"""
benchmark.py — Precise wall-clock timing of the train/val pipeline.

Simulates every real operation (GPU forward, triplet loss, backward, HDBSCAN,
sklearn metrics) using random tensors. No dataset needed.

Usage:
    python benchmark.py                       # defaults matching train.py
    python benchmark.py --seq-len 500         # test smaller window
    python benchmark.py --device cpu          # CPU-only timing

Timing method:
    torch.cuda.synchronize() is called before and after every GPU operation.
    Without this, CUDA launches are async and wall-clock time measures only
    the kernel launch overhead (~µs), not actual compute (~ms).
"""

from __future__ import annotations

import argparse
import math
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import HDBSCAN
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    v_measure_score,
)

from turing_deinterleaving_challenge.models import TransformerDeinterleaver
from turing_deinterleaving_challenge.models.config import EmbeddingConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class sync_time:
    """GPU-synced wall-clock timer used as a context manager.

    Example:
        with sync_time(device) as t:
            model(x)
        print(t.elapsed)   # seconds, guaranteed to include full GPU compute
    """
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.elapsed: float = 0.0

    def __enter__(self) -> "sync_time":
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.t0


def _fmt(seconds: float) -> str:
    """Format seconds as ms with 2 decimal places."""
    return f"{seconds * 1000:8.2f} ms"


def _fmt_duration(seconds: float) -> str:
    """Format a large duration as hours/minutes."""
    if seconds < 60:
        return f"{seconds:.1f} s"
    if seconds < 3600:
        return f"{seconds/60:.1f} min"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h {m:02d}m"


def _run(fn, warmup: int, repeats: int, device: torch.device) -> tuple[float, float]:
    """Run fn() warmup+repeats times; return (mean_seconds, std_seconds)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        with sync_time(device) as t:
            fn()
        times.append(t.elapsed)
    mean = statistics.mean(times)
    std  = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean, std


def _flag(mean_s: float, all_means: list[float]) -> str:
    """Mark the slowest operation with an arrow."""
    return "  ◄ bottleneck" if mean_s == max(all_means) else ""


# ---------------------------------------------------------------------------
# Random data generators
# ---------------------------------------------------------------------------

def make_random_pdw(batch: int, seq: int, device: torch.device) -> torch.Tensor:
    """Simulate a realistic PDW tensor: (batch, seq, 5) float32.

    Ranges match the dataset description:
      ToA  : cumulative, inter-pulse gaps 100–1100 µs  (ascending per sample)
      RF   : 1500–9500 MHz
      PW   : 0.1–2.1 µs
      AOA  : −180° to +180°
      PA   : −80 to −40 dBm
    """
    toa = torch.cumsum(torch.rand(batch, seq) * 1000.0 + 100.0, dim=1)
    rf  = torch.rand(batch, seq) * 8000.0 + 1500.0
    pw  = torch.rand(batch, seq) * 2.0   + 0.1
    aoa = torch.rand(batch, seq) * 360.0 - 180.0
    pa  = torch.rand(batch, seq) * 40.0  - 80.0
    return torch.stack([toa, rf, pw, aoa, pa], dim=-1).to(device)


def make_random_labels(batch: int, seq: int, n_emitters: int,
                       device: torch.device) -> torch.Tensor:
    """Simulate integer emitter-ID labels: (batch, seq) int64."""
    return torch.randint(0, n_emitters, (batch, seq), dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Triplet-loss forward — reproduced here so it can be timed independently
# ---------------------------------------------------------------------------

def _triplet_fwd(z: torch.Tensor, y: torch.Tensor,
                 margin: float, max_n: int) -> torch.Tensor:
    """Batch-all triplet loss forward pass (no backward)."""
    N, D = z.shape
    if N > max_n:
        classes, inverse = torch.unique(y, return_inverse=True)
        n_cls   = len(classes)
        per_cls = max(1, max_n // n_cls)
        keep = []
        for c in range(n_cls):
            idx  = (inverse == c).nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(idx), device=z.device)[:per_cls]
            keep.append(idx[perm])
        keep = torch.cat(keep)
        z, y = z[keep], y[keep]
        N    = len(z)

    sq     = (z * z).sum(dim=1)
    dist2  = sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * (z @ z.T)
    dist   = dist2.clamp(min=1e-12).sqrt()

    eye      = torch.eye(N, dtype=torch.bool, device=z.device)
    same     = y.unsqueeze(0) == y.unsqueeze(1)
    pos_mask = same & ~eye
    neg_mask = ~same

    d_ap         = dist.unsqueeze(2)
    d_an         = dist.unsqueeze(1)
    triplet_loss = (d_ap - d_an + margin).clamp(min=0.0)
    valid        = pos_mask.unsqueeze(2) & neg_mask.unsqueeze(1)
    non_easy     = valid & (triplet_loss > 0)

    return (triplet_loss * non_easy.float()).sum() / non_easy.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# Benchmark: training step
# ---------------------------------------------------------------------------

def bench_train_step(
    model: TransformerDeinterleaver,
    batch: int,
    seq: int,
    n_emitters: int,
    margin: float,
    max_triplet_n: int,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, tuple[float, float]]:
    """Time each sub-operation in one training step."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    # Shared data for a single "step" — regenerated each call inside fns
    def _make():
        return (make_random_pdw(batch, seq, device),
                make_random_labels(batch, seq, n_emitters, device))

    results: dict[str, tuple[float, float]] = {}

    # ── PDWEmbedding forward ─────────────────────────────────────────────────
    def _emb():
        data, _ = _make()
        model.embedding(data)

    results["PDWEmbedding forward"] = _run(_emb, warmup, repeats, device)

    # ── TransformerEncoder forward ───────────────────────────────────────────
    def _transformer():
        data, _ = _make()
        emb = model.embedding(data)
        model.transformer(emb)

    results["TransformerEncoder"] = _run(_transformer, warmup, repeats, device)

    # ── Full encode() (embedding + transformer + metric_head + L2 norm) ──────
    def _encode():
        data, _ = _make()
        model.encode(data)

    results["Full encode()"] = _run(_encode, warmup, repeats, device)

    # ── Triplet loss forward ─────────────────────────────────────────────────
    def _loss_fwd():
        data, labels = _make()
        with torch.no_grad():
            emb = model.encode(data)
        z = emb.reshape(batch * seq, -1)
        y = labels.reshape(batch * seq)
        _triplet_fwd(z, y, margin, max_triplet_n)

    results["Triplet loss (fwd)"] = _run(_loss_fwd, warmup, repeats, device)

    # ── Backward pass ────────────────────────────────────────────────────────
    def _backward():
        data, labels = _make()
        emb = model.encode(data)
        z   = emb.reshape(batch * seq, -1)
        y   = labels.reshape(batch * seq)
        loss = _triplet_fwd(z, y, margin, max_triplet_n)
        loss.backward()

    results["Backward pass"] = _run(_backward, warmup, repeats, device)

    # ── Optimizer step ───────────────────────────────────────────────────────
    def _opt_step():
        data, labels = _make()
        emb  = model.encode(data)
        z    = emb.reshape(batch * seq, -1)
        y    = labels.reshape(batch * seq)
        loss = _triplet_fwd(z, y, margin, max_triplet_n)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    results["Optimizer step (full)"] = _run(_opt_step, warmup, repeats, device)

    return results


# ---------------------------------------------------------------------------
# Benchmark: validation step
# ---------------------------------------------------------------------------

def bench_val_step(
    model: TransformerDeinterleaver,
    batch: int,
    seq: int,
    n_emitters: int,
    min_cluster_size: int,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, tuple[float, float]]:
    """Time each sub-operation in one validation step."""
    model.eval()
    results: dict[str, tuple[float, float]] = {}

    def _make():
        return (make_random_pdw(batch, seq, device),
                make_random_labels(batch, seq, n_emitters, device))

    # ── encode() no-grad ─────────────────────────────────────────────────────
    def _encode_nograd():
        data, _ = _make()
        with torch.no_grad():
            model.encode(data)

    results["encode() no-grad"] = _run(_encode_nograd, warmup, repeats, device)

    # ── GPU→CPU transfer ─────────────────────────────────────────────────────
    _data, _ = _make()
    with torch.no_grad():
        _emb_gpu = model.encode(_data)   # (batch, seq, d_model)

    def _transfer():
        _emb_gpu.cpu().numpy()

    results["GPU→CPU transfer"] = _run(_transfer, warmup, repeats, device)

    # ── HDBSCAN × batch ──────────────────────────────────────────────────────
    _emb_np = _emb_gpu.cpu().detach().numpy()   # (batch, seq, d_model)

    def _hdbscan():
        for i in range(batch):
            HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean").fit_predict(
                _emb_np[i]
            )

    results[f"HDBSCAN × {batch}"] = _run(_hdbscan, warmup, repeats, device)

    # ── sklearn metrics × batch ───────────────────────────────────────────────
    _labels_true = make_random_labels(batch, seq, n_emitters, device).cpu().numpy()
    _labels_pred = np.zeros((batch, seq), dtype=np.int32)
    for i in range(batch):
        _labels_pred[i] = HDBSCAN(min_cluster_size=min_cluster_size,
                                   metric="euclidean").fit_predict(_emb_np[i])

    def _metrics():
        for i in range(batch):
            lt = _labels_true[i]
            lp = _labels_pred[i]
            homogeneity_score(lt, lp)
            completeness_score(lt, lp)
            v_measure_score(lt, lp)
            adjusted_rand_score(lt, lp)
            adjusted_mutual_info_score(lt, lp)

    results[f"sklearn metrics × {batch}"] = _run(_metrics, warmup, repeats, device)

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print(f"\n── {title} " + "─" * max(0, 52 - len(title)))


def print_report(
    train_results: dict[str, tuple[float, float]],
    val_results: dict[str, tuple[float, float]],
    args: argparse.Namespace,
    device: torch.device,
    n_params: int,
) -> None:
    width = 60
    print("=" * width)

    device_str = (torch.cuda.get_device_name(0)
                  + f" ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)"
                  if device.type == "cuda" else "CPU")
    print(f"  PDW Benchmark — batch={args.batch_size}  seq={args.seq_len}"
          f"  d_model={args.d_model}")
    print(f"  device : {device_str}")
    print(f"  params : {n_params:,}")
    print("=" * width)

    # Training step
    _section(f"TRAINING STEP ({args.repeats} runs, {args.warmup} warmup)")
    t_means = [m for m, _ in train_results.values()]
    for name, (mean_s, std_s) in train_results.items():
        flag = _flag(mean_s, t_means)
        print(f"  {name:<28}: {_fmt(mean_s)} ± {std_s*1000:5.2f} ms{flag}")

    # Validation step
    _section(f"VALIDATION STEP ({args.repeats} runs, {args.warmup} warmup)")
    v_means = [m for m, _ in val_results.values()]
    for name, (mean_s, std_s) in val_results.items():
        flag = _flag(mean_s, v_means)
        print(f"  {name:<28}: {_fmt(mean_s)} ± {std_s*1000:5.2f} ms{flag}")

    # Epoch projections
    _section("EPOCH PROJECTIONS")
    # Use the "Optimizer step (full)" time for per-step train cost
    train_step_s = train_results["Optimizer step (full)"][0]
    # Validation step total = encode + transfer + hdbscan + metrics
    val_step_s = sum(m for m, _ in val_results.values())

    train_steps   = math.ceil(args.train_windows / args.batch_size)
    val_steps     = math.ceil(args.val_windows   / args.batch_size)
    train_epoch_s = train_step_s * train_steps
    val_epoch_s   = val_step_s   * val_steps
    full_epoch_s  = train_epoch_s + val_epoch_s
    total_s       = full_epoch_s  * args.epochs

    print(f"  Train steps / epoch  : {train_steps:>10,}")
    print(f"  Val   steps / epoch  : {val_steps:>10,}")
    print()
    print(f"  Train time / epoch   : ~{_fmt_duration(train_epoch_s)}")
    print(f"  Val   time / epoch   : ~{_fmt_duration(val_epoch_s)}")
    print(f"  Full epoch           : ~{_fmt_duration(full_epoch_s)}")
    print()
    print(f"  {args.epochs}-epoch total         : ~{_fmt_duration(total_s)}")
    print("=" * width)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark training and validation pipeline timing.")
    # Model
    p.add_argument("--batch-size",       type=int,   default=8)
    p.add_argument("--seq-len",          type=int,   default=1000,
                   help="Window length (pulses per sample)")
    p.add_argument("--d-model",          type=int,   default=128)
    p.add_argument("--d-feat",           type=int,   default=64)
    p.add_argument("--nhead",            type=int,   default=8)
    p.add_argument("--num-layers",       type=int,   default=4)
    p.add_argument("--dim-feedforward",  type=int,   default=512)
    p.add_argument("--dropout",          type=float, default=0.0,
                   help="Use 0 for benchmark (avoids stochastic noise in timing)")
    # Data simulation
    p.add_argument("--n-emitters",       type=int,   default=5,
                   help="Number of emitters per simulated window")
    # Loss
    p.add_argument("--margin",           type=float, default=1.9)
    p.add_argument("--max-triplet-n",    type=int,   default=512)
    # Inference
    p.add_argument("--min-cluster-size", type=int,   default=20)
    # Benchmark control
    p.add_argument("--warmup",           type=int,   default=3,
                   help="Warmup iterations (not timed) to stabilise CUDA caches")
    p.add_argument("--repeats",          type=int,   default=10,
                   help="Timed iterations; mean ± std reported")
    # Epoch projection inputs
    p.add_argument("--train-windows",    type=int,   default=329_815)
    p.add_argument("--val-windows",      type=int,   default=34_715)
    p.add_argument("--epochs",           type=int,   default=20)
    # Device
    p.add_argument("--device",           type=str,   default=None,
                   help="'cuda', 'cpu', or None to auto-detect")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Build model (feature_stats=None → identity normalization, fine for timing)
    model = TransformerDeinterleaver(
        d_model=args.d_model,
        d_feat=args.d_feat,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        feature_stats=None,
        min_cluster_size=args.min_cluster_size,
        embedding_config=EmbeddingConfig(),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    print(f"\nRunning training benchmark  ({args.warmup} warmup + {args.repeats} timed)...")
    train_results = bench_train_step(
        model=model,
        batch=args.batch_size,
        seq=args.seq_len,
        n_emitters=args.n_emitters,
        margin=args.margin,
        max_triplet_n=args.max_triplet_n,
        device=device,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    print(f"Running validation benchmark ({args.warmup} warmup + {args.repeats} timed)...")
    val_results = bench_val_step(
        model=model,
        batch=args.batch_size,
        seq=args.seq_len,
        n_emitters=args.n_emitters,
        min_cluster_size=args.min_cluster_size,
        device=device,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    print_report(train_results, val_results, args, device, n_params)


if __name__ == "__main__":
    main()
