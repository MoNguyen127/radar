# Transformer-based Pulse Deinterleaving Implementation

Implementation of the Transformer model with triplet loss for radar pulse deinterleaving, based on the paper:

**"Radar Pulse Deinterleaving with Transformer Based Deep Metric Learning"** (arXiv:2503.13476)

## 📁 Files Overview

- **`transformer_model.py`**: Transformer architecture with sequence-to-sequence processing
- **`triplet_loss.py`**: Batch all triplet loss implementation for metric learning
- **`data_utils.py`**: PDW normalization utilities (per-pulse-train normalization)
- **`train.py`**: Complete training script with evaluation
- **`inference.py`**: Inference script with HDBSCAN clustering
- **`requirements.txt`**: Additional dependencies

## 🔧 Installation

1. Install the main challenge package first:
```bash
cd ..
pip install -e .
```

2. Install additional requirements:
```bash
cd models_implementation
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Test Individual Components

Test the model architecture:
```bash
python transformer_model.py
```

Test triplet loss:
```bash
python triplet_loss.py
```

Test data normalization:
```bash
python data_utils.py
```

### 2. Train the Model

Basic training with default hyperparameters (from paper):
```bash
python train.py --data_dir ../data --output_dir ./outputs
```

Custom training:
```bash
python train.py \
    --data_dir ../data \
    --output_dir ./outputs/my_experiment \
    --batch_size 8 \
    --num_epochs 8 \
    --learning_rate 0.0001 \
    --window_length 1000 \
    --min_emitters 2
```

Training will create:
- `outputs/run_TIMESTAMP/` directory
- Model checkpoints (every epoch)
- Best model based on V-measure
- TensorBoard logs
- Configuration JSON

### 3. Monitor Training

View training progress with TensorBoard:
```bash
tensorboard --logdir ./outputs/run_TIMESTAMP/tensorboard
```

### 4. Evaluate Model

Evaluate on test set:
```bash
python inference.py \
    --checkpoint ./outputs/run_TIMESTAMP/best_model.pt \
    --data_dir ../data \
    --subset test \
    --save_results ./results.json
```

## 📊 Model Architecture

Based on paper specifications:

```python
TransformerDeinterleaver(
    input_dim=5,              # PDW features
    d_model=256,              # Residual size
    nhead=8,                  # Attention heads
    num_layers=8,             # Transformer layers
    dim_feedforward=2048,     # FFN hidden size
    embedding_dim=8,          # Output embedding dimension
    dropout=0.05,             # Dropout rate
)
```

**Key design choices:**
- ✅ No positional encodings (temporal info from ToA)
- ✅ Vanilla dot-product attention
- ✅ Sequence-to-sequence processing
- ✅ Progressive dimensionality: 5 → 256 → 8

## 🎯 Training Configuration

Default hyperparameters (from paper):

| Parameter | Value | Description |
|-----------|-------|-------------|
| Optimizer | Adam | Adaptive learning rate |
| Learning Rate | 0.0001 | Fixed throughout training |
| Batch Size | 8 | Pulse trains per batch |
| Epochs | 8 | Full passes through data |
| Triplet Margin | 1.9 | α in triplet loss |
| Window Length | 1000 | Pulses per window |
| Min Emitters | 2 | Filter single-emitter trains |
| Min Cluster Size | 20 | HDBSCAN parameter |

## 📈 Data Normalization

Per-pulse-train normalization (crucial for performance):

| Feature | Normalization Method |
|---------|---------------------|
| ToA | Min-max scale to [0, 1] |
| Frequency | Z-score (mean=0, std=1) |
| Pulse Width | Z-score (mean=0, std=1) |
| AoA | Divide by 360 |
| Amplitude | Z-score (mean=0, std=1) |

## 🔬 Loss Function

**Batch All Triplet Loss:**

```
For each pulse train:
  1. Generate embeddings Z = {z₁, z₂, ..., zₙ}
  2. Find all non-easy triplets T⁺(Z):
     - (zᵢ, zⱼ, zₖ) where i~j (same emitter), i≁k (different emitter)
     - d(zᵢ, zⱼ) + α ≥ d(zᵢ, zₖ) (violates margin)
  3. Compute: L = mean(max{d(zᵢ, zⱼ) - d(zᵢ, zₖ) + α, 0})
```

This encourages:
- Embeddings from same emitter to be **closer** than margin
- Embeddings from different emitters to be **farther** apart

## 🎪 Inference Pipeline

1. **Normalize** input pulse train (per-train statistics)
2. **Generate embeddings** using trained Transformer
3. **Cluster** embeddings using HDBSCAN
4. **Return** cluster labels as emitter predictions

```python
from transformer_model import TransformerDeinterleaverInference
from data_utils import PDWNormalizer

# Load model
model = load_model('best_model.pt')

# Create inference wrapper
deinterleaver = TransformerDeinterleaverInference(
    model, min_cluster_size=20
)

# Predict
pulse_train = ...  # Shape: (seq_len, 5)
labels = deinterleaver(pulse_train)  # Shape: (seq_len,)
```

## 📊 Expected Performance

Based on paper results on synthetic data:

| Metric | Expected Value |
|--------|---------------|
| Adjusted Mutual Information (AMI) | 0.882 |
| Adjusted Rand Index (ARI) | 0.817 |
| V-measure | 0.884 |
| Homogeneity | ~0.88 |
| Completeness | ~0.88 |

## 🔍 Troubleshooting

### Out of Memory (OOM)
- Reduce `batch_size` (try 4 or 2)
- Reduce `window_length` (try 500)
- Use gradient checkpointing (modify model)

### Loss not decreasing
- Check data normalization is applied
- Verify triplet mining finds valid triplets
- Ensure `min_emitters >= 2` (need multiple emitters for triplets)
- Check learning rate (try 0.0001 to 0.001)

### Poor clustering performance
- Adjust `min_cluster_size` in HDBSCAN (try 10-30)
- Try different triplet `margin` (1.5-2.5)
- Train for more epochs
- Increase model capacity (`d_model`, `num_layers`)

### Training too slow
- Increase `batch_size` if memory allows
- Reduce `num_workers` if CPU bottleneck
- Use smaller `window_length` initially
- Consider distributed training for full dataset

## 🎓 Citation

If you use this implementation, please cite:

```bibtex
@article{gunn2025radar,
  title={Radar Pulse Deinterleaving with Transformer Based Deep Metric Learning},
  author={Gunn, Edward and Hosford, Adam and Mannion, Daniel and Williams, Jarrod and Chhabra, Varun and Nockles, Victoria},
  journal={arXiv preprint arXiv:2503.13476},
  year={2025}
}
```

## 📝 Notes

- This is an independent implementation based on the paper description
- Hyperparameters match those reported in the paper
- Some implementation details may differ from the original (not open-sourced)
- HDBSCAN clustering introduces non-determinism in final predictions

## 🐛 Known Issues

- HDBSCAN can be slow for very long sequences (>5000 pulses)
- Noise points (label=-1) from HDBSCAN are kept as separate cluster
- Memory usage scales with `seq_len²` due to attention mechanism

## 🚀 Future Improvements

- [ ] Add learning rate scheduling
- [ ] Implement gradient checkpointing for longer sequences
- [ ] Add support for variable-length sequences without windowing
- [ ] Experiment with online/hard triplet mining strategies
- [ ] Add model distillation for faster inference
- [ ] Implement streaming inference for real-time processing
