"""
Inference Script for Transformer-based Pulse Deinterleaving
Load trained model and perform deinterleaving with HDBSCAN clustering.
"""

import argparse
from pathlib import Path
import json
import sys

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from turing_deinterleaving_challenge import (
    DeinterleavingChallengeDataset,
    evaluate_model_on_dataset,
)

from transformer_model import TransformerDeinterleaver, TransformerDeinterleaverInference
from data_utils import PDWNormalizer


class NormalizedInferenceDataset:
    """Wrapper to apply normalization for inference."""
    
    def __init__(self, dataset, normalizer):
        self.dataset = dataset
        self.normalizer = normalizer
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data, labels = self.dataset[idx]
        # Normalize data
        data_normalized = self.normalizer.normalize(data)
        return data_normalized, labels


def collate_fn(batch):
    """Custom collate function."""
    data_list, labels_list = zip(*batch)
    data = np.stack(data_list, axis=0)
    labels = np.stack(labels_list, axis=0)
    return data, labels


def load_model(checkpoint_path: Path, device: str = 'cuda') -> TransformerDeinterleaver:
    """Load trained model from checkpoint."""
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})
    
    # Create model
    model = TransformerDeinterleaver(
        input_dim=5,
        d_model=model_config.get('d_model', 256),
        nhead=model_config.get('nhead', 8),
        num_layers=model_config.get('num_layers', 8),
        dim_feedforward=model_config.get('dim_feedforward', 2048),
        embedding_dim=model_config.get('embedding_dim', 8),
        dropout=model_config.get('dropout', 0.05),
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'v_measure' in checkpoint:
        print(f"Model V-measure: {checkpoint['v_measure']:.4f}")
    
    return model


def evaluate_on_dataset(
    model: TransformerDeinterleaver,
    dataset_path: Path,
    subset: str = 'test',
    window_length: int = 1000,
    min_emitters: int = 2,
    min_cluster_size: int = 20,
    batch_size: int = 8,
    device: str = 'cuda',
    num_workers: int = 4,
):
    """Evaluate model on a dataset."""
    
    print(f"\nEvaluating on {subset} set...")
    
    # Load dataset
    dataset = DeinterleavingChallengeDataset(
        subset=subset,
        window_length=window_length,
        local_path=dataset_path,
        min_emitters=min_emitters,
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Wrap with normalization
    normalizer = PDWNormalizer()
    dataset = NormalizedInferenceDataset(dataset, normalizer)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    # Create inference model
    inference_model = TransformerDeinterleaverInference(
        model,
        min_cluster_size=min_cluster_size,
        device=device,
    )
    
    # Evaluate
    print("Computing metrics...")
    metrics = evaluate_model_on_dataset(inference_model, dataloader)
    
    print(f"\nResults on {subset} set:")
    print(f"{'='*50}")
    print(f"  V-measure:         {metrics['V-measure']:.4f}")
    print(f"  Homogeneity:       {metrics['Homogeneity']:.4f}")
    print(f"  Completeness:      {metrics['Completeness']:.4f}")
    print(f"  AMI:               {metrics['Adjusted Mutual Information']:.4f}")
    print(f"  ARI:               {metrics['Adjusted Rand Index']:.4f}")
    print(f"{'='*50}")
    
    return metrics


def predict_single_pulse_train(
    model: TransformerDeinterleaver,
    pulse_train: np.ndarray,
    min_cluster_size: int = 20,
    device: str = 'cuda',
) -> np.ndarray:
    """
    Predict emitter labels for a single pulse train.
    
    Args:
        model: Trained transformer model
        pulse_train: PDW data of shape (seq_len, 5)
        min_cluster_size: HDBSCAN parameter
        device: Device to run on
        
    Returns:
        labels: Predicted emitter labels of shape (seq_len,)
    """
    # Normalize
    normalizer = PDWNormalizer()
    pulse_train_normalized = normalizer.normalize(pulse_train)
    
    # Create inference model
    inference_model = TransformerDeinterleaverInference(
        model,
        min_cluster_size=min_cluster_size,
        device=device,
    )
    
    # Predict
    labels = inference_model(pulse_train_normalized)
    
    return labels


def main():
    parser = argparse.ArgumentParser(description='Evaluate Transformer Deinterleaver')
    
    # Required
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to data directory')
    
    # Optional
    parser.add_argument('--subset', type=str, default='test',
                       choices=['train', 'validation', 'test'],
                       help='Dataset subset to evaluate on')
    parser.add_argument('--window_length', type=int, default=1000)
    parser.add_argument('--min_emitters', type=int, default=2)
    parser.add_argument('--min_cluster_size', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_results', type=str, default=None,
                       help='Path to save results JSON')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load model
    model = load_model(Path(args.checkpoint), args.device)
    
    # Evaluate
    metrics = evaluate_on_dataset(
        model=model,
        dataset_path=Path(args.data_dir),
        subset=args.subset,
        window_length=args.window_length,
        min_emitters=args.min_emitters,
        min_cluster_size=args.min_cluster_size,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
    )
    
    # Save results
    if args.save_results:
        results = {
            'checkpoint': str(args.checkpoint),
            'subset': args.subset,
            'metrics': metrics,
        }
        
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {args.save_results}")


if __name__ == '__main__':
    main()
