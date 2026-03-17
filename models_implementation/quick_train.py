"""
Quick training script for testing on a small subset of data.
Useful for debugging and quick experiments.
"""

import sys
from pathlib import Path
import argparse
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from turing_deinterleaving_challenge import DeinterleavingChallengeDataset
from transformer_model import TransformerDeinterleaver
from triplet_loss import BatchAllTripletLoss
from data_utils import PDWNormalizer


class NormalizedDataset:
    """Wrapper to apply normalization."""
    def __init__(self, dataset, normalizer):
        self.dataset = dataset
        self.normalizer = normalizer
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        data, labels = self.dataset[idx]
        return self.normalizer.normalize(data), labels


def collate_fn(batch):
    """Collate function."""
    data, labels = zip(*batch)
    return torch.FloatTensor(np.stack(data)), torch.LongTensor(np.stack(labels))


def quick_train(
    data_dir: Path,
    subset_size: int = 100,
    num_epochs: int = 3,
    batch_size: int = 4,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 8,
    dim_feedforward: int = 2048,
    embedding_dim: int = 8,
):
    """Quick training on small subset for testing."""
    
    print("="*60)
    print("Quick Training Test")
    print("="*60)
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Load small subset of data
    print("\nLoading data...")
    train_dataset = DeinterleavingChallengeDataset(
        subset='train',
        window_length=1000,
        local_path=data_dir,
        min_emitters=2,
    )
    
    # Use only first 100 samples for quick test
    subset_size = min(subset_size, len(train_dataset))
    train_subset = Subset(train_dataset, range(subset_size))
    
    print(f"Using {len(train_subset)} training samples")
    
    # Wrap with normalization
    normalizer = PDWNormalizer()
    train_subset = NormalizedDataset(train_subset, normalizer)
    
    # Create dataloader
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Avoid multiprocessing issues
    )
    
    # Create model
    print("\nCreating model...")
    model = TransformerDeinterleaver(
        input_dim=5,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        embedding_dim=embedding_dim,
        dropout=0.05,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Create loss and optimizer
    criterion = BatchAllTripletLoss(margin=1.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # Train for a few epochs
    print(f"\nTraining for {num_epochs} epochs...")
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        
        for data, labels in pbar:
            data = data.to(device)
            labels = labels.to(device)
            
            # Forward
            embeddings = model(data)
            loss, stats = criterion(embeddings, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Stats
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'triplets': f"{stats['num_non_easy_triplets']:.0f}"
            })
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}")
    
    # Save model
    print("\nSaving model...")
    output_path = Path(__file__).parent / 'quick_test_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'dim_feedforward': dim_feedforward,
            'embedding_dim': embedding_dim,
            'dropout': 0.05,
        }
    }, output_path)
    print(f"Model saved to: {output_path}")
    
    print("\n" + "="*60)
    print("✅ Quick training test completed!")
    print("="*60)
    print("\nNext steps:")
    print("  - Run full training: python train.py --data_dir ../data")
    print("  - Test inference: python inference.py --checkpoint quick_test_model.pt")
    print("="*60)


def parse_args():
    parser = argparse.ArgumentParser(description='Quick training smoke test on a small subset')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to dataset root (contains train/validation/test)')
    parser.add_argument('--subset_size', type=int, default=100, help='Number of train windows to use')
    parser.add_argument('--epochs', type=int, default=3, help='Number of quick-test epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for quick-test')
    parser.add_argument('--d_model', type=int, default=256, help='Transformer model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of transformer encoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='Feedforward hidden dimension')
    parser.add_argument('--embedding_dim', type=int, default=8, help='Embedding output dimension')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    default_data_dir = Path(__file__).parent.parent / 'data'
    data_dir = Path(args.data_dir) if args.data_dir is not None else default_data_dir
    quick_train(
        data_dir=data_dir,
        subset_size=args.subset_size,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        embedding_dim=args.embedding_dim,
    )
