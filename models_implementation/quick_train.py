"""
Quick training script for testing on a small subset of data.
Useful for debugging and quick experiments.
"""

import sys
from pathlib import Path
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


def quick_train():
    """Quick training on small subset for testing."""
    
    print("="*60)
    print("Quick Training Test")
    print("="*60)
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Load small subset of data
    print("\nLoading data...")
    data_dir = Path(__file__).parent.parent / 'data'
    
    train_dataset = DeinterleavingChallengeDataset(
        subset='train',
        window_length=1000,
        local_path=data_dir,
        min_emitters=2,
    )
    
    # Use only first 100 samples for quick test
    subset_size = min(100, len(train_dataset))
    train_subset = Subset(train_dataset, range(subset_size))
    
    print(f"Using {len(train_subset)} training samples")
    
    # Wrap with normalization
    normalizer = PDWNormalizer()
    train_subset = NormalizedDataset(train_subset, normalizer)
    
    # Create dataloader
    train_loader = DataLoader(
        train_subset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Avoid multiprocessing issues
    )
    
    # Create model
    print("\nCreating model...")
    model = TransformerDeinterleaver(
        input_dim=5,
        d_model=256,
        nhead=8,
        num_layers=8,
        dim_feedforward=2048,
        embedding_dim=8,
        dropout=0.05,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Create loss and optimizer
    criterion = BatchAllTripletLoss(margin=1.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # Train for a few epochs
    num_epochs = 3
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
            'd_model': 256,
            'nhead': 8,
            'num_layers': 8,
            'dim_feedforward': 2048,
            'embedding_dim': 8,
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


if __name__ == '__main__':
    quick_train()
