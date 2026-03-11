"""
Triplet Loss Implementation with Batch All Mining Strategy
Based on: "Radar Pulse Deinterleaving with Transformer Based Deep Metric Learning" (arXiv:2503.13476)

The batch all triplet loss computes loss over all non-easy triplets in a batch:
- Anchor: any pulse embedding
- Positive: embedding from same emitter
- Negative: embedding from different emitter

Loss = max(d(anchor, positive) - d(anchor, negative) + margin, 0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BatchAllTripletLoss(nn.Module):
    """
    Batch All Triplet Loss for metric learning.
    
    Computes triplet loss over all valid non-easy triplets in a batch.
    A triplet (i, j, k) is valid if:
    - pulses i and j are from the same emitter
    - pulses i and k are from different emitters
    - i ≠ j ≠ k
    
    A triplet is non-easy if:
    - d(z_i, z_j) + margin >= d(z_i, z_k)
    
    Args:
        margin: Margin α in the triplet loss (default: 1.9 from paper)
        squared: If True, use squared Euclidean distance
    """
    
    def __init__(self, margin: float = 1.9, squared: bool = False):
        super().__init__()
        self.margin = margin
        self.squared = squared
        
    def forward(
        self, 
        embeddings: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute batch all triplet loss.
        
        Args:
            embeddings: Embeddings of shape (batch_size, seq_len, embedding_dim)
            labels: Ground truth emitter labels of shape (batch_size, seq_len)
            
        Returns:
            loss: Scalar loss value
            stats: Dictionary with statistics (num_triplets, num_valid, etc.)
        """
        batch_size, seq_len, embedding_dim = embeddings.shape
        
        # Reshape to treat all pulses in batch together
        # This allows us to compute distances across the entire batch
        embeddings_flat = embeddings.view(-1, embedding_dim)  # (batch*seq, embed_dim)
        labels_flat = labels.view(-1)  # (batch*seq,)
        
        # Compute pairwise distance matrix
        distances = self._pairwise_distances(embeddings_flat, squared=self.squared)
        
        # Get positive and negative masks
        positive_mask = self._get_positive_mask(labels_flat)
        negative_mask = self._get_negative_mask(labels_flat)
        
        # Compute triplet loss
        loss, stats = self._batch_all_triplet_loss(
            distances, positive_mask, negative_mask
        )
        
        return loss, stats
    
    def _pairwise_distances(
        self, 
        embeddings: torch.Tensor, 
        squared: bool = False
    ) -> torch.Tensor:
        """
        Compute pairwise Euclidean distances between embeddings.
        
        Args:
            embeddings: Tensor of shape (n, embedding_dim)
            squared: If True, return squared distances
            
        Returns:
            distances: Tensor of shape (n, n)
        """
        # Compute dot products
        dot_product = torch.matmul(embeddings, embeddings.t())
        
        # Get squared L2 norms
        square_norm = torch.diag(dot_product)
        
        # Compute squared distances: ||a-b||^2 = ||a||^2 - 2<a,b> + ||b||^2
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        
        # Ensure distances are non-negative (numerical stability)
        distances = F.relu(distances)
        
        if not squared:
            # Add small epsilon for numerical stability in sqrt
            mask = (distances == 0.0).float()
            distances = distances + mask * 1e-16
            distances = torch.sqrt(distances)
            distances = distances * (1.0 - mask)
        
        return distances
    
    def _get_positive_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Get mask for positive pairs (same emitter, different indices).
        
        Args:
            labels: Tensor of shape (n,)
            
        Returns:
            mask: Boolean tensor of shape (n, n)
        """
        # Check if labels are equal
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        # Exclude diagonal (i != j)
        indices_equal = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
        indices_not_equal = ~indices_equal
        
        # Positive pairs: same label, different indices
        positive_mask = labels_equal & indices_not_equal
        
        return positive_mask
    
    def _get_negative_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Get mask for negative pairs (different emitters).
        
        Args:
            labels: Tensor of shape (n,)
            
        Returns:
            mask: Boolean tensor of shape (n, n)
        """
        # Check if labels are not equal
        labels_not_equal = labels.unsqueeze(0) != labels.unsqueeze(1)
        
        return labels_not_equal
    
    def _batch_all_triplet_loss(
        self,
        distances: torch.Tensor,
        positive_mask: torch.Tensor,
        negative_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute batch all triplet loss from distance matrix and masks.
        
        Args:
            distances: Pairwise distances, shape (n, n)
            positive_mask: Positive pairs mask, shape (n, n)
            negative_mask: Negative pairs mask, shape (n, n)
            
        Returns:
            loss: Scalar loss value
            stats: Dictionary with statistics
        """
        # For each anchor, get positive and negative distances
        # distances[i, j] = distance from pulse i to pulse j
        
        # Expand dimensions for broadcasting
        # anchor_positive_dist: (n, n, 1) - distance from anchor to positive
        # anchor_negative_dist: (n, 1, n) - distance from anchor to negative
        anchor_positive_dist = distances.unsqueeze(2)
        anchor_negative_dist = distances.unsqueeze(1)
        
        # Triplet loss for all combinations
        # (i, j, k) where i=anchor, j=positive, k=negative
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin
        
        # Masks for valid triplets
        # positive_mask: (n, n) -> (n, n, 1)
        # negative_mask: (n, n) -> (n, 1, n)
        positive_mask_3d = positive_mask.unsqueeze(2)
        negative_mask_3d = negative_mask.unsqueeze(1)
        
        # Valid triplets: i~j (same emitter) and i≁k (different emitters)
        # Also ensure i≠j≠k is handled by the masks
        valid_triplets_mask = positive_mask_3d & negative_mask_3d
        
        # Non-easy triplets: valid triplets where loss > 0
        # These are triplets that violate the margin constraint
        non_easy_triplets = valid_triplets_mask & (triplet_loss > 0)
        
        # Apply hard cut-off: only consider non-easy triplets
        triplet_loss = triplet_loss * non_easy_triplets.float()
        
        # Count triplets
        num_valid_triplets = valid_triplets_mask.sum().float()
        num_non_easy_triplets = non_easy_triplets.sum().float()
        
        # Compute mean loss over non-easy triplets
        if num_non_easy_triplets > 0:
            loss = triplet_loss.sum() / num_non_easy_triplets
        else:
            loss = torch.tensor(0.0, device=distances.device, requires_grad=True)
        
        # Statistics
        stats = {
            'num_valid_triplets': num_valid_triplets.item(),
            'num_non_easy_triplets': num_non_easy_triplets.item(),
            'fraction_non_easy': (num_non_easy_triplets / (num_valid_triplets + 1e-16)).item(),
        }
        
        return loss, stats


def test_triplet_loss():
    """Test triplet loss computation."""
    print("Testing Batch All Triplet Loss...")
    
    # Create dummy data
    batch_size = 2
    seq_len = 10
    embedding_dim = 8
    
    # Create embeddings
    embeddings = torch.randn(batch_size, seq_len, embedding_dim)
    
    # Create labels with 3 emitters
    labels = torch.tensor([
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 2],  # Batch 0: 3 emitters
        [0, 0, 1, 1, 1, 2, 2, 2, 3, 3],  # Batch 1: 4 emitters
    ])
    
    # Create loss function
    criterion = BatchAllTripletLoss(margin=1.9)
    
    # Compute loss
    loss, stats = criterion(embeddings, labels)
    
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Valid triplets: {stats['num_valid_triplets']:.0f}")
    print(f"Non-easy triplets: {stats['num_non_easy_triplets']:.0f}")
    print(f"Fraction non-easy: {stats['fraction_non_easy']:.4f}")
    
    # Test with perfect embeddings (should have low loss)
    print("\n\nTesting with perfect embeddings...")
    perfect_embeddings = torch.zeros(batch_size, seq_len, embedding_dim)
    for b in range(batch_size):
        for i in range(seq_len):
            label = labels[b, i].item()
            # Assign same embedding for same label
            perfect_embeddings[b, i] = torch.randn(embedding_dim) * 0.1 + label * 10
    
    loss_perfect, stats_perfect = criterion(perfect_embeddings, labels)
    
    print(f"Loss (perfect): {loss_perfect.item():.4f}")
    print(f"Valid triplets: {stats_perfect['num_valid_triplets']:.0f}")
    print(f"Non-easy triplets: {stats_perfect['num_non_easy_triplets']:.0f}")
    print(f"Fraction non-easy: {stats_perfect['fraction_non_easy']:.4f}")
    
    print("\n✓ Triplet loss test completed!")


if __name__ == "__main__":
    test_triplet_loss()
