"""
Transformer-based Deinterleaver Model
Based on: "Radar Pulse Deinterleaving with Transformer Based Deep Metric Learning" (arXiv:2503.13476)

Architecture:
- 8 transformer layers
- 8 attention heads
- Feed-forward hidden size: 2048
- Residual size (d_model): 256
- Embedding dimension: 8
- No positional encodings (ToA provides temporal information)
- Dropout: 0.05
"""

import torch
import torch.nn as nn
import numpy as np
import hdbscan
from typing import Optional


class TransformerDeinterleaver(nn.Module):
    """
    Sequence-to-sequence Transformer for pulse train embedding generation.
    Generates embeddings that cluster pulses from the same emitter.
    """
    
    def __init__(
        self,
        input_dim: int = 5,  # PDW features: ToA, Freq, PW, AoA, Amplitude
        d_model: int = 256,  # Residual/hidden size
        nhead: int = 8,  # Number of attention heads
        num_layers: int = 8,  # Number of transformer layers
        dim_feedforward: int = 2048,  # Feed-forward hidden size
        embedding_dim: int = 8,  # Output embedding dimension
        dropout: float = 0.05,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.embedding_dim = embedding_dim
        
        # Input projection: PDW (5D) -> d_model (256D)
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Transformer encoder layers (no positional encoding)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Input shape: (batch, seq, feature)
            norm_first=False,
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Output projection: d_model (256D) -> embedding_dim (8D)
        self.output_projection = nn.Linear(d_model, embedding_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass to generate embeddings for each pulse.
        
        Args:
            x: Input pulse train, shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask
            
        Returns:
            embeddings: Output embeddings, shape (batch_size, seq_len, embedding_dim)
        """
        # Project input to d_model dimension
        x = self.input_projection(x)  # (batch, seq, d_model)
        
        # Pass through transformer encoder
        # No positional encoding - temporal info from ToA
        x = self.transformer_encoder(x, mask=mask)  # (batch, seq, d_model)
        
        # Project to embedding dimension
        embeddings = self.output_projection(x)  # (batch, seq, embedding_dim)
        
        return embeddings


class TransformerDeinterleaverInference:
    """
    Wrapper class for inference with HDBSCAN clustering.
    Implements the Deinterleaver interface for evaluation.
    """
    
    def __init__(
        self,
        model: TransformerDeinterleaver,
        min_cluster_size: int = 20,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.model = model.to(device)
        self.model.eval()
        self.min_cluster_size = min_cluster_size
        self.device = device
        
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Perform deinterleaving on a batch of pulse trains.
        
        Args:
            data: Input pulse trains, shape (batch_size, seq_len, feature_len)
                  Can also be single pulse train (seq_len, feature_len)
            
        Returns:
            labels: Predicted emitter labels, shape (batch_size, seq_len)
                    Or (seq_len,) for single pulse train
        """
        # Handle single pulse train
        single_input = False
        if data.ndim == 2:
            data = data[np.newaxis, ...]  # Add batch dimension
            single_input = True
            
        batch_size, seq_len, _ = data.shape
        
        # Convert to tensor and move to device
        data_tensor = torch.FloatTensor(data).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            embeddings = self.model(data_tensor)  # (batch, seq, embed_dim)
            embeddings = embeddings.cpu().numpy()
        
        # Cluster each pulse train separately
        all_labels = []
        for i in range(batch_size):
            # Get embeddings for this pulse train
            pulse_embeddings = embeddings[i]  # (seq_len, embed_dim)
            
            # Cluster using HDBSCAN
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                metric='euclidean',
            )
            labels = clusterer.fit_predict(pulse_embeddings)
            
            # HDBSCAN returns -1 for noise points, we keep them as separate cluster
            all_labels.append(labels)
        
        all_labels = np.array(all_labels)
        
        # Remove batch dimension if single input
        if single_input:
            all_labels = all_labels[0]
            
        return all_labels


def create_model(device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> TransformerDeinterleaver:
    """
    Create the Transformer model with hyperparameters from the paper.
    
    Returns:
        model: TransformerDeinterleaver instance
    """
    model = TransformerDeinterleaver(
        input_dim=5,
        d_model=256,
        nhead=8,
        num_layers=8,
        dim_feedforward=2048,
        embedding_dim=8,
        dropout=0.05,
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created with {total_params:,} total parameters")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    model = create_model()
    
    # Test forward pass
    batch_size = 4
    seq_len = 1000
    input_dim = 5
    
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
    
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output embedding shape: {output.shape}")
    print(f"Expected: ({batch_size}, {seq_len}, 8)")
