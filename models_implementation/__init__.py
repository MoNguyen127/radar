"""
Transformer-based Pulse Deinterleaving Implementation
Based on: "Radar Pulse Deinterleaving with Transformer Based Deep Metric Learning" (arXiv:2503.13476)
"""

from .transformer_model import (
    TransformerDeinterleaver,
    TransformerDeinterleaverInference,
    create_model,
)
from .triplet_loss import BatchAllTripletLoss
from .data_utils import PDWNormalizer

__all__ = [
    'TransformerDeinterleaver',
    'TransformerDeinterleaverInference',
    'create_model',
    'BatchAllTripletLoss',
    'PDWNormalizer',
]

__version__ = '1.0.0'
