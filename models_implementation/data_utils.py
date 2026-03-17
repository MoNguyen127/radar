"""
Data Normalization Utilities for Pulse Descriptor Words (PDWs)
Based on: "Radar Pulse Deinterleaving with Transformer Based Deep Metric Learning" (arXiv:2503.13476)

PDW features (5 dimensions):
0. Time of Arrival (ToA): Min-max scale to [0, 1]
1. Centre Frequency: Z-score normalization (mean=0, std=1)
2. Pulse Width (PW): Z-score normalization
3. Angle of Arrival (AoA): (aoa + 180) / 360 → scale [-180, 180] deg to [0, 1]
4. Amplitude: Z-score normalization (inf values replaced with 0 after normalization)

NOTE: Normalization is done WITHIN EACH pulse train separately.
"""

import numpy as np
import torch
from typing import Union


class PDWNormalizer:
    """
    Normalizer for Pulse Descriptor Words (PDWs).
    
    Applies feature-specific normalization as described in the paper:
    - ToA: Min-max scaling to [0, 1]
    - Frequency, PW, Amplitude: Z-score normalization (inf values in Amplitude replaced with 0)
    - AoA: (aoa + 180) / 360 → maps [-180°, 180°] to [0, 1]
    
    Normalization is applied independently to each pulse train.
    """
    
    def __init__(self):
        self.feature_names = ['ToA', 'Frequency', 'PW', 'AoA', 'Amplitude']
        
    def normalize(
        self, 
        data: Union[np.ndarray, torch.Tensor],
        return_torch: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalize PDW features.
        
        Args:
            data: PDW data of shape (seq_len, 5) or (batch_size, seq_len, 5)
            return_torch: If True, return torch tensor, else numpy array
            
        Returns:
            normalized_data: Normalized PDW data with same shape as input
        """
        # Convert to numpy if torch tensor
        is_torch = isinstance(data, torch.Tensor)
        if is_torch:
            device = data.device
            data = data.cpu().numpy()
        
        # Handle single pulse train or batch
        single_train = data.ndim == 2
        if single_train:
            data = data[np.newaxis, ...]  # Add batch dimension
        
        batch_size, seq_len, num_features = data.shape
        assert num_features == 5, f"Expected 5 features, got {num_features}"
        
        # Create output array
        normalized = np.zeros_like(data)
        
        # Normalize each pulse train in the batch separately
        for b in range(batch_size):
            pulse_train = data[b]  # (seq_len, 5)
            
            # Feature 0: ToA - Min-max scaling to [0, 1]
            toa = pulse_train[:, 0]
            toa_min = toa.min()
            toa_max = toa.max()
            if toa_max > toa_min:
                normalized[b, :, 0] = (toa - toa_min) / (toa_max - toa_min)
            else:
                normalized[b, :, 0] = 0.0  # All same value
            
            # Feature 1: Frequency - Z-score normalization
            freq = pulse_train[:, 1]
            freq_mean = freq.mean()
            freq_std = freq.std()
            if freq_std > 0:
                normalized[b, :, 1] = (freq - freq_mean) / freq_std
            else:
                normalized[b, :, 1] = 0.0
            
            # Feature 2: Pulse Width - Z-score normalization
            pw = pulse_train[:, 2]
            pw_mean = pw.mean()
            pw_std = pw.std()
            if pw_std > 0:
                normalized[b, :, 2] = (pw - pw_mean) / pw_std
            else:
                normalized[b, :, 2] = 0.0
            
            # Feature 3: AoA - Scale to [0, 1]
            # Data is in [-180, 180] degrees, so use (aoa + 180) / 360
            aoa = pulse_train[:, 3]
            aoa = np.clip(aoa, -180.0, 180.0)  # clamp any out-of-range values
            normalized[b, :, 3] = (aoa + 180.0) / 360.0
            
            # Feature 4: Amplitude - Z-score normalization
            # Replace inf/-inf before computing stats to avoid NaN loss
            amp = pulse_train[:, 4]
            amp = np.where(np.isfinite(amp), amp, np.nan)
            amp_mean = np.nanmean(amp)
            amp_std = np.nanstd(amp)
            if amp_std > 0:
                norm_amp = (amp - amp_mean) / amp_std
                # Replace NaN (originally inf) with 0 after normalization
                normalized[b, :, 4] = np.where(np.isfinite(norm_amp), norm_amp, 0.0)
            else:
                normalized[b, :, 4] = 0.0
        
        # Remove batch dimension if input was single pulse train
        if single_train:
            normalized = normalized[0]
        
        # Convert to torch tensor if requested
        if return_torch or is_torch:
            normalized = torch.FloatTensor(normalized)
            if is_torch:
                normalized = normalized.to(device)
        
        return normalized
    
    def __call__(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Allow calling the normalizer as a function."""
        return self.normalize(data)


def test_normalizer():
    """Test PDW normalization."""
    print("Testing PDW Normalizer...")
    
    # Create dummy PDW data
    batch_size = 2
    seq_len = 100
    num_features = 5
    
    # Generate realistic-looking data
    data = np.zeros((batch_size, seq_len, num_features))
    
    for b in range(batch_size):
        # ToA: Increasing timestamps
        data[b, :, 0] = np.cumsum(np.random.exponential(100, seq_len))
        
        # Frequency: Around 5000 MHz
        data[b, :, 1] = np.random.normal(5000, 500, seq_len)
        
        # Pulse Width: Around 5 µs
        data[b, :, 2] = np.random.exponential(5, seq_len)
        
        # AoA: -180 to 180 degrees
        data[b, :, 3] = np.random.uniform(-180, 180, seq_len)
        
        # Amplitude: Around -80 dBm
        data[b, :, 4] = np.random.normal(-80, 10, seq_len)
    
    print(f"\nOriginal data shape: {data.shape}")
    print(f"\nOriginal data statistics (first pulse train):")
    for i, name in enumerate(['ToA', 'Frequency', 'PW', 'AoA', 'Amplitude']):
        print(f"  {name:12s}: min={data[0, :, i].min():10.2f}, "
              f"max={data[0, :, i].max():10.2f}, "
              f"mean={data[0, :, i].mean():10.2f}, "
              f"std={data[0, :, i].std():10.2f}")
    
    # Normalize
    normalizer = PDWNormalizer()
    normalized = normalizer.normalize(data)
    
    print(f"\nNormalized data shape: {normalized.shape}")
    print(f"\nNormalized data statistics (first pulse train):")
    for i, name in enumerate(['ToA', 'Frequency', 'PW', 'AoA', 'Amplitude']):
        print(f"  {name:12s}: min={normalized[0, :, i].min():10.4f}, "
              f"max={normalized[0, :, i].max():10.4f}, "
              f"mean={normalized[0, :, i].mean():10.4f}, "
              f"std={normalized[0, :, i].std():10.4f}")
    
    # Test with torch tensor
    print("\n\nTesting with torch tensor...")
    data_torch = torch.FloatTensor(data)
    normalized_torch = normalizer.normalize(data_torch, return_torch=True)
    print(f"Torch output type: {type(normalized_torch)}")
    print(f"Torch output shape: {normalized_torch.shape}")
    
    # Test with single pulse train
    print("\n\nTesting with single pulse train...")
    single_data = data[0]  # (seq_len, 5)
    normalized_single = normalizer.normalize(single_data)
    print(f"Single pulse train input shape: {single_data.shape}")
    print(f"Single pulse train output shape: {normalized_single.shape}")
    
    # Verify ToA is in [0, 1]
    assert normalized[0, :, 0].min() >= -1e-6, "ToA min should be ~0"
    assert normalized[0, :, 0].max() <= 1 + 1e-6, "ToA max should be ~1"
    
    # Verify z-score normalized features have mean~0, std~1
    for feat_idx in [1, 2, 4]:  # Frequency, PW, Amplitude
        mean = normalized[0, :, feat_idx].mean()
        std = normalized[0, :, feat_idx].std()
        assert abs(mean) < 1e-6, f"Feature {feat_idx} mean should be ~0, got {mean}"
        assert abs(std - 1.0) < 1e-6, f"Feature {feat_idx} std should be ~1, got {std}"
    
    print("\n✓ All normalization tests passed!")


if __name__ == "__main__":
    test_normalizer()
