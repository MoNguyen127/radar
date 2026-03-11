"""
GETTING STARTED GUIDE
Complete walkthrough from installation to model deployment
"""

# ============================================================================
# STEP 1: ENVIRONMENT SETUP
# ============================================================================

"""
1.1 Create a new conda environment (recommended):
    
    conda create -n deinterleaving python=3.11
    conda activate deinterleaving

1.2 Install the challenge package:
    
    cd turing-deinterleaving-challenge
    pip install -e .

1.3 Install model implementation dependencies:
    
    cd models_implementation
    pip install -r requirements.txt

1.4 Verify installation:
    
    python test_all.py
    
    This will test all components and verify everything is working.
"""

# ============================================================================
# STEP 2: DATA PREPARATION
# ============================================================================

"""
2.1 Download the dataset:

The data should already be in ../data/ with structure:
    data/
    ├── train/
    │   ├── train_0.h5
    │   ├── train_1.h5
    │   └── ...
    ├── validation/
    │   ├── validation_0.h5
    │   └── ...
    └── test/
        ├── test_0.h5
        └── ...

2.2 Verify data is accessible:
    
    python -c "from turing_deinterleaving_challenge import PulseTrain; 
               from pathlib import Path; 
               sample = PulseTrain.load(Path('../data/validation/validation_0.h5')); 
               print(f'Data shape: {sample.data.shape}')"
"""

# ============================================================================
# STEP 3: QUICK TEST (OPTIONAL BUT RECOMMENDED)
# ============================================================================

"""
3.1 Test all components individually:
    
    python transformer_model.py  # Test model architecture
    python triplet_loss.py       # Test loss function
    python data_utils.py         # Test normalization
    python test_all.py           # Test everything together

3.2 Quick training test (100 samples, 3 epochs):
    
    python quick_train.py
    
    This trains on a small subset to verify everything works.
    Should complete in 5-10 minutes on GPU.
"""

# ============================================================================
# STEP 4: FULL TRAINING
# ============================================================================

"""
4.1 Start training with default hyperparameters (from paper):
    
    python train.py \
        --data_dir ../data \
        --output_dir ./outputs \
        --batch_size 8 \
        --num_epochs 8 \
        --learning_rate 0.0001 \
        --window_length 1000 \
        --min_emitters 2

4.2 Monitor training with TensorBoard:
    
    # In another terminal:
    tensorboard --logdir ./outputs/run_TIMESTAMP/tensorboard
    
    # Open browser to: http://localhost:6006

4.3 Training will create:
    
    outputs/run_TIMESTAMP/
    ├── config.json              # Hyperparameters
    ├── best_model.pt            # Best model by V-measure
    ├── final_model.pt           # Model after last epoch
    ├── checkpoint_epoch_N.pt    # Checkpoints every epoch
    └── tensorboard/             # Training logs

4.4 Expected training time (approximate):
    
    - Full training set (~2.5M samples): 8-12 hours on RTX 3090
    - With batch_size=8: ~6GB GPU memory
    - CPU training: 3-5x slower

4.5 If you run out of memory:
    
    python train.py \
        --data_dir ../data \
        --output_dir ./outputs \
        --batch_size 4 \           # Reduce batch size
        --window_length 500        # Or reduce sequence length
"""

# ============================================================================
# STEP 5: EVALUATION
# ============================================================================

"""
5.1 Evaluate on validation set:
    
    python inference.py \
        --checkpoint ./outputs/run_TIMESTAMP/best_model.pt \
        --data_dir ../data \
        --subset validation

5.2 Evaluate on test set:
    
    python inference.py \
        --checkpoint ./outputs/run_TIMESTAMP/best_model.pt \
        --data_dir ../data \
        --subset test \
        --save_results ./test_results.json

5.3 Expected results (based on paper):
    
    V-measure:           ~0.884
    AMI:                 ~0.882
    ARI:                 ~0.817
    Homogeneity:         ~0.88
    Completeness:        ~0.88
    
    Note: Results may vary slightly due to:
    - Random initialization
    - HDBSCAN non-determinism
    - Implementation differences
"""

# ============================================================================
# STEP 6: USING THE MODEL FOR INFERENCE
# ============================================================================

"""
6.1 Python API for single pulse train:
"""

import torch
import numpy as np
from pathlib import Path
from transformer_model import TransformerDeinterleaver, TransformerDeinterleaverInference
from data_utils import PDWNormalizer

def predict_pulse_train(checkpoint_path, pulse_train_data):
    """
    Predict emitter labels for a pulse train.
    
    Args:
        checkpoint_path: Path to trained model
        pulse_train_data: numpy array of shape (seq_len, 5)
                         Features: [ToA, Freq, PW, AoA, Amplitude]
    
    Returns:
        labels: numpy array of shape (seq_len,)
               Predicted emitter ID for each pulse
    """
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint['config']['model']
    model = TransformerDeinterleaver(
        input_dim=5,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        embedding_dim=config['embedding_dim'],
        dropout=config['dropout'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Create inference wrapper
    inference_model = TransformerDeinterleaverInference(
        model, min_cluster_size=20, device=device
    )
    
    # Normalize and predict
    normalizer = PDWNormalizer()
    pulse_train_normalized = normalizer.normalize(pulse_train_data)
    labels = inference_model(pulse_train_normalized)
    
    return labels


"""
6.2 Example usage:
"""

# Load a pulse train from dataset
from turing_deinterleaving_challenge import PulseTrain

pulse_train = PulseTrain.load('../data/validation/validation_0.h5')
pulse_data = pulse_train.data  # Shape: (seq_len, 5)
true_labels = pulse_train.labels  # Shape: (seq_len,)

# Predict
predicted_labels = predict_pulse_train(
    checkpoint_path='./outputs/run_TIMESTAMP/best_model.pt',
    pulse_train_data=pulse_data
)

# Evaluate
from sklearn.metrics import v_measure_score
v_measure = v_measure_score(true_labels, predicted_labels)
print(f"V-measure: {v_measure:.4f}")

# ============================================================================
# STEP 7: HYPERPARAMETER TUNING (OPTIONAL)
# ============================================================================

"""
7.1 Key hyperparameters to tune:

a) Model architecture:
   - d_model: [128, 256, 512]
   - num_layers: [4, 6, 8, 10]
   - nhead: [4, 8, 16]
   - dim_feedforward: [1024, 2048, 4096]

b) Training:
   - learning_rate: [0.00005, 0.0001, 0.0002]
   - batch_size: [4, 8, 16]
   - triplet_margin: [1.0, 1.5, 1.9, 2.5]
   
c) Clustering:
   - min_cluster_size: [10, 15, 20, 25, 30]

7.2 Example experiment:

python train.py \
    --data_dir ../data \
    --output_dir ./outputs/exp_large_model \
    --d_model 512 \
    --num_layers 10 \
    --learning_rate 0.00005

7.3 Track experiments:

Use TensorBoard to compare multiple runs:
    tensorboard --logdir ./outputs
"""

# ============================================================================
# STEP 8: TROUBLESHOOTING
# ============================================================================

"""
8.1 Common issues and solutions:

ISSUE: "CUDA out of memory"
SOLUTION: 
    - Reduce batch_size: --batch_size 4 or 2
    - Reduce window_length: --window_length 500
    - Use CPU: --device cpu (slower)

ISSUE: "Loss is NaN or not decreasing"
SOLUTION:
    - Check data normalization is working
    - Reduce learning rate: --learning_rate 0.00005
    - Check min_emitters >= 2 (need triplets)
    - Gradient clipping is enabled (should be by default)

ISSUE: "No valid triplets found"
SOLUTION:
    - Ensure min_emitters >= 2
    - Check that dataset has multiple emitters per pulse train
    - Verify labels are correct

ISSUE: "Poor clustering performance"
SOLUTION:
    - Train for more epochs
    - Adjust min_cluster_size for HDBSCAN
    - Try different triplet margin
    - Increase model capacity

ISSUE: "Import errors"
SOLUTION:
    - Ensure you're in the correct directory
    - Check that challenge package is installed: pip install -e ..
    - Check requirements are installed: pip install -r requirements.txt

8.2 Debugging tips:

# Enable more verbose output:
import logging
logging.basicConfig(level=logging.DEBUG)

# Check GPU memory:
nvidia-smi -l 1  # Monitor GPU usage in real-time

# Profile training:
python -m cProfile -o profile.stats train.py --num_epochs 1
python -m pstats profile.stats
"""

# ============================================================================
# STEP 9: ADVANCED USAGE
# ============================================================================

"""
9.1 Distributed training (multiple GPUs):

# Not implemented yet, but can use PyTorch DDP:
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train.py --data_dir ../data

9.2 Resume training from checkpoint:

# Add to train.py:
if args.resume:
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']

9.3 Export model for deployment:

# Convert to TorchScript for faster inference:
model.eval()
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save('model_traced.pt')

9.4 Custom data preprocessing:

# Modify data_utils.py to add your own normalization
# Or create a custom collate_fn for special requirements
"""

# ============================================================================
# STEP 10: RESOURCES
# ============================================================================

"""
10.1 Documentation:
    - Paper: arXiv:2503.13476
    - Challenge: https://github.com/egunn-turing/turing-deinterleaving-challenge
    - Dataset: Hugging Face (egunn-turing/turing-deinterleaving-challenge)

10.2 Contact:
    - Issues: GitHub Issues page
    - Email: vnockles@turing.ac.uk

10.3 Citation:
    @article{gunn2025radar,
      title={Radar Pulse Deinterleaving with Transformer Based Deep Metric Learning},
      author={Gunn, Edward and Hosford, Adam and others},
      journal={arXiv preprint arXiv:2503.13476},
      year={2025}
    }
"""

# ============================================================================
# QUICK REFERENCE COMMANDS
# ============================================================================

"""
# Test everything:
python test_all.py

# Quick training test:
python quick_train.py

# Full training:
python train.py --data_dir ../data --output_dir ./outputs

# Monitor training:
tensorboard --logdir ./outputs

# Evaluate:
python inference.py --checkpoint ./outputs/run_TIMESTAMP/best_model.pt --data_dir ../data --subset test

# Individual component tests:
python transformer_model.py
python triplet_loss.py
python data_utils.py
"""

print(__doc__)
