"""
🚀 QUICK START COLAB SCRIPT
Copy toàn bộ code này vào 1 cell trong Colab và chạy!
"""

# ==================== CONFIGURATION ====================
GITHUB_USERNAME = "YOUR_USERNAME"  # ← THAY ĐỔI NÀY!
REPO_NAME = "turing-deinterleaving-challenge"
NUM_EPOCHS = 3  # 3 = quick (~3h), 8 = standard (~12h)
USE_DRIVE = True  # True = lưu Drive, False = Colab local

# ==================== 1. CHECK GPU ====================
print("=" * 60)
print("STEP 1: Checking GPU...")
print("=" * 60)
import torch
try:
    import subprocess
    subprocess.run(["nvidia-smi"], check=True)
    print(f"\n✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  GPU not available! Enable GPU: Runtime → Change runtime type → GPU")
        exit(1)
except:
    print("⚠️  nvidia-smi failed")

# ==================== 2. MOUNT DRIVE ====================
if USE_DRIVE:
    print("\n" + "=" * 60)
    print("STEP 2: Mounting Google Drive...")
    print("=" * 60)
    from google.colab import drive
    drive.mount('/content/drive')
    import os
    os.makedirs('/content/drive/MyDrive/radar_deinterleaving', exist_ok=True)
    print("✓ Drive mounted!")

# ==================== 3. CLONE REPO ====================
print("\n" + "=" * 60)
print("STEP 3: Cloning repository...")
print("=" * 60)
import os
if os.path.exists(REPO_NAME):
    print("⚠️  Repo exists, removing old version...")
    import shutil
    shutil.rmtree(REPO_NAME)

import subprocess
subprocess.run([
    "git", "clone", 
    f"https://github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"
], check=True)
os.chdir(REPO_NAME)
print(f"✓ Repository cloned! Current dir: {os.getcwd()}")

# ==================== 4. INSTALL DEPENDENCIES ====================
print("\n" + "=" * 60)
print("STEP 4: Installing dependencies...")
print("=" * 60)
subprocess.run(["pip", "install", "-e", ".", "-q"], check=True)
os.chdir("models_implementation")
subprocess.run(["pip", "install", "-r", "requirements.txt", "-q"], check=True)
print("✓ Dependencies installed!")

# ==================== 5. DOWNLOAD DATA ====================
print("\n" + "=" * 60)
print("STEP 5: Downloading dataset...")
print("=" * 60)
import sys
sys.path.insert(0, f'/content/{REPO_NAME}/src')

from turing_deinterleaving_challenge import download_dataset
from pathlib import Path

if USE_DRIVE:
    data_dir = Path('/content/drive/MyDrive/radar_deinterleaving/data')
    print("📁 Saving to Google Drive (persistent)...")
else:
    data_dir = Path('/content/data')
    print("📁 Saving to Colab local (temporary)...")

data_dir.mkdir(parents=True, exist_ok=True)

if (data_dir / 'train').exists():
    print("✓ Data already exists! Skipping download.")
else:
    print("📥 Downloading dataset... This takes ~30 minutes ☕")
    print("⏰ Go grab a coffee!")
    download_dataset(
        save_dir=data_dir,
        subsets=['train', 'validation'],
        max_workers=3
    )
    print("✓ Download complete!")

# ==================== 6. QUICK TEST ====================
print("\n" + "=" * 60)
print("STEP 6: Running quick test...")
print("=" * 60)
os.chdir(f'/content/{REPO_NAME}/models_implementation')
try:
    subprocess.run(["python", "quick_train.py"], check=True, timeout=600)
    print("✓ Quick test passed!")
except subprocess.TimeoutExpired:
    print("⚠️  Test timeout but continuing...")
except Exception as e:
    print(f"⚠️  Test failed: {e}")
    print("Continuing anyway...")

# ==================== 7. FULL TRAINING ====================
print("\n" + "=" * 60)
print("STEP 7: Starting FULL TRAINING...")
print("=" * 60)

if USE_DRIVE:
    output_dir = '/content/drive/MyDrive/radar_deinterleaving/outputs'
else:
    output_dir = '/content/outputs'

print(f"""
Training Configuration:
- Epochs: {NUM_EPOCHS}
- Data: {data_dir}
- Output: {output_dir}
- Estimated time: {NUM_EPOCHS * 1.5:.1f} hours on T4
""")

import time
start_time = time.time()

training_cmd = [
    "python", "train.py",
    "--data_dir", str(data_dir),
    "--output_dir", output_dir,
    "--batch_size", "8",
    "--num_epochs", str(NUM_EPOCHS),
    "--learning_rate", "0.0001",
    "--window_length", "1000",
    "--min_emitters", "2",
    "--validate_every", "1" if NUM_EPOCHS <= 3 else "2",
    "--save_every", "1",
    "--num_workers", "2"
]

print(f"Command: {' '.join(training_cmd)}\n")
print("🚀 Training started! This will take several hours...")
print("💡 You can close this tab - training continues in background")
print("=" * 60)

try:
    subprocess.run(training_cmd, check=True)
    elapsed = (time.time() - start_time) / 3600
    print("\n" + "=" * 60)
    print(f"🎉 TRAINING COMPLETE! Time: {elapsed:.2f} hours")
    print("=" * 60)
except KeyboardInterrupt:
    print("\n⚠️  Training interrupted by user")
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    raise

# ==================== 8. FIND BEST MODEL ====================
print("\n" + "=" * 60)
print("STEP 8: Finding best model...")
print("=" * 60)

import glob
runs = sorted(glob.glob(f"{output_dir}/run_*"))
if runs:
    latest_run = runs[-1]
    best_model = f"{latest_run}/best_model.pt"
    
    if os.path.exists(best_model):
        print(f"✓ Best model: {best_model}")
        print(f"✓ Run directory: {latest_run}")
        
        # Show training summary
        log_file = f"{latest_run}/training_log.txt"
        if os.path.exists(log_file):
            print("\n📊 Training Summary:")
            with open(log_file, 'r') as f:
                lines = f.readlines()
                # Show last 10 lines
                print("".join(lines[-10:]))
    else:
        print("⚠️  Best model not found")
else:
    print("⚠️  No training runs found")

# ==================== DONE ====================
print("\n" + "=" * 60)
print("✅ ALL STEPS COMPLETED!")
print("=" * 60)
print(f"""
Summary:
- GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}
- Epochs: {NUM_EPOCHS}
- Data location: {data_dir}
- Output location: {output_dir}
- Best model: {best_model if runs else 'Not found'}

Next Steps:
1. Run evaluation (see HUONG_DAN_COLAB.md - Step 10)
2. Download model (see HUONG_DAN_COLAB.md - Step 8)
3. View TensorBoard logs

Happy Training! 🚀
""")
