# 🚀 HƯỚNG DẪN TRAIN MÔ HÌNH TRÊN GOOGLE COLAB - TỪ A ĐẾN Z

## ✅ **YÊU CẦU TRƯỚC KHI BẮT ĐẦU**

- [ ] Đã push code lên GitHub
- [ ] Có tài khoản Google (Gmail)
- [ ] Có tài khoản GitHub
- [ ] Internet ổn định

---

## 📋 **BƯỚC 1: MỞ GOOGLE COLAB**

1. Mở trình duyệt Chrome/Edge/Firefox
2. Vào: [https://colab.research.google.com](https://colab.research.google.com)
3. Đăng nhập bằng tài khoản Google

---

## 📝 **BƯỚC 2: TẠO NOTEBOOK MỚI**

**Option A: Tạo notebook trống**
1. Click **"File"** → **"New notebook"** (hoặc Ctrl+N)
2. Đổi tên: Click vào **"Untitled0.ipynb"** → Đổi thành `Training_Radar_Model.ipynb`

**Option B: Upload notebook có sẵn** (Khuyên dùng!)
1. Click **"File"** → **"Upload notebook"**
2. Chọn tab **"GitHub"**
3. Nhập URL repo của bạn: `https://github.com/YOUR_USERNAME/turing-deinterleaving-challenge`
4. Chọn file: `models_implementation/Colab_Training.ipynb`
5. Click **"Open"**

→ **Nếu dùng Option B, SKIP BƯỚC 3, đi thẳng BƯỚC 4**

---

## ⚙️ **BƯỚC 3: SETUP NOTEBOOK (Nếu tạo mới)**

Tạo các cells sau (click **"+ Code"** để tạo cell mới):

### **Cell 1: Check GPU**
```python
# Kiểm tra GPU
import torch
!nvidia-smi

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### **Cell 2: Mount Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')

# Tạo thư mục project
!mkdir -p /content/drive/MyDrive/radar_deinterleaving
print("✓ Google Drive mounted!")
```

### **Cell 3: Clone GitHub Repository**
```python
# Thay YOUR_USERNAME bằng username GitHub của bạn
GITHUB_USERNAME = "YOUR_USERNAME"
REPO_NAME = "turing-deinterleaving-challenge"

# Clone repo
!git clone https://github.com/{GITHUB_USERNAME}/{REPO_NAME}.git
%cd {REPO_NAME}

print("✓ Repository cloned!")
```

### **Cell 4: Install Dependencies**
```python
# Install base package
!pip install -e . -q

# Install model dependencies
%cd models_implementation
!pip install -r requirements.txt -q

print("✓ Dependencies installed!")
```

### **Cell 5: Download Dataset**
```python
import sys
sys.path.insert(0, '/content/turing-deinterleaving-challenge/src')

from turing_deinterleaving_challenge import download_dataset
from pathlib import Path

# Chọn nơi lưu data
USE_DRIVE = True  # True = lưu Drive, False = lưu Colab local

if USE_DRIVE:
    data_dir = Path('/content/drive/MyDrive/radar_deinterleaving/data')
    print("📁 Saving to Google Drive...")
else:
    data_dir = Path('/content/data')
    print("📁 Saving to Colab (temporary)...")

data_dir.mkdir(parents=True, exist_ok=True)

# Kiểm tra data đã tồn tại chưa
if (data_dir / 'train').exists():
    print("✓ Data already exists! Skipping download.")
else:
    print("📥 Downloading dataset (~30 minutes)...")
    download_dataset(
        save_dir=data_dir,
        subsets=['train', 'validation'],
        max_workers=3
    )
    print("✓ Download complete!")
```

### **Cell 6: Quick Test**
```python
%cd /content/turing-deinterleaving-challenge/models_implementation
!python quick_train.py

print("✓ Quick test passed!")
```

### **Cell 7: Full Training**
```python
# Cấu hình training
output_dir = '/content/drive/MyDrive/radar_deinterleaving/outputs'

# Chọn config
CONFIG = "quick"  # Options: "quick" (3 epochs), "standard" (8 epochs)

# Training command
!python train.py \
    --data_dir {data_dir} \
    --output_dir {output_dir} \
    --batch_size 8 \
    --num_epochs 3 \
    --learning_rate 0.0001 \
    --window_length 1000 \
    --min_emitters 2 \
    --validate_every 1 \
    --save_every 1 \
    --num_workers 2

print("🎉 Training complete!")
```

---

## 🎮 **BƯỚC 4: ENABLE GPU**

**QUAN TRỌNG - PHẢI LÀM BƯỚC NÀY!**

1. Click menu **"Runtime"** (hoặc **"Thời gian chạy"**)
2. Chọn **"Change runtime type"** (hoặc **"Thay đổi loại thời gian chạy"**)
3. Trong **"Hardware accelerator"** → Chọn **"GPU"**
4. GPU type: Chọn **"T4"** (free tier) hoặc để **"None"** (tự động)
5. Click **"Save"**

**Kiểm tra:**
- Góc trên bên phải phải hiển thị **"Connected"** với icon GPU
- Chạy Cell 1 → Phải thấy thông tin GPU (T4, 16GB)

---

## ▶️ **BƯỚC 5: CHẠY TRAINING**

### **A. Chạy từng cell theo thứ tự:**

**Lần đầu tiên (Setup):**
1. **Cell 1** → Check GPU (phải thấy T4)
2. **Cell 2** → Mount Drive (phải cho phép quyền truy cập)
3. **Cell 3** → Clone repo (~30 giây)
4. **Cell 4** → Install dependencies (~3-5 phút)
5. **Cell 5** → Download data (~30 phút) ⏰ **CHẬM NHẤT**
6. **Cell 6** → Quick test (~5-10 phút)
7. **Cell 7** → Full training (~8-12 giờ cho 8 epochs)

**Click vào mỗi cell và nhấn:**
- **Shift + Enter** để chạy và nhảy xuống cell tiếp theo
- Hoặc click nút ▶️ bên trái cell

### **B. Hoặc chạy tất cả cùng lúc:**
- Menu **"Runtime"** → **"Run all"** (Ctrl+F9)
- ⚠️ **CHỈ dùng nếu chắc chắn code đúng**

---

## 📊 **BƯỚC 6: GIÁM SÁT TRAINING**

### **Xem output trực tiếp:**
- Output hiển thị ngay dưới mỗi cell
- Thấy progress bar với tqdm
- Loss, V-measure mỗi epoch

### **TensorBoard (Optional):**

Thêm cell mới:
```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/radar_deinterleaving/outputs
```

Click vào link hiển thị để xem:
- Loss curve
- Metrics (V-measure, AMI, ARI)
- Training progress

---

## ⏱️ **BƯỚC 7: THỜI GIAN CHỜ**

### **Timeline dự kiến:**

| Bước | Thời gian | Làm gì? |
|------|-----------|---------|
| Setup (Cell 1-4) | ~5-10 phút | Chờ |
| Download data (Cell 5) | ~30 phút | ☕ Uống cà phê |
| Quick test (Cell 6) | ~5-10 phút | Kiểm tra |
| Full training (Cell 7) | **8-12 giờ** | 😴 Ngủ một giấc! |

### **Config training:**

**Quick (Khuyên dùng lần đầu):**
- 3 epochs
- ~2-3 giờ trên T4
- V-measure ~0.75

**Standard (Cho kết quả tốt):**
- 8 epochs  
- ~8-12 giờ trên T4
- V-measure ~0.88

---

## 💾 **BƯỚC 8: LƯU KẾT QUẢ**

### **Checkpoints tự động lưu:**
```
/content/drive/MyDrive/radar_deinterleaving/outputs/
└── run_20260311_123456/
    ├── best_model.pt         ← Model tốt nhất
    ├── checkpoint_epoch_1.pt
    ├── checkpoint_epoch_2.pt
    ├── ...
    ├── training_log.txt
    └── tensorboard/
```

### **Download model về máy:**

Thêm cell mới:
```python
from google.colab import files
import glob

# Tìm best model
runs = sorted(glob.glob('/content/drive/MyDrive/radar_deinterleaving/outputs/run_*'))
if runs:
    best_model = f"{runs[-1]}/best_model.pt"
    print(f"📥 Downloading: {best_model}")
    files.download(best_model)
    print("✓ Download complete!")
```

---

## 🔄 **BƯỚC 9: NẾU COLAB DISCONNECT**

### **Colab free tier giới hạn:**
- ~12-15 giờ/session
- Có thể disconnect ngẫu nhiên

### **Cách resume training:**

1. **Mount Drive lại** (Cell 2)
2. **Clone repo lại** (Cell 3) 
3. **Install dependencies** (Cell 4)
4. **Data đã có trong Drive** → Skip Cell 5
5. **Continue training** với checkpoint:

```python
# Sửa Cell 7 thêm:
--resume_from /content/drive/MyDrive/radar_deinterleaving/outputs/run_XXXXX/checkpoint_epoch_5.pt
```

⚠️ **Note:** Code hiện tại chưa hỗ trợ resume, cần thêm vào `train.py`

---

## 📈 **BƯỚC 10: ĐÁNH GIÁ MODEL**

Thêm cell mới:
```python
import glob

# Tìm best model
runs = sorted(glob.glob('/content/drive/MyDrive/radar_deinterleaving/outputs/run_*'))
best_model = f"{runs[-1]}/best_model.pt"

# Evaluate
!python inference.py \
    --checkpoint {best_model} \
    --data_dir {data_dir} \
    --subset validation \
    --batch_size 8 \
    --save_results {runs[-1]}/validation_results.json

print("✓ Evaluation complete!")
```

**Kết quả mong đợi:**
- V-measure: ~0.88 (8 epochs)
- AMI: ~0.87
- ARI: ~0.81

---

## 🛠️ **TROUBLESHOOTING - CÁC LỖI THƯỜNG GẶP**

### **1. "CUDA Out of Memory"**
```python
# Giảm batch size trong Cell 7:
--batch_size 4  # Thay vì 8
--window_length 500  # Thay vì 1000
```

### **2. "Cannot connect to GPU runtime"**
- Runtime → Disconnect and delete runtime
- Runtime → Connect
- Chạy lại Cell 1 để check GPU

### **3. "Module not found"**
```python
# Cell 4 - Cài lại dependencies
%cd /content/turing-deinterleaving-challenge
!pip install -e . --force-reinstall -q
%cd models_implementation
!pip install -r requirements.txt --force-reinstall -q
```

### **4. "Permission denied" khi mount Drive**
- Nhấn vào link authorization
- Đăng nhập Google
- Copy authorization code
- Paste vào ô trong Colab

### **5. "Repository not found"**
- Kiểm tra repo là **Public** chứ không phải Private
- Hoặc cung cấp GitHub token nếu Private

### **6. Download data quá chậm**
```python
# Giảm số workers
download_dataset(
    save_dir=data_dir,
    subsets=['train', 'validation'],
    max_workers=1  # Thay vì 3
)
```

### **7. Training bị disconnect giữa chừng**

**Cách phòng tránh:**
```javascript
// Chạy code này trong Browser Console (F12):
function KeepAlive(){
    console.log("Keeping connection alive");
    document.querySelector("colab-connect-button").click();
}
setInterval(KeepAlive, 60000);  // Mỗi 1 phút
```

---

## ⚡ **TIPS & TRICKS**

### **1. Tăng tốc training:**
```python
# Cell 7 - Thêm mixed precision
--fp16  # Nếu train.py hỗ trợ
```

### **2. Reduce dataset size (test nhanh):**
```python
# Cell 7
--max_samples 1000  # Chỉ train 1000 samples
```

### **3. Save storage:**
```python
# Chỉ download train set, không validation
download_dataset(
    save_dir=data_dir,
    subsets=['train'],  # Bỏ validation
    max_workers=3
)
```

### **4. Multi-session training:**
- Mở nhiều Colab notebooks
- Train với different configs
- So sánh kết quả

### **5. Notification khi training xong:**

Thêm vào cuối Cell 7:
```python
# Gửi email notification (cần setup)
from google.colab import auth
# ... setup email API
```

---

## 📱 **WORKFLOW KHUYẾN NGHỊ**

### **Ngày 1 - Setup & Quick Test (1-2 giờ):**
```
□ Enable GPU
□ Mount Drive
□ Clone repo
□ Install dependencies
□ Download data (làm song song với việc khác)
□ Quick test
```

### **Ngày 2 - Full Training (để qua đêm):**
```
□ Chạy Cell 7 với config "quick" (3 epochs)
□ Hoặc "standard" (8 epochs) nếu có thời gian
□ Để máy chạy qua đêm
□ Check kết quả sáng hôm sau
```

### **Ngày 3 - Evaluate & Download:**
```
□ Evaluate model
□ Download checkpoint
□ Analyze results
```

---

## 📊 **EXPECTED RESULTS**

### **Quick config (3 epochs):**
```
Training time: ~2-3 hours
V-measure: ~0.75
AMI: ~0.73
```

### **Standard config (8 epochs):**
```
Training time: ~8-12 hours
V-measure: ~0.88
AMI: ~0.87
ARI: ~0.81
```

### **Training progress:**
```
Epoch 1/8
Loss: 2.5 → 1.8
V-measure: 0.45 → 0.62

Epoch 8/8
Loss: ~0.9
V-measure: ~0.88
```

---

## 🎯 **CHECKLIST - TRƯỚC KHI BẮT ĐẦU**

- [ ] Đã đọc hướng dẫn này đầy đủ
- [ ] Đã push code lên GitHub (repo public)
- [ ] Có tài khoản Google với ít nhất 100GB Drive
- [ ] Biết username GitHub của mình
- [ ] Đã enable GPU trong Colab
- [ ] Có khoảng 12-15 giờ cho training (để qua đêm)
- [ ] Internet ổn định

---

## 🆘 **HỖ TRỢ**

### **Nếu gặp vấn đề:**

1. **Check logs:** Scroll xuống output của cell có lỗi
2. **Google error:** Copy error message → Google
3. **Restart runtime:** Runtime → Restart runtime
4. **Clear output:** Edit → Clear all outputs
5. **Fresh start:** Runtime → Disconnect and delete runtime

### **Resources:**
- 📖 [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- 📖 [GitHub Issues](https://github.com/YOUR_USERNAME/turing-deinterleaving-challenge/issues)
- 📖 Paper: arXiv:2503.13476

---

## 🎉 **HOÀN THÀNH!**

Sau khi training xong, bạn sẽ có:
- ✅ Model với V-measure ~0.88
- ✅ Checkpoints saved trong Drive
- ✅ Training logs và metrics
- ✅ Validation results

**Next steps:**
- Test trên tập test
- Fine-tune hyperparameters
- Try different architectures
- Deploy model

---

**Chúc bạn training thành công! 🚀**

Nếu có câu hỏi, tạo issue trên GitHub repo.
