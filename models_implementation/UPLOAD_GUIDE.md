# 📤 Hướng Dẫn Upload Data & Code Lên Colab

## 🎯 **TL;DR - Tóm Tắt Nhanh:**

### ❌ **KHÔNG CẦN upload data từ máy local**
### ✅ **CHỈ CẦN upload code (2 cách)**

---

## 📊 **Data (Dataset) - KHÔNG CẦN UPLOAD**

### **Tại sao không cần upload data?**

1. **Dataset rất lớn**: ~50-100 GB
2. **Upload mất nhiều giờ**: 3-10 giờ tùy internet
3. **Colab có sẵn download từ HuggingFace**: Tự động, nhanh hơn

### **Data sẽ được tải tự động trong notebook:**

```python
# Cell này có sẵn trong Colab_Training.ipynb
download_dataset(
    save_dir='/content/drive/MyDrive/deinterleaving_project/data',
    subsets=['train', 'validation'],
    hf_token=your_token,
)
```

**⏱️ Thời gian tải:** ~20-30 phút (nhanh hơn upload nhiều!)

**💾 Tải 1 lần, dùng mãi mãi:**
- Lần 1: Tải vào Google Drive (~30 phút)
- Lần sau: Dùng lại data trong Drive (0 phút)

---

## 💻 **Code (Models Implementation) - CẦN UPLOAD**

### **Chọn 1 trong 2 cách:**

---

## 📋 **OPTION 1: Upload Files Trực Tiếp** (Dễ nhất, khuyên dùng!)

### **Bước 1: Nén folder code (Windows)**

Mở PowerShell:
```powershell
cd "g:\PHÒNG TH\Nghiên cứu\turing-deinterleaving-challenge"

# Tạo file ZIP
Compress-Archive -Path models_implementation -DestinationPath models_implementation.zip

# Check file size
(Get-Item models_implementation.zip).Length / 1MB
# → Should be ~1-2 MB
```

### **Bước 2: Upload lên Colab**

1. Mở `Colab_Training.ipynb` trong Google Colab
2. Click vào **folder icon** (📁) bên trái sidebar
3. Click **Upload** (icon mũi tên lên)
4. Chọn file `models_implementation.zip`
5. Đợi upload (~1-2 phút)

### **Bước 3: Giải nén trong Colab**

Thêm cell này vào đầu notebook:

```python
# Extract uploaded code
!unzip -q models_implementation.zip
%cd models_implementation

# Clone base repository (for dataset utilities only)
%cd ..
!git clone https://github.com/egunn-turing/turing-deinterleaving-challenge.git temp_repo
%cd temp_repo
!pip install -e . -q

# Install model dependencies
%cd /content/models_implementation
!pip install -r requirements.txt -q

print("✓ Setup complete!")
```

### **Bước 4: Update đường dẫn data**

Sửa cell download data:
```python
# Change this line:
sys.path.insert(0, '/content/turing-deinterleaving-challenge/src')

# To this:
sys.path.insert(0, '/content/temp_repo/src')
```

### **✅ Xong! Run notebook như bình thường**

---

## 🐙 **OPTION 2: Push Lên GitHub** (Chuyên nghiệp, dài hơn)

### **Bước 1: Tạo GitHub Repository**

1. Đi tới [github.com](https://github.com)
2. Click **New repository**
3. Đặt tên: `turing-deinterleaving-challenge` (hoặc tên khác)
4. Click **Create repository**

### **Bước 2: Push code từ local**

Mở PowerShell:
```powershell
cd "g:\PHÒNG TH\Nghiên cứu\turing-deinterleaving-challenge"

# Initialize git (nếu chưa có)
git init

# Add all files
git add .

# Commit
git commit -m "Add Transformer model implementation"

# Link to your GitHub repo
git remote add origin https://github.com/YOUR_USERNAME/turing-deinterleaving-challenge.git

# Push
git push -u origin main
```

**⚠️ Lưu ý:** Không push folder `data/` (quá lớn). Đã có `.gitignore` loại bỏ nó.

### **Bước 3: Clone trong Colab**

Notebook đã có sẵn cell này:
```python
# Just change the URL to your repo
GITHUB_REPO = "https://github.com/YOUR_USERNAME/turing-deinterleaving-challenge.git"
!git clone {GITHUB_REPO}
```

### **Ưu điểm của Option 2:**
- ✅ Code luôn đồng bộ
- ✅ Dễ update (git pull)
- ✅ Version control
- ✅ Chia sẻ với người khác

---

## 🔍 **So Sánh 2 Options**

| Feature | Option 1: Upload ZIP | Option 2: GitHub |
|---------|---------------------|-------------------|
| **Độ khó** | ⭐ Dễ | ⭐⭐ Trung bình |
| **Thời gian setup** | ~5 phút | ~15 phút (lần đầu) |
| **Thời gian sau này** | ~5 phút (upload lại) | ~1 phút (git pull) |
| **Version control** | ❌ Không | ✅ Có |
| **Chia sẻ** | ❌ Khó | ✅ Dễ |
| **Cần Git?** | ❌ Không | ✅ Có |
| **Khuyên dùng cho** | Người mới, thử nhanh | Dự án lâu dài |

---

## 🎯 **Workflow Đầy Đủ (Từ Đầu Đến Cuối)**

### **Lần Chạy Đầu Tiên:**

#### **1. Chuẩn bị trên máy Local (5 phút)**
```powershell
# Nén code
cd "g:\PHÒNG TH\Nghiên cứu\turing-deinterleaving-challenge"
Compress-Archive -Path models_implementation -DestinationPath models_implementation.zip
```

#### **2. Mở Colab (1 phút)**
- Đi tới [colab.research.google.com](https://colab.research.google.com)
- Upload `Colab_Training.ipynb`
- Runtime → Change runtime type → GPU (T4)

#### **3. Mount Google Drive (1 phút)**
```python
from google.colab import drive
drive.mount('/content/drive')
```

#### **4. Upload code ZIP (2 phút)**
- Sidebar → Upload `models_implementation.zip`
- Chờ upload xong

#### **5. Setup trong Colab (10 phút)**
```python
# Cell 1: Extract code
!unzip -q models_implementation.zip

# Cell 2: Install dependencies
!git clone https://github.com/egunn-turing/turing-deinterleaving-challenge.git temp_repo
%cd temp_repo
!pip install -e . -q
!pip install -r /content/models_implementation/requirements.txt -q
```

#### **6. Download data (30 phút)**
```python
# Cell 3: Download dataset (chạy 1 lần, lưu vào Drive)
from turing_deinterleaving_challenge import download_dataset
download_dataset(
    save_dir='/content/drive/MyDrive/deinterleaving_project/data',
    subsets=['train', 'validation'],
)
```

#### **7. Training (8-12 giờ)**
```python
# Cell 4: Start training
!python /content/models_implementation/train.py \
    --data_dir /content/drive/MyDrive/deinterleaving_project/data \
    --output_dir /content/drive/MyDrive/deinterleaving_project/outputs \
    --batch_size 8 \
    --num_epochs 8
```

**⏱️ Tổng thời gian setup lần đầu: ~50 phút**

---

### **Lần Chạy Thứ 2+ (Nhanh hơn):**

#### **1. Upload code mới (nếu có thay đổi) (2 phút)**
- Upload `models_implementation.zip` mới

#### **2. Setup (5 phút)**
```python
!unzip -o models_implementation.zip  # -o = overwrite
%cd /content/models_implementation
```

#### **3. Training ngay (0 delay!)**
```python
# Data đã có trong Drive rồi, không cần download
!python train.py \
    --data_dir /content/drive/MyDrive/deinterleaving_project/data \
    ...
```

**⏱️ Tổng thời gian setup lần sau: ~7 phút**

---

## 🔧 **Template Notebook - Copy & Paste**

Tạo notebook mới với code này:

```python
# ========== CELL 1: CHECK GPU ==========
!nvidia-smi
import torch
print(f"CUDA: {torch.cuda.is_available()}")

# ========== CELL 2: MOUNT DRIVE ==========
from google.colab import drive
drive.mount('/content/drive')

# ========== CELL 3: UPLOAD CODE ==========
# MANUAL STEP: Upload models_implementation.zip using file browser
# Then run:
!unzip -q models_implementation.zip
print("✓ Code extracted")

# ========== CELL 4: INSTALL DEPENDENCIES ==========
# Clone base repo for dataset utilities
!git clone -q https://github.com/egunn-turing/turing-deinterleaving-challenge.git temp_repo
%cd temp_repo
!pip install -e . -q

# Install model dependencies
!pip install -r /content/models_implementation/requirements.txt -q
print("✓ Dependencies installed")

# ========== CELL 5: DOWNLOAD DATA (ONCE) ==========
import sys
sys.path.insert(0, '/content/temp_repo/src')
from turing_deinterleaving_challenge import download_dataset
from pathlib import Path

data_dir = Path('/content/drive/MyDrive/deinterleaving_project/data')
data_dir.mkdir(parents=True, exist_ok=True)

# Only run if data doesn't exist
if not (data_dir / 'train').exists():
    print("📥 Downloading data (30 min)...")
    download_dataset(save_dir=data_dir, subsets=['train', 'validation'])
else:
    print("✓ Data already downloaded, skipping")

# ========== CELL 6: QUICK TEST ==========
%cd /content/models_implementation
!python quick_train.py
print("✓ Quick test passed!")

# ========== CELL 7: FULL TRAINING ==========
output_dir = '/content/drive/MyDrive/deinterleaving_project/outputs'

!python train.py \
    --data_dir {data_dir} \
    --output_dir {output_dir} \
    --batch_size 8 \
    --num_epochs 8 \
    --learning_rate 0.0001 \
    --window_length 1000 \
    --validate_every 2 \
    --save_every 1 \
    --num_workers 2

print("🎉 Training complete!")

# ========== CELL 8: EVALUATE ==========
import glob
runs = sorted(glob.glob(f"{output_dir}/run_*"))
best_model = f"{runs[-1]}/best_model.pt"

!python inference.py \
    --checkpoint {best_model} \
    --data_dir {data_dir} \
    --subset validation \
    --batch_size 8

# ========== CELL 9: DOWNLOAD MODEL ==========
from google.colab import files
files.download(best_model)
print("✓ Model downloaded to your computer!")
```

**Copy toàn bộ vào 1 notebook mới → Run từng cell theo thứ tự!**

---

## ❓ **FAQ - Câu Hỏi Thường Gặp**

### **Q1: Có bắt buộc phải dùng Google Drive không?**
**A:** Không bắt buộc, nhưng MẠNH MẼ khuyên dùng vì:
- ✅ Data/checkpoints không mất khi Colab disconnect
- ✅ Reuse data cho lần chạy sau (không tải lại)
- ✅ Checkpoints auto-save mỗi epoch

### **Q2: Upload ZIP hay push GitHub?**
**A:** 
- **Người mới / thử nhanh**: Upload ZIP (dễ hơn)
- **Dự án dài hạn / làm nhóm**: GitHub (chuyên nghiệp hơn)

### **Q3: File ZIP bao nhiêu MB?**
**A:** ~1-2 MB (rất nhỏ, upload <1 phút)

### **Q4: Có cần upload folder `data/` không?**
**A:** **KHÔNG!** Folder đó >50 GB, upload mất cả ngày. Colab tự tải.

### **Q5: Upload mất bao lâu?**
**A:** 
- Code ZIP: ~1-2 phút
- Data download (trong Colab): ~30 phút
- **Tổng: ~35 phút** (lần đầu)

### **Q6: Lần sau phải upload lại không?**
**A:** 
- **Code**: Có (nếu thay đổi code) - ~2 phút
- **Data**: KHÔNG (đã lưu trong Drive) - 0 phút

### **Q7: Colab disconnect giữa chừng thì sao?**
**A:** 
- Checkpoints đã save trong Drive → Load lại checkpoint cuối
- Data đã tải → Không cần tải lại
- Code extract lại (~1 phút) → Continue training

---

## 🎯 **Checklist - Đảm Bảo Thành Công**

### **Trước khi bắt đầu:**
- [ ] Đã tạo file `models_implementation.zip` (~1-2 MB)
- [ ] Đã có tài khoản Google Drive (ít nhất 100 GB free)
- [ ] Đã có GPU enabled trong Colab (Runtime → Change type → T4)
- [ ] Đã có HuggingFace token (optional, nhưng tốt hơn)

### **Trong Colab:**
- [ ] Drive mounted (`/content/drive/MyDrive/` hiển thị)
- [ ] Code extracted (`/content/models_implementation/` tồn tại)
- [ ] Dependencies installed (không có lỗi pip)
- [ ] Data downloaded (`/content/drive/.../data/train/` có files .h5)
- [ ] Quick test passed (V-measure ~0.3-0.4 sau 1 epoch)

### **Sau khi training:**
- [ ] `best_model.pt` tồn tại trong output folder
- [ ] V-measure ~0.88 trên validation set
- [ ] Downloaded model về máy local

---

## 🚀 **Bắt Đầu Ngay!**

### **Quick Start Command (PowerShell):**
```powershell
# Step 1: Create ZIP (on your computer)
cd "g:\PHÒNG TH\Nghiên cứu\turing-deinterleaving-challenge"
Compress-Archive -Path models_implementation -DestinationPath models_implementation.zip -Force

# Step 2: Open Colab
start "https://colab.research.google.com/"

# Step 3: Upload Colab_Training.ipynb and models_implementation.zip
# Step 4: Run all cells!
```

---

## 📊 **Tóm Tắt Thời Gian**

| Task | Duration | Frequency |
|------|----------|-----------|
| Nén code (local) | 1 phút | Mỗi lần sửa code |
| Upload ZIP (Colab) | 2 phút | Mỗi lần sửa code |
| Install deps | 10 phút | Mỗi session |
| Download data | 30 phút | **CHỈ 1 LẦN** |
| Quick test | 10 phút | Mỗi session (optional) |
| Full training | 8-12 giờ | 1 lần |
| **TOTAL (lần đầu)** | **~9-12.5 giờ** | - |
| **TOTAL (lần sau)** | **~8-12 giờ** | (không tải data) |

---

## ✅ **Ready To Go!**

**Bạn chỉ cần:**
1. ✅ Nén folder `models_implementation/` → ZIP (1 phút)
2. ✅ Upload ZIP lên Colab (2 phút)
3. ✅ Run notebook (10 giờ)
4. ✅ Nhận model với V-measure ~0.88! 🎉

**KHÔNG CẦN upload data! colab tự tải!**

Good luck! 🚀
