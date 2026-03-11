# ⚡ COLAB TRAINING - QUICK REFERENCE

## 📝 TÓM TẮT NHANH 5 PHÚT

### Bước 1: Mở Colab
```
https://colab.research.google.com
```

### Bước 2: Enable GPU
```
Runtime → Change runtime type → GPU → Save
```

### Bước 3: Copy-Paste Code

**Option A - Script tự động (KHUYÊN DÙNG):**
```python
# 1. Tạo cell mới
# 2. Copy toàn bộ file colab_quick_start.py vào cell đó
# 3. SỬA DÒNG: GITHUB_USERNAME = "YOUR_USERNAME"
# 4. Chạy cell (Shift+Enter)
# 5. Chờ 30 phút (download data) + 3-12 giờ (training)
```

**Option B - Manual từng bước:**
Xem file `HUONG_DAN_COLAB.md` để biết chi tiết

---

## ⏱️ TIMELINE

| Bước | Thời gian | Mô tả |
|------|-----------|-------|
| Setup | 5-10 phút | Mount Drive, clone repo, install |
| Download data | 30 phút | **☕ Đợi** |
| Quick test | 5-10 phút | Verify |
| Training | 3-12 giờ | **😴 Ngủ** |

---

## 🎛️ CONFIGS

### Quick (Test nhanh)
```python
NUM_EPOCHS = 3      # ~3 giờ
# V-measure: ~0.75
```

### Standard (Production)
```python
NUM_EPOCHS = 8      # ~12 giờ
# V-measure: ~0.88
```

---

## 🔧 COMMANDS THƯỜNG DÙNG

### Check GPU
```python
!nvidia-smi
```

### Mount Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Clone Repo
```python
!git clone https://github.com/YOUR_USERNAME/turing-deinterleaving-challenge.git
%cd turing-deinterleaving-challenge
```

### Install
```python
!pip install -e . -q
%cd models_implementation
!pip install -r requirements.txt -q
```

### Train
```python
!python train.py \
    --data_dir /content/drive/MyDrive/radar_deinterleaving/data \
    --output_dir /content/drive/MyDrive/radar_deinterleaving/outputs \
    --batch_size 8 \
    --num_epochs 8
```

### Evaluate
```python
!python inference.py \
    --checkpoint outputs/run_XXX/best_model.pt \
    --data_dir data/ \
    --subset validation
```

### Download Model
```python
from google.colab import files
files.download('outputs/run_XXX/best_model.pt')
```

---

## ❌ TROUBLESHOOTING

### GPU không có
```
Runtime → Change runtime type → GPU → Save
Runtime → Restart runtime
```

### Out of Memory
```python
--batch_size 4      # Giảm từ 8
--window_length 500 # Giảm từ 1000
```

### Disconnect
```javascript
// Trong Browser Console (F12):
function keepAlive() {
    document.querySelector("colab-connect-button").click();
}
setInterval(keepAlive, 60000);
```

### Data không tải được
```python
# Giảm workers
download_dataset(..., max_workers=1)
```

---

## 📥 LOCATIONS

### Data
```
Drive: /content/drive/MyDrive/radar_deinterleaving/data/
Local: /content/data/
```

### Outputs
```
Drive: /content/drive/MyDrive/radar_deinterleaving/outputs/
Local: /content/outputs/
```

### Model
```
outputs/run_YYYYMMDD_HHMMSS/best_model.pt
```

---

## 📊 EXPECTED RESULTS

### Quick (3 epochs, ~3h)
```
V-measure: 0.75
AMI: 0.73
Loss: ~1.5
```

### Standard (8 epochs, ~12h)
```
V-measure: 0.88
AMI: 0.87
ARI: 0.81
Loss: ~0.9
```

---

## 🎯 CHECKLIST

**Trước khi train:**
- [ ] GPU enabled (thấy T4 trong nvidia-smi)
- [ ] Drive mounted (thấy MyDrive/)
- [ ] Repo cloned (thấy folder)
- [ ] Dependencies installed (không lỗi)
- [ ] Data downloaded (thấy train/ validation/)

**Sau khi train:**
- [ ] best_model.pt tồn tại
- [ ] V-measure ≥ 0.75
- [ ] Downloaded model về máy

---

## 🚀 ONE-LINER (Tự động hoàn toàn)

```python
# Copy script này, sửa YOUR_USERNAME, rồi chạy!
# Xem file: colab_quick_start.py
```

---

## 📚 DOCS

- Chi tiết: `HUONG_DAN_COLAB.md`
- Script: `colab_quick_start.py`
- Notebook: `Colab_Training.ipynb`
- Upload guide: `UPLOAD_GUIDE.md`

---

**That's it! Training thành công! 🎉**
