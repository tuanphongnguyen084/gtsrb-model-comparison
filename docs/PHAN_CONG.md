# Phân công nhóm — GTSRB 43 lớp

| Người | Phần việc | Khối lượng |
|---|---|---|
| **Phong Nguyễn** | M1 baseline · mô-đun đánh giá · TTA/ensemble · báo cáo | nhẹ nhất |
| **Hoàng** | M2 · training engine · ablation kiến trúc · nhiều seed | nặng nhất |
| **Phong Trần** | M3 ba backbone · Grad-CAM · đo tốc độ · nén model | tốn compute nhất |
| **Huy** | Đường ống dữ liệu · thí nghiệm rò rỉ · ablation dữ liệu · robustness | vừa |

Mỗi file mã nguồn ghi `CHỦ: <tên>` ở đầu docstring. Mở file là biết của ai.

> **Cách học phần của mình:** mở file code trong cột "Code", đọc docstring đầu file
> (nó giải thích **vì sao** chứ không chỉ **làm gì**), rồi đọc mục lý thuyết tương ứng
> trong [LY_THUYET.md](LY_THUYET.md). Bảng *"Bản đồ: mở file code nào thì đọc mục nào"*
> ở đầu file đó tra được cả hai chiều.

---

## Bản đồ phụ thuộc

```
   HUY ────▶ data/processed/index.csv + cache .npy
   (dữ liệu)         │
                     ├──────────────┬──────────────┬──────────────┐
                     ▼              ▼              ▼              ▼
   HOÀNG ────▶ engine/fit() ──▶ M1 (Phong N.)  M2 (Hoàng)   M3 (Phong T.)
   (engine)          │                │              │              │
                     │                └──────────────┴──────────────┘
                     │                               │
                     ▼                               ▼
                                     artifacts/runs/*/result.json
                                                     │
                      ┌──────────────────────────────┼──────────────┐
                      ▼                              ▼              ▼
             PHONG NGUYỄN                         HUY          PHONG TRẦN
             bảng so sánh · McNemar            robustness    Grad-CAM · tốc độ
             TTA/ensemble · báo cáo                           nén · ONNX
```

**Luật vàng:** không ai sửa file của người khác. Thấy sai thì nhắn người sở hữu.

---

# PHONG NGUYỄN

### Code phụ trách

| File | Nội dung |
|---|---|
| `src/gtsrb/models/m1_lenet.py` | M1 LeNet — mốc tham chiếu |
| `src/gtsrb/eval/metrics.py` | top-1/top-5, macro-F1, per-class |
| `src/gtsrb/eval/stats_tests.py` | McNemar |
| `src/gtsrb/eval/calibration.py` | ECE, reliability diagram |
| `src/gtsrb/eval/confusion.py` | confusion matrix, cặp bị nhầm |
| `src/gtsrb/eval/ensemble.py` | TTA, gộp nhiều model |
| `scripts/evaluate.py`, `make_report.py`, `make_baocao.py` | sinh bảng và báo cáo |

### Khái niệm phải nắm

| Khái niệm | Lý thuyết | Vì sao quan trọng |
|---|---|---|
| Conv vs fully-connected | [2.1](LY_THUYET.md) | nền tảng của mọi CNN |
| **97,3% tham số M1 ở một lớp FC** | [2.6](LY_THUYET.md) | dẫn thẳng tới lý do dùng GAP |
| **Macro-F1 vs accuracy** | [5.2](LY_THUYET.md) | chỉ số chính của cả dự án |
| Top-5 bão hoà trên 43 lớp | [5.1](LY_THUYET.md) | biết chỉ số nào KHÔNG dùng được |
| **McNemar** | [5.3](LY_THUYET.md) | thứ đổi kết luận của dự án |
| ECE và hiệu chỉnh | [5.4](LY_THUYET.md) | kết quả ngược lý thuyết thông thường |
| Confusion matrix | [5.5](LY_THUYET.md) | lỗi đi đâu, không chỉ lỗi bao nhiêu |
| TTA và ensemble | [8](LY_THUYET.md) | cách IDSIA đạt SOTA năm 2011 |

### Lệnh

```bash
make train-m1     # 12 phút
make eval         # bảng so sánh + McNemar + ECE + confusion
make report       # docs/KET_QUA.md
make baocao       # docs/BAO_CAO.md
```

---

# HOÀNG

### Code phụ trách

| File | Nội dung |
|---|---|
| `src/gtsrb/models/m2_vggres.py` | M2 — BN, residual, spatial dropout, GAP |
| `src/gtsrb/models/registry.py` | `build_model()` + hợp đồng model |
| `src/gtsrb/engine/train.py` | `fit()` dùng chung cho cả 3 model |
| `src/gtsrb/engine/losses.py` | cross-entropy + label smoothing |
| `src/gtsrb/engine/schedulers.py` | Adam/AdamW, cosine + warmup |
| `src/gtsrb/engine/callbacks.py` | early stopping, checkpoint, log |
| `scripts/train.py`, `run_seeds.py` | CLI huấn luyện |

### Khái niệm phải nắm

| Khái niệm | Lý thuyết | Vì sao quan trọng |
|---|---|---|
| **Hai conv 3×3 thay một 5×5** | [2.2](LY_THUYET.md) | luận điểm trung tâm của VGG |
| **BatchNorm** | [2.3](LY_THUYET.md) | kèm phản biện Santurkar 2018 |
| **Residual** | [2.4](LY_THUYET.md) | giải quyết *degradation*, không phải overfit |
| **SpatialDropout vs Dropout** | [2.5](LY_THUYET.md) | vì sao dropout thường vô dụng trên conv |
| **Global Average Pooling** | [2.6](LY_THUYET.md) | lý do M2 ít tham số hơn M1 |
| Label smoothing | [3.1](LY_THUYET.md) | và vì sao nó *sửa quá tay* trên GTSRB |
| Cosine + warmup | [3.3](LY_THUYET.md) | thay cho step decay |
| Early stopping theo macro-F1 | [3.4](LY_THUYET.md) | chọn checkpoint theo đúng chỉ số |
| **Vì sao một seed là không đủ** | [9](LY_THUYET.md) | hạn chế phương pháp lớn nhất của nhóm |

### Lệnh

```bash
make train-m2                                              # 24 phút
python scripts/run_ablation.py --axes components scaling label_smoothing --budget
make seeds                                                 # 3 seed, gộp mean±std
```

---

# PHONG TRẦN

### Code phụ trách

| File | Nội dung |
|---|---|
| `src/gtsrb/models/m3_transfer.py` | 3 backbone, discriminative LR, freeze/unfreeze |
| `src/gtsrb/explain/gradcam.py` | Grad-CAM |
| `src/gtsrb/deploy/speed.py` | latency p50/p95, FLOPs |
| `src/gtsrb/deploy/export.py` | lượng tử hoá int8, TorchScript, ONNX |
| `scripts/benchmark_speed.py`, `make_gradcam.py`, `export_edge.py` | CLI |

### Khái niệm phải nắm

| Khái niệm | Lý thuyết | Vì sao quan trọng |
|---|---|---|
| **Vì sao transfer learning hiệu quả** | [4.1](LY_THUYET.md) | đặc trưng phân tầng |
| **Vì sao phải upsample lên 224** | [4.2](LY_THUYET.md) | downsample 32×, và nó KHÔNG thêm thông tin |
| **Discriminative learning rate** | [4.3](LY_THUYET.md) | catastrophic forgetting |
| **Hai pha freeze → unfreeze** | [4.4](LY_THUYET.md) | có bằng chứng thực nghiệm: 67,9% → 96,4% |
| Chuẩn hoá bằng ImageNet stats | [4.5](LY_THUYET.md) | lỗi im lặng nếu quên |
| **FLOPs ≠ latency** | [6.4](LY_THUYET.md) | chênh **64 lần**, đo được cả hai chiều |
| **Đo latency cho đúng** | [6.5](LY_THUYET.md) | không đồng bộ thiết bị là sai hàng chục lần |
| **Grad-CAM + giới hạn** | [6.6](LY_THUYET.md) | M2 cho heatmap thô nhất, phản trực giác |
| Lượng tử hoá int8 | [7](LY_THUYET.md) | kiến trúc quyết định nén có hiệu quả không |

### Lệnh

```bash
make train-m3                                   # 3 backbone — nên chạy Colab
python scripts/run_ablation.py --axes resolution --budget
make gradcam
make speed                                      # MÁY PHẢI RẢNH khi đo
make edge                                       # nén int8 + ONNX + bảng đánh đổi
```

---

# HUY

### Code phụ trách

| File | Nội dung |
|---|---|
| `src/gtsrb/data/download.py` | tải, dựng index.csv, tách track_id |
| `src/gtsrb/data/preprocess.py` | ROI, 4 chế độ cân bằng sáng, resize |
| `src/gtsrb/data/split.py` | **split theo track** — chống rò rỉ |
| `src/gtsrb/data/transforms.py` | augmentation, danh sách cấm |
| `src/gtsrb/data/dataset.py` | Dataset, cache, chuẩn hoá |
| `src/gtsrb/robustness/corruptions.py` | 5 loại nhiễu × 5 mức |
| `src/gtsrb/robustness/benchmark.py` | quét robustness, mCE |
| `scripts/prepare_data.py`, `run_robustness.py`, `run_leakage_experiment.py` | CLI |

### Khái niệm phải nắm

| Khái niệm | Lý thuyết | Vì sao quan trọng |
|---|---|---|
| **★ Rò rỉ dữ liệu do cấu trúc track** | [1.1](LY_THUYET.md) | đóng góp kỹ thuật lớn nhất của nhóm |
| Histogram equalization và CLAHE | [1.2](LY_THUYET.md) | vì sao làm trên kênh L chứ không RGB |
| **Hai phép augmentation bị cấm** | [1.3](LY_THUYET.md) | 4 cặp lớp là ảnh gương của nhau |
| Mất cân bằng lớp 10,7:1 | [1.4](LY_THUYET.md) | can thiệp ở chỉ số, không ở dữ liệu |
| **Robustness nghĩa là gì** | [6.1](LY_THUYET.md) | luật: nhiễu CHỈ áp lúc test |
| Mô hình vật lý của 5 loại nhiễu | [6.2](LY_THUYET.md) | tán xạ khí quyển, nhiễu Poisson |
| **Relative robustness** | [6.3](LY_THUYET.md) | tách "giỏi sẵn" khỏi "bền" |

### Lệnh

```bash
make data                                       # tải + tiền xử lý + split
make leakage                                    # ĐO phần accuracy ảo (train 2 lần)
python scripts/run_ablation.py --axes augmentation preprocess --budget
make robustness
```

---

## Quy ước chung

```bash
git switch -c feat/huy-data        # hoặc feat/hoang-m2, feat/phongtran-m3, feat/phong-eval
pip install nbstripout && nbstripout --install
make test                          # 109 test, PHẢI xanh trước khi push
```

Notebook là JSON nên merge rất khó → **mỗi người một notebook riêng**.

Gặp lỗi lạ → ghi một dòng vào [SU_CO.md](SU_CO.md). File đó đã có **9 nhóm sự cố gặp
thật**, đọc trước khi code tiết kiệm được vài giờ.

## Việc còn lại

| # | Việc | Ai | Thời gian |
|---|---|---|---|
| 1 | **30 run ablation** — đề bài yêu cầu | Hoàng 12 · P.Trần 9 · Huy 8 | ~1,5 giờ/người |
| 2 | **3 seed** cho M2 và ResNet18 | Hoàng | ~2 giờ |
| 3 | **Thí nghiệm rò rỉ** — `make leakage` | Huy | 25 phút |
| 4 | Slide | Phong Nguyễn | — |
