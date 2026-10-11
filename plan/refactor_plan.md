# Kế Hoạch Tinh Gọn Cấu Trúc Mã Nguồn (GTSRB Deep Learning) — BẢN 2

> **Bản 2** đã sửa theo các quyết định đã chốt (10/10). Những thay đổi so với bản 1
> được đánh dấu **[SỬA]**. Đọc mục 0 để nắm nhanh cái gì đổi.

## 0. Tóm tắt thay đổi so với bản 1

1. **[SỬA] Tên file bỏ số đầu** (`train_m1.py` thay vì `02_train_m1.py`) — vì `evaluate.py`
   cần `import` lại class model từ file train, mà tên bắt đầu bằng số thì Python không
   import được.
2. **[SỬA] Giữ `evaluate.py`** (trước gọi `evaluate_comparison.py`) — nạp 5 checkpoint, in
   bảng so sánh Top-1 / Macro-F1 / McNemar / latency, không train lại.
3. **[SỬA] Preprocess xuất `index.csv` + `.npy` cache**, KHÔNG phải `train/val/test.csv` +
   thư mục ảnh lẻ. Giữ nguyên cơ chế `.npy` để số không lệch khỏi báo cáo.
4. **[SỬA] Thêm `assert` chống rò rỉ** ngay trong `preprocess.py` (thay cho file test đã bỏ).
5. **[SỬA] Checkpoint lấy từ `artifacts/runs/`** (5 run chính), KHÔNG phải `artifacts-1/`
   (thư mục đó là 3 run Colab lỗi, số sai, đã xoá).
6. **[SỬA] Ràng buộc thứ tự**: chép model y nguyên → test nạp cả 5 checkpoint → rồi mới xoá `src/`.
7. **[SỬA] Số nhỏ**: GAP giảm **214 lần** (không phải 213); M3 dùng **AdamW cả hai pha**.
8. **[SỬA] Bỏ `src/`, `scripts/`, `tests/`** → các báo cáo tự sinh (`KET_QUA.md`,
   `BAO_CAO.md`) và hình robustness/gradcam/ECE/ablation trở thành **đóng băng** (giữ nội
   dung, không tạo lại được). Chấp nhận có chủ ý.

---

## 1. Mục tiêu và Nguyên tắc

1. **Tập trung vào logic cốt lõi của Deep Learning**: mỗi file đọc là hiểu trọn luồng
   Dataset → Model → Train → Eval, không qua tầng trừu tượng (registry động, config YAML
   lồng nhau, kiểm git, CI).
2. **Độc lập, dễ đọc, dễ nộp**: giảng viên mở một file train là thấy đủ model + vòng huấn
   luyện + đánh giá của model đó.
3. **Bảo toàn 100% dữ liệu và kết quả đã đo**: giữ nguyên `data/`, 5 checkpoint, 41 run,
   bảng McNemar, hình, số trong báo cáo.
4. **Không train lại trong lúc tái cấu trúc**: chỉ sắp xếp và chép code.
5. **[SỬA] Chấp nhận đóng băng**: vì bỏ toàn bộ `src/` + `scripts/`, các tài liệu tự sinh
   và hình phân tích nâng cao sẽ giữ ở dạng tĩnh, không sinh lại được. `evaluate.py` chỉ
   tái tạo được bảng so sánh cơ bản (Top-1/Macro-F1/McNemar/latency).

---

## 2. Quyết định về M3 (gộp 3 backbone vào 1 file) — GIỮ NGUYÊN

> **Gộp ResNet18 / MobileNetV2 / EfficientNet-B0 vào `train_m3.py`**, chọn qua cờ
> `--backbone resnet18 | mobilenetv2 | effnetb0`.

Lý do: phần model M3 thật ra chỉ ~35 dòng (nạp từ `torchvision.models` + thay lớp cuối
thành 43 lớp); cả 3 chung input 224×224, chuẩn hoá ImageNet, và quy trình train 2 pha.
Tách 3 file sẽ nhân bản Dataset + vòng train 2 pha + Early Stopping 3 lần.

---

## 3. Kiến trúc thư mục sau khi tinh gọn  **[SỬA toàn bộ phần tên file]**

```text
gtsrb-model-comparison/
│
├── preprocess.py           # Tải, crop ROI 10%, CLAHE kênh L, split theo track, assert rò rỉ
├── train_m1.py             # Model M1 LeNet + Dataset 48x48 + Train/Eval   (tự chứa)
├── train_m2.py             # Model M2 VGG-Res + Dataset 48x48 + Train/Eval (tự chứa)
├── train_m3.py             # Model M3 Transfer (--backbone) + 224x224 + Train 2 pha
├── evaluate.py             # Nạp 5 checkpoint, in bảng Top-1/Macro-F1/McNemar/latency
│
├── data/                   # BẢO TOÀN
│   ├── raw/                # Ảnh gốc GTSRB (.ppm)
│   └── processed/          # index.csv (có cột split) + images_48_clahe.npy, ...
│
├── checkpoints/            # BẢO TOÀN — 5 file .pt chuyển từ artifacts/runs/
│   ├── m1_lenet_best.pt
│   ├── m2_vggres_best.pt
│   ├── m3_resnet18_best.pt
│   ├── m3_mobilenetv2_best.pt
│   └── m3_effnetb0_best.pt
│
├── results/                # BẢO TOÀN (ĐÓNG BĂNG — không sinh lại được sau khi bỏ src/)
│   ├── BAO_CAO.md          # báo cáo nộp thầy (tĩnh)
│   ├── KET_QUA.md          # toàn bộ 41 run, McNemar, latency p50/p95, ECE (tĩnh)
│   ├── LY_THUYET.md        # lý thuyết chi tiết
│   ├── figures/            # pareto, gradcam_*, robustness_curves, confusion_*, ...
│   └── tables/             # các bảng .csv
│
└── README.md               # Hướng dẫn chạy + giải thích ngắn
```

**Vì sao bỏ số đầu tên file:** `evaluate.py` phải `from train_m1 import M1LeNet` để nạp
checkpoint. `import 02_train_m1` là lỗi cú pháp Python. Mỗi file train đặt class model ở
đầu file và bọc phần chạy train trong `if __name__ == "__main__":`, nên `evaluate.py`
import được class mà không kích hoạt train.

**Thứ tự đọc** (thay cho số thứ tự): ghi trong README — preprocess → train_m1 → train_m2
→ train_m3 → evaluate.

---

## 4. Nội dung chi tiết các file

### `preprocess.py` (~150–200 dòng)
1. Đọc ảnh `.ppm` + annotation ROI.
2. Cắt ROI lề 10%.
3. CLAHE trên kênh **L** của **LAB** (sửa sáng, giữ màu — màu biển báo mang ngữ nghĩa).
4. Resize **48×48**.
5. **Split theo track** bằng `StratifiedGroupKFold`, nhóm = cặp **`(class_id, track_id)`**
   (bắt buộc là cặp — `track_id` đơn bị dùng lại giữa các lớp).
6. **[SỬA] Xuất `index.csv`** (thêm cột `split`) **+ cache `.npy`** (`images_48_clahe.npy`),
   KHÔNG xuất ảnh lẻ / 3 csv riêng.
7. **[SỬA] Assert chống rò rỉ** ngay sau split:
   ```python
   shared = set(train.track_id_full) & set(val.track_id_full)   # track_id_full = (class_id, track_id)
   assert len(shared) == 0, f"RÒ RỈ: {len(shared)} track chung giữa train và val"
   ```

### `train_m1.py` (~220–280 dòng)
* **[SỬA] Class `M1LeNet`** — chép y nguyên từ `src/gtsrb/models/m1_lenet.py`, giữ đúng
  cấu trúc `self.features` (Conv) + `self.classifier` (Linear) để state_dict khớp
  checkpoint (`features.0`, `features.3`, `classifier.1`, `classifier.4`).
  * 2 block Conv(5×5)+MaxPool + 2 Linear (9216 → 256 → 43). Cố ý **không** BN/Residual/Dropout.
  * Điểm nhấn: **97,3%** tham số dồn ở một lớp Linear (9216×256).
* Dataset đọc `.npy` 48×48 từ `index.csv` (lọc `split=='train'`), augmentation (cấm flip
  ngang, cấm đổi hue/saturation).
* Loss CrossEntropy + label smoothing 0.1; Adam lr=1e-3; Cosine Annealing + warmup 3 epoch.
* Early stopping theo **val Macro-F1** (không theo accuracy). Lưu `m1_lenet_best.pt`.
* `if __name__ == "__main__":` bọc phần train.

### `train_m2.py` (~280–350 dòng)
* **[SỬA] Class `M2VggRes` + `ResidualBlock`** — chép y nguyên, giữ cấu trúc
  `self.stem` + `self.stages` + `self.head` để khớp checkpoint (state_dict dùng
  `stem.*`, `stages.*`, `head.*`).
  * ResidualBlock: 2 Conv 3×3 + BN + ReLU + skip (y = F(x)+x).
  * Stem + 4 stage (32→64→128→256). `nn.Dropout2d` tăng dần 0.1→0.3.
  * **Global Average Pooling** thay Flatten+FC → **[SỬA] giảm 214 lần** tham số phần head.
  * Khởi tạo Kaiming/He cho conv.
* Cùng input 48×48, cùng loss/metric với M1 để công bằng. Lưu `m2_vggres_best.pt`.

### `train_m3.py` (~250–320 dòng)
* **[SỬA] Class `M3Transfer`** — chép y nguyên, giữ cấu trúc `self.net` để khớp
  checkpoint (state_dict dùng `net.*`).
  * Cờ `--backbone resnet18 | mobilenetv2 | effnetb0`.
  * Upsample ảnh về **224×224**, chuẩn hoá thống kê **ImageNet**
    (mean [0.485,0.456,0.406], std [0.229,0.224,0.225]).
  * Thay head cuối thành 43 lớp.
* **Train 2 pha**:
  * Pha 1 (epoch 1–3): **freeze** backbone, chỉ train head.
  * Pha 2 (epoch 4+): **unfreeze** toàn mạng, fine-tune với **discriminative learning rate**.
  * **[SỬA] Optimizer: AdamW cả hai pha** (hai pha khác nhau ở freeze/unfreeze + lr, không
    phải ở loại optimizer).
* Lưu `m3_<backbone>_best.pt`.

### `evaluate.py` (~150–250 dòng)  **[SỬA — chi tiết mới]**
* `from train_m1 import M1LeNet`, `from train_m2 import M2VggRes`,
  `from train_m3 import M3Transfer`.
* Nạp 5 checkpoint trong `checkpoints/` bằng `load_state_dict(strict=True)`.
* Chạy inference trên tập test (`split=='test'`), in bảng: **Top-1, Macro-F1, số tham số,
  latency**, và **McNemar** giữa các cặp model (so dự đoán bắt cặp trên cùng tập test).
* Không train lại. Không sinh robustness/gradcam/ECE (các phần đó đã đóng băng ở `results/`).

---

## 5. Bảo toàn tài sản  **[SỬA bảng — đúng đường dẫn checkpoint]**

| Hạng mục | Vị trí hiện tại | Nơi lưu sau tinh gọn |
|---|---|---|
| Dữ liệu | `data/raw`, `data/processed` | Giữ nguyên `data/` |
| **Checkpoint M1** | `artifacts/runs/m1_lenet_s42_20261003_222148/best.pt` | `checkpoints/m1_lenet_best.pt` |
| **Checkpoint M2** | `artifacts/runs/m2_vggres_s42_20261003_223350/best.pt` | `checkpoints/m2_vggres_best.pt` |
| **Checkpoint M3-ResNet18** | `artifacts/runs/m3_resnet18_s42_20261003_225740/best.pt` | `checkpoints/m3_resnet18_best.pt` |
| **Checkpoint M3-MobileNetV2** | `artifacts/runs/m3_mobilenetv2_s42_20261004_004220/best.pt` | `checkpoints/m3_mobilenetv2_best.pt` |
| **Checkpoint M3-EffNet-B0** | `artifacts/runs/m3_effnetb0_s42_20261004_101515/best.pt` | `checkpoints/m3_effnetb0_best.pt` |
| Báo cáo + bảng | `docs/*.md`, `reports/tables/` | `results/` (ĐÓNG BĂNG) |
| Hình | `docs/images/`, `reports/figures/` | `results/figures/` |

> **[SỬA] KHÔNG dùng `artifacts-1/`** — thư mục đó là 3 run Colab lỗi (val Macro-F1 0,94698
> thay vì 0,99957 khi đo lại), đã xoá hôm 09/10.

**Đóng băng — mất code, giữ kết quả:** robustness, Grad-CAM, ECE calibration, export ONNX,
ablation 30 run, và 2 script sinh báo cáo. Hình/bảng của chúng còn trong `results/` nhưng
không tạo lại được sau khi bỏ `src/` + `scripts/`.

---

## 6. Lộ trình triển khai (chỉ sửa code, KHÔNG train)  **[SỬA — thêm ràng buộc thứ tự]**

1. **Giai đoạn 1 — Lưu trữ tài sản (làm TRƯỚC, không xoá gì):**
   - Tạo `checkpoints/`, copy 5 `best.pt` theo bảng mục 5, đổi tên.
   - Tạo `results/`, copy `docs/*.md` + `reports/` vào.

2. **Giai đoạn 2 — `preprocess.py`:**
   - Trích logic crop ROI + CLAHE + `StratifiedGroupKFold` từ `src/gtsrb/data/`.
   - Giữ xuất `index.csv` + `.npy`. Thêm `assert` rò rỉ.

3. **Giai đoạn 3 — 3 file train:**
   - **Chép class model y nguyên** (`M1LeNet` / `M2VggRes`+`ResidualBlock` / `M3Transfer`),
     giữ đúng tên thuộc tính để state_dict khớp.
   - Gắn Dataset + vòng train + Early Stopping vào mỗi file. Bọc `if __name__=="__main__"`.
   - Viết docstring giải thích kiến trúc + ý nghĩa toán học.

4. **Giai đoạn 4 — `evaluate.py`:**
   - Import 3 class model, nạp 5 checkpoint, in bảng so sánh + McNemar.

5. **Giai đoạn 5 — KIỂM TRA (cổng bắt buộc trước khi xoá gì):**  **[SỬA]**
   - **(a)** `load_state_dict(strict=True)` cho **cả 5** checkpoint → phải load sạch.
   - **(b)** Chạy `evaluate.py`, **so số với bảng cũ** trong `results/` → không lệch.
   - Import sạch, không lỗi cú pháp.

6. **Giai đoạn 6 — Xoá cấu trúc cũ (CHỈ sau khi Giai đoạn 5 xanh):**  **[SỬA]**
   - Xoá `src/`, `scripts/`, `tests/`, `configs/`, `Makefile`, `*.sh`, `artifacts/`,
     `notebooks/` (giữ hay bỏ `00_colab_train` tuỳ bạn).
   - Viết lại `README.md` trỏ vào cấu trúc mới.

> **Ràng buộc then chốt:** Giai đoạn 1 (lưu checkpoint) và Giai đoạn 5 (test nạp) phải
> xong TRƯỚC Giai đoạn 6 (xoá `src/`). Xoá `src/` trước là mất đường đối chiếu, hỏng
> checkpoint mà không biết.

---

## 7. Điểm cần bạn duyệt lại

- Cấu trúc 5 file + tên không số — ổn chưa?
- Mục 0 (tóm tắt thay đổi) có thiếu gì không?
- `notebooks/00_colab_train.ipynb`: giữ hay xoá ở Giai đoạn 6? (plan đang để bạn chọn)
