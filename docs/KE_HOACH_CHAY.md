# Kế hoạch chạy — tiết kiệm tài nguyên

> Mọi con số thời gian dưới đây là **đo thật** trên Apple M1 Pro 16 GB (MPS),
> torch 2.14.1, không phải ước lượng.

---

## 1. Thời gian đã đo

| Việc | Thời gian | Ghi chú |
|---|---|---|
| Tải GTSRB (364 MB zip → 1,1 GB) | ~6 phút | làm **một lần** |
| Dựng `index.csv` + split theo track | 15 giây | |
| Tiền xử lý + cache 51.839 ảnh (48px, CLAHE) | **11 giây** | ra file 342 MB |
| **M1** LeNet, 30 epoch | **11,7 phút** | 23 s/epoch |
| **M2** VGG-res, 40 epoch | **23,8 phút** | 35 s/epoch |
| **M3** ResNet18, 15 epoch | **~60 phút** | 107 s/epoch (freeze) · 270 s/epoch (finetune) |
| Đánh giá 1 model trên test | ~20 giây | |
| Grad-CAM 1 model | ~2 giây | |
| Robustness 1 model (26 cấu hình, toàn tập test) | ~2 phút (M1) · ~35 phút (M3 @224) | |
| Đo tốc độ 1 model | ~1 phút | |

**Điểm nghẽn rõ ràng: M3 ở 224×224 và bộ ablation 30 run.**

---

## 2. Bốn cách tiết kiệm, xếp theo mức tiết kiệm

### ★ Cách 1 — Chia việc cho 4 máy (tiết kiệm nhiều nhất)

Nhóm có **4 máy**. Chạy tuần tự trên một máy là lãng phí nhất.

| Người | Chạy gì trên máy mình | Tổng compute |
|---|---|---|
| **Phong Nguyễn** | M1 + `make eval` + `make report` | **~15 phút** |
| **Hoàng** | M2 + 3 trục ablation (components, scaling, label_smoothing) | ~3,5 giờ |
| **Phong Trần** | M3 ×3 backbone (**Colab**) + ablation resolution + speed + gradcam | ~2 giờ |
| **Huy** | data + thí nghiệm rò rỉ + 2 trục ablation (aug, preprocess) + robustness | ~3 giờ |

Chạy tuần tự một máy: **~10 giờ**. Chạy song song 4 máy: **~3,5 giờ** (bằng người lâu nhất).

**Mỗi người cần gì trên máy mình:**
```bash
make setup-mac      # hoặc setup-colab
make data           # ~6 phút, mỗi máy làm một lần
```
Dữ liệu thô 1,1 GB + cache 342 MB. Không cần chia sẻ file — tải lại nhanh hơn gửi cho nhau.

---

### ★ Cách 2 — Ablation ở chế độ ngân sách (tiết kiệm ~60%)

30 thực nghiệm × 40 epoch ≈ **16 giờ**. Nhưng mục đích của ablation là **XẾP HẠNG** các
biến thể, không phải lấy con số cuối cùng của từng biến thể. Thứ tự giữa các biến thể ổn
định từ lâu trước khi accuracy tuyệt đối hội tụ.

```bash
make ablation-budget      # 15 epoch + patience 4  -> ~5 giờ (chia 4 máy: ~1,5 giờ)
```

**Quy trình đúng — ba bước:**
1. `--budget` xếp hạng 30 biến thể → ~5 giờ
2. Chọn cấu hình **thắng** của mỗi trục
3. Chạy lại **riêng chúng** ở độ dài đầy đủ để lấy số báo cáo → ~1 giờ

Tổng ~6 giờ thay vì 16 giờ, kết luận không đổi.

> ⚠️ **Phải nói rõ trong báo cáo** rằng bảng ablation dùng ngân sách 15 epoch.
> **Không được trộn** số 15 epoch với số 40 epoch trong cùng một bảng — đó là so sánh
> không công bằng, và giảng viên sẽ hỏi đúng chỗ này.

---

### ★ Cách 3 — Đưa M3 lên Colab (tiết kiệm ~45 phút/backbone)

**Đã có notebook sẵn: [`notebooks/00_colab_train.ipynb`](../notebooks/00_colab_train.ipynb)**
— 8 ô, chạy từ trên xuống.

T4 của Colab nhanh hơn MPS khoảng **3–4 lần** ở 224×224. M1/M2 chạy ở 48px thì MPS đủ
nhanh, **không cần** Colab (upload mất công hơn là train tại chỗ).

**Ba bước:**

```bash
# 1. Trên máy bạn — đóng gói code (chỉ 240 KB, không gồm dữ liệu)
make zip
```

```
# 2. Mở notebooks/00_colab_train.ipynb trên Colab
#    (colab.research.google.com -> Upload -> chọn file)
#    Runtime -> Change runtime type -> T4 GPU -> Save
```

```
# 3. Chạy 8 ô từ trên xuống. Ô 3 sẽ hỏi file -> chọn gtsrb_code.zip
```

**Notebook tự lo ba chuyện quan trọng:**

| Chuyện | Cách xử lý |
|---|---|
| Mất session giữa chừng | `artifacts/` được **symlink sang Google Drive**, nên checkpoint sống sót. Ô 7 tự tìm run dở dang và in ra lệnh `--resume` |
| Dữ liệu | Tải thẳng trên Colab (~4 phút, mạng ~100 MB/s) chứ **không** upload từ máy, và để ở `/content` chứ **không** để trên Drive — đọc từ Drive lúc train chậm hơn nhiều |
| Cài gói | Chỉ cài `thop`. **Tuyệt đối không `pip install torch`** — sẽ kéo về bản không khớp CUDA và bạn train trên CPU mà không biết (chậm hơn ~20 lần, không báo lỗi) |

**Hai tham số chỉ bật trên Colab:**

```bash
python scripts/train.py --config configs/m3_resnet18.yaml \
    --set train.num_workers=2 train.batch_size=96
```

| Tham số | Colab | macOS | Vì sao |
|---|---|---|---|
| `train.num_workers` | **2** | 0 | macOS dùng `fork` + OpenCV hay treo DataLoader |
| `train.batch_size` | **96** | 64 | T4 có 16 GB VRAM riêng; MPS chia sẻ 16 GB RAM với hệ thống |
| `train.amp` | tự **bật** | tự **tắt** | `fit()` tự quyết theo `device.type`, không cần làm gì |

> ⚠️ **Đừng chạy `make speed` trên Colab.** Latency của T4 không nói gì về thiết bị biên.
> Đo tốc độ phải làm trên **máy đích**, và lúc đo phải tắt mọi thứ khác đang chạy.

> **Ghi vào báo cáo:** M3 train trên Colab T4, M1/M2 trên M1 Pro (MPS). Điều này **không**
> ảnh hưởng so sánh accuracy (cùng dữ liệu, seed, `fit()`), nhưng **có** ảnh hưởng cột
> `train_seconds` — không được so thời gian train giữa hai thiết bị khác nhau.

### Cách 4 — `--resume` khi mất session Colab

Colab miễn phí hay ngắt giữa chừng. Mất session ở epoch 12/15 của M3 là mất ~50 phút.

```bash
python scripts/train.py --config configs/m3_resnet18.yaml \
    --resume artifacts/runs/m3_resnet18_s42_20261003_225740
```

Nó khôi phục **trọng số + moment của Adam + vị trí scheduler + trạng thái early stopping**,
rồi chạy tiếp từ đúng epoch đã dừng, ghi vào **cùng** thư mục run.

> Lưu ý trung thực: resume **không** hoàn toàn bằng chạy liền một mạch, vì thứ tự shuffle
> của DataLoader khác đi. Chênh lệch nhỏ, không ảnh hưởng kết luận. Nhưng con số **chính
> thức** đưa vào báo cáo thì nên train liền một mạch.

---

## 3. Những gì KHÔNG nên làm để tiết kiệm

| Cám dỗ | Vì sao đừng |
|---|---|
| Giảm số ảnh train (`--subset`) để chạy nhanh | Số liệu không còn so sánh được; `--subset` CHỈ để smoke test |
| Bỏ bớt seed, chỉ chạy 1 lần | Không biết chênh lệch 0,2 điểm là thật hay nhiễu khởi tạo |
| Dùng tập test để chọn siêu tham số | Test thành val, con số báo cáo không còn là ước lượng không chệch |
| Rút gọn robustness bằng `--max-samples` rồi đưa số vào báo cáo | Phải ghi rõ là tập con, hoặc chạy toàn bộ |
| Đo latency trong lúc máy đang train model khác | Số sai hoàn toàn, mọi kết luận về triển khai biên vô giá trị |
| Chạy 2 job train cùng lúc trên 1 máy | Trên MPS/CUDA chúng tranh GPU, **tổng** thời gian không giảm mà còn dễ OOM |

---

## 4. Quản lý đĩa

| Thư mục | Dung lượng | Có commit không? |
|---|---|---|
| `data/raw/` | 1,1 GB | **không** (trong `.gitignore`) |
| `data/processed/images_48_clahe.npy` | 342 MB | **không** |
| `artifacts/runs/` | ~115 MB cho 3 run | **không** |
| `reports/` | ~5 MB | **có** (bảng + hình cho báo cáo) |

**Cache cho ablation** — mỗi (size, preprocess) là một file riêng:

| Trục ablation | Cache cần dựng | Dung lượng |
|---|---|---|
| resolution 32/48/64 | `images_32_clahe`, `images_64_clahe` | 152 MB + 608 MB |
| preprocess 4 chế độ | `images_48_{none,he_gray,he_y}` | 342 MB × 3 |

`run_ablation.py` **tự dựng** cache khi thiếu (11 giây mỗi cache). Tổng thêm ~1,8 GB.
Xong việc thì xoá bớt:
```bash
rm data/processed/images_32_*.npy data/processed/images_64_*.npy
```

Dọn checkpoint của các run ablation (giữ lại `result.json`):
```bash
find artifacts/runs -name "last.pt" -delete          # giữ best.pt, bỏ last.pt
```

---

## 5. Thứ tự chạy khuyến nghị

**Ngày 1 — mỗi người trên máy mình**
```bash
make setup-mac && make data && make test     # ~12 phút, test phải XANH
```

**Ngày 2 — train model chính (song song 4 máy)**
```bash
make train-m1        # Phong Nguyễn   12 phút
make train-m2        # Hoàng          24 phút
make train-m3        # Phong Trần     45 phút trên Colab
make data-leaky && make train-m1   # Huy — thí nghiệm rò rỉ, 12 phút
```

**Ngày 3 — ablation ngân sách (song song)**
```bash
python scripts/run_ablation.py --axes components scaling label_smoothing --budget   # Hoàng
python scripts/run_ablation.py --axes resolution --budget                           # Phong Trần
python scripts/run_ablation.py --axes augmentation preprocess --budget              # Huy
```

**Ngày 4 — đo lường và tổng hợp**
```bash
make eval            # Phong Nguyễn — cần checkpoint của cả 3 người, gom về một máy
make robustness      # Huy
make speed gradcam   # Phong Trần — máy phải RẢNH khi đo tốc độ
make report          # Phong Nguyễn — sinh docs/KET_QUA.md
```

> **Gom checkpoint về một máy để chạy `make eval`:** chỉ cần copy thư mục
> `artifacts/runs/<run_id>/` (có `best.pt` + `config.yaml` + `result.json`). Mỗi run
> khoảng 10–50 MB.

**Ngày 5 — viết báo cáo, slide, diễn tập chất vấn chéo**
