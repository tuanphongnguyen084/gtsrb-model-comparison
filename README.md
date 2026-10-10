# GTSRB — So sánh 3 hướng Deep Learning cho nhận biết biển báo giao thông

Phân loại 43 lớp biển báo (bộ GTSRB, 51.839 ảnh), so sánh ba cách tiếp cận:

| Mô hình | Ý tưởng | Tham số | Macro-F1 (test) |
|---|---|---|---|
| **M1 LeNet** | CNN baseline xây từ số 0 | 2,42 M | 0,9778 |
| **M2 VGG-Res** | CNN sâu tự xây (BN + residual + GAP) | 1,24 M | 0,9880 |
| **M3 Transfer** | Backbone pretrained ImageNet, fine-tune | 2,28–11,2 M | 0,9796–0,9903 |

Kết quả đầy đủ (41 run, McNemar, robustness, Grad-CAM, latency) ở thư mục `results/`.

## Cấu trúc

```
preprocess.py      Tải GTSRB, dựng index.csv, chia train/val THEO TRACK, cache ảnh 48x48
train_m1.py        M1 LeNet          (model + train + eval, tự chứa)
train_m2.py        M2 VGG-Res        (model + train + eval, tự chứa)
train_m3.py        M3 Transfer       (--backbone resnet18 | mobilenetv2 | effnetb0)
evaluate.py        Nạp 5 checkpoint, in bảng so sánh + McNemar + latency (không train lại)
common.py          Dùng chung: Dataset, DataLoader, các hàm đo (macro-F1, McNemar)

data/              raw/ (ảnh gốc .ppm) + processed/ (index.csv + cache .npy)   [không commit]
checkpoints/       5 file .pt đã train                                          [không commit]
results/           Báo cáo, bảng, hình — kết quả đã đo (đóng băng)
```

Mỗi file `train_mX.py` đọc được độc lập từ trên xuống: định nghĩa kiến trúc model,
rồi vòng huấn luyện. Chỉ phần đọc dữ liệu và đo đạc (không phải logic deep learning
cốt lõi) nằm chung ở `common.py`.

## Chạy

```bash
pip install -r requirements.txt

python preprocess.py --download     # tải + tiền xử lý (lần đầu, ~5 phút)

python train_m1.py                  # ~12 phút
python train_m2.py                  # ~24 phút
python train_m3.py --backbone resnet18       # (và mobilenetv2, effnetb0)

python evaluate.py                  # in bảng so sánh từ checkpoint đã lưu
```

`evaluate.py` cần cache dữ liệu (`preprocess.py` đã chạy) và các checkpoint trong
`checkpoints/`. Trên máy đã train thì cả hai sẵn có; máy mới cần chạy `preprocess.py`
rồi train hoặc xin file `.pt`.

## Những điểm kỹ thuật cốt lõi

- **Chia dữ liệu theo track**: mỗi biển báo vật lý được quay 30 frame liên tiếp. Chia
  ngẫu nhiên theo ảnh làm lọt frame gần giống nhau sang cả train và val → val cao giả
  tạo (rò rỉ dữ liệu). `preprocess.py` nhóm theo cặp `(class_id, track_id)` và có
  `assert` chặn rò rỉ.
- **Macro-F1 là chỉ số chính**: dữ liệu mất cân bằng 10,7:1; accuracy che được lỗi ở
  lớp hiếm, macro-F1 thì không.
- **McNemar**: hai model chạy trên cùng tập test → quan sát bắt cặp → dùng McNemar,
  không dùng t-test.
- **Cấm augment lật ngang và đổi màu**: 4 cặp biển là ảnh gương của nhau (19↔20,
  33↔34, 36↔37, 38↔39); màu mang nghĩa (đỏ=cấm, xanh=bắt buộc).
- **M3 upsample 48→224**: backbone pretrained downsample 32 lần, cần 224 để tầng cuối
  còn 7×7. Upsample khớp SCALE, không thêm thông tin.

Chi tiết lý thuyết: `results/LY_THUYET.md`. Báo cáo nộp: `results/BAO_CAO.md`.
