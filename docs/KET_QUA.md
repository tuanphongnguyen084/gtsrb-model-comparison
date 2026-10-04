# Kết quả

> **File này được SINH TỰ ĐỘNG** bởi `python scripts/make_report.py` (hoặc `make report`).
> **Đừng sửa tay** — lần chạy sau sẽ ghi đè.
> Mọi số đều truy được về một `artifacts/runs/<run_id>/result.json` cụ thể.
>
> Cập nhật lần cuối: 2026-10-04 21:50 · 5 run

---

## 1. Bảng so sánh chính (tập test chính thức, 12.630 ảnh)

| Model          | run_id                               | Top-1   | Top-5   | **Macro-F1**   |    ECE |   #params (M) |   FLOPs (G) |   Latency p50 bs1 (ms) |
|:---------------|:-------------------------------------|:--------|:--------|:---------------|-------:|--------------:|------------:|-----------------------:|
| m1_lenet       | `m1_lenet_s42_20261003_222148`       | 98.46%  | 99.85%  | **0.9778**     | 0.131  |          2.42 |        0.07 |                   0.9  |
| m2_vggres      | `m2_vggres_s42_20261003_223350`      | 99.26%  | 99.94%  | **0.9880**     | 0.0535 |          1.24 |        0.29 |                   3.2  |
| m3_effnetb0    | `m3_effnetb0_s42_20261004_101515`    | 98.76%  | 99.91%  | **0.9838**     | 0.0973 |          4.06 |        0.83 |                 243.43 |
| m3_mobilenetv2 | `m3_mobilenetv2_s42_20261004_004220` | 98.73%  | 99.96%  | **0.9796**     | 0.1076 |          2.28 |        0.65 |                  47.87 |
| m3_resnet18    | `m3_resnet18_s42_20261003_225740`    | 99.33%  | 99.94%  | **0.9903**     | 0.1085 |         11.2  |        3.65 |                  16.65 |

**Cách đọc bảng — ba điều phải nói khi trình bày:**

1. **Top-5 gần như vô nghĩa ở đây.** Top-5 ra đời cho ImageNet **1000** lớp. Trên **43**
   lớp, top-5 nghĩa là "đúng trong 11,6% số lớp" — mọi model tử tế đều ~99,9%. Chỉ số
   **bão hoà** thì không phân biệt được gì. Nhóm báo cáo vì đề bài yêu cầu, nhưng
   **kết luận dựa trên top-1 và macro-F1**.
2. **Macro-F1 là chỉ số chính**, vì nó cho mỗi lớp trọng số bằng nhau (dữ liệu mất cân
   bằng 10,7:1). Bỏ hẳn một lớp 210 ảnh chỉ làm accuracy giảm ~0,54 điểm nhưng macro-F1
   mất ~2,3 điểm.
3. **Chênh lệch trong bảng chưa chắc có ý nghĩa** — xem mục 2. Trên 12.630 ảnh,
   0,2 điểm top-1 chỉ là ~25 ảnh.
> ⚠️ **Cột thời gian train của các run sau đã được ước lượng lại:**

> - `m3_mobilenetv2_s42_20261004_004220` — thô **34371s**, ước lượng **3318s**
>
> Nguyên nhân: máy ngủ giữa lúc train nên `time.time()` đếm cả thời gian ngủ.
> Số ước lượng tính bằng cách thay epoch bất thường bằng **trung vị** các epoch
> còn lại (`_detect_sleep()` trong `src/gtsrb/engine/train.py`). Số thô vẫn được
> giữ ở cột `train_seconds_raw` của `main_comparison.csv`.
> Chi tiết: `docs/SU_CO.md` mục 7.

---

## 2. Kiểm định ý nghĩa thống kê — McNemar

Hai model chạy trên **cùng** tập test → quan sát **bắt cặp** → dùng t-test hai mẫu
độc lập là **sai về mặt thống kê**. Chỉ hai ô **bất đồng** (`n01`, `n10`) mang thông tin.

| model_a        | model_b        |   n01 |   n10 | method         |     p_value | better   | verdict                                                                                                                                    |
|:---------------|:---------------|------:|------:|:---------------|------------:|:---------|:-------------------------------------------------------------------------------------------------------------------------------------------|
| m1_lenet       | m2_vggres      |    35 |   136 | chi2_corrected | 2.05417e-14 | B        | Model B tốt hơn CÓ ý nghĩa thống kê (p=2.054e-14 < 0,05). B đúng riêng 136 ảnh, A đúng riêng 35 ảnh.                                       |
| m1_lenet       | m3_effnetb0    |    83 |   122 | chi2_corrected | 0.00795355  | B        | Model B tốt hơn CÓ ý nghĩa thống kê (p=0.007954 < 0,05). B đúng riêng 122 ảnh, A đúng riêng 83 ảnh.                                        |
| m1_lenet       | m3_mobilenetv2 |    89 |   124 | chi2_corrected | 0.0198251   | B        | Model B tốt hơn CÓ ý nghĩa thống kê (p=0.01983 < 0,05). B đúng riêng 124 ảnh, A đúng riêng 89 ảnh.                                         |
| m1_lenet       | m3_resnet18    |    41 |   151 | chi2_corrected | 3.65e-15    | B        | Model B tốt hơn CÓ ý nghĩa thống kê (p=3.65e-15 < 0,05). B đúng riêng 151 ảnh, A đúng riêng 41 ảnh.                                        |
| m2_vggres      | m3_effnetb0    |   106 |    44 | chi2_corrected | 6.33779e-07 | A        | Model A tốt hơn CÓ ý nghĩa thống kê (p=6.338e-07 < 0,05). A đúng riêng 106 ảnh, B đúng riêng 44 ảnh.                                       |
| m2_vggres      | m3_mobilenetv2 |   115 |    49 | chi2_corrected | 3.86179e-07 | A        | Model A tốt hơn CÓ ý nghĩa thống kê (p=3.862e-07 < 0,05). A đúng riêng 115 ảnh, B đúng riêng 49 ảnh.                                       |
| m2_vggres      | m3_resnet18    |    51 |    60 | chi2_corrected | 0.447657    | tie      | Chênh lệch KHÔNG có ý nghĩa thống kê (p=0.4477 >= 0,05). Hai model coi như tương đương về accuracy; nên chọn dựa vào tốc độ và robustness. |
| m3_effnetb0    | m3_mobilenetv2 |    68 |    64 | chi2_corrected | 0.794003    | tie      | Chênh lệch KHÔNG có ý nghĩa thống kê (p=0.7940 >= 0,05). Hai model coi như tương đương về accuracy; nên chọn dựa vào tốc độ và robustness. |
| m3_effnetb0    | m3_resnet18    |    18 |    89 | chi2_corrected | 1.31339e-11 | B        | Model B tốt hơn CÓ ý nghĩa thống kê (p=1.313e-11 < 0,05). B đúng riêng 89 ảnh, A đúng riêng 18 ảnh.                                        |
| m3_mobilenetv2 | m3_resnet18    |    25 |   100 | chi2_corrected | 3.62221e-11 | B        | Model B tốt hơn CÓ ý nghĩa thống kê (p=3.622e-11 < 0,05). B đúng riêng 100 ảnh, A đúng riêng 25 ảnh.                                       |

---

## 3. ★ Đối chứng rò rỉ dữ liệu — split theo track vs split random

| Cách split | Track dùng chung giữa train và val |
|---|---|
| Random theo ảnh (**sai**) | **1.306** |
| Theo track, `StratifiedGroupKFold` (**đúng**) | **0** |

Để đo phần accuracy "ảo": `make data-leaky && make train-m1`, rồi so với con số
của split đúng. Dấu hiệu rò rỉ: **val cao hơn test một cách bất thường**.

---

## 4. Lớp yếu nhất (8 lớp F1 thấp nhất mỗi model)

Hai nguyên nhân phải **tách ra**:
F1 thấp + support **cao** → **hình giống nhau** (vấn đề độ phân giải);
F1 thấp + support **thấp** → **thiếu dữ liệu**.

**m1_lenet**

|   class_id | name                           |       f1 |   support |
|-----------:|:-------------------------------|---------:|----------:|
|         27 | Người đi bộ                    | 0.904348 |        60 |
|          6 | Hết giới hạn 80 km/h           | 0.905109 |       150 |
|         42 | Hết cấm vượt (xe trên 3,5 tấn) | 0.910995 |        90 |
|         30 | Cẩn thận băng/tuyết            | 0.930233 |       150 |
|         41 | Hết cấm vượt                   | 0.942149 |        60 |
|         22 | Đường xấu                      | 0.947368 |       120 |
|         26 | Đèn tín hiệu                   | 0.950549 |       180 |
|         40 | Vòng xuyến bắt buộc            | 0.95082  |        90 |

**m2_vggres**

|   class_id | name                           |       f1 |   support |
|-----------:|:-------------------------------|---------:|----------:|
|         42 | Hết cấm vượt (xe trên 3,5 tấn) | 0.896552 |        90 |
|         41 | Hết cấm vượt                   | 0.909091 |        60 |
|         22 | Đường xấu                      | 0.942731 |       120 |
|          6 | Hết giới hạn 80 km/h           | 0.951724 |       150 |
|          5 | Giới hạn 80 km/h               | 0.979719 |       630 |
|         26 | Đèn tín hiệu                   | 0.980926 |       180 |
|         40 | Vòng xuyến bắt buộc            | 0.98324  |        90 |
|         29 | Xe đạp qua đường               | 0.983425 |        90 |

**m3_effnetb0**

|   class_id | name                           |       f1 |   support |
|-----------:|:-------------------------------|---------:|----------:|
|         42 | Hết cấm vượt (xe trên 3,5 tấn) | 0.913706 |        90 |
|          6 | Hết giới hạn 80 km/h           | 0.916968 |       150 |
|         23 | Đường trơn                     | 0.9375   |       150 |
|         27 | Người đi bộ                    | 0.940171 |        60 |
|          5 | Giới hạn 80 km/h               | 0.958652 |       630 |
|         11 | Ưu tiên ở giao lộ kế tiếp      | 0.958738 |       420 |
|         22 | Đường xấu                      | 0.961039 |       120 |
|         30 | Cẩn thận băng/tuyết            | 0.963696 |       150 |

**m3_mobilenetv2**

|   class_id | name                           |       f1 |   support |
|-----------:|:-------------------------------|---------:|----------:|
|         27 | Người đi bộ                    | 0.811881 |        60 |
|         42 | Hết cấm vượt (xe trên 3,5 tấn) | 0.9      |        90 |
|         41 | Hết cấm vượt                   | 0.930233 |        60 |
|         30 | Cẩn thận băng/tuyết            | 0.940789 |       150 |
|         11 | Ưu tiên ở giao lộ kế tiếp      | 0.941458 |       420 |
|         23 | Đường trơn                     | 0.946372 |       150 |
|          6 | Hết giới hạn 80 km/h           | 0.947368 |       150 |
|         22 | Đường xấu                      | 0.965517 |       120 |

**m3_resnet18**

|   class_id | name                           |       f1 |   support |
|-----------:|:-------------------------------|---------:|----------:|
|         42 | Hết cấm vượt (xe trên 3,5 tấn) | 0.912821 |        90 |
|         30 | Cẩn thận băng/tuyết            | 0.932907 |       150 |
|          6 | Hết giới hạn 80 km/h           | 0.939929 |       150 |
|         21 | Đường cong đôi                 | 0.977273 |        90 |
|         11 | Ưu tiên ở giao lộ kế tiếp      | 0.978155 |       420 |
|          5 | Giới hạn 80 km/h               | 0.978923 |       630 |
|          3 | Giới hạn 60 km/h               | 0.983089 |       450 |
|         40 | Vòng xuyến bắt buộc            | 0.988764 |        90 |

---

## 5. Các cặp bị nhầm nhiều nhất

Bảng này dễ đọc hơn hình confusion matrix 43×43, và nó cho biết lỗi **đi đâu** —
điều mà accuracy không cho biết.

**m1_lenet**

|   true_id | true_name                 |   pred_id | pred_name                      |   count |
|----------:|:--------------------------|----------:|:-------------------------------|--------:|
|        17 | Cấm đi vào                |         9 | Cấm vượt                       |      16 |
|         6 | Hết giới hạn 80 km/h      |        42 | Hết cấm vượt (xe trên 3,5 tấn) |      14 |
|         3 | Giới hạn 60 km/h          |         5 | Giới hạn 80 km/h               |      14 |
|         8 | Giới hạn 120 km/h         |         5 | Giới hạn 80 km/h               |      11 |
|        22 | Đường xấu                 |        25 | Đang thi công                  |      10 |
|        11 | Ưu tiên ở giao lộ kế tiếp |        30 | Cẩn thận băng/tuyết            |      10 |
|         6 | Hết giới hạn 80 km/h      |         5 | Giới hạn 80 km/h               |       9 |
|        18 | Cảnh báo chung            |        26 | Đèn tín hiệu                   |       7 |

**m2_vggres**

|   true_id | true_name                      |   pred_id | pred_name                      |   count |
|----------:|:-------------------------------|----------:|:-------------------------------|--------:|
|        42 | Hết cấm vượt (xe trên 3,5 tấn) |        41 | Hết cấm vượt                   |      12 |
|        22 | Đường xấu                      |        25 | Đang thi công                  |      11 |
|         3 | Giới hạn 60 km/h               |         5 | Giới hạn 80 km/h               |      10 |
|         8 | Giới hạn 120 km/h              |         5 | Giới hạn 80 km/h               |       7 |
|        18 | Cảnh báo chung                 |        26 | Đèn tín hiệu                   |       7 |
|         6 | Hết giới hạn 80 km/h           |        42 | Hết cấm vượt (xe trên 3,5 tấn) |       6 |
|        12 | Đường ưu tiên                  |        13 | Nhường đường                   |       5 |
|         6 | Hết giới hạn 80 km/h           |         5 | Giới hạn 80 km/h               |       4 |

**m3_effnetb0**

|   true_id | true_name                 |   pred_id | pred_name                      |   count |
|----------:|:--------------------------|----------:|:-------------------------------|--------:|
|         3 | Giới hạn 60 km/h          |         5 | Giới hạn 80 km/h               |      22 |
|         8 | Giới hạn 120 km/h         |         5 | Giới hạn 80 km/h               |      20 |
|        11 | Ưu tiên ở giao lộ kế tiếp |        23 | Đường trơn                     |      19 |
|         6 | Hết giới hạn 80 km/h      |        42 | Hết cấm vượt (xe trên 3,5 tấn) |      17 |
|        22 | Đường xấu                 |        25 | Đang thi công                  |       8 |
|        11 | Ưu tiên ở giao lộ kế tiếp |        30 | Cẩn thận băng/tuyết            |       6 |
|         6 | Hết giới hạn 80 km/h      |         5 | Giới hạn 80 km/h               |       5 |
|        27 | Người đi bộ               |        11 | Ưu tiên ở giao lộ kế tiếp      |       5 |

**m3_mobilenetv2**

|   true_id | true_name                      |   pred_id | pred_name                      |   count |
|----------:|:-------------------------------|----------:|:-------------------------------|--------:|
|        27 | Người đi bộ                    |        11 | Ưu tiên ở giao lộ kế tiếp      |      18 |
|         3 | Giới hạn 60 km/h               |         5 | Giới hạn 80 km/h               |      17 |
|        11 | Ưu tiên ở giao lộ kế tiếp      |        23 | Đường trơn                     |      16 |
|         8 | Giới hạn 120 km/h              |         5 | Giới hạn 80 km/h               |      11 |
|        11 | Ưu tiên ở giao lộ kế tiếp      |        30 | Cẩn thận băng/tuyết            |      10 |
|         6 | Hết giới hạn 80 km/h           |        42 | Hết cấm vượt (xe trên 3,5 tấn) |       9 |
|        22 | Đường xấu                      |        25 | Đang thi công                  |       8 |
|        42 | Hết cấm vượt (xe trên 3,5 tấn) |        41 | Hết cấm vượt                   |       6 |

**m3_resnet18**

|   true_id | true_name                 |   pred_id | pred_name                      |   count |
|----------:|:--------------------------|----------:|:-------------------------------|--------:|
|        11 | Ưu tiên ở giao lộ kế tiếp |        30 | Cẩn thận băng/tuyết            |      17 |
|         6 | Hết giới hạn 80 km/h      |        42 | Hết cấm vượt (xe trên 3,5 tấn) |      16 |
|         3 | Giới hạn 60 km/h          |         5 | Giới hạn 80 km/h               |      13 |
|         8 | Giới hạn 120 km/h         |         5 | Giới hạn 80 km/h               |       7 |
|        38 | Đi bên phải               |         5 | Giới hạn 80 km/h               |       4 |
|        21 | Đường cong đôi            |        31 | Động vật hoang dã qua đường    |       4 |
|         5 | Giới hạn 80 km/h          |         7 | Giới hạn 100 km/h              |       2 |
|        22 | Đường xấu                 |        25 | Đang thi công                  |       2 |

---

## 6. Ablation (một-biến-một-lần)

_(chưa chạy)_  
Chạy: `python scripts/run_ablation.py --axes all` rồi `--collect-only`

---

## 7. Robustness (5 nhiễu × 5 mức, **chỉ áp ở test**)

`relative = acc_nhiễu / acc_sạch` — chia cho accuracy sạch để **tách "giỏi sẵn" khỏi "BỀN"**.
`mCE` = trung bình error qua mọi nhiễu và mức, càng **thấp** càng tốt.

| model       |    fog |   gauss_noise |   low_light |   motion_blur |   occlusion |    mCE |   clean_top1 |
|:------------|-------:|--------------:|------------:|--------------:|------------:|-------:|-------------:|
| m1_lenet    | 0.7254 |        0.9099 |      0.7795 |        0.8494 |      0.5009 | 0.2586 |       0.9846 |
| m2_vggres   | 0.7658 |        0.8801 |      0.6486 |        0.8248 |      0.4697 | 0.2875 |       0.9926 |
| m3_resnet18 | 0.9336 |        0.684  |      0.5576 |        0.9995 |      0.4408 | 0.2817 |       0.9933 |

---

## 8. Hình

| Hình | Nội dung |
|---|---|
| `reports/figures/preprocess_compare.png` | 4 chế độ cân bằng sáng trên ảnh tối |
| `reports/figures/aug_grid.png` | 8 biến thể augmentation của cùng 1 ảnh |
| `reports/figures/why_no_flip.png` | 4 cặp lớp gương — vì sao cấm flip ngang |
| `reports/figures/confusion_*.png` | confusion matrix 43×43 chuẩn hoá theo hàng |
| `reports/figures/reliability_*.png` | reliability diagram (hiệu chỉnh) |
| `reports/figures/corruption_grid.png` | 5 nhiễu × 5 mức trên cùng một ảnh |
| `reports/figures/robustness_curves.png` | accuracy giảm theo mức nhiễu |
| `reports/figures/gradcam_*_correct.png` | Grad-CAM, dự đoán đúng |
| `reports/figures/gradcam_*_wrong.png` | Grad-CAM, **dự đoán SAI** — model nhìn vào đâu |
| `reports/figures/gradcam_*_corrupted.png` | attention có trôi khi có nhiễu |
| `reports/figures/pareto.png` | Pareto accuracy–latency |

---

## 9. Khuyến nghị triển khai biên

**Khuyến nghị: `m1_lenet`** — suy ra từ dữ liệu theo ba bước bên dưới.

### Bước 1 — Model macro-F1 cao nhất: `m1_lenet` (0.9778)

### Bước 2 — Những model TƯƠNG ĐƯƠNG THỐNG KÊ với nó (McNemar, p ≥ 0,05)

| Cặp | n₀₁ | n₁₀ | p-value | Kết luận |
|---|---|---|---|---|
| m1_lenet vs m2_vggres | 35 | 136 | 2.054e-14 | khác biệt THẬT |
| m1_lenet vs m3_effnetb0 | 83 | 122 | 0.007954 | khác biệt THẬT |
| m1_lenet vs m3_mobilenetv2 | 89 | 124 | 0.01983 | khác biệt THẬT |
| m1_lenet vs m3_resnet18 | 41 | 151 | 3.65e-15 | khác biệt THẬT |

Nhóm tương đương: **m1_lenet**

### Bước 3 — Trong nhóm đó, chọn model có latency p95 thấp nhất (batch=1, cpu)

| Model | Latency p95 | #params | Dung lượng | FLOPs | Test top-1 |
|---|---|---|---|---|---|
| `m1_lenet` ← | 1.07 ms | 2.42 M | 9.7 MB | 0.075 G | 98.46% |

### Kết luận

> Chỉ `m1_lenet` nằm trong nhóm tốt nhất về accuracy (không model nào tương đương thống kê với nó), nên nó cũng là lựa chọn triển khai.

> ⚠️ Số latency này đo trên máy phát triển. Trước khi chốt triển khai thật, phải **đo lại trên thiết bị đích** — latency không so sánh được giữa hai máy khác nhau.
