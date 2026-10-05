# Kết quả

> **File này được SINH TỰ ĐỘNG** bởi `python scripts/make_report.py` (hoặc `make report`).
> **Đừng sửa tay** — lần chạy sau sẽ ghi đè.
> Mọi số đều truy được về một `artifacts/runs/<run_id>/result.json` cụ thể.
>
> Cập nhật lần cuối: 2026-10-05 11:40 · 39 run

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
2. **Macro-F1 là chỉ số chính**, vì nó cho mỗi lớp trọng số bằng nhau (train mất cân
   bằng 10,7:1 — lớp 0 có 210 ảnh, lớp 2 có 2.250). Lớp 0 cũng là lớp nhỏ nhất trên
   test với **60 ảnh**: đoán sai TOÀN BỘ lớp này chỉ làm accuracy giảm
   **0,48 điểm** (60/12.630) nhưng macro-F1 mất **2,33 điểm** (1/43, không phụ thuộc
   lớp to hay nhỏ) — gấp **4,9 lần**. Accuracy che được lỗi ở lớp thiểu số; macro-F1
   thì không.
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
|         30 | Cẩn thận băng/tuyết            | 0.879121 |       150 |
|         29 | Xe đạp qua đường               | 0.918367 |        90 |
|          5 | Giới hạn 80 km/h               | 0.937405 |       630 |
|         20 | Đường cong nguy hiểm sang phải | 0.942408 |        90 |
|         42 | Hết cấm vượt (xe trên 3,5 tấn) | 0.946237 |        90 |
|         22 | Đường xấu                      | 0.947368 |       120 |
|         26 | Đèn tín hiệu                   | 0.949721 |       180 |
|         27 | Người đi bộ                    | 0.956522 |        60 |

**m2_vggres**

|   class_id | name                           |       f1 |   support |
|-----------:|:-------------------------------|---------:|----------:|
|         42 | Hết cấm vượt (xe trên 3,5 tấn) | 0.941176 |        90 |
|         22 | Đường xấu                      | 0.947368 |       120 |
|          6 | Hết giới hạn 80 km/h           | 0.961938 |       150 |
|         41 | Hết cấm vượt                   | 0.97561  |        60 |
|          5 | Giới hạn 80 km/h               | 0.978193 |       630 |
|         21 | Đường cong đôi                 | 0.983051 |        90 |
|          3 | Giới hạn 60 km/h               | 0.983089 |       450 |
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
|         27 | Người đi bộ                    | 0.728972 |        60 |
|         41 | Hết cấm vượt                   | 0.923077 |        60 |
|         23 | Đường trơn                     | 0.9279   |       150 |
|          6 | Hết giới hạn 80 km/h           | 0.939929 |       150 |
|         22 | Đường xấu                      | 0.942731 |       120 |
|         42 | Hết cấm vượt (xe trên 3,5 tấn) | 0.955556 |        90 |
|          5 | Giới hạn 80 km/h               | 0.967391 |       630 |
|         20 | Đường cong nguy hiểm sang phải | 0.967391 |        90 |

---

## 5. Các cặp bị nhầm nhiều nhất

Bảng này dễ đọc hơn hình confusion matrix 43×43, và nó cho biết lỗi **đi đâu** —
điều mà accuracy không cho biết.

**m1_lenet**

|   true_id | true_name           |   pred_id | pred_name                      |   count |
|----------:|:--------------------|----------:|:-------------------------------|--------:|
|         3 | Giới hạn 60 km/h    |         5 | Giới hạn 80 km/h               |      29 |
|         8 | Giới hạn 120 km/h   |         5 | Giới hạn 80 km/h               |      20 |
|        30 | Cẩn thận băng/tuyết |        29 | Xe đạp qua đường               |      12 |
|         5 | Giới hạn 80 km/h    |         7 | Giới hạn 100 km/h              |      11 |
|        22 | Đường xấu           |        25 | Đang thi công                  |       9 |
|        30 | Cẩn thận băng/tuyết |        20 | Đường cong nguy hiểm sang phải |       8 |
|         9 | Cấm vượt            |         2 | Giới hạn 50 km/h               |       8 |
|        18 | Cảnh báo chung      |        26 | Đèn tín hiệu                   |       7 |

**m2_vggres**

|   true_id | true_name            |   pred_id | pred_name                      |   count |
|----------:|:---------------------|----------:|:-------------------------------|--------:|
|         3 | Giới hạn 60 km/h     |         5 | Giới hạn 80 km/h               |      13 |
|        22 | Đường xấu            |        25 | Đang thi công                  |      10 |
|        12 | Đường ưu tiên        |        13 | Nhường đường                   |       9 |
|         8 | Giới hạn 120 km/h    |         5 | Giới hạn 80 km/h               |       8 |
|         6 | Hết giới hạn 80 km/h |        42 | Hết cấm vượt (xe trên 3,5 tấn) |       8 |
|        18 | Cảnh báo chung       |        26 | Đèn tín hiệu                   |       4 |
|        21 | Đường cong đôi       |        31 | Động vật hoang dã qua đường    |       3 |
|        12 | Đường ưu tiên        |         9 | Cấm vượt                       |       3 |

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

|   true_id | true_name                  |   pred_id | pred_name                 |   count |
|----------:|:---------------------------|----------:|:--------------------------|--------:|
|         3 | Giới hạn 60 km/h           |         5 | Giới hạn 80 km/h          |      17 |
|         6 | Hết giới hạn 80 km/h       |         5 | Giới hạn 80 km/h          |       8 |
|        27 | Người đi bộ                |        11 | Ưu tiên ở giao lộ kế tiếp |       7 |
|        27 | Người đi bộ                |        23 | Đường trơn                |       7 |
|        11 | Ưu tiên ở giao lộ kế tiếp  |        23 | Đường trơn                |       6 |
|         6 | Hết giới hạn 80 km/h       |        41 | Hết cấm vượt              |       5 |
|        10 | Cấm vượt (xe trên 3,5 tấn) |         9 | Cấm vượt                  |       5 |
|        18 | Cảnh báo chung             |        27 | Người đi bộ               |       5 |

---

## 6. Ablation (một-biến-một-lần)

| trục            | tag                                     | model       |   img_size | preprocess   | aug_policy      |   label_smoothing |   test_top1 |   test_macro_f1 |   test_ece |
|:----------------|:----------------------------------------|:------------|-----------:|:-------------|:----------------|------------------:|------------:|----------------:|-----------:|
| augmentation    | augmentation-aug_geo-budget             | m2_vggres   |         48 | clahe        | geo             |               0.1 |    0.986857 |        0.980798 |   0.052647 |
| augmentation    | augmentation-aug_geo_photo-budget       | m2_vggres   |         48 | clahe        | geo_photo       |               0.1 |    0.988519 |        0.981064 |   0.052447 |
| augmentation    | augmentation-aug_geo_photo_erase-budget | m2_vggres   |         48 | clahe        | geo_photo_erase |               0.1 |    0.987094 |        0.981475 |   0.048196 |
| augmentation    | augmentation-aug_none-budget            | m2_vggres   |         48 | clahe        | none            |               0.1 |    0.987094 |        0.983438 |   0.065719 |
| components      | components-full-budget                  | m2_vggres   |         48 | clahe        | geo_photo       |               0.1 |    0.988519 |        0.981064 |   0.052447 |
| components      | components-no_bn-budget                 | m2_vggres   |         48 | clahe        | geo_photo       |               0.1 |    0.985748 |        0.979348 |   0.054168 |
| components      | components-no_residual-budget           | m2_vggres   |         48 | clahe        | geo_photo       |               0.1 |    0.987094 |        0.985038 |   0.045345 |
| components      | components-no_spatial_dropout-budget    | m2_vggres   |         48 | clahe        | geo_photo       |               0.1 |    0.990657 |        0.987772 |   0.102907 |
| label_smoothing | label_smoothing-ls0.0-budget            | m2_vggres   |         48 | clahe        | geo_photo       |               0   |    0.98844  |        0.979647 |   0.002868 |
| label_smoothing | label_smoothing-ls0.1-budget            | m2_vggres   |         48 | clahe        | geo_photo       |               0.1 |    0.988519 |        0.981064 |   0.052447 |
| label_smoothing | label_smoothing-ls0.2-budget            | m2_vggres   |         48 | clahe        | geo_photo       |               0.2 |    0.987173 |        0.979121 |   0.132674 |
| leak            | leak-A_random                           | m1_lenet    |         48 | clahe        | geo_photo       |               0.1 |    0.987094 |        0.98209  |   0.134673 |
| leak            | leak-B_track                            | m1_lenet    |         48 | clahe        | geo_photo       |               0.1 |    0.984561 |        0.977792 |   0.131021 |
| preprocess      | preprocess-prep_clahe-budget            | m2_vggres   |         48 | clahe        | geo_photo       |               0.1 |    0.988519 |        0.981064 |   0.052447 |
| preprocess      | preprocess-prep_he_gray-budget          | m2_vggres   |         48 | he_gray      | geo_photo       |               0.1 |    0.989865 |        0.985143 |   0.057356 |
| preprocess      | preprocess-prep_he_y-budget             | m2_vggres   |         48 | he_y         | geo_photo       |               0.1 |    0.991528 |        0.988554 |   0.056855 |
| preprocess      | preprocess-prep_none-budget             | m2_vggres   |         48 | none         | geo_photo       |               0.1 |    0.985827 |        0.97874  |   0.054459 |
| resolution      | resolution-res112-budget                | m3_resnet18 |        112 | clahe        | geo_photo       |               0.1 |    0.992399 |        0.98718  |   0.098085 |
| resolution      | resolution-res224-budget                | m3_resnet18 |        224 | clahe        | geo_photo       |               0.1 |    0.99327  |        0.9903   |   0.108545 |
| resolution      | resolution-res32-budget                 | m1_lenet    |         32 | clahe        | geo_photo       |               0.1 |    0.979018 |        0.965998 |   0.122297 |
| resolution      | resolution-res32-budget                 | m2_vggres   |         32 | clahe        | geo_photo       |               0.1 |    0.986223 |        0.977111 |   0.051279 |
| resolution      | resolution-res48-budget                 | m1_lenet    |         48 | clahe        | geo_photo       |               0.1 |    0.984719 |        0.978692 |   0.121636 |
| resolution      | resolution-res48-budget                 | m2_vggres   |         48 | clahe        | geo_photo       |               0.1 |    0.988519 |        0.981064 |   0.052447 |
| resolution      | resolution-res64-budget                 | m1_lenet    |         64 | clahe        | geo_photo       |               0.1 |    0.981869 |        0.97737  |   0.115839 |
| resolution      | resolution-res64-budget                 | m2_vggres   |         64 | clahe        | geo_photo       |               0.1 |    0.98369  |        0.974887 |   0.063035 |
| resolution      | resolution-res64-budget                 | m3_resnet18 |         64 | clahe        | geo_photo       |               0.1 |    0.987094 |        0.97758  |   0.101827 |
| scaling         | scaling-depth3-budget                   | m2_vggres   |         48 | clahe        | geo_photo       |               0.1 |    0.954553 |        0.904519 |   0.130405 |
| scaling         | scaling-depth4-budget                   | m2_vggres   |         48 | clahe        | geo_photo       |               0.1 |    0.988519 |        0.981064 |   0.052447 |
| scaling         | scaling-depth5-budget                   | m2_vggres   |         48 | clahe        | geo_photo       |               0.1 |    0.990657 |        0.984681 |   0.062248 |
| scaling         | scaling-width0.5-budget                 | m2_vggres   |         48 | clahe        | geo_photo       |               0.1 |    0.947823 |        0.89719  |   0.076757 |
| scaling         | scaling-width1.0-budget                 | m2_vggres   |         48 | clahe        | geo_photo       |               0.1 |    0.988519 |        0.981064 |   0.052447 |
| scaling         | scaling-width2.0-budget                 | m2_vggres   |         48 | clahe        | geo_photo       |               0.1 |    0.989707 |        0.98626  |   0.06958  |
| seed43          | seed43                                  | m2_vggres   |         48 | clahe        | geo_photo       |               0.1 |    0.993508 |        0.992971 |   0.057877 |
| seed44          | seed44                                  | m2_vggres   |         48 | clahe        | geo_photo       |               0.1 |    0.992716 |        0.990658 |   0.05689  |

---

## 7. Robustness (5 nhiễu × 5 mức, **chỉ áp ở test**)

`relative = acc_nhiễu / acc_sạch` — chia cho accuracy sạch để **tách "giỏi sẵn" khỏi "BỀN"**.
`mCE` = trung bình error qua mọi nhiễu và mức, càng **thấp** càng tốt.

| model          |    fog |   gauss_noise |   low_light |   motion_blur |   occlusion |    mCE |   clean_top1 |
|:---------------|-------:|--------------:|------------:|--------------:|------------:|-------:|-------------:|
| m1_lenet       | 0.7254 |        0.9099 |      0.7795 |        0.8494 |      0.5009 | 0.2586 |       0.9846 |
| m2_vggres      | 0.7658 |        0.8801 |      0.6486 |        0.8248 |      0.4697 | 0.2875 |       0.9926 |
| m3_effnetb0    | 0.6945 |        0.1104 |      0.0094 |        0.9969 |      0.5264 | 0.5383 |       0.9876 |
| m3_mobilenetv2 | 0.7045 |        0.2408 |      0.0881 |        0.9973 |      0.3874 | 0.5225 |       0.9873 |
| m3_resnet18    | 0.9336 |        0.684  |      0.5576 |        0.9995 |      0.4408 | 0.2817 |       0.9933 |

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

**Khuyến nghị: `m2_vggres`** — suy ra từ dữ liệu theo ba bước bên dưới.

### Bước 1 — Model macro-F1 cao nhất: `m3_resnet18` (0.9903)

### Bước 2 — Những model TƯƠNG ĐƯƠNG THỐNG KÊ với nó (McNemar, p ≥ 0,05)

| Cặp | n₀₁ | n₁₀ | p-value | Kết luận |
|---|---|---|---|---|
| m1_lenet vs m3_resnet18 | 41 | 151 | 3.65e-15 | khác biệt THẬT |
| m2_vggres vs m3_resnet18 | 51 | 60 | 0.4477 | TƯƠNG ĐƯƠNG |
| m3_effnetb0 vs m3_resnet18 | 18 | 89 | 1.313e-11 | khác biệt THẬT |
| m3_mobilenetv2 vs m3_resnet18 | 25 | 100 | 3.622e-11 | khác biệt THẬT |

Nhóm tương đương: **m2_vggres, m3_resnet18**

### Bước 3 — Trong nhóm đó, chọn model có latency p95 thấp nhất (batch=1, cpu)

| Model | Latency p95 | #params | Dung lượng | FLOPs | Test top-1 |
|---|---|---|---|---|---|
| `m2_vggres` ← | 3.75 ms | 1.24 M | 5.0 MB | 0.290 G | 99.26% |
| `m3_resnet18` | 18.31 ms | 11.20 M | 44.8 MB | 3.647 G | 99.33% |

### Kết luận

> `m2_vggres` và `m3_resnet18` **tương đương về accuracy** (chênh 0.07 điểm, McNemar p ≥ 0,05 → không có ý nghĩa thống kê).
> Nhưng `m2_vggres` **nhanh hơn 4.9 lần** (3.75 ms so với 18.31 ms ở p95, batch=1) và **nhẹ hơn 9.0 lần** (5.0 MB so với 44.8 MB).
>
> Trả thêm 4.9 lần latency để lấy 0.07 điểm accuracy **không có ý nghĩa thống kê** là lựa chọn tồi trên hệ thống thời gian thực.

**Đây là lý do phải kiểm định thống kê.** Nếu chỉ nhìn bảng accuracy, kết luận sẽ là "chọn `m3_resnet18` vì nó cao hơn" — một kết luận **sai**.

> ⚠️ Số latency này đo trên máy phát triển. Trước khi chốt triển khai thật, phải **đo lại trên thiết bị đích** — latency không so sánh được giữa hai máy khác nhau.
