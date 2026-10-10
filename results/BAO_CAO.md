# Phân loại biển báo giao thông GTSRB: so sánh ba hướng tiếp cận Deep Learning

**Môn**: Deep Learning · **Ngày**: 09/10/2026
**Nhóm**: Phong Nguyễn · Hoàng · Phong Trần · Huy

> Bảng số trong báo cáo này được **sinh tự động** từ `reports/tables/` bằng
> `python scripts/make_baocao.py`. Không có con số nào chép tay; mỗi con số truy được
> về một `artifacts/runs/<run_id>/result.json` cụ thể.

---

## Tóm tắt

Nhóm xây và so sánh 5 mô hình phân loại 43 lớp biển báo giao thông Đức trên
bộ GTSRB (39.209 ảnh train + 12.630 ảnh test chính thức), theo ba hướng: CNN đơn giản tự
xây (LeNet), CNN sâu tự xây (VGG-like có BatchNorm, residual, spatial dropout, Global
Average Pooling), và transfer learning từ ba backbone pretrained ImageNet.

GTSRB là bộ dữ liệu **đã bão hoà** — con người đạt 98,84%, nhà vô địch IJCNN 2011 đạt
99,46% — nên nhóm không coi accuracy là câu hỏi chính. Thay vào đó nhóm tập trung vào bốn
câu hỏi mà đa số báo cáo về bộ này bỏ qua: (1) split dữ liệu đúng cách thì con số thật là
bao nhiêu; (2) chênh lệch giữa các mô hình có **ý nghĩa thống kê** không; (3) mô hình giữ
được bao nhiêu hiệu năng dưới nhiễu thực tế; (4) với ràng buộc latency của thiết bị biên
thì nên chọn mô hình nào.

**Ba kết quả chính, đều đi ngược kỳ vọng thông thường:**

1. Mô hình **tự xây 1,24 M tham số đánh bại hai backbone pretrained ImageNet** với ý nghĩa
   thống kê (p < 1e-6), trong khi nhanh hơn 15–76 lần.
2. **Latency không tỉ lệ với FLOPs**, chênh tới **64 lần** về hiệu quả trên mỗi GFLOP.
   Mô hình tên "EfficientNet" lại là mô hình **chậm nhất** trong cả 5.
3. Mô hình có **accuracy sạch cao nhất lại kém bền nhất** trước nhiễu, còn `m1_lenet` — yếu nhất khi ảnh sạch — bền nhất trước nhiễu thực tế.

---

## 1. Bài toán và dữ liệu

|              |                                                                      |
| ------------ | -------------------------------------------------------------------- |
| Bộ dữ liệu   | GTSRB (German Traffic Sign Recognition Benchmark)                    |
| Số lớp       | 43                                                                   |
| Train        | 39.209 ảnh · Test chính thức                                         | 12.630 ảnh · **Tổng 51.839** |
| Định dạng    | `.ppm`, kích thước thay đổi, **trung vị 43×43 px** (min 25, max 266) |
| Mất cân bằng | **10,7 : 1** (lớp 2 có 2.250 ảnh; lớp 0, 19, 37 chỉ 210)             |

Nguồn: archive gốc Đại học Bochum. Nhóm **không** dùng `torchvision.datasets.GTSRB` vì
hàm đó tải bộ train 26.640 ảnh của giải IJCNN 2011 chứ không phải bộ Final Training
39.209 ảnh — một cái bẫy không được ghi trong tài liệu torchvision
(chi tiết: `docs/SU_CO.md` mục 1).

---

## 2. Phương pháp

### 2.1 Tiền xử lý

Cắt theo **ROI bounding box** (nới lề 10%) → cân bằng sáng → resize 48×48.

Bốn chế độ cân bằng sáng được hiện thực để làm ablation: `none`, `he_gray` (equalize ảnh
xám), `he_y` (equalize kênh Y của YUV), và `clahe` (mặc định). CLAHE được chọn vì HE toàn
cục dùng **một** CDF cho cả ảnh nên khuếch đại nhiễu ở vùng phẳng; CLAHE chia ô 8×8,
equalize từng ô và **chặn trần** histogram (`clipLimit = 2,0`).

Cân bằng sáng làm trên **kênh L của LAB**, không làm trên từng kênh RGB — equalize R, G, B
độc lập sẽ làm lệch tỉ lệ ba kênh và **đổi màu ảnh**, trong khi màu biển báo mang nghĩa
(đỏ = cấm, xanh = bắt buộc).

### 2.2 ★ Chia train/val theo track — chống rò rỉ dữ liệu

Đây là đóng góp kỹ thuật quan trọng nhất của phần dữ liệu.

GTSRB **không** chụp mỗi biển báo một lần: xe chạy tới gần một tấm biển và camera quay
**30 frame liên tiếp** của *cùng tấm biển vật lý đó*, lưu thành một "track"
(tên file `000{track}_000{frame}.ppm`). Nếu trộn tất cả ảnh rồi cắt 80/20 ngẫu nhiên,
frame 12 và frame 13 của **cùng một tấm biển** — gần như là hai bản sao — sẽ nằm một bên
train, một bên val. Mô hình không cần **khái quát**, chỉ cần **nhớ**.

Nhóm dùng `StratifiedGroupKFold` với nhóm `(class_id, track_id)`. Kết quả đối chứng:

| Cách split                | Số track dùng chung giữa train và val |
| ------------------------- | ------------------------------------- |
| Random theo ảnh (**sai**) | **1.306**                             |
| Theo track (**đúng**)     | **0**                                 |

Kết quả: train 31.379 / val 7.830 / test 12.630, lệch phân phối lớp tối đa **0,297%**.
Tập test chính thức được **niêm phong**, chỉ mở một lần ở cuối.
`tests/test_split.py` khoá bất biến này làm cổng chặn của dự án.

### 2.3 Augmentation

Xoay ±15°, dịch ±10%, zoom 0,9–1,1, jitter độ sáng/tương phản ±0,2. Chỉ áp cho tập train.

**Hai phép biến đổi bị cấm:**

- `RandomHorizontalFlip` — bốn cặp lớp là **ảnh gương** của nhau (33↔34 rẽ phải/trái,
  19↔20, 36↔37, 38↔39). Lật ngang là **tự tạo dữ liệu sai nhãn**.
- Hue/saturation jitter — màu là **đặc trưng mang nghĩa**, không phải nhiễu.

`tests/test_transforms.py` khoá luật này ở mức mã nguồn.

### 2.4 Ba hướng mô hình

|        | Mô hình             | Kiến trúc                                                    | Input |
| ------ | ------------------- | ------------------------------------------------------------ | ----- |
| **M1** | LeNet-style, tự xây | 2 × (Conv 5×5 + MaxPool) + 2 Dense                           | 48²   |
| **M2** | VGG-like, tự xây    | 4 stage × 2 Conv 3×3 + BN + residual + SpatialDropout + GAP  | 48²   |
| **M3** | Transfer learning   | ResNet18 / MobileNetV2 / EfficientNet-B0 pretrained ImageNet | 224²  |

**Một quan sát đáng chú ý về số tham số:** M1 có **2.424.299** tham số với **2** lớp conv,
trong đó **97,3%** nằm ở *một* lớp `Flatten(9216) → FC(256)`. M2 có **9** lớp conv nhưng
chỉ **1.237.451** tham số — **ít hơn M1** — vì nó thay Flatten+FC bằng Global Average
Pooling (11.051 tham số, giảm **214 lần** ở phần head). Bài học: **số lớp không tỉ lệ với số
tham số; chỗ đắt là lớp fully-connected.**

M3 dùng mean/std của **ImageNet** (không phải của GTSRB) vì trọng số pretrained và
`running_mean/var` của các lớp BatchNorm được học trên phân phối đó. Ảnh được upsample lên
224×224 vì ba backbone downsample 32 lần — đưa 48×48 vào thì feature map cuối chỉ còn
1,5×1,5. Nhóm nêu rõ: nội suy 48→224 **không thêm thông tin nào**, nó chỉ để khớp scale và
stride của mạng pretrained.

Huấn luyện M3 theo **hai pha**: (1) đóng băng backbone, train riêng head 3 epoch; (2) mở
băng toàn bộ với **discriminative learning rate** (tầng đầu 1e-5, giữa 1e-4, head 1e-3).
Bằng chứng thực nghiệm cho thiết kế này: với ResNet18, train riêng head chững ở **67,9%**
sau 3 epoch, rồi **nhảy lên 96,4% chỉ trong một epoch** ngay khi mở băng.

### 2.5 Huấn luyện

Cross-entropy + **label smoothing 0,1**; Adam/AdamW; warmup tuyến tính 3 epoch rồi
**cosine annealing**; gradient clipping 1,0; mixed precision (tự bật trên CUDA, tắt trên
MPS); **early stopping theo val macro-F1** (không theo accuracy, vì dữ liệu mất cân bằng
10,7:1 khiến accuracy bị các lớp đông chi phối). Cả 5 mô hình dùng **chung
một hàm `fit()`** để so sánh được công bằng.

---

## 3. Kết quả

### 3.1 Bảng so sánh chính (tập test chính thức, 12.630 ảnh)

| Mô hình        | Tham số (M) | FLOPs (G) | Dung lượng (MB) | Top-1  | Top-5  | Macro-F1 |    ECE | p95 CPU (ms) | Train (phút) |
| :------------- | ----------: | --------: | --------------: | :----- | :----- | -------: | -----: | -----------: | -----------: |
| m3_resnet18    |        11.2 |      3.65 |            44.8 | 99.33% | 99.94% |   0.9903 | 0.1085 |         18.3 |           61 |
| m2_vggres      |        1.24 |      0.29 |               5 | 99.26% | 99.94% |    0.988 | 0.0535 |          3.7 |           24 |
| m3_effnetb0    |        4.06 |      0.83 |            16.2 | 98.77% | 99.91% |   0.9838 | 0.0973 |        256.6 |          132 |
| m3_mobilenetv2 |        2.28 |      0.65 |             9.1 | 98.73% | 99.96% |   0.9796 | 0.1076 |           49 |           55 |
| m1_lenet       |        2.42 |      0.07 |             9.7 | 98.46% | 99.85% |   0.9778 |  0.131 |          1.1 |           12 |

**Lưu ý khi đọc:** cột **Top-5 gần như vô nghĩa** ở bài này. Top-5 ra đời cho ImageNet
1000 lớp; trên 43 lớp nó nghĩa là "đúng trong 11,6% số lớp" nên mọi mô hình tử tế đều
~99,9%. Nhóm báo cáo vì đề bài yêu cầu nhưng **kết luận dựa trên top-1 và macro-F1**.

### 3.2 ★ Kiểm định ý nghĩa thống kê (McNemar)

Hai mô hình chạy trên **cùng** tập test nên quan sát là **bắt cặp**; dùng t-test hai mẫu
độc lập là sai về mặt thống kê. McNemar chỉ xét hai ô **bất đồng** của bảng 2×2.

| Cặp                           |  n01 |  n10 |  p-value | Kết luận           |
| :---------------------------- | ---: | ---: | -------: | :----------------- |
| m1_lenet vs m2_vggres         |   35 |  136 | 2.05e-14 | m2_vggres hơn      |
| m1_lenet vs m3_effnetb0       |   83 |  122 |  0.00795 | m3_effnetb0 hơn    |
| m1_lenet vs m3_mobilenetv2    |   89 |  124 |   0.0198 | m3_mobilenetv2 hơn |
| m1_lenet vs m3_resnet18       |   41 |  151 | 3.65e-15 | m3_resnet18 hơn    |
| m2_vggres vs m3_effnetb0      |  106 |   44 | 6.34e-07 | m2_vggres hơn      |
| m2_vggres vs m3_mobilenetv2   |  115 |   49 | 3.86e-07 | m2_vggres hơn      |
| m2_vggres vs m3_resnet18      |   51 |   60 |    0.448 | **tương đương**    |
| m3_effnetb0 vs m3_mobilenetv2 |   68 |   64 |    0.794 | **tương đương**    |
| m3_effnetb0 vs m3_resnet18    |   18 |   89 | 1.31e-11 | m3_resnet18 hơn    |
| m3_mobilenetv2 vs m3_resnet18 |   25 |  100 | 3.62e-11 | m3_resnet18 hơn    |

Trên 12.630 ảnh, chênh 0,1 điểm top-1 chỉ là ~13 ảnh — hoàn toàn có thể nằm trong nhiễu.
Đây là lý do mọi so sánh đều phải kiểm định.

**Phát hiện quan trọng nhất của báo cáo:** mô hình **M2 tự xây** (1,24 M tham số, 48×48)
**vượt** cả MobileNetV2 lẫn EfficientNet-B0 — hai backbone pretrained trên 1,28 triệu ảnh
ImageNet — với p < 1e-6, và **tương đương** ResNet18 (p = 0,45).

Giải thích khả dĩ: ảnh GTSRB có trung vị chỉ **43×43 px**. Upsample lên 224 không tạo ra
thông tin nào, nên phần tri thức pretrained về chi tiết tinh không có gì để bám vào, trong
khi chi phí (tham số, FLOPs, latency, thời gian train) vẫn phải trả đủ. Kết luận cần phát
biểu thận trọng: **transfer learning hiệu quả khi dữ liệu đích ít VÀ miền đích gần miền
nguồn về độ phân giải** — GTSRB không thoả cả hai điều kiện.

### 3.3 Phân tích theo lớp

**Mô hình `m3_resnet18`:**

|  Lớp | Tên                            |    F1 | Số ảnh test |
| ---: | :----------------------------- | ----: | ----------: |
|   27 | Người đi bộ                    | 0.729 |          60 |
|   41 | Hết cấm vượt                   | 0.923 |          60 |
|   23 | Đường trơn                     | 0.928 |         150 |
|    6 | Hết giới hạn 80 km/h           |  0.94 |         150 |
|   22 | Đường xấu                      | 0.943 |         120 |
|   42 | Hết cấm vượt (xe trên 3,5 tấn) | 0.956 |          90 |
|    5 | Giới hạn 80 km/h               | 0.967 |         630 |
|   20 | Đường cong nguy hiểm sang phải | 0.967 |          90 |

Các cặp bị nhầm nhiều nhất:

| Lớp thật | Tên                       | Đoán thành | Tên                       | Số lần |
| -------: | :------------------------ | ---------: | :------------------------ | -----: |
|        3 | Giới hạn 60 km/h          |          5 | Giới hạn 80 km/h          |     17 |
|        6 | Hết giới hạn 80 km/h      |          5 | Giới hạn 80 km/h          |      8 |
|       27 | Người đi bộ               |         11 | Ưu tiên ở giao lộ kế tiếp |      7 |
|       27 | Người đi bộ               |         23 | Đường trơn                |      7 |
|       11 | Ưu tiên ở giao lộ kế tiếp |         23 | Đường trơn                |      6 |
|        6 | Hết giới hạn 80 km/h      |         41 | Hết cấm vượt              |      5 |

Hai nguyên nhân cần tách bạch: F1 thấp kèm **support cao** là do **hình giống nhau**
(vấn đề độ phân giải — các biển giới hạn tốc độ chỉ khác chữ số, mà ở 48×48 chữ số chỉ còn
vài pixel); F1 thấp kèm **support thấp** mới là do **thiếu dữ liệu**. Số liệu cho thấy
nguyên nhân thứ nhất chiếm ưu thế, nên hướng cải thiện là **tăng độ phân giải**, không
phải làm mô hình to hơn.

### 3.4 Độ hiệu chỉnh (calibration)

Cột ECE trong bảng 3.1. Kết quả **ngược với lý thuyết thông thường**: Guo et al. (2017)
cho rằng mạng sâu thường **tự tin thái quá**, nhưng nhóm đo được M1 có độ tự tin trung
bình **0,854** trong khi accuracy là **0,985** — tức **thiếu tự tin**.

Nguyên nhân: kết luận của Guo đúng trên ImageNet (bài toán khó, accuracy ~76%). GTSRB
**quá dễ** nên mức tự tin mà label smoothing kéo mô hình về (≈ `1−ε+ε/K` = 0,9023) **thấp
hơn** accuracy thật — nó **sửa quá tay**. Hệ quả thực tế: lập luận "đặt ngưỡng 0,8 thì
chuyển cho người lái" **không dùng được** với mô hình này, vì ngưỡng đó sẽ loại bỏ rất
nhiều dự đoán đúng.

### 3.5 Robustness dưới nhiễu mô phỏng

Năm loại nhiễu × năm mức độ, **chỉ áp lúc test** — train trên chúng thì phép đo mất ý
nghĩa vì robustness nghĩa là khái quát sang phân phối **chưa từng thấy**.

Bảng `relative robustness` = accuracy dưới nhiễu / accuracy trên ảnh sạch:

| model          |    fog | gauss_noise | low_light | motion_blur | occlusion |    mCE | clean_top1 |
| :------------- | -----: | ----------: | --------: | ----------: | --------: | -----: | ---------: |
| m1_lenet       | 0.7254 |      0.9099 |    0.7795 |      0.8494 |    0.5009 | 0.2586 |     0.9846 |
| m2_vggres      | 0.7658 |      0.8801 |    0.6486 |      0.8248 |    0.4697 | 0.2875 |     0.9926 |
| m3_effnetb0    | 0.6945 |      0.1104 |    0.0094 |      0.9969 |    0.5264 | 0.5383 |     0.9876 |
| m3_mobilenetv2 | 0.7045 |      0.2408 |    0.0881 |      0.9973 |    0.3874 | 0.5225 |     0.9873 |
| m3_resnet18    | 0.9336 |       0.684 |    0.5576 |      0.9995 |    0.4408 | 0.2817 |     0.9933 |

**Chỉ số dưới đây KHÔNG phải cột `mCE` của bảng trên.** `mCE` tính trên **cả 5 loại nhiễu**, kể cả `motion_blur` và `fog` — hai loại **không so sánh được** giữa 48×48 và 224×224 (xem cảnh báo bên dưới). Chỉ số dưới đây chỉ dùng **3 loại so sánh được** (`gauss_noise`, `low_light`, `occlusion`), nên con số CAO HƠN và thứ hạng có thể khác.

Và thứ hạng **đổi thật**: theo `mCE` là `m1_lenet` < `m3_resnet18` < `m2_vggres` < `m3_mobilenetv2` < `m3_effnetb0`; theo 3 nhiễu so sánh được thì khác (xem dòng cuối mục này). Kết luận của nhóm rút từ nhóm nhiễu so sánh được.

Trên 5 mô hình:

**Mô hình chính xác nhất trên ảnh sạch KHÔNG phải mô hình bền nhất.** `m3_resnet18` dẫn đầu khi ảnh sạch (top-1 = 0.9933) nhưng `m1_lenet` mới là mô hình bền nhất (error trên 3 nhiễu so sánh được = 0.2812 so với 0.4429).

Đáng chú ý hơn: `m1_lenet` — **yếu nhất** trên ảnh sạch (top-1 = 0.9846) — lại bền nhất. Thứ tự xếp hạng khi có nhiễu **đảo lại** so với khi không có.

Xếp hạng độ bền **theo 3 nhiễu so sánh được** (càng THẤP càng bền): `m1_lenet` 0.2812 < `m2_vggres` 0.3388 < `m3_resnet18` 0.4429 < `m3_mobilenetv2` 0.7643 < `m3_effnetb0` 0.7873.

⚠️ **Giới hạn phương pháp cần nêu rõ:** nhiễu được áp **sau khi** resize, mà M1/M2 chạy ở
48×48 còn M3 ở 224×224. Kernel motion blur mức 5 là **15 pixel tuyệt đối**: ở 48×48 nó phủ
**31%** chiều rộng ảnh, ở 224×224 chỉ **6,7%**. Vì vậy cột `motion_blur` **không so sánh
được** giữa hai nhóm độ phân giải. Ba cột `gauss_noise`, `low_light`, `occlusion` thì so
sánh được (phép toán theo từng pixel hoặc theo % diện tích), và kết luận chỉ rút từ chúng.
Hướng sửa triệt để: đổi kernel sang tỉ lệ % chiều rộng ảnh.

#### 3.5.1 Giới hạn NẶNG hơn: nhiễu được áp SAU bước CLAHE

**Chênh = (áp nhiễu TRƯỚC CLAHE) − (áp nhiễu SAU CLAHE).** Dương nghĩa là sửa thứ tự giúp mô hình đó:

| model          | gauss_noise | low_light | trung bình |
| :------------- | ----------: | --------: | ---------: |
| m3_effnetb0    |       0.368 |     0.361 |      0.364 |
| m3_mobilenetv2 |       0.332 |     0.265 |      0.298 |
| m3_resnet18    |       0.003 |     0.006 |      0.005 |
| m2_vggres      |      -0.111 |    -0.078 |     -0.095 |
| m1_lenet       |      -0.155 |    -0.198 |     -0.176 |

**Dấu của hiệu ứng phụ thuộc vào MÔ HÌNH** — không phải một sai số chung cộng vào mọi mô hình như nhau. CLAHE cân bằng tương phản **cục bộ**: áp nhiễu trước thì CLAHE *khuếch đại* chính cái nhiễu đó, rồi ảnh bị thu về 48×48. Với mô hình nhỏ ở độ phân giải thấp, nhiễu đã khuếch đại còn tệ hơn ảnh tối ban đầu; với mô hình 224px thì việc lấy lại độ sáng tổng thể quan trọng hơn.

Thứ hạng độ bền theo hai thứ tự:

|     | thứ tự ĐANG đo         | thứ tự TRIỂN KHAI      |
| --- | ---------------------- | ---------------------- |
| 1   | `m1_lenet` 0.798       | `m1_lenet` 0.622       |
| 2   | `m2_vggres` 0.705      | `m2_vggres` 0.611      |
| 3   | `m3_resnet18` 0.528    | `m3_resnet18` 0.532    |
| 4   | `m3_mobilenetv2` 0.133 | `m3_effnetb0` 0.456    |
| 5   | `m3_effnetb0` 0.091    | `m3_mobilenetv2` 0.431 |

**Ba điều phải nói cùng nhau, không được lẫn:**

1. Mô hình bền nhất **không đổi** (`m1_lenet`) ở cả hai thứ tự — kết luận chính vẫn đứng.
2. Nhưng khoảng cách hạng 1 với hạng 2 **co từ 0.093 xuống 0.011** — hai mô hình đầu gần như bằng nhau về độ bền. Nói 'm1_lenet bền nhất' mà không nói con số này là phóng đại.
3. Biên độ giữa 5 mô hình co từ **0.71 xuống 0.19**. Những con số thấp dưới mức đoán bừa (1/43 = 0,023) trong bảng robustness chính là **hiện vật đo**, không phải tính chất mô hình.

**Không có thứ tự nào đúng tuyệt đối.** Áp sau CLAHE thiên vị mô hình nhỏ ở 48px; áp trước CLAHE sát điều kiện triển khai hơn nhưng trừng phạt chính nhóm đó. Phép đo chính trong `robustness.csv` dùng thứ tự **sau CLAHE**; bảng trên là phép đo đối chứng. Chi tiết ở `docs/SU_CO.md` §12.

### 3.5.2 Nhiễu seed — chênh lệch bao nhiêu điểm thì mới là THẬT?

Cùng một cấu hình, chỉ đổi seed khởi tạo. Đây là phép đo **độc lập** với McNemar
cho cùng một câu hỏi, và quan trọng hơn mọi con số accuracy lẻ trong báo cáo: nó
cho biết **ngưỡng dưới** của những gì đáng kết luận.

**Macro-F1 nhiễu hơn top-1 5 lần** qua các seed của `m2_vggres`: biên độ 0.50 điểm so với 0.10 điểm. Hợp lý — macro-F1 cho mỗi lớp trọng số bằng nhau, nên nó chịu trọn dao động ở các lớp thiểu số, đúng chỗ nhạy nhất với seed. Đó là **cái giá** của việc chọn macro-F1 làm chỉ số chính, và nghĩa là ngưỡng 'đáng kể' của macro-F1 phải ĐẶT CAO HƠN ngưỡng của top-1, không dùng chung một mức.

`m2_vggres` chạy **3 seed** (42, 43, 44): macro-F1 0.99054 ± 0.00250, từ 0.98798 đến 0.99297 — **biên độ 0.50 điểm**.


Trong bảng so sánh chính, `m3_resnet18` (0.99030) nằm **bên trong** biên độ seed của `m2_vggres`. Nghĩa là chênh lệch với những mô hình đó **không phân biệt được khỏi việc đổi seed**.


★ Seed tốt nhất của `m2_vggres` (0.99297) **vượt** `m3_resnet18` (0.99030) — mô hình xếp TRÊN nó trong bảng chính. Thứ hạng giữa chúng **đảo theo seed**, nên không được trình bày như một xếp hạng cố định. Đây là phép đo ĐỘC LẬP dẫn tới cùng kết luận với McNemar.

| m2_vggres (3 seed) |    mean |     std |     min |     max | biên độ (điểm) |
| :----------------- | ------: | ------: | ------: | ------: | -------------: |
| top1               | 0.99293 | 0.00051 | 0.99256 | 0.99351 |         0.0951 |
| top5               | 0.99926 | 0.00018 | 0.99905 | 0.99937 |         0.0317 |
| macro_f1           | 0.99054 |  0.0025 | 0.98798 | 0.99297 |         0.4992 |
| ece                | 0.05608 | 0.00231 | 0.05347 | 0.05788 |         0.4403 |

### 3.6 Tốc độ suy luận và triển khai biên

Đo đúng quy trình: 20 vòng warm-up, **đồng bộ thiết bị trước và sau khi bấm giờ** (GPU
chạy bất đồng bộ — không đồng bộ thì đang đo thời gian *gửi lệnh*), 100 vòng đo, báo
**p50 và p95** (hệ thống thời gian thực quan tâm trường hợp xấu).

**Latency không tỉ lệ với FLOPs:**

| Mô hình        | FLOPs (G) | Latency p50 (ms) | **ms/GFLOP** |
| :------------- | --------: | ---------------: | -----------: |
| m3_resnet18    |      3.65 |            16.65 |          4.6 |
| m2_vggres      |      0.29 |              3.2 |           11 |
| m1_lenet       |      0.07 |              0.9 |           12 |
| m3_mobilenetv2 |      0.65 |            47.87 |         73.4 |
| m3_effnetb0    |      0.83 |           243.43 |        294.1 |

Nếu latency tỉ lệ FLOPs thì cột cuối phải bằng nhau — thực tế chênh **64 lần**.
EfficientNet-B0 **ít hơn ResNet18 4.4 lần FLOPs** nhưng chậm hơn **14.6 lần** trên CPU.
Nguyên nhân: MBConv + squeeze-excitation gồm rất nhiều lớp **mảnh**, mỗi lớp tốn chi phí
cố định (launch kernel, truy cập bộ nhớ) mà làm rất ít phép tính → bị chặn bởi **băng
thông bộ nhớ**, không bởi năng lực tính toán.

**Hệ quả thực hành: chọn mô hình theo FLOPs sẽ dẫn tới EfficientNet-B0 — mô hình chậm
nhất trong cả 5.** Muốn nói về triển khai thì phải đo wall-clock trên thiết
bị đích.

#### 3.6.1 Nén int8 và xuất cho thiết bị biên

| model          | biến thể     | dung lượng (MB) |   top-1 | macro-F1 | p50 (ms) | p95 (ms) | nhẹ hơn | nhanh hơn | mất top-1 (điểm) |
| :------------- | :----------- | --------------: | ------: | -------: | -------: | -------: | ------: | --------: | ---------------: |
| m1_lenet       | fp32         |             9.7 |   0.982 |  0.97615 |     1.07 |     1.58 |       1 |         1 |                0 |
| m1_lenet       | int8_dynamic |            2.59 |   0.982 |  0.97615 |     0.93 |     1.13 |    3.75 |      1.15 |                0 |
| m1_lenet       | int8_static  |            2.43 |  0.9815 |  0.97496 |     0.79 |     0.85 |    3.99 |      1.35 |             0.05 |
| m2_vggres      | fp32         |            4.99 | 0.99025 |  0.98459 |     3.43 |     4.07 |       1 |         1 |                0 |
| m2_vggres      | int8_dynamic |            4.95 | 0.99025 |  0.98459 |     3.27 |     3.69 |    1.01 |      1.05 |                0 |
| m3_effnetb0    | fp32         |           16.55 | 0.98625 |  0.98265 |   239.01 |   255.27 |       1 |         1 |                0 |
| m3_effnetb0    | int8_dynamic |           16.38 | 0.98625 |  0.98265 |   241.83 |   246.51 |    1.01 |      0.99 |                0 |
| m3_mobilenetv2 | fp32         |            9.36 | 0.98725 |  0.98093 |    49.22 |    50.58 |       1 |         1 |                0 |
| m3_mobilenetv2 | int8_dynamic |             9.2 | 0.98725 |  0.98093 |    51.12 |    54.52 |    1.02 |      0.96 |                0 |
| m3_resnet18    | fp32         |           44.87 |   0.993 |  0.98972 |    15.09 |    17.51 |       1 |         1 |                0 |
| m3_resnet18    | int8_dynamic |           44.81 |   0.993 |  0.98959 |       15 |    15.78 |       1 |      1.01 |                0 |

Cột **nhẹ hơn** và **nhanh hơn** so với bản `fp32` của cùng mô hình.

⚠️ **`int8_static` chỉ chạy được trên 1/5 mô hình** (`m1_lenet`). Bốn mô hình kia convert thành công nhưng forward **vỡ lúc chạy**, và lý do rất cụ thể:

| Mô hình          | op không có kernel int8 | nguyên nhân             |
| ---------------- | ----------------------- | ----------------------- |
| `m2_vggres`      | `aten::add.out`         | **residual connection** |
| `m3_mobilenetv2` | `aten::add.out`         | **residual connection** |
| `m3_resnet18`    | `aten::add.out`         | **residual connection** |
| `m3_effnetb0`    | `aten::silu.out`        | hàm hoạt hoá SiLU/Swish |
| `m1_lenet`       | *(chạy được)*           | **không có residual**   |

**Ba trong bốn ca là phép `+` của skip connection, không phải hàm hoạt hoá.** Backend `QuantizedCPU` không có kernel cho `+` giữa hai tensor đã lượng tử hoá: phép cộng đó phải được viết bằng `torch.nn.quantized.FloatFunctional().add()` để backend biết cách khớp thang lượng tử của hai nhánh. Viết `a + b` như bình thường thì convert vẫn qua, chỉ forward mới vỡ.

M1 LeNet lượng tử hoá được **chính vì nó không có residual** — kiến trúc cũ nhất lại là kiến trúc duy nhất triển khai int8 tĩnh được mà không phải sửa code.

Hệ quả: tỉ lệ nén **3.99×** chỉ đo được trên `m1_lenet`, **không ngoại suy** cho bốn mô hình kia. Bài học chung: khả năng lượng tử hoá phụ thuộc **cách VIẾT từng phép toán**, không chỉ kiến trúc hay số tham số. `_works()` thử chạy một forward sau convert nên chỉ thiếu một dòng bảng thay vì sập cả script. Chi tiết `docs/SU_CO.md` §10.

Cả 5 mô hình đều xuất được **TorchScript** và **ONNX**, và mỗi bản ONNX đều được
**kiểm lại bằng ảnh test thật** (so logit và so lớp dự đoán) — xuất thành công không
có nghĩa là xuất đúng. Chi tiết một lần báo động sai của phép kiểm này ở
`docs/SU_CO.md` §9.

### 3.7 Ablation — đổi một biến một lần

**Xếp hạng theo `val`, KHÔNG theo `test`.** Chọn cấu hình bằng điểm test là dùng tập test để *chọn*, và khi đó test không còn là ước lượng độc lập cho cấu hình được chọn. Cột `test` dưới đây chỉ để đối chiếu.

**Phép kiểm chéo:** hai cách xếp hạng cho **cùng biến thể thắng ở 6/6 trục**, tương quan val–test **0.986**. Nghĩa là val đủ tin để chọn, và không có dấu hiệu overfit vào val ở mức ảnh hưởng thứ hạng.

**32 run**, mỗi run đổi **đúng một biến** so với cấu hình gốc, ở chế độ ngân sách **15 epoch**. Chế độ này để **xếp hạng** biến thể, không phải để lấy số cuối cùng — cấu hình thắng cần chạy lại ở độ dài đầy đủ trước khi đưa vào bảng so sánh chính.

| trục            | biến thể            | model       | val_macro_f1 | test_top1 | test_macro_f1 |
| :-------------- | :------------------ | :---------- | -----------: | --------: | ------------: |
| augmentation    | aug_none            | m2_vggres   |      0.98895 |   0.98709 |       0.98344 |
| augmentation    | aug_geo_photo       | m2_vggres   |      0.98625 |   0.98852 |       0.98106 |
| augmentation    | aug_geo_photo_erase | m2_vggres   |      0.98412 |   0.98709 |       0.98148 |
| augmentation    | aug_geo             | m2_vggres   |      0.98301 |   0.98686 |        0.9808 |
| components      | no_spatial_dropout  | m2_vggres   |      0.99562 |   0.99066 |       0.98777 |
| components      | full                | m2_vggres   |      0.98625 |   0.98852 |       0.98106 |
| components      | no_residual         | m2_vggres   |      0.98535 |   0.98709 |       0.98504 |
| components      | no_bn               | m2_vggres   |      0.98481 |   0.98575 |       0.97935 |
| label_smoothing | ls0.1               | m2_vggres   |      0.98625 |   0.98852 |       0.98106 |
| label_smoothing | ls0.0               | m2_vggres   |      0.98528 |   0.98844 |       0.97965 |
| label_smoothing | ls0.2               | m2_vggres   |      0.98263 |   0.98717 |       0.97912 |
| preprocess      | prep_he_y           | m2_vggres   |      0.99245 |   0.99153 |       0.98855 |
| preprocess      | prep_he_gray        | m2_vggres   |      0.99166 |   0.98986 |       0.98514 |
| preprocess      | prep_none           | m2_vggres   |      0.98778 |   0.98583 |       0.97874 |
| preprocess      | prep_clahe          | m2_vggres   |      0.98625 |   0.98852 |       0.98106 |
| resolution      | res224              | m3_resnet18 |      0.99578 |   0.99327 |        0.9903 |
| resolution      | res112augafter      | m3_resnet18 |      0.99076 |   0.99272 |       0.99026 |
| resolution      | res112              | m3_resnet18 |      0.98956 |    0.9924 |       0.98718 |
| resolution      | res112real          | m3_resnet18 |      0.98906 |   0.99002 |       0.98688 |
| resolution      | res48               | m2_vggres   |      0.98625 |   0.98852 |       0.98106 |
| resolution      | res32               | m2_vggres   |      0.98581 |   0.98622 |       0.97711 |
| resolution      | res32               | m1_lenet    |      0.98455 |   0.97902 |         0.966 |
| resolution      | res64               | m2_vggres   |      0.98447 |   0.98369 |       0.97489 |
| resolution      | res48               | m1_lenet    |      0.98394 |   0.98472 |       0.97869 |
| resolution      | res64               | m1_lenet    |      0.98324 |   0.98187 |       0.97737 |
| resolution      | res64               | m3_resnet18 |      0.98238 |   0.98709 |       0.97758 |
| scaling         | width2.0            | m2_vggres   |      0.99247 |   0.98971 |       0.98626 |
| scaling         | depth5              | m2_vggres   |      0.98826 |   0.99066 |       0.98468 |
| scaling         | depth4              | m2_vggres   |      0.98625 |   0.98852 |       0.98106 |
| scaling         | width1.0            | m2_vggres   |      0.98625 |   0.98852 |       0.98106 |
| scaling         | width0.5            | m2_vggres   |      0.90067 |   0.94782 |       0.89719 |
| scaling         | depth3              | m2_vggres   |      0.89945 |   0.95455 |       0.90452 |

**★ 4/6 trục cho kết quả ĐI NGƯỢC cấu hình mặc định của dự án:**

- **augmentation** (`m2_vggres`): tốt nhất là `aug_none` (val 0.98895, test 0.98344), cao hơn mặc định `aug_geo_photo` (val 0.98625) **0.27 điểm val**.
- **components** (`m2_vggres`): tốt nhất là `no_spatial_dropout` (val 0.99562, test 0.98777), cao hơn mặc định `full` (val 0.98625) **0.94 điểm val**.
- **preprocess** (`m2_vggres`): tốt nhất là `prep_he_y` (val 0.99245, test 0.98855), cao hơn mặc định `prep_clahe` (val 0.98625) **0.62 điểm val**.
- **scaling** (`m2_vggres`): tốt nhất là `width2.0` (val 0.99247, test 0.98626), cao hơn mặc định `depth4` (val 0.98625) **0.62 điểm val**.

Ba điều phải nói khi trình bày, không được bỏ:

1. **15 epoch là quá ngắn để augmentation và regularisation trả lãi.** Augmentation, dropout và label smoothing đều làm bài toán huấn luyện KHÓ hơn để đổi lấy khái quát tốt hơn về sau. Ở 15 epoch, phần 'khó hơn' đã tới mà phần 'tốt hơn' chưa tới. Mô hình chính chạy 40 epoch, nên không thể dùng bảng này để kết luận 'augmentation vô dụng'.
2. **Mỗi ô ở đây là MỘT seed.** Biên độ seed đo được của M2 là **0,50 điểm macro-F1** (mục 3.5.2). Mọi chênh lệch nhỏ hơn con số đó trong bảng trên **không kết luận được gì** — phần lớn các trục có biên độ dưới 1 điểm.
3. **Nhóm KHÔNG đổi cấu hình mặc định theo bảng này**, vì (1) và (2). Đây là bước xếp hạng để biết nên chạy lại cái gì ở độ dài đầy đủ, không phải kết luận.

**Trục nào thật sự đáng kết luận?** So biên độ từng trục với biên độ seed:

- `augmentation`: biên độ 0.26 điểm — NẰM TRONG nhiễu seed (0.50 điểm)
- `components`: biên độ 0.84 điểm — **vượt** nhiễu seed (0.50 điểm)
- `label_smoothing`: biên độ 0.19 điểm — NẰM TRONG nhiễu seed (0.50 điểm)
- `preprocess`: biên độ 0.98 điểm — **vượt** nhiễu seed (0.50 điểm)
- `resolution`: biên độ 2.43 điểm — **vượt** nhiễu seed (0.50 điểm)
- `scaling`: biên độ 8.91 điểm — **vượt** nhiễu seed (0.50 điểm)

Chỉ những trục **vượt 0.50 điểm** mới đáng rút kết luận từ một seed. Các trục còn lại cần nhiều seed mới nói được gì.

#### 3.7.1 ★ Trục độ phân giải đo gì — và một phép đo đối chứng

**Trục này đo hai thứ KHÁC NHAU tuỳ mô hình** — chỗ dễ đọc sai nhất:

| Mô hình                 | cache dùng              | img_size       | thực chất đo                               |
| ----------------------- | ----------------------- | -------------- | ------------------------------------------ |
| `m1_lenet`, `m2_vggres` | 32 / 48 / 64 (**khớp**) | 32 / 48 / 64   | **độ phân giải thật**                      |
| `m3_resnet18`           | **48 cho cả ba**        | 64 / 112 / 224 | **hệ số nội suy**, không phải chi tiết ảnh |

Nên **không được** đọc "res224 tốt nhất" thành "ảnh nét hơn thì tốt hơn". Nguyên nhân thật là **dung lượng feature map**: ResNet18 thu nhỏ ảnh 32 lần, nên feature map cuối trước Global Average Pooling là

| img_size | feature map | số vị trí không gian |
| -------- | ----------- | -------------------- |
| 48 px    | 512×2×2     | **4**                |
| 64 px    | 512×2×2     | **4**                |
| 112 px   | 512×4×4     | 16                   |
| 224 px   | 512×7×7     | 49                   |

Ở 48 và 64 px, mạng chỉ còn **4 ô** để mô tả cả biển báo. Đó là lý do `res64` kém, không phải vì ảnh mờ hơn.

#### Phép đo đối chứng: cache 48 có làm hại M3 không?

Câu hỏi sắc nhất nhắm vào kết luận chính của báo cáo: *"backbone pretrained của các bạn kém chỉ vì bạn đưa cho nó ảnh đã bị làm mờ?"* Nhóm dựng **cache 112 px thật** rồi train lại cùng cấu hình để trả lời bằng số:

| Nguồn ảnh ở 112 px  | val macro-F1   | test macro-F1  |
| ------------------- | -------------- | -------------- |
| nội suy từ cache 48 | 0.98956        | 0.98718        |
| **cache 112 thật**  | 0.98906        | 0.98688        |
| chênh               | **-0.05 điểm** | **-0.03 điểm** |

Chi tiết thật **không giúp gì** — chênh 0.05 điểm val và 0.03 điểm test, nhỏ hơn nhiễu seed (0.50 điểm) khoảng **10 lần**, và còn hơi NGHIÊNG VỀ PHÍA bản nội suy.

Lý do nằm ở bản thân dữ liệu, không ở đường ống:

|                       |           |
| --------------------- | --------- |
| ROI biển báo trung vị | **31 px** |
| ảnh có ROI > 48 px    | 21 %      |
| ảnh có ROI > 224 px   | **0 %**   |

Biển báo trung vị còn **nhỏ hơn cache 48 px**, và không một ảnh nào trong 51.839 ảnh có chi tiết tới 224 px. Nên lời phản biện "ảnh bị làm mờ" không đứng được: ảnh gốc đã ở độ phân giải đó, và đưa chi tiết thật vào cũng không đổi kết quả.

**Kết luận:** việc cache ở 48 px là lựa chọn về tốc độ và **không cầm chân M3**. Nhưng nó LÀM ĐỔI NGHĨA trục ablation ở trên, nên trục đó phải đọc là *dung lượng feature map*, không phải *độ phân giải ảnh*.


#### Phép đo đối chứng thứ hai: thứ tự augmentation

Mặc định, augmentation được áp ở **độ phân giải cache (48 px)** rồi mới nội suy lên `img_size` — tối ưu tốc độ, vì xoay ảnh 48×48 rẻ hơn xoay 224×224. Nhưng với M3 điều đó khiến ảnh bị **lấy mẫu lại hai lần** (một lần lúc xoay/dịch ở 48 px, một lần lúc nội suy lên), còn M1/M2 chỉ bị một lần vì cache khớp `img_size`.

Lại là một yếu tố bất lợi cho đúng nhóm mô hình mà kết luận chính nói là "không giúp được gì". Nhóm thêm cờ `data.aug_after_resize` để đo:

| Thứ tự (ResNet18 @112 px, cache 48)  | val macro-F1   | test macro-F1  |
| ------------------------------------ | -------------- | -------------- |
| augment **trước** nội suy (mặc định) | 0.98956        | 0.98718        |
| augment **sau** nội suy              | 0.99076        | 0.99026        |
| chênh                                | **+0.12 điểm** | **+0.31 điểm** |

Chênh lệch **đúng chiều dự đoán** (augment sau có lợi cho M3), nhưng bằng **62% dải nhiễu** — gợi ý nhưng **chưa chứng minh được** từ một seed — dải nhiễu seed là 0.50 điểm (mục 3.5.2).

**Phải nói cho đúng mức chắc chắn:** đây là hai phép đo khác nhau về tính dứt khoát. Thí nghiệm cache ở trên chênh 0,03 điểm (~6% dải nhiễu) nên kết luận được là *không ảnh hưởng*. Thí nghiệm này chênh 0.31 điểm (62% dải nhiễu) nên **chỉ kết luận được là không có bằng chứng ở một seed** — không phải là *không có hiệu ứng*. Muốn chốt thì cần 3 seed cho mỗi nhánh (~2 giờ máy), và đó là hướng mở rộng ưu tiên cao nhất của phần này.

Điều có thể nói chắc: **cả hai lựa chọn đường ống đều không giải thích được khoảng cách giữa M2 và các backbone pretrained.** M2 hơn `m3_mobilenetv2` **0.84 điểm**, `m3_effnetb0` **0.42 điểm** macro-F1, đều với p < 1e-6. Hiệu ứng đường ống lớn nhất đo được là 0.31 điểm và **không có ý nghĩa thống kê** — nhỏ hơn cả khoảng cách hẹp nhất (0.42 điểm) trong các cặp đó.

### 3.8 Grad-CAM

Bản đồ nhiệt sinh theo công thức Selvaraju (2017), hook vào khối conv cuối. Ba nhóm hình
cho mỗi mô hình: dự đoán đúng, **dự đoán sai**, và cùng một ảnh trước/sau khi thêm nhiễu.

Độ phân giải heatmap đo được cho kết quả phản trực giác: **M1 mịn nhất** (feature map
12×12, mỗi ô ≈ 4×4 px), **M2 thô nhất** (3×3, mỗi ô ≈ 16×16 px), M3 ở giữa (7×7). M2 —
mô hình thiết kế công phu nhất — cho bản đồ nhiệt thô nhất vì nó có 4 lớp MaxPool so với
2 của M1. Nói "heatmap đúng vào biển báo" với M2 là phát biểu rất lỏng: cả ảnh chỉ có 9 ô.

Nhóm nhấn mạnh Grad-CAM chỉ cho biết **ở đâu**, không cho biết **đặc trưng gì**, và
**không phải bằng chứng nhân quả** — có ca mô hình nhìn đúng vào biển báo mà vẫn suy luận
sai (xem `reports/figures/gradcam_*_wrong.png`).

---

## 4. Khuyến nghị triển khai

Quyết định dựa trên **ba trục**, không chỉ accuracy:

1. Accuracy **kèm p-value McNemar** — nếu chênh lệch không có ý nghĩa thì coi như bằng nhau.
2. **Latency p95** ở batch = 1 (tình huống xe xử lý từng khung ảnh).
3. **Relative robustness** dưới nhiễu — điều kiện vận hành thật.

Mô hình macro-F1 cao nhất là `m3_resnet18` (0.9903). Nhưng trong nhóm **tương đương
thống kê** với nó, mô hình nhanh nhất là **`m2_vggres`** — nhanh hơn **4.9
lần** ở p95. Trả thêm 4.9 lần latency để lấy chênh lệch accuracy *không có ý
nghĩa thống kê* là lựa chọn tồi trên hệ thống thời gian thực.

→ **Khuyến nghị: `m2_vggres`.** Chi tiết suy luận ba bước: `docs/KET_QUA.md` mục 9.

---

## 5. Hạn chế

1. **Đây là bài phân loại, không phải phát hiện.** Mô hình nhận đầu vào là biển báo **đã
   cắt sẵn**; hệ thống thật phải tự tìm biển báo trong khung ảnh toàn cảnh trước, và lỗi
   của bước đó sẽ cộng dồn vào.
2. **Phân phối test giống phân phối train** — cùng chụp ở Đức, cùng loại camera. Chưa kiểm
   được khả năng khái quát sang biển báo nước khác.
3. **Phép đo robustness có hai giới hạn đã lượng hoá:** mức độ nhiễu không so sánh được
   giữa các độ phân giải đối với motion blur (mục 3.5), và nhiễu được áp **sau** bước
   CLAHE nên thiên vị theo mô hình (mục 3.5.1) — giới hạn thứ hai nặng hơn, và đã được
   đo bằng thực nghiệm đối chứng chứ không chỉ nêu ra.
4. **So sánh M2 với M3 lẫn hai biến** (pretrained hay không, và 48 hay 224 px). Trục
   ablation `resolution` **KHÔNG tách được hai biến đó cho M3**, vì cả ba mức 64/112/224
   đều nội suy từ cùng cache 48 px — nó đo dung lượng feature map, không đo chi tiết ảnh
   (mục 3.7.1). Phép đo đối chứng bằng cache 112 px thật cho thấy chi tiết thật **không
   đổi kết quả** (chênh 0,05 điểm, nhỏ hơn nhiễu seed 10 lần), nên kết luận chính không
   bị đe doạ — nhưng biến "pretrained hay không" vẫn chưa được tách sạch khỏi biến
   "kiến trúc", và đó là hướng mở rộng rõ ràng nhất.
5. **Thí nghiệm rò rỉ chỉ chạy trên M1.** Con số 1,09 điểm val ảo đo được trên M1;
   mức thổi phồng có thể khác với mô hình dung lượng lớn hơn, vốn dễ nhớ frame hơn.
6. **Không dùng class weighting hay resampling khi huấn luyện.** `losses.py` có hỗ trợ
   `class_weight` nhưng các config đều để tắt, trong khi dữ liệu mất cân bằng 10,7:1 và
   macro-F1 lại là chỉ số chính. Đây là lựa chọn có ý thức (giữ đường cơ sở đơn giản,
   và macro-F1 đã đủ để phát hiện lỗi ở lớp thiểu số) nhưng là một hướng mở rộng rõ ràng.
7. **Chỉ M2 được chạy nhiều seed** (3 seed); bốn mô hình còn lại một seed. Biên độ seed đo
   được của M2 dùng làm thước đo nhiễu cho cả bảng, nhưng đó là phép ngoại suy.
8. **Chưa kiểm adversarial robustness** (FGSM/PGD) — khác bản chất với nhiễu tự nhiên.

---

## 6. Kết luận

Trên một bộ dữ liệu đã bão hoà, việc đua accuracy không còn là câu hỏi khoa học. Giá trị
của báo cáo này nằm ở ba kết luận chỉ rút ra được khi **đo cẩn thận và kiểm định**:

1. Một CNN **tự xây 1,24 M tham số** vượt hai backbone pretrained ImageNet có ý nghĩa thống
   kê, trong khi nhanh hơn 15–76 lần — vì miền đích có độ phân giải quá thấp để lợi thế
   pretrained phát huy.
2. **Latency chênh 64 lần so với dự đoán từ FLOPs.** Chọn mô hình theo FLOPs sẽ chọn đúng
   mô hình chậm nhất.
3. Mô hình có **accuracy sạch cao nhất lại kém bền nhất** trước nhiễu, còn `m1_lenet` — yếu nhất khi ảnh sạch — bền nhất — và mô hình chính xác nhất cũng không phải mô hình
   nên triển khai.

Điểm chung của cả ba: chúng chỉ lộ ra khi **không tin vào con số đầu tiên nhìn thấy**.

---

## Phụ lục A — Tái lập

```bash
make setup-mac      # môi trường
make data           # tải + tiền xử lý + split theo track
make test           # 16 file test, gồm cổng chặn rò rỉ dữ liệu
make train-m1 train-m2 train-m3
make eval robustness speed gradcam
make report baocao  # sinh KET_QUA.md và BAO_CAO.md
```

Mỗi lần chạy sinh `artifacts/runs/<run_id>/` gồm `best.pt`, `config.yaml` (cấu hình **thực
tế** đã dùng), `train_log.csv` và `result.json` (kèm `seed` và `git_commit`). Mọi siêu tham
số nằm trong YAML, không hard-code. Seed cố định cho torch/numpy/random và
`cudnn.deterministic`; chạy lại cùng seed cho cùng val macro-F1 trong sai số < 0,001.

## Phụ lục B — Phân công

| Thành viên   | Phần phụ trách                                                              |
| ------------ | --------------------------------------------------------------------------- |
| Phong Nguyễn | M1 baseline, mô-đun đánh giá (metrics, McNemar, ECE), bảng so sánh, báo cáo |
| Hoàng        | M2, training engine dùng chung, ablation kiến trúc                          |
| Phong Trần   | M3 (ba backbone), Grad-CAM, đo tốc độ và Pareto                             |
| Huy          | Đường ống dữ liệu, thí nghiệm rò rỉ, ablation dữ liệu, robustness           |

Mỗi file mã nguồn ghi rõ `CHỦ: <tên>` ở đầu docstring.

## Phụ lục C — Tài liệu kèm theo

| File                | Nội dung                                                                                                                    |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `docs/KET_QUA.md`   | toàn bộ bảng số, sinh tự động (333 dòng)                                                                                    |
| `docs/LY_THUYET.md` | cơ sở lý thuyết, công thức, lý do từng lựa chọn thiết kế; có bản đồ code ↔ lý thuyết (942 dòng)                             |
| `docs/SU_CO.md`     | **13 sự cố** đã gặp thật, kèm nguyên nhân, cách sửa và test chặn (410 dòng)                                                 |
| `docs/PHAN_CONG.md` | phân công theo người, kèm khái niệm mỗi người phải nắm (216 dòng)                                                           |
| `docs/INTERFACE.md` | hợp đồng giữa các phần: chữ ký hàm, schema `result.json` (276 dòng)                                                         |
| `reports/tables/`   | **26 bảng CSV**                                                                                                             |
| `reports/figures/`  | **37 hình**                                                                                                                 |
| `notebooks/`        | **1 notebook** — `00_colab_train` chạy trên GPU Colab, là phương án tái lập dự phòng; mọi số trong báo cáo đều từ máy local |
| `tests/`            | **16 file test**, chạy bằng `make test`                                                                                     |
