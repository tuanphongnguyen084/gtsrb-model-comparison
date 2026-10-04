# Phân loại biển báo giao thông GTSRB: so sánh ba hướng tiếp cận Deep Learning

**Môn**: Deep Learning · **Ngày**: 04/10/2026
**Nhóm**: Phong Nguyễn · Hoàng · Phong Trần · Huy

> Bảng số trong báo cáo này được **sinh tự động** từ `reports/tables/` bằng
> `python scripts/make_baocao.py`. Không có con số nào chép tay; mỗi con số truy được
> về một `artifacts/runs/<run_id>/result.json` cụ thể.

---

## Tóm tắt

Nhóm xây và so sánh 1 mô hình phân loại 43 lớp biển báo giao thông Đức trên
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
   thống kê (p < 1e-6), trong khi nhanh hơn 13–68 lần.
2. **Latency không tỉ lệ với FLOPs**, chênh tới **64 lần** về hiệu quả trên mỗi GFLOP.
   Mô hình tên "EfficientNet" lại là mô hình **chậm nhất** trong cả 1.
3. Mô hình có **accuracy sạch cao nhất lại kém bền nhất** trước nhiễu thực tế.

---

## 1. Bài toán và dữ liệu

| | |
|---|---|
| Bộ dữ liệu | GTSRB (German Traffic Sign Recognition Benchmark) |
| Số lớp | 43 |
| Train | 39.209 ảnh · Test chính thức | 12.630 ảnh · **Tổng 51.839** |
| Định dạng | `.ppm`, kích thước thay đổi, **trung vị 43×43 px** (min 25, max 266) |
| Mất cân bằng | **10,7 : 1** (lớp 2 có 2.250 ảnh; lớp 0, 19, 37 chỉ 210) |

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

| Cách split | Số track dùng chung giữa train và val |
|---|---|
| Random theo ảnh (**sai**) | **1.306** |
| Theo track (**đúng**) | **0** |

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

| | Mô hình | Kiến trúc | Input |
|---|---|---|---|
| **M1** | LeNet-style, tự xây | 2 × (Conv 5×5 + MaxPool) + 2 Dense | 48² |
| **M2** | VGG-like, tự xây | 4 stage × 2 Conv 3×3 + BN + residual + SpatialDropout + GAP | 48² |
| **M3** | Transfer learning | ResNet18 / MobileNetV2 / EfficientNet-B0 pretrained ImageNet | 224² |

**Một quan sát đáng chú ý về số tham số:** M1 có **2.424.299** tham số với **2** lớp conv,
trong đó **97,3%** nằm ở *một* lớp `Flatten(9216) → FC(256)`. M2 có **9** lớp conv nhưng
chỉ **1.237.451** tham số — **ít hơn M1** — vì nó thay Flatten+FC bằng Global Average
Pooling (11.051 tham số, giảm ~213 lần ở phần head). Bài học: **số lớp không tỉ lệ với số
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
10,7:1 khiến accuracy bị các lớp đông chi phối). Cả 1 mô hình dùng **chung
một hàm `fit()`** để so sánh được công bằng.

---

## 3. Kết quả

### 3.1 Bảng so sánh chính (tập test chính thức, 12.630 ảnh)

| Mô hình   |   Tham số (M) |   FLOPs (G) |   Dung lượng (MB) | Top-1   | Top-5   |   Macro-F1 |   ECE |   p95 CPU (ms) |   Train (phút) |
|:----------|--------------:|------------:|------------------:|:--------|:--------|-----------:|------:|---------------:|---------------:|
| m1_lenet  |          2.42 |        0.07 |               9.7 | 98.46%  | 99.85%  |     0.9778 | 0.131 |            1.1 |             12 |

**Lưu ý khi đọc:** cột **Top-5 gần như vô nghĩa** ở bài này. Top-5 ra đời cho ImageNet
1000 lớp; trên 43 lớp nó nghĩa là "đúng trong 11,6% số lớp" nên mọi mô hình tử tế đều
~99,9%. Nhóm báo cáo vì đề bài yêu cầu nhưng **kết luận dựa trên top-1 và macro-F1**.

### 3.2 ★ Kiểm định ý nghĩa thống kê (McNemar)

Hai mô hình chạy trên **cùng** tập test nên quan sát là **bắt cặp**; dùng t-test hai mẫu
độc lập là sai về mặt thống kê. McNemar chỉ xét hai ô **bất đồng** của bảng 2×2.

| Cặp                           |   n01 |   n10 |   p-value | Kết luận           |
|:------------------------------|------:|------:|----------:|:-------------------|
| m1_lenet vs m2_vggres         |    35 |   136 |  2.05e-14 | m2_vggres hơn      |
| m1_lenet vs m3_effnetb0       |    83 |   122 |  0.00795  | m3_effnetb0 hơn    |
| m1_lenet vs m3_mobilenetv2    |    89 |   124 |  0.0198   | m3_mobilenetv2 hơn |
| m1_lenet vs m3_resnet18       |    41 |   151 |  3.65e-15 | m3_resnet18 hơn    |
| m2_vggres vs m3_effnetb0      |   106 |    44 |  6.34e-07 | m2_vggres hơn      |
| m2_vggres vs m3_mobilenetv2   |   115 |    49 |  3.86e-07 | m2_vggres hơn      |
| m2_vggres vs m3_resnet18      |    51 |    60 |  0.448    | **tương đương**    |
| m3_effnetb0 vs m3_mobilenetv2 |    68 |    64 |  0.794    | **tương đương**    |
| m3_effnetb0 vs m3_resnet18    |    18 |    89 |  1.31e-11 | m3_resnet18 hơn    |
| m3_mobilenetv2 vs m3_resnet18 |    25 |   100 |  3.62e-11 | m3_resnet18 hơn    |

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

**Mô hình `m1_lenet`:**

|   Lớp | Tên                            |    F1 |   Số ảnh test |
|------:|:-------------------------------|------:|--------------:|
|    27 | Người đi bộ                    | 0.904 |            60 |
|     6 | Hết giới hạn 80 km/h           | 0.905 |           150 |
|    42 | Hết cấm vượt (xe trên 3,5 tấn) | 0.911 |            90 |
|    30 | Cẩn thận băng/tuyết            | 0.93  |           150 |
|    41 | Hết cấm vượt                   | 0.942 |            60 |
|    22 | Đường xấu                      | 0.947 |           120 |
|    26 | Đèn tín hiệu                   | 0.951 |           180 |
|    40 | Vòng xuyến bắt buộc            | 0.951 |            90 |

Các cặp bị nhầm nhiều nhất:

|   Lớp thật | Tên                       |   Đoán thành | Tên                            |   Số lần |
|-----------:|:--------------------------|-------------:|:-------------------------------|---------:|
|         17 | Cấm đi vào                |            9 | Cấm vượt                       |       16 |
|          6 | Hết giới hạn 80 km/h      |           42 | Hết cấm vượt (xe trên 3,5 tấn) |       14 |
|          3 | Giới hạn 60 km/h          |            5 | Giới hạn 80 km/h               |       14 |
|          8 | Giới hạn 120 km/h         |            5 | Giới hạn 80 km/h               |       11 |
|         22 | Đường xấu                 |           25 | Đang thi công                  |       10 |
|         11 | Ưu tiên ở giao lộ kế tiếp |           30 | Cẩn thận băng/tuyết            |       10 |

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

| model       |    fog |   gauss_noise |   low_light |   motion_blur |   occlusion |    mCE |   clean_top1 |
|:------------|-------:|--------------:|------------:|--------------:|------------:|-------:|-------------:|
| m1_lenet    | 0.7254 |        0.9099 |      0.7795 |        0.8494 |      0.5009 | 0.2586 |       0.9846 |
| m2_vggres   | 0.7658 |        0.8801 |      0.6486 |        0.8248 |      0.4697 | 0.2875 |       0.9926 |
| m3_resnet18 | 0.9336 |        0.684  |      0.5576 |        0.9995 |      0.4408 | 0.2817 |       0.9933 |

**Kết quả chính: mô hình có accuracy sạch cao nhất lại KHÔNG phải mô hình bền nhất.**
M1 — yếu nhất về accuracy — có mCE tốt nhất.

⚠️ **Giới hạn phương pháp cần nêu rõ:** nhiễu được áp **sau khi** resize, mà M1/M2 chạy ở
48×48 còn M3 ở 224×224. Kernel motion blur mức 5 là **15 pixel tuyệt đối**: ở 48×48 nó phủ
**31%** chiều rộng ảnh, ở 224×224 chỉ **6,7%**. Vì vậy cột `motion_blur` **không so sánh
được** giữa hai nhóm độ phân giải. Ba cột `gauss_noise`, `low_light`, `occlusion` thì so
sánh được (phép toán theo từng pixel hoặc theo % diện tích), và kết luận chỉ rút từ chúng.
Hướng sửa triệt để: đổi kernel sang tỉ lệ % chiều rộng ảnh.

### 3.6 Tốc độ suy luận và triển khai biên

Đo đúng quy trình: 20 vòng warm-up, **đồng bộ thiết bị trước và sau khi bấm giờ** (GPU
chạy bất đồng bộ — không đồng bộ thì đang đo thời gian *gửi lệnh*), 100 vòng đo, báo
**p50 và p95** (hệ thống thời gian thực quan tâm trường hợp xấu).

**Latency không tỉ lệ với FLOPs:**

| Mô hình        |   FLOPs (G) |   Latency p50 (ms) |   **ms/GFLOP** |
|:---------------|------------:|-------------------:|---------------:|
| m3_resnet18    |        3.65 |              16.65 |            4.6 |
| m2_vggres      |        0.29 |               3.2  |           11   |
| m1_lenet       |        0.07 |               0.9  |           12   |
| m3_mobilenetv2 |        0.65 |              47.87 |           73.4 |
| m3_effnetb0    |        0.83 |             243.43 |          294.1 |

Nếu latency tỉ lệ FLOPs thì cột cuối phải bằng nhau — thực tế chênh **64 lần**.
EfficientNet-B0 có **ít hơn ResNet18 4,4 lần FLOPs** nhưng chậm hơn **14 lần** trên CPU.
Nguyên nhân: MBConv + squeeze-excitation gồm rất nhiều lớp **mảnh**, mỗi lớp tốn chi phí
cố định (launch kernel, truy cập bộ nhớ) mà làm rất ít phép tính → bị chặn bởi **băng
thông bộ nhớ**, không bởi năng lực tính toán.

**Hệ quả thực hành: chọn mô hình theo FLOPs sẽ dẫn tới EfficientNet-B0 — mô hình chậm
nhất trong cả 1.** Muốn nói về triển khai thì phải đo wall-clock trên thiết
bị đích.

### 3.7 Grad-CAM

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

Mô hình macro-F1 cao nhất là `m1_lenet` (0.9778). Nhưng trong nhóm **tương đương
thống kê** với nó, mô hình nhanh nhất là **`m1_lenet`** — nhanh hơn **1.0
lần** ở p95. Trả thêm 1.0 lần latency để lấy chênh lệch accuracy *không có ý
nghĩa thống kê* là lựa chọn tồi trên hệ thống thời gian thực.

→ **Khuyến nghị: `m1_lenet`.** Chi tiết suy luận ba bước: `docs/KET_QUA.md` mục 9.

---

## 5. Hạn chế

1. **Đây là bài phân loại, không phải phát hiện.** Mô hình nhận đầu vào là biển báo **đã
   cắt sẵn**; hệ thống thật phải tự tìm biển báo trong khung ảnh toàn cảnh trước, và lỗi
   của bước đó sẽ cộng dồn vào.
2. **Phân phối test giống phân phối train** — cùng chụp ở Đức, cùng loại camera. Chưa kiểm
   được khả năng khái quát sang biển báo nước khác.
3. **Mức độ nhiễu không so sánh được giữa các độ phân giải** đối với motion blur (mục 3.5).
4. **So sánh M2 với M3 lẫn hai biến** (pretrained hay không, và 48 hay 224 px). Ablation
   độ phân giải là bước cần thiết để tách chúng — đã chuẩn bị nhưng chưa chạy.
5. **Mỗi cấu hình chỉ chạy một seed.** Chênh lệch nhỏ hơn độ nhiễu seed không nên kết luận.
6. **Chưa kiểm adversarial robustness** (FGSM/PGD) — khác bản chất với nhiễu tự nhiên.

---

## 6. Kết luận

Trên một bộ dữ liệu đã bão hoà, việc đua accuracy không còn là câu hỏi khoa học. Giá trị
của báo cáo này nằm ở ba kết luận chỉ rút ra được khi **đo cẩn thận và kiểm định**:

1. Một CNN **tự xây 1,24 M tham số** vượt hai backbone pretrained ImageNet có ý nghĩa thống
   kê, trong khi nhanh hơn 13–68 lần — vì miền đích có độ phân giải quá thấp để lợi thế
   pretrained phát huy.
2. **Latency chênh 64 lần so với dự đoán từ FLOPs.** Chọn mô hình theo FLOPs sẽ chọn đúng
   mô hình chậm nhất.
3. **Mô hình chính xác nhất không phải mô hình bền nhất**, và cũng không phải mô hình nên
   triển khai.

Điểm chung của cả ba: chúng chỉ lộ ra khi **không tin vào con số đầu tiên nhìn thấy**.

---

## Phụ lục A — Tái lập

```bash
make setup-mac      # môi trường
make data           # tải + tiền xử lý + split theo track
make test           # 95 test, gồm cổng chặn rò rỉ dữ liệu
make train-m1 train-m2 train-m3
make eval robustness speed gradcam
make report baocao  # sinh KET_QUA.md và BAO_CAO.md
```

Mỗi lần chạy sinh `artifacts/runs/<run_id>/` gồm `best.pt`, `config.yaml` (cấu hình **thực
tế** đã dùng), `train_log.csv` và `result.json` (kèm `seed` và `git_commit`). Mọi siêu tham
số nằm trong YAML, không hard-code. Seed cố định cho torch/numpy/random và
`cudnn.deterministic`; chạy lại cùng seed cho cùng val macro-F1 trong sai số < 0,001.

## Phụ lục B — Phân công

| Thành viên | Phần phụ trách |
|---|---|
| Phong Nguyễn | M1 baseline, mô-đun đánh giá (metrics, McNemar, ECE), bảng so sánh, báo cáo |
| Hoàng | M2, training engine dùng chung, ablation kiến trúc |
| Phong Trần | M3 (ba backbone), Grad-CAM, đo tốc độ và Pareto |
| Huy | Đường ống dữ liệu, thí nghiệm rò rỉ, ablation dữ liệu, robustness |

Mỗi file mã nguồn ghi rõ `CHỦ: <tên>` ở đầu docstring.

## Phụ lục C — Tài liệu kèm theo

| File | Nội dung |
|---|---|
| `docs/KET_QUA.md` | toàn bộ bảng số, sinh tự động |
| `docs/LY_THUYET.md` | cơ sở lý thuyết, công thức, lý do từng lựa chọn thiết kế |
| `docs/SU_CO.md` | 9 nhóm sự cố đã gặp trong quá trình làm, kèm nguyên nhân và cách sửa |
| `docs/PHAN_CONG.md` | phân công chi tiết + 38 câu hỏi ôn tập |
| `reports/tables/` | 20 bảng CSV |
| `reports/figures/` | 31 hình |
| `notebooks/` | 10 notebook có diễn giải |
