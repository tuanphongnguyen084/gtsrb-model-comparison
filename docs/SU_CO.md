# Nhật ký sự cố

> Gặp lỗi lạ → ghi 1 dòng. Mục đích: cả nhóm không mất thời gian hai lần cho cùng một lỗi.
>
> Các mục dưới đây là lỗi **đã gặp thật** trong lúc dựng dự án này — đọc trước khi code
> sẽ tiết kiệm được vài giờ.

---

## 1. torchvision tải SAI bộ train (26.640 thay vì 39.209 ảnh)

**Triệu chứng.** `prepare_data.py` chỉ dựng được 26.640 ảnh train, không phải 39.209.

**Cách nhận biết nhanh.** Đếm ảnh của lớp 0:

| | lớp 0 | lớp 1 | lớp 2 | tổng |
|---|---|---|---|---|
| bộ THIẾU (IJCNN 2011) | 150 | 1.500 | 1.500 | 26.640 |
| bộ ĐỦ (Final Training) | **210** | **2.220** | **2.250** | **39.209** |

**Nguyên nhân.** `torchvision.datasets.GTSRB(split="train", download=True)` tải
`GTSRB-Training_fixed.zip` — đó là bộ train của **giải IJCNN 2011**, không phải bộ
"Final Training". Tài liệu torchvision không nói rõ chỗ này.

**Cách sửa.** `src/gtsrb/data/download.py` tải trực tiếp
`GTSRB_Final_Training_Images.zip` từ archive gốc của Đại học Bochum (ERDA), không qua
torchvision. Chạy `python scripts/prepare_data.py --download`.
`tests/test_split.py::test_du_51839_anh` chặn lỗi này.

---

## 2. Grad-CAM vỡ với `nn.ReLU(inplace=True)`

**Triệu chứng.**
```
RuntimeError: Output 0 of BackwardHookFunction is a view and is being modified
inplace. This view was created inside a custom Function ...
```

**Nguyên nhân.** `module.register_full_backward_hook()` — cách mà rất nhiều tutorial
Grad-CAM dạy — bọc output của layer thành một **view** của `BackwardHookFunction`.
Nếu lớp ngay sau sửa **tại chỗ** (`nn.ReLU(inplace=True)`, mà cả M1 và M2 đều dùng để
tiết kiệm bộ nhớ) thì PyTorch từ chối.

**Cách sửa (2 phần).**
1. `GradCAM` chỉ dùng **forward hook**, rồi gắn `Tensor.register_hook()` lên chính
   tensor output để lấy gradient. `Tensor.register_hook` không bọc view nên không xung đột.
2. `M1.gradcam_target_layer` trả về **cả khối `self.features`** (output đã qua
   ReLU + MaxPool, không bị lớp nào sửa tại chỗ) thay vì `self.features[3]` (Conv2d,
   output bị ReLU inplace ghi đè ngay).

Xem `src/gtsrb/explain/gradcam.py`.

---

## 3. Grad-CAM cho heatmap phẳng (toàn 0) do lỗi SỐ HỌC

**Triệu chứng.** `heatmap.max() == heatmap.min() == 0` → hình nhìn như "không có vùng
nào quan trọng". Gặp khi model chưa train, hoặc activation của target layer rất nhỏ.

**Nguyên nhân.** Công thức chuẩn hoá phổ biến
`(cam - min) / (max - min + 1e-8)`: khi dải giá trị của `cam` nhỏ hơn `1e-8`
(đo được **1e-13** với EfficientNet-B0 chưa train), hằng số `1e-8` **lấn át mẫu số**
và nghiền kết quả về 0. Lỗi **im lặng** — không báo gì, chỉ ra hình sai.

**Cách sửa.** Chia chính xác khi `span > 0`; khi `span == 0` thì trả về mảng 0 **kèm
`RuntimeWarning`** nêu 3 nguyên nhân thường gặp, thay vì im lặng.

---

## 4. macro-F1 tính trên SỐ LỚP KHÁC NHAU giữa các run

**Triệu chứng.** Không có triệu chứng — đây là **lỗi im lặng** hoàn toàn. Số vẫn ra,
chỉ là các run không so sánh được với nhau.

**Nguyên nhân.** `f1_score(y_true, y_pred, average="macro")` **không** truyền `labels=`
thì sklearn chỉ lấy trung bình trên các lớp **có mặt** trong `y_true`/`y_pred`.
Trên tập test đầy đủ (có cả 43 lớp) thì đúng. Nhưng khi đánh giá trên tập con
(`--subset` lúc smoke test, hoặc một slice robustness) mà thiếu vài lớp thì macro-F1
được tính trên **ít lớp hơn** → cao hơn một cách giả tạo.

**Cách sửa.** Luôn truyền `labels=list(range(NUM_CLASSES))`.
Xem `src/gtsrb/eval/metrics.py`, và `tests/test_metrics.py` có test chặn.

---

## 5. Cảnh báo `NumPy array is not writable` khi đọc cache

**Triệu chứng.**
```
UserWarning: The given NumPy array is not writable, and PyTorch does not support
non-writable tensors.
```

**Nguyên nhân.** Cache mở bằng `np.load(..., mmap_mode="r")` là **read-only**, và
`torch.from_numpy` trên mảng read-only sẽ cảnh báo.

**Cách sửa.** `np.array(cache[i], copy=True)` — copy một ảnh 48×48×3 chỉ là 6.912 byte
nên không ảnh hưởng tốc độ. Xem `src/gtsrb/data/dataset.py`.

---

## 6. `train_log.csv` có header LẶP sau khi `--resume`

**Triệu chứng.** Không có triệu chứng lúc chạy — lại là **lỗi im lặng**. Chỉ lộ ra khi
vẽ learning curve: mọi cột số trở thành kiểu chuỗi và matplotlib vỡ.

**Nguyên nhân.** `CsvLogger` khởi tạo lại khi resume, thấy `self._fieldnames is None`
nên ghi header lần thứ hai vào **giữa** file:
```
epoch,phase,lr,...
1,main,0.001,...
2,main,0.001,...
epoch,phase,lr,...     <- header thứ hai
3,main,0.0009,...
```
`pandas.read_csv` coi dòng header thứ hai là **dữ liệu**, nên cả cột thành `object`.

**Cách sửa.** `CsvLogger.__init__` kiểm tra file đã tồn tại và có nội dung chưa; nếu có
thì đọc tên cột từ dòng đầu và **không** ghi header nữa. Xem `src/gtsrb/engine/callbacks.py`.

**Bài học chung cho cả 7 mục ở trên:** loại lỗi nguy hiểm nhất trong dự án ML không phải
lỗi làm chương trình crash — crash thì thấy ngay. Nguy hiểm là lỗi **vẫn cho ra số**,
chỉ là số sai. Vì vậy mọi bất biến quan trọng đều phải có **test** khoá lại.

---

## 7. Máy NGỦ giữa lúc train → `train_seconds` thành vô nghĩa

**Triệu chứng.** `train_log.csv` có một epoch ghi **31.297 giây** (8,7 giờ) trong khi
các epoch khác đều ~250 giây. `train_seconds` của run đó là **33.858s (9,4 giờ)** thay
vì ~2.800s (47 phút).

**Nguyên nhân.** Laptop chạy **pin** ngủ lúc 01:34, thức lúc 10:05 khi mở nắp. Tiến
trình không chết — nó **ngủ theo máy** rồi chạy tiếp, nhưng `time.time()` vẫn đếm.

★ **Vì sao `caffeinate` không cứu được:** `caffeinate -s` (chặn ngủ hệ thống) **chỉ có
tác dụng khi đang CẮM SẠC**. Chạy pin thì macOS ngủ bất chấp. Đây là hành vi được ghi
trong `man caffeinate` nhưng rất dễ bỏ sót.

**Cách sửa.**
1. Khi train lâu: **cắm sạc VÀ mở nắp máy**, rồi `caffeinate -dimsu &`.
   (Đóng nắp là "clamshell sleep", `caffeinate` cũng không chặn được.)
2. Code: `_detect_sleep()` trong `src/gtsrb/engine/train.py` so thời gian mỗi epoch với
   **trung vị**; epoch nào > 5× trung vị thì đánh dấu `train_seconds_suspect: true` và
   ghi thêm `train_seconds_clean` (ước lượng bằng cách thay epoch bất thường bằng trung vị).
   **Không tự sửa số thô** — sửa ngầm thì người đọc báo cáo không biết số đã bị can thiệp.

**Đây cũng là lý do `--resume` đáng giá:** job tiếp tục được từ epoch dở thay vì làm lại.

---

## 8. ★ Mức độ nhiễu KHÔNG so sánh được giữa các độ phân giải khác nhau

**Triệu chứng.** M3 có `relative robustness` với motion blur = **0,9995** — gần như
không mất gì. Quá đẹp để là thật.

**Nguyên nhân.** Nhiễu được áp **sau khi** Dataset resize về `img_size`. Mà M1/M2 chạy
ở 48×48 còn M3 ở 224×224 (nội suy lên từ 48). Kernel motion blur mức 5 là **15 pixel**:

| Model | Input | 15px = bao nhiêu % ảnh | = bao nhiêu pixel GỐC |
|---|---|---|---|
| M1 / M2 | 48×48 | **31,2%** | 15,0 |
| M3 | 224×224 | 6,7% | **3,2** |

Cùng "mức 5" nhưng M3 chịu nhiễu **nhẹ hơn ~4,7 lần** tính theo pixel gốc.

**Loại nhiễu nào so sánh được:**

| Nhiễu | So sánh được? | Vì sao |
|---|---|---|
| `gauss_noise` | **có** | σ theo cường độ điểm ảnh, phép toán theo từng pixel |
| `low_light` | **có** | gamma theo từng pixel |
| `occlusion` | **có** | che theo **% diện tích**, không theo pixel tuyệt đối |
| `fog` | một phần | bản đồ khoảng cách chuẩn hoá, nhưng β cố định |
| `motion_blur` | **KHÔNG** | kernel tính bằng **pixel tuyệt đối** → phụ thuộc độ phân giải |

**Cách xử lý trong báo cáo.** Nhóm **không** sửa số, mà **nêu rõ giới hạn này** và chỉ
rút kết luận so sánh từ 3 loại nhiễu so sánh được (noise, low-light, occlusion).
Con số motion blur của M3 vẫn báo cáo nhưng kèm chú thích.

**Cách sửa triệt để nếu còn thời gian:** đổi kernel motion blur sang **tỉ lệ với chiều
rộng ảnh** (ví dụ 6% chiều rộng) thay vì pixel tuyệt đối, rồi chạy lại.
Đây là hướng mở rộng, không phải lỗi phải sửa gấp — miễn là báo cáo nói rõ.

---

## 9. ★ Phép kiểm ONNX báo động SAI vì kiểm bằng `torch.randn`

**Triệu chứng.** `verify_onnx()` báo EfficientNet-B0 lệch **2,70e-01** so với PyTorch,
trong khi ba model còn lại lệch ~1e-06. Trông đúng như một model xuất lỗi.

**Kiểm chứng trước khi sửa.** Chạy lại cùng phép so nhưng bằng **ảnh test thật**:

| Model | input = `torch.randn` | input = ảnh test thật | dự đoán khớp |
|---|---|---|---|
| m1_lenet | 4e-06 | 3,8e-06 | 256/256 |
| m2_vggres | 2e-06 | 2,3e-05 | 256/256 |
| m3_resnet18 | 8e-07 | 1,8e-05 | 256/256 |
| m3_effnetb0 | **2,7e-01** ✗ | **1,4e-05** ✓ | **256/256** |

ONNX của EfficientNet-B0 **hoàn toàn đúng**. Lỗi nằm ở phép kiểm.

**Nguyên nhân.** `torch.randn` là nhiễu Gauss chuẩn — nằm **ngoài phân phối** mà model
được train (ảnh đã CLAHE + chuẩn hoá ImageNet). Với input lạ, mạng chạy vào vùng mà SiLU
và khối squeeze-excite cho hoạt hoá cực lớn, nên chênh lệch dấu phẩy động ~1e-7 của từng
phép toán bị **khuếch đại qua 82 lớp**. Mạng càng sâu, input càng lạ thì càng phóng to —
đúng vì vậy mà **chỉ** model sâu nhất bị báo động, làm nó trông giống lỗi riêng của
EfficientNet.

**Vì sao đây là lỗi đáng ghi lại.** Một phép kiểm hay báo động sai **tệ hơn** là không có
phép kiểm: cả nhóm sẽ học cách phớt nó đi, và lần nó báo ĐÚNG thì cũng không ai tin. Nó
cũng dễ dẫn tới kết luận sai ngược lại — "EfficientNet không xuất ONNX được" — rồi viết
vào báo cáo.

**Cách sửa.** `verify_onnx(..., sample=<batch ảnh thật>)`, và thêm tiêu chí **thực sự
quan trọng khi triển khai**: `argmax` có đổi không. Lệch logit 1e-3 mà lớp dự đoán không
đổi thì model vẫn dùng được; lệch 1e-5 mà đổi lớp thì không. Không truyền `sample` thì
hàm vẫn chạy nhưng **ghi rõ trong log là phép kiểm đang yếu**.

**Bài học chung.** Mọi phép kiểm số học phải chạy trên **dữ liệu cùng phân phối với lúc
triển khai**. Sai số dấu phẩy động không phải hằng số — nó là hàm của độ sâu mạng *và*
của việc input có nằm trong miền đã train hay không.

---

## 10. int8 static VỠ trên EfficientNet-B0 vì `SiLU` không có kernel lượng tử hoá

**Triệu chứng.** `quantize_static` convert xong nhưng chạy thì:
`Could not run 'aten::silu.out' with arguments from the 'QuantizedCPU' backend`.

**Nguyên nhân.** Backend lượng tử hoá CPU của PyTorch chưa có kernel int8 cho `SiLU`
(Swish) — hàm hoạt hoá mà EfficientNet dùng ở mọi khối. ReLU thì có.

**Vì sao không sập cả mẻ.** `_works()` chạy thử một forward sau khi convert và bắt được
lỗi này, nên bảng chỉ thiếu dòng `int8_static` của EfficientNet-B0 thay vì cả script
chết. Đây là lý do `compare_quantized()` luôn **thử chạy** model vừa nén, chứ không tin
rằng convert thành công là dùng được. Lần đầu dựng `quantize_static` cũng vướng đúng
kiểu đó: thiếu `QuantStub`/`DeQuantStub` nên convert chạy qua nhưng forward thì vỡ —
sửa bằng `_QuantWrapper` trong `src/gtsrb/deploy/export.py`. Hai lần, cùng một bài học:
**convert được ≠ chạy được**.

**Kết luận cho báo cáo.** Khả năng lượng tử hoá **phụ thuộc hàm hoạt hoá**, không chỉ
phụ thuộc kiến trúc. Chọn model cho thiết bị biên mà định dùng int8 thì phải kiểm
backend có kernel cho mọi op trong model — chỉ đếm tham số và FLOPs là không đủ.

---

## 11. ★ Mất một mẻ Colab 2 giờ vì `!python` làm lỗi IM LẶNG

**Triệu chứng.** Chạy Run all trên Colab, đợi 2 giờ, tải kết quả về thì chỉ có
**3 run M3** — không có run ablation nào, mà ablation mới là thứ cần Colab nhất.

**Nguyên nhân — hai lỗi cộng lại.**

1. Ô ablation gọi `!python scripts/run_ablation.py --axes all --budget
   --set train.num_workers=2`, nhưng `run_ablation.py` **không có** tham số
   `--set`. argparse thoát với mã 2 ngay giây đầu.
2. Với `!python`, lệnh lỗi **chỉ in thông báo rồi notebook CHẠY TIẾP**. Run all
   đi tiếp sang ô train M3 — ô đó chạy đúng, mất 100 phút, và để lại 3 run trông
   hoàn toàn bình thường. Không có gì báo động.

**Cách nhận biết.** Trên Drive, thư mục run đầu tiên được tạo lúc 15:26 và là
một run **M3**. Nếu ablation có chạy thì phải có run ablation tạo trước đó —
ablation nằm ở ô 14, M3 ở ô 18.

**Cách sửa.**

- `run_ablation.py` nhận `--set`, và override chung được đặt **TRƯỚC** override
  của trục: nếu ai vô tình truyền đúng biến mà trục đang thí nghiệm (ví dụ
  `--set data.img_size=224` khi chạy trục `resolution`) thì **trục vẫn thắng**,
  thí nghiệm không bị phá âm thầm. Có cảnh báo khi trùng biến.
- Notebook thay `!python` bằng hàm `chay()` — nó đọc mã thoát và **raise
  SystemExit** để Run all dừng ngay tại ô lỗi. Việc đã xong nằm trên Drive nên
  lần chạy lại sẽ bỏ qua, không mất công.
- `tests/test_notebook_colab.py` trích mọi lệnh script trong notebook rồi chạy
  với đúng đối số đó, dừng ngay sau khi argparse phân tích xong.

**★ Bẫy trong chính test này — cách làm đầu tiên của tôi SAI.** Tôi thêm
`--help` vào lệnh và kiểm mã thoát. Nhưng argparse xử lý `--help` **TRƯỚC** khi
kiểm các tham số khác, nên nó in help, thoát 0, và **bỏ qua `--set` lạ**. Test
vẫn xanh dù lỗi còn nguyên. Phải để `parse_args()` chạy thật rồi chặn ngay sau
đó. Đã kiểm ngược: bỏ `--set` ra khỏi script thì test đỏ với đúng thông báo
`unrecognized arguments: --set train.num_workers=2`.

**Bài học.** Một test không đỏ khi lỗi quay lại thì không phải test. Mỗi lần
viết test chặn lỗi, phải **tái tạo lại lỗi** để xem test có bắt được không.

---

## 12. ★★ Nhiễu áp SAU CLAHE — phép đo robustness thiên vị theo model

**Triệu chứng.** Hai con số không thể tin được trong bảng robustness:

| model | `low_light` relative_top1 |
|---|---|
| m3_effnetb0 | **0,009** |
| m3_mobilenetv2 | 0,088 |
| m1_lenet | 0,779 |

Đoán bừa trên 43 lớp cho 1/43 = 0,023. EfficientNet-B0 sụp xuống **dưới mức
đoán bừa**, ngay từ mức 1 (gamma 1,5 — chỉ hơi tối), rồi phẳng tịt 0,0094–0,0100
qua cả 5 mức. Đường robustness thật thì giảm dần.

**Nguyên nhân.** Đường ống áp nhiễu **SAU** CLAHE:

```
ảnh gốc -> cắt ROI -> CLAHE -> resize 48 -> nội suy 224 -> LÀM TỐI -> model
```

Khi triển khai thật thì thứ tự **ngược lại** — ảnh vào camera đã tối rồi, CLAHE
mới chạy và **bù lại** độ tối. Áp nhiễu sau CLAHE nghĩa là bước bù sáng diễn ra
*trước* bước làm tối, nên nó không bù được gì.

**★ Đo thật, và kết quả KHÔNG như tôi đoán ★**

`scripts/kiem_thu_tu_clahe.py`, 430 ảnh (10 ảnh/lớp × 43 lớp), 2 loại nhiễu so
sánh được × 3 mức. Chênh = (trước CLAHE − sau CLAHE), dương nghĩa là sửa thứ tự
giúp model:

| model | gauss_noise | low_light | trung bình |
|---|---|---|---|
| m3_effnetb0 | +0,368 | +0,361 | **+0,364** |
| m3_mobilenetv2 | +0,332 | +0,265 | **+0,298** |
| m3_resnet18 | +0,003 | +0,006 | +0,005 |
| m2_vggres | −0,111 | −0,078 | **−0,095** |
| m1_lenet | −0,155 | −0,198 | **−0,176** |

**Dấu của hiệu ứng PHỤ THUỘC VÀO MODEL.** Tôi đã đoán sửa thứ tự sẽ giúp mọi
model. Thực tế nó giúp hai backbone ImageNet rất nhiều, trung tính với
ResNet18, và **làm M1/M2 KÉM ĐI**.

Vì sao M1/M2 kém đi: CLAHE là cân bằng tương phản **cục bộ**. Áp nhiễu trước thì
CLAHE **khuếch đại** chính cái nhiễu đó, rồi ảnh bị thu về 48×48. Với mô hình
nhỏ ở độ phân giải thấp, nhiễu đã bị khuếch đại còn tệ hơn ảnh tối ban đầu. Còn
với mô hình 224px thì việc CLAHE lấy lại độ sáng tổng thể quan trọng hơn.

**Thứ hạng độ bền đổi thế nào:**

| | thứ tự ĐANG đo | thứ tự TRIỂN KHAI |
|---|---|---|
| 1 | m1_lenet **0,798** | m1_lenet **0,622** |
| 2 | m2_vggres 0,705 | m2_vggres **0,611** |
| 3 | m3_resnet18 0,528 | m3_resnet18 0,532 |
| 4 | m3_mobilenetv2 0,133 | m3_effnetb0 0,456 |
| 5 | m3_effnetb0 0,091 | m3_mobilenetv2 0,431 |

**Kết luận cho báo cáo — ba điều, không được lẫn:**

1. Kết luận "M1 bền nhất dù accuracy sạch kém nhất" **vẫn đứng** ở cả hai thứ tự.
2. Nhưng **khoảng cách M1 với M2 co từ 0,093 xuống 0,011** — tức hai mô hình
   gần như BẰNG NHAU về độ bền. Nói "M1 bền nhất" mà không nói con số này là
   phóng đại.
3. Hai con số 0,009 và 0,088 là **hiện vật đo**, không phải tính chất mô hình.
   Thực tế là 0,456 và 0,431. Biên độ giữa 5 mô hình co từ **0,71 xuống 0,19**.

**Không có thứ tự nào "đúng tuyệt đối".** Áp sau CLAHE thì thiên vị M1/M2; áp
trước CLAHE thì mô phỏng sát điều kiện triển khai hơn nhưng lại trừng phạt mô
hình nhỏ ở độ phân giải thấp. Nhóm báo cáo **cả hai**, và nêu rõ phép đo chính
(`robustness.csv`) dùng thứ tự "sau CLAHE".

**★ Cổng tự kiểm — vì sao script này có ★**

Lần thử đầu tiên tôi dựng lại đường ống SAI: resize ảnh gốc trực tiếp sang 224
thay vì đi qua cache 48×48 rồi nội suy lên. Nhánh ảnh sạch cho accuracy **0,352**
thay vì 0,99, nên mọi con số sau đó vô nghĩa — mà bảng kết quả in ra trông hoàn
toàn bình thường, không có gì gợi ý là sai.

Vì vậy script **bắt buộc** kiểm nhánh ảnh sạch khớp `robustness.csv` trong 0,02
trước khi in bất cứ kết luận nào, lệch hơn thì `exit 1`. Lần chạy thật: lệch
0,0062–0,0129 cho cả 5 mô hình — ĐẠT.

---

## 13. Các lỗi KHÁC NÊN BIẾT TRƯỚC (chưa gặp nhưng gần như chắc chắn sẽ gặp)

| Triệu chứng | Nguyên nhân | Cách sửa |
|---|---|---|
| `mat1 and mat2 shapes cannot be multiplied` | hard-code số chiều sau `Flatten` rồi đổi `img_size` | M1 **tự tính** bằng một lần forward thử — xem `m1_lenet.py` |
| `Expected all tensors to be on the same device` | quên `.to(device)` cho tensor phụ | mọi hàm trong `src/` nhận `device` từ ngoài, không gọi `.cuda()` cứng |
| `autocast` không có tác dụng / kết quả sai trên MPS | MPS chưa hỗ trợ autocast đầy đủ | `fit()` tự **tắt AMP** khi `device.type != "cuda"` |
| `DataLoader` treo trên macOS | `num_workers>0` + fork + OpenCV | `configs/base.yaml` đặt `num_workers: 0`; trên Colab đổi thành 2–4 |
| OOM trên Colab khi train M3 | batch 128 ở 224×224 | `configs/m3_*.yaml` đặt `batch_size: 64` |
| val accuracy cao bất thường, cao hơn test nhiều | split random → **rò rỉ dữ liệu** | `pytest tests/test_split.py` |
| M3 accuracy tệ mà không báo lỗi | quên đổi sang `img_size=224` + `normalize=imagenet` | `scripts/train.py` **tự khớp** theo `model.expected_img_size` / `model.normalize_mode` |
