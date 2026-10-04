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

## 9. Các lỗi KHÁC NÊN BIẾT TRƯỚC (chưa gặp nhưng gần như chắc chắn sẽ gặp)

| Triệu chứng | Nguyên nhân | Cách sửa |
|---|---|---|
| `mat1 and mat2 shapes cannot be multiplied` | hard-code số chiều sau `Flatten` rồi đổi `img_size` | M1 **tự tính** bằng một lần forward thử — xem `m1_lenet.py` |
| `Expected all tensors to be on the same device` | quên `.to(device)` cho tensor phụ | mọi hàm trong `src/` nhận `device` từ ngoài, không gọi `.cuda()` cứng |
| `autocast` không có tác dụng / kết quả sai trên MPS | MPS chưa hỗ trợ autocast đầy đủ | `fit()` tự **tắt AMP** khi `device.type != "cuda"` |
| `DataLoader` treo trên macOS | `num_workers>0` + fork + OpenCV | `configs/base.yaml` đặt `num_workers: 0`; trên Colab đổi thành 2–4 |
| OOM trên Colab khi train M3 | batch 128 ở 224×224 | `configs/m3_*.yaml` đặt `batch_size: 64` |
| val accuracy cao bất thường, cao hơn test nhiều | split random → **rò rỉ dữ liệu** | `pytest tests/test_split.py` |
| M3 accuracy tệ mà không báo lỗi | quên đổi sang `img_size=224` + `normalize=imagenet` | `scripts/train.py` **tự khớp** theo `model.expected_img_size` / `model.normalize_mode` |
