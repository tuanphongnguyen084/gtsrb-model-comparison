# Kế Hoạch Thiết Kế Hệ Thống Notebook GTSRB (Trực Quan Hóa & Trình Bày)

## 1. Mục tiêu và Nguyên tắc Thiết kế
1. **Độc lập hoàn toàn (Self-contained)**:
   * Mỗi notebook hoạt động độc lập tuyệt đối, chứa toàn bộ mã nguồn cần thiết (từ cấu hình, Dataset, Model đến Training loop và Metric).
   * **Không gọi/import hàm từ các file `.py` ngoài thư mục** (`common.py`, `train_m1.py`,...). Khi mở bất kỳ notebook nào trên Google Colab hay Jupyter Lab, người dùng có thể chạy thẳng từ trên xuống dưới mà không lo lỗi đường dẫn `sys.path`.
2. **Linh hoạt tham số & Luôn ưu tiên tài nguyên sẵn có**:
   * Thiết lập các cờ điều khiển rõ ràng ở cell đầu: `FORCE_DOWNLOAD`, `FORCE_REPROCESS`, `FORCE_RETRAIN`.
   * Mặc định luôn là `False`: nếu máy đã có dữ liệu tiền xử lý (`index.csv`, `images_48_clahe.npy`) hoặc checkpoint đã train (`checkpoints/*.pt`), notebook sẽ nạp ngay lập tức để hiển thị kết quả, vẽ biểu đồ và phân tích mà không bắt người dùng/giảng viên phải ngồi chờ train lại.
3. **Cấu trúc Cell phân rã - Mỗi hàm một Cell Code**:
   * Tránh việc gom một khối mã nguồn quá lớn vào một cell.
   * Mỗi hàm, mỗi lớp (Class) hoặc mỗi bước logic được đặt trong đúng một Cell Code riêng biệt.
4. **Văn phong Markdown nghiêm chuẩn học thuật (Academic Report Style)**:
   * Không sử dụng biểu tượng cảm xúc (icon) bừa bãi.
   * Phân cấp tiêu đề mạch lạc theo hệ thống số thứ tự: `1.`, `1.1.`, `1.1.1.`,...
   * Mỗi cell Markdown trước một cell Code sẽ đóng vai trò giải thích:
     * Cơ sở lý thuyết của giải thuật/kiến trúc.
     * Ý nghĩa các tham số.
     * Dòng chảy dữ liệu (Input shape $\rightarrow$ Output shape).
5. **Vị trí lưu trữ**:
   * Tất cả các file notebook sẽ được đặt tại: `d:\code\Python\DSEB_ex\DL\gtsrb-model-comparison\notebooks/`.

---

## 2. Danh mục 5 Notebook Được Đề Xuất

Để quá trình báo cáo và trình diễn thuận tiện nhất, hệ thống chia thành 5 notebook chuyên biệt:

```text
notebooks/
│
├── 01_preprocess.ipynb             # [Tiền xử lý] Tải, crop ROI, CLAHE, Split chống rò rỉ, tạo cache .npy
├── 02_train_m1.ipynb               # [Mô hình M1] LeNet Baseline từ số 0, chứng minh điểm nghẽn FC 97.3%
├── 03_train_m2.ipynb               # [Mô hình M2] VGG-ResNet từ số 0 (BN, Residual, Spatial Dropout, GAP)
├── 04_train_m3.ipynb               # [Mô hình M3] Transfer Learning (ResNet18 / MobileNetV2 / EfficientNet-B0)
└── 05_evaluation_comparison.ipynb  # [Tổng kết] Bảng so sánh 5 model, kiểm định McNemar, phân tích lỗi
```

> **Lý do bổ sung notebook thứ 5 (`05_evaluation_comparison.ipynb`)**:
> Khi trình bày trước hội đồng hoặc giáo viên, việc phải mở lần lượt từng notebook train để so sánh là rất bất tiện. Notebook 05 đóng vai trò là "Bảng điều khiển trung tâm" (Executive Summary Dashboard), nạp đồng thời cả 5 checkpoint đã lưu và xuất ra: bảng so sánh đa tiêu chí, kiểm định McNemar chứng minh ý nghĩa thống kê, ma trận nhầm lẫn (Confusion Matrix) và biểu đồ Pareto đánh đổi giữa độ chính xác và độ trễ.

---

## 3. Bản Thiết Kế Chi Tiết Từng Notebook

### 3.1. `01_preprocess.ipynb` — Tiền Xử Lý Dữ Liệu & Chống Rò Rỉ Dữ Liệu
* **Mục 1. Khởi tạo môi trường và Cấu hình tham số**
  * *Markdown 1.1*: Giới thiệu bài toán phân loại biển báo GTSRB, nguồn dữ liệu gốc Đại học Bochum (Sid ERDA).
  * *Code Cell 1*: Import thư viện (`cv2`, `numpy`, `pandas`, `sklearn`, `matplotlib`, `urllib`, `zipfile`).
  * *Markdown 1.2*: Cấu hình cờ thực thi và đường dẫn dữ liệu.
  * *Code Cell 2*: Thiết lập cờ `FORCE_DOWNLOAD = False`, `FORCE_REPROCESS = False`, đường dẫn `data/raw`, `data/processed`.
* **Mục 2. Tải và Giải nén Dữ liệu Gốc**
  * *Markdown 2.1*: Cơ chế tải 3 tệp lưu trữ gốc (Full Training 39.209 ảnh, Full Test 12.630 ảnh, Test Ground Truth). Giải thích vì sao không dùng bộ dữ liệu tải qua torchvision (bị thiếu ảnh).
  * *Code Cell 3*: Định nghĩa hàm `download_gtsrb()`.
  * *Code Cell 4*: Thực thi tải dữ liệu (kiểm tra điều kiện: chỉ tải khi `FORCE_DOWNLOAD=True` hoặc thư mục chưa tồn tại).
* **Mục 3. Xây dựng Bảng Siêu Dữ Liệu (Metadata Index)**
  * *Markdown 3.1*: Cấu trúc file nhãn `.csv` (phân tách bởi dấu chấm phẩy `;`) và quy ước tên file `00005_00017.ppm` tương ứng với Track 5, Frame 17.
  * *Code Cell 5*: Định nghĩa hàm `_read_gt()`.
  * *Code Cell 6*: Định nghĩa hàm `build_index()`.
* **Mục 4. Phân chia Train/Validation theo Track (Chống Rò Rỉ Dữ Liệu)**
  * *Markdown 4.1*: Khái niệm rò rỉ dữ liệu do video sequence 30 frame liên tiếp. Lý thuyết phân chia theo nhóm bằng `StratifiedGroupKFold`.
  * *Code Cell 7*: Định nghĩa hàm `make_split()`.
  * *Markdown 4.2*: Kiểm tra toàn vẹn toán học (Assertion Test).
  * *Code Cell 8*: Lệnh assert chứng minh số track dùng chung giữa tập Train và Validation bằng đúng 0.
* **Mục 5. Thuật toán Xử lý Ảnh (ROI Crop & CLAHE)**
  * *Markdown 5.1*: Lý thuyết cắt ROI nới lề 10% (tránh mất viền mang ngữ nghĩa) và thuật toán cân bằng độ tương phản cục bộ CLAHE trên kênh L (LAB) để giữ nguyên sắc độ màu sắc.
  * *Code Cell 9*: Khởi tạo bộ cân bằng `_CLAHE` và định nghĩa hàm `_process_one()`.
* **Mục 6. Xây dựng Bộ Đệm Cache (`.npy`)**
  * *Markdown 6.1*: Cơ chế lưu trữ mảng NumPy `uint8` phục vụ nạp nhanh qua `mmap` lúc huấn luyện.
  * *Code Cell 10*: Định nghĩa hàm `build_cache()`.
  * *Code Cell 11*: Khối điều phối thực thi: Nếu `FORCE_REPROCESS=False` và file `images_48_clahe.npy` đã tồn tại thì bỏ qua; ngược lại chạy tiền xử lý và lưu cache.
* **Mục 7. Trực quan hóa Kết quả Tiền Xử Lý**
  * *Markdown 7.1*: So sánh trực quan chất lượng ảnh trước và sau khi qua bộ lọc ROI + CLAHE.
  * *Code Cell 12*: Trực quan hóa lưới 10 ảnh mẫu (Ảnh gốc vs Ảnh xử lý).

---

### 3.2. `02_train_m1.ipynb` — Mô Hình M1: LeNet Baseline Từ Số 0
* **Mục 1. Khởi tạo và Cấu hình Thực thi**
  * *Markdown 1.1*: Giới thiệu mô hình mốc tham chiếu (Baseline). Lý do cố ý không dùng BatchNorm, Residual và Spatial Dropout.
  * *Code Cell 1*: Import PyTorch, Torchvision, Scikit-learn, Matplotlib.
  * *Markdown 1.2*: Cấu hình tham số và cờ huấn luyện.
  * *Code Cell 2*: Cờ `FORCE_RETRAIN = False`, `DEMO_EPOCHS = 2`, `BATCH_SIZE = 64`, `LR = 1e-3`, cấu hình phần cứng `device`.
* **Mục 2. Pipeline Dữ liệu (Tự chứa hoàn toàn)**
  * *Markdown 2.1*: Chuẩn hóa theo phân phối GTSRB tập train và các ràng buộc Augmentation (cấm flip ngang, cấm đổi màu).
  * *Code Cell 3*: Định nghĩa hàm `_build_normalize()`.
  * *Code Cell 4*: Định nghĩa hàm `build_train_transform()`.
  * *Code Cell 5*: Định nghĩa class `GTSRBDataset` (đọc mảng `.npy` qua `mmap_mode="r"`).
  * *Code Cell 6*: Định nghĩa hàm `make_loader()`.
  * *Code Cell 7*: Khởi tạo `train_loader`, `val_loader`, `test_loader` cho M1 ($48 \times 48$).
* **Mục 3. Định nghĩa Kiến trúc M1LeNet**
  * *Markdown 3.1*: Phân tích cấu trúc 2 block Conv-Pool và 2 lớp Fully-Connected.
  * *Code Cell 8*: Định nghĩa class `M1LeNet`.
  * *Markdown 3.2*: Phân tích định lượng tham số mô hình.
  * *Code Cell 9*: Định nghĩa hàm `parameter_table()`.
  * *Code Cell 10*: Xuất bảng tham số thực tế, chứng minh 97.3% tham số tập trung tại lớp `Linear(9216, 256)`.
* **Mục 4. Thuật toán Tối ưu và Đo lường**
  * *Markdown 4.1*: Hàm mất mát Cross-Entropy (Label Smoothing 0.1), thuật toán Adam và Cosine Annealing LR.
  * *Code Cell 11*: Định nghĩa hàm `compute_metrics()`.
  * *Code Cell 12*: Định nghĩa hàm `predict()` và `evaluate()`.
  * *Code Cell 13*: Định nghĩa hàm `train_one_epoch()`.
  * *Code Cell 14*: Định nghĩa hàm `train_m1()`.
* **Mục 5. Thực thi: Nạp Checkpoint hoặc Huấn luyện**
  * *Markdown 5.1*: Điều phối luồng thực thi dựa trên cờ `FORCE_RETRAIN`.
  * *Code Cell 15*: Logic rẽ nhánh: Nạp `checkpoints/m1_lenet_best.pt` nếu có; nếu `FORCE_RETRAIN=True` thì bắt đầu train.
* **Mục 6. Báo cáo Kết quả trên Tập Test**
  * *Markdown 6.1*: Đánh giá hiệu năng trên 12.630 ảnh test chính thức.
  * *Code Cell 16*: In bảng chỉ số Top-1, Macro-F1 và bảng báo cáo phân loại chi tiết 43 lớp (Classification Report).
  * *Code Cell 17*: Vẽ biểu đồ phân bố F1-score trên 43 lớp để phát hiện các lớp yếu.

---

### 3.3. `03_train_m2.ipynb` — Mô Hình M2: VGG-ResNet Sâu Từ Số 0
* **Mục 1. Khởi tạo và Cấu hình Thực thi**
  * *Markdown 1.1*: Giới thiệu mô hình M2 và mục tiêu giải quyết triệt để hạn chế của M1.
  * *Code Cell 1*: Import các thư viện cần thiết.
  * *Code Cell 2*: Thiết lập cờ `FORCE_RETRAIN = False`, `DEMO_EPOCHS = 2`, `BATCH_SIZE = 64`, `LR = 1e-3`.
* **Mục 2. Pipeline Dữ liệu**
  * *Code Cell 3*: Tự chứa đầy đủ class `GTSRBDataset` và khởi tạo DataLoaders ($48 \times 48$, chuẩn hóa GTSRB).
* **Mục 3. Năm Lựa chọn Thiết kế Kiến trúc của M2**
  * *Markdown 3.1*: Phân tích chi tiết 5 điểm cải tiến: Cặp Conv 3×3 (VGG), BatchNorm2d, Residual Shortcut ($y = F(x) + x$), Spatial Dropout (`Dropout2d`), và Global Average Pooling (GAP).
  * *Code Cell 4*: Định nghĩa class `ResidualBlock`.
  * *Code Cell 5*: Định nghĩa class `M2VggRes`.
  * *Code Cell 6*: Định nghĩa hàm khởi tạo trọng số Kaiming He-Normal (`_init_weights()`).
  * *Code Cell 7*: So sánh số lượng tham số M2 (1.24M) với M1 (2.42M) — M2 sâu gấp 4 lần nhưng tham số giảm gần 50%.
* **Mục 4. Huấn luyện Mô hình**
  * *Code Cell 8*: Tự chứa các hàm đo lường `compute_metrics()`, `evaluate()`, `train_one_epoch()`.
  * *Code Cell 9*: Định nghĩa hàm điều phối `train_m2()`.
  * *Code Cell 10*: Rẽ nhánh: Nạp `checkpoints/m2_vggres_best.pt` (mặc định) hoặc train lại khi `FORCE_RETRAIN=True`.
* **Mục 5. Báo cáo Thực nghiệm**
  * *Code Cell 11*: Đo lường và in bảng kết quả Top-1, Macro-F1 trên tập test.
  * *Code Cell 12*: Trực quan hóa sự hội tụ của mô hình (Train/Val Loss và Macro-F1 theo Epoch nếu có lịch sử log).

---

### 3.4. `04_train_m3.ipynb` — Mô Hình M3: Transfer Learning Pretrained ImageNet
* **Mục 1. Khởi tạo và Lựa chọn Backbone**
  * *Markdown 1.1*: Cơ sở lý thuyết Transfer Learning từ 1.28 triệu ảnh ImageNet sang bài toán biển báo.
  * *Code Cell 1*: Import PyTorch, Torchvision, Scikit-learn.
  * *Code Cell 2*: Cấu hình tham số:
    * `BACKBONE = "resnet18"` (Lựa chọn giữa `"resnet18"`, `"mobilenetv2"`, `"effnetb0"`).
    * `FORCE_RETRAIN = False`, `DEMO_EPOCHS = 2`, `BATCH_SIZE = 64`.
* **Mục 2. Pipeline Dữ liệu Đặc thù cho Transfer Learning**
  * *Markdown 2.1*: Phân tích vì sao phải upsample $48 \times 48 \rightarrow 224 \times 224$ (khớp stride 32 của backbone) và vì sao phải chuẩn hóa bằng thống kê ImageNet thay vì GTSRB.
  * *Code Cell 3*: Class `GTSRBDataset` với cơ chế nội suy song tuyến (Bilinear Interpolation) lên $224 \times 224$ và ImageNet Normalization.
  * *Code Cell 4*: Khởi tạo DataLoaders ở độ phân giải $224 \times 224$.
* **Mục 3. Xây dựng Kiến trúc `M3Transfer`**
  * *Markdown 3.1*: Phân tích 3 triết lý kiến trúc (ResNet, MobileNetV2, EfficientNet-B0) và cơ chế thay thế Fully-Connected Head 43 lớp.
  * *Code Cell 5*: Định nghĩa class `M3Transfer`.
* **Mục 4. Chiến lược Huấn luyện Hai Pha (Two-Phase Training)**
  * *Markdown 4.1*: Hiện tượng Catastrophic Forgetting và nguyên lý đóng băng backbone ở Pha 1, mở băng với Discriminative Learning Rate ở Pha 2.
  * *Code Cell 6*: Định nghĩa hàm huấn luyện Pha 1 (`freeze_backbone=True`).
  * *Code Cell 7*: Định nghĩa hàm huấn luyện Pha 2 (`freeze_backbone=False`, AdamW, LR nhỏ).
  * *Code Cell 8*: Định nghĩa hàm điều phối `train_m3()`.
* **Mục 5. Nạp Checkpoint hoặc Huấn luyện**
  * *Code Cell 9*: Logic nạp checkpoint `checkpoints/m3_{BACKBONE}_best.pt` hoặc huấn luyện.
* **Mục 6. Báo cáo Kết quả trên Tập Test**
  * *Code Cell 10*: Đo lường và in bảng kết quả kiểm thử trên 12.630 ảnh.

---

### 3.5. `05_evaluation_comparison.ipynb` — Báo Cáo Đánh Giá Tổng Thể & Kiểm Định Thống Kê
* **Mục 1. Khởi tạo Môi trường và Dữ liệu Test**
  * *Markdown 1.1*: Nguyên tắc so sánh công bằng: Cùng tập test chính thức (12.630 ảnh), cùng môi trường phần cứng.
  * *Code Cell 1*: Import thư viện (PyTorch, Torchvision, Scipy, Pandas, Matplotlib, Seaborn).
  * *Code Cell 2*: Khởi tạo 2 Test DataLoader: một loader $48 \times 48$ (cho M1, M2) và một loader $224 \times 224$ (cho 3 mô hình M3).
* **Mục 2. Nạp Đồng Thời 5 Checkpoint Đã Huấn Luyện**
  * *Markdown 2.1*: Danh mục 5 mô hình thực nghiệm: `M1LeNet`, `M2VggRes`, `M3-ResNet18`, `M3-MobileNetV2`, `M3-EfficientNet-B0`.
  * *Code Cell 3*: Định nghĩa cấu trúc lớp của cả 3 họ mô hình (tự chứa gọn gàng).
  * *Code Cell 4*: Vòng lặp nạp trọng số từ thư mục `checkpoints/*.pt` vào dictionary `models = {}`.
* **Mục 3. Suy luận Đồng loạt & Đo lường Độ Trễ (Latency)**
  * *Markdown 3.1*: Phương pháp đo độ trễ chuẩn hóa: Warm-up 50 bước, đo median p50 với Batch Size = 1.
  * *Code Cell 5*: Định nghĩa hàm đo thời gian suy luận `benchmark_latency()`.
  * *Code Cell 6*: Chạy suy luận toàn bộ 5 mô hình, thu thập nhãn dự đoán `y_pred`, xác suất `y_prob` và thời gian suy luận.
* **Mục 4. Bảng Tổng Kết Đa Tiêu Chí**
  * *Markdown 4.1*: Trình bày bảng so sánh toàn diện. Giải thích vì sao Macro-F1 là chỉ số quyết định trong khi Top-5 bị bão hòa.
  * *Code Cell 7*: Tạo DataFrame so sánh gồm: Tên mô hình, Số tham số (#Params M), Tính toán (FLOPs G), Top-1 (%), Macro-F1, Độ trễ (ms).
* **Mục 5. Kiểm định Ý Nghĩa Thống Kê McNemar**
  * *Markdown 5.1*: Cơ sở toán học của kiểm định McNemar (dữ liệu bắt cặp trên cùng tập test; vì sao t-test độc lập là sai).
  * *Code Cell 8*: Định nghĩa hàm `mcnemar()`.
  * *Code Cell 9*: Tính toán ma trận kiểm định McNemar giữa các cặp mô hình (đặc biệt: M2 vs MobileNetV2, M2 vs EfficientNet-B0, M2 vs ResNet18).
* **Mục 6. Trực quan hóa Chuyên Sâu**
  * *Markdown 6.1*: Biểu đồ đánh đổi Pareto giữa Độ chính xác và Thời gian suy luận.
  * *Code Cell 10*: Vẽ biểu đồ phân tán Pareto (Macro-F1 vs Latency).
  * *Markdown 6.2*: Phân tích các ca dự đoán sai phổ biến (Confusion Matrix).
  * *Code Cell 11*: Vẽ biểu đồ Heatmap Confusion Matrix cho các cặp lớp dễ nhầm lẫn nhất (ví dụ biển hạn chế tốc độ 30 vs 50 km/h).

---

## 4. Quy Trình Khởi Tạo & Hướng Dẫn Thứ Tự Tự Chạy Kiểm Tra Thủ Công

### 4.1. Quy trình khởi tạo từ trợ lý
1. **Kiểm tra tính sẵn sàng của tài nguyên**:
   * Xác nhận các file checkpoint trong `checkpoints/` (`m1_lenet_best.pt`, `m2_vggres_best.pt`, `m3_resnet18_best.pt`,...).
   * Xác nhận dữ liệu tiền xử lý trong `data/processed/` (`index.csv`, `images_48_clahe.npy`).
2. **Khởi tạo tuần tự 5 file Notebook**:
   * Tạo lần lượt `01_preprocess.ipynb` $\rightarrow$ `02_train_m1.ipynb` $\rightarrow$ `03_train_m2.ipynb` $\rightarrow$ `04_train_m3.ipynb` $\rightarrow$ `05_evaluation_comparison.ipynb` trong thư mục `notebooks/`.
   * Đảm bảo cấu trúc file `.ipynb` đúng định dạng JSON chuẩn (JSON-compliant), không tự ý chạy (run) ngầm để bạn toàn quyền kiểm soát.

---

### 4.2. Thứ tự khuyến nghị khi bạn tự chạy kiểm tra thủ công

Sau khi các file notebook được tạo, bạn hãy mở và chạy kiểm tra thủ công theo đúng thứ tự 5 bước dưới đây:

1. **Bước 1 — Mở và chạy `notebooks/01_preprocess.ipynb`**:
   * **Mục đích**: Xác nhận dữ liệu đầu vào và các tệp cache đã sẵn sàng.
   * **Trạng thái cờ mặc định**: `FORCE_DOWNLOAD = False`, `FORCE_REPROCESS = False`.
   * **Kỳ vọng khi chạy**: Notebook nhận diện được `data/processed/index.csv` và `images_48_clahe.npy` đã có sẵn, in ra thông báo xác nhận `0 track rò rỉ ✓`, hiển thị kích thước mảng ảnh và vẽ lưới 10 ảnh mẫu trực quan trong vài giây (không tốn thời gian xử lý lại).

2. **Bước 2 — Mở và chạy `notebooks/02_train_m1.ipynb`**:
   * **Mục đích**: Kiểm tra mô hình Baseline M1 (LeNet).
   * **Trạng thái cờ mặc định**: `FORCE_RETRAIN = False`.
   * **Kỳ vọng khi chạy**:
     * In bảng đếm tham số chứng minh 97.3% tham số tập trung ở lớp Fully-Connected duy nhất `Linear(9216, 256)`.
     * Tự động nạp checkpoint `checkpoints/m1_lenet_best.pt`.
     * In bảng kết quả Top-1, Macro-F1 trên 12.630 ảnh test chính thức và vẽ biểu đồ phân bố F1-score trên 43 lớp.

3. **Bước 3 — Mở và chạy `notebooks/03_train_m2.ipynb`**:
   * **Mục đích**: Kiểm tra mô hình cải tiến sâu M2 (VGG-ResNet).
   * **Trạng thái cờ mặc định**: `FORCE_RETRAIN = False`.
   * **Kỳ vọng khi chạy**:
     * In cấu trúc 4 stage và bảng so sánh tham số (M2 chỉ 1.24M tham số so với 2.42M của M1 nhờ Global Average Pooling).
     * Nạp checkpoint `checkpoints/m2_vggres_best.pt`.
     * Xuất kết quả đánh giá tập test và biểu đồ loss/metric.

4. **Bước 4 — Mở và chạy `notebooks/04_train_m3.ipynb`**:
   * **Mục đích**: Kiểm tra mô hình Transfer Learning M3.
   * **Trạng thái cờ mặc định**: `BACKBONE = "resnet18"`, `FORCE_RETRAIN = False`.
   * **Kỳ vọng khi chạy**:
     * Tự động upsample $224 \times 224$ và chuẩn hóa theo phân phối ImageNet.
     * Nạp checkpoint `checkpoints/m3_resnet18_best.pt` và in kết quả đánh giá trên tập test.
     * *(Tùy chọn)*: Bạn có thể đổi biến `BACKBONE = "mobilenetv2"` hoặc `"effnetb0"` để kiểm tra các checkpoint còn lại nếu muốn.

5. **Bước 5 — Mở và chạy `notebooks/05_evaluation_comparison.ipynb` (Quan trọng nhất khi báo cáo)**:
   * **Mục đích**: Bảng điều khiển trung tâm nạp đồng loạt 5 mô hình để so sánh toàn diện trước giáo viên.
   * **Kỳ vọng khi chạy**:
     * Nạp cùng lúc 5 checkpoint từ thư mục `checkpoints/`.
     * Đo độ trễ suy luận median p50 thực tế trên phần cứng của bạn.
     * Xuất **Bảng tổng kết đa tiêu chí** (Top-1, Top-5, Macro-F1, FLOPs, Tham số, Latency).
     * Xuất **Ma trận kiểm định thống kê McNemar** (chứng minh M2 vượt trội MobileNetV2/EfficientNet-B0 có ý nghĩa thống kê $p < 10^{-6}$).
     * Vẽ **Đồ thị đánh đổi Pareto** (Accuracy vs Latency) và Heatmap các ca dự đoán nhầm lẫn.

