# Lý thuyết — vì sao chọn như vậy

> **Cách dùng**: đây là sổ tay khái niệm, đọc theo thứ tự. Mỗi khái niệm trình bày theo
> 3 lớp: **trực giác** (hiểu bằng lời) → **công thức** (nói được khi bị hỏi) →
> **trong bài này** (nhóm dùng nó ở đâu, giá trị bao nhiêu).
> Nhãn `[A]/[B]/[C]/[D]` là người chịu trách nhiệm viết phần đó sau khi có số liệu thật.
>
> Ký hiệu: `N` = số ảnh, `K = 43` = số lớp, `C` = số kênh, `H×W` = kích thước không gian.

---

---

## Bản đồ: mở file code nào thì đọc mục nào

Dùng bảng này khi bạn đang đọc một file và muốn biết lý thuyết đằng sau nó.
Chiều ngược lại (đọc lý thuyết rồi muốn xem code) nằm ở dòng `📁 Code:` dưới mỗi mục.

| File code | Đọc mục | Khái niệm chính |
|---|---|---|
| `data/download.py` | 0 | cấu trúc track của GTSRB, cái bẫy torchvision |
| `data/preprocess.py` | 1.2 | histogram equalization, CLAHE, vì sao làm trên kênh L |
| `data/split.py` | **1.1** | ★ rò rỉ dữ liệu, StratifiedGroupKFold |
| `data/transforms.py` | 1.3 | augmentation, vì sao cấm flip ngang và hue |
| `data/dataset.py` | 1.2, 4.2 | cache, chuẩn hoá, nội suy 48→224 |
| `models/m1_lenet.py` | 2.1, 2.6 | conv vs fully-connected, 97,3% tham số ở một lớp FC |
| `models/m2_vggres.py` | **2.2–2.6** | ★ hai conv 3×3, BatchNorm, residual, SpatialDropout, GAP |
| `models/m3_transfer.py` | **4.1–4.6** | ★ transfer learning, discriminative LR, 2 pha freeze |
| `engine/losses.py` | 3.1 | cross-entropy, label smoothing |
| `engine/schedulers.py` | 3.2, 3.3 | Adam vs AdamW, cosine annealing, warmup |
| `engine/callbacks.py` | 3.4 | early stopping theo macro-F1 |
| `engine/train.py` | 3.1–3.4, 4.4 | vòng huấn luyện, AMP, grad clip, hai pha |
| `eval/metrics.py` | **5.1, 5.2** | ★ top-1/top-5, macro-F1 và vì sao nó là chỉ số chính |
| `eval/stats_tests.py` | **5.3** | ★ McNemar — chênh lệch có thật không |
| `eval/calibration.py` | 5.4 | ECE, reliability diagram |
| `eval/confusion.py` | 5.5 | confusion matrix, cặp lớp bị nhầm |
| `eval/ensemble.py` | **8** | TTA, gộp nhiều model |
| `robustness/corruptions.py` | **6.1, 6.2** | ★ 5 loại nhiễu, luật chỉ áp lúc test |
| `robustness/benchmark.py` | 6.3 | relative robustness, mCE |
| `explain/gradcam.py` | **6.6** | ★ Grad-CAM, công thức và giới hạn |
| `deploy/speed.py` | **6.4, 6.5** | ★ FLOPs ≠ latency, đo latency cho đúng |
| `deploy/export.py` | **7** | lượng tử hoá int8, TorchScript, ONNX |
| `scripts/run_seeds.py` | **9** | vì sao một seed là không đủ |

★ = mục quan trọng nhất, đọc kỹ.

---

## Phần 0. Bối cảnh bài toán

GTSRB (German Traffic Sign Recognition Benchmark) — 43 lớp, 39.209 ảnh train,
12.630 ảnh test chính thức, tổng 51.839. Ảnh gốc `.ppm`, kích thước từ 15×15 đến 250×250,
chụp từ camera gắn trên xe nên có mờ chuyển động, ngược sáng, che khuất một phần.

**Điều phải nói ngay khi thuyết trình**: bộ dữ liệu này đã **bão hoà**.
Con người đạt 98,84%. Nhà vô địch IJCNN 2011 (committee of CNNs, IDSIA) đạt 99,46%.
Các phương pháp hiện đại ~99,7%. Nghĩa là **đua accuracy không còn là câu hỏi khoa học**.
Vì vậy nhóm đặt câu hỏi khác, bốn câu:

1. Nếu split dữ liệu đúng cách thì con số thật là bao nhiêu? → *rò rỉ dữ liệu*
2. Mỗi thành phần kiến trúc **thật sự** đóng góp bao nhiêu? → *ablation*
3. Model giữ được bao nhiêu phần trăm khi gặp mưa, sương, tối, mờ? → *robustness*
4. Với ngân sách latency của một thiết bị trên xe thì model nào thắng? → *Pareto*

---

## Phần 1. Dữ liệu  `Huy`

### 1.1 Rò rỉ dữ liệu do cấu trúc track — khái niệm quan trọng nhất của phần dữ liệu

> 📁 **Code:** `src/gtsrb/data/split.py` → `make_split()`, `check_leakage()` · test: `tests/test_split.py`

**Trực giác.** GTSRB không chụp mỗi biển báo một lần. Xe chạy tới gần một tấm biển và
camera quay **30 frame liên tiếp** của *cùng tấm biển vật lý đó*, lưu thành một "track".
Tên file `000{track}_000{frame}.ppm` mã hoá điều này. Frame 12 và frame 13 của cùng track
gần như là hai bản sao của nhau.

Nếu ta trộn tất cả ảnh rồi cắt 80/20 ngẫu nhiên, thì frame 12 vào train và frame 13 vào val.
Model không cần **khái quát** sang biển báo mới — nó chỉ cần **nhớ** tấm biển đã thấy.
Val accuracy đẹp, nhưng con số đó không nói gì về việc model hoạt động thế nào trên một
tấm biển nó chưa từng gặp. Đó là **rò rỉ dữ liệu** (data leakage).

**Công thức/kỹ thuật.** Group-aware splitting: định nghĩa nhóm `g = (class_id, track_id)`,
rồi dùng `StratifiedGroupKFold` để (a) mọi phần tử cùng một `g` chỉ nằm ở một phía, và
(b) tỉ lệ 43 lớp giữ gần như nhau hai phía.

**Trong bài này.** Nhóm chạy **cả hai** cách để định lượng phần "ảo":

| Cách split | Val top-1 | Val macro-F1 | Test top-1 (chính thức) |
|---|---|---|---|
| Random theo ảnh (sai) | `[A điền]` | `[A điền]` | `[A điền]` |
| Theo track (đúng) | `[A điền]` | `[A điền]` | `[A điền]` |

Dấu hiệu nhận biết có rò rỉ: **val cao hơn test** một cách bất thường. Split đúng thì
val và test phải gần nhau, vì cả hai đều là "biển báo chưa từng thấy".

### 1.2 Histogram equalization và CLAHE

> 📁 **Code:** `src/gtsrb/data/preprocess.py` → `apply_preprocess()`, `crop_roi()`, `resize_image()`

**Trực giác.** Ảnh chụp ngoài đường có cái tối thui, cái cháy sáng. Hai ảnh cùng một biển
báo nhưng độ sáng khác nhau thì với model là hai đầu vào rất khác. Equalization kéo
phân phối độ sáng về gần đều → các ảnh "trông giống nhau hơn" ở mức thống kê.

**Công thức.** Với histogram `h(r)` và CDF `F(r) = Σ_{k≤r} h(k)/N`:
```
s = round( (L−1) · F(r) )          L = 256
```
Hàm map `s = f(r)` đúng là CDF được scale — đó là lý do kết quả có histogram gần phẳng.

**Giới hạn và CLAHE.** HE toàn cục dùng **một** CDF cho cả ảnh. Vùng gần như phẳng (trời,
mặt đường) có histogram rất hẹp, bị kéo giãn cực mạnh → **nhiễu bị phóng đại**.
CLAHE (Contrast Limited Adaptive HE) sửa hai chỗ:
- *Adaptive*: chia ảnh thành ô 8×8, equalize **từng ô** → thích ứng theo vùng.
- *Contrast Limited*: chặn trần histogram ở `clipLimit` (nhóm dùng 2,0), phần vượt trần
  được **phân phối lại đều** cho các bin khác → không khuếch đại nhiễu vô hạn.
- Cuối cùng nội suy song tuyến tính giữa các ô để không xuất hiện đường ranh ô vuông.

**Vì sao làm trên kênh L (LAB) hoặc Y (YUV), không trên R, G, B.**
Equalize R, G, B độc lập làm lệch tỉ lệ giữa ba kênh → **ảnh đổi màu**. Mà màu biển báo
mang nghĩa: đỏ = cấm, xanh = bắt buộc, vàng/trắng = cảnh báo. LAB/YUV tách độ sáng (L/Y)
khỏi màu (A,B / U,V) → sửa độ sáng, giữ nguyên màu.

**Trong bài này.** 4 chế độ `none / he_gray / he_y / clahe`, mặc định `clahe`, có ablation.

### 1.3 Augmentation — và hai thứ bị cấm

> 📁 **Code:** `src/gtsrb/data/transforms.py` → `build_transform()` · test: `tests/test_transforms.py`

**Trực giác.** Augmentation dạy model những biến đổi mà **nhãn không đổi**. Xoay biển báo
15° thì nó vẫn là biển báo đó → model nên bất biến với xoay nhỏ. Đó là cách đưa
*tri thức tiên nghiệm* (prior) vào mà không cần thêm dữ liệu.

**Trong bài này.** `rotation ±15°`, `translate ±10%`, `zoom 0,9–1,1`,
`brightness/contrast jitter ±0,2`. Chỉ áp dụng cho train loader.

**Hai thứ bị cấm — và đây là câu hỏi hay của giảng viên:**

| Phép biến đổi | Vì sao cấm |
|---|---|
| `RandomHorizontalFlip` | Biển báo **bất đối xứng có nghĩa**. Lớp 33 "rẽ phải phía trước" lật ngang ra đúng hình lớp 34 "rẽ trái phía trước". Các cặp bị ảnh hưởng: **33↔34, 19↔20, 36↔37, 38↔39**. Flip = tự tạo dữ liệu **sai nhãn**, model sẽ học rằng rẽ phải và rẽ trái là một. |
| Hue jitter | Màu là **đặc trưng mang nghĩa**, không phải nhiễu. Đổi đỏ thành xanh là đổi "cấm" thành "bắt buộc". Brightness/contrast thì được, vì đó đúng là biến thiên có thật của điều kiện chụp. |

Bài học khái quát: augmentation **phải tôn trọng tính bất biến thật của bài toán**.
Dùng mù quáng "bộ augment tiêu chuẩn cho ảnh" là cách tự làm hỏng nhãn.

### 1.4 Mất cân bằng lớp

> 📁 **Code:** `src/gtsrb/eval/metrics.py` → `metrics_from_probs()` (macro-F1 là cách nhóm xử lý)

Lớp 2 ("giới hạn 50 km/h") có 2.250 ảnh; lớp 0 ("giới hạn 20"), 19, 37 chỉ có 210.
Tỉ lệ ~10,7:1. Nhóm **không** resample, vì: (a) phân phối này phản ánh tần suất thật trên
đường Đức; (b) nhóm đổi **chỉ số** sang macro-F1 thay vì cân bằng lại **dữ liệu** —
can thiệp ở chỗ đúng hơn; (c) vẫn stratify theo lớp khi split. Hướng mở rộng nếu còn thời
gian: `class_weight` trong cross-entropy, hoặc focal loss.

---

## Phần 2. Kiến trúc mạng từ số 0  `Hoàng`

### 2.1 Vì sao conv thay vì fully-connected

> 📁 **Code:** `src/gtsrb/models/m1_lenet.py` → `M1LeNet.__init__()`

Ảnh 48×48×3 = 6.912 giá trị. Một lớp FC 1.000 neuron là ~6,9 triệu tham số **chỉ cho một lớp**.
Conv giải quyết bằng hai ý tưởng: **kết nối cục bộ** (mỗi neuron chỉ nhìn một ô 3×3 —
vì pixel liên quan tới pixel gần nó) và **chia sẻ tham số** (cùng một kernel trượt khắp ảnh —
vì "cạnh" là "cạnh" bất kể nó nằm ở đâu). Kết quả: một kernel 3×3 cho 32 kênh đầu vào,
64 kênh đầu ra chỉ cần `3·3·32·64 = 18.432` tham số, và nó **bất biến với dịch chuyển**.

### 2.2 Hai conv 3×3 thay một conv 5×5 — luận điểm của VGG

> 📁 **Code:** `src/gtsrb/models/m2_vggres.py` → `M2VggRes._build_block()`

**Receptive field.** Conv 3×3 thứ nhất: mỗi output nhìn 3×3 input. Conv 3×3 thứ hai:
mỗi output nhìn 3×3 của tầng trước, mà mỗi cái đó lại nhìn 3×3 → tổng cộng **5×5** input.
Công thức tổng quát cho `n` lớp 3×3 stride 1: `RF = 2n + 1`.

**Tham số.** `2 · (3·3·C²) = 18C²` so với `5·5·C² = 25C²` → **giảm 28%**.
Với ba lớp 3×3 (RF 7×7) thì `27C²` so với `49C²` → giảm 45%.

**Phi tuyến.** Đây mới là điểm then chốt: 2 lớp có **2** hàm ReLU thay vì 1. Cùng một
receptive field nhưng hàm biểu diễn được phong phú hơn nhiều. Một stack conv 3×3 không
phải là "cách rẻ để làm conv 5×5", nó là **một hàm mạnh hơn**.

### 2.3 Batch Normalization

> 📁 **Code:** `src/gtsrb/models/m2_vggres.py` → `ResidualBlock.__init__()` (thứ tự Conv→BN→ReLU)

**Công thức** (cho mỗi kênh, trên một mini-batch `B`):
```
μ_B = (1/|B|) Σ x_i                σ²_B = (1/|B|) Σ (x_i − μ_B)²
x̂_i = (x_i − μ_B) / √(σ²_B + ε)
y_i = γ · x̂_i + β                 γ, β là tham số HỌC ĐƯỢC
```
`γ, β` cần thiết vì nếu luôn ép về mean 0 / var 1 thì mạng mất khả năng dùng vùng
phi tuyến của ReLU — `γ, β` cho mạng tự chọn mức chuẩn hoá nó muốn.

**Lúc suy luận** không có mini-batch, nên dùng `running_mean/running_var` tích luỹ trong
lúc train (đó chính là lý do phải gọi `model.eval()` — quên gọi là kết quả sai mà không báo lỗi).

**Tác dụng thực tế.** Bài gốc (Ioffe & Szegedy 2015) giải thích bằng "giảm internal covariate
shift"; **Santurkar et al. 2018 phản biện** và cho thấy tác dụng chính là **làm mượt bề mặt
loss**, nhờ đó dùng được learning rate lớn hơn. Nói được chỗ phản biện này là điểm cộng.
Thêm: BN có hiệu ứng regularize nhẹ (thống kê batch là nhiễu ngẫu nhiên), và giảm phụ thuộc
vào cách khởi tạo trọng số.

**Thứ tự.** Nhóm dùng `Conv → BN → ReLU` theo bài gốc. (Có trường phái
`BN → ReLU → Conv` kiểu pre-activation ResNet-v2; nhóm không dùng để giữ so sánh đơn giản.)

### 2.4 Residual connection

> 📁 **Code:** `src/gtsrb/models/m2_vggres.py` → `ResidualBlock.forward()` (cộng TRƯỚC, ReLU SAU)

**Vấn đề nó giải quyết — degradation, không phải overfitting.** He et al. 2015 quan sát:
mạng 56 lớp cho **train error** cao hơn mạng 20 lớp. Train error cao thì không thể là
overfitting — đó là **không tối ưu hoá được**. Nghịch lý: mạng sâu hơn luôn có thể bắt chước
mạng nông hơn bằng cách cho các lớp thừa thành identity, vậy mà SGD không tìm ra lời giải đó.

**Giải pháp.**
```
y = F(x) + x           thay vì      y = H(x)
```
Nếu identity là tối ưu, mạng chỉ cần đẩy `F(x) → 0` — việc rất dễ (đẩy trọng số về 0).
Học identity bằng một stack conv thì khó.

**Góc nhìn gradient.** `∂y/∂x = ∂F(x)/∂x + 1`. Số hạng `+1` nghĩa là gradient luôn có một
đường đi về không bị suy giảm, bất kể `∂F/∂x` nhỏ cỡ nào → chống vanishing gradient.
Đây là lý do người ta gọi nó là "đường cao tốc gradient".

**Khi shape không khớp.** Đổi từ 32 sang 64 kênh thì không cộng được. Dùng **conv 1×1**
(projection shortcut — "option B" trong bài ResNet) để chiếu nhánh skip về đúng số kênh,
kèm stride nếu cần khớp kích thước không gian.

### 2.5 Dropout và SpatialDropout

> 📁 **Code:** `src/gtsrb/models/m2_vggres.py` → `M2VggRes._build_stages()` (`nn.Dropout2d`)

**Dropout thường** (Srivastava 2014): lúc train, mỗi activation bị zero với xác suất `p`
độc lập; các activation còn lại được scale `1/(1−p)` (inverted dropout) để kỳ vọng không đổi.
Lúc test tắt hoàn toàn. Trực giác: mạng không được dựa vào một neuron cụ thể nào →
buộc học biểu diễn dư thừa. Một góc nhìn khác: nó xấp xỉ việc lấy trung bình của `2^n`
mạng con.

**Vì sao dropout thường yếu trên feature map của conv.** Pixel lân cận trong một feature map
**tương quan rất cao** (cùng nhìn gần như cùng một vùng ảnh). Bỏ pixel `(i,j)` thì thông tin
hầu như vẫn còn nguyên ở `(i,j+1)`. Việc "bỏ" gần như không gây mất mát → regularize kém hiệu lực.

**SpatialDropout / `nn.Dropout2d`**: bỏ **toàn bộ một kênh** (`C` chiều bị zero cả `H×W`).
Một kênh = một feature detector. Bỏ nó là buộc mạng không được phụ thuộc vào detector đó.
Đây mới là đơn vị regularize có nghĩa trên conv.

**Trong bài này.** `p` tăng dần theo độ sâu: 0,1 → 0,2 → 0,3 → 0,3. Lý do: tầng sâu có nhiều
kênh hơn, biểu diễn trừu tượng hơn và dễ overfit hơn. `Dropout(0,5)` thường ở ngay trước FC cuối.

### 2.6 Global Average Pooling

> 📁 **Code:** `src/gtsrb/models/m2_vggres.py` → `M2VggRes._build_head()` · so với `m1_lenet.py` `classifier`

M1 có `Flatten(9216) → FC(256)` = **2.359.552 tham số** ở một lớp duy nhất, tức **97,3%**
toàn bộ 2.424.299 tham số của model (số đo thật). GAP (Lin et al. 2013, Network-in-Network) thay bằng: lấy trung bình không gian
mỗi kênh → feature map `256×3×3` thành vector `256` → `FC(256→43)` chỉ còn **11.051** tham số.
Giảm ~213 lần. Thêm hai lợi ích: bất biến với dịch chuyển, và hoạt động như một
**regularizer cấu trúc** (buộc mỗi kênh mang nghĩa toàn cục, không được nhớ vị trí cụ thể).
Đây là lý do mọi kiến trúc hiện đại (ResNet, MobileNet, EfficientNet) đều dùng GAP.

**Bằng chứng từ chính dự án này** (số đo thật, `reports/tables/*_params.csv`):

| | Số lớp conv | Tổng tham số | Lớp "đắt" nhất |
|---|---|---|---|
| **M1** (Flatten + FC) | 2 | **2.424.299** | `classifier.1` Linear — 2.359.552 = **97,3%** |
| **M2** (GAP) | 9 | **1.237.451** | `stages.3.0.conv2` Conv2d — 589.824 = 47,7% |

M2 **sâu gấp hơn 4 lần mà chỉ bằng ~51% số tham số của M1**. Đây là câu trả lời
định lượng cho "số lớp không tỉ lệ với số tham số — chỗ đắt là lớp fully-connected".

---

## Phần 3. Huấn luyện  `Hoàng`

### 3.1 Cross-entropy và label smoothing

> 📁 **Code:** `src/gtsrb/engine/losses.py` → `build_criterion()`

**Cross-entropy.** `L = −Σ_k y_k · log p_k`. Với nhãn one-hot thì gọn lại thành
`L = −log p_c` (c là lớp đúng). Để `L → 0` thì cần `p_c → 1`, mà softmax chỉ đạt được
điều đó khi logit của lớp đúng **ra vô cực**. Tức là one-hot *luôn* đẩy model về phía
tự tin cực đoan, kể cả khi ảnh thật sự mơ hồ.

**Label smoothing** (Szegedy 2016):
```
y'_k = (1 − ε) · y_k + ε/K
```
Với `K = 43`, `ε = 0,1`: lớp đúng nhận `0,9 + 0,1/43 ≈ 0,9023`; mỗi lớp sai nhận
`0,1/43 ≈ 0,00233`. Giờ loss tối thiểu đạt ở một mức tự tin **hữu hạn** — có một trần.

**Hệ quả kỳ vọng.** Biên quyết định đều hơn, bền hơn một chút trước nhiễu, và —
theo lý thuyết thông thường — độ hiệu chỉnh tốt hơn. **Đánh đổi**: top-1 thô có thể
giảm nhẹ, và biểu diễn tầng cuối bị "co lại".

### ★ Kết quả ĐO THẬT trên GTSRB — và nó NGƯỢC với lý thuyết thông thường

M1 với `label_smoothing=0,1` cho:

| | giá trị |
|---|---|
| accuracy thật trên test | **0,9846** |
| độ tự tin trung bình của model | **0,8535** |
| ECE | **0,1310** |

`conf (0,854) < acc (0,985)` → model **THIẾU TỰ TIN** (underconfident), không phải
tự tin thái quá. Và ECE = 0,131 là **cao**, tức là hiệu chỉnh **kém**.

**Vì sao.** Câu chuyện "mạng sâu luôn tự tin thái quá, label smoothing sửa được"
(Guo 2017) đúng trên ImageNet — bài toán **khó**, accuracy ~76%. GTSRB **quá dễ**:
accuracy 98,5%. Mức tự tin mà label smoothing kéo model về (≈ `1−ε+ε/K` = 0,9023)
**thấp hơn** accuracy thật. Nên label smoothing ở đây **sửa quá tay**: nó kéo tự tin
xuống dưới mức đúng.

Nhìn theo từng bin thì mọi bin đều `acc > conf`, và lệch nặng nhất ở vùng tự tin thấp:
ở bin conf ≈ 0,43 thì accuracy thực tế là **0,90**. Nghĩa là khi model nói "tôi chỉ
43% chắc", nó vẫn đúng 9 trên 10 lần.

**Hệ quả thực tế, trả lời trực tiếp cho lập luận "đặt ngưỡng chuyển cho người lái":**
với model này, ngưỡng 0,8 sẽ **loại bỏ rất nhiều dự đoán đúng**. Muốn đặt ngưỡng có
nghĩa thì phải hiệu chỉnh lại (temperature scaling) hoặc giảm label smoothing.

**Một điểm chính xác về công thức.** Nói label smoothing "đặt một **trần** tự tin" là
nói **chưa đúng hẳn**. Nó làm **điểm cực tiểu của loss** nằm ở mức tự tin hữu hạn
(≈ 0,9023), chứ không chặn cứng đầu ra softmax. Thực nghiệm: độ tự tin cao nhất model
từng báo là **0,9965 > 0,9023**. Mạng không hội tụ chính xác vào cực tiểu của loss
(còn early stopping, dropout, weight decay), nên các ảnh dễ vẫn vượt mức đó được.

**Trục ablation 5** (ε = 0,0 / 0,1 / 0,2) sẽ định lượng đánh đổi này. Giả thuyết của
nhóm sau khi thấy số liệu trên: **ε = 0,0 sẽ cho ECE TỐT HƠN trên GTSRB**, ngược với
kỳ vọng từ bài báo. Nếu đúng, đó là một kết quả đáng báo cáo.

### 3.2 Adam

> 📁 **Code:** `src/gtsrb/engine/schedulers.py` → `build_optimizer()`

Adam giữ hai ước lượng trung bình động: moment 1 (`m_t`, hướng) và moment 2 (`v_t`, độ lớn
bình phương), rồi bước theo `m̂_t / (√v̂_t + ε)`. Hiệu quả thực tế: mỗi tham số có một
learning rate thích ứng riêng → ít phải tinh chỉnh LR, hội tụ nhanh ở giai đoạn đầu.
`m̂, v̂` là phiên bản đã hiệu chỉnh bias (vì `m_0 = v_0 = 0` nên những bước đầu bị lệch về 0).

Ghi chú cho M3: nhóm dùng **AdamW** — weight decay tách khỏi gradient adaptive.
Trong Adam gốc, weight decay bị chia cho `√v̂` nên tham số có gradient lớn bị decay ít hơn,
không đúng ý nghĩa của regularization.

### 3.3 Cosine annealing + warmup

> 📁 **Code:** `src/gtsrb/engine/schedulers.py` → `build_scheduler()` → hàm `lr_lambda()` bên trong

```
lr_t = lr_min + ½ (lr_max − lr_min) · (1 + cos(π t / T))
```
Đầu kỳ LR lớn → khám phá rộng, thoát khỏi vùng kém. Cuối kỳ LR → gần 0 → lắng vào đáy.
So với step decay: không phải đoán "giảm ở epoch nào", và không có bước nhảy làm loss giật.

**Warmup 3 epoch** (LR tăng tuyến tính từ 0): ở những bước đầu, `v̂_t` của Adam được ước lượng
từ rất ít mẫu nên rất nhiễu → bước đi có thể rất lớn và sai hướng. Warmup tránh chuyện đó.
Với M3 (fine-tune trọng số pretrained) thì warmup còn quan trọng hơn: một bước lớn sai
hướng ở đầu có thể phá tri thức pretrained.

### 3.4 Early stopping theo macro-F1

> 📁 **Code:** `src/gtsrb/engine/callbacks.py` → lớp `EarlyStopping`

Theo dõi **val macro-F1**, patience 10 epoch, lưu checkpoint tốt nhất.
**Không** theo accuracy — vì mất cân bằng 10,7:1, accuracy có thể gần như không đổi trong
khi một lớp thưa sập hoàn toàn. Nguyên tắc: *chọn checkpoint theo đúng chỉ số mình thật sự
quan tâm*. Và tuyệt đối không theo chỉ số trên tập test.

---

## Phần 4. Transfer learning  `Phong Trần`

### 4.1 Vì sao tri thức ImageNet dùng được cho biển báo

> 📁 **Code:** `src/gtsrb/models/m3_transfer.py` → `M3Transfer.__init__()` (thay head)

CNN học đặc trưng **có phân tầng**:

| Tầng | Học được gì | Có dùng lại được cho GTSRB? |
|---|---|---|
| Đầu (conv1, layer1) | cạnh, góc, đốm màu, gradient | **Có** — phổ quát cho mọi ảnh tự nhiên |
| Giữa (layer2, layer3) | texture, hoa văn, hình khối, bộ phận | **Phần lớn** — hình tròn/tam giác/mũi tên |
| Cuối (layer4, fc) | ngữ nghĩa "mèo", "xe tải" | **Không** — phải thay |

ImageNet có 1,28 triệu ảnh. GTSRB có 39 nghìn. Tri thức tầng đầu học từ 1,28 triệu ảnh là
thứ 39 nghìn ảnh **không học nổi**. Đó là nội dung thực sự được "chuyển giao".

### 4.2 Vì sao phải upsample lên 224×224

> 📁 **Code:** `src/gtsrb/data/dataset.py` → `GTSRBDataset.__getitem__()` (nội suy) · `explain/gradcam.py` → `feature_map_size()`

ResNet18, MobileNetV2, EfficientNet-B0 đều downsample tổng cộng **32 lần**
(conv stride 2 + maxpool + các stage stride 2).

| Input | Feature map cuối | Dùng được không? |
|---|---|---|
| 48×48 | `48/32 = 1,5` → 2×2 | Không. Mất hết cấu trúc không gian, `layer4` vô nghĩa, Grad-CAM thành một ô |
| 112×112 | 4×4 | Tạm được |
| **224×224** | **7×7** | Đúng cấu hình mà trọng số pretrained được học |

Tốn ~22 lần số pixel so với 48×48 — đó là **cái giá** của transfer learning, và chính là
nội dung của ablation resolution.

### 4.3 Discriminative learning rate

> 📁 **Code:** `src/gtsrb/models/m3_transfer.py` → `M3Transfer.param_groups()`

```python
optimizer = AdamW([
    {"params": early,  "lr": 1e-5},   # cạnh/màu — đã tốt sẵn, chỉ nhích nhẹ
    {"params": middle, "lr": 1e-4},   # hoa văn — điều chỉnh vừa
    {"params": head,   "lr": 1e-3},   # ngữ nghĩa — đổi hoàn toàn
])
```
Nếu cho toàn mạng LR 1e-3 thì gradient lớn sẽ ghi đè lên tri thức pretrained đắt giá bằng
tín hiệu từ 39 nghìn ảnh — gọi là **catastrophic forgetting**. Nếu cho toàn mạng 1e-5 thì
head (khởi tạo ngẫu nhiên) học quá chậm, không bao giờ hội tụ trong ngân sách epoch.
Hai cực đều tệ; discriminative LR là lời giải đúng.

### 4.4 Hai pha freeze → unfreeze

> 📁 **Code:** `src/gtsrb/models/m3_transfer.py` → `set_backbone_frozen()` · `engine/train.py` → `_run_one_phase()`

*Pha 1 (3 epoch)*: `requires_grad = False` cho backbone, chỉ train head.
Lý do: head khởi tạo ngẫu nhiên nên loss ban đầu rất lớn (`≈ ln 43 ≈ 3,76`) → gradient rất
lớn. Nếu backbone đang mở, gradient đó chảy ngược và phá filter pretrained ngay trong vài
trăm bước đầu, trước khi LR schedule kịp làm gì.

*Pha 2*: mở băng toàn bộ, dùng discriminative LR.

### 4.5 Chuẩn hoá bằng thống kê ImageNet

> 📁 **Code:** `src/gtsrb/__init__.py` → `IMAGENET_MEAN/STD` · `data/dataset.py` → `_setup_normalizer()`

M3 dùng `mean = (0,485, 0,456, 0,406)`, `std = (0,229, 0,224, 0,225)` — của ImageNet,
**không** của GTSRB. Hai lý do: (1) filter tầng đầu được tối ưu cho đúng phân phối đầu vào đó;
(2) `running_mean/running_var` trong các lớp BatchNorm pretrained được tích luỹ trên phân phối
đó — đổi phân phối đầu vào là làm chúng sai. Ngược lại, M1 và M2 train từ số 0 thì dùng
thống kê GTSRB tính **chỉ trên tập train**.

### 4.6 Ba backbone — ba ý tưởng kiến trúc khác nhau

> 📁 **Code:** `src/gtsrb/models/m3_transfer.py` → hằng `BACKBONES`

| Backbone | #params | FLOPs @224 | Ý tưởng cốt lõi |
|---|---|---|---|
| ResNet18 | 11,7M | ~1,82 G | residual + basic block; conv dày đặc, đơn giản, nhanh trên GPU |
| MobileNetV2 | 3,5M | ~0,30 G | **depthwise separable conv** (tách conv không gian và conv kênh) + **inverted residual** (nở ra rồi bóp lại) + **linear bottleneck** (bỏ ReLU ở lớp hẹp để không mất thông tin) |
| EfficientNet-B0 | 5,3M | ~0,39 G | **compound scaling** (tăng depth/width/resolution cùng lúc theo một hệ số) + MBConv + **squeeze-and-excitation** (attention trên trục kênh) |

So sánh ba cái này làm nổi bật một bài học: **ít FLOPs ≠ nhanh hơn**. Xem Phần 6.

---

## Phần 5. Đánh giá  `Phong Nguyễn`

### 5.1 Top-1, Top-5 — và tại sao top-5 yếu ở đây

> 📁 **Code:** `src/gtsrb/eval/metrics.py` → `topk_accuracy()`

Top-1: dự đoán có xác suất cao nhất đúng. Top-5: nhãn đúng nằm trong 5 ứng viên cao nhất.

Top-5 ra đời cho **ImageNet 1000 lớp**, nơi nhiều lớp mơ hồ một cách chính đáng
(nhiều giống chó rất giống nhau) và một ảnh có thể chứa nhiều đối tượng.
Trên **43 lớp**, top-5 nghĩa là "đúng trong 11,6% số lớp" — mọi model tốt đều ~99,9%.
**Chỉ số bão hoà thì không phân biệt được gì.** Nhóm vẫn báo cáo vì đề bài yêu cầu, nhưng
nêu rõ giới hạn này và kết luận dựa trên top-1 + macro-F1.

### 5.2 Macro-F1 — chỉ số chính

> 📁 **Code:** `src/gtsrb/eval/metrics.py` → `metrics_from_probs()` (chú ý tham số `labels=`)

```
Precision_k = TP_k / (TP_k + FP_k)        Recall_k = TP_k / (TP_k + FN_k)
F1_k = 2 · P_k · R_k / (P_k + R_k)        macro-F1 = (1/K) Σ_k F1_k
```
Điểm then chốt: `(1/K) Σ` — **mỗi lớp một phiếu bằng nhau**, bất kể lớp đó có 210 hay 2.250 ảnh.

So sánh bằng số thật. Lớp 0 là lớp nhỏ nhất: **210 ảnh train**, **60 ảnh test**. Giả sử
model đoán sai TOÀN BỘ lớp này:

| | công thức | mất bao nhiêu |
|---|---|---|
| accuracy | 60 / 12.630 | **0,48 điểm** — gần như không thấy |
| macro-F1 | 1 / 43 | **2,33 điểm** — thấy rõ |

Chênh **4,9 lần**. Chú ý mẫu số: accuracy chia cho **số ảnh test** (nên lớp nhỏ gần như
không ảnh hưởng), còn macro-F1 chia cho **số lớp** (nên mọi lớp nặng như nhau, dù 60 ảnh
hay 750 ảnh). Đừng lấy tỉ lệ 210/39.209 = 0,54% của tập **train** làm mức giảm accuracy
trên tập **test** — hai mẫu số khác nhau.
Trong bài biển báo, nhận sai một biển hiếm có thể nguy hiểm hơn nhận sai một biển phổ biến.

(Phân biệt: **micro**-F1 gộp hết TP/FP/FN rồi mới tính → trong bài đa lớp đơn nhãn nó
bằng đúng accuracy. **Weighted**-F1 lấy trung bình có trọng số theo support → lại bị lớp đông
chi phối. Nhóm báo cáo cả ba nhưng kết luận theo macro.)

### 5.3 McNemar test — "chênh lệch này có thật không?"

> 📁 **Code:** `src/gtsrb/eval/stats_tests.py` → `mcnemar()` · test: `tests/test_metrics.py`

Hai model chạy trên **cùng** tập test → quan sát **bắt cặp**, không độc lập → dùng t-test
hai mẫu độc lập là **sai về mặt thống kê**. McNemar là test đúng cho trường hợp này.

|  | B đúng | B sai |
|---|---|---|
| **A đúng** | n₀₀ | n₀₁ |
| **A sai** | n₁₀ | n₁₁ |

`n₀₀` và `n₁₁` là những ảnh hai model đồng ý — chúng **không mang thông tin** về sự khác biệt.
Chỉ hai ô bất đồng mới nói được điều gì. Giả thuyết không: `n₀₁` và `n₁₀` có cùng kỳ vọng.
```
χ² = (|n₀₁ − n₁₀| − 1)² / (n₀₁ + n₁₀)        (có hiệu chỉnh liên tục Yates), df = 1
```
Khi `n₀₁ + n₁₀ < 25` thì xấp xỉ χ² không đáng tin → dùng **binomial chính xác**.

Vì sao bài này cần nó: trên 12.630 ảnh test, chênh 0,2 điểm top-1 ≈ **25 ảnh**.
Nếu `p > 0,05` thì phải nói thẳng "không có ý nghĩa thống kê", và khi đó lựa chọn triển khai
chuyển sang dựa vào tốc độ và robustness — đó là một kết luận **mạnh hơn**, không phải yếu hơn.

### 5.4 Expected Calibration Error

> 📁 **Code:** `src/gtsrb/eval/calibration.py` → `expected_calibration_error()`, `plot_reliability()`

```
ECE = Σ_{m=1}^{M} (|B_m| / n) · | acc(B_m) − conf(B_m) |
```
Chia các dự đoán vào `M` bin theo độ tự tin (nhóm dùng 15), trong mỗi bin so sánh accuracy
thực tế với độ tự tin trung bình. Nó trả lời: *"khi model nói 90% chắc thì nó đúng 90% số
lần không?"*

Mạng sâu **thường** tự tin thái quá (Guo et al. 2017): `conf ≫ acc`, nguyên nhân là
cross-entropy + one-hot (xem 3.1).

**Nhưng trên GTSRB nhóm đo được điều NGƯỢC LẠI** — xem bảng ở mục 3.1:
`conf = 0,854 < acc = 0,985`, model **thiếu tự tin**, vì label smoothing sửa quá tay
trên một bài toán quá dễ. Đây là lý do phải **đo** chứ không áp dụng kết luận của bài
báo lên bài toán của mình.

Vì sao hiệu chỉnh quan trọng trong hệ thống thật: nếu độ tự tin đáng tin, ta đặt được
ngưỡng *"dưới 0,8 thì không tự quyết, chuyển cho người lái"*. Với model **tự tin thái quá**
thì ngưỡng đó vô nghĩa vì cả dự đoán sai cũng báo 0,99. Với model **thiếu tự tin** như
của nhóm thì ngưỡng đó cũng vô nghĩa, nhưng theo chiều khác: nó loại bỏ rất nhiều dự
đoán đúng. Cả hai chiều đều hỏng — phải hiệu chỉnh lại trước khi dùng ngưỡng.

### 5.5 Phân tích per-class và confusion matrix

> 📁 **Code:** `src/gtsrb/eval/confusion.py` → `confusion()`, `top_confusions()`

Accuracy là **một** số. Confusion matrix 43×43 cho biết lỗi **đi đâu**. Hai bức tranh rất
khác nhau: lỗi rải đều trên mọi lớp, hay lỗi tập trung vào vài **cặp**.

### Số ĐO THẬT trên M1 (test chính thức, 12.630 ảnh)

10 lớp F1 thấp nhất:

| Lớp | Tên | F1 | support |
|---|---|---|---|
| 27 | Người đi bộ | 0,904 | 60 |
| 6 | Hết giới hạn 80 km/h | 0,905 | 150 |
| 42 | Hết cấm vượt (xe > 3,5 t) | 0,911 | 90 |
| 30 | Cẩn thận băng/tuyết | 0,930 | 150 |
| 41 | Hết cấm vượt | 0,942 | 60 |
| 22 | Đường xấu | 0,947 | 120 |
| 26 | Đèn tín hiệu | 0,951 | 180 |
| 40 | Vòng xuyến bắt buộc | 0,951 | 90 |
| 29 | Xe đạp qua đường | 0,957 | 90 |
| 5 | Giới hạn 80 km/h | 0,965 | 630 |

5 cặp bị nhầm nhiều nhất:

| Lớp thật | → đoán thành | Số lần |
|---|---|---|
| 17 Cấm đi vào | 9 Cấm vượt | 16 |
| 6 Hết giới hạn 80 | 42 Hết cấm vượt (xe > 3,5 t) | 14 |
| 3 Giới hạn 60 | 5 Giới hạn 80 | 14 |
| 8 Giới hạn 120 | 5 Giới hạn 80 | 11 |
| 22 Đường xấu | 25 Đang thi công | 10 |

**Phân tích — ba nguyên nhân, không phải một:**

1. **Nhóm biển "HẾT hiệu lực" (6, 41, 42)** chiếm 3 trong 5 lớp yếu nhất, và cặp
   nhầm nhiều thứ hai là 6 → 42. Đây là các biển **xám có gạch chéo**, khác nhau
   rất ít về hình. Nhóm **không dự đoán trước** điều này — nó chỉ lộ ra khi đo.
2. **Biển giới hạn tốc độ nhầm lẫn nhau** (3→5, 8→5) — đúng như dự đoán: cùng hình
   tròn viền đỏ, chỉ khác chữ số, mà ở 48×48 chữ số chỉ còn vài pixel.
   → cách sửa là **tăng độ phân giải**, không phải làm model to hơn.
3. **Thiếu dữ liệu** là nguyên nhân **yếu hơn tưởng**: lớp 5 có 630 ảnh test (nhiều
   thứ 3) mà vẫn nằm trong top 10 yếu nhất; còn nhiều lớp chỉ 60 ảnh lại F1 = 1,0.

**Dự đoán SAI của nhóm:** nhóm 19/20/21 (đường cong nguy hiểm) **không** xuất hiện
trong top 10 yếu nhất. Nói ra chỗ mình dự đoán sai là một điểm cộng khi bảo vệ.

Cách tách hai nguyên nhân: vẽ F1 từng lớp theo số ảnh train. Nếu lớp ít ảnh mà F1 vẫn cao →
không phải do thiếu dữ liệu. Nếu lỗi tập trung ở các cặp hình giống nhau → cách sửa là
**tăng độ phân giải**, không phải làm model to hơn. Đây là kiểu lập luận mà giảng viên muốn thấy.

---

## Phần 6. Robustness và triển khai biên  `Phong Trần` + `Huy`

### 6.1 Robustness nghĩa là gì — và luật sắt

> 📁 **Code:** `src/gtsrb/robustness/corruptions.py` → `apply_corruption()` · test: `tests/test_corruptions.py`

Robustness = giữ được hiệu năng trên **phân phối chưa từng thấy lúc train**
(distribution shift). Suy ra **luật sắt**: các loại nhiễu dùng để đo **tuyệt đối không**
được đưa vào train. Train trên chúng rồi test trên chúng là test in-distribution, chỉ
chứng minh "model học được cái nó đã thấy" — đó là augmentation, không phải robustness.

### 6.2 Năm loại nhiễu và mô hình vật lý của chúng

> 📁 **Code:** `src/gtsrb/robustness/corruptions.py` → `motion_blur()`, `gauss_noise()`, `fog()`, `low_light()`, `occlusion()`

| Nhiễu | Mô phỏng tình huống | Mô hình |
|---|---|---|
| Motion blur | xe đang chạy, biển lướt qua | tích chập với kernel đường thẳng, dài 3→15 px, góc ngẫu nhiên |
| Gaussian noise | camera ISO cao, cảm biến kém | `I' = I + N(0, σ²)`, σ = 5/10/20/40/60 (trên thang 255) |
| Fog | sương mù, mưa | **tán xạ khí quyển (Koschmieder)**: `I = J·t + A·(1−t)`, `t = e^(−βd)`; `J` ảnh gốc, `A` độ sáng khí quyển, `β` hệ số tán xạ, `d` khoảng cách |
| Low-light | ban đêm | gamma `I' = 255·(I/255)^γ` với γ = 1,5→3,0, kèm nhiễu Poisson (nhiễu photon thật của ảnh tối) |
| Occlusion | lá cây, sticker, biển bị che | hình chữ nhật che 10/20/30/40/50% diện tích, vị trí ngẫu nhiên |

### 6.3 Relative robustness — vì sao không chỉ báo accuracy tuyệt đối

> 📁 **Code:** `src/gtsrb/robustness/benchmark.py` → `robustness_sweep()`, `mean_corruption_error()`

```
relative_robustness = acc_nhiễu / acc_sạch
```
Nếu chỉ báo accuracy tuyệt đối dưới nhiễu thì model nào giỏi sẵn sẽ luôn thắng, và ta
không học được gì về **tính bền**. Chia cho accuracy sạch tách hai thứ đó ra.

### ★ Kết quả ĐO THẬT — hai trong ba giả thuyết bị BÁC BỎ

| Model | fog | gauss_noise | low_light | motion_blur | occlusion | **mCE** | clean top-1 |
|---|---|---|---|---|---|---|---|
| M1 LeNet | 0,7254 | **0,9099** | **0,7795** | 0,8494 | **0,5009** | **0,2586** | 0,9846 |
| M2 VGG-res | 0,7658 | 0,8801 | 0,6486 | 0,8248 | 0,4697 | 0,2875 | 0,9926 |
| M3 ResNet18 | **0,9336** | 0,6840 | 0,5576 | 0,9995* | 0,4408 | 0,2817 | **0,9933** |

\* xem cảnh báo về motion blur bên dưới.

**Giả thuyết 1 — "model accuracy cao nhất không chắc bền nhất": ★ XÁC NHẬN MẠNH.**
Clean tốt nhất là **M3** (0,9933) nhưng mCE tốt nhất là **M1** (0,2586) — model **yếu
nhất** về accuracy lại **bền nhất**. Đây là phát hiện đáng giá nhất của phần robustness.

**Giả thuyết 2 — "pretrained ImageNet bền hơn trước nhiễu quang học": BÁC BỎ.**
M3 **tệ hơn hẳn** M1 ở cả hai loại so sánh được: gauss_noise 0,684 so với 0,910;
low_light 0,558 so với 0,780. Giải thích khả dĩ: M3 chuẩn hoá bằng thống kê ImageNet
và các lớp BatchNorm pretrained có `running_mean/var` cố định — khi độ sáng ảnh đổi
(gamma) hoặc nhiễu cộng vào, phân phối activation lệch khỏi điểm làm việc mà trọng số
pretrained quen. Model train từ số 0 trên chính GTSRB không có ràng buộc đó.

**Giả thuyết 3 — "GAP bền hơn trước occlusion": BÁC BỎ.**
M2 (có GAP) = 0,4697, **kém hơn** M1 (Flatten+FC) = 0,5009. Cả ba model đều mất
50–56% accuracy khi bị che — occlusion là loại nhiễu phá hoại nhất với mọi kiến trúc.

> **Nói ra chỗ mình dự đoán sai là điểm cộng khi bảo vệ.** Một dự đoán bị số liệu bác bỏ
> vẫn là kết quả khoa học; che giấu nó mới là vấn đề.

### ⚠️ Cảnh báo về cột `motion_blur`

Nhiễu được áp **sau khi** resize về `img_size`, mà M1/M2 ở 48×48 còn M3 ở 224×224.
Kernel motion blur mức 5 là **15 pixel tuyệt đối**: ở 48×48 nó phủ **31%** chiều rộng
ảnh, ở 224×224 chỉ **6,7%** (tương đương 3,2 pixel gốc). Cùng "mức 5" nhưng M3 chịu
nhiễu **nhẹ hơn ~4,7 lần**.

→ Con số 0,9995 của M3 **phần lớn là hiện vật của đường ống**, không phải tính bền thật.

Ba loại nhiễu **so sánh được** giữa các độ phân giải (vì chúng theo từng pixel hoặc theo
% diện tích): `gauss_noise`, `low_light`, `occlusion`. Kết luận so sánh **chỉ rút từ ba
cột đó**. Chi tiết: `docs/SU_CO.md` mục 8.

(mCE kiểu ImageNet-C: lấy trung bình error chuẩn hoá qua mọi loại nhiễu và mọi mức severity,
cho một con số duy nhất để xếp hạng.)

### 6.4 FLOPs không phải latency

Đây là bài học quan trọng nhất của phần triển khai.

### Số ĐO THẬT của nhóm (CPU, batch = 1, Apple M1 Pro)

| Model | FLOPs | Latency p50 | **ms / GFLOP** |
|---|---|---|---|
| M1 LeNet | 0,075 G | 0,93 ms | **12,41** |
| M2 VGG-res | 0,290 G | 3,16 ms | **10,89** |
| M3 ResNet18 | 3,647 G | 13,65 ms | **3,74** |

Nếu latency **tỉ lệ** với FLOPs thì cột cuối phải **bằng nhau**. Thực tế nó chênh
**3,3 lần** (trên MPS batch=64 chênh 2,2 lần). Vậy **latency không tỉ lệ với FLOPs**.

### ★ Và chiều của nó NGƯỢC với ví dụ kinh điển

Câu chuyện thường được kể: *"MobileNetV2 ít FLOPs hơn ResNet18 6 lần nhưng không nhanh
hơn 6 lần"* — tức model **nhỏ** kém hiệu quả trên mỗi FLOP.

Nhóm đo được điều ngược lại: model **lớn nhất** (M3) lại **hiệu quả nhất** trên mỗi FLOP
(3,74 so với 12,41 của M1). Ba nguyên nhân:

1. **Tensor quá nhỏ không lấp đầy phần cứng.** M1/M2 chạy ở 48×48 với vài chục kênh —
   mỗi phép conv quá bé để che được chi phí cố định (launch kernel, độ trễ truy cập bộ
   nhớ, đồng bộ). Phần cứng ngồi **chờ** nhiều hơn là tính. M3 ở 224×224 với conv dày đặc
   giữ phần cứng bận liên tục.
2. **Arithmetic intensity.** FLOPs chỉ đếm **phép tính**, không đếm **truy cập bộ nhớ**.
   Conv dày đặc ở độ phân giải lớn có tỉ lệ tính-toán/byte cao; conv nhỏ (và depthwise
   conv) thì thấp → bị chặn bởi **băng thông bộ nhớ**.
3. **Mức độ tối ưu của thư viện.** cuDNN/BLAS tối ưu rất kỹ cho các kích thước tensor
   phổ biến của ImageNet (224×224); kích thước lạ ít được chăm hơn.

**Cả hai chiều dẫn tới CÙNG một kết luận hành động: muốn nói về triển khai thì phải
đo wall-clock trên thiết bị đích, không suy từ FLOPs.** Nhưng khi trình bày phải nói
đúng chiều **mình đo được**, không chép chiều của bài báo khác — giảng viên sẽ hỏi
"số của em cho thấy điều đó ở đâu?".

### 6.5 Đo latency cho đúng

> 📁 **Code:** `src/gtsrb/deploy/speed.py` → `benchmark()`, `_synchronize()`

| Bước | Vì sao |
|---|---|
| `model.eval()` + `torch.no_grad()` | tắt dropout/BN-train và không dựng graph autograd |
| ≥ 20 vòng warm-up | lần chạy đầu tốn thời gian khởi tạo kernel, cấp phát bộ nhớ, nạp cache |
| **Đồng bộ thiết bị** trước & sau khi bấm giờ | GPU/MPS chạy **bất đồng bộ**; không sync là đang đo thời gian *gửi lệnh*, không phải thời gian *tính* — sai hàng chục lần |
| 100 vòng đo, báo **p50 và p95** | hệ thống thời gian thực quan tâm trường hợp xấu; mean che mất đuôi phân phối |
| batch=1 **và** batch=64 | batch=1 là tình huống xe thật (xử lý từng khung); batch=64 đo throughput |
| ghi rõ thiết bị + phiên bản thư viện | số latency không so sánh được giữa hai máy khác nhau |

### 6.6 Grad-CAM

> 📁 **Code:** `src/gtsrb/explain/gradcam.py` → lớp `GradCAM`, các hàm `_forward_backward()`, `_combine_channels()`

```
α_k^c = (1/Z) Σ_i Σ_j  ∂y^c / ∂A^k_{ij}            # trọng số của kênh k cho lớp c
L^c   = ReLU( Σ_k α_k^c · A^k )                     # tổ hợp có trọng số
```
**Trực giác.** `∂y^c/∂A^k` nói "nếu kênh `k` mạnh lên thì điểm số lớp `c` tăng hay giảm".
Gộp theo không gian (`1/Z Σ Σ`) cho một con số duy nhất: *kênh này quan trọng cỡ nào cho lớp c*.
Rồi tổ hợp các kênh theo trọng số đó. `ReLU` để giữ **chỉ** phần hỗ trợ lớp `c`
(phần âm là bằng chứng chống lại lớp `c`, không phải thứ ta muốn hiển thị).

**Vì sao dùng feature map cuối.** Nó có ngữ nghĩa cao nhất **mà vẫn còn giữ thông tin vị trí**
(FC sau đó mới xoá hết vị trí). Đó là lớp cuối cùng còn trả lời được "ở đâu".

**Giới hạn — phải nói ra:**

*Độ phân giải thô.* Số ĐO THẬT của dự án (`make gradcam` in ra bảng này):

| Model | Input | Target layer | Feature map | 1 ô heatmap ≈ |
|---|---|---|---|---|
| M1 LeNet | 48px | `features` (sau 2 MaxPool) | **12×12** | **4×4 px** |
| M2 VGG-res | 48px | `stages[-1]` (sau 4 MaxPool) | **3×3** | **16×16 px** |
| M3 (cả 3 backbone) | 224px | `layer4` / `features[-1]` | **7×7** | **32×32 px** |

**Kết quả này phản trực giác và đáng nói ra khi trình bày:** heatmap **mịn nhất** là
của **M1**, không phải M3. Lý do đơn giản — M1 chỉ có **2** lớp MaxPool nên downsample
4 lần, còn M2 có 4 lớp nên downsample 16 lần, và M3 downsample 32 lần.

So sánh cho công bằng thì phải tính theo **pixel gốc**: M3 ở 224 là bản upsample của
ảnh 48, nên 32px của nó tương đương ≈ 7 pixel gốc. Xếp hạng theo pixel gốc mỗi ô:
**M1 (4) mịn hơn M3 (≈7) mịn hơn M2 (16)**.

Nói cách khác: M2 — model được thiết kế công phu nhất — lại cho bản đồ nhiệt **thô nhất**,
vì mỗi stage đều có pooling. Muốn nhìn rõ hơn ở M2 thì lấy `stages[2]` (6×6), nhưng
ngữ nghĩa thấp hơn. Đó là một đánh đổi thật, không phải lỗi.

*Hai giới hạn còn lại:*
- Nó chỉ cho biết **ở đâu**, không cho biết **đặc trưng gì** ở đó.
- Nó **không** là bằng chứng nhân quả: model có thể nhìn đúng chỗ mà vẫn suy luận sai.

**Trong bài này**, 3 nhóm ảnh cho mỗi model: (a) đúng, (b) **sai — model nhìn vào đâu**,
(c) cùng ảnh trước/sau nhiễu — attention có trôi không.

---

## Phần 7. Nén model cho thiết bị biên  `Phong Trần`

> 📁 **Code:** `src/gtsrb/deploy/export.py` → `quantize_dynamic()`, `quantize_static()`, `export_onnx()` · test: `tests/test_export.py`

### 7.1 Vì sao phải nén

Đo latency của model fp32 mới là một nửa câu chuyện. Thiết bị biên thật — Raspberry Pi,
Jetson, vi điều khiển trên xe — hiếm khi chạy fp32, vì ba lý do:

- **nhẹ hơn 4 lần** trên đĩa và trong RAM;
- nhiều chip có **lệnh SIMD int8 chuyên dụng** nên tính nhanh hơn;
- tốn **ít băng thông bộ nhớ** hơn — mà băng thông mới là nút cổ chai thật (mục 6.4).

### 7.2 Hai kiểu lượng tử hoá

**Trực giác chung.** Số thực 32 bit có dải rất rộng, nhưng trọng số của một mạng đã train
thường nằm gọn trong khoảng nhỏ, ví dụ `[-0,8; 0,8]`. Ta có thể chia khoảng đó thành 256
mức và chỉ lưu *chỉ số mức* (1 byte) thay vì số thực (4 byte). Đó là lượng tử hoá.

```
giá trị thật  ≈  scale × (số_nguyên_int8 − zero_point)
```

`scale` và `zero_point` là hai số thực lưu kèm cho mỗi tensor (hoặc mỗi kênh).

| | Dynamic | Static |
|---|---|---|
| Lượng tử hoá gì | **chỉ trọng số** Linear | **trọng số + activation**, cả Conv |
| Cần dữ liệu hiệu chuẩn | không | **có** — chạy vài trăm ảnh thật để đo dải activation |
| Lợi ích chính | giảm **dung lượng** | giảm dung lượng **và tăng tốc** |
| Dùng khi | muốn nhanh gọn | làm sản phẩm thật |

### 7.3 ★ Kết quả đo thật — kiến trúc quyết định nén có hiệu quả không

| Model | fp32 | int8 dynamic | Nhẹ hơn |
|---|---|---|---|
| M1 LeNet | 9,70 MB | 2,59 MB | **3,75×** |
| M2 VGG-res | 4,99 MB | 4,95 MB | **1,0×** (không giảm) |

**Vì sao chênh nhau đến thế?** Dynamic quantization chỉ đụng tới lớp `Linear`.
M1 có **97,3%** tham số ở một lớp Linear nên nó hưởng trọn lợi ích. M2 dùng Global
Average Pooling nên Linear chỉ còn **11.051** tham số — gần như không có gì để nén.

→ **Bài học: quyết định kiến trúc (mục 2.6) ảnh hưởng tới cả khả năng nén sau này.**

Với M1, so sánh đầy đủ ba biến thể:

| Biến thể | MB | Top-1 | p50 | Nhẹ hơn | Nhanh hơn | **Mất top-1** |
|---|---|---|---|---|---|---|
| fp32 | 9,70 | 0,98167 | 1,04 ms | — | — | — |
| int8 dynamic | 2,59 | 0,98167 | 1,03 ms | 3,75× | 1,01× | **0,000** |
| int8 static | 2,43 | 0,98000 | 0,80 ms | 3,99× | **1,30×** | **0,167** |

Dynamic nhẹ đi 3,75 lần mà **không mất điểm nào**, nhưng cũng **không nhanh hơn**.
Static mới nhanh hơn thật (1,3 lần), nhưng **trả giá 0,167 điểm**.

> **Khi báo cáo phải có cột "mất top-1".** Nói "nhẹ hơn 4 lần" mà không nói mất bao
> nhiêu accuracy là báo cáo thiếu trung thực — người đọc sẽ tưởng nén là bữa trưa miễn phí.

### 7.4 TorchScript và ONNX — khác lượng tử hoá

Hai thứ này **không nén**; chúng tách model ra khỏi mã Python để chạy được trên
C++ runtime, ONNX Runtime, TensorRT, CoreML. Cần thiết vì thiết bị biên thường
không có Python.

> ⚠️ **Bẫy đã gặp thật:** bộ xuất ONNX mới của PyTorch mặc định tách trọng số ra file
> riêng `<tên>.onnx.data`, file `.onnx` còn lại chỉ ~3 KB. Copy mỗi file `.onnx` sang
> thiết bị là model **không chạy được**, và lỗi lúc nạp không nói là thiếu file.
> Hàm `export_onnx()` của nhóm ép gộp một file và báo đúng dung lượng.

> ⚠️ **Luôn gọi `verify_onnx()` sau khi xuất.** Xuất thành công không có nghĩa là xuất
> đúng — một số phép toán bị dịch sai hoặc bị xấp xỉ, và sai lệch chỉ lộ ra khi so
> output thật. Nhóm đo được sai lệch 8,34e-07, tức khớp.

---

## Phần 8. TTA và ensemble  `Phong Nguyễn`

> 📁 **Code:** `src/gtsrb/eval/ensemble.py` → `predict_tta()`, `ensemble_probs()`, `disagreement_rate()` · test: `tests/test_ensemble.py`

### 8.1 Vì sao phần này liên quan trực tiếp tới GTSRB

Bài **thắng giải IJCNN 2011 trên chính bộ dữ liệu này** (IDSIA, 99,46%) là một
*committee of CNNs* — tức ensemble. Nên đây không phải kỹ thuật phụ: nó chính là cách
người ta đạt SOTA trên bài toán này.

### 8.2 Hai kỹ thuật

**TTA (test-time augmentation).** Một model, chạy nhiều lần trên các biến thể của
**cùng một ảnh** (xoay nhẹ, zoom nhẹ), rồi lấy trung bình xác suất.
*Trực giác:* nếu model chỉ đúng nhờ một góc nhìn may mắn, lấy trung bình nhiều góc sẽ
lộ ra sự thiếu chắc chắn đó. Không cần train thêm, nhưng **latency nhân lên đúng số
biến thể**.

**Ensemble.** Nhiều model khác nhau, mỗi model chạy một lần, rồi gộp xác suất.
*Trực giác:* các model mắc lỗi ở **những chỗ khác nhau**; gộp lại thì lỗi riêng của
từng model bị đa số lấn át. Latency bằng **tổng** các model.

**Gộp bằng trung bình XÁC SUẤT, không phải trung bình logit.** Logit không cùng thang
đo giữa các model, trung bình chúng là phép toán không có nghĩa thống kê. Trung bình
xác suất thì vẫn là một phân phối hợp lệ.

### 8.3 Khi nào ensemble đáng làm

Chỉ khi các thành phần **mắc lỗi khác nhau**. Hàm `disagreement_rate()` đo điều đó:
tỉ lệ ảnh mà các model không đồng ý về nhãn. Nếu gần 0 thì các model về cơ bản giống
hệt nhau — gộp lại không được gì mà latency vẫn nhân lên.

Số đo của nhóm (M1 + M2, trên một phần tập test):

| | macro-F1 |
|---|---|
| M1 đơn lẻ | 0,97857 |
| M2 đơn lẻ | 0,98219 |
| **Ensemble 2 model** | **0,98477** (+0,00258) |

Tỉ lệ bất đồng: **1,76%** — đủ khác nhau để ensemble có lợi, nhưng lợi ích nhỏ.
TTA trên riêng M2: 0,9822 → 0,9834 (+0,0012) với latency **nhân 3**.

> **Phải nói ra cái giá.** Cả hai kỹ thuật đều đánh đổi latency lấy accuracy. Với bài
> triển khai biên — mà cả dự án này hướng tới — đó thường là đánh đổi **sai chiều**.

### 8.4 ★ Cấm flip ngang trong TTA

Giống hệt lý do ở mục 1.3: lật ngang biến lớp 33 "rẽ phải" thành đúng hình lớp 34
"rẽ trái". TTA có flip sẽ **trung bình xác suất của hai lớp khác nhau** — làm accuracy
**tệ đi**, không phải tốt lên.

---

## Phần 9. Vì sao một seed là không đủ  `Hoàng`

> 📁 **Code:** `scripts/run_seeds.py` → `summarise()`, `report_separability()`

### 9.1 Hai loại phương sai khác nhau

| Loại | Nguồn | Xử lý bằng |
|---|---|---|
| **Trong một lần chạy** | hai model đoán khác nhau trên cùng tập test | **McNemar** (mục 5.3) |
| **Giữa các lần chạy** | khởi tạo trọng số, thứ tự shuffle, augmentation ngẫu nhiên | **nhiều seed** |

McNemar xử lý loại thứ nhất rất tốt. Nhưng nó **hoàn toàn không nói gì** về loại thứ
hai. Train lại cùng model với seed khác sẽ ra một model khác, và chênh lệch giữa hai
lần chạy đó có thể lớn hơn chênh lệch giữa hai *kiến trúc*.

### 9.2 Quy tắc phát biểu kết quả

```
chênh lệch giữa hai model  >  2 × độ lệch chuẩn giữa các seed   ->  kết luận được
chênh lệch                 ≤  2 × độ lệch chuẩn                 ->  "nằm trong nhiễu"
```

Hàm `report_separability()` tự in ra kết luận này cho từng cặp model.

### 9.3 Vì sao dự án này cần nó

M2 và ResNet18 chênh nhau **9 ảnh trên 12.630**. McNemar nói p = 0,4477 (tương đương).
Nhưng ngay cả khi p < 0,05, ta vẫn chưa biết kết quả có lặp lại với seed khác không.

Nhóm hiện mới chạy **một seed** mỗi cấu hình — đó là hạn chế lớn nhất về phương pháp,
và đã ghi rõ ở `docs/BAO_CAO.md` mục 5. Chạy `make seeds` để bổ sung.
