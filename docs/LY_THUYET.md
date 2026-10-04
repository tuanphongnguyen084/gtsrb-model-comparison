# Lý thuyết — vì sao chọn như vậy

> **Cách dùng**: đây là sổ tay khái niệm, đọc theo thứ tự. Mỗi khái niệm trình bày theo
> 3 lớp: **trực giác** (hiểu bằng lời) → **công thức** (nói được khi bị hỏi) →
> **trong bài này** (nhóm dùng nó ở đâu, giá trị bao nhiêu).
> Nhãn `[A]/[B]/[C]/[D]` là người chịu trách nhiệm viết phần đó sau khi có số liệu thật.
>
> Ký hiệu: `N` = số ảnh, `K = 43` = số lớp, `C` = số kênh, `H×W` = kích thước không gian.

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

## Phần 1. Dữ liệu  `[A]`

### 1.1 Rò rỉ dữ liệu do cấu trúc track — khái niệm quan trọng nhất của phần dữ liệu

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

Lớp 2 ("giới hạn 50 km/h") có 2.250 ảnh; lớp 0 ("giới hạn 20"), 19, 37 chỉ có 210.
Tỉ lệ ~10,7:1. Nhóm **không** resample, vì: (a) phân phối này phản ánh tần suất thật trên
đường Đức; (b) nhóm đổi **chỉ số** sang macro-F1 thay vì cân bằng lại **dữ liệu** —
can thiệp ở chỗ đúng hơn; (c) vẫn stratify theo lớp khi split. Hướng mở rộng nếu còn thời
gian: `class_weight` trong cross-entropy, hoặc focal loss.

---

## Phần 2. Kiến trúc mạng từ số 0  `[B]`

### 2.1 Vì sao conv thay vì fully-connected

Ảnh 48×48×3 = 6.912 giá trị. Một lớp FC 1.000 neuron là ~6,9 triệu tham số **chỉ cho một lớp**.
Conv giải quyết bằng hai ý tưởng: **kết nối cục bộ** (mỗi neuron chỉ nhìn một ô 3×3 —
vì pixel liên quan tới pixel gần nó) và **chia sẻ tham số** (cùng một kernel trượt khắp ảnh —
vì "cạnh" là "cạnh" bất kể nó nằm ở đâu). Kết quả: một kernel 3×3 cho 32 kênh đầu vào,
64 kênh đầu ra chỉ cần `3·3·32·64 = 18.432` tham số, và nó **bất biến với dịch chuyển**.

### 2.2 Hai conv 3×3 thay một conv 5×5 — luận điểm của VGG

**Receptive field.** Conv 3×3 thứ nhất: mỗi output nhìn 3×3 input. Conv 3×3 thứ hai:
mỗi output nhìn 3×3 của tầng trước, mà mỗi cái đó lại nhìn 3×3 → tổng cộng **5×5** input.
Công thức tổng quát cho `n` lớp 3×3 stride 1: `RF = 2n + 1`.

**Tham số.** `2 · (3·3·C²) = 18C²` so với `5·5·C² = 25C²` → **giảm 28%**.
Với ba lớp 3×3 (RF 7×7) thì `27C²` so với `49C²` → giảm 45%.

**Phi tuyến.** Đây mới là điểm then chốt: 2 lớp có **2** hàm ReLU thay vì 1. Cùng một
receptive field nhưng hàm biểu diễn được phong phú hơn nhiều. Một stack conv 3×3 không
phải là "cách rẻ để làm conv 5×5", nó là **một hàm mạnh hơn**.

### 2.3 Batch Normalization

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

## Phần 3. Huấn luyện  `[B]`

### 3.1 Cross-entropy và label smoothing

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

Adam giữ hai ước lượng trung bình động: moment 1 (`m_t`, hướng) và moment 2 (`v_t`, độ lớn
bình phương), rồi bước theo `m̂_t / (√v̂_t + ε)`. Hiệu quả thực tế: mỗi tham số có một
learning rate thích ứng riêng → ít phải tinh chỉnh LR, hội tụ nhanh ở giai đoạn đầu.
`m̂, v̂` là phiên bản đã hiệu chỉnh bias (vì `m_0 = v_0 = 0` nên những bước đầu bị lệch về 0).

Ghi chú cho M3: nhóm dùng **AdamW** — weight decay tách khỏi gradient adaptive.
Trong Adam gốc, weight decay bị chia cho `√v̂` nên tham số có gradient lớn bị decay ít hơn,
không đúng ý nghĩa của regularization.

### 3.3 Cosine annealing + warmup

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

Theo dõi **val macro-F1**, patience 10 epoch, lưu checkpoint tốt nhất.
**Không** theo accuracy — vì mất cân bằng 10,7:1, accuracy có thể gần như không đổi trong
khi một lớp thưa sập hoàn toàn. Nguyên tắc: *chọn checkpoint theo đúng chỉ số mình thật sự
quan tâm*. Và tuyệt đối không theo chỉ số trên tập test.

---

## Phần 4. Transfer learning  `[C]`

### 4.1 Vì sao tri thức ImageNet dùng được cho biển báo

CNN học đặc trưng **có phân tầng**:

| Tầng | Học được gì | Có dùng lại được cho GTSRB? |
|---|---|---|
| Đầu (conv1, layer1) | cạnh, góc, đốm màu, gradient | **Có** — phổ quát cho mọi ảnh tự nhiên |
| Giữa (layer2, layer3) | texture, hoa văn, hình khối, bộ phận | **Phần lớn** — hình tròn/tam giác/mũi tên |
| Cuối (layer4, fc) | ngữ nghĩa "mèo", "xe tải" | **Không** — phải thay |

ImageNet có 1,28 triệu ảnh. GTSRB có 39 nghìn. Tri thức tầng đầu học từ 1,28 triệu ảnh là
thứ 39 nghìn ảnh **không học nổi**. Đó là nội dung thực sự được "chuyển giao".

### 4.2 Vì sao phải upsample lên 224×224

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

*Pha 1 (3 epoch)*: `requires_grad = False` cho backbone, chỉ train head.
Lý do: head khởi tạo ngẫu nhiên nên loss ban đầu rất lớn (`≈ ln 43 ≈ 3,76`) → gradient rất
lớn. Nếu backbone đang mở, gradient đó chảy ngược và phá filter pretrained ngay trong vài
trăm bước đầu, trước khi LR schedule kịp làm gì.

*Pha 2*: mở băng toàn bộ, dùng discriminative LR.

### 4.5 Chuẩn hoá bằng thống kê ImageNet

M3 dùng `mean = (0,485, 0,456, 0,406)`, `std = (0,229, 0,224, 0,225)` — của ImageNet,
**không** của GTSRB. Hai lý do: (1) filter tầng đầu được tối ưu cho đúng phân phối đầu vào đó;
(2) `running_mean/running_var` trong các lớp BatchNorm pretrained được tích luỹ trên phân phối
đó — đổi phân phối đầu vào là làm chúng sai. Ngược lại, M1 và M2 train từ số 0 thì dùng
thống kê GTSRB tính **chỉ trên tập train**.

### 4.6 Ba backbone — ba ý tưởng kiến trúc khác nhau

| Backbone | #params | FLOPs @224 | Ý tưởng cốt lõi |
|---|---|---|---|
| ResNet18 | 11,7M | ~1,82 G | residual + basic block; conv dày đặc, đơn giản, nhanh trên GPU |
| MobileNetV2 | 3,5M | ~0,30 G | **depthwise separable conv** (tách conv không gian và conv kênh) + **inverted residual** (nở ra rồi bóp lại) + **linear bottleneck** (bỏ ReLU ở lớp hẹp để không mất thông tin) |
| EfficientNet-B0 | 5,3M | ~0,39 G | **compound scaling** (tăng depth/width/resolution cùng lúc theo một hệ số) + MBConv + **squeeze-and-excitation** (attention trên trục kênh) |

So sánh ba cái này làm nổi bật một bài học: **ít FLOPs ≠ nhanh hơn**. Xem Phần 6.

---

## Phần 5. Đánh giá  `[D]`

### 5.1 Top-1, Top-5 — và tại sao top-5 yếu ở đây

Top-1: dự đoán có xác suất cao nhất đúng. Top-5: nhãn đúng nằm trong 5 ứng viên cao nhất.

Top-5 ra đời cho **ImageNet 1000 lớp**, nơi nhiều lớp mơ hồ một cách chính đáng
(nhiều giống chó rất giống nhau) và một ảnh có thể chứa nhiều đối tượng.
Trên **43 lớp**, top-5 nghĩa là "đúng trong 11,6% số lớp" — mọi model tốt đều ~99,9%.
**Chỉ số bão hoà thì không phân biệt được gì.** Nhóm vẫn báo cáo vì đề bài yêu cầu, nhưng
nêu rõ giới hạn này và kết luận dựa trên top-1 + macro-F1.

### 5.2 Macro-F1 — chỉ số chính

```
Precision_k = TP_k / (TP_k + FP_k)        Recall_k = TP_k / (TP_k + FN_k)
F1_k = 2 · P_k · R_k / (P_k + R_k)        macro-F1 = (1/K) Σ_k F1_k
```
Điểm then chốt: `(1/K) Σ` — **mỗi lớp một phiếu bằng nhau**, bất kể lớp đó có 210 hay 2.250 ảnh.

So sánh: nếu model bỏ hẳn một lớp 210 ảnh (0,54% dữ liệu) thì accuracy chỉ giảm ~0,54 điểm
(gần như không thấy), nhưng macro-F1 mất `1/43 ≈ 2,3` điểm (thấy rõ).
Trong bài biển báo, nhận sai một biển hiếm có thể nguy hiểm hơn nhận sai một biển phổ biến.

(Phân biệt: **micro**-F1 gộp hết TP/FP/FN rồi mới tính → trong bài đa lớp đơn nhãn nó
bằng đúng accuracy. **Weighted**-F1 lấy trung bình có trọng số theo support → lại bị lớp đông
chi phối. Nhóm báo cáo cả ba nhưng kết luận theo macro.)

### 5.3 McNemar test — "chênh lệch này có thật không?"

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

## Phần 6. Robustness và triển khai biên  `[C] [D]`

### 6.1 Robustness nghĩa là gì — và luật sắt

Robustness = giữ được hiệu năng trên **phân phối chưa từng thấy lúc train**
(distribution shift). Suy ra **luật sắt**: các loại nhiễu dùng để đo **tuyệt đối không**
được đưa vào train. Train trên chúng rồi test trên chúng là test in-distribution, chỉ
chứng minh "model học được cái nó đã thấy" — đó là augmentation, không phải robustness.

### 6.2 Năm loại nhiễu và mô hình vật lý của chúng

| Nhiễu | Mô phỏng tình huống | Mô hình |
|---|---|---|
| Motion blur | xe đang chạy, biển lướt qua | tích chập với kernel đường thẳng, dài 3→15 px, góc ngẫu nhiên |
| Gaussian noise | camera ISO cao, cảm biến kém | `I' = I + N(0, σ²)`, σ = 5/10/20/40/60 (trên thang 255) |
| Fog | sương mù, mưa | **tán xạ khí quyển (Koschmieder)**: `I = J·t + A·(1−t)`, `t = e^(−βd)`; `J` ảnh gốc, `A` độ sáng khí quyển, `β` hệ số tán xạ, `d` khoảng cách |
| Low-light | ban đêm | gamma `I' = 255·(I/255)^γ` với γ = 1,5→3,0, kèm nhiễu Poisson (nhiễu photon thật của ảnh tối) |
| Occlusion | lá cây, sticker, biển bị che | hình chữ nhật che 10/20/30/40/50% diện tích, vị trí ngẫu nhiên |

### 6.3 Relative robustness — vì sao không chỉ báo accuracy tuyệt đối

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

| Bước | Vì sao |
|---|---|
| `model.eval()` + `torch.no_grad()` | tắt dropout/BN-train và không dựng graph autograd |
| ≥ 20 vòng warm-up | lần chạy đầu tốn thời gian khởi tạo kernel, cấp phát bộ nhớ, nạp cache |
| **Đồng bộ thiết bị** trước & sau khi bấm giờ | GPU/MPS chạy **bất đồng bộ**; không sync là đang đo thời gian *gửi lệnh*, không phải thời gian *tính* — sai hàng chục lần |
| 100 vòng đo, báo **p50 và p95** | hệ thống thời gian thực quan tâm trường hợp xấu; mean che mất đuôi phân phối |
| batch=1 **và** batch=64 | batch=1 là tình huống xe thật (xử lý từng khung); batch=64 đo throughput |
| ghi rõ thiết bị + phiên bản thư viện | số latency không so sánh được giữa hai máy khác nhau |

### 6.6 Grad-CAM

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
