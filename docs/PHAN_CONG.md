# Phân công nhóm — GTSRB 43 lớp

| Người | Vai trò | Khối lượng | Vì sao giao phần này |
|---|---|---|---|
| **Phong Nguyễn** | M1 baseline + Bảng so sánh + Slide | **NHẸ NHẤT** | bận việc khác; M1 đã train xong, phần còn lại chủ yếu là chạy script có sẵn và trình bày |
| **Hoàng** | M2 + Training engine + 3 trục ablation | **Nặng nhất** | khái niệm khó nhất (BN, residual, spatial dropout, GAP, label smoothing); cả nhóm phụ thuộc `fit()` |
| **Phong Trần** | M3 transfer learning + Grad-CAM + Tốc độ | **Nặng, tốn compute nhất** | kỹ thuật khó (hook, discriminative LR, đo latency đúng cách); chạy Colab |
| **Huy** | Dữ liệu + Rò rỉ + 2 trục ablation + Robustness | **Vừa** | đặc tả rất rõ ràng, ít bẫy kỹ thuật, nhưng sở hữu **hai phát hiện mạnh nhất** của nhóm |

> **Cách dùng tài liệu này**: đọc **toàn bộ** mục của mình, rồi đọc *nhanh* 3 mục còn lại
> (giảng viên hỏi chéo). Mục "Bạn phải trả lời được" chính là đề cương ôn.
>
> Lịch chạy máy và cách tiết kiệm compute: [KE_HOACH_CHAY.md](KE_HOACH_CHAY.md).

---

## Bản đồ phụ thuộc

```
   HUY ────▶ data/processed/index.csv + cache .npy
   (dữ liệu)         │
                     ├──────────────┬──────────────┬──────────────┐
                     ▼              ▼              ▼              ▼
   HOÀNG ────▶ engine/fit() ──▶ M1 (Phong N.)  M2 (Hoàng)   M3 (Phong T.)
   (engine)          │                │              │              │
                     │                └──────────────┴──────────────┘
                     │                               │
                     ▼                               ▼
                                     artifacts/runs/*/result.json
                                                     │
                      ┌──────────────────────────────┼──────────────┐
                      ▼                              ▼              ▼
             PHONG NGUYỄN                         HUY          PHONG TRẦN
             bảng so sánh + McNemar            robustness    Grad-CAM + tốc độ
```

**Luật vàng**: không ai sửa file của người khác. Thấy sai → nhắn người sở hữu.
Hai người sửa cùng một file Python là cách nhanh nhất để hỏng repo trước deadline.

**Tin tốt**: toàn bộ code đã được viết và chạy thử. Việc của mỗi người bây giờ là
**hiểu** phần của mình, **chạy** thực nghiệm, và **giải thích** được kết quả.

---

# PHONG NGUYỄN — M1 + Bảng so sánh + Slide

> **Phần nhẹ nhất.** M1 đã train xong (val macro-F1 = 0,9881). Hai lệnh là ra toàn bộ
> bảng và hình. Việc chính của bạn là **hiểu để trình bày**, không phải viết code.

### Sở hữu
```
src/gtsrb/models/m1_lenet.py      scripts/evaluate.py
src/gtsrb/eval/metrics.py         scripts/make_report.py
src/gtsrb/eval/stats_tests.py     notebooks/02_m1_lenet.ipynb
docs/SLIDE_OUTLINE.md             notebooks/05_evaluation_compare.ipynb
```

### Việc cụ thể — ước tính 4 giờ

| # | Việc | Lệnh | Thời gian |
|---|---|---|---|
| 1 | Đọc hiểu M1 | mở `src/gtsrb/models/m1_lenet.py` (87 dòng, có chú thích đầy đủ) | 30 phút |
| 2 | Train lại M1 (nếu cần) | `make train-m1` | 12 phút máy |
| 3 | Sinh toàn bộ bảng + hình | `make eval` | 2 phút máy |
| 4 | Sinh báo cáo | `make report` → `docs/KET_QUA.md` | tức thì |
| 5 | Chạy `notebooks/02` và `05` | đọc output, hiểu từng hình | 1 giờ |
| 6 | Làm slide | theo `docs/SLIDE_OUTLINE.md` | 2 giờ |

### Kết quả M1 đã có (số thật, không phải ước lượng)

| | |
|---|---|
| Tham số | **2.424.299** |
| Test top-1 | **98,46%** |
| Test macro-F1 | **0,9778** |
| ECE | 0,1310 |
| Thời gian train | 11,7 phút (M1 Pro, MPS) |

### Bạn phải trả lời được — 8 câu

1. **"Mô tả kiến trúc M1."**
   → `Conv(3→32, 5×5) → ReLU → MaxPool → Conv(32→64, 5×5) → ReLU → MaxPool →
   Flatten(9216) → FC(256) → ReLU → Dropout(0,5) → FC(43)`.
   Ảnh vào 48×48, sau hai lần pool còn 12×12, nên flatten ra `64×12×12 = 9216`.

2. **"Vì sao M1 cố ý không có BatchNorm và residual?"**
   → Vì M1 là **mốc tham chiếu**. Mọi điểm accuracy mà M2 hơn M1 phải quy được về
   một thành phần cụ thể. Nếu "tinh chỉnh M1 cho mạnh lên" thì mất luôn ý nghĩa đó.

3. ★ **"M1 có bao nhiêu tham số, nằm ở đâu?"**
   → **2.424.299**, trong đó **2.359.552 = 97,3%** nằm ở **một** lớp fully-connected
   duy nhất (`Flatten(9216) → FC(256)`). Hai lớp conv cộng lại chỉ ~2,2%.
   → Đây chính là lý do mọi kiến trúc hiện đại bỏ Flatten+FC, dùng **Global Average
   Pooling**. M2 thay chỗ đó bằng `GAP → FC(256→43)` chỉ 11.051 tham số — giảm ~213 lần.
   Bằng chứng: `reports/tables/m1_lenet_params.csv`.

4. ★ **"Vì sao macro-F1 là chỉ số chính chứ không phải accuracy?"**
   → `macro-F1 = (1/K) Σ F1_k` — **mỗi lớp một phiếu bằng nhau**. Dữ liệu mất cân bằng
   **10,7:1** (lớp 2 có 2.250 ảnh, lớp 0/19/37 chỉ 210). Nếu model bỏ hẳn một lớp 210 ảnh
   thì accuracy chỉ giảm ~0,54 điểm (gần như không thấy) nhưng macro-F1 mất `1/43 ≈ 2,3`
   điểm (thấy rõ). Trong bài biển báo, nhận sai một biển **hiếm** có thể nguy hiểm hơn
   nhận sai một biển phổ biến.

5. ★ **"Top-5 accuracy của nhóm 99,85%, tốt quá nhỉ?"** *(câu bẫy)*
   → Con số đó **gần như không có ý nghĩa ở đây**. Top-5 ra đời cho ImageNet **1000** lớp,
   nơi nhiều lớp mơ hồ chính đáng. Trên **43** lớp, top-5 nghĩa là "đúng trong 11,6% số
   lớp" — mọi model tử tế đều ~99,9%. **Chỉ số bão hoà thì không phân biệt được gì.**
   Nhóm báo cáo vì đề bài yêu cầu, nhưng kết luận dựa trên top-1 và macro-F1.

6. ★ **"Chênh lệch 98,46% và 99,26% có ý nghĩa thống kê không?"**
   → Phải **kiểm định**, không nhìn số rồi kết luận. Hai model chạy trên **cùng** tập test
   → quan sát **bắt cặp** → t-test hai mẫu độc lập là **sai**. Dùng **McNemar**, chỉ xét
   hai ô **bất đồng** của bảng 2×2 (`n₀₁`, `n₁₀`); hai ô đồng ý không mang thông tin.
   **Kết quả thật của nhóm** — và đây là phát hiện đắt giá nhất:

   | Cặp | n₀₁ | n₁₀ | p-value | Kết luận |
   |---|---|---|---|---|
   | M1 vs M2 | 35 | 136 | 2,05e-14 | M2 hơn, **có** ý nghĩa |
   | M1 vs M3 | 41 | 151 | 3,65e-15 | M3 hơn, **có** ý nghĩa |
   | **M2 vs M3** | 51 | 60 | **0,4477** | ★ **KHÔNG có ý nghĩa** |

   M3 nhỉnh hơn trên giấy (99,33% vs 99,26%) nhưng chênh lệch đó **không thật** — chỉ
   9 ảnh trên 12.630. Mà M3 dùng **9 lần nhiều tham số hơn** và chạy ở 224×224.
   → Nếu chỉ nhìn bảng accuracy, nhóm sẽ kết luận "transfer learning thắng" — **sai**.
   Chính McNemar ngăn kết luận đó. Đó là lý do mọi so sánh model đều phải kiểm định.

7. **"ECE là gì? Của M1 là 0,131 — cao hay thấp, và theo chiều nào?"**
   → `ECE = Σ (|Bₘ|/n)·|acc(Bₘ) − conf(Bₘ)|`, trả lời "khi model nói 90% chắc thì nó
   đúng 90% số lần không". M1 có **0,131 — cao**, tức hiệu chỉnh **kém**. Và theo chiều
   **THIẾU tự tin**: `conf = 0,854 < acc = 0,985`.
   → Lý thuyết (Guo 2017) nói mạng sâu thường **tự tin thái quá**, nhưng điều đó đúng trên
   ImageNet (bài toán khó, acc ~76%). GTSRB **quá dễ** (acc 98,5%), nên label smoothing
   kéo tự tin về ≈ 0,9023 — **thấp hơn** accuracy thật. Nó **sửa quá tay**.
   → M2 có ECE **0,0535**, tốt hơn hẳn. Trục ablation label smoothing (Hoàng) sẽ định lượng.

8. **"Model nào tốt nhất?"** *(câu bẫy — đừng trả lời "M3 vì accuracy cao nhất")*
   → Phụ thuộc tiêu chí, và nhóm đã đo trên **ba trục**: (1) accuracy **kèm p-value
   McNemar**; (2) **latency p95** ở batch=1; (3) **relative robustness** dưới nhiễu.
   Nếu chênh lệch accuracy không có ý nghĩa thống kê thì "tốt hơn" là một kết luận sai.

---

# HOÀNG — M2 + Training engine + Ablation kiến trúc

> **Phần nặng nhất về khái niệm.** Cả nhóm dùng chung `fit()` của bạn, nên nếu nó sai
> thì cả 3 model đều sai. Bù lại, bạn sở hữu những câu trả lời "ăn điểm" nhất.

### Sở hữu
```
src/gtsrb/models/m2_vggres.py  ★   src/gtsrb/engine/train.py      ★
src/gtsrb/models/registry.py       src/gtsrb/engine/losses.py
configs/m2_vggres.yaml             src/gtsrb/engine/schedulers.py
tests/test_models.py               src/gtsrb/engine/callbacks.py
notebooks/03_m2_vggres.ipynb
```

### Việc cụ thể — ước tính 14 giờ

| # | Việc | Lệnh | Compute |
|---|---|---|---|
| 1 | Đọc `m2_vggres.py` + `engine/train.py` | — | 2 giờ |
| 2 | Train M2 | `make train-m2` | 24 phút |
| 3 | Ablation **components** (bỏ BN / residual / spatial dropout) | `run_ablation.py --axes components --budget` | ~50 phút |
| 4 | Ablation **scaling** (width ×0,5/1/2, depth 3/4/5) | `--axes scaling --budget` | ~1,5 giờ |
| 5 | Ablation **label_smoothing** (0,0 / 0,1 / 0,2) | `--axes label_smoothing --budget` | ~40 phút |
| 6 | Chạy lại cấu hình thắng ở độ dài đầy đủ | bỏ `--budget` | ~1 giờ |
| 7 | Viết phần lý thuyết + 10 Q&A | — | 3 giờ |

### Kết quả M2 đã có

| | |
|---|---|
| Tham số | **1.237.451** — **ít hơn M1** dù sâu gấp 4 lần |
| Test top-1 | **99,26%** |
| Test macro-F1 | **0,9880** |
| ECE | **0,0535** (tốt hơn M1 gấp 2,4 lần) |
| McNemar vs M1 | p = 2,05e-14 → **hơn có ý nghĩa thống kê** |

### Bạn phải trả lời được — 10 câu

1. ★ **"Vì sao hai conv 3×3 thay vì một conv 5×5?"**
   → Ba lý do. (a) **Receptive field** bằng nhau: `RF = 2n+1`, hai lớp 3×3 cho 5×5.
   (b) **Tham số**: `2·(3·3·C²) = 18C²` so với `5·5·C² = 25C²` → giảm 28%.
   (c) Quan trọng nhất: có **hai** tầng phi tuyến ReLU thay vì một → hàm biểu diễn
   phong phú hơn. Stack 3×3 không phải "cách rẻ để làm 5×5", nó là hàm **mạnh hơn**.
   Đây là luận điểm trung tâm của VGG (Simonyan & Zisserman 2014).

2. ★ **"BatchNorm làm gì? Đặt trước hay sau ReLU?"**
   → Chuẩn hoá mỗi kênh về mean 0 var 1 theo mini-batch, rồi scale-shift bằng hai tham số
   **học được** γ, β (cần γ,β vì nếu luôn ép mean 0/var 1 thì mạng mất khả năng dùng vùng
   phi tuyến của ReLU). Lúc suy luận dùng `running_mean/var` — đó là lý do phải gọi
   `model.eval()`, quên là sai mà **không báo lỗi**. Nhóm đặt **Conv → BN → ReLU**.
   → *Điểm cộng*: bài gốc (Ioffe 2015) giải thích bằng "internal covariate shift", nhưng
   **Santurkar et al. 2018 đã phản biện** — tác dụng thật chủ yếu là **làm mượt bề mặt loss**,
   nhờ đó dùng được learning rate lớn hơn.

3. ★ **"Residual connection giải quyết vấn đề gì?"**
   → Vấn đề **degradation**, **không phải** overfitting. He et al. 2015 thấy mạng 56 lớp cho
   *train error* cao hơn mạng 20 lớp — train error cao thì không thể là overfit, mà là
   **không tối ưu hoá được**. Giải pháp `y = F(x) + x`: nếu identity là tối ưu thì chỉ cần
   đẩy `F(x) → 0`, dễ hơn nhiều so với học identity bằng cả stack conv.
   → Góc nhìn gradient: `∂y/∂x = ∂F/∂x + 1`. Số hạng **+1** là "đường cao tốc" cho gradient
   chảy ngược, chống vanishing gradient.
   → Khi đổi số kênh (32→64) thì shape không khớp → dùng **conv 1×1** (projection shortcut).

4. **"Thứ tự trong residual block: cộng rồi ReLU, hay ReLU rồi cộng?"**
   → **Cộng TRƯỚC, ReLU SAU**, theo bài ResNet gốc. Làm ngược lại thì nhánh skip bị chặn
   ở 0 và mất tác dụng "đường cao tốc gradient". Xem `ResidualBlock.forward()`.

5. ★ **"SpatialDropout khác Dropout thường thế nào? Vì sao dùng nó trên conv?"**
   → `Dropout` bỏ từng **phần tử** độc lập. Trên feature map, pixel lân cận **tương quan rất
   cao** (cùng nhìn gần như cùng một vùng ảnh), nên bỏ pixel (i,j) thì thông tin vẫn còn ở
   (i,j+1) → regularize gần như **vô hiệu**. `nn.Dropout2d` bỏ **toàn bộ một kênh** = một
   feature detector → buộc mạng không được dựa vào một detector duy nhất. Nhóm tăng p dần
   0,1 → 0,3 theo độ sâu vì tầng sâu nhiều kênh hơn và dễ overfit hơn.

6. ★ **"M2 sâu gấp 4 lần M1, vậy nhiều tham số hơn bao nhiêu?"** *(câu bẫy)*
   → **Nó ÍT HƠN.** M1 = 2.424.299 với **2** lớp conv; M2 = 1.237.451 với **9** lớp conv
   — chỉ bằng **51%**. Hai lý do: (a) conv dùng **tham số chia sẻ** (một kernel 3×3 trượt
   khắp ảnh) nên mỗi lớp conv rất rẻ; (b) M1 dồn 97,3% tham số vào một lớp FC, còn M2 thay
   chỗ đó bằng GAP.
   → **Số lớp KHÔNG tỉ lệ với số tham số. Chỗ đắt là lớp fully-connected.**
   Tham số của M2 dồn vào stage cuối: `stages.3.0.conv2` chiếm 47,7%.

7. ★ **"Label smoothing là gì? Công thức? Vì sao 0,1?"**
   → `y'_k = (1−ε)·y_k + ε/K`. Với K=43, ε=0,1: lớp đúng nhận `0,9 + 0,1/43 ≈ 0,9023`,
   mỗi lớp sai nhận `≈ 0,00233`. Lý do: với one-hot, để loss → 0 thì softmax phải đẩy logit
   lớp đúng ra **vô cực** → tự tin cực đoan.
   → *Chính xác hơn*: nói nó "đặt một **trần**" là **chưa đúng hẳn**. Nó làm **điểm cực tiểu
   của loss** nằm ở mức tự tin hữu hạn, chứ không chặn cứng đầu ra softmax. Nhóm đo được
   độ tự tin cao nhất là **0,9965 > 0,9023**.
   → Kết quả thật: trên GTSRB nó làm model **thiếu tự tin** (xem câu 7 của Phong Nguyễn).
   Trục ablation của bạn kiểm chứng giả thuyết: **ε = 0,0 sẽ cho ECE tốt hơn**.

8. **"Cosine annealing khác step decay thế nào? Vì sao cần warmup?"**
   → `lr_t = lr_min + ½(lr_max − lr_min)(1 + cos(πt/T))`. Đầu kỳ LR lớn để khám phá rộng,
   cuối kỳ về gần 0 để lắng vào đáy. So với step decay: không phải đoán "giảm ở epoch nào",
   và không có bước nhảy làm loss giật.
   → **Warmup 3 epoch**: ở những bước đầu, moment bậc 2 `v̂` của Adam ước lượng từ rất ít mẫu
   nên rất nhiễu → bước đi có thể rất lớn và sai hướng.

9. **"Vì sao early stopping theo macro-F1 chứ không theo accuracy?"**
   → Mất cân bằng 10,7:1 nên accuracy bị lớp đông chi phối: model có thể bỏ hẳn một lớp
   210 ảnh mà accuracy gần như không giảm. Nguyên tắc: **chọn checkpoint theo đúng chỉ số
   mình thật sự quan tâm.** Và tuyệt đối không theo chỉ số trên tập test.

10. **"Vì sao cả 3 model dùng CHUNG một hàm `fit()`?"**
    → Nếu mỗi người viết vòng train riêng thì 3 model được huấn luyện theo 3 cách khác nhau
    và bảng so sánh **mất ý nghĩa** — không biết chênh lệch đến từ kiến trúc hay từ cách train.
    Nhu cầu 2 pha freeze/unfreeze của M3 được đưa vào `cfg.train.phases` chứ không phải
    viết vòng train thứ hai.

---

# PHONG TRẦN — M3 Transfer learning + Grad-CAM + Tốc độ

> **Phần tốn compute nhất.** Nên chạy trên Colab (T4 nhanh hơn MPS khoảng 3–4 lần ở 224px).
> Kỹ thuật khó nhất nằm ở đây: hook của Grad-CAM và cách đo latency cho đúng.

### Sở hữu
```
src/gtsrb/models/m3_transfer.py  ★   src/gtsrb/deploy/speed.py
src/gtsrb/explain/gradcam.py     ★   scripts/benchmark_speed.py
configs/m3_*.yaml (3 file)           scripts/make_gradcam.py
notebooks/04, 08, 09
```

### Việc cụ thể — ước tính 14 giờ

| # | Việc | Lệnh | Compute |
|---|---|---|---|
| 1 | Đọc `m3_transfer.py` + `gradcam.py` + `speed.py` | — | 2,5 giờ |
| 2 | Train 3 backbone (**trên Colab**) | `make train-m3` | ~45 phút Colab / ~3 giờ MPS |
| 3 | Ablation **resolution** M3 (64/112/224) | `--axes resolution --budget` | ~1 giờ |
| 4 | Grad-CAM cho cả 3 model | `make gradcam` | 2 phút |
| 5 | Đo tốc độ + Pareto (**máy local, không chạy gì khác**) | `make speed` | 5 phút |
| 6 | Phân tích ≥3 ca dự đoán SAI | notebook 08 | 1,5 giờ |
| 7 | Viết lý thuyết + 10 Q&A | — | 3 giờ |

> ⚠️ **Lúc đo tốc độ phải tắt mọi thứ khác đang chạy.** Nếu máy đang train model khác
> thì số latency sai hoàn toàn và mọi kết luận về triển khai biên vô giá trị.

### Bạn phải trả lời được — 10 câu

1. ★ **"Transfer learning vì sao hiệu quả khi ImageNet (mèo, xe) khác xa biển báo?"**
   → CNN học đặc trưng **theo tầng**: tầng đầu học cạnh/góc/đốm màu — **phổ quát cho mọi
   ảnh tự nhiên**; tầng giữa học texture/hình khối; chỉ tầng cuối mang ngữ nghĩa riêng
   ImageNet, và nhóm thay hoàn toàn. ImageNet có **1,28 triệu** ảnh, GTSRB chỉ 39 nghìn —
   tri thức tầng đầu đó là thứ 39 nghìn ảnh **không học nổi**.

2. ★ **"Vì sao phải upsample 48×48 lên 224×224? Nghe như lãng phí."**
   → Ba backbone này downsample **32 lần**: `48/32 = 1,5` → feature map cuối 2×2, mất hoàn
   toàn cấu trúc không gian và `layer4` coi như vô dụng. `224/32 = 7×7` là **đúng cấu hình**
   mà trọng số pretrained được học.
   → *Phải trung thực*: nội suy 48→224 **KHÔNG THÊM THÔNG TIN**. Trung vị kích thước ảnh gốc
   GTSRB chỉ **43×43 px**, nên dù cache ở 224 cũng không có thêm chi tiết thật. Upsample để
   khớp **scale và stride**, không phải để có thêm chi tiết. Ablation 64/112/224 đo cái giá này.

3. ★ **"Discriminative learning rate là gì? Vì sao tầng đầu LR nhỏ?"**
   → Mỗi nhóm tầng một LR, tăng dần: đầu `1e-5`, giữa `1e-4`, head `1e-3`, cài qua
   `param_groups`. Tầng đầu đã học cạnh/màu **rất tốt** từ 1,28 triệu ảnh — cho LR lớn là
   đập bỏ tri thức đó bằng tín hiệu từ 39 nghìn ảnh (**catastrophic forgetting**). Tầng cuối
   mang ngữ nghĩa "mèo/xe" nên phải đổi hoàn toàn → LR lớn.
   → Cho toàn mạng `1e-3`: phá pretrained. Cho toàn mạng `1e-5`: head không hội tụ. **Hai cực
   đều tệ.**

4. ★ **"Vì sao đóng băng rồi mới mở băng? Mở luôn có sao?"**
   → Head khởi tạo ngẫu nhiên nên loss ban đầu rất lớn (`≈ ln 43 ≈ 3,76`) → gradient rất lớn.
   Nếu backbone đang mở, gradient đó chảy ngược và **phá filter pretrained** ngay trong vài
   trăm bước đầu.
   → **Bằng chứng thực nghiệm của nhóm** (`artifacts/runs/m3_resnet18*/train_log.csv`):

   | Epoch | Pha | val top-1 |
   |---|---|---|
   | 1–3 | `freeze_head` | 0,582 → 0,679 |
   | 4 | `finetune_all` | **0,964** |

   Train riêng head chững ở ~68%; **mở băng là nhảy lên 96% chỉ trong một epoch**.

5. **"Vì sao M3 dùng mean/std của ImageNet chứ không của GTSRB?"**
   → (a) filter tầng đầu được tối ưu cho đúng phân phối đó; (b) `running_mean/running_var`
   trong các lớp BatchNorm pretrained được tích luỹ **trên** phân phối đó — đổi đầu vào là
   làm chúng sai. Quên chuyện này là **lỗi im lặng**: model vẫn train, chỉ kết quả tệ.
   Vì vậy `scripts/train.py` **tự khớp** theo `model.normalize_mode`.

6. ★ **"Grad-CAM công thức và trực giác?"**
   → `α_k^c = (1/Z)·ΣᵢΣⱼ ∂y^c/∂A^k_ij`, rồi `L^c = ReLU(Σ_k α_k^c · A^k)`.
   (1) `∂y^c/∂A^k` nói "kênh k mạnh lên thì điểm lớp c tăng hay giảm"; (2) gộp theo không
   gian cho **một** số mỗi kênh; (3) tổ hợp các kênh theo trọng số đó; (4) `ReLU` giữ **chỉ**
   phần **hỗ trợ** lớp c (phần âm là bằng chứng **chống lại** lớp c).
   → Dùng feature map **cuối** vì nó có ngữ nghĩa cao nhất **mà vẫn giữ vị trí**.

7. ★ **"Grad-CAM có giới hạn gì?"**
   → (a) **Độ phân giải thô**, và số đo thật cho kết quả **phản trực giác**:

   | Model | Input | Feature map | 1 ô ≈ |
   |---|---|---|---|
   | M1 | 48px | **12×12** | 4×4 px (**mịn nhất**) |
   | M2 | 48px | **3×3** | 16×16 px (**thô nhất**) |
   | M3 | 224px | **7×7** | 32px ≈ 7 pixel gốc |

   **M2 — model thiết kế công phu nhất — cho heatmap THÔ NHẤT**, vì nó có 4 lớp MaxPool so
   với 2 của M1. Nói "heatmap đúng vào biển báo" với M2 là phát biểu rất lỏng: cả ảnh chỉ 9 ô.
   → (b) Nó chỉ cho biết **ở đâu**, không cho biết **đặc trưng gì**.
   → (c) Nó **không** phải bằng chứng nhân quả — model có thể nhìn đúng chỗ mà vẫn suy luận sai.

8. ★ **"FLOPs có dự đoán được latency không?"** — nhóm có số đo riêng, chênh **64 lần**.

   | Model | FLOPs | Latency p50 (CPU, bs=1) | **ms/GFLOP** |
   |---|---|---|---|
   | ResNet18 | 3,65 G | 16,7 ms | **4,6** |
   | M2 VGG-res | 0,29 G | 3,2 ms | 11,0 |
   | M1 LeNet | 0,07 G | 0,9 ms | 12,0 |
   | MobileNetV2 | 0,65 G | 47,9 ms | 73,4 |
   | **EfficientNet-B0** | 0,83 G | **243,4 ms** | **294,1** |

   Nếu latency tỉ lệ FLOPs thì cột cuối phải **bằng nhau** — thực tế chênh **64 lần**.

   **Điểm nhấn: "EfficientNet" là model KÉM HIỆU QUẢ NHẤT về wall-clock.** Nó có ít hơn
   ResNet18 **4,4 lần FLOPs** nhưng chậm hơn **14 lần** trên CPU, và chậm hơn M2 **68 lần**.
   → Nguyên nhân: MBConv + squeeze-excitation gồm rất nhiều lớp **mảnh**; mỗi lớp tốn chi
   phí cố định (launch kernel, độ trễ bộ nhớ) mà làm rất ít phép tính → bị chặn bởi
   **băng thông bộ nhớ**, không bởi năng lực tính toán. SiLU cũng đắt hơn ReLU.
   → Và nhóm đo được **cả hai chiều**: ở nhóm model nhỏ (M1, M2 ở 48×48) thì model **lớn**
   hiệu quả hơn trên mỗi FLOP vì tensor quá nhỏ không lấp đầy phần cứng; ở nhóm 224×224
   thì conv **dày đặc** (ResNet18) hiệu quả hơn hẳn conv **depthwise** (MobileNet, EffNet).
   → **Kết luận: muốn nói về triển khai thì PHẢI đo wall-clock trên thiết bị đích.**
   Chọn model theo FLOPs sẽ dẫn tới EfficientNet-B0 — model **chậm nhất** trong cả 5.

9. ★ **"Đo latency thế nào cho đúng?"**
   → `eval()` + `no_grad()`; ≥20 vòng **warm-up**; ★ **đồng bộ thiết bị TRƯỚC VÀ SAU khi
   bấm giờ** — GPU chạy **bất đồng bộ**, `model(x)` trả về ngay khi xếp hàng xong chứ chưa
   tính xong, không sync là đo thời gian **gửi lệnh**, sai hàng chục lần; 100 vòng đo, báo
   **p50 và p95** (hệ thống thời gian thực quan tâm trường hợp xấu); batch 1 **và** 64;
   ghi rõ thiết bị và phiên bản thư viện.

10. **"Nếu phải gắn lên xe thật thì chọn model nào?"**
    → Quyết định trên **ba trục**: (1) accuracy **kèm p-value McNemar** — nếu không có ý
    nghĩa thống kê thì coi như bằng nhau; (2) **latency p95** ở batch=1 trên thiết bị đích;
    (3) **relative robustness** dưới sương/tối/mờ — vì đó mới là điều kiện vận hành thật.
    Trả thêm 3 lần latency để lấy 0,1 điểm accuracy không có ý nghĩa là lựa chọn tồi.

---

# HUY — Dữ liệu + Rò rỉ + Ablation dữ liệu + Robustness

> **Phần đặc tả rõ ràng nhất, ít bẫy kỹ thuật nhất.** Nhưng bạn sở hữu **hai phát hiện
> mạnh nhất của cả nhóm**: bằng chứng rò rỉ dữ liệu, và nghịch lý robustness.

### Sở hữu
```
src/gtsrb/data/download.py          src/gtsrb/robustness/corruptions.py  ★
src/gtsrb/data/preprocess.py        src/gtsrb/robustness/benchmark.py
src/gtsrb/data/split.py        ★    scripts/run_robustness.py
src/gtsrb/data/transforms.py        tests/test_split.py        ★
scripts/prepare_data.py             tests/test_corruptions.py
notebooks/01, 07
```

### Việc cụ thể — ước tính 10 giờ

| # | Việc | Lệnh | Compute |
|---|---|---|---|
| 1 | Đọc `split.py` + `preprocess.py` + `corruptions.py` | — | 2 giờ |
| 2 | Chạy đường ống dữ liệu | `make data` | 6 phút |
| 3 | ★ **Thí nghiệm rò rỉ**: train M1 trên split random, so với split đúng | `make data-leaky && make train-m1` rồi `make data` lại | 25 phút |
| 4 | Ablation **augmentation** (4 policy) | `--axes augmentation --budget` | ~50 phút |
| 5 | Ablation **preprocess** (4 chế độ) | `--axes preprocess --budget` | ~60 phút (+ dựng 3 cache) |
| 6 | **Robustness** 3 model × 5 nhiễu × 5 mức | `make robustness` | ~45 phút |
| 7 | Chạy `notebooks/01` và `07`, viết nhận xét | — | 2 giờ |
| 8 | 10 Q&A | — | 2 giờ |

### Hai phát hiện bạn sở hữu

**Phát hiện 1 — rò rỉ dữ liệu do cấu trúc track** *(số thật)*

| Cách split | Track dùng chung giữa train và val |
|---|---|
| Random theo ảnh (**sai**) | **1.306** |
| Theo track `StratifiedGroupKFold` (**đúng**) | **0** |

**Phát hiện 2 — occlusion phá nặng nhất** *(relative robustness của M1)*

| Nhiễu | relative = acc_nhiễu / acc_sạch |
|---|---|
| Nhiễu Gaussian | 0,908 |
| Mờ chuyển động | 0,854 |
| Thiếu sáng | 0,782 |
| Sương mù | 0,725 |
| **Bị che** | **0,511** ← phá nặng nhất |

M1 mất gần **một nửa** accuracy khi bị che. Giả thuyết: M1 dùng `Flatten + FC` nên phụ
thuộc **vị trí tuyệt đối** — che một vùng là phá hỏng đúng những input của lớp FC.
M2 dùng **Global Average Pooling** nên không phụ thuộc vị trí → **dự đoán M2 sẽ bền hơn
trước occlusion**. Chạy `make robustness` cho cả 3 model để kiểm chứng. Nếu đúng, đây là
liên kết đẹp giữa **lựa chọn kiến trúc** và **tính bền**.

### Bạn phải trả lời được — 10 câu

1. ★ **"Vì sao không split train/val random?"**
   → GTSRB quay mỗi biển báo **vật lý** thành một track **30 frame liên tiếp**
   (`000{track}_000{frame}.ppm`). Split random làm frame 12 và 13 của **cùng tấm biển** rơi
   vào cả train và val — chúng gần như là hai bản sao. Model không cần **khái quát**, chỉ cần
   **nhớ** → **rò rỉ dữ liệu**, val accuracy bị thổi phồng.
   → Nhóm dùng `StratifiedGroupKFold` với `groups = (class_id, track_id)`, và đo **cả hai**
   cách: split random cho **1.306** track dùng chung, split đúng cho **0**.
   → Dấu hiệu nhận biết rò rỉ: **val cao hơn test một cách bất thường**.

2. ★ **"Vì sao phải kèm `class_id` vào nhóm, không chỉ `track_id`?"**
   → Vì `track_id` được đánh số lại **trong mỗi thư mục lớp** — track 5 của lớp 2 khác track 5
   của lớp 7. Chỉ nhóm theo `track_id` sẽ gộp nhầm các track của những lớp khác nhau.

3. **"Histogram equalization làm gì? Công thức?"**
   → Trải lại phân phối độ sáng cho đều: tính histogram, lấy CDF, dùng CDF làm hàm map
   `s = (L−1)·F(r)` với L = 256. GTSRB chụp ngoài đường, điều kiện sáng rất khác nhau.

4. ★ **"CLAHE khác HE thường chỗ nào?"**
   → HE toàn cục dùng **một** CDF cho cả ảnh, nên vùng gần phẳng (trời, mặt đường) có
   histogram rất hẹp và bị kéo giãn cực mạnh → **nhiễu bị phóng đại**. CLAHE sửa ba chỗ:
   (a) chia ảnh thành ô 8×8, equalize **từng ô**; (b) **chặn trần** histogram ở
   `clipLimit = 2,0` rồi phân phối lại phần bị cắt; (c) nội suy song tuyến tính giữa các ô
   để không thấy đường ranh.

5. **"Vì sao equalize trên kênh L/Y chứ không trên từng kênh RGB?"**
   → Equalize R, G, B độc lập làm lệch tỉ lệ giữa ba kênh → **ảnh đổi màu**. Mà màu biển báo
   **mang nghĩa**: đỏ = cấm, xanh = bắt buộc. LAB/YUV tách độ sáng (L/Y) khỏi màu → sửa sáng,
   giữ màu.

6. ★ **"Vì sao không dùng horizontal flip? Augmentation ảnh thường hay dùng mà."** *(câu bẫy)*
   → Biển báo **bất đối xứng có nghĩa**. Lật ngang lớp 33 "rẽ phải phía trước" ra **đúng hình**
   lớp 34 "rẽ trái phía trước". Các cặp bị ảnh hưởng: **33↔34, 19↔20, 36↔37, 38↔39**.
   Flip = tự tạo dữ liệu **sai nhãn**. Hue jitter cũng bị cấm vì màu mang nghĩa.
   → **Bài học khái quát: augmentation phải tôn trọng tính bất biến THẬT của bài toán.**
   `tests/test_transforms.py` khoá luật này ở mức mã nguồn.

7. **"Mất cân bằng lớp 10,7:1 — xử lý thế nào?"**
   → Nhóm **không** resample, vì (a) phân phối này phản ánh tần suất biển báo thật trên đường;
   (b) nhóm can thiệp ở **chỉ số** (dùng macro-F1) thay vì ở **dữ liệu** — chỗ can thiệp đúng
   hơn; (c) vẫn stratify theo lớp khi split (lệch tối đa **0,297%**).

8. ★ **"Vì sao không train luôn trên các loại nhiễu đó cho model bền hơn?"** *(câu bẫy)*
   → Vì khi đó **phép đo mất ý nghĩa**. Robustness nghĩa là khái quát sang **phân phối chưa
   từng thấy**. Train trên chúng rồi test trên chúng là test **in-distribution** — chỉ chứng
   minh "model học được cái nó đã thấy". Đó là **augmentation**, không phải robustness.
   → `tests/test_corruptions.py` có test khẳng định `data/transforms.py` **không** import
   `robustness/corruptions.py`.

9. ★ **"Model accuracy cao nhất có phải model bền nhất?"**
   → **Không chắc — và nhóm báo cáo `relative = acc_nhiễu / acc_sạch` chính vì vậy.** Nếu chỉ
   báo accuracy tuyệt đối dưới nhiễu thì model nào giỏi sẵn sẽ luôn thắng, và ta không học
   được gì về **tính bền**. Giả thuyết của nhóm: model dùng **GAP** (M2) bền hơn trước
   **occlusion** vì nó không phụ thuộc vị trí tuyệt đối; M1 dùng Flatten+FC nên mất tới 49%
   accuracy khi bị che.

10. **"Mean/std chuẩn hoá tính trên tập nào? Tập test dùng mấy lần?"**
    → Mean/std tính **chỉ trên tập train** (tính trên cả bộ là rò rỉ thống kê của val/test vào
    huấn luyện). Số thật: `mean = [0,4597, 0,4254, 0,4376]`, `std = [0,2670, 0,2607, 0,2691]`.
    → Tập test chính thức (12.630 ảnh) bị **niêm phong**, chỉ mở **một lần ở cuối**. Mọi quyết
    định chỉ dựa trên val. Nhìn test rồi quay lại sửa model thì test đã thành val.

---

## Quy ước chung

```bash
git switch -c feat/huy-data        # hoặc feat/hoang-m2, feat/phongtran-m3, feat/phong-eval
pip install nbstripout && nbstripout --install   # xoá output notebook trước khi commit
make test                                        # 95 test, PHẢI xanh trước khi push
```

**Notebook là JSON** nên merge rất khó → **mỗi người một notebook riêng**, tuyệt đối không
hai người sửa cùng một `.ipynb`.

**Gặp lỗi lạ → ghi 1 dòng vào [SU_CO.md](SU_CO.md)** (triệu chứng → nguyên nhân → cách sửa).
File đó đã có **6 nhóm sự cố đã gặp thật** — đọc trước khi code tiết kiệm được vài giờ.

**Ngày cuối: diễn tập chất vấn chéo.** Mỗi người bị 3 người kia hỏi 10 câu về phần mình.
Ai trả lời không được thì quay lại học — **không đẩy cho người khác trả lời hộ**.
Mục tiêu: hiểu **sâu** phần của mình và **đủ** phần của 3 người kia.
