# Câu hỏi giảng viên — phần hỏi chéo toàn dự án

> **38 câu hỏi riêng của từng người** nằm trong [PHAN_CONG.md](PHAN_CONG.md):
> Phong Nguyễn 8 câu · Hoàng 10 câu · Phong Trần 10 câu · Huy 10 câu.
> File này là **những câu hỏi không thuộc riêng ai** — về thiết kế thực nghiệm, về kết luận,
> và những **câu bẫy**. Cả 4 người phải trả lời được những câu ở đây.
>
> Quy ước: ✱ = câu bẫy, trả lời sai là mất điểm nặng.

---

## A. Câu hỏi mở đầu (gần như chắc chắn bị hỏi)

**Q1. "Tóm tắt dự án trong 2 phút."**

Nhóm phân loại 43 lớp biển báo giao thông Đức trên bộ GTSRB — 39.209 ảnh train,
12.630 ảnh test chính thức. Nhóm xây 3 mô hình: M1 LeNet từ số 0 làm mốc tham chiếu,
M2 VGG-like sâu từ số 0 có BatchNorm, residual và spatial dropout, M3 transfer learning
từ ba backbone pretrained ImageNet (ResNet18, MobileNetV2, EfficientNet-B0).
Cả ba dùng **cùng** split, **cùng** seed, **cùng** ngân sách augmentation.

Điểm nhóm muốn nhấn: GTSRB đã bão hoà — con người 98,84%, nhà vô địch IJCNN 2011 đạt 99,46%,
hiện nay ~99,7%. Nên nhóm không coi accuracy là câu hỏi chính. Nhóm tập trung vào bốn thứ:
split không rò rỉ dữ liệu, ablation một-biến-một-lần có kiểm định thống kê, robustness dưới
nhiễu thực tế, và đánh đổi accuracy–latency cho triển khai biên.

**Q2. ✱ "Model nào tốt nhất?"** — ★ CÂU QUAN TRỌNG NHẤT, SỐ ĐO THẬT 5 MODEL

*Trả lời sai*: "M3 tốt nhất vì nó pretrained."

*Trả lời đúng*: Có **hai** phát hiện phá vỡ kỳ vọng thông thường.

| Model | Tham số | Input | Top-1 | Macro-F1 | ECE | p95 CPU | Train |
|---|---|---|---|---|---|---|---|
| M3 ResNet18 | 11,20 M | 224² | **99,33%** | **0,9903** | 0,1085 | 18,3 ms | 61 ph |
| **M2 VGG-res** | **1,24 M** | 48² | 99,26% | 0,9880 | **0,0535** | **3,7 ms** | 24 ph |
| M3 EfficientNet-B0 | 4,06 M | 224² | 98,77% | 0,9838 | 0,0973 | **256,6 ms** | 132 ph |
| M3 MobileNetV2 | 2,28 M | 224² | 98,73% | 0,9796 | 0,1076 | 49,0 ms | 55 ph |
| M1 LeNet | 2,42 M | 48² | 98,46% | 0,9778 | 0,1310 | **1,1 ms** | 12 ph |

**★ Phát hiện 1 — M2 (tự xây, 1,24M tham số) ĐÁNH BẠI hai backbone pretrained ImageNet,
có ý nghĩa thống kê:**

| Cặp | n₀₁ | n₁₀ | p-value | Kết luận |
|---|---|---|---|---|
| M2 vs EfficientNet-B0 | 106 | 44 | **6,3e-07** | **M2 hơn** |
| M2 vs MobileNetV2 | 115 | 49 | **3,9e-07** | **M2 hơn** |
| M2 vs ResNet18 | 51 | 60 | 0,4477 | **tương đương** |
| EfficientNet-B0 vs MobileNetV2 | 68 | 64 | 0,7940 | tương đương |

Nghĩa là: **"transfer learning luôn thắng" là SAI.** M2 nhỏ hơn EfficientNet-B0 **3,3 lần**
và nhanh hơn **68 lần**, mà vẫn **chính xác hơn có ý nghĩa thống kê**.
Lý do khả dĩ: ảnh GTSRB trung vị chỉ **43×43 px**; upsample lên 224 không thêm thông tin
nào, nên lợi thế pretrained bị triệt tiêu, còn chi phí thì vẫn phải trả đủ.

**★ Phát hiện 2 — "EfficientNet" là model KÉM HIỆU QUẢ NHẤT về wall-clock:**

| Model | FLOPs | Latency p50 (CPU, bs=1) | **ms/GFLOP** |
|---|---|---|---|
| ResNet18 | 3,65 G | 16,7 ms | **4,6** |
| M2 VGG-res | 0,29 G | 3,2 ms | 11,0 |
| M1 LeNet | 0,07 G | 0,9 ms | 12,0 |
| MobileNetV2 | 0,65 G | 47,9 ms | 73,4 |
| **EfficientNet-B0** | 0,83 G | **243,4 ms** | **294,1** |

Chênh **64 lần**. EfficientNet-B0 có **ít hơn ResNet18 4,4 lần FLOPs** nhưng chậm hơn
**14 lần** trên CPU. Nguyên nhân: MBConv + squeeze-excitation gồm rất nhiều lớp mảnh,
mỗi lớp tốn chi phí cố định (launch kernel, truy cập bộ nhớ) mà làm rất ít phép tính —
bị chặn bởi **băng thông bộ nhớ**, không bởi năng lực tính toán. Thêm nữa SiLU đắt hơn ReLU.

→ **Kết luận triển khai: chọn M2.** Tương đương ResNet18 về accuracy (p = 0,4477) nhưng
nhanh hơn **4,9 lần** và nhẹ hơn **9 lần**; đồng thời **chính xác hơn** cả hai backbone
"nhẹ" kia trong khi nhanh hơn chúng 13–68 lần.

**Vì sao đây là kết quả đáng giá:** nếu nhóm chỉ nhìn bảng accuracy, kết luận sẽ là
"chọn ResNet18". Nếu nhóm chỉ nhìn FLOPs, kết luận sẽ là "chọn EfficientNet-B0" — model
**chậm nhất** trong cả 5. Chính McNemar và phép đo wall-clock ngăn cả hai kết luận sai đó.

**Q3. "Vì sao chọn đúng ba model đó?"**

Ba model tạo thành một **thang đo có kiểm soát**, không phải ba model ngẫu nhiên:
- M1 → M2: cùng train từ số 0, cùng dữ liệu. Chênh lệch **chỉ** đến từ kiến trúc
  (độ sâu, BN, residual, spatial dropout, GAP). Trả lời được "kiến trúc đáng giá bao nhiêu".
- M2 → M3: cùng được huấn luyện tốt. Chênh lệch **chỉ** đến từ việc có hay không
  tri thức pretrained. Trả lời được "1,28 triệu ảnh ImageNet đáng giá bao nhiêu".
- Trong M3, ba backbone có ba triết lý kiến trúc khác nhau (residual dày đặc / depthwise
  separable / compound scaling + SE), cho phép so sánh accuracy–cost.

---

## B. Thiết kế thực nghiệm

**Q4. ✱ "Làm sao biết so sánh của nhóm là công bằng?"**

Nhóm cố định mọi thứ ngoại trừ biến đang xét:
cùng `index.csv` và cùng split (track-aware, seed 42); cùng ngân sách augmentation;
cùng loss (label smoothing 0,1), cùng optimizer family, cùng scheduler; **cùng một hàm
`fit()`** cho cả ba model (C không được viết vòng train riêng — xem `docs/INTERFACE.md`);
cùng tập test niêm phong. Mọi run ghi `result.json` kèm git commit và config thực tế,
nên bất kỳ số nào trong báo cáo cũng truy được về một run cụ thể.

Có **hai** thứ buộc phải khác giữa M1/M2 và M3, và nhóm nêu rõ:
input 48 vs 224, và mean/std GTSRB vs ImageNet. Cả hai đều có lý do kỹ thuật
(downsample 32× và phân phối pretrained — xem `LY_THUYET.md` 4.2 và 4.5), và cả hai đều
được định lượng bằng ablation resolution.

**Q5. "Vì sao ablation phải đổi một biến một lần?"**

Đổi hai biến cùng lúc thì không quy được kết quả về nguyên nhân nào, và còn có thể bị
**triệt tiêu**: biến thứ nhất +0,3 điểm, biến thứ hai −0,3 điểm, tổng ra 0 và ta kết luận
sai rằng "cả hai đều vô dụng". Nhóm có 5 trục ablation, mỗi cấu hình là một file YAML riêng
trong `configs/ablation/`, chạy bằng script nên không thể lẫn.

**Q6. "Nhóm chạy mấy seed? Vì sao quan trọng?"**

`[điền số thực tế]`. Một seed duy nhất thì không biết chênh lệch 0,1 điểm là do thay đổi
mình vừa làm hay chỉ là nhiễu khởi tạo. Nhóm báo mean ± std trên `[n]` seed cho các so sánh
quan trọng, và với những so sánh chỉ chạy được 1 seed thì nhóm nói rõ là **không** kết luận
về chênh lệch nhỏ hơn độ nhiễu seed đó.

**Q7. ✱ "Tập test dùng mấy lần?"**

**Một lần**, ở cuối. Mọi quyết định — chọn kiến trúc, siêu tham số, early stopping, chọn
cấu hình ablation thắng — chỉ dựa trên tập **val**. Nếu nhìn test rồi quay lại sửa model thì
test đã trở thành val, và con số báo cáo không còn là ước lượng không chệch của hiệu năng
trên dữ liệu mới. (Phần robustness cũng chạy trên test, nhưng nó là phép **đo** sau khi
model đã đóng băng, không phải căn cứ để chọn model.)

---

## C. Câu bẫy

**Q8. ✱ "Accuracy 99,5%, nghĩa là model này dùng được trên xe thật rồi?"**

Chưa. Ba lý do:
1. **Phân phối test giống phân phối train.** GTSRB test cũng chụp ở Đức, cùng loại camera,
   cùng loại biển. Biển báo Việt Nam khác, camera khác, điều kiện thời tiết khác.
2. **Robustness tụt rất nhanh.** Phần đo của nhóm cho thấy accuracy còn `[số]`% dưới
   sương mù mức 4 và `[số]`% khi bị che 30%. Đó mới là điều kiện vận hành thật.
3. **Đây là bài phân loại, không phải phát hiện.** Model nhận đầu vào là một biển báo
   **đã được cắt sẵn**. Hệ thống thật phải tự tìm biển báo trong khung ảnh toàn cảnh
   (detection) trước, và lỗi của bước đó sẽ cộng dồn vào.

**Q9. ✱ "Top-5 accuracy của nhóm 99,9%, tốt quá nhỉ?"**

Con số đó **không có nhiều ý nghĩa ở đây**. Top-5 ra đời cho ImageNet 1000 lớp. Trên 43 lớp,
top-5 nghĩa là "đúng trong 11,6% số lớp" — mọi model tử tế đều đạt ~99,9%, chỉ số bão hoà
nên không phân biệt được gì. Nhóm báo cáo vì đề bài yêu cầu, nhưng kết luận dựa trên
top-1 và macro-F1.

**Q10. ✱ "Val accuracy của nhóm 99,6% mà test 99,3%, sao val lại cao hơn?"**

Chênh 0,3 điểm là bình thường (val nhỏ hơn test nên phương sai lớn hơn, và checkpoint được
*chọn* theo val nên có một chút bias chọn lọc). Điều **đáng lo** là khi val cao hơn test
*rất nhiều* — đó là dấu hiệu rò rỉ dữ liệu. Nhóm đã chặn nguyên nhân chính bằng split
theo track, và có số liệu đối chứng: với split random thì val cao hơn `[số]` điểm so với
split đúng, đúng như dự đoán về rò rỉ.

**Q11. ✱ "Sao không dùng augmentation mạnh hơn cho tốt hơn?"**

Augmentation không phải "càng nhiều càng tốt" — nó phải **tôn trọng tính bất biến thật**
của bài toán. Horizontal flip sẽ biến lớp 33 "rẽ phải" thành đúng hình lớp 34 "rẽ trái"
(các cặp bị ảnh hưởng: 33/34, 19/20, 36/37, 38/39) — đó là tạo dữ liệu **sai nhãn**.
Hue jitter sẽ biến đỏ thành xanh, tức biến "cấm" thành "bắt buộc". Nhóm có ablation 4 mức
augmentation và `[điền: mức nào thắng]` — số liệu cho thấy tăng augmentation
`[có/không]` tiếp tục giúp sau mức đó.

**Q12. ✱ "Transfer learning có luôn tốt hơn train from scratch không?"**

**Không — và nhóm có bằng chứng thống kê ngược lại.** M2 tự xây từ số 0 (1,24M tham số,
48×48) **đánh bại** cả MobileNetV2 (p = 3,9e-07) lẫn EfficientNet-B0 (p = 6,3e-07) —
hai backbone pretrained trên 1,28 triệu ảnh ImageNet — và **tương đương** ResNet18
(p = 0,4477).

**Vì sao.** Lợi thế của pretrained là tri thức từ 1,28 triệu ảnh ở độ phân giải cao.
Nhưng ảnh GTSRB có trung vị chỉ **43×43 pixel**. Upsample lên 224×224 **không tạo ra
thông tin nào** — nó chỉ để khớp stride 32× của mạng pretrained. Nên phần tri thức
về chi tiết tinh của ImageNet không có gì để bám vào, trong khi chi phí (tham số,
FLOPs, latency, thời gian train) vẫn phải trả đủ.

**Phát biểu cho đúng:** transfer learning hiệu quả khi **dữ liệu đích ít** và **miền
đích gần miền nguồn về độ phân giải và loại đặc trưng**. GTSRB có 39 nghìn ảnh — không
quá ít — và độ phân giải rất thấp. Hai điều kiện đều không thuận lợi.

**Nói thêm nếu được hỏi sâu:** nhóm chưa kiểm được giả thuyết này một cách cô lập, vì
M2 và M3 khác nhau ở **hai** biến cùng lúc (pretrained hay không, và 48 hay 224). Trục
ablation resolution của M3 (64/112/224) là bước đi đúng để tách hai biến đó ra.

**Q13. ✱ "Nhóm có overfit không? Train accuracy bao nhiêu?"**

`[điền]`. Khoảng cách train–val là `[số]` điểm. Nhóm chống overfitting bằng 5 lớp:
spatial dropout + dropout, BatchNorm (regularize nhẹ), augmentation, label smoothing,
và early stopping theo val macro-F1. Nhưng điểm quan trọng hơn: **overfit theo nghĩa
"model nhớ dữ liệu" khác với "pipeline bị rò rỉ"**. Lớp bảo vệ mạnh nhất của nhóm không
phải dropout mà là split theo track — nó chặn loại rò rỉ mà dropout không cứu được, và
nhóm có test tự động chứng minh giao track_id giữa train và val là rỗng.

---

## D. Câu hỏi khó / nâng cao

**Q14. "Nếu cho thêm 1 tuần, nhóm làm gì tiếp?"**

Theo thứ tự giá trị trên mỗi giờ công:
1. **Thêm seed** cho các so sánh chính — hiện tại nhiều kết luận chỉ dựa trên `[n]` seed.
2. **Test-time augmentation + ensemble** — committee of CNNs chính là cách IDSIA thắng
   IJCNN 2011; nó gần như chắc chắn cải thiện và cho một mốc so sánh lịch sử rõ ràng.
3. **Adversarial robustness (FGSM/PGD)** — khác về bản chất với nhiễu tự nhiên, và liên quan
   trực tiếp tới an toàn giao thông (sticker dán lên biển báo đã được chứng minh có thể
   đánh lừa model thật).
4. **Quantization int8 + export ONNX/CoreML** và đo lại latency — làm phần triển khai biên
   thành kết luận thực sự thay vì chỉ là so sánh tương đối.
5. **Kiểm tra khái quát sang bộ khác** (ví dụ biển báo Việt Nam hoặc BTSD của Bỉ) — câu hỏi
   khoa học thú vị nhất còn lại, và là câu trả lời thẳng cho Q8.

**Q15. "Vì sao không dùng Vision Transformer?"**

ViT cần dữ liệu rất lớn hoặc pretrain mạnh để thắng CNN; với 39 nghìn ảnh 48×48 thì
inductive bias của CNN (cục bộ + bất biến dịch chuyển) đúng là thứ ta muốn, còn ViT phải
*học* nó từ dữ liệu. Thêm nữa đề bài yêu cầu một model "from scratch đơn giản" và một
"from scratch phức tạp" — ViT from scratch trên 39 nghìn ảnh gần như chắc chắn kém hơn M2.
Nếu có thời gian, cách làm đúng là thêm một ViT/DeiT **pretrained** vào nhóm M3 để so sánh
công bằng về mặt "đều có pretrained".

**Q16. "Nếu dữ liệu chỉ có 1/10 (3.900 ảnh) thì kết luận có đổi không?"**

Nhóm dự đoán khoảng cách M3 − M2 **nở rộng rõ rệt**: transfer learning có lợi thế lớn nhất
khi dữ liệu đích ít, vì phần lớn đặc trưng đã có sẵn, chỉ cần học lớp cuối. M1 và M2 train
từ số 0 sẽ overfit nặng. Đây cũng là một thực nghiệm đáng làm (learning curve: accuracy theo
% dữ liệu train) và nhóm `[đã/chưa]` làm.

**Q17. "Nhóm xử lý thế nào nếu một lớp chỉ có 5 ảnh?"**

Thứ tự: (1) macro-F1 sẽ bộc lộ ngay vấn đề, không che được; (2) `class_weight` trong loss
hoặc oversampling lớp đó; (3) augmentation mạnh hơn **riêng** cho lớp đó; (4) nếu vẫn không
đủ thì bài toán chuyển sang few-shot — dùng metric learning / prototypical network thay vì
classifier softmax. Trong GTSRB lớp thưa nhất là 210 ảnh nên chưa tới mức đó.

**Q18. "Grad-CAM của nhóm có vùng sáng đúng vào biển báo. Vậy chứng minh được model hoạt
động đúng chưa?"**

Chưa. Grad-CAM cho biết **ở đâu**, không cho biết **model dùng đặc trưng gì ở đó** và
**không phải bằng chứng nhân quả** — model có thể nhìn đúng chỗ mà vẫn suy luận sai (nhóm
có `[n]` ca sai mà heatmap vẫn nằm trên biển báo, đã phân tích trong `notebooks/08`).

Thêm nữa, **độ phân giải** heatmap rất thô, và số đo thật cho kết quả phản trực giác:
M2 chỉ có feature map **3×3** (mỗi ô = 16×16 pixel, cả ảnh 9 ô) trong khi M1 có **12×12**
(mỗi ô = 4×4 pixel) và M3 có **7×7**. Tức **M2 — model thiết kế công phu nhất — cho
heatmap THÔ NHẤT**, vì nó có 4 lớp MaxPool so với 2 của M1. Nói "đúng vào biển báo"
với M2 ở độ phân giải đó là một phát biểu rất lỏng.

**Q19. "Nếu em đổi một pixel, model có đổi dự đoán không?"**

Về nguyên tắc là có thể — đó là adversarial example, và các CNN chuẩn đã được chứng minh
rất dễ bị. Nhóm **chưa** đo phần này (ghi trong hướng mở rộng Q14), nên nhóm không khẳng
định gì về adversarial robustness. Những gì nhóm đo được là robustness trước **nhiễu tự
nhiên** (mờ, nhiễu, sương, tối, che) — một loại distribution shift khác về bản chất:
nhiễu tự nhiên không được tối ưu để đánh lừa model, còn adversarial perturbation thì có.

**Q20. "Số liệu của nhóm có tái lập được không? Chứng minh."**

Có. Mỗi run sinh `artifacts/runs/<run_id>/` gồm `best.pt`, `train_log.csv`,
`config.yaml` (bản sao cấu hình **thực tế đã dùng**, không phải bản gốc), và `result.json`
ghi cả `seed` và `git_commit`. Mọi siêu tham số nằm trong YAML, không hard-code trong code.
Seed cố định cho `torch`, `numpy`, `random`, và bật `cudnn.deterministic`. Nhóm đã kiểm:
chạy lại cùng seed cho cùng val macro-F1 trong sai số < 0,001.
Mọi số trong báo cáo đều do script sinh, không ai chép tay.

---

## E. Phần phải nhớ dạng số (học thuộc)

| Hỏi | Đáp |
|---|---|
| GTSRB có bao nhiêu ảnh? | 39.209 train + 12.630 test = **51.839**, 43 lớp |
| Split theo track cho ra bao nhiêu? | train **31.379** / val **7.830** / test **12.630** (0 track dùng chung) |
| Mất cân bằng lớp bao nhiêu? | ~10,7:1 (lớp 2: 2.250 ảnh; lớp 0/19/37: 210 ảnh) |
| Mỗi track bao nhiêu frame? | 30 frame của **cùng một biển báo vật lý** |
| Con người đạt bao nhiêu trên GTSRB? | 98,84% |
| SOTA lịch sử? | 99,46% (IDSIA, committee of CNNs, IJCNN 2011); nay ~99,7% |
| Vì sao 48×48? | khớp cấu hình bài vô địch IJCNN 2011; đủ để đọc chữ số trên biển giới hạn tốc độ |
| Receptive field của n lớp conv 3×3 stride 1? | `2n + 1` |
| Hai conv 3×3 so với một conv 5×5 về tham số? | `18C²` so với `25C²` → giảm 28%, và có thêm 1 tầng phi tuyến |
| Label smoothing ε=0,1, K=43 → nhãn lớp đúng? | `0,9 + 0,1/43 ≈ 0,9023`; lớp sai `≈ 0,00233` |
| ResNet18 / MobileNetV2 / EfficientNet-B0 — #params? | 11,7M / 3,5M / 5,3M |
| FLOPs @224 của ba cái đó? | ~1,82G / ~0,30G / ~0,39G |
| Ba backbone downsample bao nhiêu lần? | **32×** → đó là lý do phải upsample lên 224 (224/32 = 7) |
| Mean/std ImageNet? | mean (0,485; 0,456; 0,406), std (0,229; 0,224; 0,225) |
| Cặp lớp nào bị ảnh hưởng nếu flip ngang? | 33↔34, 19↔20, 36↔37, 38↔39 |
| M1 bao nhiêu tham số, nằm ở đâu? | **2.424.299**, trong đó **2.359.552 = 97,3%** ở MỘT lớp FC (9216×256) |
| M2 bao nhiêu tham số? | **1.237.451** — **ÍT HƠN M1** dù sâu gấp 4 lần (nhờ GlobalAvgPool) |
| Tại sao early stopping theo macro-F1? | mất cân bằng 10,7:1 — accuracy bị lớp đông chi phối |
| Chênh 0,2 điểm top-1 trên test = mấy ảnh? | ≈ 25 ảnh trên 12.630 → phải kiểm định McNemar |
| McNemar dùng ô nào của bảng 2×2? | chỉ hai ô **bất đồng** `n₀₁`, `n₁₀` |
| Khi nào dùng binomial chính xác thay χ²? | khi `n₀₁ + n₁₀ < 25` |

---

## F. Checklist diễn tập ngày cuối

- [ ] Mỗi người trả lời trơn các câu của mình trong `PHAN_CONG.md` **không nhìn giấy**
- [ ] Mỗi người trả lời được Q1–Q3 (mở đầu) và Q8–Q13 (câu bẫy) của file này
- [ ] Cả nhóm thuộc bảng số ở mục E
- [ ] Mỗi người giải thích được **một** khái niệm của người khác ở mức cơ bản:
      Phong Nguyễn hiểu residual là gì · Hoàng hiểu rò rỉ track là gì ·
      Phong Trần hiểu macro-F1 là gì · Huy hiểu discriminative LR là gì
- [ ] Mọi `[điền]` trong file này và trong `KET_QUA.md` đã có số thật
- [ ] Không ai nói "cái đó bạn khác làm" khi bị hỏi
