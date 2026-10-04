# Plan: So sánh 3 mô hình Deep Learning phân loại biển báo giao thông GTSRB (43 lớp)

**Nguồn yêu cầu**: đề bài Project 2 (free-form, không có PRD)
**Framework**: PyTorch 2.x (+ torchvision)
**Compute**: lai — code/debug local trên M1 Pro (MPS), train nặng trên Google Colab (T4)
**Định dạng sản phẩm**: `src/` (thư viện dùng chung) + `notebooks/` (giải thích) + `docs/` (lý thuyết & Q&A)
**Quy mô nhóm**: 4 người (A, B, C, D)
**Độ phức tạp**: LARGE
**Thời lượng mặc định**: 14 ngày (có thể nén còn 10)

---

## 1. Tóm tắt

Xây 3 mô hình phân loại 43 lớp biển báo GTSRB (39.209 ảnh train + 12.630 ảnh test chính thức)
và so sánh chúng một cách **có kỷ luật khoa học**: cùng data split, cùng seed, cùng ngân sách
augmentation, chỉ thay đổi một biến mỗi lần.

- **M1** — LeNet-style từ số 0 (2 block Conv-Pool + 2 Dense) → mốc tham chiếu (baseline).
- **M2** — VGG-like sâu từ số 0: stack conv 3×3 + BatchNorm + residual + Dropout + SpatialDropout.
- **M3** — Transfer learning: ResNet18 / MobileNetV2 / EfficientNet-B0 pretrained ImageNet,
  fine-tune với discriminative learning rate, ảnh upsample lên 224×224.

Điểm khác biệt của bài làm này so với một bài GTSRB thông thường: GTSRB hiện đã **bão hoà**
(SOTA ~99,7%, con người 98,84%, nhà vô địch IJCNN 2011 — committee of CNNs của IDSIA — đạt 99,46%).
Vì vậy giá trị học thuật **không nằm ở việc đua accuracy**, mà nằm ở 4 thứ:

1. Split **không rò rỉ dữ liệu** (track-aware) — đa số bài sinh viên làm sai chỗ này.
2. Ablation **một-biến-một-lần** có kiểm định thống kê.
3. Robustness dưới nhiễu thực tế (mưa/sương/mờ/tối/che khuất).
4. Pareto accuracy–latency cho triển khai biên (edge).

---

## 2. Pattern Grounding

Repo hiện **trống** (`/Users/nguyentuanphong/Downloads/DLprj` không có file nào).
**Không có pattern code sẵn để noi theo** — ta tự lập chuẩn ở `docs/INTERFACE.md`
và mọi người bắt buộc tuân thủ. Đây là quyết định có chủ ý: contract phải chốt TRƯỚC khi
4 người code song song, nếu không sẽ phải viết lại.

| Hạng mục | Chuẩn tự lập | Nơi ghi |
|---|---|---|
| Đặt tên | `snake_case` cho hàm/file, `PascalCase` cho class, model id dạng `m{1,2,3}_<tên>` | `docs/INTERFACE.md` |
| Xử lý lỗi | Raise `ValueError`/`FileNotFoundError` kèm thông điệp nêu rõ giá trị sai; không `except: pass` | `docs/INTERFACE.md` |
| Logging | `logging` chuẩn + 1 file `train_log.csv` mỗi run; không dùng `print` trong `src/` | `src/gtsrb/utils/logging.py` |
| Truy cập dữ liệu | Mọi thứ đi qua `GTSRBDataset` + `build_dataloaders(cfg)`; không ai đọc ảnh trực tiếp | `src/gtsrb/data/dataset.py` |
| Test | `pytest`, file `tests/test_<module>.py`, bắt buộc có test cho split (chống leakage) và corruption | `tests/` |
| Cấu hình | YAML trong `configs/`, nạp bằng `utils/config.py`; **mọi** siêu tham số phải nằm trong YAML, không hard-code | `configs/base.yaml` |
| Kết quả | Mỗi run ghi `artifacts/runs/<run_id>/result.json` theo schema cố định | `docs/INTERFACE.md` |

---

## 3. Phân công tổng quát

| Người | Vai trò | Sở hữu thư mục | Câu chuyện phải kể được trước giảng viên |
|---|---|---|---|
| **A** | Data & Infrastructure Lead | `src/gtsrb/data/`, `src/gtsrb/utils/`, `notebooks/00,01` | "Dữ liệu vào model sạch và công bằng như thế nào" |
| **B** | From-Scratch Models & Training Engine | `src/gtsrb/models/m1,m2`, `src/gtsrb/engine/`, `notebooks/02,03` | "Xây mạng từ số 0 và vì sao từng khối tồn tại" |
| **C** | Transfer Learning + Explainability + Deploy | `src/gtsrb/models/m3`, `explain/`, `deploy/`, `notebooks/04,08,09` | "Dùng tri thức ImageNet, rồi mở hộp đen và đo tốc độ" |
| **D** | Evaluation, Robustness & Reporting | `src/gtsrb/eval/`, `robustness/`, `notebooks/05,06,07` | "Đo lường trung thực và kết luận có căn cứ" |

Chi tiết từng việc, deadline, và bộ câu hỏi riêng: `docs/PHAN_CONG.md`.

---

## 4. Files to Change

| File | Action | Chủ | Vì sao |
|---|---|---|---|
| `requirements.txt`, `.gitignore`, `Makefile` | CREATE | A | môi trường tái lập được |
| `configs/base.yaml`, `configs/m*.yaml`, `configs/ablation/*.yaml` | CREATE | A (base) / B,C (model) | mọi siêu tham số ra khỏi code |
| `src/gtsrb/utils/{seed,config,logging,viz}.py` | CREATE | A (3 đầu) / D (viz) | hạ tầng dùng chung |
| `src/gtsrb/data/{download,preprocess,split,dataset,transforms}.py` | CREATE | A | đường ống dữ liệu |
| `src/gtsrb/models/{registry,m1_lenet,m2_vggres}.py` | CREATE | B | 2 model from scratch |
| `src/gtsrb/models/m3_transfer.py` | CREATE | C | 3 backbone pretrained |
| `src/gtsrb/engine/{train,losses,schedulers,callbacks}.py` | CREATE | B | vòng huấn luyện dùng chung cho cả 3 model |
| `src/gtsrb/eval/{metrics,confusion,stats_tests,calibration}.py` | CREATE | D | đo lường |
| `src/gtsrb/robustness/{corruptions,benchmark}.py` | CREATE | D | 5 loại nhiễu × 5 mức |
| `src/gtsrb/explain/gradcam.py` | CREATE | C | giải thích vùng quyết định |
| `src/gtsrb/deploy/speed.py` | CREATE | C | latency / FLOPs / size |
| `scripts/{prepare_data,train,evaluate,run_ablation,run_robustness,benchmark_speed,make_gradcam}.py` | CREATE | chủ module tương ứng | chạy lại được bằng 1 dòng CLI |
| `tests/test_{split,corruptions,metrics,models}.py` | CREATE | A,D,D,B | khoá các bất biến quan trọng |
| `notebooks/00..09.ipynb` | CREATE | xem bảng trên | bản kể chuyện có giải thích tiếng Việt |
| `docs/{PHAN_CONG,INTERFACE,LY_THUYET,QA_GIANG_VIEN,KET_QUA,SLIDE_OUTLINE}.md` | CREATE | cả nhóm | tài liệu học & bảo vệ |

---

## Step 1 — Chốt kiến trúc code, interface contract và môi trường

**Chủ**: cả 4 người họp 90 phút, A chấp bút. **Ngày 0.**

Khoá cứng 5 chữ ký hàm liên module (`GTSRBDataset`, `build_model`, `fit`, `evaluate`,
`apply_corruption`, `GradCAM`) và schema `result.json`. Dựng `requirements.txt` (torch,
torchvision, opencv, scikit-learn, pyyaml, thop, pytest), `.gitignore` (bỏ `data/`,
`artifacts/`), `Makefile`. Mỗi người tạo branch riêng `feat/{a-data,b-scratch,c-transfer,d-eval}`.
A giao ngay một **tiny fake dataset** (10 ảnh/lớp, đúng interface) để B, C, D code được
mà không phải chờ dữ liệu thật.

Out of scope: chọn siêu tham số cuối cùng, tối ưu hiệu năng.

**Acceptance**: `docs/INTERFACE.md` có đủ 6 chữ ký hàm + schema JSON; `pip install -r requirements.txt` thành công trên cả macOS/MPS và Colab; `python -c "from gtsrb.data.dataset import GTSRBDataset"` chạy được với fake dataset.

---

## Step 2 — Tải GTSRB, cắt ROI, tiền xử lý (histogram equalization) và resize 48×48

**Chủ**: A. **Ngày 1–2.**

Tải bộ train (39.209 ảnh, 43 thư mục lớp, `.ppm`) và bộ test chính thức (12.630 ảnh +
`GT-final_test.csv`). Mỗi thư mục lớp có CSV kèm **ROI bounding box** (`Roi.X1..Y2`) —
cắt theo ROI với lề 10% trước khi resize, vì ảnh gốc có viền nền thừa.
Hiện thực 4 chế độ tiền xử lý để sau này ablation: `none` (RGB thô),
`he_gray` (equalize trên ảnh xám, đúng kiểu Sermanet & LeCun 2011),
`he_y` (equalize kênh Y của YUV, giữ màu), `clahe` (CLAHE clipLimit=2.0,
tile 8×8 trên kênh L của LAB — mặc định). Resize 48×48 bằng `INTER_AREA` khi thu nhỏ.
Tính mean/std **chỉ trên tập train** rồi lưu ra YAML.

Out of scope: augmentation (Step 4), split (Step 3).

**Acceptance**: `python scripts/prepare_data.py --preprocess clahe` sinh `data/processed/index.csv` đủ 51.839 dòng; 4 chế độ tiền xử lý đều chạy và xuất ảnh mẫu ra `reports/figures/preprocess_compare.png`; mean/std ghi trong `configs/base.yaml` và được tính chỉ từ split train.

---

## Step 3 — Split train/val **theo track** để chống rò rỉ dữ liệu

**Chủ**: A. **Ngày 2–3.** ★ Đây là đóng góp kỹ thuật đáng giá nhất của phần dữ liệu.

GTSRB chụp mỗi biển báo thật thành một **track 30 frame liên tiếp**; tên file
`000{track}_000{frame}.ppm`. Nếu split random theo ảnh, các frame của **cùng một
biển báo vật lý** nằm cả ở train và val → val accuracy bị thổi phồng (thực nghiệm
thường cao hơn 2–4 điểm so với split đúng). Phải dùng `StratifiedGroupKFold`
(group = `(class_id, track_id)`) để mọi frame của một track chỉ ở đúng một phía.
Giữ nguyên tỉ lệ 43 lớp. Bộ test chính thức **niêm phong**, chỉ mở 1 lần ở cuối.

Out of scope: xử lý mất cân bằng lớp bằng resampling (chỉ phân tích, không cân bằng lại —
để macro-F1 làm việc đó).

**Acceptance**: test `tests/test_split.py` chứng minh giao của tập track_id giữa train và val là rỗng; báo cáo đối chứng val-accuracy của M1 giữa split random và split theo track (số liệu đưa vào `docs/KET_QUA.md`); phân phối lớp của train/val lệch nhau < 1%.

---

## Step 4 — Đường ống augmentation + EDA

**Chủ**: A. **Ngày 3.**

4 chính sách augmentation để ablation: `none`, `geo` (rotation ±15°, translate ±10%,
zoom 0,9–1,1), `geo_photo` (geo + brightness/contrast jitter ±0,2), `geo_photo_clahe`.
**Tuyệt đối không horizontal flip** — biển báo có tính bất đối xứng: lớp 33/34
(rẽ phải / rẽ trái), 19/20 (đường cong trái / phải), 36/37, 38/39 là các cặp ảnh gương
của nhau, flip sẽ biến ảnh thành nhãn khác. Jitter hue cũng bị cấm: màu đỏ/xanh mang
nghĩa (cấm / bắt buộc). Augmentation chỉ áp dụng cho train loader.
EDA: histogram 43 lớp (mất cân bằng ~10,7:1 — lớp 2 có 2.250 ảnh, lớp 0/19/37 chỉ 210),
phân phối kích thước ảnh gốc (15×15 → 250×250), độ sáng trung bình, lưới ảnh mẫu.

Out of scope: MixUp/CutMix (ghi vào phần "hướng mở rộng").

**Acceptance**: 4 policy đều gọi được qua `cfg.aug.policy`; `reports/figures/aug_grid.png` cho thấy 8 biến thể của cùng 1 ảnh; `notebooks/01` có đủ 4 hình EDA kèm nhận xét tiếng Việt; grep toàn repo không có `RandomHorizontalFlip`.

---

## Step 5 — Training engine dùng chung: label smoothing, Adam, cosine schedule, early stopping

**Chủ**: B. **Ngày 3–4.** ★ C và D đều phụ thuộc vào step này — không được trễ.

`fit(model, loaders, cfg)` dùng chung cho cả M1, M2, M3. Thành phần:
`CrossEntropyLoss(label_smoothing=0.1)`; Adam/AdamW; `CosineAnnealingLR` kèm warmup
tuyến tính 3 epoch; mixed precision (bật trên CUDA, tắt trên MPS); gradient clipping 1,0;
early stopping theo **val macro-F1** (không theo accuracy — vì dữ liệu mất cân bằng);
lưu checkpoint tốt nhất; `train_log.csv` mỗi epoch; cố định seed
(`torch.manual_seed`, `np.random.seed`, `cudnn.deterministic`); ghi `result.json`
đúng schema ở Step 1.

Out of scope: distributed training, hyperparameter search tự động.

**Acceptance**: `fit` train được cả 3 họ model mà không cần sửa code engine; chạy 2 lần cùng seed cho cùng val macro-F1 (sai số < 0,001); `artifacts/runs/<id>/result.json` validate được bằng `tests/test_schema.py`.

---

## Step 6 — M1: LeNet-style baseline từ số 0

**Chủ**: B. **Ngày 4.**

`Conv(3→32, 5×5, pad=2) → ReLU → MaxPool2 → Conv(32→64, 5×5, pad=2) → ReLU → MaxPool2
→ Flatten(64·12·12=9216) → FC(256) → ReLU → Dropout(0,5) → FC(43)`.
Không BatchNorm, không residual — **đây là điểm so sánh**, M1 tồn tại để cho thấy
BN/residual/dropout trong M2 đóng góp bao nhiêu. Phải chỉ ra được: ~2,4M tham số nhưng
**96% nằm ở lớp FC đầu** (9216×256) — minh hoạ vì sao kiến trúc hiện đại thay
Flatten+FC bằng Global Average Pooling.

Out of scope: tinh chỉnh M1 để đẩy accuracy (nó là baseline, giữ nó đơn giản).

**Acceptance**: test-set top-1 ≥ 95%; bảng đếm tham số theo từng lớp xuất ra `reports/tables/m1_params.csv`; đường cong train/val loss trong `notebooks/02` kèm nhận xét về overfitting.

---

## Step 7 — M2: VGG-like sâu + BatchNorm + residual + SpatialDropout

**Chủ**: B. **Ngày 5–6.**

4 stage, mỗi stage 2 conv 3×3 + BN + ReLU, có skip connection (dùng conv 1×1 để
chiếu khi số kênh đổi), MaxPool giữa các stage, `nn.Dropout2d` (= spatial dropout,
bỏ **cả kênh**) với p tăng dần 0,1 → 0,3, kết thúc bằng GlobalAvgPool + Dropout(0,5) + FC(43).
Kênh 32 → 64 → 128 → 256, feature map 48 → 24 → 12 → 6 → 3. Khoảng 4,5M tham số.
Phải giải thích được 5 lựa chọn: (1) hai conv 3×3 thay một conv 5×5 (cùng receptive field,
18C² thay vì 25C² tham số, thêm một tầng phi tuyến); (2) BN đặt giữa conv và ReLU;
(3) residual là "đường cao tốc gradient", giải quyết degradation; (4) `Dropout2d` mạnh hơn
`Dropout` trên feature map vì pixel lân cận tương quan cao; (5) GAP thay Flatten+FC
cắt gần hết tham số và hoạt động như một regularizer cấu trúc.

Out of scope: attention module, NAS.

**Acceptance**: test-set top-1 ≥ 99%, macro-F1 ≥ 0,98; bảng ablation nội bộ (bỏ BN / bỏ residual / bỏ spatial dropout, mỗi cái một dòng) cho thấy đóng góp của từng thành phần; số tham số < 6M.

---

## Step 8 — M3: Transfer learning 3 backbone với discriminative learning rate

**Chủ**: C. **Ngày 5–7.** (chạy trên Colab T4)

ResNet18 (11,7M tham số, ~1,82 GFLOPs @224), MobileNetV2 (3,5M, ~0,30 GFLOPs),
EfficientNet-B0 (5,3M, ~0,39 GFLOPs), đều `weights=IMAGENET1K_V1`.
Thay head: `fc`/`classifier[1]` → `Linear(·, 43)`. Chuẩn hoá bằng **mean/std của ImageNet**
(0,485/0,456/0,406 và 0,229/0,224/0,225), **không** dùng mean/std của GTSRB, vì trọng số
pretrained kỳ vọng đúng phân phối đó. Upsample ảnh lên 224×224 vì các backbone này
downsample 32× — đưa 48×48 vào sẽ còn feature map 1,5×1,5, mất hết thông tin không gian.
Huấn luyện 2 pha: (1) **đóng băng** backbone, train riêng head 3 epoch để gradient lớn
của head khởi tạo ngẫu nhiên không phá trọng số pretrained; (2) **mở băng** toàn bộ với
discriminative LR qua `param_groups`: tầng đầu 1e-5, tầng giữa 1e-4, head 1e-3 —
tầng đầu học cạnh/màu đã tốt rồi nên chỉ cần nhích nhẹ, tầng cuối mang ngữ nghĩa
ImageNet nên cần đổi nhiều.

Out of scope: pretrain lại trên tập biển báo khác, self-supervised.

**Acceptance**: cả 3 backbone đạt test top-1 ≥ 99,3%; so sánh có–không warmup-freeze (ít nhất 1 backbone) cho thấy chênh lệch; `param_groups` in ra đúng 3 nhóm LR khác nhau trong log.

---

## Step 9 — Module đánh giá: top-1, top-5, macro-F1, per-class, confusion, McNemar, ECE

**Chủ**: D. **Ngày 4–6** (code sớm với fake dataset, chạy thật khi có checkpoint).

`evaluate(model, loader, device)` trả về dict cố định. Chỉ số: top-1; top-5
(**và phải nêu rõ top-5 trên 43 lớp gần như bão hoà ~99,9%, nó là chỉ số yếu ở đây —
top-5 sinh ra cho ImageNet 1000 lớp**); macro-F1 (chỉ số chính, vì nó cho mỗi lớp
trọng số bằng nhau nên không bị lớp đông "che" lớp thưa); precision/recall/F1 từng lớp;
confusion matrix 43×43 + top-10 cặp nhầm nhiều nhất; **McNemar test** để trả lời
"99,3% so với 99,5% có thật là khác nhau không"; ECE + reliability diagram để thấy
tác dụng của label smoothing lên độ hiệu chỉnh.

Out of scope: bootstrap CI cho từng lớp (nếu còn thời gian thì thêm).

**Acceptance**: `evaluate` chạy đúng cho cả 3 model qua cùng một API; `reports/tables/per_class.csv` có 43 dòng; McNemar in ra p-value cho cả 3 cặp model (M1–M2, M2–M3, M1–M3); confusion matrix lưu ở `reports/figures/`.

---

## Step 10 — Ablation studies một-biến-một-lần

**Chủ**: B (scaling, label smoothing) + C (resolution của M3) + A (augmentation, preprocessing) + D (tổng hợp). **Ngày 7–9.**

5 trục, mỗi lần đổi **đúng một** biến, cùng seed, cùng split, 3 seed nếu kịp (báo mean±std):
(1) **Độ phân giải đầu vào**: 32/48/64 cho M1–M2; 64/112/224 cho M3.
(2) **Augmentation**: none / geo / geo_photo / geo_photo_clahe.
(3) **Tiền xử lý**: none / he_gray / he_y / clahe.
(4) **Model scaling của M2**: width ×0,5 / ×1 / ×2 và depth 3 / 4 / 5 stage.
(5) **Label smoothing**: 0,0 / 0,1 / 0,2.
Mỗi cấu hình là một file YAML trong `configs/ablation/`, chạy qua `scripts/run_ablation.py`.

Out of scope: grid search đa biến (tốn compute, không thêm hiểu biết).

**Acceptance**: ≥ 18 run hoàn tất, mỗi run có `result.json`; `reports/tables/ablation.csv` do script tự tổng hợp (không chép tay); mỗi trục có 1 hình và 1 đoạn kết luận "biến này đáng giá bao nhiêu điểm macro-F1".

---

## Step 11 — Robustness dưới nhiễu mô phỏng thực tế

**Chủ**: D. **Ngày 9–10.**

5 loại nhiễu × 5 mức severity, **chỉ áp dụng lúc test, tuyệt đối không train trên chúng**
(nếu train thì đó là augmentation, không còn là đo robustness nữa):
motion blur (kernel đường thẳng, đổi độ dài + góc), Gaussian noise (σ = 5/10/20/40/60 trên 255),
fog (mô hình tán xạ khí quyển `I = J·t + A·(1−t)`, `t = e^(−βd)`),
low-light (gamma γ > 1 + nhiễu Poisson), occlusion (hình chữ nhật che 10/20/30/40/50% diện tích).
Báo cáo: đường cong accuracy theo severity cho từng model, **relative robustness**
`acc_nhiễu / acc_sạch`, và mCE kiểu ImageNet-C. Dự kiến sẽ thấy nghịch lý đáng nói:
model accuracy sạch cao nhất **không chắc** là model bền nhất.

Out of scope: adversarial attack (FGSM/PGD) — ghi vào hướng mở rộng.

**Acceptance**: 3 model × 5 nhiễu × 5 mức = 75 ô số liệu đầy đủ trong `reports/tables/robustness.csv`; 1 hình lưới 5 đường cong; nêu tên model bền nhất kèm giải thích tại sao.

---

## Step 12 — Grad-CAM: trực quan hoá vùng quyết định

**Chủ**: C. **Ngày 10–11.**

`GradCAM(model, target_layer)` theo công thức Selvaraju 2017:
`α_k^c = GAP(∂y^c/∂A^k)`, `L^c = ReLU(Σ_k α_k^c · A^k)`, rồi upsample về kích thước ảnh.
Target layer: M1 → `conv2`; M2 → conv cuối của stage 4; M3 → `layer4` (ResNet18) /
`features[-1]` (MobileNetV2, EfficientNet-B0). Mỗi model trình bày 3 nhóm ảnh:
(a) dự đoán đúng, (b) **dự đoán sai — model đang nhìn vào đâu?**, (c) cùng một ảnh
trước/sau khi thêm nhiễu — attention có trôi đi không. Phải nêu **giới hạn** của
Grad-CAM: feature map cuối của M2 ở input 48×48 chỉ còn 3×3 nên heatmap rất thô;
có thể lấy stage 3 (6×6) để nhìn rõ hơn, còn M3 ở 224 cho 7×7 nên nét hơn.

Out of scope: Grad-CAM++, Score-CAM, LIME, SHAP.

**Acceptance**: `scripts/make_gradcam.py` sinh được lưới 3 model × 3 nhóm ảnh; có ít nhất 3 ca dự đoán sai được phân tích bằng chữ; nêu rõ target layer và độ phân giải heatmap của từng model.

---

## Step 13 — Benchmark tốc độ suy luận cho triển khai biên

**Chủ**: C. **Ngày 11.**

Đo đúng cách: 20 vòng warm-up, đồng bộ hoá thiết bị (`torch.cuda.synchronize()` /
`torch.mps.synchronize()`) trước và sau khi bấm giờ, 100 vòng đo, báo **p50 và p95**
(không chỉ mean). Hai chế độ: batch=1 (đúng tình huống xe chạy, xử lý từng khung) và
batch=64 (đo throughput). Hai thiết bị: CPU và GPU/MPS. Kèm số tham số, dung lượng
file fp32 (MB), FLOPs (`thop`/`ptflops`). Tuỳ chọn: dynamic quantization int8,
export TorchScript/ONNX. Kết quả cuối là **biểu đồ Pareto accuracy–latency** trả lời
"nếu phải gắn lên xe thật thì chọn model nào". Phải giải thích được nghịch lý:
MobileNetV2 ít FLOPs hơn ResNet18 nhiều nhưng latency thực tế thường không nhanh hơn
tương ứng, vì depthwise conv bị chặn bởi băng thông bộ nhớ chứ không bởi phép tính.

Out of scope: biên dịch cho phần cứng cụ thể (TensorRT, CoreML, TFLite).

**Acceptance**: `reports/tables/speed.csv` có p50/p95 cho 5 model × 2 batch size × 2 device; biểu đồ Pareto ở `reports/figures/pareto.png`; có một khuyến nghị triển khai kèm lý do.

---

## Step 14 — Khắc phục lỗi runtime PyTorch phát sinh (shape, device, AMP, DataLoader)

**Chủ**: ai gặp thì người đó xử lý, B là người hỗ trợ. **Xuyên suốt.**

Các lỗi gần như chắc chắn sẽ gặp và nên biết trước: sai shape sau khi Flatten khi đổi
`img_size` (M1 hard-code 9216); `Expected all tensors on the same device` khi quên
`.to(device)` cho tensor phụ; MPS không hỗ trợ đầy đủ `autocast`; `num_workers>0`
treo trên macOS do fork; `Dropout2d` cảnh báo khi nhận input 3 chiều;
OOM trên Colab khi dùng batch 128 ở 224×224.

**Acceptance**: mỗi lỗi đã gặp được ghi 1 dòng vào `docs/SU_CO.md` (triệu chứng → nguyên nhân → cách sửa) để cả nhóm không mất thời gian hai lần.

---

## Step 15 — Tổng hợp báo cáo, hình bảng, slide và diễn tập bảo vệ

**Chủ**: D chủ trì, cả 4 người viết phần của mình. **Ngày 12–14.**

`docs/KET_QUA.md` (bảng so sánh chính + mọi bảng ablation), `docs/LY_THUYET.md`
(mỗi người viết phần lý thuyết của mình), `docs/QA_GIANG_VIEN.md` (mỗi người soạn
10 câu hỏi–đáp cho phần mình), `docs/SLIDE_OUTLINE.md`. Ngày cuối: **diễn tập chéo** —
mỗi người bị 3 người còn lại chất vấn 10 câu về phần của mình; ai trả lời không được
thì quay lại học, không được đẩy cho người khác trả lời hộ.

**Acceptance**: mọi số trong báo cáo truy được về một `result.json` cụ thể (không có số "chép từ đâu không rõ"); 40 câu Q&A có đáp án; mỗi thành viên trả lời trơn được 10 câu về phần mình trong buổi diễn tập.

---

## 5. Validation

```bash
# môi trường
python -c "import torch, torchvision; print(torch.__version__, torch.backends.mps.is_available())"

# dữ liệu + chống rò rỉ (bắt buộc xanh trước khi train bất cứ thứ gì)
python scripts/prepare_data.py --preprocess clahe
pytest tests/test_split.py -v          # giao track_id train/val phải rỗng
pytest tests/ -v

# huấn luyện
python scripts/train.py --config configs/m1_lenet.yaml
python scripts/train.py --config configs/m2_vggres.yaml
python scripts/train.py --config configs/m3_resnet18.yaml

# đánh giá + so sánh + kiểm định
python scripts/evaluate.py --runs artifacts/runs/* --out reports/tables/main_comparison.csv

# ablation / robustness / tốc độ / gradcam
python scripts/run_ablation.py   --grid configs/ablation/
python scripts/run_robustness.py --runs artifacts/runs/*
python scripts/benchmark_speed.py --runs artifacts/runs/*
python scripts/make_gradcam.py   --runs artifacts/runs/*
```

---

## 6. Risks

| Rủi ro | Khả năng | Ảnh hưởng | Giảm thiểu |
|---|---|---|---|
| A chậm → B, C, D bị chặn (đường tới hạn) | CAO | CAO | A giao **tiny fake dataset** đúng interface ngay Ngày 0; mọi người code và test với nó trước |
| Split random vô tình được dùng → mọi số liệu bị thổi phồng | TRUNG BÌNH | CAO | `tests/test_split.py` là cổng chặn CI; báo cáo cả hai con số để làm bằng chứng |
| Colab timeout / mất session khi train M3 | CAO | TRUNG BÌNH | checkpoint mỗi epoch ra Google Drive; `--resume`; chia 3 backbone thành 3 session riêng |
| 16GB RAM + MPS không đủ cho M3 @224 | TRUNG BÌNH | TRUNG BÌNH | M3 train trên Colab; local chỉ chạy smoke test với 500 ảnh |
| Conflict git ở notebook (.ipynb là JSON) | CAO | THẤP | mỗi người một notebook riêng; `nbstripout` xoá output trước khi commit |
| GTSRB bão hoà → cả 3 model ~99%, "không có gì để so sánh" | CAO | TRUNG BÌNH | giá trị dồn vào ablation + robustness + Pareto tốc độ; dùng McNemar để nói chênh lệch có ý nghĩa hay không |
| Mỗi người chỉ hiểu phần mình, bị giảng viên hỏi chéo là tắc | TRUNG BÌNH | CAO | Ngày 14 diễn tập chất vấn chéo; `docs/LY_THUYET.md` bắt buộc đọc hết, không chỉ đọc phần mình |
| Tải dữ liệu lỗi (nguồn gốc là `.ppm`, server đôi khi chậm) | TRUNG BÌNH | TRUNG BÌNH | dự phòng 3 nguồn: archive gốc, Kaggle (PNG, sẵn cấu trúc), `torchvision.datasets.GTSRB(download=True)` |

---

## 7. Acceptance toàn dự án

- [ ] 3 model huấn luyện xong trên **cùng** split, cùng seed, cùng ngân sách augmentation
- [ ] Bảng so sánh chính: top-1, top-5, macro-F1, #params, FLOPs, latency p50/p95
- [ ] Phân tích per-class, confusion matrix, chỉ tên được các lớp khó và lý do
- [ ] ≥ 18 run ablation trên 5 trục, mỗi trục một-biến-một-lần
- [ ] 75 ô robustness (3 model × 5 nhiễu × 5 mức) + relative robustness
- [ ] Grad-CAM cho cả 3 model, gồm phân tích ca sai
- [ ] Pareto accuracy–latency + khuyến nghị triển khai biên
- [ ] McNemar p-value cho mọi cặp model
- [ ] `tests/` xanh, đặc biệt test chống rò rỉ split
- [ ] Mọi số liệu trong báo cáo truy được về `result.json`
- [ ] Mỗi thành viên trả lời được 10 câu hỏi về phần mình **và** hiểu phần của 3 người kia
