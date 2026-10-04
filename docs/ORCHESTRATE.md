# Plan-Orchestrate — bảng điều phối agent theo từng bước

**Plan**: `.claude/plans/gtsrb-3-model-comparison.plan.md`
**Lang**: `python` (phát hiện qua `requirements.txt`), sub-profile **pytorch = true**
**ECC mode**: `plugin`, marketplace `ecc` → prefix `ecc:`
**Số bước**: 15
**Scope**: all

> ## ⚠️ Lưu ý về `/orchestrate` — đọc trước khi dùng file này
>
> Skill `plan-orchestrate` sinh ra các lệnh dạng
> `/<prefix>:orchestrate custom "<agent1>,<agent2>" "<task>"`.
> **Bản ECC 2.0.0-rc.1 đang cài trên máy này KHÔNG có lệnh đó.**
> Kiểm chứng: `ls ~/.claude/plugins/cache/ecc/ecc/2.0.0-rc.1/commands/` có 75 lệnh,
> không có `orchestrate.md`. Thứ duy nhất còn tên đó là
> `~/.claude/plugins/marketplaces/ecc/legacy-command-shims/commands/orchestrate.md` —
> một shim đã bị khai tử, không nằm trong plugin đang hoạt động, và bản thân nó ghi rõ
> *"Prefer the skills directly"*, delegate sang `dmux-workflows` / `autonomous-agent-harness`.
> Nó **không** hỗ trợ cú pháp chain `custom "<agents>" "<task>"`.
>
> Vì vậy bảng dưới đây dùng **lệnh và agent thật có trong bản này**:
> - Lệnh đã xác nhận tồn tại: `/ecc:plan`, `/ecc:feature-dev`, `/ecc:python-review`,
>   `/ecc:code-review`, `/ecc:build-fix`, `/ecc:test-coverage`, `/ecc:quality-gate`,
>   `/ecc:update-docs`, `/ecc:checkpoint`, `/ecc:multi-plan`, `/ecc:multi-execute`.
> - Agent gọi trực tiếp (có trong `agents/`, 60 file): `ecc:planner`, `ecc:architect`,
>   `ecc:tdd-guide`, `ecc:python-reviewer`, `ecc:pytorch-build-resolver`,
>   `ecc:mle-reviewer`, `ecc:code-reviewer`, `ecc:doc-updater`.
>
> **Bổ sung ngoài catalogue của skill**: `ecc:mle-reviewer` không nằm trong danh sách agent
> mà `plan-orchestrate` liệt kê, nhưng nó **đúng việc hơn** `python-reviewer` cho các bước
> liên quan tới reproducibility huấn luyện, data contract và quy trình đánh giá offline —
> đó chính là nội dung của bước 3, 5, 9, 10. Nên các bước đó dùng `mle-reviewer`.

---

## Bảng tổng quan 15 bước

| # | Bước | Chủ | Tags | Agent chain |
|---|---|---|---|---|
| 1 | Chốt kiến trúc code + interface contract + môi trường | cả nhóm / A | design | `ecc:planner`, `ecc:architect` |
| 2 | Tải GTSRB, cắt ROI, tiền xử lý HE/CLAHE, resize 48×48 | A | impl | `ecc:tdd-guide`, `ecc:python-reviewer` |
| 3 | ★ Split theo track chống rò rỉ dữ liệu | A | impl, review | `ecc:tdd-guide`, `ecc:mle-reviewer` |
| 4 | Augmentation pipeline + EDA | A | impl | `ecc:tdd-guide`, `ecc:python-reviewer` |
| 5 | Training engine: label smoothing, Adam, cosine, early stop | B | impl, review | `ecc:tdd-guide`, `ecc:mle-reviewer` |
| 6 | M1 LeNet baseline | B | impl | `ecc:tdd-guide`, `ecc:python-reviewer` |
| 7 | M2 VGG-like + BN + residual + spatial dropout | B | impl | `ecc:tdd-guide`, `ecc:python-reviewer` |
| 8 | M3 transfer learning 3 backbone + discriminative LR | C | impl | `ecc:tdd-guide`, `ecc:python-reviewer` |
| 9 | Module đánh giá: top1/top5/macroF1/per-class/McNemar/ECE | D | impl, review | `ecc:tdd-guide`, `ecc:mle-reviewer` |
| 10 | Ablation 5 trục một-biến-một-lần | B+C+A, D tổng hợp | test, review | `ecc:mle-reviewer` |
| 11 | Robustness 5 nhiễu × 5 mức | D | impl | `ecc:tdd-guide`, `ecc:python-reviewer` |
| 12 | Grad-CAM | C | impl | `ecc:tdd-guide`, `ecc:python-reviewer` |
| 13 | Benchmark latency + FLOPs + Pareto | C | impl | `ecc:tdd-guide`, `ecc:python-reviewer` |
| 14 | Khắc phục lỗi runtime PyTorch | ai gặp | build | `ecc:pytorch-build-resolver` |
| 15 | Báo cáo, hình bảng, slide, diễn tập bảo vệ | D chủ trì | docs | `ecc:doc-updater` |

**Lưu ý về cách dùng**: nhóm cần **hiểu bài** để bảo vệ trước giảng viên. Đừng để agent
viết hết rồi đọc lại — bạn sẽ không trả lời được câu hỏi về code mình không viết.
Cách dùng đúng: **tự viết trước**, rồi gọi reviewer agent để soi lỗi. Chỉ dùng
`tdd-guide` để sinh **test** (phần khô khan, ít giá trị học) và tự viết phần implementation.

---

## Bước 1 — Chốt kiến trúc code, interface contract và môi trường

**Tags**: design · **Chain rationale**: bước thiết kế, không viết code →
`planner` phân rã rủi ro, `architect` chốt cấu trúc module và ranh giới sở hữu giữa 4 người.
Đây là cặp duy nhất nên đi với nhau (skill cấm ghép `planner`+`architect` ở bước `impl`
vì tốn token vô ích, nhưng ở bước `design` thì đúng).

```bash
/ecc:plan Chốt kiến trúc src/gtsrb cho dự án GTSRB 43 lớp, 4 người code song song. Rà soát docs/INTERFACE.md: 6 chữ ký hàm liên module (GTSRBDataset, build_model, fit, evaluate, apply_corruption, GradCAM) và schema result.json. Tiêu chí: mỗi người sở hữu thư mục riêng không chồng lấn; không ai phải chờ ai quá 1 ngày; đổi img_size không làm vỡ code của người khác. Acceptance: xác nhận hoặc chỉ ra lỗ hổng trong 6 chữ ký; chỉ ra mọi phụ thuộc vòng; đề xuất thứ tự giao việc để đường tới hạn ngắn nhất.
```

---

## Bước 2 — Tải GTSRB, cắt ROI, tiền xử lý, resize 48×48

**Tags**: impl · **Chain**: `tdd-guide` → `python-reviewer` (reviewer đóng chain, theo
quy tắc bước `impl` phải kết thúc bằng agent lớp reviewer).

```bash
/ecc:feature-dev Hiện thực src/gtsrb/data/{download,preprocess}.py và scripts/prepare_data.py theo docs/INTERFACE.md mục 1. Tải GTSRB (39209 train .ppm + 12630 test official kèm GT-final_test.csv), cắt theo ROI từ CSV mỗi thư mục lớp với lề 10%, 4 chế độ tiền xử lý none/he_gray/he_y/clahe (CLAHE clipLimit=2.0 tileGridSize=8x8 trên kênh L của LAB), resize 48x48 bằng INTER_AREA khi thu nhỏ, tính mean/std CHỈ trên split train. Acceptance: index.csv đủ 51839 dòng với đúng các cột trong INTERFACE.md mục 1; 4 chế độ đều chạy và xuất reports/figures/preprocess_compare.png; mean/std ghi vào configs/base.yaml. Out of scope: augmentation và split.
```
Sau khi tự viết xong, gọi review:
```bash
/ecc:python-review src/gtsrb/data/
```

---

## Bước 3 — ★ Split theo track chống rò rỉ dữ liệu

**Tags**: impl, review · **Chain**: `tdd-guide` → **`ecc:mle-reviewer`**.
Vì sao `mle-reviewer` thay vì `python-reviewer`: đây không phải vấn đề Python style mà là
vấn đề **tính đúng đắn của phép đo ML** — rò rỉ dữ liệu giữa train và val. `mle-reviewer`
được viết cho đúng loại lỗi này (data contract, offline evaluation). Code có thể hoàn hảo
về PEP 8 mà vẫn làm mọi số liệu của dự án thành vô giá trị.

```bash
/ecc:feature-dev Hiện thực src/gtsrb/data/split.py: make_split(index_csv, val_ratio=0.2, seed=42, group_by_track=True). GTSRB quay mỗi biển báo vật lý thành 1 track 30 frame, tên file 000{track}_000{frame}.ppm. group_by_track=True dùng StratifiedGroupKFold với groups=(class_id,track_id); group_by_track=False là split random theo ảnh, CHỈ để làm đối chứng rò rỉ. Kèm tests/test_split.py. Acceptance: test chứng minh giao tập track_id giữa train và val là rỗng; phân phối 43 lớp lệch nhau dưới 1% giữa train và val; chạy được cả 2 chế độ để báo cáo đối chứng. Out of scope: resampling để cân bằng lớp.
```
```bash
# soi đúng loại lỗi nguy hiểm nhất của bước này
# (gọi agent ecc:mle-reviewer trên src/gtsrb/data/split.py và tests/test_split.py)
/ecc:code-review src/gtsrb/data/split.py
```

---

## Bước 4 — Augmentation pipeline + EDA

**Tags**: impl · **Chain**: `tdd-guide` → `python-reviewer`

```bash
/ecc:feature-dev Hiện thực src/gtsrb/data/transforms.py: build_transform(policy, img_size, seed) với 4 policy none/geo/geo_photo/geo_photo_clahe. geo = rotation +/-15 do, translate +/-10%, zoom 0.9-1.1. geo_photo thêm brightness/contrast jitter +/-0.2. CẤM TUYỆT ĐỐI RandomHorizontalFlip (lớp 33 rẽ phải lật ngang thành đúng hình lớp 34 rẽ trái; các cặp 33/34, 19/20, 36/37, 38/39) và cấm hue jitter (màu mang nghĩa: đỏ=cấm, xanh=bắt buộc). Chỉ áp dụng cho train loader. Kèm EDA trong notebooks/01: phân phối 43 lớp, phân phối kích thước ảnh gốc, histogram độ sáng, lưới ảnh mẫu. Acceptance: 4 policy gọi được qua cfg.aug.policy; reports/figures/aug_grid.png có 8 biến thể của cùng 1 ảnh; grep toàn repo không có RandomHorizontalFlip. Out of scope: MixUp, CutMix.
```

---

## Bước 5 — Training engine dùng chung

**Tags**: impl, review · **Chain**: `tdd-guide` → **`ecc:mle-reviewer`**
(lý do giống bước 3: đây là vấn đề reproducibility và tính công bằng của so sánh,
không phải vấn đề style). ★ Bước chặn — C đang chờ.

```bash
/ecc:feature-dev Hiện thực src/gtsrb/engine/{train,losses,schedulers,callbacks}.py theo docs/INTERFACE.md mục 3. fit(model, loaders, cfg, device) dùng chung cho CẢ M1, M2, M3 (không ai được viết vòng train riêng, nếu không so sánh 3 model mất công bằng). Gồm: CrossEntropyLoss(label_smoothing từ cfg), Adam/AdamW, warmup tuyến tính 3 epoch rồi CosineAnnealingLR, AMP bật trên CUDA tắt trên MPS, grad clip 1.0, early stopping theo val MACRO-F1 (không theo accuracy vì dữ liệu mất cân bằng 10.7:1), checkpoint tốt nhất, train_log.csv mỗi epoch, cố định seed torch/numpy/random + cudnn.deterministic, ghi result.json đúng schema INTERFACE.md mục 6. Hỗ trợ cfg.train.phases để M3 làm 2 pha freeze/unfreeze. Acceptance: fit train được cả 3 họ model không sửa code engine; chạy 2 lần cùng seed cho cùng val macro-F1 sai số dưới 0.001; result.json validate được bằng tests/test_schema.py. Out of scope: distributed training, hyperparameter search tự động.
```

---

## Bước 6 — M1 LeNet baseline

**Tags**: impl · **Chain**: `tdd-guide` → `python-reviewer`

```bash
/ecc:feature-dev Hiện thực src/gtsrb/models/m1_lenet.py và registry.py theo docs/INTERFACE.md mục 2. M1: Conv(3->32,5x5,pad=2)+ReLU+MaxPool2, Conv(32->64,5x5,pad=2)+ReLU+MaxPool2, Flatten(64*12*12=9216), FC(256)+ReLU+Dropout(0.5), FC(43). CỐ Ý không BatchNorm không residual - M1 là mốc tham chiếu để đo đóng góp của các thành phần trong M2. build_model phải trả model có .gradcam_target_layer, .model_name, .expected_img_size, .normalize_mode. Acceptance: test top-1 >= 95%; bảng đếm tham số từng lớp ra reports/tables/m1_params.csv và chỉ ra lớp FC đầu chiếm khoảng 96% tổng tham số; model không vỡ khi đổi img_size trong config. Out of scope: tinh chỉnh M1 để đẩy accuracy - nó là baseline, giữ đơn giản.
```

---

## Bước 7 — M2 VGG-like + BN + residual + spatial dropout

**Tags**: impl · **Chain**: `tdd-guide` → `python-reviewer`

```bash
/ecc:feature-dev Hiện thực src/gtsrb/models/m2_vggres.py. 4 stage, mỗi stage 2 lần Conv3x3+BN+ReLU có skip connection (conv 1x1 projection khi đổi số kênh), MaxPool giữa các stage, nn.Dropout2d (spatial dropout, bỏ cả kênh) p tăng dần 0.1/0.2/0.3/0.3, kết thúc GlobalAvgPool + Dropout(0.5) + FC(43). Kênh 32/64/128/256, feature map 48->24->12->6->3. Tham số phải dưới 6M. Width multiplier và số stage lấy từ cfg để bước ablation dùng được. Acceptance: test top-1 >= 99% và macro-F1 >= 0.98; bảng ablation nội bộ bỏ BN / bỏ residual / bỏ spatial dropout cho thấy đóng góp từng thành phần; .gradcam_target_layer trỏ đúng conv cuối của stage 4. Out of scope: attention module, NAS.
```

---

## Bước 8 — M3 transfer learning 3 backbone

**Tags**: impl · **Chain**: `tdd-guide` → `python-reviewer`

```bash
/ecc:feature-dev Hiện thực src/gtsrb/models/m3_transfer.py cho ResNet18, MobileNetV2, EfficientNet-B0 weights=IMAGENET1K_V1. Thay head: resnet18.fc và classifier[1] của 2 cái kia thành Linear(.,43). normalize_mode='imagenet' (mean 0.485/0.456/0.406 std 0.229/0.224/0.225) KHÔNG dùng mean/std GTSRB vì trọng số pretrained kỳ vọng phân phối ImageNet và running stats của BatchNorm pretrained được tích luỹ trên đó. expected_img_size=224 vì 3 backbone downsample 32x, đưa 48x48 vào thì feature map cuối còn 1.5x1.5. Huấn luyện 2 pha qua cfg.train.phases: pha 1 đóng băng backbone train riêng head 3 epoch để gradient lớn của head khởi tạo ngẫu nhiên không phá trọng số pretrained; pha 2 mở băng toàn bộ với discriminative LR qua param_groups (tầng đầu 1e-5, giữa 1e-4, head 1e-3). Acceptance: cả 3 backbone đạt test top-1 >= 99.3%; log in ra đúng 3 nhóm LR khác nhau; so sánh có/không warmup-freeze trên ít nhất 1 backbone. Out of scope: pretrain lại trên tập biển báo khác, self-supervised.
```

---

## Bước 9 — Module đánh giá

**Tags**: impl, review · **Chain**: `tdd-guide` → **`ecc:mle-reviewer`**

```bash
/ecc:feature-dev Hiện thực src/gtsrb/eval/{metrics,confusion,stats_tests,calibration}.py theo docs/INTERFACE.md mục 4. evaluate() trả dict cố định gồm top1, top5, macro_f1, weighted_f1, per_class DataFrame 43 dòng, y_true, y_pred, y_prob. mcnemar(y_true,y_pred_a,y_pred_b) dùng chi2 có hiệu chỉnh liên tục Yates khi n01+n10>=25, binomial chính xác khi nhỏ hơn - KHÔNG dùng t-test độc lập vì 2 model chạy trên cùng tập test nên quan sát bắt cặp. expected_calibration_error với 15 bin. confusion matrix 43x43 + top-10 cặp nhầm nhiều nhất. Acceptance: evaluate chạy đúng cho cả 3 model qua cùng API; reports/tables/per_class.csv có 43 dòng; McNemar in p-value cho cả 3 cặp model; confusion matrix lưu ra reports/figures. Out of scope: bootstrap CI từng lớp.
```

---

## Bước 10 — Ablation 5 trục

**Tags**: test, review · **Chain**: **`ecc:mle-reviewer`** một mình.
Lý do chain chỉ 1 agent: bước này không viết logic mới, nó **chạy thực nghiệm và kiểm tra
tính hợp lệ của thiết kế thực nghiệm**. Rủi ro lớn nhất không phải bug code mà là
đổi hai biến cùng lúc rồi kết luận sai — đúng việc của `mle-reviewer`.
`tdd-guide` ở đây chỉ tốn token vì không có hàm mới để viết test.

```bash
/ecc:code-review configs/ablation/ scripts/run_ablation.py
```
Nội dung cần rà: 5 trục (resolution 32/48/64 cho M1-M2 và 64/112/224 cho M3;
augmentation 4 policy; preprocessing 4 mode; model scaling width ×0,5/×1/×2 và depth 3/4/5;
label smoothing 0,0/0,1/0,2), **mỗi file YAML đổi đúng một biến** so với base,
cùng seed cùng split. Acceptance: ≥18 run có `result.json`;
`reports/tables/ablation.csv` do script sinh, không chép tay; mỗi trục có 1 hình và
1 kết luận định lượng.

---

## Bước 11 — Robustness 5 nhiễu × 5 mức

**Tags**: impl · **Chain**: `tdd-guide` → `python-reviewer`

```bash
/ecc:feature-dev Hiện thực src/gtsrb/robustness/{corruptions,benchmark}.py theo docs/INTERFACE.md mục 5. apply_corruption(img, kind, severity, rng) với img HWC uint8 vào ra cùng shape cùng dtype. 5 loại: motion_blur (kernel đường thẳng dài 3-15px góc ngẫu nhiên), gauss_noise (sigma 5/10/20/40/60 trên 255), fog (tán xạ khí quyển I = J*t + A*(1-t), t = exp(-beta*d)), low_light (gamma 1.5-3.0 kèm nhiễu Poisson), occlusion (hình chữ nhật che 10/20/30/40/50% diện tích). LUẬT SẮT: nhiễu CHỈ dùng ở test, build_transform của A không bao giờ gọi apply_corruption - train trên chúng thì phép đo mất ý nghĩa vì robustness nghĩa là khái quát sang phân phối chưa từng thấy. Báo cáo accuracy theo severity, relative robustness acc_nhiễu/acc_sạch, và mCE. Acceptance: tests/test_corruptions.py kiểm severity=0 trả ảnh y nguyên, output luôn clip về [0,255] uint8, cùng rng cho cùng kết quả; đủ 75 ô số liệu 3 model x 5 nhiễu x 5 mức trong reports/tables/robustness.csv. Out of scope: adversarial attack FGSM/PGD.
```

---

## Bước 12 — Grad-CAM

**Tags**: impl · **Chain**: `tdd-guide` → `python-reviewer`

```bash
/ecc:feature-dev Hiện thực src/gtsrb/explain/gradcam.py theo docs/INTERFACE.md mục 7. GradCAM(model, target_layer=None) mặc định lấy model.gradcam_target_layer. Công thức Selvaraju 2017: alpha_k = GAP(d y_c / d A_k), L_c = ReLU(sum_k alpha_k * A_k), rồi upsample về kích thước đầu vào. Dùng register_forward_hook lấy activation và register_full_backward_hook lấy gradient. Kèm overlay(img_uint8, heatmap, alpha=0.4). Target layer: M1 conv2, M2 conv cuối stage 4, M3 layer4 cho ResNet18 và features[-1] cho MobileNetV2/EfficientNet-B0. scripts/make_gradcam.py sinh lưới 3 model x 3 nhóm ảnh: đúng, SAI (model nhìn vào đâu), và trước/sau khi thêm nhiễu. Acceptance: lưới sinh được cho cả 3 model; ít nhất 3 ca dự đoán sai được phân tích bằng chữ trong notebooks/08; ghi rõ target layer và độ phân giải heatmap từng model (M2 ở input 48 chỉ có feature map 3x3 nên heatmap rất thô - phải nêu giới hạn này). Out of scope: Grad-CAM++, Score-CAM, LIME, SHAP.
```

---

## Bước 13 — Benchmark latency + FLOPs + Pareto

**Tags**: impl · **Chain**: `tdd-guide` → `python-reviewer`

```bash
/ecc:feature-dev Hiện thực src/gtsrb/deploy/speed.py theo docs/INTERFACE.md mục 7. benchmark(model, img_size, device, batch_sizes=(1,64), warmup=20, iters=100) và model_cost(model, img_size). BẮT BUỘC đo đúng: model.eval() + torch.no_grad(); 20 vòng warm-up; ĐỒNG BỘ THIẾT BỊ (torch.cuda.synchronize hoặc torch.mps.synchronize) TRƯỚC VÀ SAU khi bấm giờ vì GPU chạy bất đồng bộ, không sync là đo thời gian gửi lệnh chứ không phải thời gian tính; 100 vòng đo; báo p50 VÀ p95 không chỉ mean. Hai batch size 1 (tình huống xe chạy từng khung) và 64 (throughput), hai thiết bị CPU và GPU/MPS. Kèm params_m, flops_g (thop), size_mb fp32. Sinh biểu đồ Pareto accuracy-latency ra reports/figures/pareto.png. Acceptance: reports/tables/speed.csv có p50/p95 cho 5 model x 2 batch size x 2 device; ghi rõ thiết bị và phiên bản thư viện vì số latency không so sánh được giữa các máy; có 1 khuyến nghị triển khai kèm lý do. Out of scope: TensorRT, CoreML, TFLite.
```

---

## Bước 14 — Khắc phục lỗi runtime PyTorch

**Tags**: build · **Chain**: **`ecc:pytorch-build-resolver`** một mình.
Theo quy tắc của skill: `python` + `pytorch=true` → dùng `pytorch-build-resolver`
(không có `python-build-resolver`; `--lang=python` mà không có pytorch thì mới rơi về
`build-error-resolver`). Bước `build` được gate bởi chính resolver nên không cần thêm reviewer.

```bash
/ecc:build-fix
```
Các lỗi gần như chắc chắn gặp, ghi vào `docs/SU_CO.md` khi gặp:
sai shape sau `Flatten` khi đổi `img_size` (M1 hard-code 9216);
`Expected all tensors to be on the same device`;
`autocast` không hoạt động đầy đủ trên MPS;
`num_workers>0` treo trên macOS do fork;
`Dropout2d` cảnh báo khi nhận input 3 chiều;
OOM trên Colab với batch 128 ở 224×224.

---

## Bước 15 — Báo cáo, hình bảng, slide, diễn tập

**Tags**: docs · **Chain**: `ecc:doc-updater`

```bash
/ecc:update-docs docs/KET_QUA.md docs/LY_THUYET.md docs/QA_GIANG_VIEN.md docs/SLIDE_OUTLINE.md
```
Điền mọi `[chờ]` và `[điền]` từ `artifacts/runs/*/result.json`.
Acceptance: mọi số trong báo cáo truy được về một `run_id` cụ thể;
40 câu Q&A trong `PHAN_CONG.md` + câu hỏi chéo trong `QA_GIANG_VIEN.md` có đáp án;
mỗi thành viên trả lời trơn 10 câu của mình trong buổi diễn tập ngày cuối.

---

## Chạy hàng loạt

Vì `/orchestrate` không tồn tại trong bản này, không có "batch execution block" dạng
một lệnh chạy tất cả. Thay vào đó, trình tự theo đường tới hạn:

```bash
# NGÀY 0 — cả nhóm
/ecc:plan                       # bước 1

# NGÀY 1-3 — A chạy, B/C/D code song song với fake dataset
/ecc:feature-dev                # bước 2, 3, 4   (A)
/ecc:feature-dev                # bước 5         (B) — song song, đây là bước chặn thứ 2
/ecc:feature-dev                # bước 9         (D) — code trước, chạy sau

# NGÀY 4-7
/ecc:feature-dev                # bước 6, 7      (B)
/ecc:feature-dev                # bước 8         (C) — trên Colab

# NGÀY 7-11
/ecc:code-review configs/ablation/    # bước 10
/ecc:feature-dev                      # bước 11 (D), 12, 13 (C)

# XUYÊN SUỐT
/ecc:build-fix                        # bước 14 khi gặp lỗi
/ecc:checkpoint                       # chốt mốc sau mỗi bước xanh

# NGÀY 12-14
/ecc:update-docs                      # bước 15
/ecc:quality-gate                     # cổng chất lượng cuối
```

**Nếu muốn điều phối song song thật** (4 người, 4 worktree): bản ECC này thay
`/orchestrate` bằng hai skill — `ecc:dmux-workflows` (chia pane/worktree, multi-agent song song)
và `ecc:autonomous-agent-harness` (vòng lặp dài, scheduling). Với một dự án môn học 4 người
thì chúng là quá nặng; git branch thường + `docs/INTERFACE.md` là đủ.
