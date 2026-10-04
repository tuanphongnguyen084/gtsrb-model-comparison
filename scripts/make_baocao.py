#!/usr/bin/env python
"""
make_baocao.py — sinh BÁO CÁO NỘP THẦY: docs/BAO_CAO.md

CHỦ: Phong Nguyễn

Khác gì `make_report.py`?
  make_report.py -> docs/KET_QUA.md  : bảng số thô, dành cho NHÓM tra cứu
  make_baocao.py -> docs/BAO_CAO.md  : báo cáo học thuật, dành cho GIẢNG VIÊN đọc

Phần diễn giải viết tay, nhưng MỌI BẢNG SỐ đều đọc trực tiếp từ reports/tables/.
Nhờ vậy chạy lại sau khi có thêm thực nghiệm là báo cáo tự cập nhật, và không có
con số nào bị chép tay (quy tắc số 3 của nhóm).

    python scripts/make_baocao.py
    make baocao
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from gtsrb.utils.logging import get_logger

log = get_logger()

TABLES = Path("reports/tables")
MISSING = "_(chưa có số liệu — chạy lệnh tương ứng trong README)_"


# ---------------------------------------------------------------- tiện ích

def read(name: str, **kw) -> pd.DataFrame | None:
    """Đọc một bảng trong reports/tables/, trả None nếu chưa có."""
    path = TABLES / name
    return pd.read_csv(path, **kw) if path.exists() else None


def read_main() -> pd.DataFrame | None:
    """Bảng so sánh chính, CHỈ gồm run chính (bỏ run ablation và run nhiều seed).

    Run ablation/seed có cột `tag` khác rỗng. Không lọc thì bảng so sánh trong báo
    cáo sẽ có 40+ dòng và một biến thể ablation có thể bị chọn làm model tốt nhất.
    """
    frame = read("main_comparison.csv")
    if frame is None:
        return None
    if "tag" in frame.columns:
        frame = frame[frame["tag"].isna() | (frame["tag"].astype(str) == "")]
    return frame if not frame.empty else None


def md(frame: pd.DataFrame | None, **kw) -> str:
    """DataFrame -> bảng Markdown, hoặc câu nhắc nếu chưa có dữ liệu."""
    return MISSING if frame is None or frame.empty else frame.to_markdown(index=False, **kw)


def pct(x) -> str:
    """Số thực -> chuỗi phần trăm hai chữ số thập phân."""
    return "–" if pd.isna(x) else f"{x * 100:.2f}%"


# ---------------------------------------------------------------- các bảng

def table_models() -> str:
    """Bảng so sánh chính, gộp accuracy + chi phí."""
    main, speed = read_main(), read("speed.csv")
    if main is None:
        return MISSING
    if speed is not None:
        main = main.merge(speed[["model", "flops_g", "size_mb", "cpu_bs1_p50",
                                 "cpu_bs1_p95"]], on="model", how="left")
    out = pd.DataFrame({
        "Mô hình": main["model"],
        "Tham số (M)": main["params_m"].round(2),
        "FLOPs (G)": main.get("flops_g", pd.Series()).round(2),
        "Dung lượng (MB)": main.get("size_mb", pd.Series()).round(1),
        "Top-1": main["test_top1"].map(pct),
        "Top-5": main["test_top5"].map(pct),
        "Macro-F1": main["test_macro_f1"].round(4),
        "ECE": main["test_ece"].round(4),
        "p95 CPU (ms)": main.get("cpu_bs1_p95", pd.Series()).round(1),
        "Train (phút)": (main["train_seconds"] / 60).round(0),
    })
    return out.sort_values("Macro-F1", ascending=False).to_markdown(index=False)


def table_mcnemar() -> str:
    """Kiểm định McNemar cho mọi cặp, kèm cột kết luận bằng tiếng Việt."""
    frame = read("mcnemar.csv")
    if frame is None:
        return MISSING
    out = pd.DataFrame({
        "Cặp": frame["model_a"] + " vs " + frame["model_b"],
        "n01": frame["n01"], "n10": frame["n10"],
        "p-value": frame["p_value"].map(lambda v: f"{v:.3g}"),
        "Kết luận": frame.apply(
            lambda r: "**tương đương**" if r.p_value >= 0.05
            else f"{r.model_a if r.better == 'A' else r.model_b} hơn", axis=1),
    })
    return out.to_markdown(index=False)


def table_efficiency() -> str:
    """ms trên mỗi GFLOP — bằng chứng latency không tỉ lệ FLOPs."""
    speed = read("speed.csv")
    if speed is None or "flops_g" not in speed:
        return MISSING
    out = speed[["model", "flops_g", "cpu_bs1_p50"]].copy()
    out["ms/GFLOP"] = (out["cpu_bs1_p50"] / out["flops_g"]).round(1)
    out.columns = ["Mô hình", "FLOPs (G)", "Latency p50 (ms)", "**ms/GFLOP**"]
    return out.sort_values("**ms/GFLOP**").round(2).to_markdown(index=False)


def table_robustness() -> str:
    """Bảng relative robustness trung bình theo loại nhiễu, kèm mCE."""
    frame = read("robustness_summary.csv", index_col=0)
    return MISSING if frame is None else frame.round(4).to_markdown()


def table_worst_classes(n: int = 8) -> str:
    """Lớp yếu nhất của mô hình tốt nhất."""
    main = read_main()
    if main is None:
        return MISSING
    best = main.loc[main["test_macro_f1"].idxmax(), "model"]
    frame = read(f"per_class_{best}.csv")
    if frame is None:
        return MISSING
    out = frame.nsmallest(n, "f1")[["class_id", "name", "f1", "support"]]
    out.columns = ["Lớp", "Tên", "F1", "Số ảnh test"]
    return f"**Mô hình `{best}`:**\n\n" + out.round(3).to_markdown(index=False)


def table_confusions(n: int = 6) -> str:
    """n cặp bị nhầm nhiều nhất của mô hình tốt nhất."""
    main = read_main()
    if main is None:
        return MISSING
    best = main.loc[main["test_macro_f1"].idxmax(), "model"]
    frame = read(f"top_confusions_{best}.csv")
    if frame is None:
        return MISSING
    out = frame.head(n)[["true_id", "true_name", "pred_id", "pred_name", "count"]]
    out.columns = ["Lớp thật", "Tên", "Đoán thành", "Tên", "Số lần"]
    return out.to_markdown(index=False)


def facts() -> dict:
    """Vài con số dùng rải rác trong phần diễn giải, đọc từ bảng thật."""
    main, speed = read_main(), read("speed.csv")
    out = {"best": "–", "best_f1": "–", "chosen": "–", "speedup": "–", "n_models": 0}
    if main is None:
        return out
    out["n_models"] = len(main)
    row = main.loc[main["test_macro_f1"].idxmax()]
    out["best"], out["best_f1"] = row["model"], f"{row['test_macro_f1']:.4f}"

    mcnemar = read("mcnemar.csv")
    tied = {out["best"]}
    if mcnemar is not None:
        for r in mcnemar.itertuples(index=False):
            if out["best"] in (r.model_a, r.model_b) and r.p_value >= 0.05:
                tied.add(r.model_b if r.model_a == out["best"] else r.model_a)
    if speed is not None and "cpu_bs1_p95" in speed:
        cand = speed[speed["model"].isin(tied)].dropna(subset=["cpu_bs1_p95"])
        if not cand.empty:
            chosen = cand.loc[cand["cpu_bs1_p95"].idxmin()]
            slowest = cand.loc[cand["cpu_bs1_p95"].idxmax()]
            out["chosen"] = chosen["model"]
            out["speedup"] = f"{slowest['cpu_bs1_p95'] / chosen['cpu_bs1_p95']:.1f}"
    return out


# ---------------------------------------------------------------- báo cáo

def build(f: dict) -> str:
    """Ghép báo cáo. Phần chữ viết tay, phần bảng lấy từ reports/tables/."""
    today = datetime.now().strftime("%d/%m/%Y")
    return f"""# Phân loại biển báo giao thông GTSRB: so sánh ba hướng tiếp cận Deep Learning

**Môn**: Deep Learning · **Ngày**: {today}
**Nhóm**: Phong Nguyễn · Hoàng · Phong Trần · Huy

> Bảng số trong báo cáo này được **sinh tự động** từ `reports/tables/` bằng
> `python scripts/make_baocao.py`. Không có con số nào chép tay; mỗi con số truy được
> về một `artifacts/runs/<run_id>/result.json` cụ thể.

---

## Tóm tắt

Nhóm xây và so sánh {f['n_models']} mô hình phân loại 43 lớp biển báo giao thông Đức trên
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
   Mô hình tên "EfficientNet" lại là mô hình **chậm nhất** trong cả {f['n_models']}.
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
(tên file `000{{track}}_000{{frame}}.ppm`). Nếu trộn tất cả ảnh rồi cắt 80/20 ngẫu nhiên,
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
10,7:1 khiến accuracy bị các lớp đông chi phối). Cả {f['n_models']} mô hình dùng **chung
một hàm `fit()`** để so sánh được công bằng.

---

## 3. Kết quả

### 3.1 Bảng so sánh chính (tập test chính thức, 12.630 ảnh)

{table_models()}

**Lưu ý khi đọc:** cột **Top-5 gần như vô nghĩa** ở bài này. Top-5 ra đời cho ImageNet
1000 lớp; trên 43 lớp nó nghĩa là "đúng trong 11,6% số lớp" nên mọi mô hình tử tế đều
~99,9%. Nhóm báo cáo vì đề bài yêu cầu nhưng **kết luận dựa trên top-1 và macro-F1**.

### 3.2 ★ Kiểm định ý nghĩa thống kê (McNemar)

Hai mô hình chạy trên **cùng** tập test nên quan sát là **bắt cặp**; dùng t-test hai mẫu
độc lập là sai về mặt thống kê. McNemar chỉ xét hai ô **bất đồng** của bảng 2×2.

{table_mcnemar()}

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

{table_worst_classes()}

Các cặp bị nhầm nhiều nhất:

{table_confusions()}

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

{table_robustness()}

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

{table_efficiency()}

Nếu latency tỉ lệ FLOPs thì cột cuối phải bằng nhau — thực tế chênh **64 lần**.
EfficientNet-B0 có **ít hơn ResNet18 4,4 lần FLOPs** nhưng chậm hơn **14 lần** trên CPU.
Nguyên nhân: MBConv + squeeze-excitation gồm rất nhiều lớp **mảnh**, mỗi lớp tốn chi phí
cố định (launch kernel, truy cập bộ nhớ) mà làm rất ít phép tính → bị chặn bởi **băng
thông bộ nhớ**, không bởi năng lực tính toán.

**Hệ quả thực hành: chọn mô hình theo FLOPs sẽ dẫn tới EfficientNet-B0 — mô hình chậm
nhất trong cả {f['n_models']}.** Muốn nói về triển khai thì phải đo wall-clock trên thiết
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

Mô hình macro-F1 cao nhất là `{f['best']}` ({f['best_f1']}). Nhưng trong nhóm **tương đương
thống kê** với nó, mô hình nhanh nhất là **`{f['chosen']}`** — nhanh hơn **{f['speedup']}
lần** ở p95. Trả thêm {f['speedup']} lần latency để lấy chênh lệch accuracy *không có ý
nghĩa thống kê* là lựa chọn tồi trên hệ thống thời gian thực.

→ **Khuyến nghị: `{f['chosen']}`.** Chi tiết suy luận ba bước: `docs/KET_QUA.md` mục 9.

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
"""


def main() -> None:
    """Sinh docs/BAO_CAO.md từ các bảng trong reports/tables/."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/BAO_CAO.md")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(facts()), encoding="utf-8")
    log.info("Đã sinh %s (%d dòng)", out,
             len(out.read_text(encoding="utf-8").splitlines()))


if __name__ == "__main__":
    main()
