#!/usr/bin/env python
"""
make_report.py — TỰ SINH docs/KET_QUA.md từ mọi result.json và bảng trong reports/.

CHỦ: Phong Nguyễn

★ VÌ SAO CÓ SCRIPT NÀY ★
Quy tắc số 3 của nhóm: "không chép số bằng tay vào báo cáo — mọi số phải truy được
về một result.json cụ thể". Quy tắc đó chỉ có hiệu lực nếu việc tuân thủ nó DỄ HƠN
việc vi phạm. Script này làm cho nó dễ hơn: chạy một lệnh, báo cáo tự cập nhật.

Lợi ích phụ: khi giảng viên hỏi "số 99,2% này ở đâu ra?", mỗi dòng trong bảng đều có
cột `run_id` trỏ thẳng về thư mục chứa checkpoint, config và log của đúng lần chạy đó.

Dùng:
    python scripts/make_report.py
    make report
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import glob
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from gtsrb.utils.logging import get_logger

log = get_logger()

CHUA_CO = "_(chưa chạy)_"


def load_results(pattern: str = "artifacts/runs/*") -> list[dict]:
    """Đọc mọi result.json khớp pattern. Bỏ qua file hỏng thay vì vỡ cả báo cáo."""
    out = []
    for path in sorted(glob.glob(f"{pattern}/result.json")):
        try:
            out.append(json.loads(Path(path).read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            log.warning("Bỏ qua %s (JSON hỏng)", path)
    return out


def fmt(value, digits: int = 4, percent: bool = False) -> str:
    """Định dạng số cho bảng Markdown. None -> dấu gạch ngang."""
    if value is None:
        return "–"
    if isinstance(value, (int, float)):
        return f"{value * 100:.2f}%" if percent else f"{value:.{digits}f}"
    return str(value)


def table_main(results: list[dict]) -> str:
    """Bảng so sánh chính. Chỉ lấy các run KHÔNG phải ablation (notes rỗng)."""
    rows = []
    for r in results:
        if r.get("notes"):          # run ablation có tag -> bỏ khỏi bảng chính
            continue
        test = r.get("test") or {}
        cost = r.get("cost") or {}
        lat = None
        for key, value in cost.items():
            if key.endswith("_bs1_p50"):
                lat = value
                break
        rows.append({
            "Model": r.get("model"),
            "run_id": f"`{r.get('run_id')}`",
            "Top-1": fmt(test.get("top1"), percent=True),
            "Top-5": fmt(test.get("top5"), percent=True),
            "**Macro-F1**": f"**{fmt(test.get('macro_f1'))}**" if test.get("macro_f1") else "–",
            "ECE": fmt(test.get("ece")),
            "#params (M)": fmt(cost.get("params_m"), 2),
            "FLOPs (G)": fmt(cost.get("flops_g"), 2),
            "Latency p50 bs1 (ms)": fmt(lat, 2),
        })
    if not rows:
        return CHUA_CO
    frame = pd.DataFrame(rows)
    return frame.to_markdown(index=False)


def table_ablation() -> str:
    """Bảng ablation, nhóm theo trục. Trả câu nhắc chạy lệnh nếu chưa có số."""
    path = Path("reports/tables/ablation.csv")
    if not path.exists():
        return CHUA_CO + "  \nChạy: `python scripts/run_ablation.py --axes all` rồi `--collect-only`"
    frame = pd.read_csv(path)
    frame = frame[frame["tag"].notna() & (frame["tag"] != "")]
    if frame.empty:
        return CHUA_CO
    frame["trục"] = frame["tag"].astype(str).str.split("-").str[0]
    keep = ["trục", "tag", "model", "img_size", "preprocess", "aug_policy",
            "label_smoothing", "test_top1", "test_macro_f1", "test_ece"]
    keep = [c for c in keep if c in frame.columns]
    return frame[keep].sort_values(["trục", "tag"]).to_markdown(index=False)


def table_mcnemar() -> str:
    """Bảng kiểm định McNemar cho mọi cặp model."""
    path = Path("reports/tables/mcnemar.csv")
    if not path.exists():
        return CHUA_CO + "  \nCần ≥ 2 model. Chạy: `make eval`"
    frame = pd.read_csv(path)
    keep = ["model_a", "model_b", "n01", "n10", "method", "p_value", "better", "verdict"]
    return frame[[c for c in keep if c in frame.columns]].to_markdown(index=False)


def table_robustness() -> str:
    """Bảng relative robustness trung bình theo loại nhiễu + mCE."""
    path = Path("reports/tables/robustness_summary.csv")
    if not path.exists():
        return CHUA_CO + "  \nChạy: `make robustness`"
    return pd.read_csv(path, index_col=0).round(4).to_markdown()


def table_worst_classes() -> str:
    """8 lớp F1 thấp nhất của mỗi model."""
    files = sorted(Path("reports/tables").glob("per_class_*.csv"))
    if not files:
        return CHUA_CO
    parts = []
    for f in files:
        name = f.stem.replace("per_class_", "")
        frame = pd.read_csv(f).nsmallest(8, "f1")[["class_id", "name", "f1", "support"]]
        parts.append(f"**{name}**\n\n" + frame.to_markdown(index=False))
    return "\n\n".join(parts)


def table_confusions() -> str:
    """8 cặp (lớp thật -> lớp đoán) bị nhầm nhiều nhất của mỗi model."""
    files = sorted(Path("reports/tables").glob("top_confusions_*.csv"))
    if not files:
        return CHUA_CO
    parts = []
    for f in files:
        name = f.stem.replace("top_confusions_", "")
        frame = pd.read_csv(f).head(8)
        parts.append(f"**{name}**\n\n" + frame.to_markdown(index=False))
    return "\n\n".join(parts)


# =====================================================================
# Các đoạn văn giải thích — tách riêng để main() chỉ còn việc ghép
# =====================================================================

NOTE_MAIN_TABLE = """**Cách đọc bảng — ba điều phải nói khi trình bày:**

1. **Top-5 gần như vô nghĩa ở đây.** Top-5 ra đời cho ImageNet **1000** lớp. Trên **43**
   lớp, top-5 nghĩa là "đúng trong 11,6% số lớp" — mọi model tử tế đều ~99,9%. Chỉ số
   **bão hoà** thì không phân biệt được gì. Nhóm báo cáo vì đề bài yêu cầu, nhưng
   **kết luận dựa trên top-1 và macro-F1**.
2. **Macro-F1 là chỉ số chính**, vì nó cho mỗi lớp trọng số bằng nhau (dữ liệu mất cân
   bằng 10,7:1). Bỏ hẳn một lớp 210 ảnh chỉ làm accuracy giảm ~0,54 điểm nhưng macro-F1
   mất ~2,3 điểm.
3. **Chênh lệch trong bảng chưa chắc có ý nghĩa** — xem mục 2. Trên 12.630 ảnh,
   0,2 điểm top-1 chỉ là ~25 ảnh."""

NOTE_MCNEMAR = """Hai model chạy trên **cùng** tập test → quan sát **bắt cặp** → dùng t-test hai mẫu
độc lập là **sai về mặt thống kê**. Chỉ hai ô **bất đồng** (`n01`, `n10`) mang thông tin."""

SECTION_LEAKAGE = """| Cách split | Track dùng chung giữa train và val |
|---|---|
| Random theo ảnh (**sai**) | **1.306** |
| Theo track, `StratifiedGroupKFold` (**đúng**) | **0** |

Để đo phần accuracy "ảo": `make data-leaky && make train-m1`, rồi so với con số
của split đúng. Dấu hiệu rò rỉ: **val cao hơn test một cách bất thường**."""

NOTE_WORST_CLASSES = """Hai nguyên nhân phải **tách ra**:
F1 thấp + support **cao** → **hình giống nhau** (vấn đề độ phân giải);
F1 thấp + support **thấp** → **thiếu dữ liệu**."""

NOTE_CONFUSION = """Bảng này dễ đọc hơn hình confusion matrix 43×43, và nó cho biết lỗi **đi đâu** —
điều mà accuracy không cho biết."""

NOTE_ROBUSTNESS = """`relative = acc_nhiễu / acc_sạch` — chia cho accuracy sạch để **tách "giỏi sẵn" khỏi "BỀN"**.
`mCE` = trung bình error qua mọi nhiễu và mức, càng **thấp** càng tốt."""

SECTION_FIGURES = """| Hình | Nội dung |
|---|---|
| `reports/figures/preprocess_compare.png` | 4 chế độ cân bằng sáng trên ảnh tối |
| `reports/figures/aug_grid.png` | 8 biến thể augmentation của cùng 1 ảnh |
| `reports/figures/why_no_flip.png` | 4 cặp lớp gương — vì sao cấm flip ngang |
| `reports/figures/confusion_*.png` | confusion matrix 43×43 chuẩn hoá theo hàng |
| `reports/figures/reliability_*.png` | reliability diagram (hiệu chỉnh) |
| `reports/figures/corruption_grid.png` | 5 nhiễu × 5 mức trên cùng một ảnh |
| `reports/figures/robustness_curves.png` | accuracy giảm theo mức nhiễu |
| `reports/figures/gradcam_*_correct.png` | Grad-CAM, dự đoán đúng |
| `reports/figures/gradcam_*_wrong.png` | Grad-CAM, **dự đoán SAI** — model nhìn vào đâu |
| `reports/figures/gradcam_*_corrupted.png` | attention có trôi khi có nhiễu |
| `reports/figures/pareto.png` | Pareto accuracy–latency |"""

def section_recommendation() -> str:
    """★ Tự SUY RA khuyến nghị triển khai từ dữ liệu, không để người điền tay.

    Thuật toán — đúng ba bước mà một kỹ sư sẽ làm:
      1. Tìm model có macro-F1 cao nhất.
      2. Dùng McNemar tìm mọi model TƯƠNG ĐƯƠNG THỐNG KÊ với nó (p >= 0,05).
         Đây là bước mà người ta hay bỏ qua, và vì bỏ qua nên chọn nhầm model.
      3. Trong nhóm tương đương đó, chọn model có latency p95 ở batch=1 THẤP NHẤT
         (batch=1 vì đó là tình huống xe chạy: xử lý từng khung ảnh).

    Vì sao p95 chứ không p50: hệ thống thời gian thực quan tâm TRƯỜNG HỢP XẤU.
    Một khung ảnh chậm gấp đôi có thể là khác biệt giữa phanh kịp và không kịp.
    """
    main_path = Path("reports/tables/main_comparison.csv")
    if not main_path.exists():
        return CHUA_CO + "  \nChạy: `make eval`"

    main = pd.read_csv(main_path)

    # ★ CHỈ xét run CHÍNH. Run ablation/seed có tag -> loại bỏ.
    # Không lọc thì một biến thể ablation (ví dụ M2 width x2) có thể có macro-F1
    # cao nhất và bị chọn làm khuyến nghị triển khai — trong khi nó không phải
    # một trong các model mà nhóm thực sự so sánh.
    if "tag" in main.columns:
        main = main[main["tag"].isna() | (main["tag"].astype(str) == "")]
    if main.empty:
        return MISSING + "  \nKhông có run CHÍNH nào (mọi run đều có tag)."

    best = main.loc[main["test_macro_f1"].idxmax()]
    best_model = best["model"]

    # --- Bước 2: nhóm tương đương thống kê ---
    tied = {best_model}
    mcnemar_path = Path("reports/tables/mcnemar.csv")
    pairs_text = []
    if mcnemar_path.exists():
        mcnemar = pd.read_csv(mcnemar_path)
        for row in mcnemar.itertuples(index=False):
            if best_model not in (row.model_a, row.model_b):
                continue
            other = row.model_b if row.model_a == best_model else row.model_a
            verdict = "TƯƠNG ĐƯƠNG" if row.p_value >= 0.05 else "khác biệt THẬT"
            pairs_text.append(
                f"| {row.model_a} vs {row.model_b} | {row.n01} | {row.n10} | "
                f"{row.p_value:.4g} | {verdict} |")
            if row.p_value >= 0.05:
                tied.add(other)

    # --- Bước 3: trong nhóm tương đương, chọn model nhanh nhất ---
    speed_path = Path("reports/tables/speed.csv")
    if not speed_path.exists():
        return (f"Model macro-F1 cao nhất: **{best_model}**. "
                f"Chưa có số latency — chạy `make speed` để ra khuyến nghị đầy đủ.")

    speed = pd.read_csv(speed_path)
    latency_col = next((c for c in speed.columns if c.endswith("_bs1_p95")), None)
    candidates = speed[speed["model"].isin(tied)].dropna(subset=[latency_col])
    if candidates.empty:
        return f"Model macro-F1 cao nhất: **{best_model}**. Thiếu số latency để kết luận."

    chosen = candidates.loc[candidates[latency_col].idxmin()]
    device = latency_col.split("_")[0]

    lines = [
        f"**Khuyến nghị: `{chosen['model']}`** — suy ra từ dữ liệu theo ba bước bên dưới.",
        "",
        f"### Bước 1 — Model macro-F1 cao nhất: `{best_model}` "
        f"({best['test_macro_f1']:.4f})",
        "",
        "### Bước 2 — Những model TƯƠNG ĐƯƠNG THỐNG KÊ với nó (McNemar, p ≥ 0,05)",
        "",
    ]
    if pairs_text:
        lines += ["| Cặp | n₀₁ | n₁₀ | p-value | Kết luận |",
                  "|---|---|---|---|---|"] + pairs_text + [""]
    lines.append(f"Nhóm tương đương: **{', '.join(sorted(tied))}**")
    lines.append("")
    lines.append(f"### Bước 3 — Trong nhóm đó, chọn model có latency p95 thấp nhất "
                 f"(batch=1, {device})")
    lines.append("")
    lines += ["| Model | Latency p95 | #params | Dung lượng | FLOPs | Test top-1 |",
              "|---|---|---|---|---|---|"]
    for row in candidates.sort_values(latency_col).itertuples(index=False):
        mark = " ←" if row.model == chosen["model"] else ""
        top1 = f"{row.test_top1 * 100:.2f}%" if pd.notna(row.test_top1) else "–"
        lines.append(f"| `{row.model}`{mark} | {getattr(row, latency_col):.2f} ms | "
                     f"{row.params_m:.2f} M | {row.size_mb:.1f} MB | "
                     f"{row.flops_g:.3f} G | {top1} |")

    # --- Câu kết luận, có số cụ thể ---
    if len(candidates) > 1:
        slowest = candidates.loc[candidates[latency_col].idxmax()]
        speedup = slowest[latency_col] / chosen[latency_col]
        size_ratio = slowest["size_mb"] / chosen["size_mb"]
        acc_gap = abs(slowest["test_top1"] - chosen["test_top1"]) * 100
        lines += [
            "",
            "### Kết luận",
            "",
            f"> `{chosen['model']}` và `{slowest['model']}` **tương đương về accuracy** "
            f"(chênh {acc_gap:.2f} điểm, McNemar p ≥ 0,05 → không có ý nghĩa thống kê).",
            f"> Nhưng `{chosen['model']}` **nhanh hơn {speedup:.1f} lần** "
            f"({chosen[latency_col]:.2f} ms so với {slowest[latency_col]:.2f} ms ở p95, "
            f"batch=1) và **nhẹ hơn {size_ratio:.1f} lần** "
            f"({chosen['size_mb']:.1f} MB so với {slowest['size_mb']:.1f} MB).",
            ">",
            f"> Trả thêm {speedup:.1f} lần latency để lấy {acc_gap:.2f} điểm accuracy "
            f"**không có ý nghĩa thống kê** là lựa chọn tồi trên hệ thống thời gian thực.",
            "",
            "**Đây là lý do phải kiểm định thống kê.** Nếu chỉ nhìn bảng accuracy, "
            f"kết luận sẽ là \"chọn `{slowest['model']}` vì nó cao hơn\" — một kết luận "
            "**sai**.",
        ]
    else:
        lines += ["", "### Kết luận", "",
                  f"> Chỉ `{chosen['model']}` nằm trong nhóm tốt nhất về accuracy "
                  f"(không model nào tương đương thống kê với nó), nên nó cũng là "
                  f"lựa chọn triển khai."]

    lines += ["",
              "> ⚠️ Số latency này đo trên máy phát triển. Trước khi chốt triển khai thật, "
              "phải **đo lại trên thiết bị đích** — latency không so sánh được giữa hai "
              "máy khác nhau."]
    return "\n".join(lines)


def note_suspect_times(results: list[dict]) -> str:
    """Cảnh báo nếu có run nào bị máy ngủ làm hỏng cột thời gian train.

    Không im lặng dùng số đã sửa — người đọc báo cáo phải biết số nào đã bị can thiệp
    và vì sao, nếu không thì bảng mất tính kiểm chứng được.
    """
    suspects = [r for r in results
                if (r.get("train") or {}).get("train_seconds_suspect")]
    if not suspects:
        return ""

    lines = ["", "> ⚠️ **Cột thời gian train của các run sau đã được ước lượng lại:**", ""]
    for r in suspects:
        train = r["train"]
        lines.append(f"> - `{r['run_id']}` — thô **{train['train_seconds']:.0f}s**, "
                     f"ước lượng **{train['train_seconds_clean']:.0f}s**")
    lines += [
        ">",
        "> Nguyên nhân: máy ngủ giữa lúc train nên `time.time()` đếm cả thời gian ngủ.",
        "> Số ước lượng tính bằng cách thay epoch bất thường bằng **trung vị** các epoch",
        "> còn lại (`_detect_sleep()` trong `src/gtsrb/engine/train.py`). Số thô vẫn được",
        "> giữ ở cột `train_seconds_raw` của `main_comparison.csv`.",
        "> Chi tiết: `docs/SU_CO.md` mục 7.",
    ]
    return "\n".join(lines)


def build_markdown(results: list[dict]) -> str:
    """Ghép toàn bộ báo cáo từ các bảng và đoạn văn ở trên."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    sections = [
        f"""# Kết quả

> **File này được SINH TỰ ĐỘNG** bởi `python scripts/make_report.py` (hoặc `make report`).
> **Đừng sửa tay** — lần chạy sau sẽ ghi đè.
> Mọi số đều truy được về một `artifacts/runs/<run_id>/result.json` cụ thể.
>
> Cập nhật lần cuối: {now} · {len(results)} run""",

        "## 1. Bảng so sánh chính (tập test chính thức, 12.630 ảnh)\n\n"
        + table_main(results) + "\n\n" + NOTE_MAIN_TABLE + note_suspect_times(results),

        "## 2. Kiểm định ý nghĩa thống kê — McNemar\n\n"
        + NOTE_MCNEMAR + "\n\n" + table_mcnemar(),

        "## 3. ★ Đối chứng rò rỉ dữ liệu — split theo track vs split random\n\n"
        + SECTION_LEAKAGE,

        "## 4. Lớp yếu nhất (8 lớp F1 thấp nhất mỗi model)\n\n"
        + NOTE_WORST_CLASSES + "\n\n" + table_worst_classes(),

        "## 5. Các cặp bị nhầm nhiều nhất\n\n"
        + NOTE_CONFUSION + "\n\n" + table_confusions(),

        "## 6. Ablation (một-biến-một-lần)\n\n" + table_ablation(),

        "## 7. Robustness (5 nhiễu × 5 mức, **chỉ áp ở test**)\n\n"
        + NOTE_ROBUSTNESS + "\n\n" + table_robustness(),

        "## 8. Hình\n\n" + SECTION_FIGURES,

        "## 9. Khuyến nghị triển khai biên\n\n" + section_recommendation(),
    ]
    return "\n\n---\n\n".join(sections) + "\n"


def main() -> None:
    """Gom mọi result.json và bảng trong reports/ thành docs/KET_QUA.md."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="artifacts/runs/*")
    parser.add_argument("--out", default="docs/KET_QUA.md")
    args = parser.parse_args()

    results = load_results(args.runs)
    log.info("Đọc %d result.json", len(results))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_markdown(results), encoding="utf-8")
    log.info("Đã sinh %s", out)


if __name__ == "__main__":
    main()
