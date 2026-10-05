#!/usr/bin/env python
"""
evaluate.py — đánh giá trên TẬP TEST, so sánh các model, kiểm định McNemar.

CHỦ: Phong Nguyễn

★ TẬP TEST CHỈ ĐƯỢC MỞ MỘT LẦN, Ở CUỐI ★
Mọi quyết định (chọn kiến trúc, siêu tham số, early stopping, chọn cấu hình ablation
thắng) chỉ dựa trên tập VAL. Nếu nhìn test rồi quay lại sửa model thì test đã trở
thành val, và con số báo cáo không còn là ước lượng không chệch.

Script làm 5 việc, mỗi việc một hàm bên dưới:
  1. evaluate_one_run   nạp checkpoint -> chỉ số + per-class + confusion + ECE
  2. save_per_class     bảng 43 lớp + 10 lớp yếu nhất
  3. save_confusion     confusion matrix + 10 cặp nhầm nhiều nhất
  4. build_main_table   bảng so sánh chính
  5. run_mcnemar_pairs  kiểm định cho MỌI cặp model

Ví dụ:
    python scripts/evaluate.py --runs "artifacts/runs/*"
    python scripts/evaluate.py --runs artifacts/runs/m1_lenet_s42_20261003_222148
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import itertools
from pathlib import Path

import pandas as pd
import torch

from _common import (TAG_CHINH, find_run_dirs, free_memory, load_run,
                     make_test_loader,
                     read_result, write_result)
from gtsrb.eval.calibration import expected_calibration_error, plot_reliability
from gtsrb.eval.confusion import confusion, plot_confusion, top_confusions
from gtsrb.eval.metrics import evaluate, worst_classes
from gtsrb.eval.stats_tests import mcnemar
from gtsrb.models.registry import count_parameters
from gtsrb.utils.logging import get_logger
from gtsrb.utils.seed import pick_device, set_seed

log = get_logger()


# =====================================================================
# 1 — Đánh giá một run
# =====================================================================

def evaluate_one_run(run_dir: Path, device: torch.device, batch_size: int) -> dict:
    """Nạp checkpoint, chạy trên tập test, in và trả về mọi chỉ số."""
    log.info("")
    log.info("=" * 68)
    log.info("Đánh giá %s", run_dir.name)

    model, cfg, _ = load_run(run_dir, device)
    stats = evaluate(model, make_test_loader(cfg, batch_size), device)
    calibration = expected_calibration_error(stats["y_true"], stats["y_prob"])

    log.info("  top-1      = %.4f", stats["top1"])
    log.info("  top-5      = %.4f   <- bão hoà trên 43 lớp, chỉ số YẾU ở đây",
             stats["top5"])
    log.info("  macro-F1   = %.4f   <- CHỈ SỐ CHÍNH", stats["macro_f1"])
    log.info("  weighted-F1= %.4f", stats["weighted_f1"])
    log.info("  ECE        = %.4f   <- càng nhỏ càng hiệu chỉnh tốt", calibration["ece"])

    out = {"model_key": model.model_name, "cfg": cfg, "stats": stats,
           "calibration": calibration, "params_m": count_parameters(model) / 1e6}
    free_memory(model, device)
    return out


# =====================================================================
# 2 — Per-class
# =====================================================================

def save_per_class(model_key: str, stats: dict, tables_dir: Path) -> None:
    """Lưu bảng 43 lớp và in 10 lớp yếu nhất.

    Hai nguyên nhân khó phải TÁCH RA, không gộp:
      F1 thấp + support CAO  -> hình giống nhau (vấn đề độ phân giải)
      F1 thấp + support THẤP -> thiếu dữ liệu
    """
    stats["per_class"].to_csv(tables_dir / f"per_class_{model_key}.csv", index=False)

    log.info("  10 lớp yếu nhất (F1 thấp + support cao -> do HÌNH GIỐNG NHAU;")
    log.info("                   F1 thấp + support thấp -> do THIẾU DỮ LIỆU):")
    for row in worst_classes(stats["per_class"], 10).itertuples(index=False):
        log.info("    lớp %2d %-34s F1=%.3f  n=%d",
                 row.class_id, row.name[:34], row.f1, row.support)


# =====================================================================
# 3 — Confusion
# =====================================================================

def save_confusion(model_key: str, stats: dict,
                   tables_dir: Path, figures_dir: Path) -> None:
    """Confusion matrix 43x43 + bảng 10 cặp nhầm nhiều nhất.

    Bảng cặp dễ đọc hơn hình 43x43 nhiều, và nó cho biết lỗi ĐI ĐÂU —
    điều mà accuracy không cho biết.
    """
    matrix = confusion(stats["y_true"], stats["y_pred"], normalize=True)
    plot_confusion(matrix, figures_dir / f"confusion_{model_key}.png",
                   title=f"Confusion matrix — {model_key} (chuẩn hoá theo hàng)")

    pairs = top_confusions(stats["y_true"], stats["y_pred"], 10)
    pairs.to_csv(tables_dir / f"top_confusions_{model_key}.csv", index=False)

    log.info("  10 cặp nhầm nhiều nhất:")
    for row in pairs.itertuples(index=False):
        log.info("    %2d %-26s -> %2d %-26s  %d lần",
                 row.true_id, row.true_name[:26], row.pred_id,
                 row.pred_name[:26], row.count)


# =====================================================================
# 4 — Bảng so sánh chính
# =====================================================================

def build_main_table(rows: list[dict], out_path: Path,
                     rebuild: bool = False) -> pd.DataFrame:
    """GỘP các run vừa đánh giá vào bảng cũ, sắp theo macro-F1 giảm dần.

    ★ GỘP, KHÔNG GHI ĐÈ — tôi đã mắc lỗi này HAI LẦN ★

    Bản đầu ghi đè thẳng. Chạy `evaluate.py --runs "artifacts/runs/*res112real*"`
    để lấy số test cho MỘT run làm bảng chính từ 7 dòng xuống còn 1 — và dòng
    đó lại là run ablation (có tag), nên bảng KHÔNG CÒN RUN CHÍNH NÀO.
    make_report.py crash ngay sau đó.

    Lần sửa thứ nhất tôi chỉ thêm CẢNH BÁO khi số run chính giảm. Rồi tôi mắc
    lại y nguyên — và lần đó guard IM LẶNG, vì bảng đã bị hỏng từ trước nên
    "0 run chính -> 0 run chính" không phải là giảm. Một cảnh báo chỉ nổ ở lần
    đầu thì vô dụng đúng lúc cần nhất.

    Nên giờ hành vi mặc định là GỘP: run nào vừa đánh giá thì cập nhật dòng của
    nó, các run khác giữ nguyên. Muốn dựng lại từ đầu (để bỏ run đã xoá khỏi
    đĩa) thì truyền `--rebuild`.
    """
    moi = pd.DataFrame(rows)

    if not rebuild and out_path.exists():
        try:
            cu = pd.read_csv(out_path)
            if "run_id" in cu.columns and "run_id" in moi.columns:
                giu = cu[~cu["run_id"].isin(set(moi["run_id"]))]
                n_giu = len(giu)
                moi = pd.concat([giu, moi], ignore_index=True)
                if n_giu:
                    log.info("Gộp vào bảng cũ: giữ %d run không đánh giá lần này, "
                             "cập nhật %d run", n_giu, len(rows))
        except Exception as exc:
            log.warning("Không đọc được bảng cũ (%s) -> dựng mới", exc)

    frame = moi.sort_values("test_macro_f1", ascending=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)

    log.info("")
    log.info("=" * 68)
    log.info("BẢNG SO SÁNH CHÍNH -> %s", out_path)
    log.info("\n%s", frame.to_string(index=False))
    return frame


def row_for_table(run_dir: Path, evaluated: dict, result: dict) -> dict:
    """Một dòng của bảng so sánh chính."""
    cfg, stats = evaluated["cfg"], evaluated["stats"]
    train_info = result.get("train", {})
    return {
        "run_id": run_dir.name,
        "model": evaluated["model_key"],
        # tag rỗng = run CHÍNH (M1/M2/M3). tag có giá trị = run ablation hoặc seed.
        # Cột này để make_report.py lọc: bảng so sánh chính và khuyến nghị triển
        # khai chỉ được xét run chính, nếu không thì một biến thể ablation có thể
        # lọt vào và bị chọn làm "model tốt nhất".
        "tag": result.get("notes", "") or TAG_CHINH,
        "img_size": cfg.data.img_size,
        "preprocess": cfg.data.preprocess,
        "aug_policy": cfg.data.aug_policy,
        "label_smoothing": cfg.train.get("label_smoothing"),
        "params_m": round(evaluated["params_m"], 3),
        "test_top1": round(stats["top1"], 5),
        "test_top5": round(stats["top5"], 5),
        "test_macro_f1": round(stats["macro_f1"], 5),
        "test_weighted_f1": round(stats["weighted_f1"], 5),
        "test_ece": round(evaluated["calibration"]["ece"], 5),
        "val_best_macro_f1": result.get("val", {}).get("best_macro_f1"),
        "epochs_run": train_info.get("epochs_run"),
        # Ưu tiên train_seconds_clean: nếu máy ngủ giữa chừng thì số thô vô nghĩa
        # (đã gặp thật: 34.371s thay vì 3.318s). Xem _detect_sleep() trong engine.
        "train_seconds": train_info.get("train_seconds_clean",
                                        train_info.get("train_seconds")),
        "train_seconds_raw": train_info.get("train_seconds"),
        "train_time_suspect": train_info.get("train_seconds_suspect", False),
    }


# =====================================================================
# 5 — McNemar cho mọi cặp
# =====================================================================

def run_mcnemar_pairs(predictions: dict, tables_dir: Path) -> None:
    """Kiểm định McNemar cho MỌI cặp model.

    Hai model chạy trên CÙNG tập test -> quan sát bắt cặp -> dùng t-test hai mẫu
    độc lập là SAI về mặt thống kê. Chỉ hai ô BẤT ĐỒNG (n01, n10) mang thông tin.
    """
    if len(predictions) < 2:
        log.info("Chỉ có 1 model -> bỏ qua McNemar (cần ít nhất 2 để so sánh).")
        return

    log.info("")
    log.info("=" * 68)
    log.info("KIỂM ĐỊNH McNEMAR — 'chênh lệch này có THẬT không?'")
    log.info("(hai model chạy trên CÙNG tập test -> quan sát bắt cặp,")
    log.info(" dùng t-test độc lập là SAI về mặt thống kê)")

    rows = []
    for name_a, name_b in itertools.combinations(sorted(predictions), 2):
        y_true, pred_a = predictions[name_a]
        _, pred_b = predictions[name_b]
        test = mcnemar(y_true, pred_a, pred_b)

        log.info("")
        log.info("  %s  vs  %s", name_a, name_b)
        log.info("    n01=%d (chỉ A đúng)  n10=%d (chỉ B đúng)  method=%s",
                 test["n01"], test["n10"], test["method"])
        log.info("    p-value = %.6f -> %s", test["p_value"], test["verdict"])
        rows.append({"model_a": name_a, "model_b": name_b, **test})

    out = tables_dir / "mcnemar.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info("")
    log.info("McNemar -> %s", out)


# =====================================================================
# Điều phối
# =====================================================================

def main() -> None:
    """Đánh giá mọi run trên tập test, xuất bảng so sánh và chạy McNemar."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=["artifacts/runs/*"])
    parser.add_argument("--out", default="reports/tables/main_comparison.csv")
    parser.add_argument("--rebuild", action="store_true",
                        help="Dựng lại bảng chính TỪ ĐẦU thay vì gộp vào bảng "
                             "cũ. Dùng khi đã xoá run khỏi đĩa.")
    parser.add_argument("--figures", default="reports/figures")
    parser.add_argument("--tables", default="reports/tables")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--no-mcnemar", action="store_true")
    args = parser.parse_args()

    set_seed(42)
    device = pick_device(args.device or "auto")
    figures_dir, tables_dir = Path(args.figures), Path(args.tables)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = find_run_dirs(args.runs)
    log.info("Tìm thấy %d run: %s", len(run_dirs), [d.name for d in run_dirs])

    rows: list[dict] = []
    predictions: dict[str, tuple] = {}        # dùng cho McNemar

    for run_dir in run_dirs:
        evaluated = evaluate_one_run(run_dir, device, args.batch_size)
        model_key, stats = evaluated["model_key"], evaluated["stats"]

        # ★ CHỈ run CHÍNH mới vào McNemar.
        #
        # LỖI ĐÃ GẶP: `predictions` đánh khoá theo TÊN MODEL, mà nhiều run có cùng
        # tên (m1_lenet thật, m1_lenet của thí nghiệm rò rỉ, 12 biến thể ablation
        # của m2_vggres...). Run xử lý sau GHI ĐÈ run trước, nên McNemar có thể
        # đang so một biến thể ablation thay vì model thật — mà không báo gì.
        tag = read_result(run_dir).get("notes", "") or ""
        if tag:
            log.info("  (run phụ, tag=%r — không đưa vào McNemar)", tag)
        elif model_key in predictions:
            log.warning("  TRÙNG TÊN: đã có run chính tên %r. Giữ run ĐẦU TIÊN, "
                        "bỏ qua %s cho McNemar.", model_key, run_dir.name)
        else:
            predictions[model_key] = (stats["y_true"], stats["y_pred"])

        save_per_class(model_key, stats, tables_dir)
        save_confusion(model_key, stats, tables_dir, figures_dir)
        plot_reliability(evaluated["calibration"]["bins"],
                         figures_dir / f"reliability_{model_key}.png",
                         title=f"Reliability — {model_key}",
                         ece=evaluated["calibration"]["ece"])

        # Ghi số test vào lại result.json để make_report.py tổng hợp tự động
        result = read_result(run_dir)
        result["test"] = {
            "top1": round(stats["top1"], 6),
            "top5": round(stats["top5"], 6),
            "macro_f1": round(stats["macro_f1"], 6),
            "weighted_f1": round(stats["weighted_f1"], 6),
            "ece": round(evaluated["calibration"]["ece"], 6),
            "per_class_f1": [round(v, 6) for v in stats["per_class"]["f1"].tolist()],
        }
        write_result(run_dir, result)
        rows.append(row_for_table(run_dir, evaluated, result))

    build_main_table(rows, Path(args.out), rebuild=args.rebuild)
    if not args.no_mcnemar:
        run_mcnemar_pairs(predictions, tables_dir)


if __name__ == "__main__":
    main()
