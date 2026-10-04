#!/usr/bin/env python
"""
run_robustness.py — 5 loại nhiễu x 5 mức x N model = bảng robustness.

CHỦ: Huy

★ NHIỄU CHỈ ÁP Ở TEST, KHÔNG BAO GIỜ Ở TRAIN ★
Robustness nghĩa là khái quát sang phân phối CHƯA TỪNG THẤY. Train trên chúng
rồi test trên chúng là test in-distribution — phép đo mất ý nghĩa.

Ví dụ:
    python scripts/run_robustness.py --runs "artifacts/runs/*"
    python scripts/run_robustness.py --runs "artifacts/runs/*" --max-samples 2000
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import pandas as pd
import torch

from gtsrb.robustness.benchmark import mean_corruption_error, robustness_sweep
from gtsrb.robustness.corruptions import CORRUPTIONS, SEVERITIES
from _common import find_run_dirs, free_memory, load_run, make_dataset
from gtsrb.utils.logging import get_logger
from gtsrb.utils.seed import pick_device, set_seed

log = get_logger()


def plot_curves(frame: pd.DataFrame, out_path: Path) -> None:
    """Một hình, 5 ô con (một ô mỗi loại nhiễu), mỗi đường là một model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(CORRUPTIONS), figsize=(4 * len(CORRUPTIONS), 4),
                             sharey=True)
    titles = {
        "motion_blur": "Mờ chuyển động", "gauss_noise": "Nhiễu Gaussian",
        "fog": "Sương mù", "low_light": "Thiếu sáng", "occlusion": "Bị che",
    }

    for ax, kind in zip(axes, CORRUPTIONS):
        part = frame[frame["corruption"] == kind]
        for model_name, group in part.groupby("model"):
            group = group.sort_values("severity")
            ax.plot(group["severity"], group["top1"] * 100, marker="o", label=model_name)
        # Đường accuracy SẠCH làm mốc
        for model_name, group in frame[frame["corruption"] == "clean"].groupby("model"):
            ax.axhline(float(group["top1"].iloc[0]) * 100, ls=":", alpha=0.4)
        ax.set_title(titles.get(kind, kind))
        ax.set_xlabel("mức độ")
        ax.set_xticks(list(SEVERITIES))
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Test top-1 (%)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Robustness: accuracy giảm theo mức nhiễu "
                 "(đường chấm = accuracy trên ảnh sạch)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    log.info("Hình đường cong -> %s", out_path)


def main() -> None:
    """Quét 5 nhiễu x 5 mức cho mọi run, xuất bảng và đường cong."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=["artifacts/runs/*"])
    parser.add_argument("--out", default="reports/tables/robustness.csv")
    parser.add_argument("--figure", default="reports/figures/robustness_curves.png")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="chi dung N anh test dau (de chay nhanh khi debug)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(42)
    device = pick_device(args.device or "auto")

    run_dirs = find_run_dirs(args.runs)

    # --- Chạy lại thì BỎ QUA model đã có trong CSV ---
    #
    # Nhờ vậy, nếu job chết giữa chừng thì chỉ cần chạy lại đúng lệnh cũ: nó đọc
    # phần đã xong và làm tiếp từ model còn thiếu. Muốn làm lại từ đầu thì xoá file.
    all_frames = []
    done_runs: set[str] = set()
    existing = Path(args.out)
    if existing.exists():
        previous = pd.read_csv(existing)
        if "run_id" in previous.columns:
            done_runs = set(previous["run_id"].unique())
            all_frames.append(previous)
            log.info("Đã có %s với %d model: %s -> bỏ qua chúng",
                     existing, len(done_runs), sorted(done_runs))

    for run_dir in run_dirs:
        if run_dir.name in done_runs:
            log.info("Bỏ qua %s (đã có trong %s)", run_dir.name, existing)
            continue
        log.info("")
        log.info("=" * 68)
        log.info("Robustness cho %s", run_dir.name)
        model, cfg, _ = load_run(run_dir, device)

        # return_uint8=True: ta cần ảnh THÔ để áp nhiễu, rồi mới chuẩn hoá.
        # Áp nhiễu lên ảnh ĐÃ chuẩn hoá là SAI — sigma=20/255 không còn nghĩa gì
        # sau khi đã trừ mean và chia std.
        dataset = make_dataset(cfg, "test", return_uint8=True)

        frame = robustness_sweep(
            model, dataset, device,
            mean=dataset.mean, std=dataset.std,
            model_name=model.model_name,
            batch_size=args.batch_size, seed=args.seed,
            max_samples=args.max_samples,
        )
        frame["run_id"] = run_dir.name
        frame["mCE"] = mean_corruption_error(frame)
        all_frames.append(frame)

        log.info("  mCE = %.4f  (trung bình error qua mọi nhiễu, càng THẤP càng tốt)",
                 frame["mCE"].iloc[0])

        # ★ GHI NGAY sau mỗi model, không đợi tới cuối.
        #
        # VÌ SAO: quét robustness cho M3 ở 224x224 mất ~40 phút. Nếu chỉ ghi CSV ở
        # cuối thì job chết ở model thứ 3 là mất sạch công của cả 3 model. Ghi dần
        # thì mất nhiều nhất là model đang dở.
        # Đây là lỗi ĐÃ GẶP THẬT: một job robustness bị kill giữa chừng và không
        # để lại gì (xem docs/SU_CO.md).
        partial = Path(args.out)
        partial.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(all_frames, ignore_index=True).to_csv(partial, index=False)
        log.info("  đã ghi tạm %s (%d/%d model)", partial, len(all_frames), len(run_dirs))

        free_memory(model, device)

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(args.out, index=False)       # ghi lần cuối cho đủ
    log.info("")
    log.info("=" * 68)
    log.info("BẢNG ROBUSTNESS -> %s (%d dòng)", args.out, len(combined))

    # --- Bảng tóm tắt: relative robustness trung bình theo loại nhiễu ---
    summary = (combined[combined["corruption"] != "clean"]
               .pivot_table(index="model", columns="corruption",
                            values="relative_top1", aggfunc="mean"))
    summary["mCE"] = combined.groupby("model")["mCE"].first()
    summary["clean_top1"] = (combined[combined["corruption"] == "clean"]
                             .set_index("model")["top1"])
    summary_path = Path(args.out).with_name("robustness_summary.csv")
    summary.to_csv(summary_path)
    log.info("")
    log.info("RELATIVE ROBUSTNESS trung bình (= acc_nhiễu / acc_sạch):")
    log.info("\n%s", summary.round(4).to_string())
    log.info("")
    log.info("CÁCH ĐỌC: relative robustness tách 'giỏi sẵn' khỏi 'BỀN'.")
    log.info("  Nếu model có clean_top1 cao nhất mà relative KHÔNG cao nhất,")
    log.info("  thì model accuracy cao nhất KHÔNG phải model bền nhất — đó là")
    log.info("  phát hiện đáng giá nhất của phần robustness.")

    plot_curves(combined, Path(args.figure))


if __name__ == "__main__":
    main()
