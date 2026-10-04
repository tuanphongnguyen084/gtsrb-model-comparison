#!/usr/bin/env python
"""
benchmark_speed.py — đo latency, FLOPs, dung lượng; vẽ Pareto accuracy-latency.

CHỦ: Phong Trần

Ví dụ:
    python scripts/benchmark_speed.py --runs "artifacts/runs/*"
    python scripts/benchmark_speed.py --runs "artifacts/runs/*" --devices cpu mps
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import pandas as pd
import torch

from gtsrb.deploy.speed import benchmark, environment_info, model_cost
from _common import (find_run_dirs, free_memory, load_run, read_result,
                     write_result)
from gtsrb.utils.logging import get_logger
from gtsrb.utils.seed import set_seed

log = get_logger()


def plot_pareto(frame: pd.DataFrame, out_path: Path, latency_col: str) -> None:
    """Pareto accuracy-latency: trả lời 'gắn lên xe thật thì chọn model nào'."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = frame.dropna(subset=["test_top1", latency_col])
    if valid.empty:
        log.warning("Chưa có số test_top1 -> bỏ qua Pareto. "
                    "Chạy scripts/evaluate.py trước.")
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(valid[latency_col], valid["test_top1"] * 100, s=90, zorder=3)
    for row in valid.itertuples(index=False):
        ax.annotate(row.model, (getattr(row, latency_col), row.test_top1 * 100),
                    textcoords="offset points", xytext=(8, 5), fontsize=9)

    # Đường biên Pareto: các điểm không bị model nào vừa nhanh hơn VỪA chính xác hơn
    ordered = valid.sort_values(latency_col)
    frontier_x, frontier_y, best = [], [], -1.0
    for row in ordered.itertuples(index=False):
        accuracy = row.test_top1 * 100
        if accuracy > best:
            best = accuracy
            frontier_x.append(getattr(row, latency_col))
            frontier_y.append(accuracy)
    ax.plot(frontier_x, frontier_y, "--", alpha=0.6, zorder=2,
            label="biên Pareto")

    ax.set_xlabel(f"Latency {latency_col} (ms, càng nhỏ càng tốt)")
    ax.set_ylabel("Test top-1 (%, càng cao càng tốt)")
    ax.set_title("Đánh đổi accuracy – latency cho triển khai biên")
    ax.grid(alpha=0.3, zorder=1)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    log.info("Pareto -> %s", out_path)


def pick_devices(requested: list[str] | None) -> list[str]:
    """Chọn thiết bị để đo. Mặc định: CPU + GPU nếu có.

    Đo cả CPU lẫn GPU vì thiết bị biên thật (Raspberry Pi, Jetson Nano) gần CPU
    hơn là gần GPU máy bàn — con số CPU mới là con số đáng tin cho triển khai.
    """
    if requested is not None:
        return requested
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    elif torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def benchmark_one_run(run_dir: Path, devices: list[str], args) -> dict:
    """Đo một run trên mọi thiết bị. Trả một dòng cho bảng tốc độ."""
    log.info("")
    log.info("=" * 68)
    log.info("Đo %s", run_dir.name)

    result = read_result(run_dir)
    row = {
        "run_id": run_dir.name,
        "test_top1": (result.get("test") or {}).get("top1"),
        "test_macro_f1": (result.get("test") or {}).get("macro_f1"),
    }

    cost_recorded = None
    for device_name in devices:
        device = torch.device(device_name)
        model, cfg, _ = load_run(run_dir, device)
        row["model"] = model.model_name
        row["img_size"] = cfg.data.img_size

        # #params / FLOPs / dung lượng không phụ thuộc thiết bị -> chỉ tính một lần
        if cost_recorded is None:
            cost_recorded = model_cost(model, cfg.data.img_size)
            row.update(cost_recorded)
            log.info("  params=%.2fM  FLOPs=%s G  size=%.1f MB",
                     cost_recorded["params_m"], cost_recorded["flops_g"],
                     cost_recorded["size_mb"])

        log.info("  thiết bị %s:", device_name)
        row.update(benchmark(model, cfg.data.img_size, device,
                             batch_sizes=tuple(args.batch_sizes),
                             warmup=args.warmup, iters=args.iters))
        free_memory(model, device)

    # Ghi lại vào result.json để make_report.py tổng hợp tự động
    result["cost"] = {k: v for k, v in row.items()
                      if k not in ("run_id", "model", "test_top1", "test_macro_f1")}
    result["cost"]["environment"] = environment_info(torch.device(devices[-1]))
    write_result(run_dir, result)
    return row


def print_speed_table(frame: pd.DataFrame, out_path: Path) -> None:
    """Lưu và in bảng tốc độ, kèm ghi chú về chuyện FLOPs không phải latency."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)

    log.info("")
    log.info("=" * 68)
    log.info("BẢNG TỐC ĐỘ -> %s", out_path)
    columns = [c for c in ("model", "params_m", "flops_g", "size_mb", "test_top1")
               if c in frame.columns]
    columns += [c for c in frame.columns if c.endswith("_p50") or c.endswith("_p95")]
    log.info("\n%s", frame[columns].to_string(index=False))

    log.info("")
    log.info("GHI CHÚ ĐỌC BẢNG — FLOPs KHÔNG PHẢI LATENCY:")
    log.info("  MobileNetV2 ít FLOPs hơn ResNet18 ~6 lần nhưng latency thường không")
    log.info("  nhanh hơn tương ứng, vì depthwise conv bị chặn bởi BĂNG THÔNG BỘ NHỚ")
    log.info("  chứ không bởi năng lực tính toán. Xem src/gtsrb/deploy/speed.py.")


def main() -> None:
    """Đo latency + FLOPs cho mọi run, vẽ Pareto accuracy-latency."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=["artifacts/runs/*"])
    parser.add_argument("--devices", nargs="+", default=None,
                        help="mac dinh: cpu + (cuda hoac mps neu co)")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 64])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--out", default="reports/tables/speed.csv")
    parser.add_argument("--pareto", default="reports/figures/pareto.png")
    args = parser.parse_args()

    # benchmark=True cho cuDNN tự dò thuật toán nhanh nhất -> đo tốc độ phải BẬT.
    # (Khi so sánh ACCURACY thì ngược lại, phải tắt để kết quả tiền định.)
    set_seed(42, deterministic=False)

    devices = pick_devices(args.devices)
    rows = [benchmark_one_run(run_dir, devices, args)
            for run_dir in find_run_dirs(args.runs)]

    frame = pd.DataFrame(rows)
    print_speed_table(frame, Path(args.out))

    # Pareto theo latency batch=1 — đúng tình huống xe thật (xử lý từng khung ảnh)
    latency_col = next((c for c in frame.columns if c.endswith("_bs1_p50")), None)
    if latency_col:
        plot_pareto(frame, Path(args.pareto), latency_col)


if __name__ == "__main__":
    main()
