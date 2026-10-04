#!/usr/bin/env python
"""
export_edge.py — nén và xuất model cho thiết bị biên, kèm ĐO LẠI accuracy.

CHỦ: Phong Trần

Làm 3 việc cho mỗi run:
  1. Lượng tử hoá int8 (dynamic, và static nếu kiến trúc hỗ trợ)
  2. Xuất TorchScript + ONNX, rồi KIỂM ONNX cho ra đúng kết quả như PyTorch
  3. Bảng so sánh: dung lượng / latency / accuracy của fp32 vs int8

★ Cột accuracy là bắt buộc. Báo "nhẹ hơn 4 lần" mà không nói mất bao nhiêu điểm
accuracy là báo cáo thiếu trung thực.

    python scripts/export_edge.py --runs "artifacts/runs/m1_lenet*"
    python scripts/export_edge.py --runs "artifacts/runs/*" --max-samples 3000
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from _common import find_run_dirs, free_memory, load_run, make_dataset
from gtsrb.deploy.export import (export_onnx, export_torchscript, model_size_mb,
                                 quantize_dynamic, quantize_static, verify_onnx)
from gtsrb.deploy.speed import benchmark
from gtsrb.eval.metrics import evaluate
from gtsrb.utils.logging import get_logger
from gtsrb.utils.seed import set_seed

log = get_logger()


def subset_loader(dataset, n: int | None, batch_size: int = 128) -> DataLoader:
    """Loader trên tập con lấy RẢI ĐỀU (không phải n ảnh đầu).

    index.csv sắp theo lớp nên n ảnh đầu toàn lớp 0 — đo accuracy trên đó là vô nghĩa.
    """
    if n is None or n >= len(dataset):
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    step = max(1, len(dataset) // n)
    return DataLoader(Subset(dataset, list(range(0, len(dataset), step))[:n]),
                      batch_size=batch_size, shuffle=False)


def measure(model, name: str, loader, img_size: int, cpu: torch.device) -> dict:
    """Đo một biến thể: dung lượng + accuracy + latency."""
    stats = evaluate(model, loader, cpu)
    timing = benchmark(model, img_size, cpu, batch_sizes=(1,), warmup=10, iters=40)
    return {
        "biến thể": name,
        "dung lượng (MB)": round(model_size_mb(model), 2),
        "top-1": round(stats["top1"], 5),
        "macro-F1": round(stats["macro_f1"], 5),
        "p50 (ms)": round(timing["cpu_bs1_p50"], 2),
        "p95 (ms)": round(timing["cpu_bs1_p95"], 2),
    }


def process_run(run_dir: Path, args, out_dir: Path) -> pd.DataFrame | None:
    """Nén + xuất + đo cho một run. Trả bảng so sánh các biến thể."""
    log.info("")
    log.info("=" * 70)
    log.info("%s", run_dir.name)

    cpu = torch.device("cpu")
    model, cfg, _ = load_run(run_dir, cpu)
    name, img_size = model.model_name, cfg.data.img_size

    dataset = make_dataset(cfg, "test")
    test_loader = subset_loader(dataset, args.max_samples)
    calib_loader = subset_loader(dataset, 512, batch_size=64)

    rows = [measure(model, "fp32", test_loader, img_size, cpu)]

    log.info("  lượng tử hoá động (Linear -> int8)...")
    rows.append(measure(quantize_dynamic(model), "int8_dynamic",
                        test_loader, img_size, cpu))

    log.info("  lượng tử hoá tĩnh (Conv + activation -> int8)...")
    static = quantize_static(model, calib_loader, img_size=img_size)
    if static is not None:
        rows.append(measure(static, "int8_static", test_loader, img_size, cpu))

    # ---- Xuất ----
    log.info("  xuất TorchScript và ONNX...")
    export_torchscript(model, img_size, out_dir / f"{name}.torchscript.pt")
    onnx_path = export_onnx(model, img_size, out_dir / f"{name}.onnx")
    if onnx_path:
        verify_onnx(onnx_path, model, img_size)

    frame = pd.DataFrame(rows)
    base = frame.iloc[0]
    frame["nhẹ hơn"] = (base["dung lượng (MB)"] / frame["dung lượng (MB)"]).round(2)
    frame["nhanh hơn"] = (base["p50 (ms)"] / frame["p50 (ms)"]).round(2)
    frame["mất top-1 (điểm)"] = ((base["top-1"] - frame["top-1"]) * 100).round(3)
    frame.insert(0, "model", name)

    log.info("\n%s", frame.to_string(index=False))
    free_memory(model, cpu)
    return frame


def main() -> None:
    """Nén int8 + xuất TorchScript/ONNX cho mọi run, kèm bảng đánh đổi."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=["artifacts/runs/*"])
    parser.add_argument("--out-dir", default="artifacts/export")
    parser.add_argument("--table", default="reports/tables/edge_export.csv")
    parser.add_argument("--max-samples", type=int, default=4000,
                        help="so anh test de do accuracy (int8 chay tren CPU rat cham)")
    args = parser.parse_args()

    set_seed(42, deterministic=False)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = [f for f in (process_run(d, args, out_dir)
                          for d in find_run_dirs(args.runs)) if f is not None]
    if not frames:
        return

    combined = pd.concat(frames, ignore_index=True)
    table = Path(args.table)
    table.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(table, index=False)

    log.info("")
    log.info("=" * 70)
    log.info("BẢNG NÉN MODEL -> %s", table)
    log.info("File đã xuất   -> %s/", out_dir)
    log.info("")
    log.info("CÁCH ĐỌC: cột 'mất top-1' là CÁI GIÁ của việc nén. Nếu nén làm nhẹ 4 lần")
    log.info("  mà mất 2 điểm accuracy thì với bài biển báo đó là đánh đổi tồi.")
    log.info("  Và chú ý: model dùng GlobalAvgPool (M2) hầu như không nhẹ đi khi")
    log.info("  lượng tử hoá động, vì nó gần như không có tham số Linear.")


if __name__ == "__main__":
    main()
