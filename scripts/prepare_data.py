#!/usr/bin/env python
"""
prepare_data.py — bước ĐẦU TIÊN của dự án. Chạy cái này trước mọi thứ khác.

CHỦ: Huy

Nó làm 5 việc, theo thứ tự:
  1. Tải GTSRB (nếu có --download hoặc chưa có dữ liệu)
  2. Dựng data/processed/index.csv  (kèm cột track_id — khoá chống rò rỉ)
  3. Chia train/val THEO TRACK      (bước quan trọng nhất)
  4. Tiền xử lý + resize tất cả ảnh, lưu cache .npy  (để train nhanh)
  5. Tính mean/std CHỈ trên tập train, ghi vào configs/base.yaml

Ví dụ:
    python scripts/prepare_data.py --download
    python scripts/prepare_data.py --preprocess clahe --img-size 48
    python scripts/prepare_data.py --preprocess none --img-size 64   # cho ablation
    python scripts/prepare_data.py --no-group-by-track               # DOI CHUNG ro ri
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (phải import đầu tiên, thêm src/ vào sys.path)

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from gtsrb.data.download import build_index, download_gtsrb
from gtsrb.data.dataset import cache_path
from gtsrb.data.preprocess import PREPROCESS_MODES, compute_mean_std, load_and_process
from gtsrb.data.split import check_leakage, make_split
from gtsrb.utils.logging import get_logger

log = get_logger()


def build_cache(index_csv: Path, out_path: Path, size: int, mode: str,
                roi_margin: float = 0.10) -> np.ndarray:
    """Tiền xử lý TẤT CẢ ảnh một lần, lưu thành một mảng numpy duy nhất.

    VÌ SAO CACHE: giải nén .ppm + CLAHE + resize cho 51.839 ảnh mất ~2 phút.
    Làm lại mỗi epoch thì GPU ngồi chờ CPU và train chậm gấp nhiều lần.
    Làm một lần, lưu ra (N, size, size, 3) uint8, rồi Dataset mở bằng mmap.

    Dung lượng: 51.839 * 48 * 48 * 3 byte = ~358 MB ở size=48.
    """
    frame = pd.read_csv(index_csv)
    total = len(frame)
    images = np.zeros((total, size, size, 3), dtype=np.uint8)

    log.info("Tiền xử lý %d ảnh: mode=%s, size=%dx%d", total, mode, size, size)
    for position, row in enumerate(tqdm(frame.itertuples(index=False),
                                        total=total, ncols=88, desc="xu ly anh",
                                        disable=not sys.stderr.isatty())):
        roi = (int(row.roi_x1), int(row.roi_y1), int(row.roi_x2), int(row.roi_y2))
        images[position] = load_and_process(str(row.path), roi, size, mode, roi_margin)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, images)
    log.info("Đã lưu cache %s (%.0f MB)", out_path, out_path.stat().st_size / 1e6)
    return images


def update_base_config(mean: list[float], std: list[float],
                       config_path: Path = Path("configs/base.yaml")) -> None:
    """Ghi mean/std vào configs/base.yaml.

    Tự ghi vào config thay vì in ra cho người copy tay — copy tay là chỗ hay sai,
    và sai mean/std thì model vẫn train bình thường, chỉ kém đi mà KHÔNG BÁO LỖI.
    """
    with open(config_path, "r", encoding="utf-8") as fh:
        text = fh.read()

    mean_text = "[" + ", ".join(f"{v:.4f}" for v in mean) + "]"
    std_text = "[" + ", ".join(f"{v:.4f}" for v in std) + "]"

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("mean:"):
            line = f"  mean: {mean_text}"
        elif stripped.startswith("std:"):
            line = f"  std: {std_text}"
        lines.append(line)

    with open(config_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    log.info("Đã ghi mean/std vào %s", config_path)


def parse_args() -> argparse.Namespace:
    """Tham số dòng lệnh. Xem docstring đầu file để biết các ví dụ thường dùng."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--download", action="store_true",
                        help="tai lai du lieu (bo train DAY DU 39.209 anh)")
    parser.add_argument("--img-size", type=int, default=48)
    parser.add_argument("--preprocess", default="clahe", choices=PREPROCESS_MODES)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-group-by-track", action="store_true",
                        help="DOI CHUNG: split random theo anh (CO RO RI)")
    parser.add_argument("--skip-cache", action="store_true",
                        help="chi dung index + split, khong tien xu ly anh")
    parser.add_argument("--roi-margin", type=float, default=0.10)
    return parser.parse_args()


def step_index(raw_dir: Path, index_csv: Path) -> None:
    """Bước 2/5 — quét dữ liệu thành index.csv (nguồn sự thật duy nhất)."""
    log.info("")
    log.info("--- Bước 2/5: dựng index.csv ---")
    frame = build_index(raw_dir, index_csv)
    log.info("Tổng %d ảnh = %d train-pool + %d test chính thức",
             len(frame),
             int((frame["source"] == "train").sum()),
             int((frame["source"] == "test_official").sum()))


def step_split(index_csv: Path, args: argparse.Namespace) -> None:
    """Bước 3/5 — chia train/val THEO TRACK để chống rò rỉ dữ liệu."""
    log.info("")
    log.info("--- Bước 3/5: chia train/val ---")
    make_split(index_csv, val_ratio=args.val_ratio, seed=args.seed,
               group_by_track=not args.no_group_by_track)


def step_cache(index_csv: Path, args: argparse.Namespace) -> tuple[np.ndarray, Path]:
    """Bước 4/5 — tiền xử lý toàn bộ ảnh một lần, lưu thành mảng numpy."""
    log.info("")
    log.info("--- Bước 4/5: tiền xử lý + cache ---")
    out_path = cache_path(args.processed_dir, args.img_size, args.preprocess)
    if out_path.exists():
        log.info("Đã có cache %s, nạp lại (xoá file đó nếu muốn dựng lại)", out_path)
        return np.load(out_path, mmap_mode="r"), out_path
    images = build_cache(index_csv, out_path, args.img_size,
                         args.preprocess, args.roi_margin)
    return images, out_path


def step_stats(index_csv: Path, images: np.ndarray, args: argparse.Namespace) -> None:
    """Bước 5/5 — tính mean/std CHỈ trên tập train rồi ghi vào configs/base.yaml."""
    log.info("")
    log.info("--- Bước 5/5: tính mean/std CHỈ trên tập train ---")
    frame = pd.read_csv(index_csv)
    train_mask = (frame["split"] == "train").to_numpy()
    mean, std = compute_mean_std(np.asarray(images[train_mask]))
    log.info("mean = [%.4f, %.4f, %.4f]", *mean)
    log.info("std  = [%.4f, %.4f, %.4f]", *std)

    # Chỉ cập nhật base.yaml cho cấu hình MẶC ĐỊNH; run ablation (size/mode khác)
    # không được ghi đè mean/std của cấu hình chính.
    if args.preprocess == "clahe" and args.img_size == 48:
        update_base_config(mean, std)
    else:
        log.info("(không ghi vào base.yaml vì đây không phải cấu hình mặc định clahe/48)")


def print_summary(index_csv: Path, cache_file: Path) -> None:
    """In tóm tắt cuối, gồm hai con số phải kiểm bằng mắt trước khi train."""
    frame = pd.read_csv(index_csv)
    report = check_leakage(frame)
    counts = frame["split"].value_counts()
    log.info("")
    log.info("=" * 62)
    log.info("XONG")
    log.info("  index.csv        : %s (%d dòng)", index_csv, len(frame))
    log.info("  cache            : %s", cache_file)
    log.info("  train/val/test   : %d / %d / %d",
             counts.get("train", 0), counts.get("val", 0), counts.get("test", 0))
    log.info("  track dùng chung : %d  <- phải là 0", report["n_shared_tracks"])
    log.info("  lệch phân phối   : %.3f%%  <- phải < 1%%",
             report["max_class_ratio_gap_pct"])
    log.info("=" * 62)


def main() -> None:
    """Chạy 5 bước: tải -> index -> split theo track -> cache -> mean/std."""
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    index_csv = Path(args.processed_dir) / "index.csv"

    # Bước 1/5 — tải (bỏ qua nếu đã có dữ liệu và không yêu cầu tải lại)
    if args.download or not any(raw_dir.glob("*")):
        download_gtsrb(raw_dir)

    step_index(raw_dir, index_csv)
    step_split(index_csv, args)

    if args.skip_cache:
        log.info("Bỏ qua bước cache (--skip-cache)")
        return

    images, cache_file = step_cache(index_csv, args)
    step_stats(index_csv, images, args)
    print_summary(index_csv, cache_file)


if __name__ == "__main__":
    main()
