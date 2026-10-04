#!/usr/bin/env python
"""
fix_sleep_times.py — bổ sung `train_seconds_clean` cho các run train bằng code CŨ.

CHỦ: Hoàng

VÌ SAO CÓ FILE NÀY: `_detect_sleep()` được thêm vào `engine/train.py` SAU khi một số
run đã chạy xong. Những run đó có `train_seconds` bị máy ngủ làm hỏng (ví dụ 34.370s
thay vì ~2.800s) nhưng không có trường `train_seconds_clean` để báo cáo dùng.

★ Script này KHÔNG sửa tay số liệu. Nó gọi ĐÚNG hàm `_detect_sleep()` mà engine dùng,
trên ĐÚNG dữ liệu `train_log.csv` mà run đó đã ghi. Kết quả vì vậy giống hệt như khi
run đó được train bằng code mới — truy vết được, lặp lại được.

Giữ nguyên `train_seconds` thô và chỉ THÊM trường mới, để người đọc báo cáo thấy
cả số thô lẫn số ước lượng và biết đã có can thiệp.

    python scripts/fix_sleep_times.py            # xem sẽ sửa gì
    python scripts/fix_sleep_times.py --apply    # ghi thật
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

from gtsrb.engine.train import _detect_sleep
from gtsrb.utils.logging import get_logger

log = get_logger()


def main() -> None:
    """Quét mọi run, bổ sung train_seconds_clean cho run nào còn thiếu."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="artifacts/runs/*")
    parser.add_argument("--apply", action="store_true",
                        help="ghi that (mac dinh chi xem truoc)")
    args = parser.parse_args()

    changed = 0
    for run_dir in sorted(Path(p) for p in glob.glob(args.runs)):
        result_path = run_dir / "result.json"
        log_path = run_dir / "train_log.csv"
        if not (result_path.exists() and log_path.exists()):
            continue

        result = json.loads(result_path.read_text(encoding="utf-8"))
        if "train_seconds_clean" in result.get("train", {}):
            continue        # đã có, bỏ qua

        history = pd.read_csv(log_path).to_dict("records")
        info = _detect_sleep(history)

        raw = result["train"].get("train_seconds")
        result["train"]["train_seconds_clean"] = round(info["clean_seconds"], 1)
        result["train"]["train_seconds_suspect"] = info["suspect"]
        result["train"]["train_time_note"] = info["note"]

        mark = "NGHI NGỜ" if info["suspect"] else "bình thường"
        log.info("%-46s thô %9.0fs -> sạch %8.0fs  [%s]",
                 run_dir.name, raw or 0, info["clean_seconds"], mark)

        if args.apply:
            result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        changed += 1

    if not changed:
        log.info("Mọi run đã có train_seconds_clean, không cần sửa.")
    elif args.apply:
        log.info("Đã cập nhật %d run.", changed)
    else:
        log.info("Xem trước %d run. Thêm --apply để ghi thật.", changed)


if __name__ == "__main__":
    main()
