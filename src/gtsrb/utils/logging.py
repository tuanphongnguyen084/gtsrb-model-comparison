"""
logging — thiết lập logger dùng chung.

CHỦ: Huy

QUY TẮC: KHÔNG dùng print() trong src/. Dùng logging, vì:
  - bật/tắt được theo mức (DEBUG khi debug, INFO khi chạy thật)
  - có timestamp, biết bước nào mất bao lâu
  - ghi được đồng thời ra file log của run
Trong notebook thì print() tự do.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str = "gtsrb", log_file: str | Path | None = None,
               level: int = logging.INFO) -> logging.Logger:
    """Tạo (hoặc lấy lại) logger. Gọi nhiều lần không bị nhân đôi dòng log."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:            # đã thiết lập trước đó
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
