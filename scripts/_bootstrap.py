"""Thêm src/ vào sys.path để chạy script trực tiếp, không cần `pip install -e .`

Mọi script trong scripts/ import file này ở DÒNG ĐẦU TIÊN.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
