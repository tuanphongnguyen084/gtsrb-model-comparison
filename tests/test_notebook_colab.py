"""
test_notebook_colab.py — khoá các lệnh trong notebook Colab.

CHỦ: Phong Nguyễn

★ LỖI ĐÃ GẶP THẬT — mất một mẻ Colab 2 giờ ★

Ô ablation gọi `!python scripts/run_ablation.py --axes all --budget
--set train.num_workers=2`, nhưng run_ablation.py KHÔNG CÓ tham số `--set`.
argparse thoát với mã 2. Với `!python`, lệnh lỗi chỉ IN thông báo rồi notebook
CHẠY TIẾP — nên Run all đi tiếp sang ô train M3 và sau 2 giờ cho ra 3 run M3
mà không có run ablation nào, không có gì báo động.

Test này mô phỏng lỗi đó: trích mọi lệnh script từ notebook rồi chạy với
`--help`. argparse từ chối tham số lạ NGAY ở bước phân tích đối số, trước khi
làm gì, nên `--help` đủ để phát hiện mà không tốn một giây train nào.
"""
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "00_colab_train.ipynb"


def _cells() -> list[str]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _lenh_trong_notebook() -> list[list[str]]:
    """Mọi lệnh gọi script của dự án trong notebook, dạng danh sách đối số.

    Nhận cả ba cách viết đang dùng: `chay("python", ...)`, `subprocess.run([...])`
    và `!python scripts/...` (dạng cũ, vẫn phải kiểm nếu còn sót).
    """
    ra: list[list[str]] = []
    for src in _cells():
        # ast.parse không đọc được magic của IPython (`!pip`, `%cd`) — đổi thành
        # chú thích để phần Python còn lại vẫn phân tích được.
        sach = "\n".join("# " + d if d.lstrip().startswith(("!", "%")) else d
                          for d in src.splitlines())
        for cay in ast.walk(ast.parse(sach)):
            if not isinstance(cay, ast.Call):
                continue
            ten = getattr(cay.func, "id", None) or getattr(cay.func, "attr", None)
            if ten not in ("chay", "run"):
                continue
            doi_so = cay.args[0].elts if (
                len(cay.args) == 1 and isinstance(cay.args[0], ast.List)
            ) else cay.args
            try:
                lenh = [ast.literal_eval(a) for a in doi_so]
            except ValueError:
                continue                      # có biến, không đánh giá tĩnh được
            if any(isinstance(x, str) and x.startswith("scripts/") for x in lenh):
                ra.append([str(x) for x in lenh])

        for dong in re.findall(r"^\s*!python\s+(scripts/\S+.*)$", src, re.M):
            ra.append(["python"] + dong.split())
    return ra


def test_notebook_co_lenh_de_kiem():
    assert _lenh_trong_notebook(), "không trích được lệnh nào — regex/AST đã lỗi?"


# Chạy script với ĐÚNG đối số của notebook, nhưng DỪNG ngay sau khi argparse
# phân tích xong — chưa train, chưa tải gì.
#
# ★ CÁCH LÀM ĐẦU TIÊN CỦA TÔI SAI: thêm `--help` vào lệnh.
# argparse xử lý `--help` TRƯỚC khi kiểm các tham số khác, nên nó in help rồi
# thoát 0 và BỎ QUA tham số lạ. Test vẫn xanh dù `--set` không tồn tại — đúng
# lỗi mà nó phải bắt. Phải để argparse chạy parse_args() thật.
_KIEM_DOI_SO = """
import argparse, runpy, sys

class _DaPhanTichXong(Exception):
    pass

_goc = argparse.ArgumentParser.parse_args
def _chan(self, *a, **k):
    _goc(self, *a, **k)          # đối số sai -> SystemExit(2) ngay tại đây
    raise _DaPhanTichXong        # đối số đúng -> dừng, không làm gì thêm
argparse.ArgumentParser.parse_args = _chan

sys.argv = sys.argv[1:]
# `python scripts/x.py` tự thêm scripts/ vào sys.path, còn runpy thì KHÔNG —
# thiếu dòng này thì `import _bootstrap` trong script sẽ lỗi.
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
except _DaPhanTichXong:
    print("DOI_SO_OK")
"""


@pytest.mark.parametrize("lenh", _lenh_trong_notebook(),
                         ids=lambda l: Path(l[1]).stem if len(l) > 1 else "?")
def test_moi_lenh_trong_notebook_duoc_argparse_chap_nhan(lenh):
    ket = subprocess.run([sys.executable, "-c", _KIEM_DOI_SO, *lenh[1:]],
                         cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert "DOI_SO_OK" in ket.stdout, (
        f"notebook gọi lệnh mà script KHÔNG nhận:\n"
        f"  {' '.join(lenh)}\n"
        f"  {(ket.stderr or ket.stdout).strip().splitlines()[-1] if (ket.stderr or ket.stdout).strip() else '(không có thông báo)'}")


def test_khong_con_dung_bang_python_cho_lenh_dai():
    """`!python` làm lỗi IM LẶNG — lệnh nặng phải đi qua chay() để dừng notebook."""
    nang = ("run_ablation.py", "run_seeds.py", "train.py")
    for i, src in enumerate(_cells()):
        for dong in re.findall(r"^\s*!python\s+(scripts/\S+)", src, re.M):
            assert not any(n in dong for n in nang), (
                f"ô code #{i} dùng `!python {dong}` cho một lệnh dài. "
                f"Lệnh lỗi sẽ chỉ in ra rồi notebook chạy tiếp. Dùng chay().")


def test_ham_chay_kiem_ma_thoat():
    """chay() phải THẬT SỰ dừng khi mã thoát khác 0."""
    nguon = "\n".join(_cells())
    assert "def chay(" in nguon, "notebook thiếu hàm chay()"
    than = nguon.split("def chay(", 1)[1].split("\ndef ", 1)[0]
    assert "returncode" in than, "chay() không đọc mã thoát"
    assert "SystemExit" in than or "raise" in than, "chay() không dừng khi lỗi"
