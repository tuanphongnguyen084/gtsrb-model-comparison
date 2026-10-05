"""
test_gop_bang_chinh.py — khoá hành vi GỘP của build_main_table().

CHỦ: Phong Nguyễn (tổng hợp báo cáo)

★ LỖI ĐÃ MẮC HAI LẦN ★

build_main_table() bản đầu GHI ĐÈ main_comparison.csv. Chạy
`evaluate.py --runs "artifacts/runs/*res112real*"` để lấy số test cho MỘT run
làm bảng chính từ 7 dòng xuống còn 1 — và dòng đó lại là run ablation (có tag),
nên bảng KHÔNG CÒN RUN CHÍNH NÀO. make_report.py crash ngay sau đó với
NameError: MISSING (một hằng chưa bao giờ được định nghĩa, nằm im suốt dự án
vì chỉ nổ khi bảng chính trống).

Lần sửa thứ nhất chỉ thêm CẢNH BÁO khi số run chính giảm. Rồi tôi mắc lại y
nguyên — và lần đó guard IM LẶNG, vì bảng đã hỏng từ trước nên "0 run chính ->
0 run chính" không phải là giảm.

Bài học: một cảnh báo chỉ nổ ở lần đầu thì vô dụng đúng lúc cần nhất. Cách sửa
đúng là đổi HÀNH VI, không phải thêm thông báo.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from _common import TAG_CHINH, chi_run_chinh
from evaluate import build_main_table


def _hang(run_id: str, model: str, mf1: float, tag: str = TAG_CHINH) -> dict:
    return {"run_id": run_id, "model": model, "tag": tag, "test_macro_f1": mf1,
            "test_top1": mf1 + 0.003}


@pytest.fixture
def bang_cu(tmp_path):
    """Bảng đã có 3 run chính."""
    out = tmp_path / "main_comparison.csv"
    build_main_table([_hang("m1_a", "m1_lenet", 0.9778),
                      _hang("m2_a", "m2_vggres", 0.9880),
                      _hang("m3_a", "m3_resnet18", 0.9903)], out)
    return out


def test_danh_gia_mot_run_KHONG_lam_mat_run_khac(bang_cu):
    """Lõi của lỗi: đánh giá 1 run phụ không được xoá 3 run chính."""
    build_main_table([_hang("abl_x", "m3_resnet18", 0.9890, tag="resolution-res112")],
                     bang_cu)
    d = pd.read_csv(bang_cu)
    assert len(d) == 4, f"phải còn 4 dòng, có {len(d)}"
    assert len(chi_run_chinh(d)) == 3, "ba run CHÍNH phải còn nguyên"


def test_danh_gia_lai_mot_run_thi_CAP_NHAT_chu_khong_nhan_doi(bang_cu):
    build_main_table([_hang("m2_a", "m2_vggres", 0.9999)], bang_cu)
    d = pd.read_csv(bang_cu)
    assert len(d) == 3, f"không được thêm dòng mới, có {len(d)}"
    assert d.loc[d.run_id == "m2_a", "test_macro_f1"].iloc[0] == pytest.approx(0.9999)


def test_rebuild_thi_dung_la_GHI_DE(bang_cu):
    """--rebuild phải dựng lại từ đầu, để bỏ được run đã xoá khỏi đĩa."""
    build_main_table([_hang("m1_a", "m1_lenet", 0.9778)], bang_cu, rebuild=True)
    d = pd.read_csv(bang_cu)
    assert len(d) == 1, f"rebuild phải cho đúng 1 dòng, có {len(d)}"


def test_sap_theo_macro_f1_giam_dan(bang_cu):
    d = pd.read_csv(bang_cu)
    assert list(d.test_macro_f1) == sorted(d.test_macro_f1, reverse=True)


def test_bang_cu_hong_thi_van_ghi_duoc(tmp_path):
    """File cũ không đọc được thì phải dựng mới, không được crash."""
    out = tmp_path / "main_comparison.csv"
    out.write_text("đây không phải csv hợp lệ\x00\x01", encoding="utf-8", errors="ignore")
    build_main_table([_hang("m1_a", "m1_lenet", 0.9778)], out)
    assert len(pd.read_csv(out)) == 1
