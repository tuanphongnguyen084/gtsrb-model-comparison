"""
test_loc_run.py — khoá phép lọc "run chính" vs "run phụ" trong các bảng CSV.

CHỦ: Hoàng (bảo trì scripts/_common.py)

Vì sao cần test: artifacts/runs/ chứa lẫn run CHÍNH (5 model của báo cáo) với run
PHỤ (30 biến thể ablation, các seed khác, 2 run của thí nghiệm rò rỉ). Ba script
evaluate / make_report / make_baocao đều phải lọc GIỐNG NHAU. Trước đây mỗi script
tự viết lại biểu thức lọc, nên chúng phải trùng nhau mà không có gì bắt buộc.

Hậu quả nếu lọc sai: một run ablation, hoặc tệ hơn là một run TRAIN VỚI SPLIT RÒ RỈ,
lọt vào bảng so sánh chính của báo cáo nộp thầy. Dòng leak-A_random có macro-F1
0,98209 — xếp TRÊN MobileNetV2 — nên nó sẽ trông hoàn toàn hợp lý.
"""
import pandas as pd
import pytest
from _common import TAG_CHINH, chi_run_chinh, chi_run_phu

# Giống đúng cấu trúc main_comparison.csv thật, gồm cả run rò rỉ.
BANG = pd.DataFrame({
    "model": ["m3_resnet18", "m2_vggres", "m1_lenet", "m1_lenet",
              "m2_vggres", "m2_vggres", "m1_lenet"],
    "tag": [TAG_CHINH, TAG_CHINH, "leak-A_random", "leak-B_track",
            "resolution-64", "", None],
    "test_macro_f1": [0.99030, 0.98798, 0.98209, 0.97779,
                      0.98500, 0.98798, 0.97779],
})


def test_run_chinh_khong_lan_run_phu():
    chinh = chi_run_chinh(BANG)
    assert set(chinh["tag"].fillna("")) <= {"", TAG_CHINH}
    assert "leak-A_random" not in set(chinh["tag"].fillna(""))


def test_run_ro_ri_KHONG_duoc_vao_bang_chinh():
    """Dòng quan trọng nhất: run rò rỉ xếp hạng 3 nếu không bị lọc."""
    chinh = chi_run_chinh(BANG)
    assert not (chinh["tag"].astype(str).str.startswith("leak")).any()


def test_hai_ham_PHAN_HOACH_dung_bang():
    """Không dòng nào thuộc cả hai, không dòng nào rơi ngoài cả hai.

    Đây là tính chất ngăn hai hàm trôi khỏi nhau: thêm một giá trị tag mới mà
    chỉ sửa một hàm thì test này đỏ ngay.
    """
    chinh, phu = chi_run_chinh(BANG), chi_run_phu(BANG)
    assert len(chinh) + len(phu) == len(BANG), "có dòng bị mất hoặc bị đếm hai lần"
    assert set(chinh.index).isdisjoint(phu.index), "có dòng thuộc cả hai nhóm"


def test_bang_cu_khong_co_TAG_CHINH_van_doc_duoc():
    """Bảng sinh TRƯỚC khi có TAG_CHINH dùng ô TRỐNG — phải vẫn lọc đúng."""
    cu = pd.DataFrame({"model": ["m1", "m2"], "tag": ["", "resolution-64"]})
    assert list(chi_run_chinh(cu)["model"]) == ["m1"]
    assert list(chi_run_phu(cu)["model"]) == ["m2"]


def test_khong_co_cot_tag_thi_coi_tat_ca_la_chinh():
    khong_tag = pd.DataFrame({"model": ["m1", "m2"]})
    assert len(chi_run_chinh(khong_tag)) == 2
    assert len(chi_run_phu(khong_tag)) == 0


@pytest.mark.parametrize("gia_tri", ["", TAG_CHINH, None, float("nan")])
def test_moi_cach_viet_cua_run_chinh(gia_tri):
    """NaN, None, ô trống và '(chính)' đều phải được coi là run chính."""
    f = pd.DataFrame({"model": ["m"], "tag": [gia_tri]})
    assert len(chi_run_chinh(f)) == 1, f"{gia_tri!r} bị loại khỏi bảng chính"
    assert len(chi_run_phu(f)) == 0
