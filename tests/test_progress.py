"""
test_progress.py — khoá cách đọc log mẻ của scripts/progress.py.

CHỦ: Phong Nguyễn

Thanh tiến độ là thứ cả nhóm nhìn để biết "còn bao lâu". Nó đọc logs/_all.log
bằng regex, nên MỖI LẦN đổi câu chữ trong run_everything.sh là một lần nó có
thể đọc sai mà KHÔNG báo lỗi gì — chỉ hiện một con số vô lý.

LỖI ĐÃ GẶP (đúng bộ log dưới đây): hai bước đẩy lên Colab và bước tổng hợp
cuối đều bị tính là "đang chạy", nên ETA báo còn 6h42m trong khi mẻ đã xong
từ 15 phút trước và dòng "★ HOÀN TẤT" hiện ngay bên dưới.
"""
import pytest
from progress import STEPS, parse_steps

# Log THẬT của mẻ chạy đêm 03→04/10/2026, giữ nguyên từng dấu.
LOG_THAT = """\
[22:09] ================ BẮT ĐẦU ================
[22:09] Máy phải CẮM SẠC và MỞ NẮP. caffeinate chỉ chặn ngủ khi có sạc.
[22:09] 1/5 NÉN int8 + XUẤT TorchScript/ONNX cho 5 model  (~15 phút)
[22:30]     -> mã thoát 0
[22:30] 2/5 THÍ NGHIỆM RÒ RỈ: train M1 hai lần, split sai vs split đúng  (~25 phút)
[22:54]     -> mã thoát 0
[22:54] 3/5 ABLATION — BỎ QUA (SKIP_HEAVY=1, chạy trên Colab)
[22:54] 4/5 SEED — BỎ QUA (SKIP_HEAVY=1, chạy trên Colab)
[22:54] 5/5 ĐÁNH GIÁ TẤT CẢ + TỔNG HỢP  (~30 phút)
[22:57]     evaluate OK
[22:57]     BAO_CAO.md OK
[22:57] ================ HOÀN TẤT ================
""".splitlines()


@pytest.fixture
def state():
    return parse_steps(LOG_THAT)


def test_khong_con_buoc_nao_dang_chay_khi_log_da_hoan_tat(state):
    """Sau dòng HOÀN TẤT thì KHÔNG bước nào được coi là đang chạy."""
    treo = [k for k, v in state.items() if v.get("running")]
    assert treo == [], f"các bước bị treo ở 'đang chạy': {treo}"


def test_doc_duoc_het_nam_buoc(state):
    assert sorted(state) == [key for key, _, _ in STEPS]


def test_buoc_bi_SKIP_HEAVY_duoc_danh_dau_bo_qua(state):
    """'BỎ QUA' viết HOA — so chuỗi chữ thường là lỗi đã từng xảy ra."""
    for key in ("3/5", "4/5"):
        assert state[key].get("skipped") is True, f"{key} phải là bỏ qua"


def test_buoc_cuoi_ket_thuc_bang_dong_HOAN_TAT(state):
    """Bước 5/5 KHÔNG có dòng '-> mã thoát'; chỉ 'HOÀN TẤT' đóng nó."""
    assert state["5/5"]["running"] is False
    assert state["5/5"]["end"] == "22:57"


def test_eta_bang_khong_khi_moi_buoc_da_xong_hoac_bo_qua(state):
    """Đúng chỗ đã sinh ra con số sai 6h42m (= 300 + 72 + 30 phút)."""
    con_lai = sum(est for key, _, est in STEPS
                  if not state[key].get("skipped")
                  and "start" not in state[key])
    con_lai += sum(est for key, _, est in STEPS if state[key].get("running"))
    assert con_lai == 0, f"ETA phải là 0 nhưng tính ra {con_lai} phút"


def test_moc_thoi_gian_bat_dau_doc_dung(state):
    assert state["1/5"]["start"] == "22:09"
    assert state["2/5"]["start"] == "22:30"
    assert state["5/5"]["start"] == "22:54"
