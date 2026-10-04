"""
test_canh_bao_tai.py — khoá cảnh báo "máy đang bận" của benchmark().

CHỦ: Phong Trần

Vì sao cần: số latency đo trên máy đang bận chậm 1,3-2,3 lần so với thực tế, và
KHÔNG có gì báo động. Lỗi này đã xảy ra thật — chạy edge_export song song với
run_robustness làm cả 5 model bị méo, và bảng edge_export.csv trông hoàn toàn
bình thường. Xem quy tắc 7 trong docstring của src/gtsrb/deploy/speed.py.

Điểm then chốt mà các test dưới khoá lại: PHẢI có hai dấu hiệu, vì mỗi dấu hiệu
một mình đều bỏ sót. Đo thật trên ResNet18 lúc máy bận: chậm 2,55 lần nhưng
p95/p50 chỉ 1,26 — dưới ngưỡng jitter. Chỉ tín hiệu TẢI bắt được.
"""
import pytest
import torch

from gtsrb.deploy import speed
from gtsrb.deploy.speed import NGUONG_JITTER, NGUONG_TAI_MOI_LOI, benchmark
from gtsrb.models.registry import build_model


@pytest.fixture
def m1():
    return build_model("m1_lenet", img_size=48).eval()


def _do(m1, monkeypatch, tai):
    """Đo nhanh với tải hệ thống được GIẢ LẬP, để test không phụ thuộc máy."""
    monkeypatch.setattr(speed, "_tai_moi_loi", lambda: tai)
    return benchmark(m1, 48, torch.device("cpu"), batch_sizes=(1,),
                     warmup=3, iters=15)


def test_may_ranh_thi_KHONG_gan_co_nhieu(m1, monkeypatch):
    r = _do(m1, monkeypatch, tai=0.0)
    assert r.get("cpu_bs1_nhieu_he_thong", False) is False


def test_tai_cao_thi_gan_co_du_jitter_binh_thuong(m1, monkeypatch):
    """Đây là trường hợp mà jitter MỘT MÌNH bỏ sót — model chậm, máy bận."""
    r = _do(m1, monkeypatch, tai=NGUONG_TAI_MOI_LOI + 0.5)
    assert r["cpu_bs1_nhieu_he_thong"] is True


def test_khong_doc_duoc_tai_thi_van_chay(m1, monkeypatch):
    """os.getloadavg() không có trên mọi hệ — thiếu nó không được làm vỡ phép đo."""
    r = _do(m1, monkeypatch, tai=None)
    assert "cpu_bs1_p50" in r and r["cpu_bs1_p50"] > 0


def test_luon_bao_ty_le_jitter(m1, monkeypatch):
    """jitter phải LUÔN có trong kết quả để ghi được vào bảng, không chỉ khi báo động."""
    r = _do(m1, monkeypatch, tai=0.0)
    assert r["cpu_bs1_jitter"] == pytest.approx(
        r["cpu_bs1_p95"] / r["cpu_bs1_p50"], rel=0.02)


def test_nguong_khop_voi_so_do_that():
    """Ngưỡng phải nằm ĐÚNG giữa vùng máy rảnh và vùng máy bận đã đo.

    Số đo thật trên 5 model: máy rảnh p95/p50 = 1,04-1,34; máy bận lên tới 3,71.
    Tải: lúc chạy song song đo được ~0,35-0,93 trên mỗi lõi.
    """
    assert 1.34 < NGUONG_JITTER < 3.71, "ngưỡng jitter nằm ngoài vùng đã đo"
    assert 0.0 < NGUONG_TAI_MOI_LOI < 0.35, "ngưỡng tải phải thấp hơn 0,35 đã đo"
