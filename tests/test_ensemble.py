"""
test_ensemble.py — khoá tính đúng của TTA và ensemble.

CHỦ: Phong Nguyễn
"""
import numpy as np
import pytest
import torch

from gtsrb import NUM_CLASSES
from gtsrb.eval.ensemble import (disagreement_rate, ensemble_probs,
                                 ensemble_report, tta_variants)


def test_tta_sinh_dung_so_bien_the():
    x = torch.randn(4, 3, 48, 48)
    variants = tta_variants(x, angles=(-7.5, 0.0, 7.5))
    assert len(variants) == 3
    assert all(v.shape == x.shape for v in variants), "TTA không được đổi shape"


def test_tta_goc_0_tra_ve_anh_y_nguyen():
    """Biến thể góc 0 phải là CHÍNH ảnh gốc, không qua nội suy.

    Nếu nội suy cả ở góc 0 thì ảnh bị mờ đi một chút và baseline của TTA
    không còn bằng dự đoán thường — làm sai phép so sánh.
    """
    x = torch.randn(2, 3, 48, 48)
    variants = tta_variants(x, angles=(0.0,))
    assert torch.equal(variants[0], x)


def test_ensemble_hai_model_giong_het_thi_khong_doi():
    """Gộp hai bản sao của cùng một dự đoán phải cho đúng dự đoán đó."""
    probs = np.random.dirichlet(np.ones(NUM_CLASSES), size=50)
    combined = ensemble_probs([probs, probs])
    assert np.allclose(combined, probs)


def test_ensemble_van_la_phan_phoi_hop_le():
    """Tổng xác suất mỗi dòng phải bằng 1 sau khi gộp."""
    a = np.random.dirichlet(np.ones(NUM_CLASSES), size=30)
    b = np.random.dirichlet(np.ones(NUM_CLASSES), size=30)
    assert np.allclose(ensemble_probs([a, b]).sum(axis=1), 1.0)
    assert np.allclose(ensemble_probs([a, b], weights=[0.7, 0.3]).sum(axis=1), 1.0)


def test_ensemble_bao_loi_khi_shape_khac_nhau():
    """Hai model chạy trên tập khác nhau là lỗi lập trình, phải vỡ NGAY."""
    with pytest.raises(ValueError, match="shape khác nhau"):
        ensemble_probs([np.zeros((10, 43)), np.zeros((20, 43))])


def test_disagreement_bang_0_khi_hai_model_giong_het():
    probs = np.random.dirichlet(np.ones(NUM_CLASSES), size=40)
    assert disagreement_rate([probs, probs]) == 0.0


def test_disagreement_bang_1_khi_hoan_toan_khac():
    """Hai model đoán khác nhau ở MỌI ảnh -> tỉ lệ bất đồng = 1."""
    n = 20
    a = np.zeros((n, NUM_CLASSES)); a[:, 0] = 1.0
    b = np.zeros((n, NUM_CLASSES)); b[:, 1] = 1.0
    assert disagreement_rate([a, b]) == 1.0


def test_ensemble_report_co_dong_tong_hop():
    y_true = np.random.randint(0, NUM_CLASSES, 60)
    members = {f"m{i}": np.random.dirichlet(np.ones(NUM_CLASSES), size=60)
               for i in range(2)}
    frame = ensemble_report(y_true, members)
    assert len(frame) == 3, "phải có 2 model + 1 dòng ENSEMBLE"
    assert "ENSEMBLE" in frame.iloc[-1]["thành phần"]
    assert "so với model đơn tốt nhất" in frame.columns
