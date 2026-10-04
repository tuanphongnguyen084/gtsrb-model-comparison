"""
test_metrics.py — khoá tính đúng của tầng đo lường.

CHỦ: Phong Nguyễn

Chỉ số sai thì mọi kết luận sai, và sai chỉ số là lỗi IM LẶNG — không ai thấy
cho tới khi giảng viên hỏi.
"""
import numpy as np
import pytest

from gtsrb import NUM_CLASSES
from gtsrb.eval.calibration import expected_calibration_error
from gtsrb.eval.confusion import confusion, top_confusions
from gtsrb.eval.metrics import metrics_from_probs
from gtsrb.eval.stats_tests import mcnemar


def make_probs(y_true: np.ndarray, accuracy: float, seed: int = 0) -> np.ndarray:
    """Tạo y_prob giả sao cho top-1 accuracy xấp xỉ `accuracy`."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    probs = rng.uniform(0, 0.01, size=(n, NUM_CLASSES))
    correct_mask = rng.random(n) < accuracy
    for i in range(n):
        target = y_true[i] if correct_mask[i] else (y_true[i] + 1) % NUM_CLASSES
        probs[i, target] = 0.9
    return probs / probs.sum(axis=1, keepdims=True)


def test_du_doan_hoan_hao_cho_moi_chi_so_bang_1():
    y_true = np.arange(NUM_CLASSES)
    probs = np.eye(NUM_CLASSES)
    stats = metrics_from_probs(y_true, probs)
    assert stats["top1"] == pytest.approx(1.0)
    assert stats["top5"] == pytest.approx(1.0)
    assert stats["macro_f1"] == pytest.approx(1.0)


def test_top5_luon_lon_hon_hoac_bang_top1():
    """Bất biến toán học. Nếu vi phạm thì công thức top-5 sai."""
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, NUM_CLASSES, 500)
    probs = make_probs(y_true, 0.7)
    stats = metrics_from_probs(y_true, probs)
    assert stats["top5"] >= stats["top1"]


def test_macro_f1_bi_anh_huong_manh_hon_accuracy_khi_bo_lop_thua():
    """★ Test chứng minh LÝ DO chọn macro-F1 làm chỉ số chính.

    Dựng tình huống mất cân bằng rồi cho model bỏ hẳn lớp thưa:
    accuracy giảm rất ít, macro-F1 giảm nhiều. Đây là bằng chứng cho báo cáo.
    """
    # 990 ảnh lớp 0, 10 ảnh lớp 1
    y_true = np.array([0] * 990 + [1] * 10)
    # Model đoán TẤT CẢ là lớp 0 -> bỏ hẳn lớp 1
    probs = np.zeros((1000, NUM_CLASSES))
    probs[:, 0] = 1.0

    stats = metrics_from_probs(y_true, probs)
    assert stats["top1"] == pytest.approx(0.99), "accuracy vẫn 99% dù bỏ hẳn một lớp"
    # macro-F1 tính trên 43 lớp: chỉ lớp 0 có F1 > 0
    assert stats["macro_f1"] < 0.03, (
        f"macro-F1 = {stats['macro_f1']:.4f}, phải RẤT thấp. "
        f"Đây là điểm khác biệt cốt lõi so với accuracy."
    )


def test_per_class_co_du_43_dong():
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, NUM_CLASSES, 300)
    stats = metrics_from_probs(y_true, make_probs(y_true, 0.8))
    assert len(stats["per_class"]) == NUM_CLASSES
    assert list(stats["per_class"].columns) == \
        ["class_id", "name", "precision", "recall", "f1", "support"]
    assert stats["per_class"]["support"].sum() == 300


# ---------------- McNemar ----------------

def test_mcnemar_hai_model_giong_het_thi_p_bang_1():
    y_true = np.array([0, 1, 2, 3, 4] * 20)
    pred = y_true.copy()
    out = mcnemar(y_true, pred, pred)
    assert out["n01"] == 0 and out["n10"] == 0
    assert out["p_value"] == pytest.approx(1.0)
    assert out["better"] == "tie"


def test_mcnemar_chi_dung_o_bat_dong():
    """n00 và n11 (hai model đồng ý) KHÔNG được ảnh hưởng tới p-value.

    Nếu thêm 10.000 ảnh mà cả hai model đều đúng, kết luận phải KHÔNG đổi.
    """
    y_true = np.array([0] * 100)
    pred_a = y_true.copy(); pred_a[:30] = 1      # A sai 30
    pred_b = y_true.copy(); pred_b[70:] = 1      # B sai 30, ở chỗ khác
    out_small = mcnemar(y_true, pred_a, pred_b)

    # Thêm 5.000 ảnh mà CẢ HAI đều đúng
    y_true_big = np.concatenate([y_true, np.zeros(5000, dtype=int)])
    pred_a_big = np.concatenate([pred_a, np.zeros(5000, dtype=int)])
    pred_b_big = np.concatenate([pred_b, np.zeros(5000, dtype=int)])
    out_big = mcnemar(y_true_big, pred_a_big, pred_b_big)

    assert out_small["p_value"] == pytest.approx(out_big["p_value"]), (
        "p-value đổi khi thêm ảnh mà cả hai model đều đúng — "
        "công thức đang dùng sai ô của bảng 2x2."
    )
    assert out_big["n00"] > out_small["n00"]


def test_mcnemar_chon_binomial_khi_it_bat_dong():
    """n01+n10 < 25 -> xấp xỉ chi-square không đáng tin -> phải dùng binomial."""
    y_true = np.zeros(1000, dtype=int)
    pred_a = y_true.copy(); pred_a[:3] = 1       # A sai 3
    pred_b = y_true.copy(); pred_b[500:502] = 1  # B sai 2
    out = mcnemar(y_true, pred_a, pred_b)
    assert out["n01"] + out["n10"] < 25
    assert out["method"] == "exact_binomial"


def test_mcnemar_phat_hien_duoc_khac_biet_that():
    """Khác biệt LỚN phải cho p < 0,05 và chỉ đúng model tốt hơn."""
    y_true = np.zeros(1000, dtype=int)
    pred_a = y_true.copy()                       # A đúng hết
    pred_b = y_true.copy(); pred_b[:100] = 1     # B sai 100
    out = mcnemar(y_true, pred_a, pred_b)
    assert out["p_value"] < 0.05
    assert out["better"] == "A"


# ---------------- Calibration ----------------

def test_ece_bang_0_khi_hieu_chinh_hoan_hao():
    """Model nói 100% chắc và đúng 100% -> ECE = 0."""
    y_true = np.arange(NUM_CLASSES)
    probs = np.eye(NUM_CLASSES)
    out = expected_calibration_error(y_true, probs)
    assert out["ece"] == pytest.approx(0.0, abs=1e-6)


def test_ece_cao_khi_tu_tin_thai_qua():
    """Model nói 99% chắc nhưng chỉ đúng 50% -> ECE phải gần 0,49."""
    n = 1000
    y_true = np.zeros(n, dtype=int)
    probs = np.full((n, NUM_CLASSES), 0.01 / (NUM_CLASSES - 1))
    probs[:, 0] = 0.99
    # Cho một nửa bị sai bằng cách đổi nhãn thật
    y_true[: n // 2] = 1
    out = expected_calibration_error(y_true, probs)
    assert 0.4 < out["ece"] < 0.55, f"ECE = {out['ece']:.4f}, mong đợi ~0,49"


def test_ece_bins_cong_lai_bang_tong_so_mau():
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, NUM_CLASSES, 500)
    out = expected_calibration_error(y_true, make_probs(y_true, 0.8), n_bins=15)
    assert out["bins"]["count"].sum() == 500, "có mẫu bị rơi ra ngoài mọi bin"


# ---------------- Confusion ----------------

def test_confusion_dung_kich_thuoc_va_tong():
    rng = np.random.default_rng(4)
    y_true = rng.integers(0, NUM_CLASSES, 400)
    y_pred = rng.integers(0, NUM_CLASSES, 400)
    matrix = confusion(y_true, y_pred)
    assert matrix.shape == (NUM_CLASSES, NUM_CLASSES)
    assert matrix.sum() == 400


def test_confusion_normalize_moi_hang_bang_1():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    matrix = confusion(y_true, y_pred, normalize=True)
    assert matrix[0].sum() == pytest.approx(1.0)
    assert matrix[1].sum() == pytest.approx(1.0)
    # Lớp không có mẫu -> hàng toàn 0, KHÔNG được là NaN
    assert matrix[5].sum() == pytest.approx(0.0)
    assert not np.isnan(matrix).any()


def test_top_confusions_bo_duong_cheo():
    """Chỉ quan tâm LỖI, không quan tâm dự đoán đúng."""
    y_true = np.array([0] * 100 + [1] * 10)
    y_pred = np.array([0] * 100 + [2] * 10)      # lớp 1 bị đoán thành lớp 2
    pairs = top_confusions(y_true, y_pred, n=5)
    assert len(pairs) >= 1
    assert pairs.iloc[0]["true_id"] == 1
    assert pairs.iloc[0]["pred_id"] == 2
    assert pairs.iloc[0]["count"] == 10
    # Không có dòng nào true_id == pred_id
    assert not (pairs["true_id"] == pairs["pred_id"]).any()
