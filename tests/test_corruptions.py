"""
test_corruptions.py — khoá các bất biến của tầng robustness.

CHỦ: Huy
"""
import numpy as np
import pytest

from gtsrb.robustness.corruptions import (CORRUPTIONS, SEVERITIES,
                                          apply_corruption)


@pytest.fixture
def sample_image():
    """Ảnh 48x48x3 uint8 có gradient — không dùng ảnh toàn 0 vì nhiều lỗi
    (ví dụ chia cho 0, hoặc quên nhân) sẽ không bộc lộ trên ảnh đen."""
    rng = np.random.default_rng(0)
    base = np.linspace(0, 255, 48 * 48 * 3).reshape(48, 48, 3)
    noise = rng.integers(0, 30, size=(48, 48, 3))
    return np.clip(base + noise, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("kind", CORRUPTIONS)
def test_severity_0_tra_ve_anh_y_nguyen(sample_image, kind):
    """severity=0 nghĩa là KHÔNG làm gì — bất biến cần cho đường cơ sở 'clean'."""
    out = apply_corruption(sample_image, kind, 0, np.random.default_rng(0))
    assert np.array_equal(out, sample_image)


@pytest.mark.parametrize("kind", CORRUPTIONS)
@pytest.mark.parametrize("severity", SEVERITIES)
def test_giu_nguyen_shape_va_dtype(sample_image, kind, severity):
    """Nhiễu KHÔNG được đổi shape hay dtype — nếu đổi thì vỡ ở chỗ khác rất xa."""
    out = apply_corruption(sample_image, kind, severity, np.random.default_rng(0))
    assert out.shape == sample_image.shape, f"{kind} đổi shape"
    assert out.dtype == np.uint8, f"{kind} đổi dtype thành {out.dtype}"
    assert out.min() >= 0 and out.max() <= 255, (
        f"{kind} cho giá trị ngoài [0,255] — thiếu bước clip. "
        f"Bỏ clip là lỗi im lặng: giá trị 300 wrap thành 44 và biến vùng sáng "
        f"thành vùng tối."
    )


@pytest.mark.parametrize("kind", CORRUPTIONS)
def test_cung_seed_cho_cung_ket_qua(sample_image, kind):
    """Tái lập được: cùng rng -> cùng ảnh. Thiếu tính này thì bảng robustness
    không lặp lại được giữa các lần chạy."""
    a = apply_corruption(sample_image, kind, 3, np.random.default_rng(123))
    b = apply_corruption(sample_image, kind, 3, np.random.default_rng(123))
    assert np.array_equal(a, b), f"{kind} không tiền định với cùng seed"


@pytest.mark.parametrize("kind", CORRUPTIONS)
def test_nhieu_cang_manh_thi_anh_cang_khac(sample_image, kind):
    """Mức độ phải thực sự tăng dần: severity 5 lệch nhiều hơn severity 1."""
    def deviation(severity: int) -> float:
        out = apply_corruption(sample_image, kind, severity,
                               np.random.default_rng(7))
        return float(np.abs(out.astype(float) - sample_image.astype(float)).mean())

    assert deviation(5) > deviation(1), (
        f"{kind}: severity 5 không lệch nhiều hơn severity 1 — "
        f"bảng mức độ có thể bị đặt sai."
    )


def test_kind_sai_thi_bao_loi_ro_rang(sample_image):
    with pytest.raises(ValueError, match="kind"):
        apply_corruption(sample_image, "khong_ton_tai", 1)


def test_severity_sai_thi_bao_loi(sample_image):
    with pytest.raises(ValueError, match="severity"):
        apply_corruption(sample_image, "fog", 99)


def test_dtype_sai_thi_bao_loi():
    """Truyền float vào là lỗi lập trình, phải vỡ NGAY với thông điệp rõ,
    không được âm thầm cho kết quả vô nghĩa."""
    with pytest.raises(TypeError, match="uint8"):
        apply_corruption(np.zeros((8, 8, 3), dtype=np.float32), "fog", 1)


def test_occlusion_che_dung_ti_le_dien_tich(sample_image):
    """Che 50% thì số pixel bị đổi phải xấp xỉ 50% (sai số do làm tròn kích thước)."""
    out = apply_corruption(sample_image, "occlusion", 5, np.random.default_rng(0))
    changed = (out != sample_image).any(axis=2).mean()
    assert 0.35 < changed < 0.65, f"Che 50% mà đổi {changed:.1%} pixel"


def test_transforms_khong_import_corruptions():
    """★ LUẬT SẮT: nhiễu CHỈ dùng ở test, KHÔNG BAO GIỜ ở train.

    Nếu data/transforms.py (augmentation lúc train) import corruptions thì phép
    đo robustness mất ý nghĩa: train trên nhiễu rồi test trên nhiễu là test
    in-distribution, chỉ chứng minh 'model học được cái nó đã thấy'.
    """
    from pathlib import Path
    source = Path("src/gtsrb/data/transforms.py").read_text(encoding="utf-8")
    assert "corruptions" not in source, (
        "data/transforms.py KHÔNG được import robustness.corruptions. "
        "Xem ghi chú ở đầu src/gtsrb/robustness/corruptions.py."
    )
