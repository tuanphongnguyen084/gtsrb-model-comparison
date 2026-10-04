"""
test_transforms.py — khoá hai điều BỊ CẤM trong augmentation.

CHỦ: Huy

Đây là test bảo vệ tính đúng của NHÃN, không phải tính đúng của code.
"""
from pathlib import Path

import pytest
import torch

from gtsrb import MIRROR_PAIRS
from gtsrb.data.transforms import (AUG_POLICIES, FORBIDDEN_TRANSFORMS,
                                   build_transform)


def test_khong_co_horizontal_flip_trong_bat_ky_policy_nao():
    """★ Lật ngang lớp 33 'rẽ phải' ra ĐÚNG HÌNH lớp 34 'rẽ trái'.

    Các cặp bị ảnh hưởng: 33/34, 19/20, 36/37, 38/39.
    Flip = tự tạo dữ liệu SAI NHÃN. Model sẽ học rằng rẽ phải và rẽ trái là một.
    """
    for policy in AUG_POLICIES:
        transform = build_transform(policy, img_size=48)
        text = repr(transform)
        for forbidden in FORBIDDEN_TRANSFORMS:
            assert forbidden not in text, (
                f"policy {policy!r} có {forbidden} — bị CẤM. "
                f"Các cặp lớp gương sẽ bị dán sai nhãn: {MIRROR_PAIRS}"
            )


def test_khong_co_hue_hay_saturation_jitter():
    """Màu MANG NGHĨA: đỏ = cấm, xanh = bắt buộc. Đổi hue là đổi nghĩa biển báo.

    ColorJitter chỉ được dùng brightness và contrast.
    """
    for policy in AUG_POLICIES:
        transform = build_transform(policy, img_size=48)
        if transform is None:
            continue
        for step in getattr(transform, "transforms", []):
            if type(step).__name__ == "ColorJitter":
                assert step.hue is None, f"{policy}: ColorJitter có hue — bị cấm"
                assert step.saturation is None, \
                    f"{policy}: ColorJitter có saturation — bị cấm"


def test_source_code_khong_chua_flip():
    """Chặn ở mức mã nguồn luôn, phòng người sau thêm vào mà quên đọc ghi chú."""
    source = Path("src/gtsrb/data/transforms.py").read_text(encoding="utf-8")
    # Bỏ các dòng chú thích/cấm (chúng có nhắc tên để giải thích)
    code_lines = [
        line for line in source.splitlines()
        if not line.strip().startswith("#")
        and "FORBIDDEN" not in line
        and "CẤM" not in line
    ]
    code = "\n".join(code_lines)
    assert "RandomHorizontalFlip(" not in code
    assert "RandomVerticalFlip(" not in code


@pytest.mark.parametrize("policy", AUG_POLICIES)
def test_giu_nguyen_shape_va_dtype(policy):
    """Augmentation KHÔNG được đổi shape hay dtype — dataset resize sau đó."""
    transform = build_transform(policy, img_size=48)
    x = torch.randint(0, 256, (3, 48, 48), dtype=torch.uint8)
    if transform is None:
        assert policy == "none"
        return
    out = transform(x)
    assert out.shape == x.shape, f"{policy} đổi shape thành {tuple(out.shape)}"
    assert out.dtype == torch.uint8, f"{policy} đổi dtype thành {out.dtype}"


def test_policy_none_tra_ve_None():
    """policy='none' trả None để dataset bỏ qua hẳn, không gọi transform vô ích."""
    assert build_transform("none", 48) is None


def test_policy_sai_thi_bao_loi():
    with pytest.raises(ValueError, match="policy"):
        build_transform("khong_ton_tai", 48)


def test_cac_cap_lop_guong_duoc_khai_bao_dung():
    """MIRROR_PAIRS phải khớp với thực tế GTSRB — đây là cơ sở của lệnh cấm flip."""
    from gtsrb import CLASS_NAMES
    expected = {
        (19, 20): ("trái", "phải"),
        (33, 34): ("phải", "trái"),
        (36, 37): ("phải", "trái"),
        (38, 39): ("phải", "trái"),
    }
    assert set(MIRROR_PAIRS) == set(expected)
    for (a, b), (word_a, word_b) in expected.items():
        assert word_a in CLASS_NAMES[a], f"lớp {a} ({CLASS_NAMES[a]}) phải chứa {word_a!r}"
        assert word_b in CLASS_NAMES[b], f"lớp {b} ({CLASS_NAMES[b]}) phải chứa {word_b!r}"
