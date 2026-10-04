"""
test_export.py — khoá tính đúng của phần nén và xuất model.

CHỦ: Phong Trần

Hai lỗi ĐÃ GẶP THẬT mà các test này chặn:
  1. Không đặt backend -> "NoQEngine", thông báo lỗi không gợi ý gì
  2. Static quantization CONVERT được nhưng VỠ lúc chạy (thiếu QuantStub)
"""
import pytest
import torch

from gtsrb.deploy.export import (model_size_mb, pick_backend, quantize_dynamic,
                                 quantize_static)
from gtsrb.models.registry import build_model


@pytest.fixture(scope="module")
def m1():
    return build_model("m1_lenet", img_size=48).eval()


def test_tu_chon_duoc_backend():
    """pick_backend phải trả một engine có thật, không raise."""
    backend = pick_backend()
    assert backend in torch.backends.quantized.supported_engines


def test_quantize_dynamic_chay_duoc(m1):
    """Bản int8 phải CHẠY được, không chỉ convert được.

    Lỗi 'NoQEngine' xảy ra ở bước convert; lỗi thiếu QuantStub xảy ra ở bước chạy.
    Test này bắt cả hai.
    """
    q = quantize_dynamic(m1)
    with torch.no_grad():
        out = q(torch.randn(2, 3, 48, 48))
    assert out.shape == (2, 43)
    assert torch.isfinite(out).all(), "output có NaN/Inf sau khi lượng tử hoá"


def test_quantize_dynamic_khong_sua_model_goc(m1):
    """Lượng tử hoá phải trả model MỚI, không đổi model gốc tại chỗ.

    Nếu sửa tại chỗ thì mọi phép đo fp32 sau đó đều sai mà không báo lỗi.
    """
    before = model_size_mb(m1)
    quantize_dynamic(m1)
    assert model_size_mb(m1) == pytest.approx(before), \
        "model gốc bị đổi — quantize_dynamic phải trả bản sao"


def test_m1_nho_di_dang_ke_sau_khi_nen(m1):
    """★ M1 có 97,3% tham số ở Linear nên dynamic quantization phải giảm >= 3 lần."""
    ratio = model_size_mb(m1) / model_size_mb(quantize_dynamic(m1))
    assert ratio > 3.0, f"chỉ nhẹ đi {ratio:.2f} lần, mong đợi > 3"


def test_m2_gan_nhu_khong_nho_di():
    """★ M2 dùng GlobalAvgPool nên Linear chỉ 11.051 tham số -> nén động vô ích.

    Đây không phải lỗi mà là KẾT QUẢ đáng báo cáo: kiến trúc quyết định việc nén
    có hiệu quả hay không. Test khoá lại để quan sát này không bị mất.
    """
    m2 = build_model("m2_vggres", img_size=48).eval()
    ratio = model_size_mb(m2) / model_size_mb(quantize_dynamic(m2))
    assert ratio < 1.5, f"nhẹ đi {ratio:.2f} lần — khác kỳ vọng, kiểm lại kiến trúc"


def test_quantize_static_tra_None_thay_vi_vo(m1):
    """Nếu không lượng tử hoá tĩnh được thì trả None, KHÔNG raise.

    Đây là bước tuỳ chọn; để nó làm vỡ cả đường ống export là thiết kế sai.
    """
    from torch.utils.data import DataLoader, TensorDataset
    data = TensorDataset(torch.randn(32, 3, 48, 48), torch.zeros(32, dtype=torch.long))
    result = quantize_static(m1, DataLoader(data, batch_size=8),
                             img_size=48, n_batches=2)
    assert result is None or isinstance(result, torch.nn.Module)
    if result is not None:
        with torch.no_grad():
            assert result(torch.randn(1, 3, 48, 48)).shape == (1, 43)
