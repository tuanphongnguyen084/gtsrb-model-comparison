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


# =====================================================================
# verify_onnx — xem sự cố 9 trong docs/SU_CO.md
# =====================================================================

onnxruntime = pytest.importorskip("onnxruntime",
                                  reason="cần onnxruntime để kiểm ONNX")


@pytest.fixture(scope="module")
def m1_onnx(m1, tmp_path_factory):
    """Xuất M1 ra ONNX một lần, dùng lại cho mọi test dưới."""
    from gtsrb.deploy.export import export_onnx
    path = export_onnx(m1, 48, tmp_path_factory.mktemp("onnx") / "m1.onnx")
    if path is None:
        pytest.skip("không xuất được ONNX trong môi trường này")
    return path


def test_onnx_khop_khi_kiem_bang_anh_that(m1, m1_onnx):
    """Truyền sample thì phải KHỚP — đây là đường dùng thật của export_edge."""
    from gtsrb.deploy.export import verify_onnx
    torch.manual_seed(0)
    anh = torch.rand(8, 3, 48, 48)      # rand [0,1], giống thang ảnh đã chuẩn hoá
    assert verify_onnx(m1_onnx, m1, 48, sample=anh) is True


def test_verify_onnx_bat_duoc_model_SAI(m1, m1_onnx):
    """Phép kiểm phải thật sự phát hiện lệch, không phải luôn trả True.

    So file ONNX của M1 với một model M1 KHÁC (khởi tạo ngẫu nhiên, trọng số
    khác hẳn). Nếu hàm vẫn báo KHỚP thì nó chẳng kiểm gì cả.
    """
    from gtsrb.deploy.export import verify_onnx
    torch.manual_seed(999)
    model_khac = build_model("m1_lenet", img_size=48).eval()
    anh = torch.rand(8, 3, 48, 48)
    assert verify_onnx(m1_onnx, model_khac, 48, sample=anh) is False


def test_so_ca_argmax_chu_khong_chi_so_logit(m1, m1_onnx):
    """Tiêu chí triển khai là LỚP DỰ ĐOÁN, nên tolerance lỏng vẫn phải so argmax.

    Đặt tolerance = 1e9 (bỏ hẳn điều kiện lệch logit). Với model đúng thì vẫn
    KHỚP; với model khác hẳn thì argmax lệch nên phải FAIL — chứng tỏ argmax là
    một điều kiện độc lập, không ăn theo tolerance.
    """
    from gtsrb.deploy.export import verify_onnx
    anh = torch.rand(8, 3, 48, 48)
    assert verify_onnx(m1_onnx, m1, 48, tolerance=1e9, sample=anh) is True

    torch.manual_seed(12345)
    model_khac = build_model("m1_lenet", img_size=48).eval()
    assert verify_onnx(m1_onnx, model_khac, 48, tolerance=1e9, sample=anh) is False
