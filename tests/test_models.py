"""
test_models.py — khoá HỢP ĐỒNG của tầng model (docs/INTERFACE.md mục 2).

CHỦ: Hoàng

Các test này bắt những lỗi làm C hoặc D tắc việc:
  - thiếu .gradcam_target_layer  -> C không làm được Grad-CAM
  - sai expected_img_size        -> M3 nhận ảnh 48 và cho kết quả tệ, KHÔNG BÁO LỖI
  - vỡ khi đổi img_size          -> không làm được ablation resolution
"""
import pytest
import torch
import torch.nn as nn

from gtsrb import NUM_CLASSES
from gtsrb.models.registry import (MODEL_NAMES, build_model, count_parameters,
                                   parameter_table)

# M3 cần tải trọng số pretrained (~45 MB) nên tách riêng, có thể bỏ qua khi offline
SCRATCH_MODELS = ["m1_lenet", "m2_vggres"]


@pytest.mark.parametrize("name", SCRATCH_MODELS)
def test_forward_dung_shape(name):
    """(B,3,48,48) -> (B,43). Sai shape đầu ra là vỡ ngay ở loss."""
    model = build_model(name, img_size=48)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(4, 3, 48, 48))
    assert out.shape == (4, NUM_CLASSES), f"{name} cho shape {tuple(out.shape)}"


@pytest.mark.parametrize("name", SCRATCH_MODELS)
def test_co_du_thuoc_tinh_theo_hop_dong(name):
    model = build_model(name, img_size=48)
    assert isinstance(model.gradcam_target_layer, nn.Module)
    assert isinstance(model.model_name, str)
    assert model.expected_img_size == 48
    assert model.normalize_mode == "gtsrb"


@pytest.mark.parametrize("name", SCRATCH_MODELS)
@pytest.mark.parametrize("img_size", [32, 48, 64])
def test_doi_img_size_khong_vo(name, img_size):
    """Ablation resolution cần điều này.

    Lỗi kinh điển: hard-code 9216 cho lớp Flatten của M1 -> đổi img_size là
    'mat1 and mat2 shapes cannot be multiplied'. M1 tự tính nên không vỡ.
    """
    model = build_model(name, img_size=img_size)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, img_size, img_size))
    assert out.shape == (2, NUM_CLASSES)


def test_gradcam_target_layer_khong_lam_nhan_doi_state_dict():
    """★ Lỗi tinh vi: nếu gán self.gradcam_target_layer = module (thay vì dùng
    @property) thì module đó bị đăng ký LẦN THỨ HAI vào _modules, và state_dict
    chứa trọng số NHÂN ĐÔI -> checkpoint phình ra và gây lẫn lộn khi nạp lại."""
    for name in SCRATCH_MODELS:
        model = build_model(name, img_size=48)
        keys = list(model.state_dict().keys())
        assert not any(k.startswith("gradcam_target_layer") for k in keys), (
            f"{name}: gradcam_target_layer bị đăng ký thành submodule. "
            f"Dùng @property thay vì gán thuộc tính."
        )


def test_m1_co_97_phan_tram_tham_so_o_mot_lop_fc():
    """★ Luận điểm trung tâm của M1 — đây là thứ phải nói với giảng viên.

    Nếu test này đỏ thì M1 đã bị sửa và mất ý nghĩa làm mốc tham chiếu.
    """
    model = build_model("m1_lenet", img_size=48)
    table = parameter_table(model)
    biggest = table.iloc[0]
    assert biggest["type"] == "Linear", "Lớp nhiều tham số nhất của M1 phải là Linear"
    assert biggest["percent"] > 90, (
        f"Lớp FC đầu chỉ chiếm {biggest['percent']:.1f}% — mong đợi >90%. "
        f"Đây là bằng chứng cho việc kiến trúc hiện đại dùng GlobalAvgPool."
    )
    assert count_parameters(model) == 2_424_299, (
        f"Số tham số M1 đổi thành {count_parameters(model)}. Nếu đổi kiến trúc "
        f"có chủ ý thì cập nhật cả docs/ cho khớp."
    )


def test_m2_it_tham_so_hon_M1_du_sau_hon_nhieu():
    """★ Kết quả ĐO THẬT, và là luận điểm mạnh nhất về Global Average Pooling.

        M1:  2.424.299 tham số,  2 lớp conv
        M2:  1.237.451 tham số,  9 lớp conv (4 stage x 2 + stem)

    M2 sâu gấp hơn 4 lần mà chỉ bằng ~51% số tham số của M1. Hai lý do:
      (a) conv dùng THAM SỐ CHIA SẺ (một kernel 3x3 trượt khắp ảnh) nên rẻ;
      (b) M2 thay Flatten+FC (2.359.552 tham số ở M1) bằng GlobalAvgPool + FC
          (chỉ 11.051 tham số) -> giảm ~213 lần ở riêng phần head.

    BÀI HỌC: SỐ LỚP KHÔNG TỈ LỆ VỚI SỐ THAM SỐ. Chỗ đắt là lớp fully-connected.
    """
    m2_params = count_parameters(build_model("m2_vggres", img_size=48))
    m1_params = count_parameters(build_model("m1_lenet", img_size=48))

    assert m2_params < 6_000_000, f"M2 có {m2_params} tham số, vượt ngưỡng 6M"
    assert m2_params < m1_params, (
        f"M2 ({m2_params:,}) phải ÍT tham số hơn M1 ({m1_params:,}) "
        f"dù sâu hơn nhiều — đó là tác dụng của GlobalAvgPool."
    )
    assert m2_params == 1_237_451, (
        f"Số tham số M2 đổi thành {m2_params:,}. Nếu đổi kiến trúc có chủ ý "
        f"thì cập nhật cả docs/ và các test cho khớp."
    )

    # Tham số của M2 dồn vào stage CUỐI (nhiều kênh nhất) — không dồn vào FC như M1
    table = parameter_table(build_model("m2_vggres", img_size=48))
    biggest = table.iloc[0]
    assert biggest["type"] == "Conv2d", (
        "Lớp nhiều tham số nhất của M2 phải là Conv2d (không phải Linear như M1)"
    )


def test_m2_cac_co_ablation_doi_so_tham_so():
    """3 cờ use_bn / use_residual / use_spatial_dropout phải thực sự có tác dụng."""
    full = count_parameters(build_model("m2_vggres", img_size=48))
    no_bn = count_parameters(build_model("m2_vggres", img_size=48, use_bn=False))
    no_res = count_parameters(build_model("m2_vggres", img_size=48, use_residual=False))

    assert no_bn != full, "use_bn=False không đổi gì — cờ ablation không hoạt động"
    assert no_res != full, "use_residual=False không đổi gì"


def test_m2_scaling_hoat_dong():
    """width_mult và n_stages phải đổi được, nếu không thì không ablation được."""
    base = count_parameters(build_model("m2_vggres", img_size=48, width_mult=1.0))
    wide = count_parameters(build_model("m2_vggres", img_size=48, width_mult=2.0))
    thin = count_parameters(build_model("m2_vggres", img_size=48, width_mult=0.5))
    assert thin < base < wide, f"width_mult không hoạt động: {thin}/{base}/{wide}"

    shallow = count_parameters(build_model("m2_vggres", img_size=48, n_stages=3))
    assert shallow < base, "n_stages không hoạt động"


def test_ten_model_sai_thi_bao_loi_ro():
    with pytest.raises(ValueError, match="name"):
        build_model("khong_ton_tai")


def test_bang_tham_so_cong_lai_bang_tong():
    """parameter_table phải dùng recurse=False, nếu không module cha đếm lại
    tham số của con và tổng vượt 100%."""
    model = build_model("m2_vggres", img_size=48)
    table = parameter_table(model)
    assert abs(table["params"].sum() - count_parameters(model)) == 0
    assert abs(table["percent"].sum() - 100.0) < 0.01


@pytest.mark.slow
@pytest.mark.parametrize("name", ["m3_resnet18", "m3_mobilenetv2", "m3_effnetb0"])
def test_m3_hop_dong(name):
    """M3: cần mạng, tải trọng số pretrained. Chạy: pytest -m slow"""
    model = build_model(name, img_size=224, pretrained=False)  # False để test nhanh
    assert model.expected_img_size == 224
    assert model.normalize_mode == "imagenet", (
        "M3 PHẢI dùng normalize='imagenet' — trọng số pretrained và running stats "
        "của BatchNorm được học trên phân phối đó."
    )
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, NUM_CLASSES)

    # Freeze/unfreeze phải thực sự đổi số tham số train được
    model.set_backbone_frozen(True)
    frozen = count_parameters(model, only_trainable=True)
    model.set_backbone_frozen(False)
    unfrozen = count_parameters(model, only_trainable=True)
    assert frozen < unfrozen, "set_backbone_frozen không có tác dụng"

    # Discriminative LR phải cho ra các nhóm LR KHÁC NHAU
    groups = model.param_groups(lr_early=1e-5, lr_middle=1e-4, lr_head=1e-3)
    lrs = [g["lr"] for g in groups]
    assert len(set(lrs)) == len(lrs) >= 2, f"Các nhóm LR không khác nhau: {lrs}"
    assert lrs == sorted(lrs), "LR phải TĂNG dần từ tầng đầu đến head"
