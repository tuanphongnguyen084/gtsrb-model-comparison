"""
registry — một cửa duy nhất để dựng model: build_model(name).

CHỦ: Hoàng

HỢP ĐỒNG (docs/INTERFACE.md mục 2): mọi model trả về từ đây BẮT BUỘC có:
    .gradcam_target_layer  -> nn.Module   (C cần cho Grad-CAM)
    .model_name            -> str
    .expected_img_size     -> int          (48 cho M1/M2, 224 cho M3)
    .normalize_mode        -> str          ("gtsrb" | "imagenet")

VÌ SAO cần expected_img_size và normalize_mode GẮN TRÊN MODEL:
    để scripts/train.py tự chọn đúng cấu hình dataset, không ai phải nhớ
    "M3 thì nhớ đổi sang 224 và ImageNet stats". Quên chuyện đó là một LỖI IM LẶNG:
    model vẫn train bình thường, chỉ là kết quả tệ mà không ai biết vì sao.
"""

from __future__ import annotations

import torch.nn as nn

from gtsrb import NUM_CLASSES
from gtsrb.models.m1_lenet import M1LeNet
from gtsrb.models.m2_vggres import M2VggRes
from gtsrb.models.m3_transfer import M3Transfer

MODEL_NAMES = (
    "m1_lenet",
    "m2_vggres",
    "m3_resnet18",
    "m3_mobilenetv2",
    "m3_effnetb0",
)


def build_model(name: str, num_classes: int = NUM_CLASSES, **kwargs) -> nn.Module:
    """Dựng model theo tên. kwargs được truyền thẳng vào constructor.

    Ví dụ:
        build_model("m2_vggres", img_size=48, width_mult=2.0)
        build_model("m3_resnet18", img_size=224, pretrained=True)
    """
    if name == "m1_lenet":
        model = M1LeNet(num_classes=num_classes, **kwargs)
    elif name == "m2_vggres":
        model = M2VggRes(num_classes=num_classes, **kwargs)
    elif name.startswith("m3_"):
        backbone = name[len("m3_"):]
        model = M3Transfer(backbone=backbone, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"name phải thuộc {MODEL_NAMES}, nhận được: {name!r}")

    # Kiểm tra hợp đồng ngay tại đây. Thà vỡ ở dòng này với thông điệp rõ ràng
    # còn hơn vỡ ở giữa buổi chạy Grad-CAM lúc 2 giờ sáng.
    for attribute in ("gradcam_target_layer", "model_name",
                      "expected_img_size", "normalize_mode"):
        if not hasattr(model, attribute):
            raise RuntimeError(
                f"Model {name!r} thiếu thuộc tính bắt buộc {attribute!r}. "
                f"Xem hợp đồng ở docs/INTERFACE.md mục 2."
            )
    return model


def count_parameters(model: nn.Module, only_trainable: bool = False) -> int:
    """Đếm tham số. only_trainable=True để kiểm pha freeze của M3 có đúng không."""
    params = model.parameters()
    if only_trainable:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def parameter_table(model: nn.Module):
    """Bảng đếm tham số theo từng lớp, sắp giảm dần.

    Dùng để chứng minh luận điểm của M1: ~96% tham số nằm ở MỘT lớp FC.
    Trả về pandas.DataFrame.
    """
    import pandas as pd

    rows = []
    total = count_parameters(model)
    for module_name, module in model.named_modules():
        # Chỉ đếm tham số TRỰC TIẾP của module (recurse=False), nếu không thì
        # module cha sẽ đếm lại tham số của con -> tổng vượt 100%.
        own = sum(p.numel() for p in module.parameters(recurse=False))
        if own > 0:
            rows.append({
                "layer": module_name or "(root)",
                "type": type(module).__name__,
                "params": own,
                "percent": 100.0 * own / total if total else 0.0,
            })
    frame = pd.DataFrame(rows).sort_values("params", ascending=False)
    return frame.reset_index(drop=True)
