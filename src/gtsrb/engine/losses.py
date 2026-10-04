"""
losses — cross-entropy + label smoothing.

CHỦ: Hoàng

★ LABEL SMOOTHING ★

Nhãn one-hot:    y = [0, 0, 1, 0, ...]
Loss = -log p_c. Để loss -> 0 thì cần p_c -> 1, mà softmax chỉ đạt được điều đó
khi LOGIT của lớp đúng ra VÔ CỰC. Tức là one-hot LUÔN đẩy model về phía tự tin
cực đoan, kể cả khi ảnh thật sự mơ hồ.

Label smoothing (Szegedy et al. 2016):

    y'_k = (1 - eps) * y_k + eps / K

Với K = 43, eps = 0,1:
    lớp đúng nhận  0,9 + 0,1/43 = 0,902326
    mỗi lớp sai    0,1/43       = 0,002326

Giờ loss tối thiểu đạt ở một mức tự tin HỮU HẠN — có một TRẦN.

HỆ QUẢ:
  + độ hiệu chỉnh tốt hơn (ECE giảm — xem eval/calibration.py)
  + biên quyết định đều hơn, bền hơn một chút trước nhiễu
  - top-1 thô CÓ THỂ giảm nhẹ  <- đây là đánh đổi, phải đo chứ không đoán

Nhóm có ablation eps = 0,0 / 0,1 / 0,2 và đo cả ECE để nói về đánh đổi này bằng số.

GHI CHÚ: PyTorch có sẵn label_smoothing trong CrossEntropyLoss từ 1.10, không
cần tự hiện thực. Tự viết lại là cơ hội để sai công thức.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def build_criterion(cfg) -> nn.Module:
    """Dựng loss function từ config.

    class_weight="balanced" là hướng mở rộng cho dữ liệu mất cân bằng:
    cân trọng số nghịch với tần suất lớp. Nhóm KHÔNG dùng mặc định, vì đã chọn
    can thiệp ở CHỈ SỐ (macro-F1) thay vì ở loss — xem docs/LY_THUYET.md 1.4.
    """
    smoothing = float(cfg.train.get("label_smoothing", 0.1))
    if not 0.0 <= smoothing < 1.0:
        raise ValueError(f"label_smoothing phải trong [0,1), nhận: {smoothing}")

    weight = None
    class_weight = cfg.train.get("class_weight")
    if class_weight is not None and class_weight != "none":
        weight = torch.tensor(class_weight, dtype=torch.float32)

    return nn.CrossEntropyLoss(label_smoothing=smoothing, weight=weight)
