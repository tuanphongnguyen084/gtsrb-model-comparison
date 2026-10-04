"""
schedulers — optimizer và learning rate schedule.

CHỦ: Hoàng

★ COSINE ANNEALING + WARMUP ★

    lr_t = lr_min + 1/2 * (lr_max - lr_min) * (1 + cos(pi * t / T))

  Đầu kỳ LR lớn  -> khám phá rộng, thoát khỏi vùng kém.
  Cuối kỳ LR -> 0 -> lắng vào đáy.

So với step decay: không phải đoán "giảm ở epoch nào", và không có bước nhảy
làm loss giật.

WARMUP 3 epoch (LR tăng tuyến tính từ 0):
  Ở những bước đầu, ước lượng moment bậc 2 (v_hat) của Adam được tính từ RẤT ÍT
  mẫu nên rất nhiễu -> bước đi có thể rất lớn và sai hướng.
  Với M3 (fine-tune trọng số pretrained) thì warmup còn quan trọng hơn: một bước
  lớn sai hướng ở đầu có thể phá tri thức pretrained.

ADAM vs ADAMW:
  Adam gốc cộng weight decay vào gradient, nên nó bị chia cho sqrt(v_hat) ->
  tham số có gradient lớn bị decay ÍT hơn, không đúng ý nghĩa regularization.
  AdamW tách weight decay ra khỏi gradient adaptive. Dùng AdamW cho M3.
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(model: torch.nn.Module, cfg) -> Optimizer:
    """Dựng optimizer. Nếu config bật discriminative LR thì dùng param_groups của model."""
    train_cfg = cfg.train
    name = str(train_cfg.get("optimizer", "adam")).lower()
    lr = float(train_cfg.lr)
    weight_decay = float(train_cfg.get("weight_decay", 0.0))

    # --- Discriminative LR (chỉ M3 có hàm param_groups) ---
    disc = train_cfg.get("discriminative_lr")
    if disc is not None and hasattr(model, "param_groups"):
        groups = model.param_groups(
            lr_early=float(disc.get("early", 1e-5)),
            lr_middle=float(disc.get("middle", 1e-4)),
            lr_head=float(disc.get("head", 1e-3)),
        )
        params = groups
    else:
        # Chỉ truyền tham số CÓ requires_grad -> quan trọng cho pha freeze của M3,
        # nếu truyền hết thì optimizer vẫn giữ state cho tham số đã đóng băng.
        params = [p for p in model.parameters() if p.requires_grad]

    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9,
                               weight_decay=weight_decay, nesterov=True)
    raise ValueError(f"optimizer phải là adam/adamw/sgd, nhận được: {name!r}")


def build_scheduler(optimizer: Optimizer, cfg, steps_per_epoch: int,
                    total_epochs: int) -> LambdaLR | None:
    """Cosine schedule có warmup, cập nhật theo TỪNG BƯỚC (không theo epoch).

    Cập nhật theo bước cho đường LR mượt hơn hẳn, nhất là khi chỉ train ít epoch.

    Trả None nếu scheduler='none'.
    """
    name = str(cfg.train.get("scheduler", "cosine")).lower()
    if name == "none":
        return None
    if name != "cosine":
        raise ValueError(f"scheduler phải là cosine/none, nhận được: {name!r}")

    warmup_epochs = int(cfg.train.get("warmup_epochs", 3))
    min_lr_ratio = float(cfg.train.get("min_lr_ratio", 0.01))

    total_steps = max(1, steps_per_epoch * total_epochs)
    warmup_steps = min(steps_per_epoch * warmup_epochs, total_steps - 1)

    def lr_lambda(step: int) -> float:
        """Trả về HỆ SỐ NHÂN với lr ban đầu (không phải lr tuyệt đối).

        Nhờ vậy nó hoạt động đúng cả khi có nhiều param_group với lr khác nhau
        (discriminative LR của M3) — mỗi nhóm được nhân cùng một hệ số.
        """
        if step < warmup_steps:
            # Warmup tuyến tính: 0 -> 1
            return (step + 1) / max(1, warmup_steps)
        # Cosine: 1 -> min_lr_ratio
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)
