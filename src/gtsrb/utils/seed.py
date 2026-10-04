"""
seed — cố định mọi nguồn ngẫu nhiên để thực nghiệm tái lập được.

CHỦ: Huy

VÌ SAO QUAN TRỌNG: nếu không cố định seed thì chạy lại cùng config ra kết quả khác nhau,
và ta không phân biệt được "thay đổi của tôi giúp +0,2 điểm" với "chỉ là nhiễu khởi tạo".
Mọi so sánh trong dự án này đều vô nghĩa nếu thiếu bước này.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Cố định seed cho random, numpy, torch (CPU + CUDA + MPS).

    deterministic=True làm cuDNN chọn thuật toán tiền định (chậm hơn ~5-10%
    nhưng cho kết quả lặp lại được). Khi chạy benchmark tốc độ thì nên để False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                 # CPU
    torch.cuda.manual_seed_all(seed)        # mọi GPU CUDA (no-op nếu không có)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # benchmark=True cho cuDNN tự dò thuật toán conv nhanh nhất -> nhanh hơn
        # nhưng không tiền định. Dùng khi đo tốc độ, không dùng khi so sánh accuracy.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """Hàm truyền vào DataLoader(worker_init_fn=...).

    Mỗi worker của DataLoader là một process riêng với seed riêng; không set
    thì augmentation ngẫu nhiên khác nhau giữa các lần chạy.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def pick_device(prefer: str = "auto") -> torch.device:
    """Chọn thiết bị tính toán.

    'auto' -> cuda nếu có (Colab), rồi mps (Apple Silicon), cuối cùng cpu.
    """
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
