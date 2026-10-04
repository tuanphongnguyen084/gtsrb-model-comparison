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

    limit_threads()          # tôn trọng GTSRB_THREADS nếu được đặt

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


def limit_threads(n: int | None = None) -> int:
    """Giới hạn số luồng CPU mà PyTorch dùng, để máy còn dùng được khi train nền.

    ★ VÌ SAO CẦN: mặc định PyTorch dùng HẾT số lõi. Trên MacBook 8 lõi, một mẻ
    train nền sẽ chiếm 600–700% CPU và máy giật tới mức không gõ phím được.
    `nice` chỉ hạ ưu tiên, không giảm số lõi bị chiếm — vẫn giật.

    Đọc biến môi trường GTSRB_THREADS; không đặt thì để OS tự quyết (dùng hết).
    Khuyến nghị khi chạy nền trên máy cá nhân: đặt bằng số lõi trừ 2.

        GTSRB_THREADS=6 python scripts/train.py ...

    Trả về số luồng thực tế đang dùng.
    """
    if n is None:
        env = os.environ.get("GTSRB_THREADS")
        n = int(env) if env and env.isdigit() else None
    if n and n > 0:
        torch.set_num_threads(n)
        # Một số thư viện nền (OpenMP, MKL) đọc biến môi trường riêng
        os.environ.setdefault("OMP_NUM_THREADS", str(n))
        os.environ.setdefault("MKL_NUM_THREADS", str(n))
    return torch.get_num_threads()


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
