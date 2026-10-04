"""
benchmark — chạy 3 model x 5 nhiễu x 5 mức = 75 ô số liệu.

CHỦ: Huy
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from gtsrb.data.transforms import build_normalize
from gtsrb.eval.metrics import metrics_from_probs
from gtsrb.robustness.corruptions import CORRUPTIONS, SEVERITIES, apply_corruption
from gtsrb.utils.logging import get_logger

log = get_logger()


@torch.no_grad()
def evaluate_corrupted(model: torch.nn.Module, dataset, device: torch.device,
                       kind: str, severity: int, mean, std,
                       batch_size: int = 128, seed: int = 0,
                       max_samples: int | None = None) -> dict:
    """Đánh giá model trên một (loại nhiễu, mức độ).

    `dataset` phải là GTSRBDataset khởi tạo với return_uint8=True — ta cần ảnh
    THÔ để áp nhiễu, rồi mới chuẩn hoá. Áp nhiễu lên ảnh ĐÃ chuẩn hoá là sai:
    sigma=20/255 không còn nghĩa gì sau khi trừ mean chia std.
    """
    model.eval()
    normalizer = build_normalize(mean, std)
    rng = np.random.default_rng(seed)

    total = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    all_probs, all_targets = [], []
    batch_images, batch_targets = [], []

    def flush() -> None:
        """Đẩy batch đang gom qua model rồi xoá bộ đệm."""
        if not batch_images:
            return
        images = torch.stack(batch_images).to(device)
        logits = model(images)
        all_probs.append(torch.softmax(logits.float(), dim=1).cpu())
        all_targets.append(torch.tensor(batch_targets))
        batch_images.clear()
        batch_targets.clear()

    for i in range(total):
        tensor_uint8, label = dataset[i]              # CHW uint8
        array = tensor_uint8.permute(1, 2, 0).numpy()  # -> HWC uint8
        array = apply_corruption(array, kind, severity, rng)
        tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
        batch_images.append(normalizer(tensor))
        batch_targets.append(label)
        if len(batch_images) == batch_size:
            flush()
    flush()

    y_prob = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_targets).numpy()
    stats = metrics_from_probs(y_true, y_prob)
    return {"top1": stats["top1"], "macro_f1": stats["macro_f1"]}


def robustness_sweep(model: torch.nn.Module, dataset, device: torch.device,
                     mean, std, model_name: str = "model",
                     batch_size: int = 128, seed: int = 0,
                     max_samples: int | None = None) -> pd.DataFrame:
    """Quét toàn bộ 5 nhiễu x 5 mức + 1 lần sạch. Trả DataFrame dài.

    Cột: model, corruption, severity, top1, macro_f1, relative_top1
    """
    rows = []

    clean = evaluate_corrupted(model, dataset, device, "gauss_noise", 0,
                               mean, std, batch_size, seed, max_samples)
    rows.append({"model": model_name, "corruption": "clean", "severity": 0,
                 "top1": clean["top1"], "macro_f1": clean["macro_f1"],
                 "relative_top1": 1.0})
    log.info("[%s] sạch: top1=%.4f", model_name, clean["top1"])

    for kind in tqdm(CORRUPTIONS, desc=f"robustness {model_name}", ncols=88,
                     disable=not sys.stderr.isatty()):
        for severity in SEVERITIES:
            stats = evaluate_corrupted(model, dataset, device, kind, severity,
                                       mean, std, batch_size, seed, max_samples)
            rows.append({
                "model": model_name,
                "corruption": kind,
                "severity": severity,
                "top1": stats["top1"],
                "macro_f1": stats["macro_f1"],
                # ★ relative robustness: tách "giỏi sẵn" khỏi "bền"
                "relative_top1": stats["top1"] / clean["top1"] if clean["top1"] else 0.0,
            })

    return pd.DataFrame(rows)


def mean_corruption_error(frame: pd.DataFrame) -> float:
    """mCE kiểu ImageNet-C: trung bình error trên mọi nhiễu và mọi mức.

    Một con số duy nhất để xếp hạng. Càng THẤP càng tốt.
    """
    corrupted = frame[frame["corruption"] != "clean"]
    return float((1.0 - corrupted["top1"]).mean())
