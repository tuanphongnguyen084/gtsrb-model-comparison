"""
metrics — top-1, top-5, macro-F1, per-class. Hàm evaluate() dùng chung.

CHỦ: Phong Nguyễn

★ CHỈ SỐ NÀO QUAN TRỌNG VÀ VÌ SAO ★

  top-1        : chỉ số trực giác, dùng để báo cáo.

  top-5        : đề bài yêu cầu, NHƯNG trên 43 lớp nó BÃO HOÀ ~99,9%.
                 Top-5 ra đời cho ImageNet 1000 lớp, nơi nhiều lớp mơ hồ chính đáng.
                 Trên 43 lớp, top-5 nghĩa là "đúng trong 11,6% số lớp" — mọi model
                 tử tế đều ~99,9%. CHỈ SỐ BÃO HOÀ THÌ KHÔNG PHÂN BIỆT ĐƯỢC GÌ.
                 Nhóm vẫn báo cáo vì đề bài yêu cầu, nhưng kết luận dựa trên top-1
                 và macro-F1. Phải nói rõ điều này khi trình bày.

  macro-F1     : ★ CHỈ SỐ CHÍNH.
                 macro-F1 = (1/K) * sum_k F1_k    <- mỗi lớp MỘT PHIẾU BẰNG NHAU
                 Dữ liệu mất cân bằng ~10,7:1 nên accuracy bị lớp đông chi phối:
                 model bỏ hẳn một lớp 210 ảnh (0,54% dữ liệu) thì accuracy chỉ giảm
                 ~0,54 điểm (gần như không thấy), nhưng macro-F1 mất 1/43 ~ 2,3 điểm.
                 Trong bài biển báo, nhận sai một biển HIẾM có thể nguy hiểm hơn
                 nhận sai một biển phổ biến.

  weighted-F1  : trung bình có trọng số theo support -> lại bị lớp đông chi phối.
                 Báo cáo cho đủ nhưng không dùng để kết luận.
                 (micro-F1 trong bài đa lớp đơn nhãn BẰNG ĐÚNG accuracy.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support

from gtsrb import CLASS_NAMES, NUM_CLASSES


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor,
                  ks: tuple[int, ...] = (1, 5)) -> dict[int, float]:
    """Top-k accuracy cho một batch.

    logits:  (B, K)
    targets: (B,)
    """
    max_k = min(max(ks), logits.shape[1])
    # topk trả (giá trị, chỉ số); ta chỉ cần chỉ số -> (B, max_k)
    _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
    # So sánh với nhãn đúng, broadcast targets ra (B, max_k)
    correct = pred.eq(targets.view(-1, 1))

    out: dict[int, float] = {}
    for k in ks:
        k_eff = min(k, max_k)
        # any(dim=1): nhãn đúng có nằm trong k ứng viên đầu không
        out[k] = correct[:, :k_eff].any(dim=1).float().mean().item()
    return out


@torch.no_grad()
def predict(model: torch.nn.Module, loader, device: torch.device,
            use_amp: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Chạy model trên toàn bộ loader, trả (y_true, y_prob).

    y_true: (N,) int64
    y_prob: (N, 43) float32 — xác suất sau softmax.
            D cần y_prob cho ECE, C cần cho Grad-CAM, cả hai cần cho top-5.
    """
    model.eval()                    # BẮT BUỘC: tắt dropout, BatchNorm dùng running stats
    all_probs, all_targets = [], []

    amp_device = device.type if device.type in ("cuda", "cpu") else "cpu"
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
            logits = model(images)
        probs = torch.softmax(logits.float(), dim=1)
        all_probs.append(probs.cpu())
        all_targets.append(targets)

    y_prob = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_targets).numpy()
    return y_true, y_prob


def metrics_from_probs(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Tính mọi chỉ số từ (y_true, y_prob). Tách riêng để test được không cần model."""
    y_pred = y_prob.argmax(axis=1)

    top1 = float((y_pred == y_true).mean())

    # top-5: nhãn đúng có nằm trong 5 lớp xác suất cao nhất không
    top5_idx = np.argsort(-y_prob, axis=1)[:, :5]
    top5 = float((top5_idx == y_true[:, None]).any(axis=1).mean())

    # ★ labels=range(43) là BẮT BUỘC, không được bỏ.
    # Không truyền labels thì sklearn chỉ lấy trung bình trên các lớp CÓ MẶT trong
    # y_true/y_pred. Trên tập test đầy đủ (có cả 43 lớp) thì giống nhau, nhưng khi
    # đánh giá trên tập con (ablation với --subset, hoặc một slice robustness) mà
    # thiếu vài lớp thì macro-F1 được tính trên ÍT lớp hơn -> các run KHÔNG SO SÁNH
    # ĐƯỢC với nhau. Đó là lỗi im lặng: số vẫn ra, chỉ là sai ý nghĩa.
    all_labels = list(range(NUM_CLASSES))
    macro_f1 = float(f1_score(y_true, y_pred, labels=all_labels,
                              average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=all_labels,
                                 average="weighted", zero_division=0))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, zero_division=0,
    )
    per_class = pd.DataFrame({
        "class_id": list(range(NUM_CLASSES)),
        "name": CLASS_NAMES,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })

    return {
        "top1": top1,
        "top5": top5,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def evaluate(model: torch.nn.Module, loader, device: torch.device,
             use_amp: bool = False) -> dict:
    """API chính của tầng đánh giá (docs/INTERFACE.md mục 4).

    Trả dict có: top1, top5, macro_f1, weighted_f1, per_class, y_true, y_pred, y_prob.
    Dùng ĐƯỢC cho cả 3 model qua cùng một chữ ký — đó là điều kiện để so sánh công bằng.
    """
    y_true, y_prob = predict(model, loader, device, use_amp=use_amp)
    return metrics_from_probs(y_true, y_prob)


def worst_classes(per_class: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """n lớp có F1 thấp nhất, kèm support để tách hai nguyên nhân:

      F1 thấp + support thấp  -> có thể do THIẾU DỮ LIỆU
      F1 thấp + support cao   -> do HÌNH GIỐNG NHAU (vấn đề độ phân giải)

    Đây là lập luận mà giảng viên muốn thấy: không chỉ nói "lớp này khó"
    mà nói được "khó vì nguyên nhân nào".
    """
    return per_class.nsmallest(n, "f1").reset_index(drop=True)
