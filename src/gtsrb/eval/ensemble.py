"""
ensemble — Test-Time Augmentation (TTA) và gộp nhiều model.

CHỦ: Phong Nguyễn

★ VÌ SAO CÓ PHẦN NÀY ★
Bài thắng giải IJCNN 2011 trên chính bộ GTSRB này (IDSIA, 99,46%) **là một
committee of CNNs** — tức ensemble. Nên đây không phải kỹ thuật phụ: nó chính là
cách người ta đạt SOTA trên bài toán này. Có nó thì nhóm mới đặt kết quả của mình
cạnh mốc lịch sử một cách công bằng.

HAI KỸ THUẬT, KHÁC NHAU VỀ CHI PHÍ:

  TTA (test-time augmentation)
      Một model, chạy nhiều lần trên các biến thể của CÙNG một ảnh (xoay nhẹ,
      dịch nhẹ...), rồi lấy trung bình xác suất.
      Chi phí: latency nhân lên đúng số biến thể. KHÔNG cần train thêm.
      Trực giác: nếu model chỉ đúng nhờ một góc nhìn may mắn, lấy trung bình
      nhiều góc sẽ lộ ra sự thiếu chắc chắn đó.

  Ensemble (gộp nhiều model)
      Nhiều model khác nhau, mỗi model chạy một lần, rồi gộp xác suất.
      Chi phí: latency bằng TỔNG các model. Phải train đủ số model.
      Trực giác: các model mắc lỗi ở những chỗ KHÁC nhau; gộp lại thì lỗi riêng
      của từng model bị đa số lấn át.

★ ĐIỀU PHẢI NÓI KHI BÁO CÁO ★
Cả hai đều ĐÁNH ĐỔI LATENCY LẤY ACCURACY. Với bài triển khai biên — mà cả dự án này
đang hướng tới — đó thường là đánh đổi SAI chiều. Báo cáo ensemble mà không báo
latency tăng bao nhiêu lần là báo cáo thiếu.

★ CẤM FLIP NGANG TRONG TTA ★
Giống như augmentation lúc train: lật ngang biến lớp 33 "rẽ phải" thành đúng hình
lớp 34 "rẽ trái". TTA có flip sẽ trung bình xác suất của hai lớp khác nhau —
làm accuracy TỆ ĐI, không phải tốt lên. Xem src/gtsrb/data/transforms.py.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from gtsrb.eval.metrics import metrics_from_probs
from gtsrb.utils.logging import get_logger

log = get_logger()


# =====================================================================
# Test-Time Augmentation
# =====================================================================

def tta_variants(x: torch.Tensor, angles: tuple[float, ...] = (-7.5, 0.0, 7.5),
                 scales: tuple[float, ...] = (1.0,)) -> list[torch.Tensor]:
    """Sinh các biến thể của một batch ảnh để làm TTA.

    x: (B, C, H, W) đã chuẩn hoá.

    CHỈ dùng biến đổi mà nhãn KHÔNG đổi: xoay nhẹ và zoom nhẹ. Không flip, không
    đổi màu — xem ghi chú đầu file.

    Mặc định 3 biến thể (xoay -7,5° / 0° / +7,5°): đủ để thấy tác dụng mà latency
    chỉ nhân 3. Thêm nữa thì lợi ích giảm dần nhưng chi phí vẫn tuyến tính.
    """
    variants = []
    for angle in angles:
        for scale in scales:
            if angle == 0.0 and scale == 1.0:
                variants.append(x)
                continue
            # Ma trận affine 2x3 cho xoay + zoom quanh tâm ảnh
            radian = np.deg2rad(angle)
            cos, sin = np.cos(radian) / scale, np.sin(radian) / scale
            theta = torch.tensor([[cos, -sin, 0.0], [sin, cos, 0.0]],
                                 dtype=x.dtype, device=x.device)
            theta = theta.unsqueeze(0).expand(x.size(0), -1, -1)
            grid = F.affine_grid(theta, list(x.shape), align_corners=False)
            variants.append(F.grid_sample(x, grid, align_corners=False,
                                          padding_mode="zeros"))
    return variants


@torch.no_grad()
def predict_tta(model: torch.nn.Module, loader, device: torch.device,
                angles: tuple[float, ...] = (-7.5, 0.0, 7.5)) -> tuple[np.ndarray, np.ndarray]:
    """Dự đoán có TTA. Trả (y_true, y_prob) y như hàm predict() thường.

    Gộp bằng TRUNG BÌNH XÁC SUẤT (sau softmax), không phải trung bình logit.
    Lý do: logit không cùng thang đo giữa các biến thể, trung bình chúng là phép
    toán không có nghĩa thống kê. Trung bình xác suất thì vẫn là một phân phối hợp lệ.
    """
    model.eval()
    all_probs, all_targets = [], []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        probs = None
        for variant in tta_variants(images, angles=angles):
            p = torch.softmax(model(variant).float(), dim=1)
            probs = p if probs is None else probs + p
        all_probs.append((probs / len(angles)).cpu())
        all_targets.append(targets)

    return torch.cat(all_targets).numpy(), torch.cat(all_probs).numpy()


# =====================================================================
# Ensemble nhiều model
# =====================================================================

def ensemble_probs(prob_list: list[np.ndarray],
                   weights: list[float] | None = None) -> np.ndarray:
    """Gộp xác suất của nhiều model thành một.

    weights=None -> trung bình đều. Truyền trọng số nếu muốn ưu tiên model mạnh hơn,
    nhưng CẢNH BÁO: chọn trọng số dựa trên kết quả TEST là rò rỉ — phải chọn trên
    tập VAL rồi mới áp lên test.
    """
    if not prob_list:
        raise ValueError("prob_list rỗng")
    shapes = {p.shape for p in prob_list}
    if len(shapes) != 1:
        raise ValueError(f"Các model cho shape khác nhau: {shapes}. "
                         f"Chúng phải chạy trên CÙNG tập test, CÙNG thứ tự.")

    stacked = np.stack(prob_list)                      # (n_model, N, 43)
    if weights is None:
        return stacked.mean(axis=0)

    if len(weights) != len(prob_list):
        raise ValueError(f"Cần {len(prob_list)} trọng số, nhận {len(weights)}")
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return (stacked * w[:, None, None]).sum(axis=0)


def ensemble_report(y_true: np.ndarray, members: dict[str, np.ndarray]) -> "pd.DataFrame":
    """Bảng: từng model riêng lẻ, rồi ensemble của tất cả.

    members: {tên model: y_prob}. Mọi y_prob phải cùng thứ tự với y_true.
    """
    import pandas as pd

    rows = []
    for name, probs in members.items():
        stats = metrics_from_probs(y_true, probs)
        rows.append({"thành phần": name,
                     "top-1": round(stats["top1"], 5),
                     "macro-F1": round(stats["macro_f1"], 5)})

    combined = ensemble_probs(list(members.values()))
    stats = metrics_from_probs(y_true, combined)
    rows.append({"thành phần": f"ENSEMBLE ({len(members)} model)",
                 "top-1": round(stats["top1"], 5),
                 "macro-F1": round(stats["macro_f1"], 5)})

    frame = pd.DataFrame(rows)

    # Cột "hơn model tốt nhất bao nhiêu" — câu hỏi thật sự của phần này
    best_single = frame.iloc[:-1]["macro-F1"].max()
    frame["so với model đơn tốt nhất"] = (frame["macro-F1"] - best_single).round(5)
    return frame


def disagreement_rate(prob_list: list[np.ndarray]) -> float:
    """Tỉ lệ ảnh mà các model KHÔNG đồng ý về nhãn dự đoán.

    ★ Đây là chỉ số dự báo ensemble có đáng làm không.
    Ensemble chỉ có lợi khi các thành phần mắc lỗi ở những chỗ KHÁC nhau. Nếu tỉ lệ
    bất đồng gần 0 thì các model về cơ bản giống hệt nhau, gộp lại không được gì mà
    latency vẫn nhân lên — lúc đó đừng làm ensemble.
    """
    preds = np.stack([p.argmax(axis=1) for p in prob_list])    # (n_model, N)
    return float((preds != preds[0]).any(axis=0).mean())
