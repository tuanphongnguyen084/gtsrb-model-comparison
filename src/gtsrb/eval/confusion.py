"""
confusion — confusion matrix 43x43 và các cặp lớp bị nhầm nhiều nhất.

CHỦ: Phong Nguyễn

VÌ SAO CẦN confusion matrix khi đã có accuracy:
  Accuracy là MỘT số. Confusion matrix cho biết lỗi ĐI ĐÂU.
  Hai bức tranh rất khác nhau:
    - lỗi rải đều trên mọi lớp        -> model yếu toàn diện
    - lỗi tập trung vào vài CẶP lớp   -> vấn đề cụ thể, sửa được cụ thể
  Nếu lỗi tập trung ở cặp "giới hạn 60 <-> 80" thì cách sửa là TĂNG ĐỘ PHÂN GIẢI
  hoặc thêm dữ liệu cho đúng hai lớp đó — hoàn toàn khác với "cần model to hơn".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from gtsrb import CLASS_NAMES, NUM_CLASSES


def confusion(y_true: np.ndarray, y_pred: np.ndarray,
              normalize: bool = False) -> np.ndarray:
    """Confusion matrix (43, 43). matrix[i][j] = số ảnh lớp thật i bị dự đoán là j.

    normalize=True chia theo từng hàng -> mỗi hàng là phân phối dự đoán của lớp đó.
    Nên dùng normalize khi vẽ hình, vì lớp 2.250 ảnh sẽ làm lớp 210 ảnh vô hình
    nếu dùng số đếm thô.
    """
    matrix = sk_confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            matrix = np.where(row_sums > 0, matrix / row_sums, 0.0)
    return matrix


def top_confusions(y_true: np.ndarray, y_pred: np.ndarray, n: int = 10) -> pd.DataFrame:
    """n cặp (lớp thật, lớp dự đoán) bị nhầm nhiều nhất. Bỏ đường chéo (dự đoán đúng).

    Đây là bảng đáng đưa vào báo cáo hơn cả hình confusion matrix 43x43,
    vì hình đó nhìn vào rất khó đọc ra con số cụ thể.
    """
    matrix = confusion(y_true, y_pred)
    np.fill_diagonal(matrix, 0)             # chỉ quan tâm LỖI

    rows = []
    flat_order = np.argsort(matrix, axis=None)[::-1][:n]
    for flat_index in flat_order:
        true_id, pred_id = np.unravel_index(flat_index, matrix.shape)
        count = int(matrix[true_id, pred_id])
        if count == 0:
            break
        rows.append({
            "true_id": int(true_id),
            "true_name": CLASS_NAMES[true_id],
            "pred_id": int(pred_id),
            "pred_name": CLASS_NAMES[pred_id],
            "count": count,
        })
    return pd.DataFrame(rows)


def plot_confusion(matrix: np.ndarray, out_path: str | Path,
                   title: str = "Confusion matrix") -> Path:
    """Vẽ confusion matrix ra file PNG."""
    import matplotlib
    matplotlib.use("Agg")           # backend không cần màn hình (chạy được trên server)
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=matrix.max() or 1)
    ax.set_xlabel("Lớp dự đoán")
    ax.set_ylabel("Lớp thật")
    ax.set_title(title)
    ax.set_xticks(range(0, NUM_CLASSES, 2))
    ax.set_yticks(range(0, NUM_CLASSES, 2))
    fig.colorbar(image, ax=ax, label="tỉ lệ" if matrix.max() <= 1.0 else "số ảnh")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
