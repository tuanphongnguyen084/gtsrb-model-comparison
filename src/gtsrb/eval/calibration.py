"""
calibration — Expected Calibration Error và reliability diagram.

CHỦ: Phong Nguyễn

ECE TRẢ LỜI CÂU HỎI: "khi model nói 90% chắc thì nó đúng 90% số lần không?"

    ECE = sum_m  (|B_m| / n) * | acc(B_m) - conf(B_m) |

  Chia các dự đoán vào M bin theo độ tự tin (nhóm dùng 15 bin), trong mỗi bin
  so sánh accuracy THỰC TẾ với độ tự tin TRUNG BÌNH, rồi lấy trung bình có trọng số.

MẠNG SÂU NỔI TIẾNG TỰ TIN THÁI QUÁ (Guo et al. 2017): conf >> acc.
Nguyên nhân chính là cross-entropy + nhãn one-hot: để loss -> 0 thì softmax phải
đẩy logit lớp đúng ra vô cực. Label smoothing đặt một TRẦN tự tin nên thường
giảm ECE rõ rệt, đôi khi ĐỔI LẤY một chút top-1.

VÌ SAO QUAN TRỌNG TRONG HỆ THỐNG THẬT:
  Nếu độ tự tin có hiệu chỉnh, ta đặt được ngưỡng "dưới 0,8 thì không tự quyết,
  chuyển cho người lái". Với model tự tin thái quá thì ngưỡng đó vô nghĩa, vì mọi
  dự đoán — kể cả sai — đều báo 0,99.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                               n_bins: int = 15) -> dict:
    """Trả {"ece": float, "bins": DataFrame}.

    bins có các cột: bin_lo, bin_hi, count, conf_mean, acc, gap
    """
    confidence = y_prob.max(axis=1)         # độ tự tin = xác suất của lớp được chọn
    prediction = y_prob.argmax(axis=1)
    correct = (prediction == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)
    ece = 0.0
    rows = []

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Bin cuối lấy cả biên phải để không bỏ sót dự đoán có confidence = 1,0
        in_bin = (confidence > lo) & (confidence <= hi) if i > 0 else (confidence <= hi)
        count = int(in_bin.sum())
        if count == 0:
            rows.append({"bin_lo": lo, "bin_hi": hi, "count": 0,
                         "conf_mean": np.nan, "acc": np.nan, "gap": np.nan})
            continue

        conf_mean = float(confidence[in_bin].mean())
        accuracy = float(correct[in_bin].mean())
        gap = abs(accuracy - conf_mean)
        ece += (count / total) * gap

        rows.append({"bin_lo": lo, "bin_hi": hi, "count": count,
                     "conf_mean": conf_mean, "acc": accuracy, "gap": gap})

    return {"ece": float(ece), "bins": pd.DataFrame(rows)}


def plot_reliability(bins: pd.DataFrame, out_path: str | Path,
                     title: str = "Reliability diagram", ece: float | None = None) -> Path:
    """Vẽ reliability diagram: accuracy theo độ tự tin.

    Đường chéo y=x là model hiệu chỉnh HOÀN HẢO.
    Cột nằm DƯỚI đường chéo = tự tin thái quá (overconfident).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    valid = bins.dropna(subset=["conf_mean"])
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="hiệu chỉnh hoàn hảo")
    ax.bar(valid["conf_mean"], valid["acc"], width=0.055,
           edgecolor="black", alpha=0.8, label="model")
    ax.set_xlabel("Độ tự tin trung bình trong bin")
    ax.set_ylabel("Accuracy thực tế trong bin")
    ax.set_title(title + (f"  (ECE = {ece:.4f})" if ece is not None else ""))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
