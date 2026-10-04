"""
stats_tests — kiểm định McNemar: "chênh lệch này có THẬT không?"

CHỦ: Phong Nguyễn

★ VÌ SAO KHÔNG DÙNG t-test ★
  Hai model chạy trên CÙNG tập test -> quan sát BẮT CẶP (paired), không độc lập.
  Dùng t-test hai mẫu độc lập là SAI VỀ MẶT THỐNG KÊ.
  McNemar là test đúng cho trường hợp này.

BẢNG 2x2:
                 B đúng    B sai
    A đúng        n00       n01
    A sai         n10       n11

  n00 và n11 là những ảnh hai model ĐỒNG Ý — chúng KHÔNG MANG THÔNG TIN về
  sự khác biệt giữa hai model. Chỉ hai ô BẤT ĐỒNG (n01, n10) mới nói được gì.

  Giả thuyết không H0: n01 và n10 có cùng kỳ vọng (hai model tương đương).

      chi2 = (|n01 - n10| - 1)^2 / (n01 + n10)      df = 1
                              ^-- hiệu chỉnh liên tục Yates

  Khi n01 + n10 < 25 thì xấp xỉ chi-square không đáng tin -> dùng BINOMIAL CHÍNH XÁC.

VÌ SAO BÀI NÀY CẦN NÓ:
  Trên 12.630 ảnh test, chênh 0,2 điểm top-1 chỉ là ~25 ảnh.
  Nếu p > 0,05 thì phải nói thẳng "chênh lệch KHÔNG CÓ Ý NGHĨA THỐNG KÊ",
  và khi đó lựa chọn triển khai chuyển sang dựa vào TỐC ĐỘ và ROBUSTNESS.
  Đó là một kết luận MẠNH HƠN, không phải yếu hơn.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def mcnemar(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray,
            exact_threshold: int = 25) -> dict:
    """Kiểm định McNemar giữa hai model trên cùng tập test.

    Trả về:
      n00, n01, n10, n11 : bảng 2x2
      statistic          : chi2 (hoặc min(n01,n10) nếu dùng binomial)
      p_value            : float
      method             : "chi2_corrected" | "exact_binomial"
      better             : "A" | "B" | "tie"
      verdict            : câu kết luận bằng tiếng Việt, dán thẳng vào báo cáo được
    """
    y_true = np.asarray(y_true)
    correct_a = np.asarray(y_pred_a) == y_true
    correct_b = np.asarray(y_pred_b) == y_true

    n00 = int(np.sum(correct_a & correct_b))      # cả hai đúng
    n01 = int(np.sum(correct_a & ~correct_b))     # chỉ A đúng
    n10 = int(np.sum(~correct_a & correct_b))     # chỉ B đúng
    n11 = int(np.sum(~correct_a & ~correct_b))    # cả hai sai

    discordant = n01 + n10

    if discordant == 0:
        # Hai model cho dự đoán giống hệt nhau trên mọi ảnh
        statistic, p_value, method = 0.0, 1.0, "identical"
    elif discordant < exact_threshold:
        # Binomial chính xác: dưới H0, n01 ~ Binomial(discordant, 0.5)
        statistic = float(min(n01, n10))
        p_value = float(stats.binomtest(min(n01, n10), discordant, 0.5,
                                        alternative="two-sided").pvalue)
        method = "exact_binomial"
    else:
        statistic = float((abs(n01 - n10) - 1) ** 2 / discordant)
        p_value = float(stats.chi2.sf(statistic, df=1))
        method = "chi2_corrected"

    if p_value >= 0.05:
        better = "tie"
        verdict = (f"Chênh lệch KHÔNG có ý nghĩa thống kê (p={p_value:.4f} >= 0,05). "
                   f"Hai model coi như tương đương về accuracy; nên chọn dựa vào "
                   f"tốc độ và robustness.")
    elif n01 > n10:
        better = "A"
        verdict = (f"Model A tốt hơn CÓ ý nghĩa thống kê (p={p_value:.4g} < 0,05). "
                   f"A đúng riêng {n01} ảnh, B đúng riêng {n10} ảnh.")
    else:
        better = "B"
        verdict = (f"Model B tốt hơn CÓ ý nghĩa thống kê (p={p_value:.4g} < 0,05). "
                   f"B đúng riêng {n10} ảnh, A đúng riêng {n01} ảnh.")

    return {
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "n_discordant": discordant,
        "statistic": statistic,
        "p_value": p_value,
        "method": method,
        "better": better,
        "verdict": verdict,
    }
