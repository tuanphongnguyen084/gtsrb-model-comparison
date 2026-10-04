"""
corruptions — 5 loại nhiễu mô phỏng điều kiện thực tế, 5 mức độ mỗi loại.

CHỦ: Huy

★ LUẬT SẮT CỦA FILE NÀY ★
Các loại nhiễu ở đây CHỈ ĐƯỢC DÙNG LÚC TEST, TUYỆT ĐỐI KHÔNG ĐƯA VÀO TRAIN.

VÌ SAO: robustness nghĩa là khái quát sang PHÂN PHỐI CHƯA TỪNG THẤY
(distribution shift). Nếu train trên chúng rồi test trên chúng thì đó là test
in-distribution — chỉ chứng minh "model học được cái nó đã thấy". Đó là
augmentation, không phải robustness, và phép đo mất hoàn toàn ý nghĩa.

(src/gtsrb/data/transforms.py KHÔNG BAO GIỜ import file này. Có test kiểm điều đó.)

5 LOẠI NHIỄU VÀ MÔ HÌNH VẬT LÝ:

  motion_blur  xe đang chạy, biển lướt qua
               tích chập với kernel ĐƯỜNG THẲNG, dài 3->15 px, góc ngẫu nhiên

  gauss_noise  camera ISO cao, cảm biến kém
               I' = I + N(0, sigma^2),  sigma = 5/10/20/40/60 (trên thang 255)

  fog          sương mù, mưa
               MÔ HÌNH TÁN XẠ KHÍ QUYỂN (Koschmieder):
                   I = J * t + A * (1 - t),   t = exp(-beta * d)
               J = ảnh gốc, A = độ sáng khí quyển, beta = hệ số tán xạ,
               d = khoảng cách. beta càng lớn -> sương càng dày.

  low_light    ban đêm
               gamma correction I' = 255 * (I/255)^gamma, gamma = 1,5 -> 3,0
               kèm NHIỄU POISSON — nhiễu photon thật của ảnh tối, không phải
               Gaussian. Ảnh càng tối thì tỉ lệ nhiễu/tín hiệu càng cao.

  occlusion    lá cây, sticker, biển bị che
               hình chữ nhật che 10/20/30/40/50% diện tích, vị trí ngẫu nhiên

★ BÁO CÁO RELATIVE ROBUSTNESS, không chỉ accuracy tuyệt đối ★
    relative = acc_nhiễu / acc_sạch
Nếu chỉ báo accuracy tuyệt đối thì model nào giỏi sẵn sẽ luôn thắng và ta không
học được gì về TÍNH BỀN. Chia cho accuracy sạch tách hai thứ đó ra.
"""

from __future__ import annotations

import cv2
import numpy as np

CORRUPTIONS = ("motion_blur", "gauss_noise", "fog", "low_light", "occlusion")
SEVERITIES = (1, 2, 3, 4, 5)


def _as_uint8(array: np.ndarray) -> np.ndarray:
    """Clip về [0,255] rồi đổi sang uint8. Mọi hàm nhiễu kết thúc bằng hàm này.

    Bỏ bước clip là lỗi im lặng kinh điển: giá trị 300 bị wrap thành 44 (300-256),
    biến vùng sáng thành vùng tối và tạo ra ảnh hoàn toàn khác ý định.
    """
    return np.clip(array, 0, 255).astype(np.uint8)


def motion_blur(image: np.ndarray, severity: int,
                rng: np.random.Generator) -> np.ndarray:
    """Mờ chuyển động: tích chập với kernel đường thẳng."""
    lengths = {1: 3, 2: 5, 3: 7, 4: 11, 5: 15}
    size = lengths[severity]

    # Kernel toàn 0, vẽ một đường thẳng qua tâm rồi chuẩn hoá tổng = 1
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0
    angle = float(rng.uniform(0, 180))
    rotation = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (size, size))
    total = kernel.sum()
    if total > 0:
        kernel /= total          # chuẩn hoá để độ sáng trung bình không đổi

    return _as_uint8(cv2.filter2D(image.astype(np.float32), -1, kernel))


def gauss_noise(image: np.ndarray, severity: int,
                rng: np.random.Generator) -> np.ndarray:
    """Nhiễu Gaussian cộng tính."""
    sigmas = {1: 5.0, 2: 10.0, 3: 20.0, 4: 40.0, 5: 60.0}
    noise = rng.normal(0.0, sigmas[severity], size=image.shape)
    return _as_uint8(image.astype(np.float32) + noise)


def fog(image: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Sương mù theo mô hình tán xạ khí quyển.

        I = J * t + A * (1 - t),   t = exp(-beta * d)

    Dùng bản đồ khoảng cách d tăng dần từ tâm ra biên: biển báo ở giữa ảnh nên
    tâm "gần" hơn, giống cảnh thật khi xe tiến tới biển báo.
    """
    betas = {1: 0.4, 2: 0.8, 3: 1.3, 4: 1.9, 5: 2.6}
    beta = betas[severity]
    atmospheric_light = 235.0           # màu sương: gần trắng

    height, width = image.shape[:2]
    # Bản đồ khoảng cách: 0 ở tâm, 1 ở góc
    ys, xs = np.mgrid[0:height, 0:width]
    center_y, center_x = (height - 1) / 2.0, (width - 1) / 2.0
    distance = np.sqrt(((ys - center_y) / center_y) ** 2 +
                       ((xs - center_x) / center_x) ** 2)
    distance = distance / (distance.max() or 1.0)

    # Cộng thêm một chút biến thiên ngẫu nhiên cho tự nhiên hơn
    distance = np.clip(distance + rng.normal(0, 0.03, distance.shape), 0, None)

    transmission = np.exp(-beta * (0.4 + distance))[..., None]   # (H,W,1) để broadcast
    hazy = image.astype(np.float32) * transmission + \
        atmospheric_light * (1.0 - transmission)
    return _as_uint8(hazy)


def low_light(image: np.ndarray, severity: int,
              rng: np.random.Generator) -> np.ndarray:
    """Điều kiện ban đêm: gamma > 1 làm tối, kèm nhiễu Poisson (nhiễu photon)."""
    gammas = {1: 1.5, 2: 1.9, 3: 2.3, 4: 2.7, 5: 3.2}
    gamma = gammas[severity]

    normalized = image.astype(np.float32) / 255.0
    darkened = np.power(normalized, gamma) * 255.0

    # Nhiễu Poisson: phương sai BẰNG kỳ vọng -> vùng tối nhiễu tương đối nhiều hơn.
    # Đây là nhiễu photon thật của cảm biến, không phải Gaussian.
    photon_scale = 12.0        # càng nhỏ càng ít photon -> càng nhiễu
    noisy = rng.poisson(np.clip(darkened, 0, None) * photon_scale / 255.0)
    noisy = noisy / photon_scale * 255.0
    return _as_uint8(noisy)


def occlusion(image: np.ndarray, severity: int,
              rng: np.random.Generator) -> np.ndarray:
    """Che một phần bằng hình chữ nhật xám (lá cây, sticker, biển bị che)."""
    area_fractions = {1: 0.10, 2: 0.20, 3: 0.30, 4: 0.40, 5: 0.50}
    fraction = area_fractions[severity]

    height, width = image.shape[:2]
    # Hình chữ nhật tỉ lệ ngẫu nhiên nhưng diện tích đúng bằng fraction
    aspect = float(rng.uniform(0.6, 1.6))
    box_h = int(round(np.sqrt(fraction * height * width / aspect)))
    box_w = int(round(aspect * box_h))
    box_h, box_w = min(box_h, height), min(box_w, width)

    top = int(rng.integers(0, max(1, height - box_h + 1)))
    left = int(rng.integers(0, max(1, width - box_w + 1)))

    out = image.copy()
    out[top:top + box_h, left:left + box_w] = 114    # xám trung tính
    return out


_DISPATCH = {
    "motion_blur": motion_blur,
    "gauss_noise": gauss_noise,
    "fog": fog,
    "low_light": low_light,
    "occlusion": occlusion,
}


def apply_corruption(image: np.ndarray, kind: str, severity: int,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """API chính (docs/INTERFACE.md mục 5).

    image: HWC uint8 -> HWC uint8, CÙNG shape CÙNG dtype.
    kind in CORRUPTIONS, severity in 0..5 (0 = không làm gì).

    BẤT BIẾN (tests/test_corruptions.py kiểm):
      - severity=0 trả về ảnh y NGUYÊN
      - output luôn là uint8 trong [0,255], cùng shape
      - cùng rng (cùng seed) -> cùng kết quả
    """
    if kind not in _DISPATCH:
        raise ValueError(f"kind phải thuộc {CORRUPTIONS}, nhận được: {kind!r}")
    if severity == 0:
        return image
    if severity not in SEVERITIES:
        raise ValueError(f"severity phải thuộc {(0,) + SEVERITIES}, nhận: {severity}")
    if image.dtype != np.uint8:
        raise TypeError(f"image phải là uint8, nhận được {image.dtype}")

    if rng is None:
        rng = np.random.default_rng(0)
    return _DISPATCH[kind](image, severity, rng)
