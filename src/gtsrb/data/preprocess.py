"""
preprocess — cắt ROI, cân bằng sáng (histogram equalization), resize.

CHỦ: Huy

4 chế độ cân bằng sáng, cố ý làm cả 4 để bước ablation có cái so sánh:

  none     : RGB thô, chỉ resize.                          -> đối chứng
  he_gray  : chuyển ảnh xám rồi cv2.equalizeHist.           -> cách của Sermanet & LeCun 2011
  he_y     : equalize kênh Y của YUV, giữ nguyên U,V.       -> giữ được MÀU
  clahe    : CLAHE trên kênh L của LAB.                     -> MẶC ĐỊNH

VÌ SAO equalize trên kênh L/Y mà không trên từng kênh R,G,B:
  Equalize R, G, B độc lập làm lệch tỉ lệ giữa 3 kênh -> ẢNH ĐỔI MÀU.
  Mà màu biển báo MANG NGHĨA: đỏ = cấm, xanh = bắt buộc.
  LAB/YUV tách độ sáng (L, Y) khỏi màu (A,B / U,V) -> sửa sáng, giữ màu.

VÌ SAO CLAHE thay vì HE toàn cục:
  HE toàn cục dùng MỘT CDF cho cả ảnh, nên vùng gần phẳng (trời, mặt đường) có
  histogram rất hẹp và bị kéo giãn cực mạnh -> NHIỄU BỊ PHÓNG ĐẠI.
  CLAHE chia ảnh thành ô 8x8, equalize từng ô, và CHẶN TRẦN histogram
  (clipLimit=2.0) rồi phân phối lại phần bị cắt -> không khuếch đại nhiễu vô hạn.
"""

from __future__ import annotations

import cv2
import numpy as np

PREPROCESS_MODES = ("none", "he_gray", "he_y", "clahe")

# Tạo sẵn một đối tượng CLAHE và dùng lại — tạo mới mỗi ảnh thì chậm hơn nhiều
# khi phải xử lý 51.839 ảnh.
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def crop_roi(image: np.ndarray, x1: int, y1: int, x2: int, y2: int,
             margin: float = 0.10) -> np.ndarray:
    """Cắt theo ROI bounding box, nới thêm `margin` ở mỗi phía.

    image: HWC uint8 -> HWC uint8

    VÌ SAO NỚI LỀ 10% thay vì cắt sát: cắt sát có nguy cơ mất phần viền biển báo
    (viền đỏ của biển cấm là đặc trưng quan trọng). Nới 10% vẫn loại được phần lớn
    nền trời/đường không mang thông tin phân loại.
    """
    height, width = image.shape[:2]
    box_w, box_h = x2 - x1, y2 - y1
    pad_x, pad_y = int(box_w * margin), int(box_h * margin)

    # np.clip để không ra ngoài biên ảnh
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:        # ROI lỗi trong metadata -> dùng cả ảnh
        return image
    return image[y1:y2, x1:x2]


def apply_preprocess(image: np.ndarray, mode: str) -> np.ndarray:
    """Cân bằng sáng. image: HWC uint8 RGB -> HWC uint8 RGB (luôn 3 kênh).

    Luôn trả 3 kênh, kể cả chế độ he_gray (nhân bản kênh xám ra 3), để mọi
    chế độ dùng được cùng một kiến trúc model — nếu không thì đổi chế độ tiền xử lý
    lại phải đổi số kênh đầu vào của model, và ablation mất tính một-biến-một-lần.
    """
    if mode not in PREPROCESS_MODES:
        raise ValueError(f"mode phải thuộc {PREPROCESS_MODES}, nhận được: {mode!r}")

    if mode == "none":
        return image

    if mode == "he_gray":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)      # HWC -> HW
        gray = cv2.equalizeHist(gray)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)       # HW -> HWC (3 kênh giống nhau)

    if mode == "he_y":
        yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])       # chỉ sửa kênh độ sáng Y
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

    # mode == "clahe"
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = _CLAHE.apply(lab[:, :, 0])               # chỉ sửa kênh độ sáng L
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    """Resize về size x size.

    Chọn phép nội suy theo chiều resize:
      - THU NHỎ  -> INTER_AREA: lấy trung bình vùng, chống aliasing (răng cưa).
      - PHÓNG TO -> INTER_CUBIC: mượt hơn INTER_LINEAR.
    Dùng sai (ví dụ INTER_NEAREST khi thu nhỏ) làm mất chi tiết chữ số trên
    biển giới hạn tốc độ — đúng cái đặc trưng mà model cần nhất.
    """
    height, width = image.shape[:2]
    shrinking = (size < height) or (size < width)
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    return cv2.resize(image, (size, size), interpolation=interp)


def load_and_process(path: str, roi: tuple[int, int, int, int], size: int,
                     mode: str, roi_margin: float = 0.10) -> np.ndarray:
    """Đường ống hoàn chỉnh cho một ảnh: đọc -> cắt ROI -> cân bằng sáng -> resize.

    Trả về HWC uint8 RGB, kích thước size x size.
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)    # OpenCV đọc ra BGR, không phải RGB
    if bgr is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    rgb = crop_roi(rgb, *roi, margin=roi_margin)
    rgb = apply_preprocess(rgb, mode)
    rgb = resize_image(rgb, size)
    return rgb


def compute_mean_std(images: np.ndarray) -> tuple[list[float], list[float]]:
    """Tính mean/std theo từng kênh, chuẩn hoá về [0,1].

    images: (N, H, W, 3) uint8  ->  ([3 mean], [3 std])

    ★ CHỈ ĐƯỢC GỌI TRÊN TẬP TRAIN.
    Tính trên cả bộ dữ liệu là rò rỉ thống kê của val/test vào quá trình huấn luyện:
    model được chuẩn hoá bằng thông tin mà lẽ ra nó chưa được thấy.
    """
    # float64 để tổng của 51.839 ảnh không bị mất chính xác
    flat = images.reshape(-1, 3).astype(np.float64) / 255.0
    return flat.mean(axis=0).tolist(), flat.std(axis=0).tolist()
