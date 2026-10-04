"""
transforms — chính sách augmentation.

CHỦ: Huy

4 chính sách, để bước ablation có cái so sánh:

  none            : không augment.                              -> đối chứng
  geo             : xoay +/-15 độ, dịch +/-10%, zoom 0,9-1,1.   -> biến đổi hình học
  geo_photo       : geo + jitter độ sáng/tương phản +/-0,2.      -> MẶC ĐỊNH
  geo_photo_erase : geo_photo + RandomErasing (che ngẫu nhiên).  -> mạnh nhất

★ HAI PHÉP BIẾN ĐỔI BỊ CẤM TUYỆT ĐỐI TRONG DỰ ÁN NÀY ★

1) RandomHorizontalFlip
   Biển báo BẤT ĐỐI XỨNG CÓ NGHĨA. Lật ngang lớp 33 "rẽ phải phía trước" ra
   ĐÚNG HÌNH lớp 34 "rẽ trái phía trước". Các cặp bị ảnh hưởng:
       33 <-> 34,  19 <-> 20,  36 <-> 37,  38 <-> 39
   Flip = tự tạo dữ liệu SAI NHÃN. Model sẽ học rằng rẽ phải và rẽ trái là một.

2) Hue / saturation jitter
   Màu là ĐẶC TRƯNG MANG NGHĨA, không phải nhiễu: đỏ = cấm, xanh = bắt buộc.
   Đổi hue là biến "cấm" thành "bắt buộc".
   Brightness/contrast thì ĐƯỢC, vì đó đúng là biến thiên có thật của điều kiện chụp.

BÀI HỌC KHÁI QUÁT: augmentation phải tôn trọng TÍNH BẤT BIẾN THẬT của bài toán.
Dùng mù quáng "bộ augment tiêu chuẩn cho ảnh" là cách tự làm hỏng nhãn.

LƯU Ý: augmentation CHỈ áp dụng cho loader 'train'. val/test không bao giờ augment,
nếu không thì phép đo không lặp lại được.
"""

from __future__ import annotations

import torch
from torchvision.transforms import v2

AUG_POLICIES = ("none", "geo", "geo_photo", "geo_photo_erase")

# Danh sách đen — tests/test_transforms.py kiểm rằng không có cái nào xuất hiện
FORBIDDEN_TRANSFORMS = ("RandomHorizontalFlip", "RandomVerticalFlip")


def build_transform(policy: str = "geo_photo", img_size: int = 48,
                    rotation_deg: float = 15.0, translate: float = 0.10,
                    zoom: tuple[float, float] = (0.9, 1.1),
                    jitter: float = 0.20) -> v2.Transform | None:
    """Trả về transform áp lên tensor uint8 CHW, ra tensor uint8 CHW (cùng shape).

    Trả None nếu policy='none' (để dataset biết bỏ qua hẳn, không gọi vô ích).
    """
    if policy not in AUG_POLICIES:
        raise ValueError(f"policy phải thuộc {AUG_POLICIES}, nhận được: {policy!r}")

    if policy == "none":
        return None

    steps: list[v2.Transform] = []

    # --- Biến đổi hình học ---
    # RandomAffine làm xoay + dịch + zoom trong MỘT phép nội suy duy nhất.
    # Làm 3 phép riêng lẻ sẽ nội suy 3 lần -> ảnh mờ đi đáng kể ở 48x48.
    steps.append(v2.RandomAffine(
        degrees=rotation_deg,
        translate=(translate, translate),
        scale=zoom,
        interpolation=v2.InterpolationMode.BILINEAR,
        fill=0,
    ))

    # --- Biến đổi quang học ---
    if policy in ("geo_photo", "geo_photo_erase"):
        # CHỈ brightness và contrast. KHÔNG hue, KHÔNG saturation (xem ghi chú đầu file).
        steps.append(v2.ColorJitter(brightness=jitter, contrast=jitter))

    # --- Che ngẫu nhiên ---
    if policy == "geo_photo_erase":
        # Mô phỏng lá cây / sticker che một phần biển báo.
        # p=0.25 để phần lớn ảnh vẫn nguyên vẹn; che quá nhiều thì model học kém đi.
        steps.append(v2.RandomErasing(p=0.25, scale=(0.02, 0.15), value=0))

    return v2.Compose(steps)


def build_normalize(mean: tuple[float, ...], std: tuple[float, ...]) -> v2.Transform:
    """Đổi uint8 [0,255] -> float32 đã chuẩn hoá.

    scale=True trong ToDtype tự chia 255. Thứ tự BẮT BUỘC là
    uint8 -> float [0,1] -> normalize; làm ngược lại thì mean/std sai thang đo.
    """
    return v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(mean), std=list(std)),
    ])
