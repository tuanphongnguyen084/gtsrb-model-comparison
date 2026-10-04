"""
M1 — CNN kiểu LeNet, xây từ số 0. MỐC THAM CHIẾU (baseline).

CHỦ: Phong Nguyễn

KIẾN TRÚC (đề bài: 2 block Conv-Pool + 2 lớp dense):
    đầu vào           (B, 3, 48, 48)
    Conv(3->32, 5x5)  (B, 32, 48, 48)    pad=2 nên giữ nguyên kích thước
    ReLU + MaxPool2   (B, 32, 24, 24)
    Conv(32->64, 5x5) (B, 64, 24, 24)
    ReLU + MaxPool2   (B, 64, 12, 12)
    Flatten           (B, 9216)          64 * 12 * 12 = 9216
    Linear(9216->256) (B, 256)           <- 2.359.552 tham số = 97,3% CẢ MODEL
    ReLU + Dropout
    Linear(256->43)   (B, 43)

★ M1 CỐ Ý KHÔNG CÓ BatchNorm, KHÔNG CÓ residual, KHÔNG CÓ spatial dropout.
Nó tồn tại để làm MỐC: mọi điểm accuracy mà M2 hơn M1 đều phải quy được về một
thành phần cụ thể. Nếu "tinh chỉnh M1 cho mạnh lên" thì mất luôn ý nghĩa đó.

★ BÀI HỌC CHÍNH TỪ M1: in bảng đếm tham số ra sẽ thấy 97,3% tham số nằm ở MỘT
lớp fully-connected duy nhất (9216 x 256). Đó chính là lý do mọi kiến trúc
hiện đại (ResNet, MobileNet, EfficientNet, và M2 của nhóm) bỏ Flatten+FC và
dùng Global Average Pooling. Xem docs/LY_THUYET.md mục 2.6.

GHI CHÚ VỀ MỘT LỖI HAY GẶP: nếu hard-code 9216 thì đổi img_size=64 là vỡ ngay
("mat1 and mat2 shapes cannot be multiplied"). Code dưới đây TỰ TÍNH kích thước
sau khi flatten bằng một lần forward thử, nên đổi img_size không vỡ.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gtsrb import NUM_CLASSES


class M1LeNet(nn.Module):
    """CNN kiểu LeNet: 2 block Conv-Pool + 2 lớp Dense.

    Cố ý KHÔNG có BatchNorm / residual / spatial dropout — xem ghi chú đầu file.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, img_size: int = 48,
                 channels: tuple[int, int] = (32, 64), hidden: int = 256,
                 dropout: float = 0.5) -> None:
        super().__init__()
        c1, c2 = channels

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, c1, kernel_size=5, padding=2),   # (B,3,S,S) -> (B,c1,S,S)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # -> (B,c1,S/2,S/2)
            # Block 2
            nn.Conv2d(c1, c2, kernel_size=5, padding=2),  # -> (B,c2,S/2,S/2)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # -> (B,c2,S/4,S/4)
        )

        # Tự tính số chiều sau flatten thay vì hard-code 9216.
        # torch.no_grad() để lần forward thử này không dựng graph autograd.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            flat_dim = self.features(dummy).flatten(1).shape[1]
        self.flat_dim = flat_dim

        self.classifier = nn.Sequential(
            nn.Flatten(),                                  # -> (B, flat_dim)
            nn.Linear(flat_dim, hidden),                   # <- lớp "đắt" nhất
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),                # -> (B, 43)
        )

        # Siêu dữ liệu theo hợp đồng docs/INTERFACE.md mục 2
        self.model_name = "m1_lenet"
        self.expected_img_size = img_size
        self.normalize_mode = "gtsrb"

    @property
    def gradcam_target_layer(self) -> nn.Module:
        """Toàn bộ khối features — output (B, c2, S/4, S/4) là feature map sâu nhất
        còn giữ thông tin VỊ TRÍ (Flatten phía sau mới xoá hết vị trí).

        VÌ SAO trả về self.features chứ không phải self.features[3] (lớp Conv2d):
        output của Conv2d bị nn.ReLU(inplace=True) ngay sau đó SỬA TẠI CHỖ.
        Hook Grad-CAM gắn lên tensor đó sẽ bắt được giá trị đã bị ghi đè.
        Lấy output của cả Sequential thì nó đã qua ReLU + MaxPool và không bị
        lớp nào sửa tại chỗ nữa. Xem thêm ghi chú trong src/gtsrb/explain/gradcam.py.

        Dùng @property chứ không phải self.xxx = module, vì gán thẳng sẽ đăng ký
        module đó lần thứ hai vào _modules -> state_dict bị nhân đôi trọng số.
        """
        return self.features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B,3,S,S) -> (B,43). features trích đặc trưng, classifier ra logit."""
        return self.classifier(self.features(x))
