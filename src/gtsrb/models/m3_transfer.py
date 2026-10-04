"""
M3 — Transfer learning từ 3 backbone pretrained ImageNet.

CHỦ: Phong Trần

BA BACKBONE — ba triết lý kiến trúc khác nhau, cố ý chọn vậy để so sánh accuracy/cost:

  ResNet18          11,7M tham số, ~1,82 GFLOPs @224
      residual + basic block; conv dày đặc, đơn giản, nhanh trên GPU.
  MobileNetV2        3,5M tham số, ~0,30 GFLOPs @224
      depthwise separable conv (tách conv không gian khỏi conv kênh)
      + inverted residual (nở ra rồi bóp lại)
      + linear bottleneck (bỏ ReLU ở lớp hẹp để không mất thông tin).
  EfficientNet-B0    5,3M tham số, ~0,39 GFLOPs @224
      compound scaling (tăng depth/width/resolution cùng lúc theo một hệ số)
      + MBConv + squeeze-and-excitation (attention trên trục kênh).

★ BỐN ĐIỂM PHẢI GIẢI THÍCH ĐƯỢC ★

1) VÌ SAO TRANSFER LEARNING HIỆU QUẢ khi ImageNet (mèo, xe) khác xa biển báo?
   CNN học đặc trưng THEO TẦNG:
     tầng đầu  -> cạnh, góc, đốm màu, gradient   : PHỔ QUÁT cho mọi ảnh tự nhiên
     tầng giữa -> texture, hình khối, bộ phận     : phần lớn dùng lại được
     tầng cuối -> ngữ nghĩa "mèo", "xe tải"       : phải thay hoàn toàn
   ImageNet có 1,28 TRIỆU ảnh; GTSRB chỉ 39 nghìn. Tri thức tầng đầu học từ
   1,28 triệu ảnh là thứ 39 nghìn ảnh KHÔNG HỌC NỔI. Đó là nội dung được chuyển giao.

2) VÌ SAO PHẢI UPSAMPLE LÊN 224x224?
   Ba backbone này downsample tổng cộng 32 LẦN.
       input  48 -> feature map cuối  48/32 = 1,5  -> làm tròn 2x2  : MẤT HẾT vị trí
       input 112 -> 4x4                                             : tạm được
       input 224 -> 7x7                                             : ĐÚNG cấu hình pretrain
   Lưu ý trung thực: nội suy 48->224 KHÔNG THÊM THÔNG TIN. Upsample để khớp
   SCALE và STRIDE của mạng pretrained, không phải để có thêm chi tiết.

3) DISCRIMINATIVE LEARNING RATE
   Mỗi nhóm tầng một LR riêng, tăng dần từ đầu đến cuối mạng:
       tầng đầu  1e-5  : đã học cạnh/màu RẤT TỐT, chỉ nhích nhẹ
       tầng giữa 1e-4  : điều chỉnh vừa
       head      1e-3  : khởi tạo ngẫu nhiên, phải học từ đầu
   Nếu cho toàn mạng 1e-3: gradient lớn ghi đè tri thức pretrained đắt giá bằng
   tín hiệu từ 39 nghìn ảnh -> CATASTROPHIC FORGETTING.
   Nếu cho toàn mạng 1e-5: head học quá chậm, không hội tụ trong ngân sách epoch.
   Hai cực đều tệ.

4) HUẤN LUYỆN 2 PHA: FREEZE rồi UNFREEZE
   Pha 1 (3 epoch): đóng băng backbone, chỉ train head.
       Head khởi tạo ngẫu nhiên nên loss ban đầu rất lớn (~ln 43 ~ 3,76)
       -> gradient rất lớn. Nếu backbone đang mở, gradient đó chảy ngược và PHÁ
       filter pretrained ngay trong vài trăm bước đầu, trước khi LR schedule kịp làm gì.
   Pha 2: mở băng toàn bộ với discriminative LR.

5) CHUẨN HOÁ BẰNG THỐNG KÊ IMAGENET, không phải của GTSRB
   (a) filter tầng đầu được tối ưu cho đúng phân phối đầu vào đó;
   (b) running_mean/running_var trong các lớp BatchNorm pretrained được tích luỹ
       TRÊN phân phối đó — đổi đầu vào là làm chúng sai.
   Vì vậy normalize_mode = "imagenet". M1/M2 train từ số 0 thì dùng "gtsrb".
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

from gtsrb import NUM_CLASSES

# Tên backbone -> (hàm dựng torchvision, tên enum weights)
BACKBONES = {
    "resnet18": ("resnet18", "ResNet18_Weights"),
    "mobilenetv2": ("mobilenet_v2", "MobileNet_V2_Weights"),
    "effnetb0": ("efficientnet_b0", "EfficientNet_B0_Weights"),
}


class M3Transfer(nn.Module):
    """Bọc một backbone pretrained, thay head thành 43 lớp."""

    def __init__(self, backbone: str = "resnet18", num_classes: int = NUM_CLASSES,
                 img_size: int = 224, pretrained: bool = True,
                 head_dropout: float = 0.2) -> None:
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(
                f"backbone phải thuộc {sorted(BACKBONES)}, nhận được: {backbone!r}"
            )
        self.backbone_name = backbone
        builder_name, weights_enum_name = BACKBONES[backbone]

        builder = getattr(torchvision.models, builder_name)
        weights = None
        if pretrained:
            weights_enum = getattr(torchvision.models, weights_enum_name)
            weights = weights_enum.IMAGENET1K_V1
        self.net = builder(weights=weights)

        # ---- Thay head: 1000 lớp ImageNet -> 43 lớp biển báo ----
        if backbone == "resnet18":
            in_features = self.net.fc.in_features              # 512
            self.net.fc = nn.Linear(in_features, num_classes)
        else:
            # MobileNetV2: classifier = Sequential(Dropout, Linear(1280, 1000))
            # EfficientNet-B0: classifier = Sequential(Dropout, Linear(1280, 1000))
            in_features = self.net.classifier[-1].in_features  # 1280
            self.net.classifier = nn.Sequential(
                nn.Dropout(head_dropout),
                nn.Linear(in_features, num_classes),
            )
        self.in_features = in_features

        self.model_name = f"m3_{backbone}"
        self.expected_img_size = img_size
        self.normalize_mode = "imagenet"     # <- KHÔNG phải "gtsrb", xem ghi chú 5

    # ------------------------------------------------------------------
    # Grad-CAM
    # ------------------------------------------------------------------
    @property
    def gradcam_target_layer(self) -> nn.Module:
        """Khối conv cuối cùng. Ở input 224 cho feature map 7x7 -> heatmap nét
        hơn hẳn M2 (chỉ 3x3 ở input 48). Lợi thế này là do ĐỘ PHÂN GIẢI,
        không phải do model 'hiểu' hơn — phải nói rõ khi trình bày."""
        if self.backbone_name == "resnet18":
            return self.net.layer4
        return self.net.features[-1]

    # ------------------------------------------------------------------
    # Freeze / unfreeze
    # ------------------------------------------------------------------
    def head_parameters(self):
        """Tham số của head (lớp phân loại mới, khởi tạo ngẫu nhiên)."""
        if self.backbone_name == "resnet18":
            return self.net.fc.parameters()
        return self.net.classifier.parameters()

    def set_backbone_frozen(self, frozen: bool) -> None:
        """Bật/tắt requires_grad cho toàn bộ backbone (giữ nguyên head).

        Dùng cho pha 1 (frozen=True) rồi pha 2 (frozen=False).
        """
        head_ids = {id(p) for p in self.head_parameters()}
        for param in self.net.parameters():
            if id(param) not in head_ids:
                param.requires_grad = not frozen

    # ------------------------------------------------------------------
    # Discriminative learning rate
    # ------------------------------------------------------------------
    def param_groups(self, lr_early: float = 1e-5, lr_middle: float = 1e-4,
                     lr_head: float = 1e-3) -> list[dict]:
        """Chia tham số thành 3 nhóm, mỗi nhóm một learning rate.

        Trả về list[dict] truyền thẳng vào torch.optim.AdamW(...).

        Cách chia: lấy danh sách các khối con theo thứ tự trong backbone, nửa đầu
        là "early", nửa sau là "middle", head riêng. Chia theo thứ tự kiến trúc
        (không theo tên cứng) nên dùng được cho cả 3 backbone.
        """
        head_params = list(self.head_parameters())
        head_ids = {id(p) for p in head_params}

        # Các khối con theo thứ tự forward
        if self.backbone_name == "resnet18":
            blocks = [self.net.conv1, self.net.bn1,
                      self.net.layer1, self.net.layer2,
                      self.net.layer3, self.net.layer4]
        else:
            blocks = list(self.net.features)

        split = len(blocks) // 2
        early_params, middle_params = [], []
        for position, block in enumerate(blocks):
            target = early_params if position < split else middle_params
            target += [p for p in block.parameters() if id(p) not in head_ids]

        groups = [
            {"params": early_params, "lr": lr_early, "name": "early"},
            {"params": middle_params, "lr": lr_middle, "name": "middle"},
            {"params": head_params, "lr": lr_head, "name": "head"},
        ]
        # Bỏ nhóm rỗng để optimizer không báo lỗi
        return [g for g in groups if len(g["params"]) > 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B,3,224,224) -> backbone pretrained -> head 43 lớp -> (B,43)."""
        return self.net(x)
