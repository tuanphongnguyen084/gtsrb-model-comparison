"""
M2 — CNN kiểu VGG sâu, xây từ số 0, có BatchNorm + residual + spatial dropout + GAP.

CHỦ: Hoàng

KIẾN TRÚC (4 stage, đầu vào 48x48):
    Stem      Conv(3->32,3x3)+BN+ReLU            (B, 32, 48, 48)
    Stage 1   ResBlock(32->32)  -> Pool -> SD    (B, 32, 24, 24)   spatial dropout p=0,1
    Stage 2   ResBlock(32->64)  -> Pool -> SD    (B, 64, 12, 12)                  p=0,2
    Stage 3   ResBlock(64->128) -> Pool -> SD    (B,128,  6,  6)                  p=0,3
    Stage 4   ResBlock(128->256)-> Pool -> SD    (B,256,  3,  3)                  p=0,3
    Head      GAP -> Dropout(0,5) -> FC(256->43) (B, 43)

★ NĂM LỰA CHỌN THIẾT KẾ — PHẢI GIẢI THÍCH ĐƯỢC TỪNG CÁI ★

1) HAI conv 3x3 thay MỘT conv 5x5
   Cùng receptive field 5x5 (công thức RF = 2n+1 cho n lớp 3x3 stride 1), nhưng:
     tham số 2*(3*3*C^2) = 18C^2  so với  5*5*C^2 = 25C^2   -> giảm 28%
     và có HAI tầng phi tuyến ReLU thay vì một -> hàm biểu diễn phong phú hơn.
   Đây là luận điểm trung tâm của VGG (Simonyan & Zisserman 2014).
   Stack 3x3 không phải "cách rẻ để làm 5x5", nó là một hàm MẠNH HƠN.

2) BatchNorm, đặt giữa Conv và ReLU
   Chuẩn hoá mỗi kênh về mean 0 / var 1 theo mini-batch rồi scale-shift bằng 2
   tham số HỌC ĐƯỢC gamma, beta (cần gamma,beta vì nếu luôn ép mean 0 var 1 thì
   mạng mất khả năng dùng vùng phi tuyến của ReLU).
   Tác dụng thật: LÀM MƯỢT BỀ MẶT LOSS -> dùng được learning rate lớn hơn.
   (Bài gốc Ioffe 2015 giải thích bằng "internal covariate shift", nhưng
    Santurkar et al. 2018 đã phản biện điểm đó. Nói được chỗ này là điểm cộng.)

3) Residual connection  y = F(x) + x
   Giải quyết DEGRADATION, không phải overfitting: He et al. 2015 thấy mạng 56 lớp
   cho TRAIN error cao hơn mạng 20 lớp -> không thể là overfit, mà là không tối
   ưu hoá được. Nếu identity là tối ưu thì chỉ cần đẩy F(x)->0, dễ hơn nhiều so
   với học identity bằng cả stack conv.
   Về gradient: dy/dx = dF/dx + 1. Số hạng "+1" là ĐƯỜNG CAO TỐC cho gradient
   chảy ngược -> chống vanishing gradient.
   Khi số kênh đổi (32->64) thì shape không khớp -> dùng conv 1x1 để chiếu
   (projection shortcut, "option B" trong bài ResNet).

4) SpatialDropout (nn.Dropout2d) thay Dropout thường
   Dropout thường bỏ từng PHẦN TỬ độc lập. Trên feature map, pixel lân cận tương
   quan RẤT CAO (cùng nhìn gần như cùng một vùng ảnh), nên bỏ pixel (i,j) thì
   thông tin vẫn còn ở (i,j+1) -> regularize gần như vô hiệu.
   Dropout2d bỏ TOÀN BỘ MỘT KÊNH = một feature detector -> buộc mạng không được
   dựa vào một detector duy nhất, phải học đặc trưng dư thừa và độc lập.
   p tăng dần theo độ sâu vì tầng sâu nhiều kênh hơn và dễ overfit hơn.

5) Global Average Pooling thay Flatten + FC
   M1 có 9216x256 = 2.359.552 tham số ở MỘT lớp (97,3% cả model, số ĐO THẬT).
   GAP lấy trung bình không gian
   mỗi kênh: (B,256,3,3) -> (B,256) -> FC(256->43) chỉ còn 11.051 tham số.
   Giảm ~213 lần. Thêm: bất biến với dịch chuyển, và là một REGULARIZER CẤU TRÚC
   (buộc mỗi kênh mang nghĩa toàn cục chứ không nhớ vị trí cụ thể).
   Ý tưởng từ Network-in-Network (Lin et al. 2013).

`width_mult` và `n_stages` lấy từ config để bước ablation model scaling dùng được.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gtsrb import NUM_CLASSES


class ResidualBlock(nn.Module):
    """Hai conv 3x3 + BN, cộng với nhánh skip.

        x ──┬─> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN ──> (+) -> ReLU -> out
            └────────────── skip (identity hoặc Conv1x1+BN) ──┘

    Lưu ý thứ tự: ReLU CUỐI nằm SAU phép cộng, không phải trước.
    Cộng trước rồi ReLU là đúng theo bài ResNet gốc; làm ngược lại thì nhánh skip
    bị chặn ở 0 và mất tác dụng "đường cao tốc gradient".
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        # bias=False ở conv vì ngay sau là BatchNorm — BN có beta làm việc của bias,
        # để thêm bias là thừa tham số và không có tác dụng gì.

        if in_ch == out_ch:
            self.skip: nn.Module = nn.Identity()
        else:
            # Projection shortcut: conv 1x1 để khớp số kênh
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """y = ReLU(BN(Conv(ReLU(BN(Conv(x))))) + skip(x)). Cộng TRƯỚC, ReLU SAU."""
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)        # <- cộng TRƯỚC, ReLU SAU


class M2VggRes(nn.Module):
    """VGG-like 4 stage, mỗi stage một ResidualBlock + MaxPool + SpatialDropout."""

    def __init__(self, num_classes: int = NUM_CLASSES, img_size: int = 48,
                 base_channels: int = 32, n_stages: int = 4,
                 width_mult: float = 1.0,
                 spatial_dropout: tuple[float, ...] = (0.1, 0.2, 0.3, 0.3),
                 head_dropout: float = 0.5,
                 use_bn: bool = True, use_residual: bool = True,
                 use_spatial_dropout: bool = True) -> None:
        super().__init__()

        # Ba cờ use_* để làm ABLATION NỘI BỘ: bỏ BN / bỏ residual / bỏ spatial
        # dropout, mỗi cái một dòng trong bảng, chứng minh từng thành phần đáng
        # giá bao nhiêu. Giữ nguyên mọi thứ khác -> đúng nguyên tắc một-biến-một-lần.
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_spatial_dropout = use_spatial_dropout

        stem_channels = self._channels_at(0, base_channels, width_mult)
        self.stem = self._build_stem(stem_channels, use_bn)
        self.stages, self.out_channels = self._build_stages(
            stem_channels, base_channels, n_stages, width_mult, spatial_dropout)
        self.head = self._build_head(self.out_channels, num_classes, head_dropout)

        self._init_weights()

        # Siêu dữ liệu theo hợp đồng docs/INTERFACE.md mục 2
        self.model_name = "m2_vggres"
        self.expected_img_size = img_size
        self.normalize_mode = "gtsrb"

    # ------------------------------------------------------------------
    # Dựng từng phần
    # ------------------------------------------------------------------

    @staticmethod
    def _channels_at(stage: int, base_channels: int, width_mult: float) -> int:
        """Số kênh của stage: 32 -> 64 -> 128 -> 256, nhân với width_mult.

        max(8, ...) để width_mult=0.25 không tạo ra stage 0 kênh.
        """
        return max(8, int(base_channels * (2 ** stage) * width_mult))

    @staticmethod
    def _build_stem(out_channels: int, use_bn: bool) -> nn.Sequential:
        """Conv 3x3 đầu tiên, đưa 3 kênh RGB lên số kênh làm việc.

        bias=not use_bn: khi có BatchNorm ngay sau thì bias là THỪA, vì BN có
        tham số beta làm đúng việc đó.
        """
        layers: list[nn.Module] = [nn.Conv2d(3, out_channels, 3, padding=1,
                                             bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _build_stages(self, in_channels: int, base_channels: int, n_stages: int,
                      width_mult: float,
                      spatial_dropout: tuple[float, ...]) -> tuple[nn.Sequential, int]:
        """Dựng n_stages, mỗi stage: [2 conv 3x3 + skip] -> MaxPool -> SpatialDropout.

        Kích thước không gian giảm một nửa sau mỗi stage:
        48 -> 24 -> 12 -> 6 -> 3 (với n_stages=4, img_size=48).
        """
        stages: list[nn.Module] = []
        for stage in range(n_stages):
            out_channels = self._channels_at(stage, base_channels, width_mult)
            block = self._build_block(in_channels, out_channels)

            block.append(nn.MaxPool2d(2))        # giảm một nửa kích thước không gian

            if self.use_spatial_dropout:
                # p tăng dần theo độ sâu: tầng sâu nhiều kênh hơn và dễ overfit hơn
                p = spatial_dropout[min(stage, len(spatial_dropout) - 1)]
                block.append(nn.Dropout2d(p))    # <- bỏ CẢ KÊNH, không phải từng pixel

            stages.append(nn.Sequential(*block))
            in_channels = out_channels

        return nn.Sequential(*stages), in_channels

    def _build_block(self, in_channels: int, out_channels: int) -> list[nn.Module]:
        """Phần tích chập của một stage: có residual hoặc không (cho ablation)."""
        if self.use_residual:
            return [ResidualBlock(in_channels, out_channels)]

        # Biến thể ablation: CÙNG 2 conv 3x3 nhưng KHÔNG cộng skip.
        # Giữ nguyên số conv để chênh lệch chỉ đến từ việc có/không có skip.
        return [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=not self.use_bn),
            nn.BatchNorm2d(out_channels) if self.use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=not self.use_bn),
            nn.BatchNorm2d(out_channels) if self.use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        ]

    @staticmethod
    def _build_head(in_channels: int, num_classes: int,
                    dropout: float) -> nn.Sequential:
        """Global Average Pooling -> Dropout -> FC.

        ★ Đây là chỗ M2 khác M1 nhiều nhất về số tham số.
        M1: Flatten(9216) -> FC(256)        = 2.359.552 tham số (97,3% cả model)
        M2: GAP -> FC(256 -> 43)            =    11.051 tham số
        Giảm ~213 lần. Thêm: bất biến với dịch chuyển, và là một REGULARIZER CẤU TRÚC
        (buộc mỗi kênh mang nghĩa toàn cục chứ không nhớ vị trí cụ thể).
        """
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),            # (B,C,H,W) -> (B,C,1,1)  <- GAP
            nn.Flatten(),                        # -> (B,C)
            nn.Dropout(dropout),
            nn.Linear(in_channels, num_classes), # -> (B,43)
        )

    def _init_weights(self) -> None:
        """Khởi tạo He/Kaiming cho conv — đúng cho mạng dùng ReLU.

        Khởi tạo Xavier (cho tanh/sigmoid) làm phương sai activation co lại dần
        qua các tầng ReLU vì ReLU chặn một nửa giá trị.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @property
    def gradcam_target_layer(self) -> nn.Module:
        """Stage cuối — feature map có ngữ nghĩa cao nhất mà còn giữ vị trí.

        LƯU Ý GIỚI HẠN: ở img_size=48, stage 4 cho feature map 3x3. Heatmap
        Grad-CAM upsample từ 3x3 lên 48x48 nên rất thô — mỗi "ô" là 16x16 pixel.
        Muốn nhìn rõ hơn thì lấy self.stages[2] (6x6), nhưng ngữ nghĩa thấp hơn.
        Phải nêu giới hạn này khi trình bày.
        """
        return self.stages[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B,3,48,48) -> stem -> 4 stage (48->24->12->6->3) -> GAP+FC -> (B,43)."""
        return self.head(self.stages(self.stem(x)))
