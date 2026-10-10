"""
train_m2.py — M2 VGG-Res: CNN sâu tự xây, có BatchNorm + residual + spatial
dropout + Global Average Pooling.

    python train_m2.py          # train, lưu checkpoints/m2_vggres_best.pt

KIẾN TRÚC (4 stage, đầu vào 48x48):
    Stem      Conv(3->32,3x3)+BN+ReLU
    Stage 1   ResBlock(32->32)  -> MaxPool -> Dropout2d(0.1)   (B,32,24,24)
    Stage 2   ResBlock(32->64)  -> MaxPool -> Dropout2d(0.2)   (B,64,12,12)
    Stage 3   ResBlock(64->128) -> MaxPool -> Dropout2d(0.3)   (B,128,6,6)
    Stage 4   ResBlock(128->256)-> MaxPool -> Dropout2d(0.3)   (B,256,3,3)
    Head      GAP -> Dropout(0.5) -> Linear(256->43)

★ NĂM LỰA CHỌN THIẾT KẾ:
1) Hai conv 3x3 thay một conv 5x5: cùng receptive field nhưng ít tham số hơn 28%
   và có 2 tầng phi tuyến -> biểu diễn mạnh hơn (luận điểm VGG).
2) BatchNorm giữa Conv và ReLU: làm mượt bề mặt loss -> dùng được lr lớn hơn.
3) Residual y = F(x)+x: chống degradation; "+1" trong đạo hàm là đường cao tốc
   cho gradient -> chống vanishing. Đổi số kênh thì skip dùng Conv1x1 chiếu.
4) SpatialDropout (Dropout2d) bỏ CẢ KÊNH, không bỏ pixel lẻ — vì pixel lân cận
   trên feature map tương quan cao, bỏ pixel thì regularize gần như vô hiệu.
5) Global Average Pooling thay Flatten+FC: head chỉ còn 11.051 tham số, giảm
   214 lần so với M1. Thêm: bất biến dịch chuyển, là regularizer cấu trúc.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from common import (NUM_CLASSES, build_train_transform, evaluate, make_loader)

INDEX_CSV = "data/processed/index.csv"
CKPT = "checkpoints/m2_vggres_best.pt"


class ResidualBlock(nn.Module):
    """2 conv 3x3 + BN, cộng nhánh skip. Cộng TRƯỚC, ReLU SAU (theo ResNet gốc)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        # bias=False vì ngay sau là BatchNorm (beta của BN làm việc của bias)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        if in_ch == out_ch:
            self.skip: nn.Module = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)          # cộng TRƯỚC, ReLU SAU


class M2VggRes(nn.Module):
    """VGG-like 4 stage. Cấu trúc (stem/stages/head) phải GIỮ NGUYÊN để khớp checkpoint."""

    def __init__(self, num_classes: int = NUM_CLASSES, base_channels: int = 32,
                 n_stages: int = 4, spatial_dropout: tuple[float, ...] = (0.1, 0.2, 0.3, 0.3),
                 head_dropout: float = 0.5) -> None:
        super().__init__()

        # Stem: Conv 3x3 đưa 3 kênh RGB lên base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # 4 stage: [ResBlock] -> MaxPool (giảm nửa kích thước) -> Dropout2d
        stages = []
        in_ch = base_channels
        for stage in range(n_stages):
            out_ch = base_channels * (2 ** stage)          # 32, 64, 128, 256
            p = spatial_dropout[min(stage, len(spatial_dropout) - 1)]
            stages.append(nn.Sequential(
                ResidualBlock(in_ch, out_ch),
                nn.MaxPool2d(2),
                nn.Dropout2d(p),                            # p tăng dần theo độ sâu
            ))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)

        # Head: Global Average Pooling -> Dropout -> FC
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                        # (B,C,H,W) -> (B,C,1,1)
            nn.Flatten(),
            nn.Dropout(head_dropout),
            nn.Linear(in_ch, num_classes),                  # chỉ 256*43+43 = 11.051 tham số
        )

        self._init_weights()
        self.model_name = "m2_vggres"

    def _init_weights(self) -> None:
        """He/Kaiming cho conv (đúng cho ReLU). Xavier sẽ làm phương sai co lại qua ReLU."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stages(self.stem(x)))


def train() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    epochs, warmup, batch_size, lr = 40, 3, 128, 1e-3

    model = M2VggRes().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    head = sum(p.numel() for p in model.head.parameters())
    print(f"M2 VGG-Res · {n_params:,} tham số · head (GAP+FC) chỉ {head:,} = {head/n_params*100:.1f}%")

    train_loader = make_loader(INDEX_CSV, "train", 48, batch_size, "gtsrb",
                               transform=build_train_transform(48))
    val_loader = make_loader(INDEX_CSV, "val", 48, batch_size, "gtsrb")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def lr_at(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

    best_f1 = 0.0
    for epoch in range(epochs):
        for g in optimizer.param_groups:
            g["lr"] = lr * lr_at(epoch)
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        m = evaluate(model, val_loader, device)
        flag = ""
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            torch.save({"model_state": model.state_dict()}, CKPT)
            flag = " <- lưu best"
        print(f"epoch {epoch+1:2}/{epochs} | val top1 {m['top1']:.4f} "
              f"macroF1 {m['macro_f1']:.4f}{flag}")

    print(f"Xong. best val macro-F1 = {best_f1:.4f} -> {CKPT}")


if __name__ == "__main__":
    train()
