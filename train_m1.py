"""
train_m1.py — M1 LeNet: CNN baseline xây từ số 0. MỐC THAM CHIẾU.

    python train_m1.py          # train, lưu checkpoints/m1_lenet_best.pt

KIẾN TRÚC (2 block Conv-Pool + 2 Dense):
    (B,3,48,48)
    Conv(3->32,5x5,pad2) + ReLU + MaxPool2   -> (B,32,24,24)
    Conv(32->64,5x5,pad2) + ReLU + MaxPool2  -> (B,64,12,12)
    Flatten                                   -> (B,9216)
    Linear(9216->256) + ReLU + Dropout        <- 2.359.552 tham số = 97,3% CẢ MODEL
    Linear(256->43)                           -> (B,43)

★ M1 CỐ Ý KHÔNG có BatchNorm / residual / spatial dropout. Nó là MỐC: mọi điểm
accuracy M2 hơn M1 phải quy được về một thành phần cụ thể. "Tinh chỉnh M1 cho
mạnh" là mất ý nghĩa đó.

★ BÀI HỌC CHÍNH: in bảng đếm tham số ra sẽ thấy 97,3% nằm ở MỘT lớp Linear
(9216 x 256). Đó là lý do mọi kiến trúc hiện đại (ResNet, MobileNet, EfficientNet,
và M2) bỏ Flatten+FC, dùng Global Average Pooling (xem train_m2.py).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from common import (NUM_CLASSES, build_train_transform, evaluate, make_loader)

INDEX_CSV = "data/processed/index.csv"
CKPT = "checkpoints/m1_lenet_best.pt"


class M1LeNet(nn.Module):
    """LeNet: 2 block Conv-Pool + 2 Dense. Cố ý KHÔNG BatchNorm/residual/dropout kênh.

    ★ Cấu trúc (features / classifier) phải GIỮ NGUYÊN để state_dict khớp
    checkpoint đã train: features.0, features.3, classifier.1, classifier.4.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, img_size: int = 48,
                 channels: tuple[int, int] = (32, 64), hidden: int = 256,
                 dropout: float = 0.5) -> None:
        super().__init__()
        c1, c2 = channels

        self.features = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=5, padding=2),   # giữ kích thước nhờ pad=2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # -> S/2
            nn.Conv2d(c1, c2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # -> S/4
        )

        # Tự tính số chiều sau flatten (không hard-code 9216 -> đổi img_size không vỡ)
        with torch.no_grad():
            flat_dim = self.features(torch.zeros(1, 3, img_size, img_size)).flatten(1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, hidden),                   # <- lớp "đắt" nhất
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
        self.model_name = "m1_lenet"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def train() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    epochs, warmup, batch_size, lr = 30, 3, 128, 1e-3

    model = M1LeNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    fc1 = model.classifier[1].weight.numel() + model.classifier[1].bias.numel()
    print(f"M1 LeNet · {n_params:,} tham số · lớp Linear đầu chiếm {fc1/n_params*100:.1f}%")

    # Dữ liệu 48x48, chuẩn hoá gtsrb (M1 train từ số 0). Augment geo_photo cho train.
    train_loader = make_loader(INDEX_CSV, "train", 48, batch_size, "gtsrb",
                               transform=build_train_transform(48))
    val_loader = make_loader(INDEX_CSV, "val", 48, batch_size, "gtsrb")

    # CrossEntropy + label smoothing 0.1; Adam; cosine annealing + warmup 3 epoch
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def lr_at(epoch: int) -> float:
        """warmup tuyến tính 3 epoch rồi cosine về 1% lr."""
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
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # chặn gradient đột biến
            optimizer.step()

        m = evaluate(model, val_loader, device)
        flag = ""
        if m["macro_f1"] > best_f1:        # early stopping theo val MACRO-F1, không accuracy
            best_f1 = m["macro_f1"]
            torch.save({"model_state": model.state_dict()}, CKPT)
            flag = " <- lưu best"
        print(f"epoch {epoch+1:2}/{epochs} | val top1 {m['top1']:.4f} "
              f"macroF1 {m['macro_f1']:.4f}{flag}")

    print(f"Xong. best val macro-F1 = {best_f1:.4f} -> {CKPT}")


if __name__ == "__main__":
    train()
