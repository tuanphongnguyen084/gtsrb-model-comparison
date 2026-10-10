"""
train_m3.py — M3 Transfer learning từ 3 backbone pretrained ImageNet.

    python train_m3.py --backbone resnet18       # hoặc mobilenetv2 | effnetb0

Lưu checkpoints/m3_<backbone>_best.pt.

GỘP 3 BACKBONE VÀO 1 FILE vì phần model chỉ khác ~3 dòng (nạp backbone + thay
lớp cuối), còn Dataset 224x224, chuẩn hoá ImageNet và quy trình train 2 pha thì
giống hệt. Đặt chung giúp thấy rõ tương phản 3 triết lý: ResNet (conv dày đặc),
MobileNetV2 (depthwise separable), EfficientNet-B0 (compound scaling + SE).

★ BỐN ĐIỂM CỐT LÕI:
1) Transfer learning hiệu quả vì CNN học theo tầng: tầng đầu (cạnh/màu) PHỔ QUÁT
   cho mọi ảnh, học từ 1,28 triệu ảnh ImageNet — thứ 39 nghìn ảnh GTSRB không học
   nổi. Chỉ tầng cuối (ngữ nghĩa) phải thay.
2) Upsample 48->224 vì backbone downsample 32 lần: 224/32 = 7x7 là đúng cấu hình
   pretrain; 48/32 làm tròn 2x2 thì tầng cuối vô dụng. Upsample để khớp SCALE,
   KHÔNG thêm thông tin.
3) Discriminative learning rate: tầng đầu 1e-5 (đã tốt, nhích nhẹ), head 1e-3
   (ngẫu nhiên, học từ đầu). Cho cả mạng 1e-3 -> gradient lớn phá tri thức
   pretrained (catastrophic forgetting).
4) Train 2 pha: Pha 1 freeze backbone chỉ train head (head ngẫu nhiên -> loss
   ~ln43 ~3,76 -> gradient lớn, nếu backbone mở sẽ bị phá ngay); Pha 2 mở băng.
5) Chuẩn hoá bằng thống kê ImageNet (không phải GTSRB) vì BatchNorm pretrained
   tích luỹ running stats trên phân phối đó.
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn
import torchvision

from common import NUM_CLASSES, evaluate, make_loader

INDEX_CSV = "data/processed/index.csv"

BACKBONES = {
    "resnet18": ("resnet18", "ResNet18_Weights"),
    "mobilenetv2": ("mobilenet_v2", "MobileNet_V2_Weights"),
    "effnetb0": ("efficientnet_b0", "EfficientNet_B0_Weights"),
}


class M3Transfer(nn.Module):
    """Bọc backbone pretrained, thay head thành 43 lớp. Cấu trúc self.net giữ nguyên."""

    def __init__(self, backbone: str = "resnet18", num_classes: int = NUM_CLASSES,
                 pretrained: bool = True, head_dropout: float = 0.2) -> None:
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"backbone phải thuộc {sorted(BACKBONES)}, nhận: {backbone!r}")
        self.backbone_name = backbone
        builder_name, weights_name = BACKBONES[backbone]

        builder = getattr(torchvision.models, builder_name)
        weights = getattr(torchvision.models, weights_name).IMAGENET1K_V1 if pretrained else None
        self.net = builder(weights=weights)

        # Thay head: 1000 lớp ImageNet -> 43 lớp biển báo
        if backbone == "resnet18":
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)   # 512 -> 43
        else:
            in_f = self.net.classifier[-1].in_features                       # 1280
            self.net.classifier = nn.Sequential(nn.Dropout(head_dropout),
                                                nn.Linear(in_f, num_classes))
        self.model_name = f"m3_{backbone}"

    # --- freeze/unfreeze + discriminative lr ---
    def head_parameters(self):
        return (self.net.fc if self.backbone_name == "resnet18"
                else self.net.classifier).parameters()

    def set_backbone_frozen(self, frozen: bool) -> None:
        head_ids = {id(p) for p in self.head_parameters()}
        for p in self.net.parameters():
            if id(p) not in head_ids:
                p.requires_grad = not frozen

    def param_groups(self, lr_early=1e-5, lr_middle=1e-4, lr_head=1e-3) -> list[dict]:
        """Chia tham số thành 3 nhóm lr theo thứ tự kiến trúc (nửa đầu/nửa sau/head)."""
        head_params = list(self.head_parameters())
        head_ids = {id(p) for p in head_params}
        if self.backbone_name == "resnet18":
            blocks = [self.net.conv1, self.net.bn1, self.net.layer1,
                      self.net.layer2, self.net.layer3, self.net.layer4]
        else:
            blocks = list(self.net.features)
        split = len(blocks) // 2
        early, middle = [], []
        for pos, block in enumerate(blocks):
            tgt = early if pos < split else middle
            tgt += [p for p in block.parameters() if id(p) not in head_ids]
        groups = [{"params": early, "lr": lr_early}, {"params": middle, "lr": lr_middle},
                  {"params": head_params, "lr": lr_head}]
        return [g for g in groups if g["params"]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train(backbone: str) -> None:
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64                    # 224x224 tốn bộ nhớ ~22 lần 48x48
    phase1_epochs, phase2_epochs = 3, 12

    model = M3Transfer(backbone).to(device)
    print(f"M3 {backbone} · {sum(p.numel() for p in model.parameters()):,} tham số")

    # Dữ liệu 224x224 (nội suy từ cache 48), chuẩn hoá ImageNet. Không augment để
    # giữ đơn giản (bản train này không nhằm tái tạo checkpoint, chỉ minh hoạ luồng).
    train_loader = make_loader(INDEX_CSV, "train", 224, batch_size, "imagenet")
    val_loader = make_loader(INDEX_CSV, "val", 224, batch_size, "imagenet")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    ckpt = f"checkpoints/m3_{backbone}_best.pt"
    best_f1 = 0.0

    def run_epochs(optimizer, n_epochs, tag, epoch_base, total):
        nonlocal best_f1
        for e in range(n_epochs):
            # cosine annealing trong pha (giữ lr nền của từng nhóm)
            scale = 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * e / max(1, n_epochs)))
            base_lrs = [g.get("initial_lr", g["lr"]) for g in optimizer.param_groups]
            for g, base in zip(optimizer.param_groups, base_lrs):
                g.setdefault("initial_lr", base)
                g["lr"] = base * scale
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
                torch.save({"model_state": model.state_dict()}, ckpt)
                flag = " <- lưu best"
            print(f"[{tag}] epoch {epoch_base+e+1:2}/{total} | val top1 {m['top1']:.4f} "
                  f"macroF1 {m['macro_f1']:.4f}{flag}")

    total = phase1_epochs + phase2_epochs
    # Pha 1: freeze backbone, chỉ train head với AdamW lr 1e-3
    model.set_backbone_frozen(True)
    opt1 = torch.optim.AdamW(model.head_parameters(), lr=1e-3, weight_decay=1e-4)
    run_epochs(opt1, phase1_epochs, "freeze", 0, total)

    # Pha 2: mở băng toàn bộ, discriminative lr (early 1e-5 / middle 1e-4 / head 1e-3)
    model.set_backbone_frozen(False)
    opt2 = torch.optim.AdamW(model.param_groups(), weight_decay=1e-4)
    run_epochs(opt2, phase2_epochs, "finetune", phase1_epochs, total)

    print(f"Xong. best val macro-F1 = {best_f1:.4f} -> {ckpt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default="resnet18", choices=sorted(BACKBONES))
    train(ap.parse_args().backbone)
