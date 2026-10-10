"""
evaluate.py — nạp 5 checkpoint đã train, in bảng so sánh trên tập test chính thức
(12.630 ảnh), KHÔNG train lại.

    python evaluate.py

In: số tham số, Top-1, Macro-F1 (chỉ số chính), latency CPU; và kiểm định
McNemar giữa các cặp model để biết chênh lệch có Ý NGHĨA THỐNG KÊ không.

Macro-F1 là chỉ số CHÍNH (dữ liệu mất cân bằng 10,7:1). Top-5 bị bỏ vì trên 43
lớp nó bão hoà ~99,9%, không phân biệt được gì.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from common import compute_metrics, mcnemar, make_loader, predict
from train_m1 import M1LeNet
from train_m2 import M2VggRes
from train_m3 import M3Transfer

INDEX_CSV = "data/processed/index.csv"

# (nhãn, hàm dựng model, img_size, normalize, checkpoint)
MODELS = [
    ("m1_lenet",       lambda: M1LeNet(),                              48,  "gtsrb",    "checkpoints/m1_lenet_best.pt"),
    ("m2_vggres",      lambda: M2VggRes(),                             48,  "gtsrb",    "checkpoints/m2_vggres_best.pt"),
    ("m3_resnet18",    lambda: M3Transfer("resnet18", pretrained=False),    224, "imagenet", "checkpoints/m3_resnet18_best.pt"),
    ("m3_mobilenetv2", lambda: M3Transfer("mobilenetv2", pretrained=False), 224, "imagenet", "checkpoints/m3_mobilenetv2_best.pt"),
    ("m3_effnetb0",    lambda: M3Transfer("effnetb0", pretrained=False),    224, "imagenet", "checkpoints/m3_effnetb0_best.pt"),
]


def load_checkpoint(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """Nạp trọng số. Checkpoint lưu dạng {'model_state': ...} hoặc state_dict trần."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj
    model.load_state_dict(state, strict=True)    # strict=True: sai cấu trúc là báo ngay
    return model.eval()


@torch.no_grad()
def measure_latency(model, img_size, device, n=40, warmup=10) -> float:
    """Latency p50 (ms) cho batch=1. Warmup + đồng bộ thiết bị trước/sau khi bấm giờ."""
    x = torch.randn(1, 3, img_size, img_size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        model(x)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(times, 50))


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    cpu = torch.device("cpu")
    print(f"Thiết bị: {device}\n")

    rows, preds = [], {}
    for name, build, img_size, normalize, ckpt in MODELS:
        if not Path(ckpt).exists():
            print(f"  ⚠ thiếu {ckpt}, bỏ qua {name}")
            continue
        model = load_checkpoint(build().to(device), ckpt)
        loader = make_loader(INDEX_CSV, "test", img_size, 128, normalize)
        y_true, y_prob = predict(model, loader, device)
        m = compute_metrics(y_true, y_prob)
        preds[name] = m["y_pred"]
        n_params = sum(p.numel() for p in model.parameters())
        lat = measure_latency(model.to(cpu), img_size, cpu)   # latency đo trên CPU
        rows.append((name, n_params, m["top1"], m["macro_f1"], lat))
        print(f"  {name:16} top1={m['top1']:.4f}  macroF1={m['macro_f1']:.4f}")

    # --- Bảng so sánh ---
    print("\n" + "=" * 72)
    print(f"{'Model':16}{'Tham số':>12}{'Top-1':>9}{'Macro-F1':>10}{'CPU p50 (ms)':>14}")
    print("-" * 72)
    for name, n, top1, f1, lat in sorted(rows, key=lambda r: -r[3]):
        print(f"{name:16}{n/1e6:>10.2f}M{top1*100:>8.2f}%{f1:>10.4f}{lat:>13.1f}")

    # --- McNemar giữa các cặp ---
    print("\n" + "=" * 72)
    print("McNemar — chênh lệch có ý nghĩa thống kê không? (p<0.05 = có)")
    print("-" * 72)
    names = [r[0] for r in rows]
    # lấy lại y_true của tập test (giống nhau cho các model cùng img_size/normalize
    # — ở đây dùng nhãn từ loader test của model đầu; nhãn test cố định theo index.csv)
    ref_true = make_loader(INDEX_CSV, "test", 48, 128, "gtsrb").dataset.frame["class_id"].to_numpy()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            r = mcnemar(ref_true, preds[a], preds[b])
            kl = {"A": f"{a} hơn", "B": f"{b} hơn", "tie": "tương đương"}[r["better"]]
            print(f"  {a:14} vs {b:14}  p={r['p_value']:.4f}  {kl}")


if __name__ == "__main__":
    main()
