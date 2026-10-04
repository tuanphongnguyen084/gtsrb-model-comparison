#!/usr/bin/env python
"""
make_gradcam.py — lưới Grad-CAM: dự đoán ĐÚNG, dự đoán SAI, và trước/sau nhiễu.

CHỦ: Phong Trần

★ NHÓM ẢNH SAI THÚ VỊ HƠN NHÓM ĐÚNG ★
Heatmap trên ảnh đúng gần như luôn nằm trên biển báo — không nói được gì mới.
Ảnh SAI mới cho biết model đang nhìn vào đâu khi nó lầm.

★ PHẢI BÁO CÁO ĐỘ PHÂN GIẢI HEATMAP ★
M2 ở input 48 cho feature map cuối 3x3 -> mỗi "ô" heatmap là 16x16 pixel.
M3 ở input 224 cho 7x7 -> nét hơn hẳn. Nhưng đó là lợi thế ĐỘ PHÂN GIẢI,
không phải vì model "hiểu" hơn. Script tự in con số này ra.

Ví dụ:
    python scripts/make_gradcam.py --runs "artifacts/runs/*"
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import numpy as np
import torch

from gtsrb import CLASS_NAMES
from gtsrb.data.transforms import build_normalize
from gtsrb.explain.gradcam import GradCAM, feature_map_size
from gtsrb.robustness.corruptions import apply_corruption
from _common import find_run_dirs, free_memory, load_run, make_dataset
from gtsrb.utils.logging import get_logger
from gtsrb.utils.seed import pick_device, set_seed

log = get_logger()


def draw_grid(panels: list[dict], out_path: Path, title: str) -> None:
    """panels: list các dict {image, heatmap_overlay, caption}. Vẽ 2 hàng/panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(panels)
    if n == 0:
        log.warning("Không có panel nào để vẽ (%s)", title)
        return

    fig, axes = plt.subplots(2, n, figsize=(2.3 * n, 5.4), squeeze=False)
    for col, panel in enumerate(panels):
        axes[0][col].imshow(panel["image"])
        axes[0][col].set_title(panel["caption"], fontsize=7.5)
        axes[1][col].imshow(panel["overlay"])
        for row in (0, 1):
            axes[row][col].set_xticks([]); axes[row][col].set_yticks([])
    axes[0][0].set_ylabel("ảnh gốc", fontsize=9)
    axes[1][0].set_ylabel("Grad-CAM", fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    log.info("  -> %s", out_path)


def report_resolution(model, cfg, device) -> tuple[int, int]:
    """In ra ĐỘ PHÂN GIẢI heatmap — phải báo cáo TRƯỚC khi diễn giải bất cứ hình nào.

    Số đo thật của dự án:
        M1 @48  -> feature map 12x12, mỗi ô ~ 4x4 pixel    (MỊN NHẤT)
        M2 @48  -> feature map  3x3,  mỗi ô ~ 16x16 pixel  (THÔ NHẤT)
        M3 @224 -> feature map  7x7,  mỗi ô ~ 32x32 pixel
    M2 — model thiết kế công phu nhất — lại cho heatmap thô nhất, vì nó có 4 lớp
    MaxPool so với 2 của M1. Nói "heatmap đúng vào biển báo" với M2 là phát biểu
    rất lỏng: cả ảnh chỉ có 9 ô.
    """
    fm_h, fm_w = feature_map_size(model, cfg.data.img_size, device)
    ratio = cfg.data.img_size / max(1, fm_h)
    log.info("  target layer: %s", type(model.gradcam_target_layer).__name__)
    log.info("  feature map: %dx%d ở input %dx%d -> mỗi ô heatmap ~ %.0fx%.0f pixel",
             fm_h, fm_w, cfg.data.img_size, cfg.data.img_size, ratio, ratio)
    if fm_h <= 3:
        log.warning("  Heatmap RẤT THÔ (%dx%d). Phải nêu giới hạn này khi trình bày "
                    "— xem src/gtsrb/explain/gradcam.py.", fm_h, fm_w)
    return fm_h, fm_w


def collect_panels(model, dataset, normalizer, cam, device,
                   n_correct: int, n_wrong: int, scan: int) -> tuple[list, list]:
    """Quét tập test, thu thập panel cho dự đoán ĐÚNG và dự đoán SAI.

    Lấy mẫu RẢI ĐỀU (bước nhảy `step`) thay vì n ảnh đầu, vì index.csv sắp theo lớp
    nên n ảnh đầu sẽ toàn lớp 0.
    """
    correct_panels, wrong_panels = [], []
    step = max(1, len(dataset) // min(scan, len(dataset)))

    for i in range(0, len(dataset), step):
        if len(correct_panels) >= n_correct and len(wrong_panels) >= n_wrong:
            break

        tensor_uint8, label = dataset[i]
        image = tensor_uint8.permute(1, 2, 0).numpy()
        x = normalizer(tensor_uint8).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.softmax(model(x).float(), dim=1)[0]
        pred = int(probs.argmax().item())
        confidence = float(probs[pred].item())

        is_correct = (pred == label)
        bucket = correct_panels if is_correct else wrong_panels
        limit = n_correct if is_correct else n_wrong
        if len(bucket) >= limit:
            continue

        heatmap = cam(x, class_idx=pred)
        bucket.append({
            "image": image,
            "overlay": GradCAM.overlay(image, heatmap),
            "caption": (f"thật: {CLASS_NAMES[label][:18]}\n"
                        f"đoán: {CLASS_NAMES[pred][:18]} ({confidence:.2f})"),
        })

    return correct_panels, wrong_panels


def corrupted_panels(model, dataset, normalizer, cam, device,
                     base_index: int = 0) -> tuple[list, int]:
    """Cùng MỘT ảnh, trước và sau 5 loại nhiễu — attention có TRÔI không?

    Nếu attention trôi khỏi biển báo đúng lúc dự đoán sai, ta có một GIẢI THÍCH
    CƠ CHẾ cho con số robustness, không chỉ nói "accuracy giảm".
    """
    rng = np.random.default_rng(0)
    tensor_uint8, label = dataset[base_index]
    base_image = tensor_uint8.permute(1, 2, 0).numpy()

    panels = []
    for kind, severity in [("clean", 0), ("fog", 3), ("low_light", 3),
                           ("motion_blur", 3), ("gauss_noise", 3), ("occlusion", 3)]:
        array = (base_image if kind == "clean"
                 else apply_corruption(base_image, kind, severity, rng))
        tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
        x = normalizer(tensor).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.softmax(model(x).float(), dim=1)[0]
        pred = int(probs.argmax().item())
        heatmap = cam(x, class_idx=pred)

        panels.append({
            "image": array,
            "overlay": GradCAM.overlay(array, heatmap),
            "caption": (f"{kind}\n{'OK' if pred == label else 'SAI'}: "
                        f"{CLASS_NAMES[pred][:16]} ({float(probs[pred]):.2f})"),
        })
    return panels, label


def gradcam_for_run(run_dir: Path, device, figures_dir: Path, args) -> None:
    """Sinh đủ 3 lưới hình cho một run: đúng / SAI / trước-sau nhiễu."""
    log.info("")
    log.info("=" * 68)
    log.info("Grad-CAM cho %s", run_dir.name)

    model, cfg, _ = load_run(run_dir, device)
    name = model.model_name
    fm_h, fm_w = report_resolution(model, cfg, device)

    # return_uint8=True: cần ảnh THÔ để vẽ overlay, rồi mới chuẩn hoá cho model
    dataset = make_dataset(cfg, "test", return_uint8=True)
    normalizer = build_normalize(dataset.mean, dataset.std)

    cam = GradCAM(model)
    try:
        correct, wrong = collect_panels(model, dataset, normalizer, cam, device,
                                        args.n_correct, args.n_wrong, args.scan)
        draw_grid(correct, figures_dir / f"gradcam_{name}_correct.png",
                  f"Grad-CAM {name} — dự đoán ĐÚNG (feature map {fm_h}x{fm_w})")
        draw_grid(wrong, figures_dir / f"gradcam_{name}_wrong.png",
                  f"Grad-CAM {name} — dự đoán SAI: model nhìn vào đâu? "
                  f"(feature map {fm_h}x{fm_w})")

        panels, label = corrupted_panels(model, dataset, normalizer, cam, device)
        draw_grid(panels, figures_dir / f"gradcam_{name}_corrupted.png",
                  f"Grad-CAM {name} — attention có TRÔI khi có nhiễu? "
                  f"(nhãn thật: {CLASS_NAMES[label]})")
    finally:
        cam.close()      # BẮT BUỘC tháo hook, nếu không mọi forward sau đều chậm

    free_memory(model, device)


def main() -> None:
    """Sinh 3 lưới Grad-CAM (đúng / SAI / trước-sau nhiễu) cho mọi run."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=["artifacts/runs/*"])
    parser.add_argument("--figures", default="reports/figures")
    parser.add_argument("--n-correct", type=int, default=6)
    parser.add_argument("--n-wrong", type=int, default=6)
    parser.add_argument("--scan", type=int, default=3000,
                        help="quet N anh test de tim ca dung/sai")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    set_seed(42)
    device = pick_device(args.device or "auto")
    figures_dir = Path(args.figures)
    figures_dir.mkdir(parents=True, exist_ok=True)

    for run_dir in find_run_dirs(args.runs, main_only=True):
        gradcam_for_run(run_dir, device, figures_dir, args)


if __name__ == "__main__":
    main()
