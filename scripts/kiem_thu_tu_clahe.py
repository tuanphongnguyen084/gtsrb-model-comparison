#!/usr/bin/env python
"""
kiem_thu_tu_clahe.py — nhiễu áp TRƯỚC hay SAU CLAHE thì khác bao nhiêu?

CHỦ: Huy (robustness)

★ VẤN ĐỀ ★

Bảng robustness cho hai con số không tin được:

    model            low_light (relative_top1)
    m3_effnetb0      0,009   <- dưới mức đoán bừa 1/43 = 0,023
    m3_mobilenetv2   0,088
    m1_lenet         0,779

EfficientNet-B0 sụp xuống 0,0100 ngay ở mức 1 (gamma 1,5 — chỉ hơi tối) rồi
PHẲNG TỊT 0,0094–0,0100 qua cả 5 mức. Đường robustness thật thì giảm dần.

★ GIẢ THUYẾT ★

Đường ống hiện tại áp nhiễu SAU CLAHE:

    ảnh gốc -> cắt ROI -> CLAHE -> resize 48 -> nội suy 224 -> LÀM TỐI -> model

Nhưng khi triển khai thật, thứ tự NGƯỢC LẠI:

    camera -> ảnh đã tối -> cắt ROI -> CLAHE -> resize -> model
                                        ^^^^^ CLAHE BÙ LẠI độ tối

CLAHE là cân bằng histogram thích nghi — việc của nó chính là kéo lại tương
phản. Áp nhiễu sau CLAHE nghĩa là bước bù sáng diễn ra TRƯỚC bước làm tối, nên
nó không bù được gì. Phép đo vì vậy nặng tay hơn thực tế, và nặng tay nhất với
model dựa nhiều vào thống kê độ sáng tuyệt đối — đúng chỗ khác nhau giữa một
backbone pretrained ImageNet và một CNN nhỏ train từ đầu.

Nếu giả thuyết đúng thì kết luận "M1 bền nhất" một phần là HIỆN VẬT ĐO, không
phải tính chất của model.

★ CỔNG TỰ KIỂM — VÌ SAO CÓ ★

Lần thử đầu của tôi dựng lại đường ống SAI: resize ảnh gốc trực tiếp sang 224
thay vì qua cache 48×48. Nhánh ảnh sạch cho accuracy 0,352 thay vì 0,99, nên
mọi con số sau đó vô nghĩa — mà nhìn bảng kết quả thì không thấy gì bất thường.

Vì vậy script này BẮT BUỘC kiểm nhánh ảnh sạch khớp với số đã biết trong
reports/tables/robustness.csv trước khi in bất cứ kết luận nào. Lệch quá
NGUONG_SACH thì DỪNG.

    python scripts/kiem_thu_tu_clahe.py --n 512
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from _common import load_run, make_dataset
from pathlib import Path

from gtsrb.data.preprocess import apply_preprocess, crop_roi
from gtsrb.data.transforms import build_normalize
from gtsrb.robustness.corruptions import apply_corruption
from gtsrb.utils.logging import get_logger

log = get_logger()

# Nhánh ảnh sạch phải khớp số trong robustness.csv trong khoảng này.
# 0,02 là đủ rộng cho việc lấy mẫu con, đủ chặt để bắt lỗi dựng sai đường ống
# (lần trước lệch 0,64 — sai kiểu đó thì không thể lọt).
NGUONG_SACH = 0.02


def _resize_cache(rgb: np.ndarray, size: int) -> np.ndarray:
    """Giống hệt gtsrb.data.preprocess: thu nhỏ dùng INTER_AREA."""
    shrinking = rgb.shape[0] > size
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    return cv2.resize(rgb, (size, size), interpolation=interp)


def _noi_suy(rgb: np.ndarray, img_size: int) -> np.ndarray:
    """Giống hệt GTSRBDataset.__getitem__: bilinear + clamp + uint8."""
    if rgb.shape[0] == img_size:
        return rgb
    t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
    t = F.interpolate(t.unsqueeze(0).float(), size=(img_size, img_size),
                      mode="bilinear", align_corners=False)
    return t.squeeze(0).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()


def dung_anh(rows, thu_tu: str, kind: str, sev: int,
             cache_size: int, img_size: int, preprocess: str,
             seed: int = 0) -> np.ndarray:
    """Dựng batch ảnh uint8 HWC theo một trong hai thứ tự.

    'sau'   = CLAHE -> resize -> nội suy -> NHIỄU      (đường ống ĐANG dùng)
    'truoc' = NHIỄU -> CLAHE -> resize -> nội suy      (thứ tự khi TRIỂN KHAI)
    """
    rng = np.random.default_rng(seed)
    out = []
    for r in rows.itertuples():
        bgr = cv2.imread(r.path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = crop_roi(rgb, r.roi_x1, r.roi_y1, r.roi_x2, r.roi_y2, margin=0.10)

        if thu_tu == "truoc" and sev:
            rgb = apply_corruption(rgb, kind, sev, rng)     # ở độ phân giải GỐC

        rgb = apply_preprocess(rgb, preprocess)
        rgb = _resize_cache(rgb, cache_size)
        rgb = _noi_suy(rgb, img_size)

        if thu_tu == "sau" and sev:
            rgb = apply_corruption(rgb, kind, sev, rng)     # ở img_size, như hiện tại
        out.append(rgb)
    return np.stack(out)


@torch.no_grad()
def do_acc(model, imgs: np.ndarray, y: np.ndarray, norm, device) -> float:
    x = torch.stack([norm(torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1))
                     for im in imgs]).to(device)
    pred = model(x).argmax(1).cpu().numpy()
    return float((pred == y).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=512, help="số ảnh test lấy mẫu")
    ap.add_argument("--models", nargs="+",
                    default=["m3_effnetb0", "m3_mobilenetv2", "m3_resnet18",
                             "m2_vggres", "m1_lenet"])
    ap.add_argument("--corruptions", nargs="+", default=["low_light", "gauss_noise"])
    ap.add_argument("--severities", nargs="+", type=int, default=[1, 3, 5])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="reports/tables/thu_tu_clahe.csv")
    args = ap.parse_args()

    device = torch.device(args.device or
                          ("mps" if torch.backends.mps.is_available() else "cpu"))
    idx = pd.read_csv("data/processed/index.csv")
    # Lấy mẫu RẢI ĐỀU theo lớp, không lấy 512 ảnh đầu (index.csv sắp theo lớp,
    # 512 ảnh đầu chỉ có vài lớp -> macro-F1 và accuracy đều lệch).
    test = (idx[idx.split == "test"]
            .groupby("class_id", group_keys=False)
            .apply(lambda g: g.head(max(1, args.n // 43)), include_groups=False)
            .reset_index(drop=True))
    test["path"] = idx.loc[test.index, "path"].values if "path" not in test else test["path"]
    log.info("Lấy %d ảnh test, %d lớp", len(test), test.class_id.nunique())

    sach_biet = (pd.read_csv("reports/tables/robustness.csv")
                 .query("corruption == 'clean'").set_index("model")["top1"].to_dict())

    ket = []
    for ten in args.models:
        runs = [d for d in sorted(Path("artifacts/runs").glob(f"{ten}_s42_2*"))
                if (d / "result.json").exists()]
        if not runs:
            log.warning("%s: không có run -> bỏ qua", ten)
            continue
        model, cfg, _ = load_run(runs[0], device)
        ds = make_dataset(cfg, "test")
        norm = build_normalize(ds.mean, ds.std)
        cache_size = ds.cache_img_size or cfg.data.img_size
        y = test["class_id"].to_numpy()

        # ---- CỔNG: nhánh ảnh sạch phải khớp robustness.csv ----
        imgs = dung_anh(test, "sau", "low_light", 0, cache_size,
                        cfg.data.img_size, cfg.data.preprocess)
        acc_sach = do_acc(model, imgs, y, norm, device)
        mong_doi = sach_biet.get(ten)
        log.info("")
        log.info("=" * 70)
        log.info("%s — ảnh sạch: dựng lại %.4f  ·  robustness.csv %.4f",
                 ten, acc_sach, mong_doi if mong_doi else float("nan"))
        if mong_doi is not None and abs(acc_sach - mong_doi) > NGUONG_SACH:
            log.error("  ★ DỪNG: lệch %.4f > %.2f -> đường ống dựng lại KHÔNG khớp "
                      "đường ống thật. Mọi số sau đây sẽ vô nghĩa.",
                      abs(acc_sach - mong_doi), NGUONG_SACH)
            raise SystemExit(1)
        log.info("  cổng tự kiểm: ĐẠT")

        for kind in args.corruptions:
            for sev in args.severities:
                a = {}
                for tt in ("sau", "truoc"):
                    imgs = dung_anh(test, tt, kind, sev, cache_size,
                                    cfg.data.img_size, cfg.data.preprocess)
                    a[tt] = do_acc(model, imgs, y, norm, device)
                ket.append({"model": ten, "corruption": kind, "severity": sev,
                            "acc_sach": round(acc_sach, 4),
                            "nhieu_SAU_clahe": round(a["sau"], 4),
                            "nhieu_TRUOC_clahe": round(a["truoc"], 4),
                            "chenh": round(a["truoc"] - a["sau"], 4)})
                log.info("  %-12s mức %d:  sau CLAHE %.4f  |  trước CLAHE %.4f  "
                         "|  chênh %+.4f", kind, sev, a["sau"], a["truoc"],
                         a["truoc"] - a["sau"])

    frame = pd.DataFrame(ket)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    log.info("")
    log.info("=" * 70)
    log.info("BẢNG -> %s", args.out)
    if not frame.empty:
        log.info("\n%s", frame.to_string(index=False))
        log.info("")
        log.info("CÁCH ĐỌC: `chênh` > 0 nghĩa là áp nhiễu TRƯỚC CLAHE cho accuracy")
        log.info("  CAO HƠN, tức CLAHE thật sự bù lại được, tức bảng robustness hiện")
        log.info("  tại đang NẶNG TAY hơn thực tế. Chênh càng lớn thì phép đo càng")
        log.info("  lệch với điều kiện triển khai.")


if __name__ == "__main__":
    main()
