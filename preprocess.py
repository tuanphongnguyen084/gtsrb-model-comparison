"""
preprocess.py — tải GTSRB, dựng index.csv, chia train/val theo track, và tạo
cache ảnh đã tiền xử lý (crop ROI + CLAHE + resize 48x48) dưới dạng .npy.

Chạy MỘT LẦN trước khi train:
    python preprocess.py            # dùng data/raw đã có
    python preprocess.py --download # tải từ archive gốc nếu chưa có

Đầu ra trong data/processed/:
    index.csv                 nguồn sự thật: path, class_id, track_id, ROI, split
    images_48_clahe.npy       mảng (N, 48, 48, 3) uint8 — cache để train nhanh

★ VÌ SAO CÁC BƯỚC NHƯ VẬY — phần kiến thức cốt lõi:

1. CROP ROI lề 10%: cắt sát mất viền biển báo (viền đỏ của biển cấm là đặc trưng
   quan trọng); nới 10% vẫn bỏ được phần lớn nền trời/đường vô nghĩa.

2. CLAHE trên kênh L của LAB: cân bằng tương phản cục bộ để biển báo rõ hơn
   trong điều kiện sáng khác nhau. Làm trên kênh L (độ sáng) chứ KHÔNG trên
   R,G,B — vì cân bằng từng kênh màu làm ĐỔI MÀU, mà màu biển báo mang nghĩa.
   CLAHE (không phải HE toàn cục) để không phóng đại nhiễu ở vùng trời phẳng.

3. SPLIT THEO TRACK: GTSRB quay mỗi biển báo vật lý thành 30 frame liên tiếp.
   Chia ngẫu nhiên theo ảnh thì frame 12 vào train, frame 13 (gần giống hệt)
   vào val -> model chỉ cần NHỚ, không cần KHÁI QUÁT -> val cao giả tạo (rò rỉ
   dữ liệu). Phải gom cả 30 frame của một track về cùng một phía, dùng
   StratifiedGroupKFold với nhóm = cặp (class_id, track_id).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
INDEX_CSV = OUT_DIR / "index.csv"
IMG_SIZE = 48

# Archive gốc của GTSRB (Đại học Bochum, lưu trên ERDA). Tải trực tiếp để chắc
# chắn lấy bộ "Final Training" 39.209 ảnh, KHÔNG phải bộ 26.640 ảnh của torchvision.
BASE_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370"
ARCHIVES = {
    "GTSRB_Final_Training_Images.zip": "full_train",
    "GTSRB_Final_Test_Images.zip": "full_test",
    "GTSRB_Final_Test_GT.zip": "full_test_gt",
}

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ---------------------------------------------------------------------
# 1. Tải (tuỳ chọn)
# ---------------------------------------------------------------------
def download_gtsrb(root: Path = RAW_DIR) -> None:
    import urllib.request
    import zipfile
    root.mkdir(parents=True, exist_ok=True)
    for zip_name, out_name in ARCHIVES.items():
        out_dir = root / out_name
        if out_dir.exists() and any(out_dir.rglob("*")):
            print(f"  đã có {out_name}, bỏ qua")
            continue
        print(f"  tải {zip_name} ...")
        zip_path = root / zip_name
        urllib.request.urlretrieve(f"{BASE_URL}/{zip_name}", zip_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out_dir)
        zip_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------
# 2. Dựng index.csv (tách track_id từ tên file — khoá chống rò rỉ)
# ---------------------------------------------------------------------
def _find_dir(root: Path, *candidates: str) -> Path:
    for c in candidates:
        if (root / c).is_dir():
            return root / c
    raise FileNotFoundError(f"Không thấy thư mục dữ liệu, đã thử {candidates} trong {root}")


def _read_gt(csv_path: Path) -> list[dict]:
    with open(csv_path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter=";"))   # GTSRB dùng ';'


def build_index(root: Path = RAW_DIR) -> pd.DataFrame:
    rows: list[dict] = []

    # --- tập train: 43 thư mục lớp, mỗi thư mục một GT-*.csv ---
    train_dir = _find_dir(root, "full_train/GTSRB/Final_Training/Images",
                          "GTSRB/Final_Training/Images")
    class_dirs = sorted(d for d in train_dir.iterdir() if d.is_dir())
    if len(class_dirs) != 43:
        raise RuntimeError(f"Mong đợi 43 thư mục lớp, thấy {len(class_dirs)}")
    for class_dir in class_dirs:
        class_id = int(class_dir.name)
        for rec in _read_gt(next(class_dir.glob("GT-*.csv"))):
            # '00005_00017.ppm' -> track 5, frame 17
            track, _, frame = Path(rec["Filename"]).stem.partition("_")
            rows.append({
                "path": (class_dir / rec["Filename"]).as_posix(),
                "class_id": class_id, "track_id": int(track), "frame_id": int(frame),
                "roi_x1": int(rec["Roi.X1"]), "roi_y1": int(rec["Roi.Y1"]),
                "roi_x2": int(rec["Roi.X2"]), "roi_y2": int(rec["Roi.Y2"]),
                "source": "train",
            })
    n_train = len(rows)
    if n_train < 39_000:
        print(f"  ⚠ chỉ {n_train} ảnh train — bộ ĐẦY ĐỦ phải 39.209. Có thể tải nhầm "
              f"bộ 26.640 của torchvision. Chạy: python preprocess.py --download")

    # --- tập test chính thức: 12.630 ảnh + 1 file nhãn chung ---
    test_dir = _find_dir(root, "full_test/GTSRB/Final_Test/Images",
                         "GTSRB/Final_Test/Images")
    test_gt = next(root.rglob("GT-final_test.csv"))
    for pos, rec in enumerate(_read_gt(test_gt)):
        rows.append({
            "path": (test_dir / rec["Filename"]).as_posix(),
            "class_id": int(rec["ClassId"]),
            # test không công bố track; mỗi ảnh một track riêng, đánh số ÂM để
            # không bao giờ trùng track train (nếu trùng, check rò rỉ sẽ báo giả).
            "track_id": -1 - pos, "frame_id": 0,
            "roi_x1": int(rec["Roi.X1"]), "roi_y1": int(rec["Roi.Y1"]),
            "roi_x2": int(rec["Roi.X2"]), "roi_y2": int(rec["Roi.Y2"]),
            "source": "test_official",
        })

    frame = pd.DataFrame(rows)
    print(f"  index: {n_train} train + {len(rows)-n_train} test = {len(rows)} ảnh")
    return frame


# ---------------------------------------------------------------------
# 3. Chia train/val theo track + ASSERT chống rò rỉ
# ---------------------------------------------------------------------
def make_split(frame: pd.DataFrame, val_ratio: float = 0.2, seed: int = 42) -> pd.DataFrame:
    frame["split"] = "test"                       # test_official luôn là test
    pool = frame[frame["source"] == "train"]
    labels = pool["class_id"].to_numpy()
    # nhóm = (lớp, track): phải kèm class_id vì track đánh số lại trong mỗi lớp
    groups = np.array([f"{c}_{t}" for c, t in zip(pool["class_id"], pool["track_id"])])

    n_splits = max(2, round(1.0 / val_ratio))     # val_ratio=0.2 -> 5 fold, lấy fold đầu
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_pos, val_pos = next(splitter.split(pool, labels, groups))
    idx = pool.index.to_numpy()
    frame.loc[idx[train_pos], "split"] = "train"
    frame.loc[idx[val_pos], "split"] = "val"

    # ★ ASSERT CHỐNG RÒ RỈ — cổng an toàn quan trọng nhất của cả dự án.
    # Nếu bất kỳ track nào (cặp lớp+track) nằm ở CẢ train và val thì split sai.
    def track_keys(part):
        return {f"{c}_{t}" for c, t in zip(part["class_id"], part["track_id"])}
    tr, va = frame[frame.split == "train"], frame[frame.split == "val"]
    shared = track_keys(tr) & track_keys(va)
    assert len(shared) == 0, f"RÒ RỈ DỮ LIỆU: {len(shared)} track chung giữa train và val"

    c = frame["split"].value_counts()
    print(f"  split: train={c.get('train',0)} val={c.get('val',0)} test={c.get('test',0)} "
          f"— 0 track rò rỉ ✓")
    return frame


# ---------------------------------------------------------------------
# 4. Cache ảnh đã tiền xử lý -> .npy
# ---------------------------------------------------------------------
def _process_one(path: str, roi: tuple[int, int, int, int], size: int) -> np.ndarray:
    """Đọc -> crop ROI lề 10% -> CLAHE kênh L -> resize. Trả HWC uint8 RGB size x size."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # crop ROI nới lề 10%
    x1, y1, x2, y2 = roi
    h, w = rgb.shape[:2]
    px, py = int((x2 - x1) * 0.10), int((y2 - y1) * 0.10)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    if x2 > x1 and y2 > y1:
        rgb = rgb[y1:y2, x1:x2]

    # CLAHE trên kênh L của LAB (giữ màu)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = _CLAHE.apply(lab[:, :, 0])
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # resize: thu nhỏ -> INTER_AREA (chống răng cưa), phóng to -> INTER_CUBIC
    shrink = size < rgb.shape[0] or size < rgb.shape[1]
    interp = cv2.INTER_AREA if shrink else cv2.INTER_CUBIC
    return cv2.resize(rgb, (size, size), interpolation=interp)


def build_cache(frame: pd.DataFrame, size: int = IMG_SIZE) -> np.ndarray:
    """Tiền xử lý toàn bộ ảnh theo đúng thứ tự index.csv, lưu (N, size, size, 3) uint8."""
    out = np.zeros((len(frame), size, size, 3), dtype=np.uint8)
    for i, row in enumerate(frame.itertuples()):
        roi = (row.roi_x1, row.roi_y1, row.roi_x2, row.roi_y2)
        out[i] = _process_one(row.path, roi, size)
        if (i + 1) % 5000 == 0:
            print(f"  cache {i+1}/{len(frame)}")
    return out


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true", help="tải GTSRB nếu chưa có")
    ap.add_argument("--size", type=int, default=IMG_SIZE)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.download:
        print("1. Tải dữ liệu"); download_gtsrb()

    print("2. Dựng index.csv")
    frame = build_index()

    print("3. Chia train/val theo track")
    frame = make_split(frame)
    frame.to_csv(INDEX_CSV, index=False)
    print(f"  đã ghi {INDEX_CSV}")

    print("4. Tạo cache ảnh (crop + CLAHE + resize)")
    cache = build_cache(frame, args.size)
    cache_path = OUT_DIR / f"images_{args.size}_clahe.npy"
    np.save(cache_path, cache)
    print(f"  đã ghi {cache_path}  {cache.shape}  ({cache.nbytes/1e6:.0f} MB)")

    print("Xong. Giờ train: python train_m1.py")


if __name__ == "__main__":
    main()
