"""
download — tải GTSRB và dựng bảng chỉ mục `index.csv`.

CHỦ: Huy

GTSRB gồm 2 phần:
  - Tập train: 39.209 ảnh .ppm, chia trong 43 thư mục theo lớp (00000..00042).
    Mỗi thư mục có kèm `GT-000xx.csv` chứa ROI bounding box của từng ảnh.
  - Tập test chính thức: 12.630 ảnh + 1 file `GT-final_test.csv` chứa nhãn.
  Tổng: 51.839 ảnh.

★ CÁI BẪY ĐÃ GẶP THẬT (ghi vào docs/SU_CO.md):
  `torchvision.datasets.GTSRB(split="train", download=True)` KHÔNG tải bộ train
  đầy đủ. Nó tải "GTSRB-Training_fixed.zip" — tức là bộ train 26.640 ảnh của
  GIẢI IJCNN 2011, không phải bộ "Final Training" 39.209 ảnh.
  Cách nhận biết: đếm ảnh của lớp 0.
      bộ thiếu  -> lớp 0 có 150 ảnh,  lớp 1: 1500,  lớp 2: 1500  (tổng 26.640)
      bộ đủ     -> lớp 0 có 210 ảnh,  lớp 1: 2220,  lớp 2: 2250  (tổng 39.209)
  Vì vậy hàm download_gtsrb() dưới đây tải TRỰC TIẾP từ archive gốc của
  Đại học Bochum (ERDA), không qua torchvision.

★ ĐIỂM QUAN TRỌNG NHẤT CỦA FILE NÀY: cấu trúc TRACK.
Tên file ảnh train có dạng  000{track}_000{frame}.ppm
GTSRB không chụp mỗi biển báo một lần — xe chạy tới gần một tấm biển và camera
quay 30 FRAME LIÊN TIẾP của CÙNG MỘT TẤM BIỂN VẬT LÝ, lưu thành một "track".
Hàm dưới đây tách `track_id` ra khỏi tên file để split.py dùng chống rò rỉ dữ liệu.
Không có cột track_id thì không chống được rò rỉ. Xem docs/LY_THUYET.md mục 1.1.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from gtsrb.utils.logging import get_logger

log = get_logger()

# Các cột của index.csv — hợp đồng với cả nhóm, xem docs/INTERFACE.md mục 1
INDEX_COLUMNS = [
    "path",        # đường dẫn tới ảnh (tương đối từ gốc repo)
    "class_id",    # 0..42
    "track_id",    # khoá chống rò rỉ dữ liệu
    "frame_id",    # thứ tự frame trong track
    "roi_x1", "roi_y1", "roi_x2", "roi_y2",   # bounding box từ CSV gốc
    "width", "height",                         # kích thước ảnh gốc
    "source",      # "train" | "test_official"
]


# Archive gốc của GTSRB (Ruhr-Universität Bochum, lưu trên ERDA).
# Tải trực tiếp để chắc chắn lấy đúng bộ "Final Training" 39.209 ảnh.
BASE_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370"
ARCHIVES = {
    # tên file zip                      : (thư mục giải nén, mô tả, dung lượng xấp xỉ)
    "GTSRB_Final_Training_Images.zip": ("full_train", "tập train 39.209 ảnh", "276 MB"),
    "GTSRB_Final_Test_Images.zip": ("full_test", "tập test 12.630 ảnh", "88 MB"),
    "GTSRB_Final_Test_GT.zip": ("full_test_gt", "nhãn của tập test", "0,1 MB"),
}


def _download_file(url: str, dest: Path) -> None:
    """Tải một file, in tiến độ. Đã có file (và không rỗng) thì bỏ qua."""
    import urllib.request

    if dest.exists() and dest.stat().st_size > 1024:
        log.info("  đã có %s, bỏ qua tải", dest.name)
        return

    def hook(block_count: int, block_size: int, total_size: int) -> None:
        """Báo tiến độ tải, chỉ in mỗi 10% để log không bị tràn."""
        if total_size <= 0:
            return
        done = min(block_count * block_size, total_size)
        percent = 100.0 * done / total_size
        # Chỉ in mỗi 10% để log không bị tràn
        if int(percent) % 10 == 0 and done % (total_size // 10 or 1) < block_size:
            log.info("  %s: %.0f%% (%.0f/%.0f MB)",
                     dest.name, percent, done / 1e6, total_size / 1e6)

    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    tmp.rename(dest)


def download_gtsrb(root: str | Path = "data/raw", keep_zip: bool = False) -> Path:
    """Tải và giải nén GTSRB từ archive gốc. Gọi lại lần 2 thì tự bỏ qua.

    keep_zip=False xoá file zip sau khi giải nén (tiết kiệm ~364 MB).
    """
    import zipfile

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    for zip_name, (out_dir_name, description, size_text) in ARCHIVES.items():
        out_dir = root / out_dir_name
        if out_dir.exists() and any(out_dir.rglob("*")):
            log.info("Đã có %s (%s), bỏ qua.", out_dir_name, description)
            continue

        log.info("Tải %s (~%s)...", description, size_text)
        zip_path = root / zip_name
        _download_file(f"{BASE_URL}/{zip_name}", zip_path)

        log.info("  giải nén vào %s/", out_dir_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(out_dir)

        if not keep_zip:
            zip_path.unlink(missing_ok=True)

    log.info("Tải xong. Dữ liệu ở: %s", root)
    return root


def _find_dir(root: Path, *candidates: str) -> Path:
    """Tìm thư mục đầu tiên tồn tại trong danh sách ứng viên.

    Cần hàm này vì layout sau khi giải nén phụ thuộc phiên bản torchvision.
    Thà dò vài đường dẫn còn hơn hard-code một cái rồi vỡ ở máy người khác.
    """
    for candidate in candidates:
        path = root / candidate
        if path.is_dir():
            return path
    raise FileNotFoundError(
        f"Không tìm thấy thư mục dữ liệu. Đã thử: {candidates} trong {root}\n"
        f"Hãy chạy: python scripts/prepare_data.py --download"
    )


def _find_file(root: Path, pattern: str) -> Path:
    """Tìm file đầu tiên khớp pattern (đệ quy)."""
    hits = sorted(root.rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"Không tìm thấy file khớp {pattern!r} trong {root}")
    return hits[0]


def _read_gt_csv(csv_path: Path) -> list[dict]:
    """Đọc file CSV của GTSRB. Nó dùng dấu CHẤM PHẨY làm phân cách, không phải phẩy."""
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter=";"))


def _parse_train_filename(filename: str) -> tuple[int, int]:
    """'00005_00017.ppm' -> (track_id=5, frame_id=17)

    Đây là chỗ lấy ra thông tin quan trọng nhất cho việc split đúng.
    """
    stem = Path(filename).stem          # '00005_00017'
    track_text, _, frame_text = stem.partition("_")
    return int(track_text), int(frame_text)


def _index_train_split(root: Path) -> list[dict]:
    """Quét 43 thư mục lớp của tập train, trả danh sách dòng cho index.csv.

    Mỗi thư mục lớp có một file GT-000xx.csv chứa ROI bounding box của từng ảnh.
    Đây cũng là nơi tách ra `track_id` — khoá chống rò rỉ dữ liệu.
    """
    # Thứ tự ứng viên QUAN TRỌNG: bộ "Final_Training" (39.209 ảnh) phải được
    # ưu tiên trước bộ "Training" của torchvision (chỉ 26.640 ảnh). Xem bẫy ở đầu file.
    train_dir = _find_dir(
        root,
        "full_train/GTSRB/Final_Training/Images",   # <- bộ ĐẦY ĐỦ, ưu tiên
        "GTSRB/Final_Training/Images",
        "gtsrb/GTSRB/Training",                      # <- bộ thiếu của torchvision
        "GTSRB/Training",
        "Training",
    )
    log.info("Đọc tập train từ: %s", train_dir)

    class_dirs = sorted(d for d in train_dir.iterdir() if d.is_dir())
    if len(class_dirs) != 43:
        raise RuntimeError(
            f"Mong đợi 43 thư mục lớp, tìm thấy {len(class_dirs)} trong {train_dir}"
        )

    rows: list[dict] = []
    for class_dir in class_dirs:
        class_id = int(class_dir.name)              # '00002' -> 2
        gt_files = list(class_dir.glob("GT-*.csv"))
        if not gt_files:
            raise FileNotFoundError(f"Thiếu file GT-*.csv trong {class_dir}")

        for record in _read_gt_csv(gt_files[0]):
            filename = record["Filename"]
            track_id, frame_id = _parse_train_filename(filename)
            rows.append({
                "path": str((class_dir / filename).as_posix()),
                "class_id": class_id,
                "track_id": track_id,
                "frame_id": frame_id,
                "roi_x1": int(record["Roi.X1"]),
                "roi_y1": int(record["Roi.Y1"]),
                "roi_x2": int(record["Roi.X2"]),
                "roi_y2": int(record["Roi.Y2"]),
                "width": int(record["Width"]),
                "height": int(record["Height"]),
                "source": "train",
            })

    log.info("  -> %d ảnh train", len(rows))
    if len(rows) < 39_000:
        log.warning(
            "CHÚ Ý: chỉ có %d ảnh train. Bộ ĐẦY ĐỦ phải có 39.209 ảnh. "
            "Rất có thể đang dùng bộ 26.640 ảnh của giải IJCNN 2011 do torchvision tải. "
            "Chạy: python scripts/prepare_data.py --download  để lấy bộ đầy đủ.",
            len(rows),
        )
    return rows


def _index_test_split(root: Path, id_offset: int) -> list[dict]:
    """Quét tập test chính thức (12.630 ảnh + 1 file nhãn chung).

    `id_offset` để đánh track_id âm không trùng nhau — xem ghi chú bên dưới.
    """
    test_dir = _find_dir(
        root,
        "full_test/GTSRB/Final_Test/Images",
        "gtsrb/GTSRB/Final_Test/Images",
        "GTSRB/Final_Test/Images",
        "Final_Test/Images",
    )
    test_gt = _find_file(root, "GT-final_test.csv")
    log.info("Đọc tập test từ: %s (nhãn: %s)", test_dir, test_gt.name)

    rows: list[dict] = []
    for position, record in enumerate(_read_gt_csv(test_gt)):
        rows.append({
            "path": str((test_dir / record["Filename"]).as_posix()),
            "class_id": int(record["ClassId"]),
            # Tập test chính thức không công bố track_id. Mỗi ảnh test coi như một
            # track riêng, đánh số ÂM để chắc chắn không trùng track của tập train
            # (nếu trùng, check_leakage sẽ báo rò rỉ giả).
            "track_id": -1 - (id_offset + position),
            "frame_id": 0,
            "roi_x1": int(record["Roi.X1"]),
            "roi_y1": int(record["Roi.Y1"]),
            "roi_x2": int(record["Roi.X2"]),
            "roi_y2": int(record["Roi.Y2"]),
            "width": int(record["Width"]),
            "height": int(record["Height"]),
            "source": "test_official",
        })

    log.info("  -> %d ảnh test chính thức", len(rows))
    return rows


def _verify_paths_exist(frame: pd.DataFrame, root: Path, n_check: int = 50) -> None:
    """Kiểm tra tỉnh táo: index.csv có trỏ tới ảnh thật không.

    Thiếu ảnh là dấu hiệu giải nén lỗi. Phải báo NGAY tại đây, vì nếu để lọt thì
    lỗi chỉ lộ ra ở giữa buổi tiền xử lý 51.839 ảnh.
    """
    missing = [path for path in frame["path"].head(n_check) if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            f"index.csv trỏ tới ảnh không tồn tại, ví dụ: {missing[0]}\n"
            f"Có thể giải nén lỗi. Thử xoá {root} rồi tải lại."
        )


def build_index(root: str | Path = "data/raw",
                out_csv: str | Path = "data/processed/index.csv") -> pd.DataFrame:
    """Quét dữ liệu đã tải, dựng index.csv với đủ các cột trong INDEX_COLUMNS.

    index.csv là NGUỒN SỰ THẬT DUY NHẤT về dữ liệu cho cả nhóm.
    Không ai được đọc ảnh trực tiếp từ thư mục; mọi thứ đi qua bảng này.
    """
    root = Path(root)

    train_rows = _index_train_split(root)
    test_rows = _index_test_split(root, id_offset=len(train_rows))
    frame = pd.DataFrame(train_rows + test_rows, columns=INDEX_COLUMNS)

    _verify_paths_exist(frame, root)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False)
    log.info("Đã ghi %s (%d dòng)", out_csv, len(frame))
    return frame
