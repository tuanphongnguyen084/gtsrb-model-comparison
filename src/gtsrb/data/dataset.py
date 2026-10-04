"""
dataset — GTSRBDataset và build_dataloaders. Cửa duy nhất để lấy dữ liệu.

CHỦ: Huy

HỢP ĐỒNG VỚI CẢ NHÓM (docs/INTERFACE.md mục 1): không ai đọc ảnh trực tiếp từ
thư mục. Mọi thứ đi qua đây. Lý do: nếu 4 người tự đọc ảnh theo 4 cách thì 3 model
nhận đầu vào khác nhau và so sánh mất ý nghĩa.

CƠ CHẾ CACHE:
  Giải nén + CLAHE + resize cho 51.839 ảnh mất ~2 phút. Làm lại mỗi epoch thì
  GPU ngồi chờ CPU. Nên scripts/prepare_data.py tiền xử lý MỘT LẦN rồi lưu thành
  một mảng numpy duy nhất: data/processed/images_{size}_{preprocess}.npy
  Dataset mở file đó bằng mmap_mode='r' -> các worker của DataLoader dùng chung
  page cache của OS, không copy 358 MB cho mỗi worker.

VỀ M3 VÀ ĐỘ PHÂN GIẢI 224 — điểm này phải hiểu rõ để trả lời giảng viên:
  Cache lưu ở 48x48. M3 cần 224x224 nên Dataset nội suy 48 -> 224.
  Nội suy KHÔNG THÊM THÔNG TIN. Vậy upsample để làm gì?
  Vì ResNet18/MobileNetV2/EfficientNet-B0 downsample 32 lần: đưa 48x48 vào thì
  feature map cuối còn 1,5x1,5 (làm tròn 2x2), layer4 coi như vô dụng.
  224/32 = 7x7 là ĐÚNG CẤU HÌNH mà trọng số pretrained được học.
  Nói cách khác: upsample để khớp SCALE và STRIDE của mạng pretrained,
  không phải để có thêm chi tiết.
  (Thực tế ảnh gốc GTSRB phần lớn chỉ khoảng 50x50 pixel, nên dù cache ở 224
   cũng không có thêm bao nhiêu chi tiết thật. Đây là một hạn chế của bộ dữ liệu.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from gtsrb import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES
from gtsrb.data.preprocess import load_and_process
from gtsrb.data.transforms import build_normalize, build_transform
from gtsrb.utils.logging import get_logger
from gtsrb.utils.seed import seed_worker

log = get_logger()


def cache_path(processed_dir: str | Path, size: int, preprocess: str) -> Path:
    """Đường dẫn file cache. Tên file mã hoá cả size và chế độ tiền xử lý,
    nên các thực nghiệm ablation không ghi đè cache của nhau."""
    return Path(processed_dir) / f"images_{size}_{preprocess}.npy"


class GTSRBDataset(Dataset):
    """Một split (train/val/test) của GTSRB.

    Trả về (ảnh CHW float32 đã chuẩn hoá, nhãn int 0..42).
    """

    def __init__(self,
                 index_csv: str | Path = "data/processed/index.csv",
                 split: str = "train",
                 img_size: int = 48,
                 preprocess: str = "clahe",
                 normalize: str = "gtsrb",
                 mean: tuple[float, ...] | None = None,
                 std: tuple[float, ...] | None = None,
                 transform=None,
                 processed_dir: str | Path = "data/processed",
                 cache_size: int | None = None,
                 return_uint8: bool = False) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split phải là train/val/test, nhận được: {split!r}")

        self.split = split
        self.img_size = img_size
        self.preprocess = preprocess
        self.transform = transform
        self.return_uint8 = return_uint8   # True khi cần ảnh thô (Grad-CAM, robustness)

        # Ba bước khởi tạo, tách riêng cho dễ đọc và dễ tìm lỗi
        full_frame = self._load_index(index_csv, split)
        self._setup_normalizer(normalize, mean, std)
        self._load_cache(processed_dir, img_size, preprocess, cache_size, full_frame)

        self.class_names = CLASS_NAMES
        counts = self.frame["class_id"].value_counts().to_dict()
        self.class_counts = {c: int(counts.get(c, 0)) for c in range(NUM_CLASSES)}

    # ------------------------------------------------------------------
    # Ba bước khởi tạo
    # ------------------------------------------------------------------

    def _load_index(self, index_csv: str | Path, split: str) -> pd.DataFrame:
        """Đọc index.csv, lọc lấy split cần dùng. Trả về bảng ĐẦY ĐỦ (chưa lọc).

        Cần bảng đầy đủ vì file cache .npy được lưu theo thứ tự gốc của index.csv,
        nên muốn lấy ảnh thứ i của split này thì phải biết nó nằm ở hàng nào của cache.
        """
        index_csv = Path(index_csv)
        if not index_csv.exists():
            raise FileNotFoundError(
                f"Chưa có {index_csv}. Chạy trước: python scripts/prepare_data.py"
            )

        frame = pd.read_csv(index_csv)
        if "split" not in frame.columns:
            raise ValueError(
                f"{index_csv} chưa có cột 'split'. Chạy prepare_data.py để tạo split."
            )

        self.frame = frame[frame["split"] == split].reset_index(drop=True)
        if len(self.frame) == 0:
            raise ValueError(f"Split {split!r} rỗng trong {index_csv}")

        # Vị trí của từng dòng trong mảng cache (cache theo thứ tự gốc của index.csv)
        self._cache_rows = frame.index[frame["split"] == split].to_numpy()
        return frame

    def _setup_normalizer(self, normalize: str,
                          mean: tuple[float, ...] | None,
                          std: tuple[float, ...] | None) -> None:
        """Chọn thống kê chuẩn hoá.

        'imagenet' cho M3: trọng số pretrained và running stats của BatchNorm được
        học trên phân phối đó.
        'gtsrb' cho M1/M2: tính CHỈ trên tập train (tính trên cả bộ là rò rỉ thống kê).
        """
        self.normalize_mode = normalize
        if normalize == "imagenet":
            self.mean, self.std = IMAGENET_MEAN, IMAGENET_STD
        elif normalize == "gtsrb":
            if mean is None or std is None:
                raise ValueError(
                    "normalize='gtsrb' cần truyền mean/std (tính CHỈ trên tập train, "
                    "xem configs/base.yaml -> data.mean/data.std)"
                )
            self.mean, self.std = tuple(mean), tuple(std)
        else:
            raise ValueError(
                f"normalize phải là 'gtsrb' hoặc 'imagenet', nhận: {normalize!r}")

        self.normalizer = build_normalize(self.mean, self.std)

    def _load_cache(self, processed_dir: str | Path, img_size: int, preprocess: str,
                    cache_size: int | None, full_frame: pd.DataFrame) -> None:
        """Mở file cache .npy bằng mmap (các worker dùng chung page cache của OS).

        Ưu tiên cache đúng img_size; nếu không có thì dùng cache nhỏ hơn và nội suy
        trong __getitem__ — đúng trường hợp M3: cache 48, cần 224.
        """
        self.cache: np.ndarray | None = None
        self.cache_img_size: int | None = None

        sizes_to_try = [img_size] if cache_size is None else [cache_size, img_size]
        for size in dict.fromkeys(sizes_to_try):     # giữ thứ tự, bỏ trùng
            path = cache_path(processed_dir, size, preprocess)
            if path.exists():
                self.cache = np.load(path, mmap_mode="r")
                self.cache_img_size = size
                break

        if self.cache is None:
            # Thử bất kỳ cache nào cùng chế độ tiền xử lý
            candidates = sorted(Path(processed_dir).glob(f"images_*_{preprocess}.npy"))
            if candidates:
                self.cache = np.load(candidates[0], mmap_mode="r")
                self.cache_img_size = int(candidates[0].stem.split("_")[1])

        if self.cache is None:
            log.warning("[%s] KHÔNG có cache -> đọc ảnh từ đĩa mỗi lần (chậm hơn ~20 lần). "
                        "Chạy prepare_data.py để tạo cache.", self.split)
            return

        if len(self.cache) != len(full_frame):
            raise RuntimeError(
                f"Cache có {len(self.cache)} ảnh nhưng index.csv có {len(full_frame)} "
                f"dòng. Chạy lại prepare_data.py để dựng lại cache."
            )
        log.info("[%s] %d ảnh, cache %dx%d -> img_size %d, preprocess=%s, normalize=%s",
                 self.split, len(self.frame), self.cache_img_size, self.cache_img_size,
                 img_size, preprocess, self.normalize_mode)

    # ------------------------------------------------------------------
    # Giao diện Dataset
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.frame)

    def _read_uint8(self, i: int) -> np.ndarray:
        """Lấy ảnh HWC uint8 ở độ phân giải cache (hoặc đọc từ đĩa nếu không có cache)."""
        if self.cache is not None:
            # np.array(..., copy=True) vì cache mở bằng mmap_mode='r' là READ-ONLY,
            # torch.from_numpy trên mảng read-only sẽ cảnh báo. Copy 1 ảnh 48x48x3
            # chỉ là 6912 byte nên không đáng kể.
            return np.array(self.cache[self._cache_rows[i]], copy=True)
        row = self.frame.iloc[i]
        roi = (int(row.roi_x1), int(row.roi_y1), int(row.roi_x2), int(row.roi_y2))
        return load_and_process(str(row.path), roi, self.img_size, self.preprocess)

    def __getitem__(self, i: int):
        image = self._read_uint8(i)                       # HWC uint8
        label = int(self.frame.iloc[i]["class_id"])

        # HWC -> CHW. torchvision v2 làm việc trên tensor CHW.
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)

        # Augmentation (chỉ train). Làm ở ĐỘ PHÂN GIẢI CACHE để rẻ hơn,
        # rồi mới resize lên img_size.
        if self.transform is not None:
            tensor = self.transform(tensor)

        # Resize về img_size nếu cache khác size (trường hợp M3: 48 -> 224)
        if tensor.shape[-1] != self.img_size:
            tensor = F.interpolate(
                tensor.unsqueeze(0).float(),
                size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0).clamp(0, 255).to(torch.uint8)

        if self.return_uint8:
            return tensor, label                          # CHW uint8, cho Grad-CAM/robustness

        return self.normalizer(tensor), label             # CHW float32 đã chuẩn hoá


def build_dataloaders(cfg) -> dict[str, DataLoader]:
    """Dựng 3 DataLoader từ config. Trả {"train":..., "val":..., "test":...}.

    - train: shuffle=True, CÓ augmentation
    - val/test: shuffle=False, KHÔNG augmentation (phép đo phải lặp lại được)
    """
    data_cfg = cfg.data
    train_tf = build_transform(
        policy=data_cfg.aug_policy,
        img_size=data_cfg.img_size,
        rotation_deg=data_cfg.get("rotation_deg", 15.0),
        translate=data_cfg.get("translate", 0.10),
        zoom=tuple(data_cfg.get("zoom", (0.9, 1.1))),
        jitter=data_cfg.get("jitter", 0.20),
    )

    common = dict(
        index_csv=data_cfg.index_csv,
        img_size=data_cfg.img_size,
        preprocess=data_cfg.preprocess,
        normalize=data_cfg.normalize,
        mean=data_cfg.get("mean"),
        std=data_cfg.get("std"),
        processed_dir=data_cfg.get("processed_dir", "data/processed"),
        cache_size=data_cfg.get("cache_size"),
    )

    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        dataset = GTSRBDataset(
            split=split,
            transform=train_tf if split == "train" else None,
            **common,
        )
        # num_workers>0 trên macOS hay treo (fork + OpenCV). Mặc định 0 trong
        # configs/base.yaml, trên Colab thì đặt 2-4 để nhanh hơn.
        num_workers = int(cfg.train.get("num_workers", 0))
        loaders[split] = DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=(num_workers > 0),
            drop_last=(split == "train"),     # bỏ batch cuối lẻ để BatchNorm ổn định
            persistent_workers=(num_workers > 0),
            worker_init_fn=seed_worker if num_workers > 0 else None,
        )
    return loaders
