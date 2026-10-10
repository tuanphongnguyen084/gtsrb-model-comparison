"""
common.py — phần dùng chung cho cả 4 file train/evaluate.

Chỉ chứa thứ KHÔNG phải logic deep learning cốt lõi: hằng số, đọc dữ liệu
(Dataset + DataLoader), và các hàm đo (macro-F1, McNemar). Model và vòng
huấn luyện của mỗi mô hình nằm trọn trong file train_mX.py tương ứng, để đọc
một file là thấy trọn kiến trúc + cách train của mô hình đó.

Vì sao tách ra đây: nếu chép Dataset + hàm metrics vào cả 4 file thì sửa một
chỗ phải sửa 4 lần, và ba model rất dễ nhận đầu vào khác nhau -> so sánh mất ý
nghĩa. Một bản dữ liệu duy nhất đi qua đây đảm bảo 3 model so sánh công bằng.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

# =====================================================================
# Hằng số dùng chung
# =====================================================================
NUM_CLASSES = 43

# Thống kê chuẩn hoá. gtsrb = tính CHỈ trên tập train của GTSRB (M1, M2).
# imagenet = dùng cho M3 vì backbone pretrained học trên phân phối đó.
GTSRB_MEAN = (0.4597, 0.4254, 0.4376)
GTSRB_STD = (0.2670, 0.2607, 0.2691)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 4 cặp lớp là ẢNH GƯƠNG của nhau -> CẤM lật ngang khi augment (lật lớp 33 "rẽ
# phải" ra đúng hình lớp 34 "rẽ trái" = tự tạo dữ liệu sai nhãn).
MIRROR_PAIRS = [(19, 20), (33, 34), (36, 37), (38, 39)]


# =====================================================================
# Dataset — cửa duy nhất để lấy dữ liệu
# =====================================================================
def _build_normalize(mean, std) -> v2.Transform:
    """uint8 [0,255] -> float32 đã chuẩn hoá. ToDtype(scale=True) tự chia 255."""
    return v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(mean), std=list(std)),
    ])


class GTSRBDataset(Dataset):
    """Một split (train/val/test) của GTSRB, đọc từ cache .npy đã tiền xử lý.

    Trả về (ảnh CHW float32 đã chuẩn hoá, nhãn int 0..42).

    Cache lưu ở 48x48 (xem preprocess.py). M3 cần 224x224 nên Dataset nội suy
    48 -> 224. Nội suy KHÔNG thêm thông tin; nó chỉ khớp SCALE và STRIDE mà
    backbone pretrained (downsample 32 lần) mong đợi — 224/32 = 7x7 là đúng cấu
    hình pretrain, còn 48/32 làm tròn 2x2 thì tầng cuối coi như vô dụng.
    """

    def __init__(self, index_csv: str | Path, split: str, img_size: int,
                 preprocess: str = "clahe", normalize: str = "gtsrb",
                 processed_dir: str | Path = "data/processed",
                 cache_size: int = 48, transform=None) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split phải là train/val/test, nhận: {split!r}")
        self.split = split
        self.img_size = img_size
        self.transform = transform

        # --- Đọc index, lọc split ---
        frame = pd.read_csv(index_csv)
        if "split" not in frame.columns:
            raise ValueError(f"{index_csv} chưa có cột 'split'. Chạy preprocess.py trước.")
        self.frame = frame[frame["split"] == split].reset_index(drop=True)
        # vị trí dòng trong cache (.npy lưu theo thứ tự gốc của index.csv)
        self._cache_rows = frame.index[frame["split"] == split].to_numpy()

        # --- Chọn thống kê chuẩn hoá ---
        if normalize == "imagenet":
            self.normalizer = _build_normalize(IMAGENET_MEAN, IMAGENET_STD)
        elif normalize == "gtsrb":
            self.normalizer = _build_normalize(GTSRB_MEAN, GTSRB_STD)
        else:
            raise ValueError(f"normalize phải là gtsrb/imagenet, nhận: {normalize!r}")

        # --- Mở cache .npy bằng mmap (worker dùng chung page cache của OS) ---
        path = Path(processed_dir) / f"images_{cache_size}_{preprocess}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Chưa có cache {path}. Chạy: python preprocess.py")
        self.cache = np.load(path, mmap_mode="r")
        self.cache_size = cache_size
        if len(self.cache) != len(frame):
            raise RuntimeError(
                f"Cache {len(self.cache)} ảnh != index.csv {len(frame)} dòng. "
                f"Chạy lại preprocess.py.")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, i: int):
        # copy=True vì mmap read-only; 1 ảnh 48x48x3 chỉ ~7KB nên không đáng kể
        image = np.array(self.cache[self._cache_rows[i]], copy=True)   # HWC uint8
        label = int(self.frame.iloc[i]["class_id"])

        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)  # CHW

        if self.transform is not None:        # augment (chỉ train)
            tensor = self.transform(tensor)

        if tensor.shape[-1] != self.img_size:  # nội suy lên img_size nếu cần (M3)
            tensor = F.interpolate(
                tensor.unsqueeze(0).float(),
                size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0).clamp(0, 255).to(torch.uint8)

        return self.normalizer(tensor), label


def make_loader(index_csv: str | Path, split: str, img_size: int, batch_size: int,
                normalize: str, transform=None, num_workers: int = 0) -> DataLoader:
    """Dựng DataLoader cho một split. train: shuffle + drop_last; val/test: không."""
    ds = GTSRBDataset(index_csv, split, img_size, normalize=normalize, transform=transform)
    is_train = split == "train"
    return DataLoader(ds, batch_size=batch_size, shuffle=is_train,
                      num_workers=num_workers, drop_last=is_train,
                      pin_memory=(num_workers > 0))


def build_train_transform(img_size: int) -> v2.Transform:
    """Augmentation geo_photo: xoay ±15°, dịch ±10%, zoom 0.9-1.1, jitter sáng/tương phản.

    ★ CẤM lật ngang (phá cặp gương) và CẤM đổi hue/saturation (màu mang nghĩa:
    đỏ=cấm, xanh=bắt buộc). Chỉ brightness/contrast là biến thiên chụp có thật.
    """
    return v2.Compose([
        v2.RandomAffine(degrees=15.0, translate=(0.10, 0.10), scale=(0.9, 1.1),
                        interpolation=v2.InterpolationMode.BILINEAR, fill=0),
        v2.ColorJitter(brightness=0.20, contrast=0.20),
    ])


# =====================================================================
# Đo lường
# =====================================================================
@torch.no_grad()
def predict(model: torch.nn.Module, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Chạy model trên loader, trả (y_true, y_prob). model.eval() tắt dropout/BN-train."""
    model.eval()
    probs, targets = [], []
    for images, y in loader:
        logits = model(images.to(device))
        probs.append(torch.softmax(logits.float(), dim=1).cpu())
        targets.append(y)
    return torch.cat(targets).numpy(), torch.cat(probs).numpy()


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """top-1, top-5, macro-F1, weighted-F1 + per-class từ (y_true, y_prob).

    ★ labels=range(43) BẮT BUỘC: thiếu nó thì macro-F1 chỉ tính trên các lớp CÓ
    MẶT trong batch -> các run thiếu lớp không so sánh được (lỗi im lặng).
    """
    y_pred = y_prob.argmax(axis=1)
    top1 = float((y_pred == y_true).mean())
    top5_idx = np.argsort(-y_prob, axis=1)[:, :5]
    top5 = float((top5_idx == y_true[:, None]).any(axis=1).mean())

    labels = list(range(NUM_CLASSES))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    per_class = pd.DataFrame({"class_id": labels, "precision": prec, "recall": rec,
                              "f1": f1, "support": support})
    return {"top1": top1, "top5": top5, "macro_f1": macro_f1,
            "weighted_f1": weighted_f1, "per_class": per_class,
            "y_true": y_true, "y_pred": y_pred}


def evaluate(model, loader, device) -> dict:
    """Tiện ích: predict + compute_metrics."""
    y_true, y_prob = predict(model, loader, device)
    return compute_metrics(y_true, y_prob)


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Kiểm định McNemar giữa hai model trên CÙNG tập test (quan sát bắt cặp).

    Chỉ hai ô BẤT ĐỒNG mang thông tin: n01 (chỉ A đúng), n10 (chỉ B đúng).
    discordant < 25 -> binomial chính xác; ngược lại -> chi2 có hiệu chỉnh Yates.
    Dùng t-test ở đây là SAI vì quan sát không độc lập.
    """
    from scipy import stats
    ca = pred_a == y_true
    cb = pred_b == y_true
    n01 = int(np.sum(ca & ~cb))
    n10 = int(np.sum(~ca & cb))
    discordant = n01 + n10
    if discordant == 0:
        p_value, method = 1.0, "identical"
    elif discordant < 25:
        p_value = float(stats.binomtest(min(n01, n10), discordant, 0.5).pvalue)
        method = "exact_binomial"
    else:
        chi2 = (abs(n01 - n10) - 1) ** 2 / discordant
        p_value = float(stats.chi2.sf(chi2, df=1))
        method = "chi2_corrected"
    better = "tie" if p_value >= 0.05 else ("A" if n01 > n10 else "B")
    return {"n01": n01, "n10": n10, "p_value": p_value, "method": method, "better": better}
