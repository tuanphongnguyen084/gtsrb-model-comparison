# INTERFACE CONTRACT — chốt ngày 0, không đổi tuỳ ý

> Đây là **hợp đồng** giữa 4 người. Nó tồn tại để 4 người code song song mà không chặn nhau.
> Muốn đổi một chữ ký hàm ở đây → **thông báo cả nhóm trước**, vì có người đang code dựa vào nó.
>
> Nguyên tắc: **ai sở hữu hàm thì người đó quyết định bên trong, nhưng không được đổi bên ngoài.**

## Quy ước chung

| Hạng mục | Quy ước |
|---|---|
| Tensor ảnh | `float32`, layout `CHW`, đã chuẩn hoá, giá trị khoảng `[-3, 3]` |
| Nhãn | `int64`, miền `0..42` |
| Ảnh numpy (dùng trong corruption/gradcam) | `uint8`, layout `HWC`, miền `[0, 255]` |
| Số lớp | `NUM_CLASSES = 43` — khai báo một lần ở `src/gtsrb/__init__.py`, không hard-code chỗ khác |
| Device | luôn nhận từ ngoài vào qua tham số `device`, **không** gọi `.cuda()` cứng trong `src/` |
| Siêu tham số | **mọi** siêu tham số nằm trong YAML; code chỉ đọc `cfg`, không hard-code |
| Random | mọi hàm có tính ngẫu nhiên phải nhận `seed` hoặc `rng` |
| Lỗi | raise `ValueError`/`FileNotFoundError` kèm giá trị sai trong thông điệp; **cấm** `except: pass` |
| Print | cấm `print` trong `src/`; dùng `logging`. Notebook thì tự do |

---

## 1. Dữ liệu — chủ: **Huy**

```python
# src/gtsrb/data/dataset.py
class GTSRBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        index_csv: str,                  # data/processed/index.csv
        split: str,                      # "train" | "val" | "test"
        img_size: int = 48,
        preprocess: str = "clahe",       # "none" | "he_gray" | "he_y" | "clahe"
        normalize: str = "gtsrb",        # "gtsrb" (M1, M2) | "imagenet" (M3)
        transform=None,                  # callable augmentation, chỉ cho split="train"
    ) -> None: ...

    def __len__(self) -> int: ...

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        """(ảnh CHW float32 đã chuẩn hoá, nhãn int 0..42)"""

    # thuộc tính bắt buộc, D cần để phân tích per-class
    class_counts: dict[int, int]         # {class_id: số ảnh trong split này}
    class_names:  list[str]              # 43 tên tiếng Việt, index = class_id
```

```python
# src/gtsrb/data/dataset.py
def build_dataloaders(cfg) -> dict[str, torch.utils.data.DataLoader]:
    """Trả về {"train": ..., "val": ..., "test": ...}.
    train: shuffle=True + transform; val/test: shuffle=False, không transform."""
```

```python
# src/gtsrb/data/split.py
def make_split(index_csv: str, val_ratio: float = 0.2, seed: int = 42,
               group_by_track: bool = True) -> None:
    """Ghi cột 'split' vào index_csv.
    group_by_track=True  -> StratifiedGroupKFold theo (class_id, track_id)  [MẶC ĐỊNH]
    group_by_track=False -> split random theo ảnh  [CHỈ để làm đối chứng rò rỉ]
    """
```

```python
# src/gtsrb/data/preprocess.py
def apply_preprocess(img: np.ndarray, mode: str) -> np.ndarray:
    """img: HWC uint8 BGR/RGB -> HWC uint8. mode in {none, he_gray, he_y, clahe}."""

# src/gtsrb/data/transforms.py
def build_transform(policy: str, img_size: int, seed: int | None = None):
    """policy in {none, geo, geo_photo, geo_photo_clahe}.
    CẤM RandomHorizontalFlip và hue jitter — xem docs/PHAN_CONG.md mục A6."""
```

**Format `data/processed/index.csv`** (A giao, mọi người đọc):

| cột | kiểu | ví dụ | ghi chú |
|---|---|---|---|
| `path` | str | `data/raw/Train/00002/00005_00017.ppm` | tương đối từ gốc repo |
| `class_id` | int | `2` | 0..42 |
| `track_id` | int | `5` | **khoá chống rò rỉ** |
| `frame_id` | int | `17` | |
| `roi_x1,roi_y1,roi_x2,roi_y2` | int | `5,6,43,44` | từ CSV gốc của GTSRB |
| `width,height` | int | `48,50` | kích thước ảnh gốc |
| `source` | str | `train` \| `test_official` | |
| `split` | str | `train` \| `val` \| `test` | do `make_split` ghi vào |

---

## 2. Model — chủ: **Hoàng** (registry, m1, m2) / **C** (m3)

```python
# src/gtsrb/models/registry.py
def build_model(name: str, num_classes: int = 43, **kw) -> torch.nn.Module:
    """name in {"m1_lenet", "m2_vggres",
                "m3_resnet18", "m3_mobilenetv2", "m3_effnetb0"}

    HỢP ĐỒNG BẮT BUỘC — mọi model trả về PHẢI có:
      model.gradcam_target_layer -> nn.Module   # C cần, không có là C tắc
      model.model_name           -> str
      model.expected_img_size    -> int          # 48 cho M1/M2, 224 cho M3
      model.normalize_mode       -> str          # "gtsrb" | "imagenet"
    """
```

Vì sao có `expected_img_size` và `normalize_mode` trên model: để `scripts/train.py`
tự chọn đúng cấu hình dataset, không ai phải nhớ "M3 thì nhớ đổi sang 224 và ImageNet stats".
Quên chuyện đó là lỗi **im lặng** — model vẫn train, chỉ là kết quả tệ mà không ai biết vì sao.

---

## 3. Huấn luyện — chủ: **Hoàng**

```python
# src/gtsrb/engine/train.py
def fit(model: torch.nn.Module,
        loaders: dict[str, DataLoader],
        cfg,
        device: torch.device) -> dict:
    """Trả về:
      {"run_id": str,
       "best_ckpt": str,                 # đường dẫn .pt
       "history": pandas.DataFrame,      # 1 dòng / epoch
       "best_val_macro_f1": float,
       "best_epoch": int,
       "train_seconds": float}

    Tác dụng phụ bắt buộc:
      artifacts/runs/<run_id>/best.pt
      artifacts/runs/<run_id>/train_log.csv
      artifacts/runs/<run_id>/config.yaml       # bản sao cfg thực tế đã dùng
      artifacts/runs/<run_id>/result.json       # theo schema mục 6
    """
```

Dùng chung cho cả M1, M2, M3. **C không được viết vòng train riêng** — nếu M3 cần
2 pha freeze/unfreeze thì C đưa vào `cfg.train.phases` và B hỗ trợ `fit` đọc nó.
Lý do: hai vòng train khác nhau thì so sánh 3 model không còn công bằng.

---

## 4. Đánh giá — chủ: **Phong Nguyễn**

```python
# src/gtsrb/eval/metrics.py
def evaluate(model, loader, device) -> dict:
    """Trả về:
      {"top1": float, "top5": float, "macro_f1": float, "weighted_f1": float,
       "per_class": pandas.DataFrame,    # 43 dòng: class_id, name, precision, recall, f1, support
       "y_true": np.ndarray,             # (N,) int
       "y_pred": np.ndarray,             # (N,) int
       "y_prob": np.ndarray}             # (N, 43) float — D cần cho ECE, C cần cho Grad-CAM
    """

# src/gtsrb/eval/stats_tests.py
def mcnemar(y_true, y_pred_a, y_pred_b) -> dict:
    """{"n01": int, "n10": int, "statistic": float, "p_value": float, "method": str}
    method = "chi2_corrected" nếu n01+n10 >= 25, ngược lại "exact_binomial"."""

# src/gtsrb/eval/calibration.py
def expected_calibration_error(y_true, y_prob, n_bins: int = 15) -> dict:
    """{"ece": float, "bins": DataFrame}  # bins: conf_mean, acc, count"""
```

---

## 5. Robustness — chủ: **Phong Nguyễn**

```python
# src/gtsrb/robustness/corruptions.py
def apply_corruption(img: np.ndarray, kind: str, severity: int,
                     rng: np.random.Generator) -> np.ndarray:
    """img: HWC uint8 -> HWC uint8 (CÙNG shape, CÙNG dtype).
    kind in {"motion_blur", "gauss_noise", "fog", "low_light", "occlusion"}
    severity in 1..5 (1 = nhẹ, 5 = nặng)

    BẤT BIẾN (có test): severity=0 phải trả về ảnh y nguyên;
    output luôn clip về [0,255] uint8; cùng rng -> cùng kết quả."""

CORRUPTIONS: list[str]   # 5 tên, để script lặp
SEVERITIES:  list[int]   # [1,2,3,4,5]
```

Nhiễu **chỉ dùng ở test**. `build_transform` của A không bao giờ gọi `apply_corruption`.

---

## 6. Schema `result.json` — **mọi run đều phải ghi đúng cái này**

D viết script tổng hợp bằng cách glob `artifacts/runs/*/result.json`.
Sai schema = run đó bị bỏ khỏi báo cáo.

```json
{
  "run_id": "m2_vggres_s42_20260310_1432",
  "model": "m2_vggres",
  "seed": 42,
  "git_commit": "a1b2c3d",
  "config_path": "configs/m2_vggres.yaml",
  "data": {"img_size": 48, "preprocess": "clahe", "aug_policy": "geo_photo",
           "split_group_by_track": true, "n_train": 31367, "n_val": 7842, "n_test": 12630},
  "train": {"epochs_run": 42, "best_epoch": 35, "label_smoothing": 0.1,
            "optimizer": "adam", "lr": 0.001, "scheduler": "cosine_warmup3",
            "batch_size": 128, "train_seconds": 1523.4, "device": "cuda:T4"},
  "val":  {"top1": 0.9941, "best_macro_f1": 0.9917},
  "test": {"top1": 0.9936, "top5": 0.9999, "macro_f1": 0.9905,
           "weighted_f1": 0.9936, "ece": 0.0535,
           "per_class_f1": [0.99, 0.99, "... đủ 43 số ..."]},
  "cost": {"img_size": 48, "params_m": 4.51, "flops_g": 0.42, "size_mb": 17.2,
           "cpu_bs1_p50": 8.1, "cpu_bs1_p95": 11.4, "cpu_bs1_mean": 8.4,
           "cpu_bs1_imgs_per_sec": 123.5,
           "cpu_bs64_p50": 21.0, "mps_bs1_p50": 1.9, "mps_bs64_p50": 9.2,
           "environment": {"torch": "2.x", "device": "mps"}},
  "ckpt_path": "artifacts/runs/m2_vggres_s42_20260310_1432/best.pt",
  "notes": "ablation: width x1, depth 4 stage"
}
```

Khoá nào chưa chạy thì để `null`, **không bỏ khoá**. `tests/test_schema.py` kiểm
việc này trên MỌI `result.json` có trong `artifacts/runs/`.

**Hai chỗ dễ viết sai, đã từng viết sai trong chính tài liệu này:**

| Sai | Đúng | Vì sao |
|---|---|---|
| `val.macro_f1`, `val.top5` | **`val.best_macro_f1`** | `val` chỉ ghi `top1` và `best_macro_f1` (giá trị TỐT NHẤT qua các epoch, do early stopping theo nó). Không script nào đọc `val.macro_f1`. |
| `cost.latency_ms.cpu_bs1_p50` | **`cost.cpu_bs1_p50`** | latency nằm ở khoá **phẳng**, không lồng. `benchmark()` trả dict phẳng dạng `{device}_bs{N}_{p50\|p95\|mean\|imgs_per_sec}`. |

Hai lỗi này tồn tại trong tài liệu suốt dự án mà không ai phát hiện, vì
`tests/test_schema.py` được HỨA trong mục này nhưng chưa bao giờ được viết. Soát
lại 39 run thật thì `val.top5` và `val.macro_f1` thiếu ở **39/39 run** —
nghĩa là tài liệu sai, không phải code sai.

---

## 7. Giải thích & Tốc độ — chủ: **Phong Trần**

```python
# src/gtsrb/explain/gradcam.py
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        """target_layer=None -> dùng model.gradcam_target_layer"""

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """x: (1,C,H,W). class_idx=None -> dùng lớp được dự đoán.
        Trả về heatmap (H, W) float trong [0,1], đã upsample về kích thước đầu vào."""

    def overlay(self, img_uint8: np.ndarray, heatmap: np.ndarray,
                alpha: float = 0.4) -> np.ndarray:
        """(HWC uint8, HW float) -> HWC uint8 đã phủ colormap"""

# src/gtsrb/deploy/speed.py
def benchmark(model, img_size: int, device, batch_sizes=(1, 64),
              warmup: int = 20, iters: int = 100) -> dict:
    """{"<device>_bs<N>_p50": ms, "<device>_bs<N>_p95": ms, ...}
    BẮT BUỘC: eval() + no_grad(); đồng bộ thiết bị trước & sau khi bấm giờ."""

def model_cost(model, img_size: int) -> dict:
    """{"params_m": float, "flops_g": float, "size_mb": float}"""
```

---

## 8. Ai gọi được gì — bảng phụ thuộc

| Người | Được import từ | Không bao giờ sửa |
|---|---|---|
| **Huy** | `utils/` | `models/`, `engine/`, `eval/`, `explain/`, `deploy/` |
| **Hoàng** | `data/`, `utils/` | `data/`, `eval/`, `robustness/`, `explain/`, `deploy/` |
| **Phong Trần** | `data/`, `models/registry`, `engine/`, `utils/` | `data/`, `engine/`, `eval/`, `robustness/` |
| **Phong Nguyễn** | `data/`, `models/registry`, `utils/` | `data/`, `models/m2,m3`, `engine/`, `explain/`, `deploy/` |

Thấy bug trong file của người khác → **nhắn người đó**, không tự sửa.
Hai người sửa cùng một file là cách nhanh nhất để hỏng repo trước deadline.
