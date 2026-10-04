"""
_common — tiện ích dùng chung cho mọi script trong scripts/.

CHỦ: dùng chung (Hoàng bảo trì)

Trước đây 4 script (evaluate, benchmark_speed, run_robustness, make_gradcam) đều
tự viết lại cùng một đoạn: mở rộng glob -> lọc thư mục có best.pt -> nạp model.
Gom vào đây để sửa một chỗ là cả 4 cùng đúng.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import torch

from gtsrb.data.dataset import GTSRBDataset
from gtsrb.models.registry import build_model
from gtsrb.utils.config import load_config
from gtsrb.utils.logging import get_logger

log = get_logger()


def find_run_dirs(patterns: list[str]) -> list[Path]:
    """Mở rộng các glob thành danh sách thư mục run CÓ checkpoint.

    Bỏ qua thư mục chưa train xong (không có best.pt) thay vì vỡ giữa chừng.
    """
    dirs: list[Path] = []
    for pattern in patterns:
        dirs += [Path(p) for p in sorted(glob.glob(pattern)) if Path(p).is_dir()]

    with_ckpt = [d for d in dirs if (d / "best.pt").exists()]
    skipped = len(dirs) - len(with_ckpt)
    if skipped:
        log.warning("Bỏ qua %d thư mục chưa có best.pt (train chưa xong?)", skipped)
    if not with_ckpt:
        raise SystemExit(
            f"Không tìm thấy run nào có best.pt trong {patterns}.\n"
            f"Chạy trước: make train-m1  (hoặc make train-m2 / make train-m3)"
        )
    return with_ckpt


def load_run(run_dir: Path, device: torch.device):
    """Nạp lại model + config của một run. Trả (model, cfg, checkpoint_dict).

    Dựng model từ chính config đã lưu trong run, không phải từ config hiện tại —
    nếu không thì sửa configs/ hôm nay sẽ làm hỏng việc đọc lại run của hôm qua.
    """
    cfg = load_config(run_dir / "config.yaml")
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Không có best.pt trong {run_dir}")

    model = build_model(cfg.model.name, **dict(cfg.model.get("kwargs", {})),
                        img_size=cfg.data.img_size)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    return model.to(device).eval(), cfg, state


def make_dataset(cfg, split: str = "test", return_uint8: bool = False) -> GTSRBDataset:
    """Dựng dataset dùng ĐÚNG cấu hình dữ liệu của run đó.

    Quan trọng: M3 cần img_size=224 + normalize=imagenet. Dùng cấu hình của M1 để
    đánh giá M3 thì kết quả sai hoàn toàn mà KHÔNG báo lỗi.
    """
    return GTSRBDataset(
        index_csv=cfg.data.index_csv,
        split=split,
        img_size=cfg.data.img_size,
        preprocess=cfg.data.preprocess,
        normalize=cfg.data.normalize,
        mean=cfg.data.get("mean"),
        std=cfg.data.get("std"),
        processed_dir=cfg.data.get("processed_dir", "data/processed"),
        cache_size=cfg.data.get("cache_size"),
        return_uint8=return_uint8,
    )


def make_test_loader(cfg, batch_size: int = 256):
    """DataLoader cho tập test. shuffle=False để phép đo lặp lại được."""
    from torch.utils.data import DataLoader
    return DataLoader(make_dataset(cfg, "test"), batch_size=batch_size,
                      shuffle=False, num_workers=0)


def read_result(run_dir: Path) -> dict:
    """Đọc result.json của một run (trả {} nếu chưa có)."""
    path = run_dir / "result.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_result(run_dir: Path, result: dict) -> None:
    """Ghi lại result.json. Dùng khi evaluate/benchmark bổ sung mục `test`/`cost`."""
    (run_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def free_memory(model, device: torch.device) -> None:
    """Giải phóng model khỏi bộ nhớ thiết bị.

    Cần thiết khi lặp qua nhiều run: không giải phóng thì 5 model cùng nằm trên
    GPU và OOM ở run thứ 3-4, nhất là với M3 ở 224x224.
    """
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
