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


def find_run_dirs(patterns: list[str], main_only: bool = False) -> list[Path]:
    """Mở rộng các glob thành danh sách thư mục run ĐÃ TRAIN XONG.

    ★ Điều kiện là có `result.json`, KHÔNG phải `best.pt`.
    Lý do (lỗi đã gặp thật): `fit()` lưu best.pt sau MỖI epoch nhưng chỉ ghi
    result.json khi train xong hẳn. Một run bị kill giữa chừng vẫn có best.pt,
    và nếu chỉ lọc theo file đó thì nó lọt vào bảng kết quả với con số vô nghĩa —
    đã xảy ra: một run M1 bị kill ở epoch 3 lọt vào bảng nén với top-1 = 0,66
    thay vì 0,98, mà không có gì cảnh báo.

    `main_only=True` bỏ thêm các run có tag (ablation, seed, thí nghiệm rò rỉ) —
    dùng khi chỉ quan tâm 5 model chính.
    """
    dirs: list[Path] = []
    for pattern in patterns:
        dirs += [Path(p) for p in sorted(glob.glob(pattern)) if Path(p).is_dir()]

    complete, incomplete = [], []
    for d in dirs:
        (complete if (d / "result.json").exists() else incomplete).append(d)

    if incomplete:
        log.warning("Bỏ qua %d run TRAIN CHƯA XONG (có best.pt nhưng thiếu result.json):",
                    len(incomplete))
        for d in incomplete[:5]:
            log.warning("    %s", d.name)

    if main_only:
        tagged = [d for d in complete if _run_tag(d)]
        if tagged:
            log.info("Bỏ qua %d run phụ (ablation/seed/thí nghiệm): %s",
                     len(tagged), ", ".join(_run_tag(d) for d in tagged[:4]))
        complete = [d for d in complete if not _run_tag(d)]

    if not complete:
        raise SystemExit(
            f"Không tìm thấy run nào ĐÃ TRAIN XONG trong {patterns}.\n"
            f"Chạy trước: make train-m1  (hoặc make train-m2 / make train-m3)"
        )
    return complete


def _run_tag(run_dir: Path) -> str:
    """Tag của một run ('' nếu là run chính). Đọc từ result.json."""
    try:
        return json.loads((run_dir / "result.json").read_text(encoding="utf-8")
                          ).get("notes", "") or ""
    except Exception:
        return ""


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
