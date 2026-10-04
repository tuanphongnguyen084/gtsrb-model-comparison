#!/usr/bin/env python
"""
train.py — huấn luyện MỘT model theo một file config.

CHỦ: Hoàng

Ví dụ:
    python scripts/train.py --config configs/m1_lenet.yaml
    python scripts/train.py --config configs/m2_vggres.yaml
    python scripts/train.py --config configs/m3_resnet18.yaml

    # Chạy thử nhanh 2 epoch để kiểm đường ống không vỡ:
    python scripts/train.py --config configs/smoke.yaml

    # Override siêu tham số từ dòng lệnh (cho ablation, không cần sửa file):
    python scripts/train.py --config configs/m2_vggres.yaml \
        --set train.label_smoothing=0.0 --tag ls0 --subset 2000
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse

import torch

from gtsrb.data.dataset import build_dataloaders
from gtsrb.engine.train import fit
from gtsrb.models.registry import build_model, count_parameters, parameter_table
from gtsrb.utils.config import load_config
from gtsrb.utils.logging import get_logger
from gtsrb.utils.seed import pick_device, set_seed

log = get_logger()


def subset_loaders(loaders: dict, n: int) -> dict:
    """Thu nhỏ mọi loader về n mẫu — dùng cho smoke test, KHÔNG dùng để báo cáo.

    Giữ nguyên mọi tham số khác của DataLoader để smoke test đi qua đúng code path.
    """
    from torch.utils.data import DataLoader, Subset

    out = {}
    for name, loader in loaders.items():
        dataset = loader.dataset
        keep = min(n, len(dataset))
        # Lấy mẫu RẢI ĐỀU thay vì n mẫu đầu, để có đủ các lớp khác nhau
        # (index.csv sắp theo lớp nên n mẫu đầu toàn lớp 0).
        step = max(1, len(dataset) // keep)
        indices = list(range(0, len(dataset), step))[:keep]
        out[name] = DataLoader(
            Subset(dataset, indices),
            batch_size=loader.batch_size,
            shuffle=(name == "train"),
            num_workers=0,
            drop_last=False,
        )
    return out


def main() -> None:
    """Nạp config, dựng model + dataloader, gọi fit(), rồi in bảng tham số."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], dest="overrides",
                        help="ghi de sieu tham so, vd: train.epochs=3 data.img_size=64")
    parser.add_argument("--tag", default=None,
                        help="nhan them vao run_id, de phan biet cac run ablation")
    parser.add_argument("--subset", type=int, default=None,
                        help="chi dung N mau moi split (SMOKE TEST, khong de bao cao)")
    parser.add_argument("--device", default=None, help="cuda | mps | cpu | auto")
    parser.add_argument("--resume", default=None, metavar="RUN_DIR",
                        help="chay tiep tu last.pt cua mot run cu "
                             "(dung khi Colab mat session giua chung)")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    set_seed(int(cfg.get("seed", 42)), deterministic=True)
    device = pick_device(args.device or cfg.get("device", "auto"))

    # ---- Dựng model TRƯỚC, để lấy expected_img_size và normalize_mode của nó ----
    model = build_model(cfg.model.name, **dict(cfg.model.get("kwargs", {})),
                        img_size=cfg.data.img_size)

    # ★ Tự khớp cấu hình dataset với yêu cầu của model.
    # Nhờ bước này, không ai phải nhớ "M3 thì nhớ đổi sang 224 và ImageNet stats".
    # Quên chuyện đó là một LỖI IM LẶNG: model vẫn train, chỉ là kết quả tệ.
    if cfg.data.img_size != model.expected_img_size:
        log.warning("Model %s cần img_size=%d nhưng config ghi %d -> dùng %d",
                    model.model_name, model.expected_img_size,
                    cfg.data.img_size, model.expected_img_size)
        cfg.data.img_size = model.expected_img_size
    if cfg.data.normalize != model.normalize_mode:
        log.warning("Model %s cần normalize=%r nhưng config ghi %r -> dùng %r",
                    model.model_name, model.normalize_mode,
                    cfg.data.normalize, model.normalize_mode)
        cfg.data.normalize = model.normalize_mode

    log.info("Model %s: %.2fM tham số, img_size=%d, normalize=%s",
             model.model_name, count_parameters(model) / 1e6,
             model.expected_img_size, model.normalize_mode)

    loaders = build_dataloaders(cfg)
    if args.subset:
        log.warning("SMOKE TEST: chỉ dùng %d mẫu mỗi split. KHÔNG dùng số này "
                    "để báo cáo.", args.subset)
        loaders = subset_loaders(loaders, args.subset)

    outcome = fit(model, loaders, cfg, device, tag=args.tag,
                  resume_from=args.resume)

    # ---- Bảng đếm tham số theo lớp ----
    # Dùng để chứng minh luận điểm của M1: ~96% tham số ở MỘT lớp FC.
    table = parameter_table(model)
    out_csv = f"reports/tables/{model.model_name}_params.csv"
    import os
    os.makedirs("reports/tables", exist_ok=True)
    table.to_csv(out_csv, index=False)
    log.info("Bảng tham số -> %s", out_csv)
    log.info("5 lớp nhiều tham số nhất:")
    for row in table.head(5).itertuples(index=False):
        log.info("  %-28s %-14s %10d  (%5.1f%%)",
                 row.layer, row.type, row.params, row.percent)

    log.info("")
    log.info("KẾT QUẢ: val macro-F1 tốt nhất = %.4f (epoch %d)",
             outcome["best_val_macro_f1"] or 0.0, outcome["best_epoch"])
    log.info("Chạy tiếp để có số trên tập test:")
    log.info("  python scripts/evaluate.py --runs %s", outcome["run_dir"])


if __name__ == "__main__":
    main()
