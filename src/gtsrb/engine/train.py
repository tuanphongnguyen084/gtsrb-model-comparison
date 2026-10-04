"""
train — vòng huấn luyện dùng chung cho CẢ M1, M2, M3.

CHỦ: Hoàng

★ VÌ SAO MỘT HÀM fit() CHO CẢ BA MODEL ★
Nếu mỗi người viết vòng train riêng thì 3 model được huấn luyện theo 3 cách khác nhau,
và bảng so sánh cuối cùng MẤT Ý NGHĨA — không biết chênh lệch đến từ kiến trúc hay từ
cách train. Vì vậy đây là hợp đồng cứng (docs/INTERFACE.md mục 3): nhu cầu 2 pha
freeze/unfreeze của M3 được đưa vào `cfg.train.phases` và fit() đọc nó.

CÁC THÀNH PHẦN VÀ LÝ DO:
  loss        CrossEntropy + label smoothing 0,1   chống tự tin thái quá
  optimizer   Adam / AdamW                          ít phải tinh chỉnh LR
  scheduler   warmup 3 epoch rồi cosine             xem engine/schedulers.py
  AMP         bật trên CUDA, TẮT trên MPS           MPS chưa hỗ trợ đầy đủ
  grad clip   norm 1,0                              chặn gradient explode (M1 không có BN)
  early stop  theo val MACRO-F1, patience 10        KHÔNG theo accuracy
  checkpoint  lưu best + last                       Colab hay mất session
  seed        cố định, cudnn.deterministic          chạy lại ra cùng số

BỐ CỤC FILE (đọc từ dưới lên nếu muốn hiểu tổng thể trước):
  Phần 1  tiện ích        _git_commit, make_run_id
  Phần 2  một epoch       train_one_epoch
  Phần 3  cấu hình pha    _resolve_phases, _phase_config
  Phần 4  trạng thái      _TrainState
  Phần 5  chạy một pha    _run_one_phase, _setup_phase_optimizer
  Phần 6  ghi kết quả     _build_result
  Phần 7  API chính       fit   <- bắt đầu đọc ở đây
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from gtsrb.engine.callbacks import CheckpointSaver, CsvLogger, EarlyStopping
from gtsrb.engine.losses import build_criterion
from gtsrb.engine.schedulers import build_optimizer, build_scheduler
from gtsrb.eval.metrics import evaluate
from gtsrb.models.registry import count_parameters
from gtsrb.utils.config import save_config
from gtsrb.utils.logging import get_logger

log = get_logger()


# =====================================================================
# Phần 1 — Tiện ích
# =====================================================================

def _git_commit() -> str:
    """Lấy git commit hiện tại để ghi vào result.json (phục vụ tái lập)."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "no-git"
    except Exception:
        return "no-git"


def make_run_id(model_name: str, seed: int, tag: str | None = None) -> str:
    """run_id duy nhất cho mỗi lần chạy: tên model + seed + (tag) + thời gian.

    Ví dụ: 'm2_vggres_s42_scaling-width2.0_20261003_224906'
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [model_name, f"s{seed}", stamp]
    if tag:
        parts.insert(2, tag)
    return "_".join(parts)


# =====================================================================
# Phần 2 — Huấn luyện một epoch
# =====================================================================

def train_one_epoch(model: nn.Module, loader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    scheduler=None, scaler=None, grad_clip: float = 1.0,
                    use_amp: bool = False, desc: str = "") -> dict:
    """Train đúng một epoch. Trả {"loss": ..., "acc": ...} trên tập train."""
    model.train()                 # BẬT dropout; BatchNorm dùng thống kê của batch
    total_loss, total_correct, total_seen = 0.0, 0, 0

    amp_device = device.type if device.type in ("cuda", "cpu") else "cpu"
    # disable khi không chạy trong terminal thật -> log ghi ra file không bị tràn
    # hàng nghìn dòng progress bar.
    progress = tqdm(loader, desc=desc, leave=False, ncols=88,
                    disable=not sys.stderr.isatty())

    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # set_to_none=True nhanh hơn và tốn ít bộ nhớ hơn zero_grad() thường
        optimizer.zero_grad(set_to_none=True)

        # --- Lượt thuận ---
        with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        # --- Lượt nghịch + cập nhật ---
        _backward_and_step(loss, model, optimizer, scaler, grad_clip)

        if scheduler is not None:
            scheduler.step()      # cập nhật LR theo TỪNG BƯỚC, không theo epoch

        # --- Thống kê ---
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(1) == targets).sum().item()
        total_seen += batch_size
        progress.set_postfix(loss=f"{total_loss / total_seen:.4f}",
                             acc=f"{total_correct / total_seen:.4f}")

    return {"loss": total_loss / max(1, total_seen),
            "acc": total_correct / max(1, total_seen)}


def _backward_and_step(loss: torch.Tensor, model: nn.Module,
                       optimizer: torch.optim.Optimizer, scaler, grad_clip: float) -> None:
    """Lan truyền ngược, cắt gradient, rồi cập nhật trọng số.

    Tách riêng vì nhánh mixed-precision có thứ tự KHÁC nhánh thường và dễ làm sai:
    phải `unscale_` TRƯỚC khi clip, nếu không ta đang clip gradient đã bị nhân hệ số.
    """
    if scaler is not None and scaler.is_enabled():
        # Mixed precision: nhân loss lên để gradient fp16 không underflow về 0
        scaler.scale(loss).backward()
        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)        # <- phải unscale TRƯỚC khi clip
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()


# =====================================================================
# Phần 3 — Cấu hình các pha huấn luyện
# =====================================================================

def _resolve_phases(cfg) -> list[dict]:
    """Chuẩn hoá cfg.train.phases thành danh sách pha.

    M1/M2 không khai báo `phases` -> tạo MỘT pha duy nhất từ cfg.train.epochs.
    M3 khai báo 2 pha: đóng băng backbone (train head) rồi mở băng toàn bộ.
    """
    phases = cfg.train.get("phases")
    if not phases:
        return [{"name": "main",
                 "epochs": int(cfg.train.epochs),
                 "freeze_backbone": False,
                 "lr": None,
                 "discriminative_lr": None}]

    out = []
    for i, phase in enumerate(phases):
        out.append({
            "name": phase.get("name", f"phase{i + 1}"),
            "epochs": int(phase.get("epochs", cfg.train.epochs)),
            "freeze_backbone": bool(phase.get("freeze_backbone", False)),
            "lr": phase.get("lr"),
            "discriminative_lr": phase.get("discriminative_lr"),
        })
    return out


def _phase_config(cfg, phase: dict):
    """Trả bản sao cfg với lr / discriminative_lr của riêng pha này.

    Vì sao cần: pha 1 của M3 chỉ train head nên dùng MỘT lr; pha 2 mở băng và dùng
    discriminative LR (3 nhóm). Nếu dùng chung một cfg thì pha 1 sẽ vô tình tạo
    param_groups cho cả backbone đang đóng băng.
    """
    if phase.get("lr") is None and phase.get("discriminative_lr") is None:
        return cfg          # không có gì riêng -> dùng luôn cfg gốc, khỏi copy

    out = copy.deepcopy(cfg)
    if phase.get("lr") is not None:
        out.train.lr = phase["lr"]
    # Gán None sẽ khiến build_optimizer bỏ qua discriminative LR
    out.train["discriminative_lr"] = phase.get("discriminative_lr")
    return out


# =====================================================================
# Phần 4 — Trạng thái dùng chung giữa các pha
# =====================================================================

@dataclass
class _TrainState:
    """Gom các đối tượng sống XUYÊN SUỐT nhiều pha vào một chỗ.

    Nếu để rời rạc thì fit() phải truyền 6 biến vào từng hàm con và rất dễ quên
    một cái (ví dụ quên truyền `stopper` thì early stopping im lặng không hoạt động).
    """
    saver: CheckpointSaver
    csv_logger: CsvLogger
    stopper: EarlyStopping
    criterion: nn.Module
    scaler: torch.amp.GradScaler
    use_amp: bool
    device: torch.device
    total_epochs: int
    seed: int
    run_id: str
    epoch_global: int = 0
    history: list[dict] = field(default_factory=list)
    # --- phục vụ --resume ---
    phase_index: int = 0        # pha đang chạy
    skip_until: int = 0         # bỏ qua các epoch <= số này (đã train ở lần trước)
    resume_optimizer: dict | None = None
    resume_scheduler: dict | None = None


# =====================================================================
# Phần 5 — Chạy một pha
# =====================================================================

def _setup_phase_optimizer(model: nn.Module, cfg, phase: dict, loader_len: int):
    """Đóng/mở băng backbone, rồi dựng optimizer + scheduler cho pha này."""
    # --- Đóng băng / mở băng (chỉ M3 có hàm set_backbone_frozen) ---
    if hasattr(model, "set_backbone_frozen"):
        model.set_backbone_frozen(phase["freeze_backbone"])
        trainable = count_parameters(model, only_trainable=True)
        log.info("[pha %s] backbone %s -> %.2fM/%.2fM tham số được train",
                 phase["name"],
                 "ĐÓNG BĂNG" if phase["freeze_backbone"] else "mở",
                 trainable / 1e6, count_parameters(model) / 1e6)

    # --- Optimizer riêng cho pha ---
    phase_cfg = _phase_config(cfg, phase)
    optimizer = build_optimizer(model, phase_cfg)
    lrs = [f"{g['lr']:.2e}" for g in optimizer.param_groups]
    log.info("[pha %s] optimizer=%s  lr theo nhóm = %s",
             phase["name"], type(optimizer).__name__, lrs)

    scheduler = build_scheduler(optimizer, phase_cfg,
                               steps_per_epoch=loader_len,
                               total_epochs=phase["epochs"])
    return optimizer, scheduler


def _run_one_phase(model: nn.Module, loaders: dict, cfg, phase: dict,
                   state: _TrainState) -> None:
    """Chạy trọn một pha. Cập nhật state tại chỗ. Dừng sớm nếu early stopping bật."""
    optimizer, scheduler = _setup_phase_optimizer(
        model, cfg, phase, loader_len=len(loaders["train"]))
    grad_clip = float(cfg.train.get("grad_clip", 1.0))

    # --resume: khôi phục moment của optimizer và vị trí của scheduler
    if state.resume_optimizer is not None:
        optimizer.load_state_dict(state.resume_optimizer)
        if scheduler is not None and state.resume_scheduler is not None:
            scheduler.load_state_dict(state.resume_scheduler)
        log.info("[pha %s] đã khôi phục trạng thái optimizer/scheduler từ last.pt",
                 phase["name"])
        state.resume_optimizer = state.resume_scheduler = None   # chỉ dùng một lần

    for _ in range(phase["epochs"]):
        state.epoch_global += 1

        # --resume: các epoch đã train ở lần chạy trước thì bỏ qua
        if state.epoch_global <= state.skip_until:
            continue

        started = time.time()

        # --- Train ---
        train_stats = train_one_epoch(
            model, loaders["train"], state.criterion, optimizer, state.device,
            scheduler=scheduler, scaler=state.scaler, grad_clip=grad_clip,
            use_amp=state.use_amp,
            desc=f"epoch {state.epoch_global}/{state.total_epochs} [{phase['name']}]",
        )

        # --- Đánh giá trên val ---
        val_stats = evaluate(model, loaders["val"], state.device, use_amp=state.use_amp)

        # --- Early stopping theo val MACRO-F1 (không theo accuracy) ---
        is_best = state.stopper.step(val_stats["macro_f1"], state.epoch_global)

        # --- Ghi log + checkpoint ---
        row = {
            "epoch": state.epoch_global,
            "phase": phase["name"],
            "lr": optimizer.param_groups[-1]["lr"],
            "train_loss": round(train_stats["loss"], 6),
            "train_acc": round(train_stats["acc"], 6),
            "val_top1": round(val_stats["top1"], 6),
            "val_top5": round(val_stats["top5"], 6),
            "val_macro_f1": round(val_stats["macro_f1"], 6),
            "seconds": round(time.time() - started, 2),
            "is_best": int(is_best),
        }
        state.history.append(row)
        state.csv_logger.log(row)
        state.saver.save(
            model, state.epoch_global,
            {k: row[k] for k in ("val_top1", "val_top5", "val_macro_f1")},
            is_best=is_best,
            extra={"run_id": state.run_id, "seed": state.seed},
            optimizer=optimizer, scheduler=scheduler,
            phase_index=state.phase_index,
            stopper_state=state.stopper.state_dict(),
        )

        log.info("epoch %2d/%d | train loss %.4f acc %.4f | "
                 "val top1 %.4f macroF1 %.4f | %5.1fs %s",
                 state.epoch_global, state.total_epochs,
                 train_stats["loss"], train_stats["acc"],
                 val_stats["top1"], val_stats["macro_f1"], row["seconds"],
                 "<- BEST" if is_best else "")

        if state.stopper.should_stop:
            log.info("Early stopping: val macro-F1 không cải thiện sau %d epoch "
                     "(tốt nhất %.4f ở epoch %d)",
                     state.stopper.patience, state.stopper.best, state.stopper.best_epoch)
            return


def _restore_from_checkpoint(model: nn.Module, state: _TrainState,
                             resume_dir: Path, device: torch.device) -> None:
    """Khôi phục model + early stopping + optimizer từ last.pt của một run cũ.

    Dùng khi Colab mất session giữa chừng: thay vì train lại từ đầu (có thể mất
    cả giờ với M3 ở 224x224), ta chạy tiếp từ epoch đã dừng.

        python scripts/train.py --config configs/m3_resnet18.yaml \
            --resume artifacts/runs/m3_resnet18_s42_20261003_225740

    LƯU Ý: resume KHÔNG hoàn toàn bằng chạy liền một mạch, vì thứ tự shuffle của
    DataLoader khác đi. Chênh lệch nhỏ và không ảnh hưởng kết luận, nhưng khi báo
    cáo con số CHÍNH THỨC thì nên train liền một mạch.
    """
    last_path = Path(resume_dir) / "last.pt"
    if not last_path.exists():
        raise FileNotFoundError(
            f"Không có {last_path} để resume. Kiểm tra lại đường dẫn run."
        )

    checkpoint = torch.load(last_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    state.epoch_global = 0                              # đếm lại từ đầu...
    state.skip_until = int(checkpoint["epoch"])         # ...nhưng bỏ qua tới đây
    state.phase_index = int(checkpoint.get("phase_index", 0))
    state.resume_optimizer = checkpoint.get("optimizer_state")
    state.resume_scheduler = checkpoint.get("scheduler_state")

    if checkpoint.get("stopper_state"):
        state.stopper.load_state_dict(checkpoint["stopper_state"])

    log.info("RESUME từ %s: đã train %d epoch, val macro-F1 tốt nhất %.4f",
             resume_dir, state.skip_until, state.stopper.best or 0.0)


# =====================================================================
# Phần 6 — Ghi kết quả ra result.json
# =====================================================================

def _detect_sleep(history: list[dict]) -> dict:
    """Phát hiện máy đã NGỦ giữa lúc train, dựa vào epoch có thời gian bất thường.

    ★ LỖI ĐÃ GẶP THẬT: laptop chạy pin ngủ lúc 01:34, thức lúc 10:05. Epoch đang dở
    bị ghi nhận là **31.297 giây** (8,7 giờ) thay vì ~250 giây. `train_seconds` của
    run đó thành vô nghĩa, và nếu đưa thẳng vào bảng báo cáo thì sai nghiêm trọng —
    "M3 train 9 tiếng" là con số không ai kiểm chứng lại được.

    Không tự sửa số, chỉ ĐÁNH DẤU. Sửa ngầm còn tệ hơn: người đọc báo cáo sẽ không
    biết số đã bị can thiệp. Đánh dấu thì scripts/make_report.py bỏ qua được và
    người đọc biết vì sao.

    Trả: {"suspect": bool, "clean_seconds": float, "note": str}
    """
    times = [row["seconds"] for row in history if row.get("seconds")]
    if len(times) < 3:
        return {"suspect": False, "clean_seconds": sum(times), "note": ""}

    ordered = sorted(times)
    median = ordered[len(ordered) // 2]
    outliers = [t for t in times if t > 5 * median]

    if not outliers:
        return {"suspect": False, "clean_seconds": sum(times), "note": ""}

    # Thay mỗi epoch bất thường bằng trung vị -> ước lượng thời gian train THẬT
    clean = sum(median if t > 5 * median else t for t in times)
    note = (f"{len(outliers)} epoch có thời gian bất thường "
            f"(lớn nhất {max(outliers):.0f}s so với trung vị {median:.0f}s) — "
            f"gần như chắc chắn máy đã NGỦ giữa chừng. "
            f"train_seconds thô không dùng để báo cáo được; "
            f"dùng train_seconds_clean ({clean:.0f}s).")
    log.warning("=" * 70)
    log.warning("CHÚ Ý: %s", note)
    log.warning("Lần sau: cắm sạc VÀ mở nắp máy, hoặc chạy `caffeinate -dimsu &`.")
    log.warning("(caffeinate CHỈ chặn ngủ khi đang cắm sạc — chạy pin thì vô tác dụng.)")
    log.warning("=" * 70)
    return {"suspect": True, "clean_seconds": clean, "note": note}


def _build_result(model_name: str, cfg, loaders: dict, state: _TrainState,
                  train_seconds: float, ckpt_path: Path, tag: str | None) -> dict:
    """Dựng dict theo schema docs/INTERFACE.md mục 6.

    Schema này là hợp đồng với Phong Nguyễn (bảng so sánh) và Huy (robustness):
    họ glob `artifacts/runs/*/result.json` để tổng hợp tự động. Sai schema =
    run đó bị bỏ khỏi báo cáo.
    """
    sleep_info = _detect_sleep(state.history)
    return {
        "run_id": state.run_id,
        "model": model_name,
        "seed": state.seed,
        "git_commit": _git_commit(),
        "config_path": str(cfg.get("config_path", "")),
        "data": {
            "img_size": cfg.data.img_size,
            "preprocess": cfg.data.preprocess,
            "aug_policy": cfg.data.aug_policy,
            "normalize": cfg.data.normalize,
            "split_group_by_track": bool(cfg.data.get("group_by_track", True)),
            "n_train": len(loaders["train"].dataset),
            "n_val": len(loaders["val"].dataset),
            "n_test": len(loaders["test"].dataset),
        },
        "train": {
            "epochs_run": state.epoch_global,
            "best_epoch": state.stopper.best_epoch,
            "label_smoothing": cfg.train.get("label_smoothing", 0.1),
            "optimizer": cfg.train.get("optimizer", "adam"),
            "lr": cfg.train.lr,
            "scheduler": cfg.train.get("scheduler", "cosine"),
            "batch_size": cfg.train.batch_size,
            "train_seconds": round(train_seconds, 1),
            # Nếu máy ngủ giữa chừng thì train_seconds vô nghĩa — xem _detect_sleep()
            "train_seconds_clean": round(sleep_info["clean_seconds"], 1),
            "train_seconds_suspect": sleep_info["suspect"],
            "train_time_note": sleep_info["note"],
            "device": str(state.device),
            "amp": state.use_amp,
        },
        "val": {
            "top1": state.history[-1]["val_top1"] if state.history else None,
            "best_macro_f1": state.stopper.best,
        },
        "test": None,        # scripts/evaluate.py điền
        "cost": None,        # scripts/benchmark_speed.py điền
        "ckpt_path": str(ckpt_path),
        "notes": tag or "",
    }


# =====================================================================
# Phần 7 — API chính
# =====================================================================

def fit(model: nn.Module, loaders: dict, cfg, device: torch.device,
        run_dir: str | Path | None = None, tag: str | None = None,
        resume_from: str | Path | None = None) -> dict:
    """Huấn luyện model. Trả dict theo hợp đồng docs/INTERFACE.md mục 3.

    Tác dụng phụ (bắt buộc — cả nhóm phụ thuộc vào chúng để tổng hợp tự động):
        artifacts/runs/<run_id>/best.pt
        artifacts/runs/<run_id>/last.pt
        artifacts/runs/<run_id>/train_log.csv
        artifacts/runs/<run_id>/config.yaml      cấu hình THỰC TẾ đã dùng
        artifacts/runs/<run_id>/result.json
    """
    # ---------- Chuẩn bị ----------
    seed = int(cfg.get("seed", 42))
    model_name = getattr(model, "model_name", "unknown")

    # --resume: ghi tiếp vào ĐÚNG thư mục run cũ, không tạo run_id mới.
    # Nếu tạo mới thì train_log.csv bị chia đôi và result.json mất lịch sử.
    if resume_from is not None:
        run_dir = Path(resume_from)
        run_id = run_dir.name
    else:
        run_id = make_run_id(model_name, seed, tag)
        run_dir = Path(run_dir or f"artifacts/runs/{run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    # AMP chỉ bật trên CUDA. MPS chưa hỗ trợ autocast đầy đủ (lỗi im lặng hoặc
    # kết quả sai), CPU thì không có lợi gì.
    use_amp = bool(cfg.train.get("amp", True)) and device.type == "cuda"
    if cfg.train.get("amp", True) and not use_amp:
        log.info("AMP bị tắt (thiết bị %s không hỗ trợ hiệu quả)", device.type)

    phases = _resolve_phases(cfg)
    state = _TrainState(
        saver=CheckpointSaver(run_dir),
        csv_logger=CsvLogger(run_dir / "train_log.csv"),
        stopper=EarlyStopping(patience=int(cfg.train.get("patience", 10)), mode="max"),
        criterion=build_criterion(cfg).to(device),
        scaler=torch.amp.GradScaler(device.type, enabled=use_amp),
        use_amp=use_amp,
        device=device,
        total_epochs=sum(p["epochs"] for p in phases),
        seed=seed,
        run_id=run_id,
    )

    if resume_from is not None:
        _restore_from_checkpoint(model, state, Path(resume_from), device)

    save_config(cfg, run_dir / "config.yaml")
    log.info("=" * 70)
    log.info("RUN %s", run_id)
    log.info("model=%s  params=%.2fM  device=%s  epochs=%d  seed=%d",
             model_name, count_parameters(model) / 1e6, device,
             state.total_epochs, seed)
    log.info("theo dõi: val macro-F1 (KHÔNG theo accuracy — dữ liệu mất cân bằng)")
    log.info("=" * 70)

    # ---------- Chạy từng pha ----------
    started = time.time()
    for index, phase in enumerate(phases):
        state.phase_index = index
        _run_one_phase(model, loaders, cfg, phase, state)
        if state.stopper.should_stop:
            break
    train_seconds = time.time() - started

    # ---------- Nạp lại trọng số TỐT NHẤT ----------
    # Không nạp lại thì model đang giữ trọng số của epoch CUỐI, có thể tệ hơn.
    best_state = torch.load(state.saver.best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model_state"])
    log.info("Đã nạp lại checkpoint tốt nhất (epoch %d, val macro-F1 %.4f)",
             state.stopper.best_epoch, state.stopper.best or 0.0)

    # ---------- Ghi result.json ----------
    result = _build_result(model_name, cfg, loaders, state, train_seconds,
                           state.saver.best_path, tag)
    with open(run_dir / "result.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    log.info("Xong sau %.1f phút. Kết quả: %s", train_seconds / 60, run_dir)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "best_ckpt": str(state.saver.best_path),
        "history": pd.DataFrame(state.history),
        "best_val_macro_f1": state.stopper.best,
        "best_epoch": state.stopper.best_epoch,
        "train_seconds": train_seconds,
        "result": result,
    }
