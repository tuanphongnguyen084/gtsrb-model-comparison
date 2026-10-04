"""
callbacks — early stopping, lưu checkpoint, ghi log CSV.

CHỦ: Hoàng
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch


class EarlyStopping:
    """Dừng sớm khi chỉ số theo dõi không cải thiện sau `patience` epoch.

    ★ THEO DÕI val MACRO-F1, KHÔNG THEO accuracy ★
    Dữ liệu mất cân bằng 10,7:1 nên accuracy bị lớp đông chi phối: model có thể
    bỏ hẳn một lớp 210 ảnh mà accuracy gần như không giảm.
    Nguyên tắc: CHỌN CHECKPOINT THEO ĐÚNG CHỈ SỐ MÌNH THẬT SỰ QUAN TÂM.
    Và tuyệt đối không theo chỉ số trên tập TEST.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4,
                 mode: str = "max") -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode phải là max/min, nhận: {mode!r}")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: float | None = None
        self.best_epoch = -1
        self.counter = 0
        self.should_stop = False

    def _is_better(self, value: float) -> bool:
        """So sánh có tính tới min_delta: cải thiện quá nhỏ thì KHÔNG tính là tốt hơn."""
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def step(self, value: float, epoch: int) -> bool:
        """Trả True nếu đây là kết quả TỐT NHẤT (gọi hàm này mỗi epoch)."""
        if self._is_better(value):
            self.best = value
            self.best_epoch = epoch
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False

    def state_dict(self) -> dict:
        """Trạng thái để lưu vào checkpoint, phục vụ --resume."""
        return {"best": self.best, "best_epoch": self.best_epoch,
                "counter": self.counter, "should_stop": self.should_stop}

    def load_state_dict(self, state: dict) -> None:
        """Khôi phục trạng thái early stopping.

        Không khôi phục thì sau khi resume, `counter` về 0 và model được cho thêm
        `patience` epoch nữa dù đã hết kiên nhẫn từ trước — lãng phí compute.
        """
        self.best = state.get("best")
        self.best_epoch = state.get("best_epoch", -1)
        self.counter = state.get("counter", 0)
        self.should_stop = state.get("should_stop", False)


class CheckpointSaver:
    """Lưu trọng số tốt nhất và trọng số cuối.

    Lưu CẢ HAI vì: 'best' để báo cáo kết quả, 'last' để --resume khi Colab
    mất session giữa buổi train.
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.best_path = self.run_dir / "best.pt"
        self.last_path = self.run_dir / "last.pt"

    def save(self, model: torch.nn.Module, epoch: int, metrics: dict,
             is_best: bool, extra: dict | None = None,
             optimizer: torch.optim.Optimizer | None = None,
             scheduler=None, phase_index: int = 0,
             stopper_state: dict | None = None) -> None:
        """Lưu checkpoint. `last.pt` kèm đủ trạng thái để --resume chạy tiếp được.

        VÌ SAO lưu cả optimizer và scheduler:
        Adam giữ hai moment trung bình động (m, v) tích luỹ qua hàng nghìn bước.
        Resume mà khởi tạo lại optimizer thì hai moment đó về 0, và vài chục bước
        đầu sau khi resume sẽ đi sai hướng — loss nhảy vọt rồi mới ổn lại.
        Lưu thêm ~2x dung lượng model nhưng tiết kiệm được cả giờ train trên Colab.
        """
        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "metrics": metrics,
            "model_name": getattr(model, "model_name", "unknown"),
            "expected_img_size": getattr(model, "expected_img_size", None),
            "normalize_mode": getattr(model, "normalize_mode", None),
            # --- phần dành riêng cho resume ---
            "phase_index": phase_index,
            "optimizer_state": optimizer.state_dict() if optimizer else None,
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "stopper_state": stopper_state,
            **(extra or {}),
        }
        torch.save(payload, self.last_path)
        if is_best:
            # best.pt KHÔNG cần trạng thái optimizer (chỉ dùng để suy luận)
            # -> bỏ ra cho nhẹ file.
            slim = {k: v for k, v in payload.items()
                    if k not in ("optimizer_state", "scheduler_state", "stopper_state")}
            torch.save(slim, self.best_path)


class CsvLogger:
    """Ghi một dòng mỗi epoch vào train_log.csv.

    VÌ SAO CSV chứ không chỉ in ra màn hình: D cần ĐỌC TỰ ĐỘNG để vẽ learning curve
    và tổng hợp bảng. Quy tắc của dự án là không ai chép số bằng tay.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None

        # Nếu file đã có nội dung (trường hợp --resume), đọc lại tên cột từ dòng
        # đầu và KHÔNG ghi header lần nữa.
        #
        # Lỗi này đã gặp thật: không kiểm tra thì sau khi resume, file có header
        # thứ hai ở giữa và pandas.read_csv coi dòng đó là DỮ LIỆU -> toàn bộ cột
        # số thành kiểu chuỗi, mọi hình learning curve vỡ. Lỗi im lặng điển hình.
        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path, "r", encoding="utf-8", newline="") as fh:
                header = fh.readline().strip()
            if header:
                self._fieldnames = header.split(",")

    def log(self, row: dict) -> None:
        """Ghi thêm một dòng. Tự ghi header ở lần gọi đầu nếu file còn trống."""
        write_header = self._fieldnames is None
        if write_header:
            self._fieldnames = list(row.keys())
        with open(self.path, "a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k) for k in self._fieldnames})
