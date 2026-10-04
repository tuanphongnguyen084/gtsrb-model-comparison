"""
config — nạp cấu hình YAML, cho phép kế thừa và override từ dòng lệnh.

CHỦ: Huy

QUY TẮC CỦA DỰ ÁN: MỌI siêu tham số phải nằm trong YAML, không hard-code trong src/.
Hard-code một lần là mất khả năng ablation (muốn thử img_size=64 phải sửa code).

Cách dùng:
    cfg = load_config("configs/m2_vggres.yaml", overrides=["train.epochs=3"])
    print(cfg.train.epochs)     # truy cập bằng dấu chấm cho dễ đọc
    print(cfg["train"]["epochs"])  # hoặc như dict bình thường
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


class Config(dict):
    """dict nhưng truy cập được bằng dấu chấm: cfg.train.epochs

    Lý do không dùng dict thường: cfg["train"]["optimizer"]["lr"] đọc rất mệt,
    cfg.train.optimizer.lr dễ đọc hơn nhiều khi code đầy chỗ như vậy.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Bọc đệ quy mọi dict con thành Config
        for key, value in list(self.items()):
            if isinstance(value, dict):
                self[key] = Config(value)
            elif isinstance(value, list):
                self[key] = [Config(v) if isinstance(v, dict) else v for v in value]

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(
                f"Config không có khoá '{name}'. Các khoá hiện có: {sorted(self.keys())}"
            ) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = Config(value) if isinstance(value, dict) else value

    def to_plain(self) -> dict:
        """Trả về dict thuần (để yaml.safe_dump được)."""
        out: dict = {}
        for key, value in self.items():
            if isinstance(value, Config):
                out[key] = value.to_plain()
            elif isinstance(value, list):
                out[key] = [v.to_plain() if isinstance(v, Config) else v for v in value]
            else:
                out[key] = value
        return out


def _deep_merge(base: dict, extra: dict) -> dict:
    """Trộn extra vào base theo chiều sâu. extra thắng khi trùng khoá."""
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _parse_scalar(text: str) -> Any:
    """Đổi chuỗi từ CLI thành kiểu Python đúng ('3' -> int, 'true' -> bool...)."""
    return yaml.safe_load(text)


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> Config:
    """Nạp file YAML, xử lý khoá `inherit`, rồi áp các override dạng 'a.b=c'.

    `inherit: base.yaml` cho phép mỗi config model chỉ ghi phần KHÁC base,
    nên đọc file config của một thực nghiệm là thấy ngay nó khác gì so với mặc định.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file config: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    # Kế thừa: nạp file cha trước rồi trộn file con lên trên
    parent_name = raw.pop("inherit", None)
    if parent_name:
        parent_path = path.parent / parent_name
        parent = load_config(parent_path).to_plain()
        raw = _deep_merge(parent, raw)

    # Override từ CLI: --set train.epochs=3 data.img_size=64
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override phải có dạng 'khoa.con=giatri', nhận được: {item!r}")
        dotted_key, _, value_text = item.partition("=")
        cursor = raw
        parts = dotted_key.strip().split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = _parse_scalar(value_text)

    cfg = Config(raw)
    cfg.config_path = str(path)
    return cfg


def save_config(cfg: Config, path: str | Path) -> None:
    """Lưu lại cấu hình THỰC TẾ đã dùng (sau khi merge + override).

    Quan trọng cho tái lập: file config gốc không phản ánh các override từ CLI.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_plain(), fh, allow_unicode=True, sort_keys=False)
