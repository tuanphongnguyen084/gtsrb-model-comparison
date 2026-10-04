#!/usr/bin/env python
"""
merge_runs.py — gộp kết quả chạy từ máy khác (Colab, máy bạn cùng nhóm) vào đây.

CHỦ: dùng chung

Mỗi run là một thư mục `artifacts/runs/<run_id>/` tự chứa đủ thông tin, nên gộp
chỉ là chép thư mục vào. Script này làm việc đó AN TOÀN:

  - không ghi đè run đã có (trừ khi --force), vì run_id có timestamp nên trùng
    tên gần như chắc chắn là cùng một run đã chép rồi
  - kiểm `result.json` đọc được và có đủ khoá bắt buộc trước khi chép
  - báo rõ cái nào chép, cái nào bỏ qua, vì sao

    python scripts/merge_runs.py colab_results.zip
    python scripts/merge_runs.py ~/Downloads/runs_from_hoang/   # hoặc một thư mục
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from gtsrb.utils.logging import get_logger

log = get_logger()

REQUIRED = ("run_id", "model", "seed")


def valid_run(folder: Path) -> tuple[bool, str]:
    """Thư mục này có phải một run hợp lệ không? Trả (ok, lý do nếu không)."""
    result = folder / "result.json"
    if not result.exists():
        return False, "thiếu result.json"
    try:
        data = json.loads(result.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"result.json hỏng ({exc})"
    missing = [k for k in REQUIRED if k not in data]
    if missing:
        return False, f"result.json thiếu khoá {missing}"
    if not (folder / "config.yaml").exists():
        return False, "thiếu config.yaml (không tái lập được)"
    return True, ""


def iter_runs(source: Path):
    """Duyệt mọi thư mục run trong `source` (tìm theo sự có mặt của result.json)."""
    if (source / "result.json").exists():
        yield source
        return
    for path in sorted(source.rglob("result.json")):
        yield path.parent


def main() -> None:
    """Gộp các thư mục run từ zip hoặc thư mục vào artifacts/runs/."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="file .zip hoac thu muc chua cac run")
    parser.add_argument("--dest", default="artifacts/runs")
    parser.add_argument("--force", action="store_true",
                        help="ghi de run da co (mac dinh: bo qua)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"Không có {source}")

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    tmp = None
    if source.suffix == ".zip":
        tmp = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(source) as archive:
            archive.extractall(tmp.name)
        source = Path(tmp.name)
        log.info("Đã giải nén %s", args.source)

    copied = skipped = invalid = 0
    try:
        for folder in iter_runs(source):
            ok, reason = valid_run(folder)
            if not ok:
                log.warning("  BỎ (%s): %s", reason, folder.name)
                invalid += 1
                continue

            target = dest / folder.name
            if target.exists() and not args.force:
                log.info("  bỏ qua (đã có): %s", folder.name)
                skipped += 1
                continue

            size_mb = sum(f.stat().st_size for f in folder.rglob("*")
                          if f.is_file()) / 1e6
            if args.dry_run:
                log.info("  SẼ CHÉP: %s (%.1f MB)", folder.name, size_mb)
            else:
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(folder, target)
                log.info("  chép: %-52s %.1f MB", folder.name, size_mb)
            copied += 1
    finally:
        if tmp:
            tmp.cleanup()

    log.info("")
    log.info("Chép %d · bỏ qua %d (đã có) · không hợp lệ %d", copied, skipped, invalid)
    if copied and not args.dry_run:
        log.info("")
        log.info("Bước tiếp — gộp vào bảng:")
        log.info('  python scripts/evaluate.py --runs "artifacts/runs/*"')
        log.info("  python scripts/run_ablation.py --collect-only")
        log.info("  python scripts/run_seeds.py --collect-only")
        log.info("  make report baocao")


if __name__ == "__main__":
    main()
