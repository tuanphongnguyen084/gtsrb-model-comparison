#!/usr/bin/env python
"""
run_leakage_experiment.py — ĐO phần accuracy "ảo" do rò rỉ dữ liệu.

CHỦ: Huy

★ ĐÂY LÀ THÍ NGHIỆM ĐẮT GIÁ NHẤT CỦA PHẦN DỮ LIỆU ★

Nhóm đã đếm được: split random cho 1.306 track dùng chung giữa train và val, split
theo track cho 0. Nhưng con số đó mới chỉ nói RÒ RỈ CÓ TỒN TẠI, chưa nói nó làm
accuracy cao lên BAO NHIÊU. Script này đo điều đó.

CÁCH LÀM — train CÙNG một model, CÙNG seed, chỉ đổi cách split:

    A. split RANDOM theo ảnh  (sai)   -> val accuracy = ?
    B. split THEO TRACK       (đúng)  -> val accuracy = ?

    Phần "ảo" = val_A - val_B

Dấu hiệu nhận biết rò rỉ mà không cần làm thí nghiệm này: val cao hơn test một cách
bất thường. Split đúng thì val và test phải GẦN nhau, vì cả hai đều là biển báo
chưa từng thấy.

AN TOÀN: script sao lưu index.csv trước khi đổi split và LUÔN khôi phục ở cuối,
kể cả khi bị lỗi giữa chừng (khối finally). Không bao giờ để lại index.csv ở
trạng thái split sai.

    python scripts/run_leakage_experiment.py
    python scripts/run_leakage_experiment.py --epochs 15   # chạy nhanh hơn
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from gtsrb.data.split import check_leakage, make_split
from gtsrb.utils.logging import get_logger

log = get_logger()
PYTHON = sys.executable


def train_and_read(config: str, tag: str, epochs: int | None) -> dict:
    """Train một lần rồi đọc result.json của nó. Trả dict rỗng nếu thất bại."""
    command = [PYTHON, "scripts/train.py", "--config", config, "--tag", tag]
    if epochs:
        command += ["--set", f"train.epochs={epochs}"]

    log.info("  $ %s", " ".join(command))
    if subprocess.run(command).returncode != 0:
        log.error("  Train thất bại với tag=%s", tag)
        return {}

    runs = sorted(Path("artifacts/runs").glob(f"*{tag}*"))
    if not runs:
        return {}
    result_path = runs[-1] / "result.json"
    return json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}


def evaluate_run(tag: str) -> None:
    """Chạy evaluate.py để điền số TEST vào result.json của run vừa train."""
    subprocess.run([PYTHON, "scripts/evaluate.py",
                    "--runs", f"artifacts/runs/*{tag}*", "--no-mcnemar"],
                   capture_output=True)


def run_one_arm(name: str, group_by_track: bool, index_csv: Path,
                config: str, epochs: int | None, seed: int, val_ratio: float) -> dict:
    """Một nhánh của thí nghiệm: đặt split, train, đánh giá, trả số liệu."""
    log.info("")
    log.info("=" * 70)
    log.info("NHÁNH %s — split %s", name,
             "THEO TRACK (đúng)" if group_by_track else "RANDOM theo ảnh (SAI)")
    log.info("=" * 70)

    make_split(index_csv, val_ratio=val_ratio, seed=seed,
               group_by_track=group_by_track)
    shared = check_leakage(pd.read_csv(index_csv))["n_shared_tracks"]
    log.info("  track dùng chung giữa train và val: %d", shared)

    tag = f"leak-{name}"
    result = train_and_read(config, tag, epochs)
    evaluate_run(tag)

    # Đọc lại sau khi evaluate đã điền phần "test"
    runs = sorted(Path("artifacts/runs").glob(f"*{tag}*"))
    if runs:
        path = runs[-1] / "result.json"
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))

    val = (result.get("val") or {})
    test = (result.get("test") or {})
    return {
        "nhánh": name,
        "cách split": "theo track" if group_by_track else "random theo ảnh",
        "track dùng chung": shared,
        "val_macro_f1": val.get("best_macro_f1"),
        "test_top1": test.get("top1"),
        "test_macro_f1": test.get("macro_f1"),
        "run_id": result.get("run_id"),
    }


def main() -> None:
    """Train cùng model hai lần, chỉ khác cách split, để đo phần accuracy ảo."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/m1_lenet.yaml",
                        help="dung M1 vi no nhanh nhat (~12 phut moi nhanh)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--index", default="data/processed/index.csv")
    parser.add_argument("--out", default="reports/tables/leakage_experiment.csv")
    args = parser.parse_args()

    index_csv = Path(args.index)
    if not index_csv.exists():
        raise SystemExit(f"Chưa có {index_csv}. Chạy: make data")

    # ---- Chốt an toàn 1: index.csv phải đang ở trạng thái ĐÚNG trước khi bắt đầu ----
    # Nếu một lần chạy trước bị kill giữa chừng, index.csv có thể còn kẹt ở split
    # RANDOM. Chạy tiếp từ trạng thái đó sẽ cho kết quả vô nghĩa mà không báo lỗi.
    before = check_leakage(pd.read_csv(index_csv))["n_shared_tracks"]
    if before != 0:
        raise SystemExit(
            f"index.csv đang ở trạng thái SPLIT SAI ({before} track dùng chung).\n"
            f"Nhiều khả năng một lần chạy trước bị dừng giữa chừng.\n"
            f"Khôi phục bằng: make data   (hoặc python scripts/prepare_data.py --skip-cache)"
        )

    # ---- Chốt an toàn 2: không cho hai bản chạy cùng lúc ----
    # Thí nghiệm này SỬA index.csv. Hai bản chạy song song sẽ giẫm chân nhau và
    # làm hỏng split cho MỌI thực nghiệm khác. Đã suýt xảy ra thật.
    lock = index_csv.parent / ".leakage.lock"
    if lock.exists():
        try:
            other = int(lock.read_text())
            import os
            os.kill(other, 0)           # còn sống?
            raise SystemExit(
                f"Đã có một thí nghiệm rò rỉ đang chạy (PID {other}). "
                f"Hai bản cùng sửa index.csv sẽ hỏng dữ liệu."
            )
        except (ValueError, ProcessLookupError, PermissionError):
            lock.unlink(missing_ok=True)      # khoá cũ của tiến trình đã chết
    import os
    lock.write_text(str(os.getpid()))

    # ---- Sao lưu để LUÔN khôi phục được ----
    backup = index_csv.with_suffix(".csv.leakage_backup")
    shutil.copy(index_csv, backup)
    log.info("Đã sao lưu %s -> %s", index_csv, backup.name)

    rows = []
    try:
        rows.append(run_one_arm("A_random", False, index_csv, args.config,
                                args.epochs, args.seed, args.val_ratio))
        rows.append(run_one_arm("B_track", True, index_csv, args.config,
                                args.epochs, args.seed, args.val_ratio))
    finally:
        # ★ LUÔN khôi phục split đúng, kể cả khi lỗi giữa chừng.
        # Để sót index.csv ở trạng thái split random là tai hoạ: mọi thực nghiệm
        # chạy sau đó sẽ bị rò rỉ mà không ai biết.
        shutil.copy(backup, index_csv)
        backup.unlink(missing_ok=True)
        (index_csv.parent / ".leakage.lock").unlink(missing_ok=True)
        log.info("")
        log.info("Đã KHÔI PHỤC %s về split theo track.", index_csv)
        report = check_leakage(pd.read_csv(index_csv))
        log.info("  kiểm lại: %d track dùng chung (phải là 0)",
                 report["n_shared_tracks"])

    frame = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)

    log.info("")
    log.info("=" * 70)
    log.info("KẾT QUẢ THÍ NGHIỆM RÒ RỈ -> %s", out)
    log.info("\n%s", frame.to_string(index=False))

    # ---- Diễn giải ----
    try:
        a = frame[frame["nhánh"] == "A_random"].iloc[0]
        b = frame[frame["nhánh"] == "B_track"].iloc[0]
        log.info("")
        log.info("CÁCH ĐỌC:")
        if a["val_macro_f1"] and b["val_macro_f1"]:
            gap = a["val_macro_f1"] - b["val_macro_f1"]
            log.info("  val macro-F1: random %.4f  vs  theo track %.4f",
                     a["val_macro_f1"], b["val_macro_f1"])
            log.info("  -> phần 'ẢO' do rò rỉ = %.4f (%.2f điểm)", gap, gap * 100)
        if a["val_macro_f1"] and a["test_macro_f1"]:
            log.info("")
            log.info("  Dấu hiệu rò rỉ rõ nhất là KHOẢNG CÁCH val-test:")
            log.info("    nhánh A (random)    : val %.4f, test %.4f -> cách %.4f",
                     a["val_macro_f1"], a["test_macro_f1"],
                     a["val_macro_f1"] - a["test_macro_f1"])
            log.info("    nhánh B (theo track): val %.4f, test %.4f -> cách %.4f",
                     b["val_macro_f1"], b["test_macro_f1"],
                     b["val_macro_f1"] - b["test_macro_f1"])
            log.info("  Split ĐÚNG thì val và test phải GẦN nhau, vì cả hai đều là")
            log.info("  biển báo chưa từng thấy. Split SAI thì val cao hơn hẳn.")
    except (IndexError, TypeError):
        log.warning("Thiếu số liệu để diễn giải — kiểm tra lại hai nhánh có chạy xong không.")


if __name__ == "__main__":
    main()
