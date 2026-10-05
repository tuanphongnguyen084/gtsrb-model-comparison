#!/usr/bin/env python
"""
run_seeds.py — chạy cùng một cấu hình với nhiều seed, rồi gộp thành mean ± std.

CHỦ: Hoàng

★ VÌ SAO CẦN FILE NÀY ★
Toàn bộ kết luận của dự án đang dựa trên MỘT lần chạy mỗi cấu hình. McNemar xử lý
được phương sai *trong một lần chạy* (hai model trên cùng tập test), nhưng KHÔNG xử lý
được phương sai *giữa các lần khởi tạo khác nhau*.

Ví dụ cụ thể của dự án này: M2 và ResNet18 chênh nhau 9 ảnh trên 12.630. Rất có thể
thứ hạng sẽ đảo nếu đổi seed. Nếu không đo, ta không có cách nào biết.

Nguyên tắc phát biểu kết quả:
  - Chênh lệch LỚN HƠN 2x độ lệch chuẩn giữa các seed  -> có thể kết luận
  - Chênh lệch NHỎ HƠN                                  -> KHÔNG được kết luận,
    phải nói "nằm trong nhiễu khởi tạo"

Ví dụ:
    # 3 seed cho M2 (mặc định)
    python scripts/run_seeds.py --config configs/m2_vggres.yaml

    # 5 seed, rút ngắn epoch cho nhanh
    python scripts/run_seeds.py --config configs/m2_vggres.yaml \
        --seeds 42 43 44 45 46 --set train.epochs=20

    # chỉ gộp lại bảng từ các run đã có
    python scripts/run_seeds.py --collect-only
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from gtsrb.utils.logging import get_logger

log = get_logger()
PYTHON = sys.executable

# Các chỉ số được gộp mean/std. Chỉ lấy chỉ số trên tập TEST — val đã dùng để
# chọn checkpoint nên nó có bias chọn lọc, không phải ước lượng không chệch.
METRICS = ["top1", "top5", "macro_f1", "ece"]


def train_one_seed(config: str, seed: int, overrides: list[str],
                   subset: int | None) -> bool:
    """Chạy train.py cho một seed. Trả True nếu thành công."""
    tag = f"seed{seed}"
    command = [PYTHON, "scripts/train.py", "--config", config, "--tag", tag,
               "--set", f"seed={seed}", *overrides]
    if subset:
        command += ["--subset", str(subset)]

    log.info("  $ %s", " ".join(command))
    return subprocess.run(command).returncode == 0


# Các khoá cấu hình phải GIỐNG NHAU để hai run được coi là "cùng thí nghiệm,
# khác seed". Thiếu một khoá nào ở đây là gộp lẫn hai thí nghiệm khác nhau rồi
# gọi chênh lệch đó là "nhiễu seed" — kết luận sẽ sai hẳn.
KHOA_CAU_HINH = ("epochs_run", "batch_size", "lr", "label_smoothing", "optimizer")


def _van_tay(data: dict) -> tuple:
    """Dấu vết cấu hình của một run, để so xem có cùng thí nghiệm không."""
    tr = data.get("train") or {}
    da = data.get("data") or {}
    return (data.get("model"),
            tuple(tr.get(k) for k in KHOA_CAU_HINH),
            da.get("img_size"), da.get("preprocess"), da.get("aug_policy"))


def collect(pattern: str = "artifacts/runs/*seed*",
            them_run_goc: bool = True) -> pd.DataFrame:
    """Gom các run khác seed của CÙNG một cấu hình thành bảng dài.

    ★ VÌ SAO PHẢI THÊM RUN GỐC (`them_run_goc`) ★

    Run chính của mỗi model cũng là MỘT SEED (42), chỉ là nó không có tag
    `seedN`. Bản đầu của hàm này chỉ khớp `*seed*` nên bỏ nó ra, và bảng
    mean±std chỉ còn 2 seed thay vì 3.

    Mất mát không nhỏ: chính câu hỏi của phần này là "chênh lệch giữa các model
    có phải chỉ là nhiễu seed?". Với m2_vggres, 3 seed cho macro-F1 0,98798 /
    0,99297 / 0,99066 — biên độ 0,50 điểm. Bỏ seed 42 thì biên độ chỉ còn 0,23
    điểm, tức BÁO CÁO SAI rằng mô hình ổn định hơn thực tế gấp đôi.

    Chỉ thêm run gốc khi cấu hình TRÙNG KHỚP (xem KHOA_CAU_HINH). Nếu run chính
    train 40 epoch mà run seed train 15 epoch thì chúng không so được, và gộp
    vào sẽ biến "khác số epoch" thành "nhiễu seed".
    """
    rows, van_tay_seed = [], set()

    def them(data: dict) -> None:
        test = data.get("test") or {}
        if not test:
            log.warning("Bỏ qua %s — chưa có số test. Chạy scripts/evaluate.py trước.",
                        data.get("run_id"))
            return
        rows.append({
            "model": data.get("model"),
            "seed": data.get("seed"),
            "run_id": data.get("run_id"),
            **{m: test.get(m) for m in METRICS},
        })

    for path in sorted(glob.glob(f"{pattern}/result.json")):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        them(data)
        if data.get("test"):
            van_tay_seed.add(_van_tay(data))

    if them_run_goc and van_tay_seed:
        da_co = {r["run_id"] for r in rows}
        for path in sorted(glob.glob("artifacts/runs/*/result.json")):
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if data.get("run_id") in da_co or (data.get("notes") or ""):
                continue                       # đã có, hoặc là run ablation/thí nghiệm
            if _van_tay(data) in van_tay_seed:
                log.info("Thêm run gốc %s (seed %s) — cấu hình TRÙNG với các run seed",
                         data.get("run_id"), data.get("seed"))
                them(data)

    return pd.DataFrame(rows)


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """Gộp theo model: mean, std, min, max và số seed.

    Báo cả std VÀ số seed, vì std tính từ 2 seed gần như vô nghĩa — người đọc
    cần biết nó dựa trên bao nhiêu lần chạy.
    """
    if frame.empty:
        return frame
    grouped = frame.groupby("model")[METRICS].agg(["mean", "std", "min", "max"])
    grouped[("n_seeds", "")] = frame.groupby("model").size()
    return grouped


def report_separability(frame: pd.DataFrame) -> None:
    """In ra: chênh lệch giữa các model có VƯỢT nhiễu seed không.

    Đây là câu hỏi thật sự quan trọng. Một bảng mean±std đẹp nhưng không ai đọc ra
    được kết luận thì vô dụng.
    """
    if frame["model"].nunique() < 2:
        return

    stats = frame.groupby("model")["macro_f1"].agg(["mean", "std", "size"])
    stats = stats.sort_values("mean", ascending=False)

    log.info("")
    log.info("=" * 70)
    log.info("CHÊNH LỆCH GIỮA CÁC MODEL CÓ VƯỢT NHIỄU SEED KHÔNG?")
    log.info("(quy tắc: chênh lệch phải > 2x std gộp thì mới kết luận được)")
    log.info("")

    models = list(stats.index)
    for i in range(len(models) - 1):
        a, b = models[i], models[i + 1]
        gap = stats.loc[a, "mean"] - stats.loc[b, "mean"]
        # std gộp của hai nhóm (lấy căn trung bình bình phương, xấp xỉ đơn giản)
        pooled = ((stats.loc[a, "std"] ** 2 + stats.loc[b, "std"] ** 2) / 2) ** 0.5
        if pd.isna(pooled) or pooled == 0:
            verdict = "không tính được (cần >= 2 seed mỗi model)"
        elif gap > 2 * pooled:
            verdict = f"KẾT LUẬN ĐƯỢC (chênh {gap:.4f} > 2x std {2*pooled:.4f})"
        else:
            verdict = f"NẰM TRONG NHIỄU (chênh {gap:.4f} <= 2x std {2*pooled:.4f})"
        log.info("  %-16s vs %-16s  %s", a, b, verdict)


def main() -> None:
    """Chạy nhiều seed cho một config rồi gộp mean/std, hoặc chỉ gộp bảng."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None,
                        help="config can chay. Bo qua neu dung --collect-only")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--out", default="reports/tables/seeds.csv")
    args = parser.parse_args()

    if not args.collect_only:
        if not args.config:
            parser.error("cần --config, hoặc dùng --collect-only")
        log.info("Chạy %d seed cho %s: %s", len(args.seeds), args.config, args.seeds)
        failures = []
        for i, seed in enumerate(args.seeds, 1):
            log.info("")
            log.info("=" * 70)
            log.info("[%d/%d] seed = %d", i, len(args.seeds), seed)
            if not train_one_seed(args.config, seed, args.overrides, args.subset):
                log.error("  THẤT BẠI với seed %d — ghi lại và chạy tiếp", seed)
                failures.append(seed)
        if failures:
            log.warning("Các seed thất bại: %s", failures)
        log.info("")
        log.info("Bước tiếp: python scripts/evaluate.py --runs \"artifacts/runs/*seed*\"")
        log.info("           python scripts/run_seeds.py --collect-only")
        return

    frame = collect()
    if frame.empty:
        log.warning("Chưa có run nào có tag seedN kèm số test.")
        log.warning("Chạy: python scripts/run_seeds.py --config <config>")
        log.warning("  rồi: python scripts/evaluate.py --runs \"artifacts/runs/*seed*\"")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    log.info("Chi tiết từng run -> %s", out)
    log.info("\n%s", frame.round(5).to_string(index=False))

    summary = summarise(frame)
    summary_path = out.with_name("seeds_summary.csv")
    summary.to_csv(summary_path)
    log.info("")
    log.info("Gộp theo model -> %s", summary_path)
    log.info("\n%s", summary.round(5).to_string())

    report_separability(frame)


if __name__ == "__main__":
    main()
