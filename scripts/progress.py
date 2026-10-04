#!/usr/bin/env python
"""
progress.py — xem tiến độ mẻ thực nghiệm đang chạy, có thanh tiến trình và ETA.

CHỦ: dùng chung

    python scripts/progress.py           # xem một lần
    python scripts/progress.py -w        # tự làm mới mỗi 5 giây
    make progress                        # lối tắt

Đọc trạng thái từ logs/ và artifacts/runs/ nên KHÔNG ảnh hưởng gì tới mẻ đang chạy.
Không import torch để khởi động tức thì.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Năm bước của run_everything.sh, kèm ước lượng thời gian (phút) để tính ETA
STEPS = [
    ("1/5", "Nén int8 + xuất ONNX", 15),
    ("2/5", "Thí nghiệm rò rỉ", 25),
    ("3/5", "Ablation 30 run", 300),
    ("4/5", "Thêm seed cho M2", 72),
    ("5/5", "Tổng hợp + báo cáo", 30),
]
ABLATION_AXES = ("resolution", "augmentation", "preprocess",
                 "scaling", "label_smoothing", "components")
N_ABLATION_TOTAL = 30

# Mã màu ANSI; tắt khi đầu ra không phải terminal (ví dụ ghi vào file)
_TTY = sys.stdout.isatty()
def c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text

DIM, BOLD, GREEN, YELLOW, CYAN, RED = "2", "1", "32", "33", "36", "31"


def bar(done: float, total: float, width: int = 30) -> str:
    """Thanh tiến trình ASCII. done/total có thể là số thực."""
    if total <= 0:
        return "░" * width
    frac = max(0.0, min(1.0, done / total))
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


def read_log() -> list[str]:
    path = ROOT / "logs" / "_all.log"
    return path.read_text(encoding="utf-8", errors="replace").splitlines() \
        if path.exists() else []


def parse_steps(lines: list[str]) -> dict:
    """Từ _all.log suy ra: bước nào đã xong, bỏ qua, hay đang chạy.

    Mỗi bước mở đầu bằng một dòng `[hh:mm] n/5 TÊN`. Bước đó kết thúc theo
    MỘT TRONG BA cách, và phải nhận cả ba, không thì bước bị treo ở trạng
    thái "đang chạy" mãi và ETA cộng thêm thời gian của một việc đã xong:

      `-> mã thoát 0`   bước chạy xong bình thường
      `HOÀN TẤT`        dòng cuối mẻ — bước CUỐI chỉ kết thúc bằng dòng này
      `BỎ QUA`          bước bị SKIP_HEAVY=1 bỏ, nằm ngay trên cùng dòng mở

    LỖI ĐÃ GẶP: so chuỗi "bỏ qua" chữ thường trong khi log ghi "BỎ QUA"
    chữ hoa, nên hai bước đẩy lên Colab vẫn tính là đang chạy; cộng với
    bước cuối không có "mã thoát", ETA báo còn 6h42m sau khi mẻ đã xong.
    """
    state: dict[str, dict] = {}
    running: str | None = None          # bước đang mở, để gán dòng kết thúc

    def close(step: str | None, when: str) -> None:
        if step and step in state:
            state[step]["running"] = False
            state[step].setdefault("end", when)

    for line in lines:
        m = re.match(r"\[(\d\d:\d\d)\]\s+(\d/5)\s+(.*)", line)
        if m:
            when, step, rest = m.groups()
            close(running, when)        # bước trước chưa đóng thì đóng tại đây
            if "BỎ QUA" in rest.upper():
                state[step] = {"skipped": True, "running": False, "end": when}
                running = None
            else:
                state[step] = {"start": when, "running": True}
                running = step
            continue

        m2 = re.match(r"\[(\d\d:\d\d)\]\s+-> mã thoát", line)
        if m2:
            close(running, m2.group(1))
            running = None
            continue

        m3 = re.match(r"\[(\d\d:\d\d)\].*HOÀN TẤT", line)
        if m3:
            close(running, m3.group(1))
            running = None

    return state


def count_runs(pattern: str) -> int:
    return len([d for d in glob.glob(str(ROOT / "artifacts/runs" / pattern))
                if Path(d, "result.json").exists()])


def ablation_done() -> int:
    """Số run ablation ĐÃ XONG (có result.json)."""
    total = 0
    for axis in ABLATION_AXES:
        total += count_runs(f"*{axis}-*")
    return total


def training_alive() -> bool:
    """Có tiến trình train.py THẬT đang chạy không?

    ★ Không kiểm cái này thì progress báo nhầm: một run bị kill để lại
    train_log.csv mới sửa, và hàm dưới tưởng nó đang train. Đã gặp thật.
    """
    out = subprocess.run(["ps", "-eo", "command="], capture_output=True, text=True)
    return any("Python" in line and "scripts/train.py" in line
               for line in out.stdout.splitlines())


def current_training() -> tuple[str, int, int, float] | None:
    """Run đang train: (tên, epoch hiện tại, tổng epoch, giây/epoch).

    Trả None nếu không có tiến trình train.py nào thật sự chạy.
    """
    if not training_alive():
        return None
    best = None
    now = time.time()
    for d in glob.glob(str(ROOT / "artifacts/runs/*")):
        p = Path(d)
        log = p / "train_log.csv"
        if (p / "result.json").exists() or not log.exists():
            continue
        if now - log.stat().st_mtime > 180:       # không đụng tới trong 3 phút -> bỏ
            continue
        try:
            rows = list(csv.DictReader(log.open(encoding="utf-8")))
        except Exception:
            continue
        if not rows:
            continue
        secs = [float(r["seconds"]) for r in rows if r.get("seconds")]
        median = sorted(secs)[len(secs) // 2] if secs else 0.0
        cfg = p / "config.yaml"
        total_ep = 0
        if cfg.exists():
            text = cfg.read_text(encoding="utf-8")
            m = re.search(r"^\s*epochs:\s*(\d+)", text, re.M)
            total_ep = int(m.group(1)) if m else 0
        cand = (p.name, len(rows), total_ep, median)
        if best is None or log.stat().st_mtime > best[1]:
            best = (cand, log.stat().st_mtime)
    return best[0] if best else None


def system_info() -> dict:
    out = {}
    try:
        batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True,
                              text=True, timeout=3).stdout
        out["power"] = "AC" if "AC Power" in batt else "PIN (NGUY HIỂM!)"
    except Exception:
        out["power"] = "?"
    out["caffeinate"] = bool(subprocess.run(["pgrep", "-x", "caffeinate"],
                                            capture_output=True).returncode == 0)
    try:
        ps = subprocess.run(["ps", "-eo", "pcpu,rss,command"], capture_output=True,
                            text=True, timeout=3).stdout.splitlines()
        cpu = sum(float(l.split()[0]) for l in ps if "Python" in l and l.split()[0][0].isdigit())
        rss = sum(float(l.split()[1]) for l in ps if "Python" in l and l.split()[0][0].isdigit())
        out["cpu"], out["ram_gb"] = cpu, rss / 1048576
    except Exception:
        out["cpu"], out["ram_gb"] = 0.0, 0.0
    out["alive"] = subprocess.run(["pgrep", "-f", "run_everything.sh"],
                                  capture_output=True).returncode == 0
    return out


def render() -> str:
    lines = []
    now = datetime.now()
    lines.append(c(BOLD, "╭" + "─" * 62 + "╮"))
    lines.append(c(BOLD, f"│  GTSRB — tiến độ thực nghiệm" + " " * 23 + now.strftime("%H:%M:%S") + "   │"))
    lines.append(c(BOLD, "╰" + "─" * 62 + "╯"))

    log_lines = read_log()
    if not log_lines:
        lines.append(c(YELLOW, "  Chưa có logs/_all.log — mẻ chưa khởi động."))
        lines.append(f"  Chạy: {c(CYAN, './run_everything.sh')}")
        return "\n".join(lines)

    state = parse_steps(log_lines)
    done_steps = sum(1 for s in state.values() if not s.get("running", False))
    lines.append("")
    lines.append(f"  {c(BOLD,'TỔNG THỂ')}  [{c(GREEN, bar(done_steps, len(STEPS)))}]  "
                 f"{done_steps}/{len(STEPS)} bước")
    lines.append("")

    remaining_min = 0
    for key, name, est in STEPS:
        info = state.get(key, {})
        if info.get("skipped"):
            mark, extra = c(DIM, "–"), c(DIM, "bỏ qua (chạy trên Colab)")
        elif info.get("running"):
            mark = c(CYAN, "▶")
            start = info.get("start", "")
            extra = c(CYAN, f"đang chạy, bắt đầu {start}")
            remaining_min += est
        elif "start" in info:
            mark = c(GREEN, "✓")
            extra = c(DIM, f"xong {info.get('end','?')}")
        else:
            mark, extra = c(DIM, " "), c(DIM, "chờ")
            remaining_min += est
        lines.append(f"    {mark} {c(DIM,key)}  {name:<32} {extra}")

    # ---- chi tiết run đang train ----
    cur = current_training()
    if cur:
        name, ep, total_ep, per_ep = cur
        lines.append("")
        lines.append(f"  {c(BOLD,'ĐANG TRAIN')}  {c(CYAN, name[:46])}")
        if total_ep:
            eta = timedelta(seconds=int((total_ep - ep) * per_ep))
            lines.append(f"    epoch {ep}/{total_ep}  [{bar(ep,total_ep,26)}]  "
                         f"{ep/total_ep*100:4.0f}%   {per_ep:.0f}s/epoch   còn ~{eta}")
        else:
            lines.append(f"    epoch {ep}   {per_ep:.0f}s/epoch")

    # ---- ablation ----
    n_abl = ablation_done()
    lines.append("")
    colour = GREEN if n_abl >= N_ABLATION_TOTAL else YELLOW
    lines.append(f"  {c(BOLD,'ABLATION')}  [{c(colour, bar(n_abl, N_ABLATION_TOTAL))}]  "
                 f"{n_abl}/{N_ABLATION_TOTAL} run  ({n_abl/N_ABLATION_TOTAL*100:.0f}%)")

    n_seed = count_runs("*seed4*")
    if n_seed:
        lines.append(f"  {c(BOLD,'SEED')}      {n_seed} run đã xong")

    # ---- hệ thống ----
    info = system_info()
    lines.append("")
    power = c(GREEN, info["power"]) if info["power"] == "AC" else c(RED, info["power"])
    caff = c(GREEN, "bật") if info["caffeinate"] else c(RED, "TẮT")
    alive = c(GREEN, "đang chạy") if info["alive"] else c(RED, "KHÔNG CHẠY")
    lines.append(f"  {c(BOLD,'HỆ THỐNG')}  nguồn {power} · caffeinate {caff} · mẻ {alive}")
    lines.append(f"            CPU {info['cpu']:.0f}% · RAM {info['ram_gb']:.1f} GB")

    all_done = (ROOT / "logs/.ALL_DONE").exists()
    if remaining_min and not all_done:
        finish = now + timedelta(minutes=remaining_min)
        lines.append("")
        lines.append(f"  {c(BOLD,'DỰ KIẾN XONG')}  ~{finish.strftime('%H:%M')} "
                     f"{c(DIM, f'(còn ~{remaining_min//60}h{remaining_min%60:02d}m)')}")

    if all_done:
        lines.append("")
        lines.append(c(GREEN, "  ★ HOÀN TẤT — xem docs/BAO_CAO.md và docs/KET_QUA.md"))

    return "\n".join(lines)


def main() -> None:
    """Hiện tiến độ một lần, hoặc tự làm mới với -w."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-w", "--watch", action="store_true", help="tu lam moi")
    parser.add_argument("-n", "--interval", type=int, default=5, help="giay giua 2 lan")
    args = parser.parse_args()

    if not args.watch:
        print(render())
        return

    try:
        while True:
            os.system("clear")
            print(render())
            print(f"\n  {c(DIM, 'Ctrl-C để thoát · làm mới mỗi %ds' % args.interval)}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
