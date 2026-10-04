#!/usr/bin/env python
"""
run_ablation.py — chạy các thực nghiệm ablation MỘT-BIẾN-MỘT-LẦN rồi tổng hợp.

CHỦ: dùng chung — mỗi người chạy trục của mình, Phong Nguyễn tổng hợp bảng

  Hoàng       --axes components scaling label_smoothing
  Phong Trần  --axes resolution
  Huy         --axes augmentation preprocess

★ VÌ SAO PHẢI ĐỔI MỘT BIẾN MỘT LẦN ★
Đổi hai biến cùng lúc thì không quy được kết quả về nguyên nhân nào, và còn có
thể bị TRIỆT TIÊU: biến thứ nhất +0,3 điểm, biến thứ hai -0,3 điểm, tổng ra 0
và ta kết luận SAI rằng "cả hai đều vô dụng".

5 TRỤC:
  1. resolution     32 / 48 / 64        (M1, M2)   |  64 / 112 / 224 (M3)
  2. augmentation   none / geo / geo_photo / geo_photo_erase
  3. preprocess     none / he_gray / he_y / clahe
  4. model scaling  width x0,5 / x1 / x2  và  depth 3 / 4 / 5 stage
  5. label smooth   0,0 / 0,1 / 0,2
  + ablation nội bộ M2: bỏ BN / bỏ residual / bỏ spatial dropout

LƯU Ý: trục 1 và 3 cần CACHE RIÊNG cho mỗi cấu hình. Script tự gọi prepare_data
khi thiếu cache.

Ví dụ:
    # xem sẽ chạy những gì mà chưa chạy thật
    python scripts/run_ablation.py --axes label_smoothing --dry-run

    # chạy một trục
    python scripts/run_ablation.py --axes label_smoothing --epochs 15

    # chạy nhanh để kiểm đường ống (KHÔNG dùng số này để báo cáo)
    python scripts/run_ablation.py --axes all --epochs 2 --subset 3000

    # chỉ tổng hợp lại bảng từ các run đã có
    python scripts/run_ablation.py --collect-only
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


def build_jobs(axes: list[str], base_epochs: int | None) -> list[dict]:
    """Sinh danh sách thực nghiệm. Mỗi job đổi ĐÚNG MỘT biến so với config gốc."""
    jobs: list[dict] = []

    def add(axis: str, config: str, tag: str, overrides: list[str],
            needs_cache: tuple[int, str] | None = None) -> None:
        """Thêm một job. `needs_cache` = (size, preprocess) nếu job cần cache riêng."""
        jobs.append({"axis": axis, "config": config, "tag": tag,
                     "overrides": overrides, "needs_cache": needs_cache})

    # --- Trục 1: độ phân giải ---
    if "resolution" in axes:
        for size in (32, 48, 64):
            for config in ("configs/m1_lenet.yaml", "configs/m2_vggres.yaml"):
                short = Path(config).stem.split("_")[0]
                add("resolution", config, f"res{size}",
                    [f"data.img_size={size}", f"data.cache_size={size}"],
                    needs_cache=(size, "clahe"))
        for size in (64, 112, 224):
            # M3: đổi img_size nhưng GIỮ cache 48 (nội suy lên) -> chỉ đổi 1 biến
            add("resolution", "configs/m3_resnet18.yaml", f"res{size}",
                [f"data.img_size={size}"])

    # --- Trục 2: augmentation ---
    if "augmentation" in axes:
        for policy in ("none", "geo", "geo_photo", "geo_photo_erase"):
            add("augmentation", "configs/m2_vggres.yaml", f"aug_{policy}",
                [f"data.aug_policy={policy}"])

    # --- Trục 3: tiền xử lý ---
    if "preprocess" in axes:
        for mode in ("none", "he_gray", "he_y", "clahe"):
            add("preprocess", "configs/m2_vggres.yaml", f"prep_{mode}",
                [f"data.preprocess={mode}"], needs_cache=(48, mode))

    # --- Trục 4: model scaling ---
    if "scaling" in axes:
        for width in (0.5, 1.0, 2.0):
            add("scaling", "configs/m2_vggres.yaml", f"width{width}",
                [f"model.kwargs.width_mult={width}"])
        for depth in (3, 4, 5):
            add("scaling", "configs/m2_vggres.yaml", f"depth{depth}",
                [f"model.kwargs.n_stages={depth}"])

    # --- Trục 5: label smoothing ---
    if "label_smoothing" in axes:
        for eps in (0.0, 0.1, 0.2):
            add("label_smoothing", "configs/m2_vggres.yaml", f"ls{eps}",
                [f"train.label_smoothing={eps}"])

    # --- Ablation nội bộ M2: từng thành phần kiến trúc ---
    if "components" in axes:
        add("components", "configs/m2_vggres.yaml", "full", [])
        add("components", "configs/m2_vggres.yaml", "no_bn",
            ["model.kwargs.use_bn=false"])
        add("components", "configs/m2_vggres.yaml", "no_residual",
            ["model.kwargs.use_residual=false"])
        add("components", "configs/m2_vggres.yaml", "no_spatial_dropout",
            ["model.kwargs.use_spatial_dropout=false"])

    if base_epochs is not None:
        for job in jobs:
            job["overrides"] = job["overrides"] + [f"train.epochs={base_epochs}"]
    return jobs


def ensure_cache(size: int, preprocess: str) -> None:
    """Dựng cache nếu thiếu (trục resolution và preprocess cần cache riêng)."""
    cache_file = Path(f"data/processed/images_{size}_{preprocess}.npy")
    if cache_file.exists():
        return
    log.info("Thiếu cache %s -> dựng (một lần, dùng lại cho mọi run sau)", cache_file)
    subprocess.run(
        [PYTHON, "scripts/prepare_data.py",
         "--img-size", str(size), "--preprocess", preprocess],
        check=True,
    )


def collect(pattern: str = "artifacts/runs/*") -> pd.DataFrame:
    """Gom mọi result.json thành một bảng. KHÔNG AI CHÉP SỐ BẰNG TAY.

    Đây là lý do mọi run bắt buộc ghi result.json đúng schema: bảng báo cáo
    được SINH RA, nên mọi số truy được về một run_id cụ thể.
    """
    rows = []
    for path in sorted(glob.glob(f"{pattern}/result.json")):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        test = data.get("test") or {}
        rows.append({
            "run_id": data.get("run_id"),
            "model": data.get("model"),
            "tag": data.get("notes"),
            "img_size": (data.get("data") or {}).get("img_size"),
            "preprocess": (data.get("data") or {}).get("preprocess"),
            "aug_policy": (data.get("data") or {}).get("aug_policy"),
            "label_smoothing": (data.get("train") or {}).get("label_smoothing"),
            "epochs_run": (data.get("train") or {}).get("epochs_run"),
            "val_macro_f1": (data.get("val") or {}).get("best_macro_f1"),
            "test_top1": test.get("top1"),
            "test_top5": test.get("top5"),
            "test_macro_f1": test.get("macro_f1"),
            "test_ece": test.get("ece"),
            "train_seconds": (data.get("train") or {}).get("train_seconds"),
        })
    return pd.DataFrame(rows)


ALL_AXES = ["resolution", "augmentation", "preprocess", "scaling",
            "label_smoothing", "components"]

BUDGET_OVERRIDES = ["train.epochs=15", "train.patience=4"]


def parse_args() -> argparse.Namespace:
    """Tham số dòng lệnh. Xem docstring đầu file để biết các ví dụ thường dùng."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--axes", nargs="+", default=["label_smoothing"],
                        choices=ALL_AXES + ["all"])
    parser.add_argument("--epochs", type=int, default=None,
                        help="ghi de so epoch cho MOI job")
    parser.add_argument("--budget", action="store_true",
                        help="CHE DO NGAN SACH THAP: 15 epoch + patience 4. "
                             "Du de XEP HANG cac bien the, tiet kiem ~60%% compute.")
    parser.add_argument("--subset", type=int, default=None,
                        help="chi N mau moi split (SMOKE TEST)")
    parser.add_argument("--dry-run", action="store_true",
                        help="chi in ra se chay gi, khong chay that")
    parser.add_argument("--collect-only", action="store_true",
                        help="chi tong hop lai bang tu cac run da co")
    parser.add_argument("--out", default="reports/tables/ablation.csv")
    parser.add_argument("--figures", default="reports/figures",
                        help="noi luu hinh ablation (sinh cung voi --collect-only)")
    return parser.parse_args()


def plan_jobs(args: argparse.Namespace) -> list[dict]:
    """Sinh danh sách thực nghiệm, áp chế độ ngân sách nếu được yêu cầu.

    ★ VÌ SAO CÓ CHẾ ĐỘ NGÂN SÁCH ★
    30 thực nghiệm x 40 epoch là khoảng 16 giờ trên M1 Pro. Nhưng mục đích của
    ablation là XẾP HẠNG các biến thể, không phải lấy con số cuối cùng của từng
    biến thể. Thứ tự giữa các biến thể ổn định từ lâu trước khi accuracy tuyệt đối
    hội tụ, nên 15 epoch là đủ để xếp hạng.

    Quy trình đúng: (1) --budget xếp hạng 30 biến thể ~5 giờ; (2) chọn cấu hình
    thắng mỗi trục; (3) chạy lại RIÊNG chúng ở độ dài đầy đủ ~1 giờ.
    Tổng ~6 giờ thay vì 16 giờ, kết luận không đổi.

    PHẢI NÓI RÕ trong báo cáo rằng bảng ablation dùng ngân sách 15 epoch — không
    được trộn số 15 epoch với số 40 epoch trong cùng một bảng.
    """
    axes = ALL_AXES if "all" in args.axes else args.axes
    jobs = build_jobs(axes, args.epochs)

    if args.budget:
        log.info("CHẾ ĐỘ NGÂN SÁCH: 15 epoch + patience 4 cho mọi job.")
        log.info("  Dùng để XẾP HẠNG biến thể. Cấu hình thắng phải chạy lại ở")
        log.info("  độ dài đầy đủ trước khi đưa số vào báo cáo.")
        for job in jobs:
            job["overrides"] = job["overrides"] + BUDGET_OVERRIDES
            job["tag"] = f"{job['tag']}-budget"

    log.info("Sẽ chạy %d thực nghiệm trên %d trục: %s", len(jobs), len(axes), axes)
    for i, job in enumerate(jobs, 1):
        log.info("  %2d. [%-16s] %-28s %s",
                 i, job["axis"], Path(job["config"]).stem,
                 " ".join(job["overrides"]) or "(gốc)")
    return jobs


def already_done(job: dict) -> Path | None:
    """Trả thư mục run nếu job này ĐÃ chạy xong trước đó, None nếu chưa.

    ★ Quan trọng với mẻ 30 job chạy 5 giờ: nếu máy ngủ hoặc job bị kill giữa chừng,
    chạy lại đúng lệnh cũ sẽ làm tiếp từ chỗ dở thay vì bắt đầu lại từ đầu.
    Nhận diện bằng tag `<trục>-<giá trị>` nằm trong tên run, và có result.json
    (có result.json = fit() đã chạy xong).
    """
    marker = f"{job['axis']}-{job['tag']}"
    for run_dir in sorted(Path("artifacts/runs").glob(f"*{marker}*")):
        if (run_dir / "result.json").exists():
            return run_dir
    return None


def execute_jobs(jobs: list[dict], args: argparse.Namespace) -> list[dict]:
    """Chạy từng job bằng subprocess. Job lỗi được ghi lại và BỎ QUA, không dừng cả mẻ.

    Vì sao không dừng: một mẻ 30 job chạy 5 giờ, nếu job thứ 3 lỗi mà dừng hết thì
    mất cả buổi. Ghi lại và chạy tiếp, cuối cùng báo danh sách thất bại.

    Job đã chạy xong trước đó thì BỎ QUA (xem already_done).
    """
    failures = []
    for i, job in enumerate(jobs, 1):
        log.info("")
        log.info("=" * 68)
        log.info("[%d/%d] trục=%s  tag=%s", i, len(jobs), job["axis"], job["tag"])

        done = already_done(job)
        if done is not None:
            log.info("  BỎ QUA — đã chạy xong: %s", done.name)
            continue

        if job["needs_cache"]:
            ensure_cache(*job["needs_cache"])

        command = [PYTHON, "scripts/train.py", "--config", job["config"],
                   "--tag", f"{job['axis']}-{job['tag']}"]
        if job["overrides"]:
            command += ["--set"] + job["overrides"]
        if args.subset:
            command += ["--subset", str(args.subset)]

        log.info("  $ %s", " ".join(command))
        if subprocess.run(command).returncode != 0:
            log.error("  THẤT BẠI — ghi lại và tiếp tục job sau")
            failures.append(job)
    return failures


# Cách lấy giá trị biến ra khỏi tên tag, cho mỗi trục
AXIS_XLABEL = {
    "resolution": "img_size", "augmentation": "chính sách augment",
    "preprocess": "chế độ tiền xử lý", "scaling": "cấu hình",
    "label_smoothing": "epsilon", "components": "biến thể",
}


def plot_axes(frame: pd.DataFrame, out_dir: Path,
              metric: str = "test_macro_f1") -> list[Path]:
    """Vẽ một hình cho mỗi trục ablation. Trả danh sách file đã tạo.

    Mỗi hình trả lời đúng một câu: **biến này đáng bao nhiêu điểm macro-F1?**
    Biên độ (max - min) in ngay trên tiêu đề, vì đó mới là con số đưa vào báo cáo,
    không phải hình dáng đường cong.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if frame.empty or metric not in frame.columns:
        log.warning("Chưa có cột %s — chạy scripts/evaluate.py trước khi vẽ.", metric)
        return []

    frame = frame.dropna(subset=[metric]).copy()
    if frame.empty:
        log.warning("Mọi run đều thiếu số TEST. Chạy: "
                    'python scripts/evaluate.py --runs "artifacts/runs/*"')
        return []

    frame["axis"] = frame["tag"].astype(str).str.split("-").str[0]
    # bỏ hậu tố -budget khi hiển thị, nhưng giữ thông tin để ghi chú
    frame["value"] = (frame["tag"].astype(str)
                      .str.split("-").str[1:].str.join("-")
                      .str.replace("-budget", "", regex=False))
    is_budget = frame["tag"].astype(str).str.contains("budget").any()

    out_dir.mkdir(parents=True, exist_ok=True)
    created = []

    for axis, group in frame.groupby("axis"):
        if axis not in AXIS_XLABEL:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 4))
        for model, part in group.groupby("model"):
            part = part.sort_values("value")
            ax.plot(part["value"].astype(str), part[metric],
                    marker="o", label=model)

        spread = group[metric].max() - group[metric].min()
        best = group.loc[group[metric].idxmax(), "value"]
        title = f"Ablation: {axis} — biên độ {spread:.4f} ({spread*100:.2f} điểm), tốt nhất: {best}"
        if is_budget:
            title += "\n(ngân sách 15 epoch — dùng để XẾP HẠNG, không phải số cuối cùng)"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(AXIS_XLABEL[axis])
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()

        path = out_dir / f"ablation_{axis}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        created.append(path)
        log.info("  %-18s biên độ %.4f (%.2f điểm), tốt nhất %-14s -> %s",
                 axis, spread, spread * 100, str(best), path.name)

    return created


def main() -> None:
    """Sinh danh sách thực nghiệm rồi chạy tuần tự, hoặc chỉ tổng hợp bảng."""
    args = parse_args()

    if args.collect_only:
        frame = collect()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.out, index=False)
        log.info("Tổng hợp %d run -> %s", len(frame), args.out)
        log.info("\n%s", frame.to_string(index=False))

        log.info("")
        log.info("Vẽ hình cho từng trục:")
        created = plot_axes(frame, Path(args.figures))
        log.info("Đã tạo %d hình.", len(created))
        return

    jobs = plan_jobs(args)
    if args.dry_run:
        log.info("")
        log.info("--dry-run: không chạy gì. Bỏ cờ này để chạy thật.")
        return

    failures = execute_jobs(jobs, args)

    log.info("")
    log.info("=" * 68)
    log.info("Xong %d/%d thực nghiệm", len(jobs) - len(failures), len(jobs))
    if failures:
        log.warning("Thất bại: %s", [f"{j['axis']}-{j['tag']}" for j in failures])
    log.info("")
    log.info("Bước tiếp: đánh giá trên test rồi tổng hợp")
    log.info('  python scripts/evaluate.py --runs "artifacts/runs/*"')
    log.info("  python scripts/run_ablation.py --collect-only")


if __name__ == "__main__":
    main()
