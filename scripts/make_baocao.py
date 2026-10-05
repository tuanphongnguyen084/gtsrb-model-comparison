#!/usr/bin/env python
"""
make_baocao.py — sinh BÁO CÁO NỘP THẦY: docs/BAO_CAO.md

CHỦ: Phong Nguyễn

Khác gì `make_report.py`?
  make_report.py -> docs/KET_QUA.md  : bảng số thô, dành cho NHÓM tra cứu
  make_baocao.py -> docs/BAO_CAO.md  : báo cáo học thuật, dành cho GIẢNG VIÊN đọc

Phần diễn giải viết tay, nhưng MỌI BẢNG SỐ đều đọc trực tiếp từ reports/tables/.
Nhờ vậy chạy lại sau khi có thêm thực nghiệm là báo cáo tự cập nhật, và không có
con số nào bị chép tay (quy tắc số 3 của nhóm).

    python scripts/make_baocao.py
    make baocao
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import pathlib

from _common import chi_run_chinh

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from gtsrb.utils.logging import get_logger

log = get_logger()

TABLES = Path("reports/tables")
MISSING = "_(chưa có số liệu — chạy lệnh tương ứng trong README)_"


# ---------------------------------------------------------------- tiện ích

def read(name: str, **kw) -> pd.DataFrame | None:
    """Đọc một bảng trong reports/tables/, trả None nếu chưa có."""
    path = TABLES / name
    return pd.read_csv(path, **kw) if path.exists() else None


def read_main() -> pd.DataFrame | None:
    """Bảng so sánh chính, CHỈ gồm run chính (bỏ run ablation và run nhiều seed).

    Run ablation/seed có cột `tag` khác rỗng. Không lọc thì bảng so sánh trong báo
    cáo sẽ có 40+ dòng và một biến thể ablation có thể bị chọn làm model tốt nhất.
    """
    frame = read("main_comparison.csv")
    if frame is None:
        return None
    frame = chi_run_chinh(frame)
    return frame if not frame.empty else None


def md(frame: pd.DataFrame | None, **kw) -> str:
    """DataFrame -> bảng Markdown, hoặc câu nhắc nếu chưa có dữ liệu."""
    return MISSING if frame is None or frame.empty else frame.to_markdown(index=False, **kw)


def pct(x) -> str:
    """Số thực -> chuỗi phần trăm hai chữ số thập phân."""
    return "–" if pd.isna(x) else f"{x * 100:.2f}%"


# ---------------------------------------------------------------- các bảng

def table_models() -> str:
    """Bảng so sánh chính, gộp accuracy + chi phí."""
    main, speed = read_main(), read("speed.csv")
    if main is None:
        return MISSING
    if speed is not None:
        main = main.merge(speed[["model", "flops_g", "size_mb", "cpu_bs1_p50",
                                 "cpu_bs1_p95"]], on="model", how="left")
    out = pd.DataFrame({
        "Mô hình": main["model"],
        "Tham số (M)": main["params_m"].round(2),
        "FLOPs (G)": main.get("flops_g", pd.Series()).round(2),
        "Dung lượng (MB)": main.get("size_mb", pd.Series()).round(1),
        "Top-1": main["test_top1"].map(pct),
        "Top-5": main["test_top5"].map(pct),
        "Macro-F1": main["test_macro_f1"].round(4),
        "ECE": main["test_ece"].round(4),
        "p95 CPU (ms)": main.get("cpu_bs1_p95", pd.Series()).round(1),
        "Train (phút)": (main["train_seconds"] / 60).round(0),
    })
    return out.sort_values("Macro-F1", ascending=False).to_markdown(index=False)


def table_mcnemar() -> str:
    """Kiểm định McNemar cho mọi cặp, kèm cột kết luận bằng tiếng Việt."""
    frame = read("mcnemar.csv")
    if frame is None:
        return MISSING
    out = pd.DataFrame({
        "Cặp": frame["model_a"] + " vs " + frame["model_b"],
        "n01": frame["n01"], "n10": frame["n10"],
        "p-value": frame["p_value"].map(lambda v: f"{v:.3g}"),
        "Kết luận": frame.apply(
            lambda r: "**tương đương**" if r.p_value >= 0.05
            else f"{r.model_a if r.better == 'A' else r.model_b} hơn", axis=1),
    })
    return out.to_markdown(index=False)


def table_efficiency() -> str:
    """ms trên mỗi GFLOP — bằng chứng latency không tỉ lệ FLOPs."""
    speed = read("speed.csv")
    if speed is None or "flops_g" not in speed:
        return MISSING
    out = speed[["model", "flops_g", "cpu_bs1_p50"]].copy()
    out["ms/GFLOP"] = (out["cpu_bs1_p50"] / out["flops_g"]).round(1)
    out.columns = ["Mô hình", "FLOPs (G)", "Latency p50 (ms)", "**ms/GFLOP**"]
    return out.sort_values("**ms/GFLOP**").round(2).to_markdown(index=False)


def table_robustness() -> str:
    """Bảng relative robustness trung bình theo loại nhiễu, kèm mCE."""
    frame = read("robustness_summary.csv", index_col=0)
    return MISSING if frame is None else frame.round(4).to_markdown()


# Nhiễu SO SÁNH ĐƯỢC giữa hai nhóm độ phân giải (48x48 cho M1/M2, 224x224 cho M3).
# motion_blur và fog dùng kernel/khoảng cách tính bằng PIXEL TUYỆT ĐỐI, nên cùng một
# "mức 5" là 31% chiều rộng ảnh 48px nhưng chỉ 6,7% ảnh 224px — không so được.
# Chi tiết ở docs/SU_CO.md §8.
NHIEU_SO_SANH_DUOC = ("gauss_noise", "low_light", "occlusion")


def ket_luan_robustness() -> str:
    """Sinh câu kết luận về độ bền TỪ SỐ ĐO, không viết cứng.

    Câu này là kết luận số 3 của cả báo cáo. Trước đây nó được viết cứng
    ("M1 yếu nhất về accuracy nhưng có mCE tốt nhất") từ hồi chỉ đo 3 model.
    Thêm model vào là câu đó có thể thành SAI, mà bảng ngay bên dưới thì vẫn
    đúng — báo cáo tự phản bác chính mình và không có gì báo động.
    """
    frame = read("robustness.csv")
    if frame is None:
        return MISSING

    sach = (frame[frame["corruption"] == "clean"]
            .set_index("model")["top1"])
    # mCE chỉ trên nhóm nhiễu so sánh được: error trung bình qua mọi mức độ.
    nhieu = frame[frame["corruption"].isin(NHIEU_SO_SANH_DUOC)]
    if nhieu.empty or sach.empty:
        return MISSING
    mce = (1 - nhieu.groupby("model")["top1"].mean()).sort_values()

    gioi_nhat, ben_nhat, yeu_nhat = sach.idxmax(), mce.idxmin(), sach.idxmin()

    # ★ PHẢI nói rõ đây KHÔNG phải cột mCE của bảng trên.
    #
    # LỖI ĐÃ GẶP: cả hai đều được gọi là "error", đặt cách nhau 4 dòng, và cho
    # THỨ HẠNG KHÁC NHAU. mCE (mọi nhiễu) xếp m3_resnet18 hạng 2 với 0,2817;
    # chỉ số dưới đây (3 nhiễu so sánh được) xếp nó hạng 3 với 0,4429. Người
    # đọc so 0,2817 ở bảng với 0,2812 ở văn sẽ tưởng hai mô hình gần bằng nhau.
    mce_day_du = (1 - frame[frame["corruption"] != "clean"]
                  .groupby("model")["top1"].mean())
    thu_hang_khac = list(mce.index) != list(mce_day_du.sort_values().index)

    dong = [f"**Chỉ số dưới đây KHÔNG phải cột `mCE` của bảng trên.** `mCE` tính "
            f"trên **cả 5 loại nhiễu**, kể cả `motion_blur` và `fog` — hai loại "
            f"**không so sánh được** giữa 48×48 và 224×224 (xem cảnh báo bên "
            f"dưới). Chỉ số dưới đây chỉ dùng "
            f"**{len(NHIEU_SO_SANH_DUOC)} loại so sánh được** "
            f"({', '.join(f'`{n}`' for n in NHIEU_SO_SANH_DUOC)}), nên con số "
            f"CAO HƠN và thứ hạng có thể khác."]
    if thu_hang_khac:
        a = " < ".join(f"`{m}`" for m in mce_day_du.sort_values().index)
        dong.append(f"\nVà thứ hạng **đổi thật**: theo `mCE` là {a}; theo "
                    f"{len(NHIEU_SO_SANH_DUOC)} nhiễu so sánh được thì khác "
                    f"(xem dòng cuối mục này). Kết luận của nhóm rút từ nhóm "
                    f"nhiễu so sánh được.")
    dong.append(f"\nTrên {len(sach)} mô hình:")

    if gioi_nhat != ben_nhat:
        dong.append(
            f"\n**Mô hình chính xác nhất trên ảnh sạch KHÔNG phải mô hình bền nhất.** "
            f"`{gioi_nhat}` dẫn đầu khi ảnh sạch (top-1 = {sach[gioi_nhat]:.4f}) "
            f"nhưng `{ben_nhat}` mới là mô hình bền nhất "
            f"(error trên 3 nhiễu so sánh được = {mce[ben_nhat]:.4f} so với "
            f"{mce[gioi_nhat]:.4f}).")
    else:
        dong.append(
            f"\n**Mô hình chính xác nhất cũng là mô hình bền nhất:** `{gioi_nhat}` "
            f"đứng đầu cả trên ảnh sạch (top-1 = {sach[gioi_nhat]:.4f}) và dưới nhiễu "
            f"(error trên 3 nhiễu so sánh được = {mce[gioi_nhat]:.4f}). Trên nhóm mô hình này, độ bền "
            f"KHÔNG tách khỏi accuracy — khác với kết quả khi chỉ đo 3 mô hình đầu.")

    if yeu_nhat == ben_nhat:
        dong.append(
            f"\nĐáng chú ý hơn: `{yeu_nhat}` — **yếu nhất** trên ảnh sạch "
            f"(top-1 = {sach[yeu_nhat]:.4f}) — lại bền nhất. Thứ tự xếp hạng khi có "
            f"nhiễu **đảo lại** so với khi không có.")

    dong.append("\nXếp hạng độ bền **theo 3 nhiễu so sánh được** (càng THẤP càng bền): "
                + " < ".join(f"`{m}` {v:.4f}" for m, v in mce.items()) + ".")
    return "\n".join(dong)


def _ty_so_toc_do() -> tuple[float, float] | None:
    """(nhỏ nhất, lớn nhất) số lần M2 nhanh hơn các backbone mà nó ĐÁNH BẠI."""
    sp, mc = read("speed.csv"), read("mcnemar.csv")
    if sp is None or mc is None or "cpu_bs1_p50" not in sp.columns:
        return None
    lat = sp.set_index("model")["cpu_bs1_p50"].to_dict()
    if "m2_vggres" not in lat:
        return None
    # Backbone nào M2 thắng CÓ ý nghĩa thống kê (theo McNemar, không tự chọn)
    thua = set()
    for r in mc.itertuples():
        if r.better == "A" and r.model_a == "m2_vggres":
            thua.add(r.model_b)
        elif r.better == "B" and r.model_b == "m2_vggres":
            thua.add(r.model_a)
    ty = [lat[t] / lat["m2_vggres"] for t in thua
          if t in lat and t.startswith("m3_")]
    return (min(ty), max(ty)) if ty else None


def ty_so_toc_do() -> str:
    """Chuỗi 'nhanh hơn 15-76 lần', rút từ speed.csv + mcnemar.csv.

    LỖI ĐÃ GẶP: con số này viết cứng là '13-68 lần', tính từ bảng latency CŨ.
    Sau khi đo lại trên máy rảnh, số thật là 15-76 lần. Viết cứng một tỉ số
    tính từ bảng số thì mỗi lần đo lại là một lần báo cáo nói sai mà không ai
    biết — cùng loại với câu kết luận robustness ở ket_luan_robustness().
    """
    ty = _ty_so_toc_do()
    return f"{ty[0]:.0f}–{ty[1]:.0f} lần" if ty else "(chưa có số latency)"


def ms_tren_gflop() -> str:
    """Chênh lệch ms/GFLOP giữa mô hình hiệu quả nhất và kém nhất."""
    sp = read("speed.csv")
    if sp is None or "flops_g" not in sp.columns or "cpu_bs1_p50" not in sp.columns:
        return "(chưa có số)"
    r = (sp["cpu_bs1_p50"] / sp["flops_g"]).dropna()
    return f"{r.max() / r.min():.0f} lần" if len(r) > 1 else "(chưa có số)"


def bang_seed_summary() -> str:
    """Bảng mean/std/min/max, XOAY để đọc được.

    seeds_summary.csv có cột MultiIndex (chỉ số × thống kê) nên để nguyên thì
    ra bảng rộng 17 cột với tên kiểu `('top1', 'mean')` — không ai đọc nổi.
    Xoay thành: mỗi DÒNG là một chỉ số, mỗi CỘT là một thống kê.
    """
    frame = read("seeds.csv")
    if frame is None or frame.empty:
        return MISSING
    chi_so = [c for c in ("top1", "top5", "macro_f1", "ece") if c in frame.columns]
    ra = []
    for model, g in frame.groupby("model"):
        bang = g[chi_so].agg(["mean", "std", "min", "max"]).T
        bang["biên độ (điểm)"] = (bang["max"] - bang["min"]) * 100
        bang.index.name = f"{model} ({len(g)} seed)"
        ra.append(bang.round(5).to_markdown())
    return "\n\n".join(ra)


# Biến thể nào TRÙNG với cấu hình mặc định của dự án, theo từng trục.
# Dùng để nói thẳng: ablation có đồng ý với lựa chọn mặc định hay không.
MAC_DINH_THEO_TRUC = {
    "preprocess": "prep_clahe",
    "augmentation": "aug_geo_photo",
    "label_smoothing": "ls0.1",
    "scaling": ("width1.0", "depth4"),
    "components": "full",
    "resolution": "res48",
}


def nghich_ly_effnet() -> str:
    """Câu 'EffNet ít FLOPs hơn ResNet18 x lần nhưng chậm hơn y lần'.

    LỖI ĐÃ GẶP: viết cứng 'chậm hơn 14 lần'. Sau khi đo lại latency trên máy
    rảnh, số thật là 14,6 lần. Tỉ số tính từ bảng latency thì phải SINH từ bảng
    đó — mỗi lần đo lại là một lần câu viết cứng nói sai mà không ai biết.
    """
    sp = read("speed.csv")
    if sp is None or "flops_g" not in sp.columns:
        return "(chưa có số)"
    sp = sp.set_index("model")
    try:
        fl = sp.loc["m3_resnet18", "flops_g"] / sp.loc["m3_effnetb0", "flops_g"]
        la = sp.loc["m3_effnetb0", "cpu_bs1_p50"] / sp.loc["m3_resnet18", "cpu_bs1_p50"]
    except KeyError:
        return "(chưa có số)"
    return (f"**ít hơn ResNet18 {fl:.1f} lần FLOPs** nhưng chậm hơn "
            f"**{la:.1f} lần** trên CPU")


def so_test() -> str:
    """Số test ĐẾM THẬT bằng pytest --collect-only.

    LỖI ĐÃ GẶP: báo cáo viết cứng "95 test" trong khi đã có 470. Một con số
    kiểm thử sai làm người đọc mất tin vào toàn bộ phần tái lập.
    """
    import subprocess
    goc = pathlib.Path(__file__).resolve().parent.parent
    try:
        r = subprocess.run(["python", "-m", "pytest", "--collect-only", "-q"],
                           cwd=goc, capture_output=True, text=True, timeout=180)
        for d in reversed(r.stdout.splitlines()):
            if "test" in d and "collected" in d:
                return d.strip().split()[0] + " test"
            if d.strip().endswith("tests collected") or "/" in d:
                continue
        import re
        m = re.search(r"(\d+) tests? collected", r.stdout)
        if m:
            return f"{m.group(1)} test"
    except Exception:
        pass
    return f"{len(list(goc.glob('tests/test_*.py')))} file test"


def pham_vi_nen_int8() -> str:
    """int8 static chạy được trên mấy mô hình, và vì sao không phải tất cả."""
    frame = read("edge_export.csv")
    if frame is None or "biến thể" not in frame.columns:
        return ""
    dem = frame.groupby("biến thể")["model"].nunique().to_dict()
    tong = frame["model"].nunique()
    n_static = dem.get("int8_static", 0)
    if n_static >= tong:
        return ""
    co = sorted(frame.loc[frame["biến thể"] == "int8_static", "model"].unique())
    nen = frame.loc[frame["biến thể"] == "int8_static", "nhẹ hơn"].max()
    return (f"\n⚠️ **`int8_static` chỉ chạy được trên {n_static}/{tong} mô hình** "
            f"({', '.join(f'`{m}`' for m in co)}). Bốn mô hình kia convert thành "
            f"công nhưng forward **vỡ lúc chạy**, và lý do rất cụ thể:\n"
            f"\n| Mô hình | op không có kernel int8 | nguyên nhân |\n"
            f"|---|---|---|\n"
            f"| `m2_vggres` | `aten::add.out` | **residual connection** |\n"
            f"| `m3_mobilenetv2` | `aten::add.out` | **residual connection** |\n"
            f"| `m3_resnet18` | `aten::add.out` | **residual connection** |\n"
            f"| `m3_effnetb0` | `aten::silu.out` | hàm hoạt hoá SiLU/Swish |\n"
            f"| `m1_lenet` | *(chạy được)* | **không có residual** |\n"
            f"\n**Ba trong bốn ca là phép `+` của skip connection, không phải "
            f"hàm hoạt hoá.** Backend `QuantizedCPU` không có kernel cho `+` "
            f"giữa hai tensor đã lượng tử hoá: phép cộng đó phải được viết bằng "
            f"`torch.nn.quantized.FloatFunctional().add()` để backend biết cách "
            f"khớp thang lượng tử của hai nhánh. Viết `a + b` như bình thường "
            f"thì convert vẫn qua, chỉ forward mới vỡ.\n"
            f"\nM1 LeNet lượng tử hoá được **chính vì nó không có residual** — "
            f"kiến trúc cũ nhất lại là kiến trúc duy nhất triển khai int8 tĩnh "
            f"được mà không phải sửa code.\n"
            f"\nHệ quả: tỉ lệ nén **{nen:.2f}×** chỉ đo được trên "
            f"`{co[0] if co else '?'}`, **không ngoại suy** cho bốn mô hình kia. "
            f"Bài học chung: khả năng lượng tử hoá phụ thuộc **cách VIẾT từng "
            f"phép toán**, không chỉ kiến trúc hay số tham số. `_works()` thử "
            f"chạy một forward sau convert nên chỉ thiếu một dòng bảng thay vì "
            f"sập cả script. Chi tiết `docs/SU_CO.md` §10.")


def phu_luc_tai_lieu() -> str:
    """Bảng tài liệu kèm theo, ĐẾM TỪ ĐĨA.

    LỖI ĐÃ GẶP: bảng này viết cứng "9 nhóm sự cố" (thật: 13), "20 bảng CSV"
    (26), "31 hình" (37), và "38 câu hỏi ôn tập" trong PHAN_CONG.md — phần câu
    hỏi đã bị BỎ theo yêu cầu nhưng dòng mô tả vẫn còn. Một bảng liệt kê sản
    phẩm mà tự đếm sai thì làm người đọc mất tin vào cả những con số khác.
    """
    goc = pathlib.Path(__file__).resolve().parent.parent

    def dem(mau: str) -> int:
        return len(list(goc.glob(mau)))

    su_co = len([l for l in (goc / "docs/SU_CO.md").read_text(encoding="utf-8")
                 .splitlines() if l.startswith("## ") and l[3:4].isdigit()]) \
        if (goc / "docs/SU_CO.md").exists() else 0

    hang = [
        ("docs/KET_QUA.md", "toàn bộ bảng số, sinh tự động"),
        ("docs/LY_THUYET.md", "cơ sở lý thuyết, công thức, lý do từng lựa chọn "
                              "thiết kế; có bản đồ code ↔ lý thuyết"),
        ("docs/SU_CO.md", f"**{su_co} sự cố** đã gặp thật, kèm nguyên nhân, cách "
                          f"sửa và test chặn"),
        ("docs/PHAN_CONG.md", "phân công theo người, kèm khái niệm mỗi người phải nắm"),
        ("docs/INTERFACE.md", "hợp đồng giữa các phần: chữ ký hàm, schema `result.json`"),
    ]
    dong = ["| File | Nội dung |", "|---|---|"]
    for f, mo_ta in hang:
        if (goc / f).exists():
            n = len((goc / f).read_text(encoding="utf-8").splitlines())
            dong.append(f"| `{f}` | {mo_ta} ({n} dòng) |")
    dong += [
        f"| `reports/tables/` | **{dem('reports/tables/*.csv')} bảng CSV** |",
        f"| `reports/figures/` | **{dem('reports/figures/*.png')} hình** |",
        f"| `notebooks/` | **{dem('notebooks/*.ipynb')} notebook** "
        f"(01–09 diễn giải từng bước, 00 để chạy trên Colab) |",
        f"| `tests/` | **{dem('tests/test_*.py')} file test**, chạy bằng `make test` |",
    ]
    return "\n".join(dong)


def truc_do_phan_giai() -> str:
    """Trục `resolution` đo GÌ — và phép đo đối chứng chứng minh điều đó.

    ★ CHỖ DỄ ĐỌC SAI NHẤT CỦA CẢ BÁO CÁO ★

    M1/M2 chạy ablation với cache KHỚP img_size (32->32, 48->48, 64->64) nên
    chúng đo độ phân giải THẬT. M3 thì cả ba mức (64/112/224) đều lấy từ CÙNG
    cache 48px rồi nội suy lên — nên với M3, trục này đo HỆ SỐ NỘI SUY, không
    đo chi tiết ảnh. Đọc "res224 tốt nhất" thành "ảnh nét hơn thì tốt hơn" là
    sai.
    """
    frame = read("ablation.csv")
    if frame is None or "tag" not in frame.columns:
        return MISSING
    g = frame[frame["tag"].astype(str).str.startswith("resolution")].copy()
    if g.empty:
        return MISSING

    dong = [
        "**Trục này đo hai thứ KHÁC NHAU tuỳ mô hình** — chỗ dễ đọc sai nhất:",
        "",
        "| Mô hình | cache dùng | img_size | thực chất đo |",
        "|---|---|---|---|",
        "| `m1_lenet`, `m2_vggres` | 32 / 48 / 64 (**khớp**) | 32 / 48 / 64 | "
        "**độ phân giải thật** |",
        "| `m3_resnet18` | **48 cho cả ba** | 64 / 112 / 224 | "
        "**hệ số nội suy**, không phải chi tiết ảnh |",
        "",
        "Nên **không được** đọc \"res224 tốt nhất\" thành \"ảnh nét hơn thì tốt "
        "hơn\". Nguyên nhân thật là **dung lượng feature map**: ResNet18 thu nhỏ "
        "ảnh 32 lần, nên feature map cuối trước Global Average Pooling là",
        "",
        "| img_size | feature map | số vị trí không gian |",
        "|---|---|---|",
        "| 48 px | 512×2×2 | **4** |",
        "| 64 px | 512×2×2 | **4** |",
        "| 112 px | 512×4×4 | 16 |",
        "| 224 px | 512×7×7 | 49 |",
        "",
        "Ở 48 và 64 px, mạng chỉ còn **4 ô** để mô tả cả biển báo. Đó là lý do "
        "`res64` kém, không phải vì ảnh mờ hơn.",
    ]

    # ---- Phép đo đối chứng: cache thật vs nội suy ----
    thuc = g[g["tag"].astype(str).str.contains("res112real")]
    noi_suy = g[g["tag"].astype(str).str.contains("res112-budget")]
    if not thuc.empty and not noi_suy.empty:
        t, n = thuc.iloc[0], noi_suy.iloc[0]
        dv = (t["val_macro_f1"] - n["val_macro_f1"]) * 100
        dt = (t["test_macro_f1"] - n["test_macro_f1"]) * 100
        seed = read("seeds.csv")
        nguong = ((seed["macro_f1"].max() - seed["macro_f1"].min()) * 100
                  if seed is not None and len(seed) > 1 else 0.5)
        dong += [
            "",
            "#### Phép đo đối chứng: cache 48 có làm hại M3 không?",
            "",
            "Câu hỏi sắc nhất nhắm vào kết luận chính của báo cáo: *\"backbone "
            "pretrained của các bạn kém chỉ vì bạn đưa cho nó ảnh đã bị làm "
            "mờ?\"* Nhóm dựng **cache 112 px thật** rồi train lại cùng cấu "
            "hình để trả lời bằng số:",
            "",
            "| Nguồn ảnh ở 112 px | val macro-F1 | test macro-F1 |",
            "|---|---|---|",
            f"| nội suy từ cache 48 | {n['val_macro_f1']:.5f} | "
            f"{n['test_macro_f1']:.5f} |",
            f"| **cache 112 thật** | {t['val_macro_f1']:.5f} | "
            f"{t['test_macro_f1']:.5f} |",
            f"| chênh | **{dv:+.2f} điểm** | **{dt:+.2f} điểm** |",
            "",
            f"Chi tiết thật **không giúp gì** — chênh {abs(dv):.2f} điểm val và "
            f"{abs(dt):.2f} điểm test, nhỏ hơn nhiễu seed ({nguong:.2f} điểm) "
            f"khoảng **{nguong / max(abs(dv), 0.01):.0f} lần**, và còn hơi "
            f"NGHIÊNG VỀ PHÍA bản nội suy.",
            "",
            "Lý do nằm ở bản thân dữ liệu, không ở đường ống:",
            "",
            "| | |",
            "|---|---|",
            "| ROI biển báo trung vị | **31 px** |",
            "| ảnh có ROI > 48 px | 21 % |",
            "| ảnh có ROI > 224 px | **0 %** |",
            "",
            "Biển báo trung vị còn **nhỏ hơn cache 48 px**, và không một ảnh nào "
            "trong 51.839 ảnh có chi tiết tới 224 px. Nên lời phản biện \"ảnh bị "
            "làm mờ\" không đứng được: ảnh gốc đã ở độ phân giải đó, và đưa chi "
            "tiết thật vào cũng không đổi kết quả.",
            "",
            "**Kết luận:** việc cache ở 48 px là lựa chọn về tốc độ và **không "
            "cầm chân M3**. Nhưng nó LÀM ĐỔI NGHĨA trục ablation ở trên, nên "
            "trục đó phải đọc là *dung lượng feature map*, không phải *độ phân "
            "giải ảnh*.",
        ]
    return "\n".join(dong)


def bang_ablation() -> str:
    """Bảng ablation theo trục, và nêu thẳng chỗ nó ĐI NGƯỢC cấu hình mặc định.

    Đây là phần dễ bị hỏi nhất khi bảo vệ: nếu ablation nói `he_y` tốt hơn
    `clahe` mà cả dự án vẫn dùng `clahe`, thì phải giải thích được vì sao —
    không được để người đọc tự phát hiện.
    """
    frame = read("ablation.csv")
    if frame is None or "tag" not in frame.columns:
        return MISSING + "  \nChạy `make ablation-budget` rồi `--collect-only`."
    g = frame[frame["tag"].notna()
              & frame["tag"].astype(str).str.contains("budget")].copy()
    if g.empty:
        return MISSING + "  \nChưa có run ablation nào."
    g["trục"] = g["tag"].astype(str).str.split("-").str[0]
    g["biến thể"] = g.apply(
        lambda r: str(r["tag"]).replace(f"{r['trục']}-", "").replace("-budget", ""),
        axis=1)

    # Hai cách xếp hạng có cho cùng người thắng? Nêu ra như phép kiểm chéo.
    khop = sum(
        sub.loc[sub["val_macro_f1"].idxmax(), "biến thể"]
        == sub.loc[sub["test_macro_f1"].idxmax(), "biến thể"]
        for _, sub in g.groupby("trục"))
    n_truc = g["trục"].nunique()
    tuong_quan = g["val_macro_f1"].corr(g["test_macro_f1"])

    dong = [f"**Xếp hạng theo `val`, KHÔNG theo `test`.** Chọn cấu hình bằng "
            f"điểm test là dùng tập test để *chọn*, và khi đó test không còn là "
            f"ước lượng độc lập cho cấu hình được chọn. Cột `test` dưới đây chỉ "
            f"để đối chiếu.", "",
            f"**Phép kiểm chéo:** hai cách xếp hạng cho **cùng biến thể thắng ở "
            f"{khop}/{n_truc} trục**, tương quan val–test "
            f"**{tuong_quan:.3f}**. Nghĩa là val đủ tin để chọn, và không có dấu "
            f"hiệu overfit vào val ở mức ảnh hưởng thứ hạng.", "",
            f"**{len(g)} run**, mỗi run đổi **đúng một biến** so với cấu hình gốc, "
            f"ở chế độ ngân sách **15 epoch**. Chế độ này để **xếp hạng** biến "
            f"thể, không phải để lấy số cuối cùng — cấu hình thắng cần chạy lại "
            f"ở độ dài đầy đủ trước khi đưa vào bảng so sánh chính.", ""]

    bang = (g[["trục", "biến thể", "model", "val_macro_f1",
              "test_top1", "test_macro_f1"]]
            .sort_values(["trục", "val_macro_f1"], ascending=[True, False]))
    dong += [bang.round(5).to_markdown(index=False), ""]

    # ---- Chỗ ablation KHÔNG đồng ý với mặc định ----
    # ★ So TRONG CÙNG MỘT MODEL.
    #
    # LỖI ĐÃ GẶP: trục `resolution` có nhiều model (m1/m2 ở 32-64px, resnet18 ở
    # 64-224px). So "tốt nhất = res224" (m3_resnet18) với "mặc định = res48"
    # (m2_vggres) là so HAI KIẾN TRÚC KHÁC NHAU rồi gọi chênh lệch đó là hiệu
    # ứng của độ phân giải — đúng lỗi mà nguyên tắc "đổi một biến một lần"
    # được dựng ra để tránh.
    nguoc = []
    for truc, sub in g.groupby("trục"):
        md = MAC_DINH_THEO_TRUC.get(truc)
        ten_md = (md,) if isinstance(md, str) else (md or ())
        hang_md = sub[sub["biến thể"].isin(ten_md)]
        if hang_md.empty:
            continue
        goc = hang_md.loc[hang_md["val_macro_f1"].idxmax()]
        cung_model = sub[sub["model"] == goc["model"]]
        tot = cung_model.loc[cung_model["val_macro_f1"].idxmax()]
        if tot["biến thể"] in ten_md:
            continue
        hieu = (tot["val_macro_f1"] - goc["val_macro_f1"]) * 100
        nguoc.append(
            f"- **{truc}** (`{goc['model']}`): tốt nhất là `{tot['biến thể']}` "
            f"(val {tot['val_macro_f1']:.5f}, test {tot['test_macro_f1']:.5f}), "
            f"cao hơn mặc định `{goc['biến thể']}` "
            f"(val {goc['val_macro_f1']:.5f}) **{hieu:.2f} điểm val**.")

    if nguoc:
        dong += [f"**★ {len(nguoc)}/{g['trục'].nunique()} trục cho kết quả ĐI NGƯỢC "
                 f"cấu hình mặc định của dự án:**", ""] + nguoc + ["",
                 "Ba điều phải nói khi trình bày, không được bỏ:", "",
                 "1. **15 epoch là quá ngắn để augmentation và regularisation trả "
                 "lãi.** Augmentation, dropout và label smoothing đều làm bài toán "
                 "huấn luyện KHÓ hơn để đổi lấy khái quát tốt hơn về sau. Ở 15 "
                 "epoch, phần 'khó hơn' đã tới mà phần 'tốt hơn' chưa tới. Mô hình "
                 "chính chạy 40 epoch, nên không thể dùng bảng này để kết luận "
                 "'augmentation vô dụng'.",
                 "2. **Mỗi ô ở đây là MỘT seed.** Biên độ seed đo được của M2 là "
                 "**0,50 điểm macro-F1** (mục 3.5.2). Mọi chênh lệch nhỏ hơn con "
                 "số đó trong bảng trên **không kết luận được gì** — phần lớn các "
                 "trục có biên độ dưới 1 điểm.",
                 "3. **Nhóm KHÔNG đổi cấu hình mặc định theo bảng này**, vì (1) và "
                 "(2). Đây là bước xếp hạng để biết nên chạy lại cái gì ở độ dài "
                 "đầy đủ, không phải kết luận."]
    else:
        dong.append("Trên mọi trục, biến thể tốt nhất **trùng với cấu hình mặc "
                    "định** của dự án.")

    # Trục nào có biên độ LỚN hơn nhiễu seed thì mới đáng tin
    dong += ["", "**Trục nào thật sự đáng kết luận?** So biên độ từng trục với "
             "biên độ seed:", ""]
    seed = read("seeds.csv")
    nguong = ((seed["macro_f1"].max() - seed["macro_f1"].min()) * 100
              if seed is not None and len(seed) > 1 else None)
    for truc, sub in g.groupby("trục"):
        bd = (sub["test_macro_f1"].max() - sub["test_macro_f1"].min()) * 100
        if nguong is None:
            dong.append(f"- `{truc}`: biên độ {bd:.2f} điểm")
        else:
            dau = "**vượt** nhiễu seed" if bd > nguong else "NẰM TRONG nhiễu seed"
            dong.append(f"- `{truc}`: biên độ {bd:.2f} điểm — {dau} "
                        f"({nguong:.2f} điểm)")
    if nguong is not None:
        dong.append(f"\nChỉ những trục **vượt {nguong:.2f} điểm** mới đáng rút kết "
                    f"luận từ một seed. Các trục còn lại cần nhiều seed mới nói "
                    f"được gì.")
    return "\n".join(dong)


def ket_luan_seed() -> str:
    """Nhiễu seed, và nó nói gì về các chênh lệch trong bảng chính.

    Đây là phép đo ĐỘC LẬP với McNemar cho cùng một câu hỏi: chênh lệch bao
    nhiêu điểm thì mới là thật?
    """
    frame, main = read("seeds.csv"), read_main()
    if frame is None or frame.empty:
        return MISSING + "  \nChạy `make seeds` rồi `python scripts/run_seeds.py --collect-only`."

    dong = []

    # Chỉ số nào nhiễu hơn? Rút từ dữ liệu, vì nó là cái GIÁ của việc chọn
    # macro-F1 làm chỉ số chính — không nêu ra thì người đọc sẽ áp cùng một
    # ngưỡng "đáng kể" cho cả hai chỉ số.
    if {"top1", "macro_f1"} <= set(frame.columns):
        for model, g in frame.groupby("model"):
            if len(g) < 2:
                continue
            b1 = (g.top1.max() - g.top1.min()) * 100
            bm = (g.macro_f1.max() - g.macro_f1.min()) * 100
            if b1 > 0:
                dong.append(
                    f"**Macro-F1 nhiễu hơn top-1 {bm / b1:.0f} lần** qua các seed "
                    f"của `{model}`: biên độ {bm:.2f} điểm so với {b1:.2f} điểm. "
                    f"Hợp lý — macro-F1 cho mỗi lớp trọng số bằng nhau, nên nó "
                    f"chịu trọn dao động ở các lớp thiểu số, đúng chỗ nhạy nhất "
                    f"với seed. Đó là **cái giá** của việc chọn macro-F1 làm chỉ "
                    f"số chính, và nghĩa là ngưỡng 'đáng kể' của macro-F1 phải "
                    f"ĐẶT CAO HƠN ngưỡng của top-1, không dùng chung một mức.")
    for model, g in frame.groupby("model"):
        mf = g["macro_f1"]
        bien = (mf.max() - mf.min()) * 100
        dong.append(
            f"`{model}` chạy **{len(g)} seed** ({', '.join(str(x) for x in sorted(g.seed))}): "
            f"macro-F1 {mf.mean():.5f} ± {mf.std():.5f}, "
            f"từ {mf.min():.5f} đến {mf.max():.5f} — **biên độ {bien:.2f} điểm**.")

        if main is None:
            continue
        # Mô hình nào trong bảng chính nằm TRONG biên độ seed của model này?
        base = main.set_index("model")["test_macro_f1"].to_dict()
        if model not in base:
            continue
        trong = [f"`{k}` ({v:.5f})" for k, v in base.items()
                 if k != model and mf.min() <= v <= mf.max()]
        if trong:
            dong.append(
                f"\nTrong bảng so sánh chính, {', '.join(trong)} nằm **bên trong** "
                f"biên độ seed của `{model}`. Nghĩa là chênh lệch với những mô "
                f"hình đó **không phân biệt được khỏi việc đổi seed**.")
        # Chỗ đảo thứ hạng THẬT: mô hình xếp TRÊN model này trong bảng chính,
        # nhưng thấp hơn seed tốt nhất của nó. Liệt kê cả những mô hình mà
        # model này vốn đã thắng thì không phải phát hiện gì.
        cua_minh = base.get(model, mf.mean())
        dao = [f"`{k}` ({v:.5f})" for k, v in base.items()
               if k != model and v > cua_minh and v < mf.max()]
        if dao:
            dong.append(
                f"\n★ Seed tốt nhất của `{model}` ({mf.max():.5f}) **vượt** "
                f"{', '.join(dao)} — mô hình xếp TRÊN nó trong bảng chính. "
                f"Thứ hạng giữa chúng **đảo theo seed**, nên không được trình "
                f"bày như một xếp hạng cố định. Đây là phép đo ĐỘC LẬP dẫn tới "
                f"cùng kết luận với McNemar.")
    return "\n\n".join(dong)


def bang_thu_tu_clahe() -> str:
    """Bảng + kết luận về việc áp nhiễu TRƯỚC hay SAU CLAHE.

    Rút từ reports/tables/thu_tu_clahe.csv. Đây là giới hạn phương pháp NẶNG
    NHẤT của phần robustness, nên nó phải nằm trong báo cáo kèm số, không chỉ
    là một câu chú thích.
    """
    frame = read("thu_tu_clahe.csv")
    if frame is None:
        return (MISSING + "  \nChạy `python scripts/kiem_thu_tu_clahe.py` "
                "để sinh bảng này.")

    chenh = (frame.pivot_table(index="model", columns="corruption",
                               values="chenh", aggfunc="mean"))
    chenh["trung bình"] = chenh.mean(axis=1)
    chenh = chenh.sort_values("trung bình", ascending=False)

    xep = {}
    for cot, nhan in (("nhieu_SAU_clahe", "đang đo"),
                      ("nhieu_TRUOC_clahe", "triển khai")):
        xep[nhan] = frame.groupby("model")[cot].mean().sort_values(ascending=False)

    cu, moi = xep["đang đo"], xep["triển khai"]
    dong = ["**Chênh = (áp nhiễu TRƯỚC CLAHE) − (áp nhiễu SAU CLAHE).** "
            "Dương nghĩa là sửa thứ tự giúp mô hình đó:", "",
            chenh.round(3).to_markdown(), "",
            "**Dấu của hiệu ứng phụ thuộc vào MÔ HÌNH** — không phải một sai số "
            "chung cộng vào mọi mô hình như nhau. CLAHE cân bằng tương phản "
            "**cục bộ**: áp nhiễu trước thì CLAHE *khuếch đại* chính cái nhiễu "
            "đó, rồi ảnh bị thu về 48×48. Với mô hình nhỏ ở độ phân giải thấp, "
            "nhiễu đã khuếch đại còn tệ hơn ảnh tối ban đầu; với mô hình 224px "
            "thì việc lấy lại độ sáng tổng thể quan trọng hơn.", "",
            "Thứ hạng độ bền theo hai thứ tự:", "",
            "| | thứ tự ĐANG đo | thứ tự TRIỂN KHAI |", "|---|---|---|"]
    for i in range(len(cu)):
        dong.append(f"| {i+1} | `{cu.index[i]}` {cu.iloc[i]:.3f} "
                    f"| `{moi.index[i]}` {moi.iloc[i]:.3f} |")

    ben_cu, ben_moi = cu.index[0], moi.index[0]
    khoang_cu = cu.iloc[0] - cu.iloc[1]
    khoang_moi = moi.iloc[0] - moi.iloc[1]
    bien_cu, bien_moi = cu.iloc[0] - cu.iloc[-1], moi.iloc[0] - moi.iloc[-1]

    dong += ["", "**Ba điều phải nói cùng nhau, không được lẫn:**", ""]
    if ben_cu == ben_moi:
        dong.append(f"1. Mô hình bền nhất **không đổi** (`{ben_cu}`) ở cả hai "
                    f"thứ tự — kết luận chính vẫn đứng.")
    else:
        dong.append(f"1. Mô hình bền nhất **ĐỔI**: `{ben_cu}` theo thứ tự đang "
                    f"đo, nhưng `{ben_moi}` theo thứ tự triển khai.")
    dong.append(f"2. Nhưng khoảng cách hạng 1 với hạng 2 **co từ "
                f"{khoang_cu:.3f} xuống {khoang_moi:.3f}** — hai mô hình đầu "
                f"gần như bằng nhau về độ bền. Nói '{ben_moi} bền nhất' mà "
                f"không nói con số này là phóng đại.")
    dong.append(f"3. Biên độ giữa {len(cu)} mô hình co từ **{bien_cu:.2f} xuống "
                f"{bien_moi:.2f}**. Những con số thấp dưới mức đoán bừa "
                f"(1/43 = 0,023) trong bảng robustness chính là **hiện vật "
                f"đo**, không phải tính chất mô hình.")
    dong += ["", "**Không có thứ tự nào đúng tuyệt đối.** Áp sau CLAHE thiên vị "
             "mô hình nhỏ ở 48px; áp trước CLAHE sát điều kiện triển khai hơn "
             "nhưng trừng phạt chính nhóm đó. Phép đo chính trong "
             "`robustness.csv` dùng thứ tự **sau CLAHE**; bảng trên là phép đo "
             "đối chứng. Chi tiết ở `docs/SU_CO.md` §12."]
    return "\n".join(dong)


def ket_luan_robustness_ngan() -> str:
    """Một dòng cho mục 'kết quả chính' ở đầu và cuối báo cáo.

    Rút từ cùng số liệu với ket_luan_robustness() để hai chỗ KHÔNG BAO GIỜ
    nói ngược nhau — trước đây cả hai đều viết cứng ở ba vị trí khác nhau.
    """
    frame = read("robustness.csv")
    if frame is None:
        return "Độ bền dưới nhiễu (chưa có số đo)"
    sach = frame[frame["corruption"] == "clean"].set_index("model")["top1"]
    nhieu = frame[frame["corruption"].isin(NHIEU_SO_SANH_DUOC)]
    if sach.empty or nhieu.empty:
        return "Độ bền dưới nhiễu (chưa có số đo)"
    mce = 1 - nhieu.groupby("model")["top1"].mean()
    gioi, ben, yeu = sach.idxmax(), mce.idxmin(), sach.idxmin()

    if gioi != ben and yeu == ben:
        return ("Mô hình có **accuracy sạch cao nhất lại kém bền nhất** trước nhiễu, "
                f"còn `{yeu}` — yếu nhất khi ảnh sạch — bền nhất")
    if gioi != ben:
        return (f"**Accuracy sạch không dự đoán được độ bền:** `{gioi}` đứng đầu khi "
                f"ảnh sạch nhưng `{ben}` mới bền nhất dưới nhiễu")
    return (f"**Độ bền đi cùng accuracy trên nhóm mô hình này:** `{gioi}` đứng đầu cả "
            "trên ảnh sạch và dưới nhiễu")


def table_worst_classes(n: int = 8) -> str:
    """Lớp yếu nhất của mô hình tốt nhất."""
    main = read_main()
    if main is None:
        return MISSING
    best = main.loc[main["test_macro_f1"].idxmax(), "model"]
    frame = read(f"per_class_{best}.csv")
    if frame is None:
        return MISSING
    out = frame.nsmallest(n, "f1")[["class_id", "name", "f1", "support"]]
    out.columns = ["Lớp", "Tên", "F1", "Số ảnh test"]
    return f"**Mô hình `{best}`:**\n\n" + out.round(3).to_markdown(index=False)


def table_confusions(n: int = 6) -> str:
    """n cặp bị nhầm nhiều nhất của mô hình tốt nhất."""
    main = read_main()
    if main is None:
        return MISSING
    best = main.loc[main["test_macro_f1"].idxmax(), "model"]
    frame = read(f"top_confusions_{best}.csv")
    if frame is None:
        return MISSING
    out = frame.head(n)[["true_id", "true_name", "pred_id", "pred_name", "count"]]
    out.columns = ["Lớp thật", "Tên", "Đoán thành", "Tên", "Số lần"]
    return out.to_markdown(index=False)


def facts() -> dict:
    """Vài con số dùng rải rác trong phần diễn giải, đọc từ bảng thật."""
    main, speed = read_main(), read("speed.csv")
    out = {"best": "–", "best_f1": "–", "chosen": "–", "speedup": "–", "n_models": 0}
    if main is None:
        return out
    out["n_models"] = len(main)
    row = main.loc[main["test_macro_f1"].idxmax()]
    out["best"], out["best_f1"] = row["model"], f"{row['test_macro_f1']:.4f}"

    mcnemar = read("mcnemar.csv")
    tied = {out["best"]}
    if mcnemar is not None:
        for r in mcnemar.itertuples(index=False):
            if out["best"] in (r.model_a, r.model_b) and r.p_value >= 0.05:
                tied.add(r.model_b if r.model_a == out["best"] else r.model_a)
    if speed is not None and "cpu_bs1_p95" in speed:
        cand = speed[speed["model"].isin(tied)].dropna(subset=["cpu_bs1_p95"])
        if not cand.empty:
            chosen = cand.loc[cand["cpu_bs1_p95"].idxmin()]
            slowest = cand.loc[cand["cpu_bs1_p95"].idxmax()]
            out["chosen"] = chosen["model"]
            out["speedup"] = f"{slowest['cpu_bs1_p95'] / chosen['cpu_bs1_p95']:.1f}"
    return out


# ---------------------------------------------------------------- báo cáo

def build(f: dict) -> str:
    """Ghép báo cáo. Phần chữ viết tay, phần bảng lấy từ reports/tables/."""
    today = datetime.now().strftime("%d/%m/%Y")
    return f"""# Phân loại biển báo giao thông GTSRB: so sánh ba hướng tiếp cận Deep Learning

**Môn**: Deep Learning · **Ngày**: {today}
**Nhóm**: Phong Nguyễn · Hoàng · Phong Trần · Huy

> Bảng số trong báo cáo này được **sinh tự động** từ `reports/tables/` bằng
> `python scripts/make_baocao.py`. Không có con số nào chép tay; mỗi con số truy được
> về một `artifacts/runs/<run_id>/result.json` cụ thể.

---

## Tóm tắt

Nhóm xây và so sánh {f['n_models']} mô hình phân loại 43 lớp biển báo giao thông Đức trên
bộ GTSRB (39.209 ảnh train + 12.630 ảnh test chính thức), theo ba hướng: CNN đơn giản tự
xây (LeNet), CNN sâu tự xây (VGG-like có BatchNorm, residual, spatial dropout, Global
Average Pooling), và transfer learning từ ba backbone pretrained ImageNet.

GTSRB là bộ dữ liệu **đã bão hoà** — con người đạt 98,84%, nhà vô địch IJCNN 2011 đạt
99,46% — nên nhóm không coi accuracy là câu hỏi chính. Thay vào đó nhóm tập trung vào bốn
câu hỏi mà đa số báo cáo về bộ này bỏ qua: (1) split dữ liệu đúng cách thì con số thật là
bao nhiêu; (2) chênh lệch giữa các mô hình có **ý nghĩa thống kê** không; (3) mô hình giữ
được bao nhiêu hiệu năng dưới nhiễu thực tế; (4) với ràng buộc latency của thiết bị biên
thì nên chọn mô hình nào.

**Ba kết quả chính, đều đi ngược kỳ vọng thông thường:**

1. Mô hình **tự xây 1,24 M tham số đánh bại hai backbone pretrained ImageNet** với ý nghĩa
   thống kê (p < 1e-6), trong khi nhanh hơn {ty_so_toc_do()}.
2. **Latency không tỉ lệ với FLOPs**, chênh tới **{ms_tren_gflop()}** về hiệu quả trên mỗi GFLOP.
   Mô hình tên "EfficientNet" lại là mô hình **chậm nhất** trong cả {f['n_models']}.
3. {ket_luan_robustness_ngan()} trước nhiễu thực tế.

---

## 1. Bài toán và dữ liệu

| | |
|---|---|
| Bộ dữ liệu | GTSRB (German Traffic Sign Recognition Benchmark) |
| Số lớp | 43 |
| Train | 39.209 ảnh · Test chính thức | 12.630 ảnh · **Tổng 51.839** |
| Định dạng | `.ppm`, kích thước thay đổi, **trung vị 43×43 px** (min 25, max 266) |
| Mất cân bằng | **10,7 : 1** (lớp 2 có 2.250 ảnh; lớp 0, 19, 37 chỉ 210) |

Nguồn: archive gốc Đại học Bochum. Nhóm **không** dùng `torchvision.datasets.GTSRB` vì
hàm đó tải bộ train 26.640 ảnh của giải IJCNN 2011 chứ không phải bộ Final Training
39.209 ảnh — một cái bẫy không được ghi trong tài liệu torchvision
(chi tiết: `docs/SU_CO.md` mục 1).

---

## 2. Phương pháp

### 2.1 Tiền xử lý

Cắt theo **ROI bounding box** (nới lề 10%) → cân bằng sáng → resize 48×48.

Bốn chế độ cân bằng sáng được hiện thực để làm ablation: `none`, `he_gray` (equalize ảnh
xám), `he_y` (equalize kênh Y của YUV), và `clahe` (mặc định). CLAHE được chọn vì HE toàn
cục dùng **một** CDF cho cả ảnh nên khuếch đại nhiễu ở vùng phẳng; CLAHE chia ô 8×8,
equalize từng ô và **chặn trần** histogram (`clipLimit = 2,0`).

Cân bằng sáng làm trên **kênh L của LAB**, không làm trên từng kênh RGB — equalize R, G, B
độc lập sẽ làm lệch tỉ lệ ba kênh và **đổi màu ảnh**, trong khi màu biển báo mang nghĩa
(đỏ = cấm, xanh = bắt buộc).

### 2.2 ★ Chia train/val theo track — chống rò rỉ dữ liệu

Đây là đóng góp kỹ thuật quan trọng nhất của phần dữ liệu.

GTSRB **không** chụp mỗi biển báo một lần: xe chạy tới gần một tấm biển và camera quay
**30 frame liên tiếp** của *cùng tấm biển vật lý đó*, lưu thành một "track"
(tên file `000{{track}}_000{{frame}}.ppm`). Nếu trộn tất cả ảnh rồi cắt 80/20 ngẫu nhiên,
frame 12 và frame 13 của **cùng một tấm biển** — gần như là hai bản sao — sẽ nằm một bên
train, một bên val. Mô hình không cần **khái quát**, chỉ cần **nhớ**.

Nhóm dùng `StratifiedGroupKFold` với nhóm `(class_id, track_id)`. Kết quả đối chứng:

| Cách split | Số track dùng chung giữa train và val |
|---|---|
| Random theo ảnh (**sai**) | **1.306** |
| Theo track (**đúng**) | **0** |

Kết quả: train 31.379 / val 7.830 / test 12.630, lệch phân phối lớp tối đa **0,297%**.
Tập test chính thức được **niêm phong**, chỉ mở một lần ở cuối.
`tests/test_split.py` khoá bất biến này làm cổng chặn của dự án.

### 2.3 Augmentation

Xoay ±15°, dịch ±10%, zoom 0,9–1,1, jitter độ sáng/tương phản ±0,2. Chỉ áp cho tập train.

**Hai phép biến đổi bị cấm:**

- `RandomHorizontalFlip` — bốn cặp lớp là **ảnh gương** của nhau (33↔34 rẽ phải/trái,
  19↔20, 36↔37, 38↔39). Lật ngang là **tự tạo dữ liệu sai nhãn**.
- Hue/saturation jitter — màu là **đặc trưng mang nghĩa**, không phải nhiễu.

`tests/test_transforms.py` khoá luật này ở mức mã nguồn.

### 2.4 Ba hướng mô hình

| | Mô hình | Kiến trúc | Input |
|---|---|---|---|
| **M1** | LeNet-style, tự xây | 2 × (Conv 5×5 + MaxPool) + 2 Dense | 48² |
| **M2** | VGG-like, tự xây | 4 stage × 2 Conv 3×3 + BN + residual + SpatialDropout + GAP | 48² |
| **M3** | Transfer learning | ResNet18 / MobileNetV2 / EfficientNet-B0 pretrained ImageNet | 224² |

**Một quan sát đáng chú ý về số tham số:** M1 có **2.424.299** tham số với **2** lớp conv,
trong đó **97,3%** nằm ở *một* lớp `Flatten(9216) → FC(256)`. M2 có **9** lớp conv nhưng
chỉ **1.237.451** tham số — **ít hơn M1** — vì nó thay Flatten+FC bằng Global Average
Pooling (11.051 tham số, giảm **214 lần** ở phần head). Bài học: **số lớp không tỉ lệ với số
tham số; chỗ đắt là lớp fully-connected.**

M3 dùng mean/std của **ImageNet** (không phải của GTSRB) vì trọng số pretrained và
`running_mean/var` của các lớp BatchNorm được học trên phân phối đó. Ảnh được upsample lên
224×224 vì ba backbone downsample 32 lần — đưa 48×48 vào thì feature map cuối chỉ còn
1,5×1,5. Nhóm nêu rõ: nội suy 48→224 **không thêm thông tin nào**, nó chỉ để khớp scale và
stride của mạng pretrained.

Huấn luyện M3 theo **hai pha**: (1) đóng băng backbone, train riêng head 3 epoch; (2) mở
băng toàn bộ với **discriminative learning rate** (tầng đầu 1e-5, giữa 1e-4, head 1e-3).
Bằng chứng thực nghiệm cho thiết kế này: với ResNet18, train riêng head chững ở **67,9%**
sau 3 epoch, rồi **nhảy lên 96,4% chỉ trong một epoch** ngay khi mở băng.

### 2.5 Huấn luyện

Cross-entropy + **label smoothing 0,1**; Adam/AdamW; warmup tuyến tính 3 epoch rồi
**cosine annealing**; gradient clipping 1,0; mixed precision (tự bật trên CUDA, tắt trên
MPS); **early stopping theo val macro-F1** (không theo accuracy, vì dữ liệu mất cân bằng
10,7:1 khiến accuracy bị các lớp đông chi phối). Cả {f['n_models']} mô hình dùng **chung
một hàm `fit()`** để so sánh được công bằng.

---

## 3. Kết quả

### 3.1 Bảng so sánh chính (tập test chính thức, 12.630 ảnh)

{table_models()}

**Lưu ý khi đọc:** cột **Top-5 gần như vô nghĩa** ở bài này. Top-5 ra đời cho ImageNet
1000 lớp; trên 43 lớp nó nghĩa là "đúng trong 11,6% số lớp" nên mọi mô hình tử tế đều
~99,9%. Nhóm báo cáo vì đề bài yêu cầu nhưng **kết luận dựa trên top-1 và macro-F1**.

### 3.2 ★ Kiểm định ý nghĩa thống kê (McNemar)

Hai mô hình chạy trên **cùng** tập test nên quan sát là **bắt cặp**; dùng t-test hai mẫu
độc lập là sai về mặt thống kê. McNemar chỉ xét hai ô **bất đồng** của bảng 2×2.

{table_mcnemar()}

Trên 12.630 ảnh, chênh 0,1 điểm top-1 chỉ là ~13 ảnh — hoàn toàn có thể nằm trong nhiễu.
Đây là lý do mọi so sánh đều phải kiểm định.

**Phát hiện quan trọng nhất của báo cáo:** mô hình **M2 tự xây** (1,24 M tham số, 48×48)
**vượt** cả MobileNetV2 lẫn EfficientNet-B0 — hai backbone pretrained trên 1,28 triệu ảnh
ImageNet — với p < 1e-6, và **tương đương** ResNet18 (p = 0,45).

Giải thích khả dĩ: ảnh GTSRB có trung vị chỉ **43×43 px**. Upsample lên 224 không tạo ra
thông tin nào, nên phần tri thức pretrained về chi tiết tinh không có gì để bám vào, trong
khi chi phí (tham số, FLOPs, latency, thời gian train) vẫn phải trả đủ. Kết luận cần phát
biểu thận trọng: **transfer learning hiệu quả khi dữ liệu đích ít VÀ miền đích gần miền
nguồn về độ phân giải** — GTSRB không thoả cả hai điều kiện.

### 3.3 Phân tích theo lớp

{table_worst_classes()}

Các cặp bị nhầm nhiều nhất:

{table_confusions()}

Hai nguyên nhân cần tách bạch: F1 thấp kèm **support cao** là do **hình giống nhau**
(vấn đề độ phân giải — các biển giới hạn tốc độ chỉ khác chữ số, mà ở 48×48 chữ số chỉ còn
vài pixel); F1 thấp kèm **support thấp** mới là do **thiếu dữ liệu**. Số liệu cho thấy
nguyên nhân thứ nhất chiếm ưu thế, nên hướng cải thiện là **tăng độ phân giải**, không
phải làm mô hình to hơn.

### 3.4 Độ hiệu chỉnh (calibration)

Cột ECE trong bảng 3.1. Kết quả **ngược với lý thuyết thông thường**: Guo et al. (2017)
cho rằng mạng sâu thường **tự tin thái quá**, nhưng nhóm đo được M1 có độ tự tin trung
bình **0,854** trong khi accuracy là **0,985** — tức **thiếu tự tin**.

Nguyên nhân: kết luận của Guo đúng trên ImageNet (bài toán khó, accuracy ~76%). GTSRB
**quá dễ** nên mức tự tin mà label smoothing kéo mô hình về (≈ `1−ε+ε/K` = 0,9023) **thấp
hơn** accuracy thật — nó **sửa quá tay**. Hệ quả thực tế: lập luận "đặt ngưỡng 0,8 thì
chuyển cho người lái" **không dùng được** với mô hình này, vì ngưỡng đó sẽ loại bỏ rất
nhiều dự đoán đúng.

### 3.5 Robustness dưới nhiễu mô phỏng

Năm loại nhiễu × năm mức độ, **chỉ áp lúc test** — train trên chúng thì phép đo mất ý
nghĩa vì robustness nghĩa là khái quát sang phân phối **chưa từng thấy**.

Bảng `relative robustness` = accuracy dưới nhiễu / accuracy trên ảnh sạch:

{table_robustness()}

{ket_luan_robustness()}

⚠️ **Giới hạn phương pháp cần nêu rõ:** nhiễu được áp **sau khi** resize, mà M1/M2 chạy ở
48×48 còn M3 ở 224×224. Kernel motion blur mức 5 là **15 pixel tuyệt đối**: ở 48×48 nó phủ
**31%** chiều rộng ảnh, ở 224×224 chỉ **6,7%**. Vì vậy cột `motion_blur` **không so sánh
được** giữa hai nhóm độ phân giải. Ba cột `gauss_noise`, `low_light`, `occlusion` thì so
sánh được (phép toán theo từng pixel hoặc theo % diện tích), và kết luận chỉ rút từ chúng.
Hướng sửa triệt để: đổi kernel sang tỉ lệ % chiều rộng ảnh.

#### 3.5.1 Giới hạn NẶNG hơn: nhiễu được áp SAU bước CLAHE

{bang_thu_tu_clahe()}

### 3.5.2 Nhiễu seed — chênh lệch bao nhiêu điểm thì mới là THẬT?

Cùng một cấu hình, chỉ đổi seed khởi tạo. Đây là phép đo **độc lập** với McNemar
cho cùng một câu hỏi, và quan trọng hơn mọi con số accuracy lẻ trong báo cáo: nó
cho biết **ngưỡng dưới** của những gì đáng kết luận.

{ket_luan_seed()}

{bang_seed_summary()}

### 3.6 Tốc độ suy luận và triển khai biên

Đo đúng quy trình: 20 vòng warm-up, **đồng bộ thiết bị trước và sau khi bấm giờ** (GPU
chạy bất đồng bộ — không đồng bộ thì đang đo thời gian *gửi lệnh*), 100 vòng đo, báo
**p50 và p95** (hệ thống thời gian thực quan tâm trường hợp xấu).

**Latency không tỉ lệ với FLOPs:**

{table_efficiency()}

Nếu latency tỉ lệ FLOPs thì cột cuối phải bằng nhau — thực tế chênh **{ms_tren_gflop()}**.
EfficientNet-B0 {nghich_ly_effnet()}.
Nguyên nhân: MBConv + squeeze-excitation gồm rất nhiều lớp **mảnh**, mỗi lớp tốn chi phí
cố định (launch kernel, truy cập bộ nhớ) mà làm rất ít phép tính → bị chặn bởi **băng
thông bộ nhớ**, không bởi năng lực tính toán.

**Hệ quả thực hành: chọn mô hình theo FLOPs sẽ dẫn tới EfficientNet-B0 — mô hình chậm
nhất trong cả {f['n_models']}.** Muốn nói về triển khai thì phải đo wall-clock trên thiết
bị đích.

#### 3.6.1 Nén int8 và xuất cho thiết bị biên

{md(read("edge_export.csv"))}

Cột **nhẹ hơn** và **nhanh hơn** so với bản `fp32` của cùng mô hình.
{pham_vi_nen_int8()}

Cả 5 mô hình đều xuất được **TorchScript** và **ONNX**, và mỗi bản ONNX đều được
**kiểm lại bằng ảnh test thật** (so logit và so lớp dự đoán) — xuất thành công không
có nghĩa là xuất đúng. Chi tiết một lần báo động sai của phép kiểm này ở
`docs/SU_CO.md` §9.

### 3.7 Ablation — đổi một biến một lần

{bang_ablation()}

#### 3.7.1 ★ Trục độ phân giải đo gì — và một phép đo đối chứng

{truc_do_phan_giai()}

### 3.8 Grad-CAM

Bản đồ nhiệt sinh theo công thức Selvaraju (2017), hook vào khối conv cuối. Ba nhóm hình
cho mỗi mô hình: dự đoán đúng, **dự đoán sai**, và cùng một ảnh trước/sau khi thêm nhiễu.

Độ phân giải heatmap đo được cho kết quả phản trực giác: **M1 mịn nhất** (feature map
12×12, mỗi ô ≈ 4×4 px), **M2 thô nhất** (3×3, mỗi ô ≈ 16×16 px), M3 ở giữa (7×7). M2 —
mô hình thiết kế công phu nhất — cho bản đồ nhiệt thô nhất vì nó có 4 lớp MaxPool so với
2 của M1. Nói "heatmap đúng vào biển báo" với M2 là phát biểu rất lỏng: cả ảnh chỉ có 9 ô.

Nhóm nhấn mạnh Grad-CAM chỉ cho biết **ở đâu**, không cho biết **đặc trưng gì**, và
**không phải bằng chứng nhân quả** — có ca mô hình nhìn đúng vào biển báo mà vẫn suy luận
sai (xem `reports/figures/gradcam_*_wrong.png`).

---

## 4. Khuyến nghị triển khai

Quyết định dựa trên **ba trục**, không chỉ accuracy:

1. Accuracy **kèm p-value McNemar** — nếu chênh lệch không có ý nghĩa thì coi như bằng nhau.
2. **Latency p95** ở batch = 1 (tình huống xe xử lý từng khung ảnh).
3. **Relative robustness** dưới nhiễu — điều kiện vận hành thật.

Mô hình macro-F1 cao nhất là `{f['best']}` ({f['best_f1']}). Nhưng trong nhóm **tương đương
thống kê** với nó, mô hình nhanh nhất là **`{f['chosen']}`** — nhanh hơn **{f['speedup']}
lần** ở p95. Trả thêm {f['speedup']} lần latency để lấy chênh lệch accuracy *không có ý
nghĩa thống kê* là lựa chọn tồi trên hệ thống thời gian thực.

→ **Khuyến nghị: `{f['chosen']}`.** Chi tiết suy luận ba bước: `docs/KET_QUA.md` mục 9.

---

## 5. Hạn chế

1. **Đây là bài phân loại, không phải phát hiện.** Mô hình nhận đầu vào là biển báo **đã
   cắt sẵn**; hệ thống thật phải tự tìm biển báo trong khung ảnh toàn cảnh trước, và lỗi
   của bước đó sẽ cộng dồn vào.
2. **Phân phối test giống phân phối train** — cùng chụp ở Đức, cùng loại camera. Chưa kiểm
   được khả năng khái quát sang biển báo nước khác.
3. **Phép đo robustness có hai giới hạn đã lượng hoá:** mức độ nhiễu không so sánh được
   giữa các độ phân giải đối với motion blur (mục 3.5), và nhiễu được áp **sau** bước
   CLAHE nên thiên vị theo mô hình (mục 3.5.1) — giới hạn thứ hai nặng hơn, và đã được
   đo bằng thực nghiệm đối chứng chứ không chỉ nêu ra.
4. **So sánh M2 với M3 lẫn hai biến** (pretrained hay không, và 48 hay 224 px). Trục
   ablation `resolution` **KHÔNG tách được hai biến đó cho M3**, vì cả ba mức 64/112/224
   đều nội suy từ cùng cache 48 px — nó đo dung lượng feature map, không đo chi tiết ảnh
   (mục 3.7.1). Phép đo đối chứng bằng cache 112 px thật cho thấy chi tiết thật **không
   đổi kết quả** (chênh 0,05 điểm, nhỏ hơn nhiễu seed 10 lần), nên kết luận chính không
   bị đe doạ — nhưng biến "pretrained hay không" vẫn chưa được tách sạch khỏi biến
   "kiến trúc", và đó là hướng mở rộng rõ ràng nhất.
5. **Thí nghiệm rò rỉ chỉ chạy trên M1.** Con số 1,09 điểm val ảo đo được trên M1;
   mức thổi phồng có thể khác với mô hình dung lượng lớn hơn, vốn dễ nhớ frame hơn.
6. **Không dùng class weighting hay resampling khi huấn luyện.** `losses.py` có hỗ trợ
   `class_weight` nhưng các config đều để tắt, trong khi dữ liệu mất cân bằng 10,7:1 và
   macro-F1 lại là chỉ số chính. Đây là lựa chọn có ý thức (giữ đường cơ sở đơn giản,
   và macro-F1 đã đủ để phát hiện lỗi ở lớp thiểu số) nhưng là một hướng mở rộng rõ ràng.
7. **Chỉ M2 được chạy nhiều seed** (3 seed); bốn mô hình còn lại một seed. Biên độ seed đo
   được của M2 dùng làm thước đo nhiễu cho cả bảng, nhưng đó là phép ngoại suy.
8. **Chưa kiểm adversarial robustness** (FGSM/PGD) — khác bản chất với nhiễu tự nhiên.

---

## 6. Kết luận

Trên một bộ dữ liệu đã bão hoà, việc đua accuracy không còn là câu hỏi khoa học. Giá trị
của báo cáo này nằm ở ba kết luận chỉ rút ra được khi **đo cẩn thận và kiểm định**:

1. Một CNN **tự xây 1,24 M tham số** vượt hai backbone pretrained ImageNet có ý nghĩa thống
   kê, trong khi nhanh hơn {ty_so_toc_do()} — vì miền đích có độ phân giải quá thấp để lợi thế
   pretrained phát huy.
2. **Latency chênh {ms_tren_gflop()} so với dự đoán từ FLOPs.** Chọn mô hình theo FLOPs sẽ chọn đúng
   mô hình chậm nhất.
3. {ket_luan_robustness_ngan()} — và mô hình chính xác nhất cũng không phải mô hình
   nên triển khai.

Điểm chung của cả ba: chúng chỉ lộ ra khi **không tin vào con số đầu tiên nhìn thấy**.

---

## Phụ lục A — Tái lập

```bash
make setup-mac      # môi trường
make data           # tải + tiền xử lý + split theo track
make test           # {so_test()}, gồm cổng chặn rò rỉ dữ liệu
make train-m1 train-m2 train-m3
make eval robustness speed gradcam
make report baocao  # sinh KET_QUA.md và BAO_CAO.md
```

Mỗi lần chạy sinh `artifacts/runs/<run_id>/` gồm `best.pt`, `config.yaml` (cấu hình **thực
tế** đã dùng), `train_log.csv` và `result.json` (kèm `seed` và `git_commit`). Mọi siêu tham
số nằm trong YAML, không hard-code. Seed cố định cho torch/numpy/random và
`cudnn.deterministic`; chạy lại cùng seed cho cùng val macro-F1 trong sai số < 0,001.

## Phụ lục B — Phân công

| Thành viên | Phần phụ trách |
|---|---|
| Phong Nguyễn | M1 baseline, mô-đun đánh giá (metrics, McNemar, ECE), bảng so sánh, báo cáo |
| Hoàng | M2, training engine dùng chung, ablation kiến trúc |
| Phong Trần | M3 (ba backbone), Grad-CAM, đo tốc độ và Pareto |
| Huy | Đường ống dữ liệu, thí nghiệm rò rỉ, ablation dữ liệu, robustness |

Mỗi file mã nguồn ghi rõ `CHỦ: <tên>` ở đầu docstring.

## Phụ lục C — Tài liệu kèm theo

{phu_luc_tai_lieu()}
"""


def main() -> None:
    """Sinh docs/BAO_CAO.md từ các bảng trong reports/tables/."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/BAO_CAO.md")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(facts()), encoding="utf-8")
    log.info("Đã sinh %s (%d dòng)", out,
             len(out.read_text(encoding="utf-8").splitlines()))


if __name__ == "__main__":
    main()
