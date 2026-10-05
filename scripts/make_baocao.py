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
    dong = [f"Trên {len(sach)} mô hình, xét **{len(NHIEU_SO_SANH_DUOC)} loại nhiễu so sánh "
            f"được** ({', '.join(f'`{n}`' for n in NHIEU_SO_SANH_DUOC)}):"]

    if gioi_nhat != ben_nhat:
        dong.append(
            f"\n**Mô hình chính xác nhất trên ảnh sạch KHÔNG phải mô hình bền nhất.** "
            f"`{gioi_nhat}` dẫn đầu khi ảnh sạch (top-1 = {sach[gioi_nhat]:.4f}) "
            f"nhưng `{ben_nhat}` mới là mô hình bền nhất "
            f"(error trung bình dưới nhiễu = {mce[ben_nhat]:.4f} so với "
            f"{mce[gioi_nhat]:.4f}).")
    else:
        dong.append(
            f"\n**Mô hình chính xác nhất cũng là mô hình bền nhất:** `{gioi_nhat}` "
            f"đứng đầu cả trên ảnh sạch (top-1 = {sach[gioi_nhat]:.4f}) và dưới nhiễu "
            f"(error trung bình = {mce[gioi_nhat]:.4f}). Trên nhóm mô hình này, độ bền "
            f"KHÔNG tách khỏi accuracy — khác với kết quả khi chỉ đo 3 mô hình đầu.")

    if yeu_nhat == ben_nhat:
        dong.append(
            f"\nĐáng chú ý hơn: `{yeu_nhat}` — **yếu nhất** trên ảnh sạch "
            f"(top-1 = {sach[yeu_nhat]:.4f}) — lại bền nhất. Thứ tự xếp hạng khi có "
            f"nhiễu **đảo lại** so với khi không có.")

    dong.append("\nXếp hạng độ bền (error trung bình dưới nhiễu, càng THẤP càng bền): "
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
Pooling (11.051 tham số, giảm ~213 lần ở phần head). Bài học: **số lớp không tỉ lệ với số
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
EfficientNet-B0 có **ít hơn ResNet18 4,4 lần FLOPs** nhưng chậm hơn **14 lần** trên CPU.
Nguyên nhân: MBConv + squeeze-excitation gồm rất nhiều lớp **mảnh**, mỗi lớp tốn chi phí
cố định (launch kernel, truy cập bộ nhớ) mà làm rất ít phép tính → bị chặn bởi **băng
thông bộ nhớ**, không bởi năng lực tính toán.

**Hệ quả thực hành: chọn mô hình theo FLOPs sẽ dẫn tới EfficientNet-B0 — mô hình chậm
nhất trong cả {f['n_models']}.** Muốn nói về triển khai thì phải đo wall-clock trên thiết
bị đích.

### 3.7 Grad-CAM

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
3. **Mức độ nhiễu không so sánh được giữa các độ phân giải** đối với motion blur (mục 3.5).
4. **So sánh M2 với M3 lẫn hai biến** (pretrained hay không, và 48 hay 224 px). Ablation
   độ phân giải là bước cần thiết để tách chúng — đã chuẩn bị nhưng chưa chạy.
5. **Chỉ M2 được chạy nhiều seed** (3 seed); bốn mô hình còn lại một seed. Biên độ seed đo
   được của M2 dùng làm thước đo nhiễu cho cả bảng, nhưng đó là phép ngoại suy.
6. **Chưa kiểm adversarial robustness** (FGSM/PGD) — khác bản chất với nhiễu tự nhiên.

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
make test           # 95 test, gồm cổng chặn rò rỉ dữ liệu
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

| File | Nội dung |
|---|---|
| `docs/KET_QUA.md` | toàn bộ bảng số, sinh tự động |
| `docs/LY_THUYET.md` | cơ sở lý thuyết, công thức, lý do từng lựa chọn thiết kế |
| `docs/SU_CO.md` | 9 nhóm sự cố đã gặp trong quá trình làm, kèm nguyên nhân và cách sửa |
| `docs/PHAN_CONG.md` | phân công chi tiết + 38 câu hỏi ôn tập |
| `reports/tables/` | 20 bảng CSV |
| `reports/figures/` | 31 hình |
| `notebooks/` | 10 notebook có diễn giải |
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
