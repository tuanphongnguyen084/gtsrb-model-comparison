"""
test_schema.py — khoá schema của `result.json`.

CHỦ: Phong Nguyễn (tổng hợp báo cáo)

★ VÌ SAO FILE NÀY TỒN TẠI ★

`docs/INTERFACE.md` §6 viết: *"Khoá nào chưa chạy thì để `null`, không bỏ khoá.
`tests/test_schema.py` kiểm việc này."* — nhưng file đó **chưa bao giờ được
viết**. Một cam kết trong tài liệu mà không có gì thi hành thì sớm muộn sẽ lệch,
và nó đã lệch: soát 39 run thật thấy `val.top5` và `val.macro_f1` thiếu ở
**39/39 run**, còn `cost.latency_ms` không hề tồn tại (latency nằm ở các khoá
PHẲNG như `cpu_bs1_p50`).

Hậu quả nếu không ai phát hiện: mọi script tổng hợp đều glob
`artifacts/runs/*/result.json`, nên một run sai schema **bị bỏ im lặng khỏi báo
cáo**. Cộng tác viên đọc INTERFACE.md rồi viết `result["val"]["macro_f1"]` sẽ
nhận KeyError ở một chỗ không liên quan gì tới nguyên nhân.

Các test dưới đây khoá **schema THẬT** — cái mà code đang ghi và các script đang
đọc — và `docs/INTERFACE.md` đã được sửa theo. Thêm khoá mới thì cứ thêm; BỎ
hoặc ĐỔI TÊN một khoá ở đây là làm hỏng một script tổng hợp nào đó.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RUNS = sorted(ROOT.glob("artifacts/runs/*/result.json"))

# Khoá gốc — PHẢI CÓ MẶT ở mọi run, dù giá trị là null.
KHOA_GOC = ("run_id", "model", "seed", "git_commit", "config_path",
            "data", "train", "val", "test", "cost", "ckpt_path", "notes")

# Khoá trong từng khối. Chỉ liệt kê những khoá mà SCRIPT TỔNG HỢP thật sự đọc —
# đó mới là hợp đồng. Khoá nào không ai đọc thì không nên khoá cứng ở đây.
KHOA_DATA = ("img_size", "preprocess", "aug_policy",
             "split_group_by_track", "n_train", "n_val", "n_test")
KHOA_TRAIN = ("epochs_run", "best_epoch", "label_smoothing", "optimizer",
              "lr", "scheduler", "batch_size", "train_seconds", "device")
KHOA_VAL = ("top1", "best_macro_f1")          # ★ best_macro_f1, KHÔNG phải macro_f1
KHOA_TEST = ("top1", "top5", "macro_f1", "weighted_f1", "ece", "per_class_f1")
KHOA_COST = ("params_m", "flops_g", "size_mb")   # latency ở khoá PHẲNG, xem dưới

pytestmark = pytest.mark.skipif(
    not RUNS, reason="chưa có run nào trong artifacts/runs/ — chạy `make train-all`")


def _doc(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", RUNS, ids=lambda p: p.parent.name[:44])
def test_co_du_khoa_goc(path):
    """Thiếu khoá gốc = run bị bỏ im lặng khỏi báo cáo."""
    d = _doc(path)
    thieu = [k for k in KHOA_GOC if k not in d]
    assert not thieu, f"thiếu khoá gốc {thieu}"


@pytest.mark.parametrize("path", RUNS, ids=lambda p: p.parent.name[:44])
def test_khoi_data_va_train_khong_bao_gio_null(path):
    """`data` và `train` luôn ghi được: chúng có TRƯỚC khi train xong.

    Khác với `test` và `cost` — hai khối đó phụ thuộc bước đánh giá chạy sau,
    nên schema CHO PHÉP chúng là null (các run Colab đúng như vậy).
    """
    d = _doc(path)
    for khoi in ("data", "train"):
        assert d.get(khoi), f"`{khoi}` là null — phải có từ lúc train"


@pytest.mark.parametrize("path", RUNS, ids=lambda p: p.parent.name[:44])
def test_khoi_data_du_khoa(path):
    d = _doc(path)
    thieu = [k for k in KHOA_DATA if k not in (d.get("data") or {})]
    assert not thieu, f"data thiếu {thieu}"


@pytest.mark.parametrize("path", RUNS, ids=lambda p: p.parent.name[:44])
def test_khoi_train_du_khoa(path):
    d = _doc(path)
    thieu = [k for k in KHOA_TRAIN if k not in (d.get("train") or {})]
    assert not thieu, f"train thiếu {thieu}"


@pytest.mark.parametrize("path", RUNS, ids=lambda p: p.parent.name[:44])
def test_khoi_val_du_khoa(path):
    """★ `best_macro_f1`, KHÔNG phải `macro_f1`.

    INTERFACE.md từng ghi sai là `macro_f1` + `top5`; không script nào đọc hai
    khoá đó, và 39/39 run không có chúng. run_seeds và make_report đọc
    `val.best_macro_f1`.
    """
    d = _doc(path)
    thieu = [k for k in KHOA_VAL if k not in (d.get("val") or {})]
    assert not thieu, f"val thiếu {thieu}"


@pytest.mark.parametrize("path", RUNS, ids=lambda p: p.parent.name[:44])
def test_khoi_test_du_khoa_khi_khong_null(path):
    """`test` được phép null (chưa đánh giá), nhưng có thì phải đủ khoá."""
    d = _doc(path)
    t = d.get("test")
    if t is None:
        pytest.skip("chưa đánh giá — schema cho phép null")
    thieu = [k for k in KHOA_TEST if k not in t]
    assert not thieu, f"test thiếu {thieu}"
    assert len(t["per_class_f1"]) == 43, \
        f"per_class_f1 phải có đúng 43 số, có {len(t['per_class_f1'])}"


@pytest.mark.parametrize("path", RUNS, ids=lambda p: p.parent.name[:44])
def test_khoi_cost_du_khoa_khi_khong_null(path):
    """`cost` được phép null, nhưng có thì phải đủ params/flops/size + latency.

    ★ Latency nằm ở khoá PHẲNG (`cpu_bs1_p50`, `mps_bs64_p95`...), KHÔNG lồng
    trong `latency_ms` như INTERFACE.md từng ghi.
    """
    d = _doc(path)
    c = d.get("cost")
    if c is None:
        pytest.skip("chưa đo tốc độ — schema cho phép null")
    thieu = [k for k in KHOA_COST if k not in c]
    assert not thieu, f"cost thiếu {thieu}"
    assert any(k.endswith("_p50") for k in c), \
        f"cost không có khoá latency nào kết thúc bằng _p50: {sorted(c)}"


@pytest.mark.parametrize("path", RUNS, ids=lambda p: p.parent.name[:44])
def test_run_id_khop_ten_thu_muc(path):
    """run_id lệch tên thư mục thì mọi phép dò 'đã chạy xong chưa' sai theo."""
    assert _doc(path)["run_id"] == path.parent.name


@pytest.mark.parametrize("path", RUNS, ids=lambda p: p.parent.name[:44])
def test_notes_la_chuoi_khong_bao_gio_null(path):
    """`notes` phân biệt run CHÍNH với run ablation/thí nghiệm.

    Để null thì `chi_run_chinh()` và `find_run_dirs(main_only=True)` phải đoán,
    và một run ablation có thể lọt vào bảng so sánh chính.
    """
    notes = _doc(path).get("notes")
    assert isinstance(notes, str), f"notes phải là chuỗi (có thể rỗng), nhận {notes!r}"


def test_moi_run_chinh_deu_da_duoc_danh_gia():
    """Run CHÍNH (notes rỗng) phải có số test — nó đi vào bảng so sánh chính."""
    chua = [p.parent.name for p in RUNS
            if not (_doc(p).get("notes") or "") and not _doc(p).get("test")]
    assert not chua, f"run chính chưa đánh giá: {chua}. Chạy scripts/evaluate.py"


def test_khong_co_hai_run_trung_run_id():
    """run_id trùng nhau thì bảng tổng hợp sẽ có dòng ghi đè nhau."""
    ids = [_doc(p)["run_id"] for p in RUNS]
    trung = {i for i in ids if ids.count(i) > 1}
    assert not trung, f"run_id trùng: {trung}"
