"""
test_gop_seed.py — khoá cách gộp các run khác seed thành mean ± std.

CHỦ: Hoàng (thực nghiệm nhiều seed)

★ LỖI ĐÃ GẶP — báo cáo nói mô hình ổn định hơn thực tế gấp đôi ★

collect() chỉ khớp `artifacts/runs/*seed*`, nên nó bỏ RUN CHÍNH — mà run chính
cũng là một seed (42), chỉ là không có tag `seedN`. Bảng mean±std vì vậy dựa
trên 2 seed thay vì 3:

    macro-F1      2 seed        3 seed
    mean          0,99181       0,99054
    std           0,00164       0,00250
    biên độ       0,23 điểm     0,50 điểm   <- gấp hơn 2 lần

Hệ quả thẳng vào kết luận chính: chênh lệch M2 vs M3-ResNet18 là 0,23 điểm.
Với biên độ seed 0,23 thì nó trông như "sát nhau nhưng có thật"; với biên độ
thật 0,50 thì nó NẰM TRONG nhiễu seed — khớp đúng với McNemar p = 0,4477.
Và seed 43 của M2 cho 0,99297, CAO HƠN ResNet18 0,99030: thứ hạng đảo theo seed.

Nhưng chỉ được gộp khi cấu hình TRÙNG KHỚP. Gộp một run 40 epoch với một run
15 epoch rồi gọi chênh lệch đó là "nhiễu seed" là sai nặng hơn cả lỗi ban đầu.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from run_seeds import collect, summarise


def _ghi(thu_muc: Path, run_id: str, seed: int, macro_f1: float,
         notes: str = "", epochs: int = 40) -> None:
    thu_muc.mkdir(parents=True, exist_ok=True)
    (thu_muc / "result.json").write_text(json.dumps({
        "run_id": run_id, "model": "m2_vggres", "seed": seed, "notes": notes,
        "data": {"img_size": 48, "preprocess": "clahe", "aug_policy": "geo_photo"},
        "train": {"epochs_run": epochs, "batch_size": 128, "lr": 1e-3,
                  "label_smoothing": 0.1, "optimizer": "adam"},
        "test": {"top1": macro_f1 + 0.002, "top5": 0.999,
                 "macro_f1": macro_f1, "ece": 0.05},
    }), encoding="utf-8")


@pytest.fixture
def ba_run(tmp_path, monkeypatch):
    """Run chính (seed 42, không tag) + hai run seed 43/44, cùng cấu hình."""
    runs = tmp_path / "artifacts" / "runs"
    _ghi(runs / "m2_vggres_s42_20261003_223350", "m2_vggres_s42_20261003_223350",
         42, 0.98798)
    _ghi(runs / "m2_vggres_s43_seed43_20261005_081339",
         "m2_vggres_s43_seed43_20261005_081339", 43, 0.99297, notes="seed43")
    _ghi(runs / "m2_vggres_s44_seed44_20261005_083630",
         "m2_vggres_s44_seed44_20261005_083630", 44, 0.99066, notes="seed44")
    monkeypatch.chdir(tmp_path)
    return runs


def test_run_chinh_duoc_tinh_la_mot_seed(ba_run):
    """Lõi của lỗi: seed 42 nằm ở run KHÔNG có tag, vẫn phải được gộp."""
    f = collect()
    assert len(f) == 3, f"phải có 3 seed, chỉ thấy {sorted(f.seed)}"
    assert set(f.seed) == {42, 43, 44}


def test_bien_do_seed_khong_bi_bao_hep_lai(ba_run):
    """Biên độ thật là 0,50 điểm. Bỏ seed 42 thì chỉ còn 0,23 — hẹp hơn 2 lần."""
    f = collect()
    bien_do = (f.macro_f1.max() - f.macro_f1.min()) * 100
    assert bien_do == pytest.approx(0.499, abs=0.01), \
        f"biên độ phải ~0,50 điểm, ra {bien_do:.2f}"


def test_bao_so_seed_de_doc_duoc_std_dua_tren_may_lan(ba_run):
    g = summarise(collect())
    assert int(g[("n_seeds", "")].iloc[0]) == 3


def test_KHONG_gop_run_khac_cau_hinh(tmp_path, monkeypatch):
    """Run 15 epoch không được gộp với run 40 epoch.

    Gộp vào thì chênh lệch do SỐ EPOCH sẽ bị gọi là "nhiễu seed" — sai nặng
    hơn cả việc thiếu một seed.
    """
    runs = tmp_path / "artifacts" / "runs"
    _ghi(runs / "m2_vggres_s43_seed43_x", "m2_vggres_s43_seed43_x", 43, 0.99297,
         notes="seed43", epochs=40)
    _ghi(runs / "m2_vggres_s42_ngan", "m2_vggres_s42_ngan", 42, 0.95000,
         epochs=15)                                     # 15 epoch -> KHÁC
    monkeypatch.chdir(tmp_path)
    f = collect()
    assert set(f.seed) == {43}, \
        f"run 15 epoch phải bị loại, nhưng thấy seed {sorted(f.seed)}"


def test_khong_co_run_seed_nao_thi_khong_tu_gop_run_chinh(tmp_path, monkeypatch):
    """Không có thí nghiệm seed thì bảng phải TRỐNG, không phải 1 'seed'."""
    runs = tmp_path / "artifacts" / "runs"
    _ghi(runs / "m2_vggres_s42_a", "m2_vggres_s42_a", 42, 0.98798)
    monkeypatch.chdir(tmp_path)
    assert collect().empty
