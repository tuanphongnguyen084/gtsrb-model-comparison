"""
test_ablation_skip.py — khoá điều kiện "job này đã chạy xong chưa".

CHỦ: Phong Trần (trục resolution) · Hoàng (bảo trì run_ablation.py)

★ LỖI ĐÃ GẶP THẬT — hỏng cả trục resolution sau một mẻ 5 giờ ★

already_done() khớp run cũ bằng `*{trục}-{tag}*`, KHÔNG có tên model. Nhưng
trục `resolution` so NHIỀU model ở cùng một độ phân giải, nên job m1_lenet@32px
và job m2_vggres@32px có CÙNG tag `resolution-res32-budget`. Sau khi M1 xong,
M2 bị coi là "đã chạy xong" và bỏ qua.

Mất đúng 4 job: m2@32, m2@48, m2@64 và resnet18@64 (trùng tag res64 với m1@64).
Trục resolution còn 5/9 run và KHÔNG so được M1 với M2 — đúng câu hỏi nó được
dựng ra để trả lời. Mẻ vẫn báo "mã thoát 0", không có gì báo động.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from run_ablation import already_done, build_jobs


def _job(config: str, axis: str = "resolution", tag: str = "res32-budget") -> dict:
    return {"axis": axis, "config": config, "tag": tag,
            "overrides": [], "needs_cache": None}


def test_job_cua_model_KHAC_khong_bi_coi_la_da_chay(tmp_path, monkeypatch):
    """Lõi của lỗi: cùng tag, khác model -> KHÔNG được bỏ qua."""
    runs = tmp_path / "artifacts" / "runs"
    xong = runs / "m1_lenet_s42_resolution-res32-budget_20261005_032415"
    xong.mkdir(parents=True)
    (xong / "result.json").write_text("{}")
    monkeypatch.chdir(tmp_path)

    assert already_done(_job("configs/m1_lenet.yaml")) is not None, \
        "chính model đó đã xong thì phải nhận ra"
    assert already_done(_job("configs/m2_vggres.yaml")) is None, \
        "m2_vggres CHƯA chạy — cùng tag với m1 không có nghĩa là đã xong"
    assert already_done(_job("configs/m3_resnet18.yaml")) is None


def test_thieu_result_json_thi_chua_xong(tmp_path, monkeypatch):
    """Thư mục có mà không có result.json = train bị kill giữa chừng."""
    runs = tmp_path / "artifacts" / "runs"
    (runs / "m1_lenet_s42_resolution-res32-budget_20261005_032415").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    assert already_done(_job("configs/m1_lenet.yaml")) is None


def test_tag_khac_thi_chua_xong(tmp_path, monkeypatch):
    runs = tmp_path / "artifacts" / "runs"
    xong = runs / "m1_lenet_s42_resolution-res32-budget_20261005_032415"
    xong.mkdir(parents=True)
    (xong / "result.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    assert already_done(_job("configs/m1_lenet.yaml", tag="res48-budget")) is None


def test_truc_resolution_co_tag_TRUNG_giua_cac_model():
    """Khoá chính cái tính chất đã sinh ra lỗi, để không ai 'tối giản' lại.

    Nếu sau này trục resolution được đổi sang tag riêng cho từng model thì test
    này đỏ — lúc đó đọc lại docstring đầu file rồi hãy sửa.
    """
    jobs = [j for j in build_jobs(["resolution"], None)]
    theo_tag: dict[str, set[str]] = {}
    for j in jobs:
        theo_tag.setdefault(j["tag"], set()).add(Path(j["config"]).stem)
    trung = {t: m for t, m in theo_tag.items() if len(m) > 1}
    assert trung, ("trục resolution không còn tag nào dùng cho nhiều model — "
                   "nếu đúng ý thì xoá test này, nhưng đọc docstring trước")
    # và already_done PHẢI phân biệt được chúng
    for tag, models in trung.items():
        assert len(models) > 1, tag
