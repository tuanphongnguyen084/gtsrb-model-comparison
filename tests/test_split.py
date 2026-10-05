"""
test_split.py — ★ CỔNG CHẶN QUAN TRỌNG NHẤT CỦA DỰ ÁN ★

CHỦ: Huy

Test này chứng minh KHÔNG CÓ RÒ RỈ DỮ LIỆU giữa train và val.
Nếu nó đỏ thì MỌI SỐ LIỆU của dự án đều vô giá trị — vì val accuracy đang đo
"model nhớ được bao nhiêu" chứ không phải "model khái quát được bao nhiêu".

Chạy: pytest tests/test_split.py -v
"""
from pathlib import Path

import pandas as pd
import pytest

from gtsrb.data.split import check_leakage, make_split

INDEX_CSV = Path("data/processed/index.csv")

requires_data = pytest.mark.skipif(
    not INDEX_CSV.exists(),
    reason="Chưa có data/processed/index.csv. Chạy: python scripts/prepare_data.py",
)


@requires_data
def test_khong_co_track_dung_chung_giua_train_va_val():
    """★ Test cốt lõi: giao của tập track giữa train và val phải RỖNG.

    GTSRB quay mỗi biển báo vật lý thành 30 frame liên tiếp. Nếu một track
    xuất hiện ở cả train và val thì model đã thấy chính tấm biển đó lúc train.
    """
    frame = pd.read_csv(INDEX_CSV)
    report = check_leakage(frame)
    assert report["n_shared_tracks"] == 0, (
        f"RÒ RỈ DỮ LIỆU: {report['n_shared_tracks']} track xuất hiện ở cả train và val. "
        f"Ví dụ: {report['shared_tracks']}. "
        f"Chạy lại: python scripts/prepare_data.py (không có --no-group-by-track)"
    )


@requires_data
def test_phan_phoi_lop_train_va_val_gan_nhau():
    """Stratify phải giữ tỉ lệ 43 lớp gần như nhau ở hai phía."""
    frame = pd.read_csv(INDEX_CSV)
    report = check_leakage(frame)
    assert report["max_class_ratio_gap_pct"] < 1.0, (
        f"Lệch phân phối lớp {report['max_class_ratio_gap_pct']:.3f}% > 1%. "
        f"StratifiedGroupKFold có thể đã không stratify đúng."
    )


@requires_data
def test_tap_test_khong_bao_gio_nam_trong_train_hay_val():
    """Tập test chính thức phải được NIÊM PHONG hoàn toàn."""
    frame = pd.read_csv(INDEX_CSV)
    official_test = frame[frame["source"] == "test_official"]
    assert len(official_test) == 12_630, \
        f"Mong đợi 12.630 ảnh test chính thức, có {len(official_test)}"
    assert (official_test["split"] == "test").all(), \
        "Có ảnh test chính thức bị gán vào train hoặc val — tập test đã bị phá niêm phong"

    train_pool = frame[frame["source"] == "train"]
    assert not (train_pool["split"] == "test").any(), \
        "Có ảnh train-pool bị gán split='test' — làm test không còn là test chính thức"


@requires_data
def test_du_51839_anh():
    """Bộ dữ liệu đầy đủ: 39.209 train + 12.630 test.

    Test này bắt đúng cái bẫy đã gặp thật: torchvision tải bộ train 26.640 ảnh
    của giải IJCNN 2011 thay vì bộ Final Training 39.209 ảnh.
    """
    frame = pd.read_csv(INDEX_CSV)
    n_train = int((frame["source"] == "train").sum())
    assert n_train == 39_209, (
        f"Chỉ có {n_train} ảnh train, mong đợi 39.209. "
        f"Nếu là 26.640 thì đang dùng bộ IJCNN 2011 của torchvision — "
        f"chạy: python scripts/prepare_data.py --download"
    )
    assert len(frame) == 51_839


@requires_data
def test_co_du_cac_cot_theo_hop_dong():
    """index.csv phải có đủ cột theo docs/INTERFACE.md mục 1."""
    frame = pd.read_csv(INDEX_CSV)
    required = {"path", "class_id", "track_id", "frame_id",
                "roi_x1", "roi_y1", "roi_x2", "roi_y2",
                "width", "height", "source", "split"}
    missing = required - set(frame.columns)
    assert not missing, f"index.csv thiếu cột: {missing}"
    assert frame["class_id"].between(0, 42).all(), "class_id phải trong 0..42"


@requires_data
def test_split_random_CO_ro_ri__doi_chung():
    """Chứng minh ngược lại: split random theo ảnh THỰC SỰ gây rò rỉ.

    Test này không kiểm code đúng, nó CHỨNG MINH VẤN ĐỀ TỒN TẠI.
    Đây là bằng chứng cho phần báo cáo. Nó tạo file tạm, không phá index.csv thật.
    """
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        temp_csv = Path(tmp) / "index.csv"
        shutil.copy(INDEX_CSV, temp_csv)

        make_split(temp_csv, val_ratio=0.2, seed=42, group_by_track=False)
        leaky = check_leakage(pd.read_csv(temp_csv))

    assert leaky["n_shared_tracks"] > 0, (
        "Split random mà KHÔNG có track dùng chung — bất thường. "
        "Kiểm lại hàm make_split(group_by_track=False)."
    )
    # Con số này chính là bằng chứng đưa vào báo cáo
    print(f"\n  ĐỐI CHỨNG: split random cho {leaky['n_shared_tracks']} track "
          f"dùng chung giữa train và val (split theo track cho 0).")


# =====================================================================
# Cái bẫy: track_id MỘT MÌNH không định danh được biển báo
# =====================================================================

def test_track_id_mot_minh_KHONG_dinh_danh_duoc_bien_bao():
    """★ Nhóm phải là (class_id, track_id), KHÔNG được chỉ track_id.

    GTSRB đánh số track LẠI TỪ ĐẦU trong mỗi lớp: lớp 0 có track 0..N, lớp 1
    cũng có track 0..M. Nên `track_id` một mình bị dùng chung giữa các lớp.

    Số đo thật trên index.csv của dự án:

        nhóm theo track_id        ->    75 nhóm / 39.209 ảnh = 523 ảnh/nhóm (VÔ LÝ)
        nhóm theo (class, track)  -> 1.307 nhóm / 39.209 ảnh =  30 ảnh/nhóm (ĐÚNG)

    30 ảnh/nhóm khớp đặc tả GTSRB: mỗi biển báo vật lý được quay 30 frame.

    Hậu quả nếu ai "tối giản" `groups` về chỉ `track_id`: StratifiedGroupKFold
    sẽ gom các biển báo KHÁC NHAU của các lớp khác nhau vào cùng một nhóm, rồi
    chia nhóm đó nguyên khối — split trông vẫn "group-aware" và check_leakage
    kiểu cũ vẫn báo 0, nhưng phân phối lớp sẽ vỡ và nhiều lớp mất hẳn khỏi val.
    Test này chặn đúng việc đó.
    """
    frame = pd.read_csv(INDEX_CSV)
    pool = frame[frame["split"].isin(["train", "val"])]

    chi_track = pool["track_id"].nunique()
    doi = pool.groupby(["class_id", "track_id"]).ngroups

    assert doi > chi_track * 5, (
        f"khoá đôi phải cho NHIỀU nhóm hơn hẳn: {doi} vs {chi_track}. "
        f"Nếu hai số gần nhau thì track_id đã là duy nhất toàn cục và chú thích "
        f"trong split.py cần xem lại.")

    anh_moi_nhom = len(pool) / doi
    assert 25 <= anh_moi_nhom <= 35, (
        f"{anh_moi_nhom:.1f} ảnh mỗi nhóm — GTSRB quay 30 frame/biển báo, nên "
        f"con số này phải quanh 30. Lệch nhiều nghĩa là khoá nhóm sai.")


def test_make_split_dung_khoa_DOI_khong_phai_track_id_don():
    """Đọc trực tiếp nguồn make_split để chắc nó ghép class_id vào groups."""
    import inspect

    from gtsrb.data import split as mod
    src = inspect.getsource(mod.make_split)
    dong_groups = [l.strip() for l in src.splitlines()
                   if "groups" in l and "=" in l and "fake" not in l]
    assert dong_groups, "không tìm thấy dòng gán `groups` trong make_split"
    ghep = "\n".join(dong_groups)
    assert "class_id" in ghep, (
        f"`groups` không có class_id — đây là lỗi RÒ RỈ IM LẶNG.\n"
        f"Dòng tìm được:\n{ghep}")
