"""
test_aug_thu_tu.py — khoá cờ `aug_after_resize` của GTSRBDataset.

CHỦ: Huy (đường ống dữ liệu)

★ VÌ SAO CÓ CỜ NÀY ★

Mặc định, augmentation được áp ở ĐỘ PHÂN GIẢI CACHE rồi mới nội suy lên
img_size — một tối ưu tốc độ hợp lý (xoay ảnh 48×48 rẻ hơn xoay 224×224).

Nhưng với M3 (cache 48, img_size 224) nó khiến ảnh bị **lấy mẫu lại hai lần**:
một lần khi quay/dịch ở 48px, một lần khi nội suy lên 224px. M1/M2 chỉ bị một
lần vì cache khớp img_size.

Tức đường ống mặc định BẤT LỢI cho M3 — đúng nhóm mô hình mà kết luận chính
của báo cáo nói là "không giúp được gì". Cờ này tồn tại để ĐO xem bất lợi đó
lớn bao nhiêu, chứ không phải để sửa mặc định.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from gtsrb.data.dataset import GTSRBDataset
from gtsrb.data.transforms import build_transform

INDEX = Path(__file__).resolve().parent.parent / "data/processed/index.csv"
pytestmark = pytest.mark.skipif(
    not INDEX.exists(), reason="chưa có data/processed/index.csv")


def _ds(aug_after_resize: bool, img_size: int = 112, transform=None):
    return GTSRBDataset(
        index_csv=str(INDEX), split="train", img_size=img_size,
        preprocess="clahe", normalize="imagenet",
        processed_dir=str(INDEX.parent), cache_size=48,
        aug_after_resize=aug_after_resize, transform=transform,
        return_uint8=True)


@pytest.mark.parametrize("sau", [False, True])
def test_ca_hai_nhanh_cho_dung_img_size(sau):
    """Đổi thứ tự không được làm sai kích thước đầu ra."""
    x, _ = _ds(sau, transform=build_transform("geo_photo", 48))[0]
    assert tuple(x.shape) == (3, 112, 112)


def test_khong_augment_thi_hai_nhanh_GIONG_NHAU():
    """transform=None -> chỉ còn resize, hai nhánh phải ra ảnh y hệt.

    Đây là phép kiểm quan trọng nhất: nó chứng minh cờ CHỈ đổi thứ tự
    augmentation, không vô tình đổi cách nội suy hay cách chuẩn hoá.
    """
    a, _ = _ds(False)[0]
    b, _ = _ds(True)[0]
    assert torch.equal(a, b), "không có augmentation thì hai nhánh phải trùng khít"


def test_cache_khop_img_size_thi_thu_tu_KHONG_quan_trong():
    """Khi cache == img_size (M1/M2) thì không có bước resize nào -> giống nhau.

    Chính vì vậy vấn đề này CHỈ ảnh hưởng M3.
    """
    tf = build_transform("geo_photo", 48)
    torch.manual_seed(0)
    a, _ = _ds(False, img_size=48, transform=tf)[0]
    torch.manual_seed(0)
    b, _ = _ds(True, img_size=48, transform=tf)[0]
    assert torch.equal(a, b), "cache khớp img_size thì hai nhánh phải trùng"


def test_co_augment_va_cache_khac_size_thi_hai_nhanh_KHAC_nhau():
    """Có augmentation + cache khác img_size -> phải khác, nếu không cờ vô dụng."""
    tf = build_transform("geo_photo", 48)
    torch.manual_seed(0)
    a, _ = _ds(False, transform=tf)[0]
    torch.manual_seed(0)
    b, _ = _ds(True, transform=tf)[0]
    assert not torch.equal(a, b), \
        "hai nhánh ra ảnh y hệt -> cờ aug_after_resize không có tác dụng gì"


def test_mac_dinh_la_augment_TRUOC_resize():
    """Mặc định phải giữ nguyên hành vi cũ, để mọi run trước đây còn so được.

    Kiểm ở mức chữ ký hàm, không dựng dataset — normalize='gtsrb' đòi mean/std
    nên dựng thật sẽ lỗi vì lý do không liên quan tới cờ này.
    """
    import inspect
    mac_dinh = inspect.signature(GTSRBDataset.__init__).parameters["aug_after_resize"]
    assert mac_dinh.default is False, \
        "đổi mặc định sẽ làm mọi run trước đây không so được với run sau"
