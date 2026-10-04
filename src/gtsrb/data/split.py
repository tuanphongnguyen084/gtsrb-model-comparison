"""
split — chia train/val THEO TRACK để chống rò rỉ dữ liệu.

CHỦ: Huy

VẤN ĐỀ:
  GTSRB quay mỗi biển báo VẬT LÝ thành một track 30 frame liên tiếp.
  Frame 12 và frame 13 của cùng một track gần như là hai bản sao của nhau.

  Nếu trộn tất cả ảnh rồi cắt 80/20 ngẫu nhiên:
      frame 12 -> train
      frame 13 -> val        <- gần giống hệt frame 12!
  Model không cần KHÁI QUÁT sang biển báo mới, nó chỉ cần NHỚ tấm biển đã thấy.
  Val accuracy đẹp, nhưng con số đó không nói gì về hiệu năng trên biển báo lạ.
  Đó là RÒ RỈ DỮ LIỆU (data leakage).

CÁCH ĐÚNG:
  Group-aware split. Định nghĩa nhóm g = (class_id, track_id), rồi dùng
  StratifiedGroupKFold để (a) mọi phần tử cùng một g chỉ nằm ở MỘT phía, và
  (b) tỉ lệ 43 lớp vẫn giữ gần như nhau hai phía.

DẤU HIỆU NHẬN BIẾT CÓ RÒ RỈ: val cao hơn test một cách bất thường.
Split đúng thì val và test phải gần nhau, vì cả hai đều là "biển báo chưa từng thấy".

Hàm này hỗ trợ CẢ HAI cách để nhóm báo cáo đối chứng — group_by_track=False
không phải để dùng thật, nó là để ĐO xem rò rỉ làm số liệu cao lên bao nhiêu.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from gtsrb.utils.logging import get_logger

log = get_logger()


def make_split(index_csv: str | Path, val_ratio: float = 0.2, seed: int = 42,
               group_by_track: bool = True) -> pd.DataFrame:
    """Ghi thêm cột 'split' ('train' | 'val' | 'test') vào index.csv.

    - Ảnh có source='test_official' luôn được gán split='test' và KHÔNG BAO GIỜ
      bị chia vào train/val. Tập test bị niêm phong, chỉ mở một lần ở cuối.
    - Phần source='train' được chia thành train/val theo val_ratio.
    """
    index_csv = Path(index_csv)
    frame = pd.read_csv(index_csv)

    frame["split"] = "test"                                  # mặc định
    is_train_pool = frame["source"] == "train"
    pool = frame[is_train_pool]

    if len(pool) == 0:
        raise ValueError("index.csv không có dòng nào source='train'")

    labels = pool["class_id"].to_numpy()
    # Nhóm = (lớp, track). Phải kèm class_id vì track_id được đánh số lại
    # trong mỗi thư mục lớp -> track 5 của lớp 2 khác track 5 của lớp 7.
    groups = np.array([f"{c}_{t}" for c, t in zip(pool["class_id"], pool["track_id"])])

    if group_by_track:
        # n_splits = round(1/val_ratio) -> val_ratio=0.2 cho 5 fold, lấy fold đầu
        n_splits = max(2, int(round(1.0 / val_ratio)))
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_pos, val_pos = next(splitter.split(pool, labels, groups))
        mode_text = f"THEO TRACK (StratifiedGroupKFold, {n_splits} fold)"
    else:
        # ĐỐI CHỨNG: split random theo ảnh. CÓ RÒ RỈ. Chỉ dùng để báo cáo so sánh.
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        # groups = chính index của từng ảnh -> mỗi ảnh một nhóm -> hoá thành random split
        fake_groups = np.arange(len(pool))
        train_pos, val_pos = next(splitter.split(pool, labels, fake_groups))
        mode_text = "RANDOM THEO ẢNH (CÓ RÒ RỈ — chỉ để đối chứng)"

    # train_pos/val_pos là vị trí trong `pool`, cần đổi về index của `frame`
    pool_index = pool.index.to_numpy()
    frame.loc[pool_index[train_pos], "split"] = "train"
    frame.loc[pool_index[val_pos], "split"] = "val"

    frame.to_csv(index_csv, index=False)

    counts = frame["split"].value_counts()
    log.info("Split: %s", mode_text)
    log.info("  train=%d  val=%d  test=%d",
             counts.get("train", 0), counts.get("val", 0), counts.get("test", 0))

    report = check_leakage(frame)
    if group_by_track and report["n_shared_tracks"] > 0:
        raise RuntimeError(
            f"Split theo track mà vẫn có {report['n_shared_tracks']} track dùng chung "
            f"giữa train và val. Đây là lỗi nghiêm trọng, dừng lại."
        )
    log.info("  số track dùng chung giữa train và val: %d (%s)",
             report["n_shared_tracks"],
             "OK, không rò rỉ" if report["n_shared_tracks"] == 0 else "CÓ RÒ RỈ")
    log.info("  lệch phân phối lớp train vs val lớn nhất: %.3f%%",
             report["max_class_ratio_gap_pct"])
    return frame


def check_leakage(frame: pd.DataFrame) -> dict:
    """Kiểm tra rò rỉ. Dùng trong tests/test_split.py — đây là cổng chặn của dự án.

    Trả về:
      n_shared_tracks        : số track xuất hiện ở CẢ train và val (phải = 0)
      shared_tracks          : vài ví dụ, để debug
      max_class_ratio_gap_pct: lệch tỉ lệ lớp lớn nhất giữa train và val, tính %
    """
    train = frame[frame["split"] == "train"]
    val = frame[frame["split"] == "val"]

    def group_keys(part: pd.DataFrame) -> set[str]:
        """Tập khoá (class_id, track_id) của một phần dữ liệu."""
        return {f"{c}_{t}" for c, t in zip(part["class_id"], part["track_id"])}

    shared = group_keys(train) & group_keys(val)

    # So sánh phân phối lớp: tỉ lệ từng lớp ở train so với ở val
    train_ratio = train["class_id"].value_counts(normalize=True).sort_index()
    val_ratio = val["class_id"].value_counts(normalize=True).sort_index()
    aligned = pd.concat([train_ratio, val_ratio], axis=1).fillna(0.0)
    max_gap = float((aligned.iloc[:, 0] - aligned.iloc[:, 1]).abs().max() * 100)

    return {
        "n_shared_tracks": len(shared),
        "shared_tracks": sorted(shared)[:10],
        "max_class_ratio_gap_pct": max_gap,
    }
