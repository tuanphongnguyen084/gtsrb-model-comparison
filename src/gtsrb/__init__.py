"""
gtsrb — thư viện dùng chung cho dự án phân loại 43 lớp biển báo GTSRB.

Mọi hằng số dùng chung khai báo ở đây, KHÔNG hard-code lại ở chỗ khác.
"""

NUM_CLASSES = 43

# Tên 43 lớp. Thứ tự = class_id (0..42), đúng theo thứ tự thư mục của GTSRB.
CLASS_NAMES = [
    "Giới hạn 20 km/h",                  # 0
    "Giới hạn 30 km/h",                  # 1
    "Giới hạn 50 km/h",                  # 2
    "Giới hạn 60 km/h",                  # 3
    "Giới hạn 70 km/h",                  # 4
    "Giới hạn 80 km/h",                  # 5
    "Hết giới hạn 80 km/h",              # 6
    "Giới hạn 100 km/h",                 # 7
    "Giới hạn 120 km/h",                 # 8
    "Cấm vượt",                          # 9
    "Cấm vượt (xe trên 3,5 tấn)",        # 10
    "Ưu tiên ở giao lộ kế tiếp",         # 11
    "Đường ưu tiên",                     # 12
    "Nhường đường",                      # 13
    "Dừng lại",                          # 14
    "Cấm mọi loại xe",                   # 15
    "Cấm xe trên 3,5 tấn",               # 16
    "Cấm đi vào",                        # 17
    "Cảnh báo chung",                    # 18
    "Đường cong nguy hiểm sang trái",    # 19  <- cặp gương với 20
    "Đường cong nguy hiểm sang phải",    # 20
    "Đường cong đôi",                    # 21
    "Đường xấu",                         # 22
    "Đường trơn",                        # 23
    "Đường hẹp bên phải",                # 24
    "Đang thi công",                     # 25
    "Đèn tín hiệu",                      # 26
    "Người đi bộ",                       # 27
    "Trẻ em qua đường",                  # 28
    "Xe đạp qua đường",                  # 29
    "Cẩn thận băng/tuyết",               # 30
    "Động vật hoang dã qua đường",       # 31
    "Hết mọi giới hạn tốc độ và cấm vượt",  # 32
    "Rẽ phải phía trước",                # 33  <- cặp gương với 34
    "Rẽ trái phía trước",                # 34
    "Chỉ đi thẳng",                      # 35
    "Đi thẳng hoặc rẽ phải",             # 36  <- cặp gương với 37
    "Đi thẳng hoặc rẽ trái",             # 37
    "Đi bên phải",                       # 38  <- cặp gương với 39
    "Đi bên trái",                       # 39
    "Vòng xuyến bắt buộc",               # 40
    "Hết cấm vượt",                      # 41
    "Hết cấm vượt (xe trên 3,5 tấn)",    # 42
]
assert len(CLASS_NAMES) == NUM_CLASSES

# Các cặp lớp là ẢNH GƯƠNG của nhau.
# Đây là lý do CẤM RandomHorizontalFlip: lật ngang lớp 33 ra đúng hình lớp 34,
# tức là tự tạo dữ liệu SAI NHÃN. Xem docs/LY_THUYET.md mục 1.3.
MIRROR_PAIRS = [(19, 20), (33, 34), (36, 37), (38, 39)]

# Thống kê chuẩn hoá của ImageNet — chỉ dùng cho M3 (transfer learning),
# vì trọng số pretrained và running stats của BatchNorm được học trên phân phối này.
# M1/M2 train từ số 0 thì dùng thống kê của GTSRB, tính CHỈ trên tập train.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

__all__ = [
    "NUM_CLASSES", "CLASS_NAMES", "MIRROR_PAIRS", "IMAGENET_MEAN", "IMAGENET_STD",
]
