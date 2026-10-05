"""
speed — đo latency, FLOPs, dung lượng model cho triển khai biên.

CHỦ: Phong Trần

★ ĐO LATENCY CHO ĐÚNG — SAI MỘT BƯỚC LÀ MỌI KẾT LUẬN VỀ EDGE VÔ GIÁ TRỊ ★

  1. model.eval() + torch.no_grad()
     Tắt dropout/BN-train và không dựng graph autograd.

  2. >= 20 vòng WARM-UP
     Lần chạy đầu tốn thời gian khởi tạo kernel, cấp phát bộ nhớ, nạp cache.
     Đo luôn từ lần đầu thì số bị thổi lên nhiều lần.

  3. ★ ĐỒNG BỘ THIẾT BỊ TRƯỚC VÀ SAU KHI BẤM GIỜ ★
     GPU/MPS chạy BẤT ĐỒNG BỘ: lệnh model(x) trả về NGAY khi đã xếp hàng xong,
     chưa tính xong. Không sync thì ta đang đo thời gian GỬI LỆNH, không phải
     thời gian TÍNH — sai hàng chục lần. Đây là lỗi phổ biến nhất khi benchmark.

  4. 100 vòng đo, báo p50 VÀ p95
     Hệ thống thời gian thực quan tâm TRƯỜNG HỢP XẤU. Mean che mất đuôi phân phối.

  5. batch=1 VÀ batch=64
     batch=1 là tình huống xe thật (xử lý từng khung ảnh);
     batch=64 đo throughput.

  6. Ghi rõ THIẾT BỊ và PHIÊN BẢN thư viện
     Số latency không so sánh được giữa hai máy khác nhau.

  7. ★ MÁY PHẢI RẢNH — quy tắc này đã bị vi phạm thật trong dự án ★
     Chạy edge_export song song với run_robustness làm MỌI model chậm
     1,3-2,3 lần, và không có gì báo động:

       model            p50 máy rảnh   p50 máy bận   p95/p50 rảnh   p95/p50 bận
       m1_lenet              0,95 ms       2,16 ms           1,34          3,71
       m2_vggres             3,22 ms       4,12 ms           1,12          1,33
       m3_resnet18          15,62 ms      21,67 ms           1,07          1,26
       m3_mobilenetv2       35,94 ms      58,56 ms           1,04          1,24
       m3_effnetb0         163,85 ms     272,17 ms           1,05          1,08

     Model càng NHANH càng bị méo nặng: kernel của M1 ngắn hơn một lát
     scheduler, nên nhiễu hệ điều hành chiếm phần lớn thời gian đo. Đúng
     những model nhẹ — thứ ta muốn chọn cho thiết bị biên — lại là thứ bị
     đo sai nhiều nhất.

     Tỉ lệ p95/p50 là DẤU HIỆU TỰ PHÁT HIỆN: máy rảnh cho 1,0-1,4; vượt
     2,0 thì gần như chắc chắn có tiến trình khác tranh CPU. benchmark()
     tự kiểm và cảnh báo (xem `nhieu_he_thong` trong kết quả).

★ FLOPs KHÔNG PHẢI LATENCY — bài học quan trọng nhất của phần này ★

  SỐ ĐO THẬT của dự án (CPU, batch=1, M1 Pro):

      model          FLOPs      latency p50     ms/GFLOP
      m1_lenet       0,075 G      0,93 ms         12,41
      m2_vggres      0,290 G      3,16 ms         10,89
      m3_resnet18    3,647 G     13,65 ms          3,74

  Nếu latency tỉ lệ với FLOPs thì cột "ms/GFLOP" phải BẰNG NHAU. Thực tế nó
  chênh 3,3 lần. Trên MPS batch=64 cũng vậy (chênh 2,2 lần).

  ★ VÀ CHIỀU CỦA NÓ NGƯỢC VỚI VÍ DỤ KINH ĐIỂN ★
  Câu chuyện thường được kể là "MobileNetV2 ít FLOPs hơn ResNet18 6 lần nhưng
  không nhanh hơn 6 lần" — tức model NHỎ kém hiệu quả trên mỗi FLOP.
  Ở đây ta đo được: model LỚN NHẤT (M3) lại HIỆU QUẢ NHẤT trên mỗi FLOP
  (3,74 so với 12,41 của M1). Vì sao:

    (a) TENSOR QUÁ NHỎ KHÔNG LẤP ĐẦY PHẦN CỨNG. M1/M2 chạy ở 48x48 với vài chục
        kênh — mỗi phép conv quá bé để che được chi phí cố định: launch kernel,
        độ trễ truy cập bộ nhớ, đồng bộ. Phần cứng ngồi chờ nhiều hơn là tính.
        M3 chạy ở 224x224 với conv dày đặc -> giữ phần cứng bận liên tục.
    (b) ARITHMETIC INTENSITY. FLOPs chỉ đếm phép tính, không đếm TRUY CẬP BỘ NHỚ.
        Conv dày đặc ở độ phân giải lớn có tỉ lệ tính-toán/byte cao; conv nhỏ
        (và depthwise conv) thì thấp -> bị chặn bởi BĂNG THÔNG BỘ NHỚ.
    (c) MỨC ĐỘ TỐI ƯU THƯ VIỆN. cuDNN/BLAS tối ưu rất kỹ cho các kích thước
        tensor phổ biến của ImageNet (224x224); kích thước lạ ít được chăm hơn.

  Lưu ý cho báo cáo: cả hai chiều đều dẫn tới CÙNG MỘT kết luận hành động —
  muốn nói về triển khai thì PHẢI ĐO WALL-CLOCK trên thiết bị đích, không suy
  từ FLOPs. Nhưng phải trình bày đúng chiều mà MÌNH đo được, không chép chiều
  của bài báo khác.
"""

from __future__ import annotations

import platform
import time

import numpy as np
import torch
import torch.nn as nn

from gtsrb.utils.logging import get_logger

log = get_logger()


def _synchronize(device: torch.device) -> None:
    """Chờ thiết bị làm xong mọi việc đang xếp hàng. BƯỚC KHÔNG ĐƯỢC BỎ."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    # CPU thì đồng bộ sẵn, không cần làm gì


# ---------------------------------------------------------------------
# Hai dấu hiệu "máy đang bận", vì MỘT dấu hiệu là không đủ
# ---------------------------------------------------------------------
# Đo trên chính 5 model của bài, máy rảnh so với máy bận:
#
#   model            chậm hơn   p95/p50 rảnh -> bận
#   m1_lenet            2,27x       1,34 -> 3,71   jitter BẮT ĐƯỢC
#   m2_vggres           1,28x       1,12 -> 1,33   jitter bỏ sót
#   m3_resnet18         1,39x       1,07 -> 1,26   jitter bỏ sót
#   m3_mobilenetv2      1,63x       1,04 -> 1,24   jitter bỏ sót
#   m3_effnetb0         1,66x       1,05 -> 1,08   jitter bỏ sót
#
# jitter chỉ bắt được model NHANH: kernel của M1 ngắn hơn một lát scheduler
# nên nhiễu hệ điều hành lộ ra ở đuôi phân phối. Với model chậm, nhiễu rải
# đều trên một kernel dài nên p50 và p95 cùng giãn ra — tỉ lệ không đổi, dù
# số tuyệt đối sai 66%.
#
# Vì vậy phải kiểm thêm TẢI HỆ THỐNG, dấu hiệu không phụ thuộc model chạy
# nhanh hay chậm. Hai dấu hiệu bù cho nhau: jitter nhạy với model nhanh,
# tải hệ thống nhạy với mọi model nhưng thô hơn.
NGUONG_JITTER = 2.0
NGUONG_TAI_MOI_LOI = 0.25


def _tai_moi_loi(loi_cua_minh: float = 1.0) -> float | None:
    """Tải do TIẾN TRÌNH KHÁC gây ra, trên mỗi lõi.

    ★ BẢN ĐẦU CỦA HÀM NÀY BÁO ĐỘNG SAI ★

    Nó trừ đúng 1,0 cho "phần của chính benchmark", tức ngầm cho rằng benchmark
    chạy một luồng. Nhưng PyTorch trên CPU mặc định dùng torch.get_num_threads()
    lõi — ở máy này là 6, và đo được 326% CPU. Nên khi đo EfficientNet-B0 trên
    một máy HOÀN TOÀN RẢNH, tải đọc được là 5,1 trên 8 lõi, trừ 1,0 còn
    4,1/8 = 0,51 > 0,25 -> BÁO ĐỘNG dù không có gì khác chạy.

    Đó đúng là thứ hàm này được viết ra để chống: một phép kiểm hay báo động
    sai sẽ bị phớt đi, rồi lần nó báo đúng cũng không ai tin.

    Cách sửa: `loi_cua_minh` là số lõi mà CHÍNH tiến trình này dùng trong lúc
    đo, tính từ thời gian CPU thật (time.process_time() đếm CPU của mọi luồng)
    chia cho thời gian thực. Trừ con số đó thay vì trừ 1,0.
    """
    try:
        import os
        loi = os.cpu_count() or 1
        return max(0.0, os.getloadavg()[0] - loi_cua_minh) / loi
    except (OSError, AttributeError):
        return None


def _tai_he_thong() -> str:
    """Chuỗi tải hệ thống để ghi vào cảnh báo."""
    try:
        import os
        tai, loi = os.getloadavg()[0], os.cpu_count() or 1
        return f"{tai:.1f} trên {loi} lõi = {tai / loi:.2f}/lõi"
    except (OSError, AttributeError):
        return "không đọc được"


@torch.no_grad()
def benchmark(model: nn.Module, img_size: int, device: torch.device,
              batch_sizes: tuple[int, ...] = (1, 64),
              warmup: int = 20, iters: int = 100) -> dict:
    """Đo latency. Trả dict dạng {"<device>_bs<N>_p50": ms, "..._p95": ms, ...}."""
    model = model.to(device).eval()
    results: dict[str, float] = {}

    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 3, img_size, img_size, device=device)

        # --- Warm-up ---
        for _ in range(warmup):
            model(x)
        _synchronize(device)

        # --- Đo ---
        # Bấm cả thời gian CPU của chính tiến trình, để biết nó dùng mấy lõi.
        # process_time() đếm CPU của MỌI luồng, nên tỉ số này ra đúng số lõi
        # mà PyTorch đang thật sự dùng (xem _tai_moi_loi).
        cpu_dau, thuc_dau = time.process_time(), time.perf_counter()
        timings_ms: list[float] = []
        for _ in range(iters):
            _synchronize(device)                 # chắc chắn mọi việc trước đã xong
            start = time.perf_counter()
            model(x)
            _synchronize(device)                 # chờ tính xong MỚI bấm giờ kết thúc
            timings_ms.append((time.perf_counter() - start) * 1000.0)

        thuc_het = time.perf_counter() - thuc_dau
        loi_cua_minh = ((time.process_time() - cpu_dau) / thuc_het
                        if thuc_het > 0 else 1.0)
        array = np.array(timings_ms)
        prefix = f"{device.type}_bs{batch_size}"
        results[f"{prefix}_p50"] = float(np.percentile(array, 50))
        results[f"{prefix}_p95"] = float(np.percentile(array, 95))
        results[f"{prefix}_mean"] = float(array.mean())
        # Throughput: ảnh/giây, tính từ p50
        results[f"{prefix}_imgs_per_sec"] = float(
            batch_size / (np.percentile(array, 50) / 1000.0))

        # ---- Tự phát hiện máy đang BẬN (xem quy tắc 7 ở đầu file) ----
        p50, p95 = results[f"{prefix}_p50"], results[f"{prefix}_p95"]
        jitter = p95 / p50 if p50 > 0 else 0.0
        results[f"{prefix}_jitter"] = round(jitter, 2)

        log.info("  %-14s p50=%7.2f ms  p95=%7.2f ms  (%.0f ảnh/s)  p95/p50=%.2f",
                 prefix, p50, p95, results[f"{prefix}_imgs_per_sec"], jitter)
        tai = _tai_moi_loi(loi_cua_minh)
        results[f"{prefix}_loi_dung"] = round(loi_cua_minh, 2)
        ly_do = []
        if jitter > NGUONG_JITTER:
            ly_do.append(f"p95/p50 = {jitter:.2f} > {NGUONG_JITTER:.1f}")
        if tai is not None and tai > NGUONG_TAI_MOI_LOI:
            ly_do.append(f"tải NGOÀI {tai:.2f}/lõi > {NGUONG_TAI_MOI_LOI:.2f} "
                         f"(đã trừ {loi_cua_minh:.1f} lõi của chính phép đo)")
        if ly_do:
            log.warning("  ★ MÁY ĐANG BẬN (%s) — số latency này có thể chậm "
                        "1,3-2,3 lần so với thực tế. Đóng các việc nặng khác rồi "
                        "ĐO LẠI. (tải hệ thống: %s)",
                        " · ".join(ly_do), _tai_he_thong())
            results[f"{prefix}_nhieu_he_thong"] = True

    return results


def model_cost(model: nn.Module, img_size: int) -> dict:
    """Số tham số, FLOPs, dung lượng file fp32.

    FLOPs tính bằng thop nếu có; nếu thiếu thop thì trả None thay vì vỡ
    (phần latency vẫn đo được và quan trọng hơn).
    """
    params = sum(p.numel() for p in model.parameters())
    # fp32 = 4 byte mỗi tham số
    size_mb = params * 4 / 1e6

    flops_g = None
    try:
        from thop import profile
        device = next(model.parameters()).device
        dummy = torch.zeros(1, 3, img_size, img_size, device=device)
        # thop in log khá ồn -> verbose=False
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        # thop trả MACs (multiply-accumulate). 1 MAC = 2 FLOPs theo quy ước
        # phổ biến trong các bài báo. Ta báo theo quy ước đó cho so sánh được.
        flops_g = float(macs * 2 / 1e9)
    except ImportError:
        log.warning("Chưa cài thop -> không tính được FLOPs. pip install thop")
    except Exception as exc:
        log.warning("thop lỗi trên model này: %s", exc)

    return {
        "params_m": round(params / 1e6, 4),
        "flops_g": round(flops_g, 4) if flops_g is not None else None,
        "size_mb": round(size_mb, 2),
    }


def environment_info(device: torch.device) -> dict:
    """Thông tin môi trường — BẮT BUỘC ghi kèm, vì latency không so sánh được
    giữa hai máy khác nhau."""
    info = {
        "device": str(device),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    return info
