"""
export — nén và xuất model cho thiết bị biên: quantization int8, TorchScript, ONNX.

CHỦ: Phong Trần

★ VÌ SAO PHẦN NÀY QUAN TRỌNG VỚI ĐỀ BÀI ★
Đề yêu cầu "inference speed comparison for edge deployment". Đo latency của model
fp32 mới là một nửa câu chuyện: thiết bị biên thật (Raspberry Pi, Jetson, vi điều
khiển trên xe) hiếm khi chạy fp32. Chúng chạy int8 vì:
  - nhẹ hơn 4 lần trên đĩa và trong RAM
  - nhiều chip có lệnh SIMD int8 chuyên dụng -> nhanh hơn đáng kể
  - tốn ít băng thông bộ nhớ hơn, mà băng thông mới là nút cổ chai (xem speed.py)

BA CÁCH XUẤT, DÙNG CHO BA MỤC ĐÍCH KHÁC NHAU:

  dynamic quantization  Lượng tử hoá TRỌNG SỐ sang int8, activation vẫn fp32 và
                        được lượng tử hoá tại chỗ lúc chạy. KHÔNG cần dữ liệu hiệu
                        chuẩn. Chỉ áp cho Linear/LSTM — với CNN thuần thì lợi ích
                        chủ yếu là GIẢM DUNG LƯỢNG, không phải tăng tốc.
                        -> dùng khi muốn nhanh và không có thời gian hiệu chuẩn.

  static quantization   Lượng tử hoá cả trọng số lẫn activation. CẦN chạy vài trăm
                        ảnh thật để đo dải giá trị activation (calibration).
                        Áp được cho Conv -> lợi ích thật sự cho CNN.
                        -> đây mới là cách dùng cho sản phẩm.

  TorchScript / ONNX    Không nén, nhưng tách model khỏi mã Python để chạy được
                        trên C++ runtime, ONNX Runtime, TensorRT, CoreML...
                        -> cần khi triển khai thật, vì thiết bị biên không có Python.

★ CẢNH BÁO PHẢI NÊU KHI BÁO CÁO ★
Lượng tử hoá LÀM GIẢM ACCURACY. Bắt buộc đo lại accuracy sau khi nén, không được
chỉ báo "nhẹ hơn 4 lần" rồi thôi. Hàm `compare_quantized()` dưới đây làm việc đó.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from gtsrb.utils.logging import get_logger

log = get_logger()


# =====================================================================
# Quantization
# =====================================================================

def pick_backend() -> str:
    """Chọn backend lượng tử hoá theo kiến trúc CPU.

    'qnnpack' cho ARM (Apple Silicon, Raspberry Pi, điện thoại);
    'fbgemm' cho x86 (máy bàn Intel/AMD, đa số server).

    ★ KHÔNG đặt backend là lỗi ĐÃ GẶP THẬT:
        RuntimeError: Didn't find engine for operation quantized::linear_prepack
                      NoQEngine
    PyTorch không tự chọn, và thông báo lỗi không hề gợi ý nguyên nhân.
    """
    available = torch.backends.quantized.supported_engines
    for candidate in ("qnnpack", "fbgemm", "x86"):
        if candidate in available:
            return candidate
    raise RuntimeError(
        f"Không có backend lượng tử hoá nào khả dụng. Hiện có: {available}"
    )


def quantize_dynamic(model: nn.Module, backend: str | None = None) -> nn.Module:
    """Lượng tử hoá động: trọng số Linear -> int8. Không cần dữ liệu hiệu chuẩn.

    Trả về model MỚI; model gốc không bị đổi.

    LƯU Ý cho dự án này: M1 có 97,3% tham số ở lớp Linear nên nó hưởng lợi nhiều
    nhất. M2 dùng GlobalAvgPool nên Linear chỉ còn 11.051 tham số -> dynamic
    quantization gần như không giảm được gì. Đó là một quan sát đáng báo cáo:
    **kiến trúc quyết định việc nén có hiệu quả hay không.**
    """
    torch.backends.quantized.engine = backend or pick_backend()
    model = model.cpu().eval()
    return torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8,
    )


class _QuantWrapper(nn.Module):
    """Bọc model bằng QuantStub/DeQuantStub để lượng tử hoá TĨNH hoạt động.

    ★ VÌ SAO CẦN: static quantization yêu cầu model phải đánh dấu RÕ chỗ nào đổi
    fp32 -> int8 (QuantStub) và chỗ nào đổi ngược lại (DeQuantStub). Model không có
    hai mốc đó vẫn CONVERT THÀNH CÔNG, nhưng khi chạy thì vỡ:

        NotImplementedError: Could not run 'aten::...' with arguments from the
        'QuantizedCPU' backend

    Lỗi xảy ra ở lúc SUY LUẬN chứ không phải lúc convert — nên nếu chỉ bọc try/except
    quanh bước convert thì không bắt được. Đây là lỗi đã gặp thật.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.model = model
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """fp32 vào -> lượng tử hoá -> model int8 -> giải lượng tử -> fp32 ra."""
        return self.dequant(self.model(self.quant(x)))


def _works(model: nn.Module, img_size: int) -> bool:
    """Chạy thử một forward để chắc model lượng tử hoá KHÔNG vỡ lúc suy luận.

    Convert thành công không có nghĩa là chạy được — xem ghi chú ở _QuantWrapper.
    """
    try:
        with torch.no_grad():
            model(torch.randn(1, 3, img_size, img_size))
        return True
    except Exception as exc:
        log.warning("  Model int8 convert được nhưng VỠ lúc chạy: %s",
                    str(exc).splitlines()[0][:110])
        return False


@torch.no_grad()
def quantize_static(model: nn.Module, calibration_loader, img_size: int = 48,
                    n_batches: int = 8, backend: str | None = None) -> nn.Module | None:
    """Lượng tử hoá tĩnh: cả Conv lẫn activation -> int8. CẦN dữ liệu hiệu chuẩn.

    `backend`: 'qnnpack' cho ARM (Apple Silicon, Raspberry Pi), 'fbgemm' cho x86.
    Để None thì tự chọn theo máy.

    Trả None nếu kiến trúc không lượng tử hoá được — trả None thay vì vỡ, vì đây
    là bước tuỳ chọn, không nên chặn cả đường ống.

    Ba bước: fuse (gộp Conv+BN+ReLU) -> prepare + hiệu chuẩn -> convert.
    Bước fuse quan trọng: không fuse thì BatchNorm vẫn chạy fp32 riêng và phần lớn
    lợi ích tốc độ mất đi.
    """
    import copy

    backend = backend or pick_backend()
    wrapped = _QuantWrapper(copy.deepcopy(model).cpu().eval()).eval()

    try:
        torch.backends.quantized.engine = backend
        wrapped.qconfig = torch.ao.quantization.get_default_qconfig(backend)

        # Gộp Conv+BN+ReLU thành một op trước khi lượng tử hoá
        try:
            wrapped = torch.ao.quantization.fuse_modules(wrapped, [], inplace=False)
        except Exception:
            pass        # không fuse được thì vẫn chạy tiếp, chỉ kém tối ưu hơn

        prepared = torch.ao.quantization.prepare(wrapped, inplace=False)

        # Hiệu chuẩn: chạy vài batch ảnh THẬT để đo dải giá trị activation.
        # Dùng ảnh ngẫu nhiên ở bước này là sai — dải giá trị sẽ không giống dữ
        # liệu thật và accuracy sau nén sẽ tệ hơn nhiều.
        log.info("  hiệu chuẩn trên %d batch ảnh thật...", n_batches)
        for i, (images, _) in enumerate(calibration_loader):
            if i >= n_batches:
                break
            prepared(images)

        converted = torch.ao.quantization.convert(prepared, inplace=False)
    except Exception as exc:
        log.warning("  Không lượng tử hoá tĩnh được: %s", str(exc).splitlines()[0][:110])
        return None

    # ★ Kiểm chạy được THẬT trước khi trả về
    return converted if _works(converted, img_size) else None


# =====================================================================
# Xuất sang định dạng chạy được ngoài Python
# =====================================================================

def export_torchscript(model: nn.Module, img_size: int, out_path: str | Path,
                       device: torch.device | None = None) -> Path | None:
    """Xuất TorchScript (.pt) — chạy được bằng LibTorch C++, không cần Python.

    Dùng `torch.jit.trace` chứ không phải `script`: trace đơn giản hơn và đủ cho
    CNN thuần (không có luồng điều khiển phụ thuộc dữ liệu).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cpu")
    model = model.to(device).eval()

    try:
        example = torch.randn(1, 3, img_size, img_size, device=device)
        with torch.no_grad():
            traced = torch.jit.trace(model, example)
        traced.save(str(out_path))
        log.info("  TorchScript -> %s (%.1f MB)",
                 out_path, out_path.stat().st_size / 1e6)
        return out_path
    except Exception as exc:
        log.warning("  Không xuất TorchScript được: %s", exc)
        return None


def export_onnx(model: nn.Module, img_size: int, out_path: str | Path,
                opset: int = 18, single_file: bool = True) -> Path | None:
    """Xuất ONNX — chạy được bằng ONNX Runtime, TensorRT, OpenVINO, CoreML...

    ★ BẪY TRIỂN KHAI ĐÃ GẶP THẬT ★
    Bộ xuất ONNX mới của PyTorch mặc định tách trọng số ra file riêng
    `<tên>.onnx.data` (external data format). File `.onnx` còn lại chỉ ~3 KB — chứa
    đồ thị, không chứa trọng số.

    Hậu quả: copy mỗi file `.onnx` sang thiết bị biên thì model KHÔNG CHẠY ĐƯỢC,
    và thông báo lỗi lúc nạp không hề nói là thiếu file. Tệ hơn, nếu chỉ nhìn dung
    lượng file `.onnx` thì tưởng model nhẹ 3 KB.

    `single_file=True` (mặc định) ép nhúng trọng số vào MỘT file duy nhất — an toàn
    hơn cho việc đem đi triển khai. Đặt False nếu model quá lớn (> 2 GB, giới hạn
    của protobuf) thì buộc phải tách.

    `dynamic_axes` cho phép đổi batch size lúc chạy mà không phải xuất lại.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()

    try:
        example = torch.randn(1, 3, img_size, img_size)
        kwargs = dict(
            input_names=["input"], output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=opset,
        )
        # external_data chỉ có ở bộ xuất mới; bản cũ không nhận tham số này.
        import inspect
        if "external_data" in inspect.signature(torch.onnx.export).parameters:
            kwargs["external_data"] = not single_file

        torch.onnx.export(model, example, str(out_path), **kwargs)

        # Báo dung lượng THẬT: gồm cả file .data nếu có
        sidecar = Path(str(out_path) + ".data")
        total = out_path.stat().st_size + (sidecar.stat().st_size
                                           if sidecar.exists() else 0)
        if sidecar.exists():
            log.warning("  ONNX -> %s + %s (tổng %.1f MB) — PHẢI copy CẢ HAI file "
                        "khi đem triển khai!", out_path.name, sidecar.name, total / 1e6)
        else:
            log.info("  ONNX -> %s (%.1f MB, trọng số nhúng trong một file)",
                     out_path, total / 1e6)
        return out_path
    except Exception as exc:
        log.warning("  Không xuất ONNX được: %s", exc)
        return None


def verify_onnx(onnx_path: str | Path, model: nn.Module, img_size: int,
                tolerance: float = 1e-4) -> bool:
    """Kiểm model ONNX cho ra ĐÚNG kết quả như model PyTorch gốc.

    ★ BƯỚC NÀY KHÔNG ĐƯỢC BỎ. Xuất thành công không có nghĩa là xuất đúng: một số
    phép toán bị dịch sai hoặc bị xấp xỉ, và sai lệch chỉ lộ ra khi so output thật.
    Triển khai một model ONNX chưa kiểm là đưa lỗi im lặng lên thiết bị.
    """
    try:
        import numpy as np
        import onnxruntime
    except ImportError:
        log.warning("  Chưa cài onnxruntime -> bỏ qua bước kiểm. pip install onnxruntime")
        return False

    try:
        x = torch.randn(2, 3, img_size, img_size)
        with torch.no_grad():
            expected = model.cpu().eval()(x).numpy()

        session = onnxruntime.InferenceSession(str(onnx_path),
                                               providers=["CPUExecutionProvider"])
        actual = session.run(None, {"input": x.numpy()})[0]

        diff = float(np.abs(expected - actual).max())
        ok = diff < tolerance
        log.info("  kiểm ONNX: sai lệch lớn nhất %.2e -> %s",
                 diff, "KHỚP" if ok else "LỆCH QUÁ NGƯỠNG")
        return ok
    except Exception as exc:
        log.warning("  Không kiểm được ONNX: %s", exc)
        return False


# =====================================================================
# Đo tác động của việc nén
# =====================================================================

def model_size_mb(model: nn.Module) -> float:
    """Dung lượng model khi lưu ra đĩa (MB).

    Phải lưu thật rồi đo, không ước lượng bằng số tham số × 4 byte: model đã lượng
    tử hoá có cả tham số int8 lẫn scale/zero_point fp32, công thức đơn giản sẽ sai.
    """
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as tmp:
        torch.save(model.state_dict(), tmp.name)
        return Path(tmp.name).stat().st_size / 1e6


def compare_quantized(model: nn.Module, test_loader, device: torch.device,
                      calibration_loader=None, img_size: int = 48) -> "pd.DataFrame":
    """So sánh fp32 với các bản đã nén: dung lượng, latency, VÀ accuracy.

    ★ Cột accuracy là bắt buộc. Báo "nhẹ hơn 4 lần" mà không nói accuracy giảm bao
    nhiêu là báo cáo thiếu trung thực — người đọc sẽ tưởng nén là bữa trưa miễn phí.
    """
    import pandas as pd

    from gtsrb.deploy.speed import benchmark
    from gtsrb.eval.metrics import evaluate

    cpu = torch.device("cpu")
    rows = []

    variants = [("fp32", model.cpu().eval())]

    dynamic = quantize_dynamic(model)
    variants.append(("int8_dynamic", dynamic))

    if calibration_loader is not None:
        static = quantize_static(model, calibration_loader, img_size=img_size)
        if static is not None:
            variants.append(("int8_static", static))

    for name, variant in variants:
        log.info("  đo biến thể %s...", name)
        stats = evaluate(variant, test_loader, cpu)
        timing = benchmark(variant, img_size, cpu, batch_sizes=(1,),
                           warmup=10, iters=50)
        rows.append({
            "biến thể": name,
            "dung lượng (MB)": round(model_size_mb(variant), 2),
            "top-1": round(stats["top1"], 5),
            "macro-F1": round(stats["macro_f1"], 5),
            "latency p50 (ms)": round(timing.get("cpu_bs1_p50", float("nan")), 2),
            "latency p95 (ms)": round(timing.get("cpu_bs1_p95", float("nan")), 2),
        })

    frame = pd.DataFrame(rows)

    # Thêm cột so với fp32 để người đọc thấy ngay cái giá phải trả
    if len(frame) > 1:
        base = frame.iloc[0]
        frame["nhẹ hơn"] = (base["dung lượng (MB)"] / frame["dung lượng (MB)"]).round(2)
        frame["nhanh hơn"] = (base["latency p50 (ms)"] /
                              frame["latency p50 (ms)"]).round(2)
        frame["mất top-1"] = ((base["top-1"] - frame["top-1"]) * 100).round(3)

    return frame
