"""
gradcam — trực quan hoá vùng ảnh quyết định dự đoán.

CHỦ: Phong Trần

★ CÔNG THỨC (Selvaraju et al. 2017) ★

    alpha_k^c = (1/Z) * sum_i sum_j  d y^c / d A^k_ij      <- trọng số của kênh k
    L^c       = ReLU( sum_k alpha_k^c * A^k )               <- bản đồ nhiệt

TRỰC GIÁC TỪNG BƯỚC:
  1. d y^c / d A^k  nói "nếu kênh k mạnh lên thì điểm số lớp c tăng hay giảm".
  2. Gộp theo không gian (1/Z * sum sum) cho MỘT con số cho mỗi kênh:
     "kênh này quan trọng cỡ nào cho lớp c".
  3. Tổ hợp các kênh theo trọng số đó.
  4. ReLU để giữ CHỈ phần HỖ TRỢ lớp c. Phần âm là bằng chứng CHỐNG LẠI lớp c,
     không phải thứ ta muốn hiển thị.

VÌ SAO DÙNG FEATURE MAP CUỐI: nó có ngữ nghĩa cao nhất MÀ VẪN CÒN giữ thông tin
VỊ TRÍ (các lớp fully-connected sau đó mới xoá hết vị trí). Đó là lớp cuối cùng
còn trả lời được câu hỏi "ở đâu".

★ GIỚI HẠN — PHẢI NÓI RA KHI TRÌNH BÀY ★
  1. ĐỘ PHÂN GIẢI RẤT THÔ. M2 ở input 48x48 cho feature map cuối 3x3; upsample
     lên 48x48 thì mỗi "ô" heatmap tương ứng 16x16 pixel. Nói "heatmap đúng vào
     biển báo" ở độ phân giải đó là một phát biểu rất lỏng.
     M3 ở input 224 cho 7x7 -> nét hơn hẳn, nhưng đó là lợi thế về ĐỘ PHÂN GIẢI,
     không phải vì model "hiểu" hơn.
  2. Nó chỉ cho biết Ở ĐÂU, không cho biết ĐẶC TRƯNG GÌ ở đó.
  3. Nó KHÔNG LÀ BẰNG CHỨNG NHÂN QUẢ: model có thể nhìn đúng chỗ mà vẫn suy luận sai.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Tính Grad-CAM cho một model + một target layer.

    Cách dùng:
        cam = GradCAM(model)                 # tự lấy model.gradcam_target_layer
        heatmap = cam(x)                     # x: (1,C,H,W) -> (H,W) trong [0,1]
        overlay = cam.overlay(img_uint8, heatmap)
        cam.close()                          # BẮT BUỘC: tháo hook khi dùng xong
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        self.model = model
        if target_layer is None:
            if not hasattr(model, "gradcam_target_layer"):
                raise AttributeError(
                    "Model không có .gradcam_target_layer. Xem hợp đồng ở "
                    "docs/INTERFACE.md mục 2 — mọi model phải expose thuộc tính này."
                )
            target_layer = model.gradcam_target_layer
        self.target_layer = target_layer

        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        # ★ CHỈ dùng forward hook, rồi gắn hook lên CHÍNH TENSOR output.
        #
        # VÌ SAO KHÔNG dùng register_full_backward_hook (cách nhiều tutorial dạy):
        # nó bọc output của layer thành một "view" của BackwardHookFunction. Nếu lớp
        # ngay sau đó sửa tại chỗ — ví dụ nn.ReLU(inplace=True), mà cả M1 và M2 đều
        # dùng để tiết kiệm bộ nhớ — thì PyTorch báo lỗi:
        #
        #   RuntimeError: Output 0 of BackwardHookFunction is a view and is being
        #   modified inplace.
        #
        # Đây là lỗi ĐÃ GẶP THẬT trong dự án này (xem docs/SU_CO.md).
        # Cách sửa: Tensor.register_hook() gắn trực tiếp lên tensor trong graph,
        # không bọc view nên không xung đột với in-place op.
        self._handle_forward = target_layer.register_forward_hook(self._save_activation)

    # ---- hooks ----
    def _save_activation(self, module, inputs, output) -> None:
        """Bắt activation A, và gắn hook lấy gradient d y / d A."""
        # .clone() để lớp in-place phía sau không ghi đè giá trị ta vừa bắt được.
        self._activations = output.clone().detach()

        if output.requires_grad:
            # Hook này gắn trên tensor nên tự mất khi tensor bị giải phóng —
            # không cần (và không thể) remove thủ công.
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad: torch.Tensor) -> None:
        """Hook trên tensor: bắt dy/dA khi backward chạy qua target layer."""
        self._gradients = grad.detach()

    def close(self) -> None:
        """Tháo forward hook. Không gọi thì hook còn sống và làm chậm mọi forward sau."""
        self._handle_forward.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- tính CAM ----
    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """x: (1, C, H, W). class_idx=None -> dùng lớp được model dự đoán.

        Trả heatmap (H, W) float trong [0,1], đã upsample về kích thước đầu vào.

        Bốn bước, mỗi bước một hàm con bên dưới:
          1. _forward_backward  chạy thuận rồi nghịch để hook bắt A và dA
          2. _combine_channels  alpha_k = GAP(gradient); L = ReLU(sum alpha_k * A^k)
          3. _upsample          đưa L về kích thước ảnh đầu vào
          4. _normalize         đưa về [0,1] để vẽ
        """
        if x.dim() != 4 or x.shape[0] != 1:
            raise ValueError(f"x phải có shape (1,C,H,W), nhận được {tuple(x.shape)}")

        was_training = self.model.training
        self.model.eval()      # tắt dropout để heatmap lặp lại được
        try:
            self._forward_backward(x, class_idx)
            cam = self._combine_channels()
            cam = self._upsample(cam, x.shape[-2:])
            return self._normalize(cam)
        finally:
            if was_training:
                self.model.train()

    def _forward_backward(self, x: torch.Tensor, class_idx: int | None) -> None:
        """Chạy thuận rồi nghịch từ điểm số của lớp c, để hai hook bắt được A và dy/dA.

        KHÔNG dùng torch.no_grad() ở đây — ta CẦN gradient, đó là cả cơ chế của Grad-CAM.
        """
        logits = self.model(x)                       # (1, K)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad(set_to_none=True)
        score = logits[0, class_idx]                 # y^c — một số vô hướng
        score.backward(retain_graph=False)

        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "Hook không bắt được activation/gradient. Có thể target_layer "
                "không nằm trên đường forward của model."
            )

    def _combine_channels(self) -> torch.Tensor:
        """alpha_k = trung bình không gian của gradient; L = ReLU(sum_k alpha_k * A^k)."""
        activations = self._activations              # (1, K_ch, h, w)
        gradients = self._gradients                  # (1, K_ch, h, w)

        # Bước 1-2 của công thức: gộp gradient theo không gian -> (1, K_ch, 1, 1)
        # "kênh k quan trọng cỡ nào cho lớp c"
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Bước 3: tổ hợp có trọng số các kênh -> (1, 1, h, w)
        cam = (weights * activations).sum(dim=1, keepdim=True)

        # Bước 4: ReLU — giữ CHỈ phần hỗ trợ lớp c.
        # Phần âm là bằng chứng CHỐNG LẠI lớp c, không phải thứ ta muốn hiển thị.
        return F.relu(cam)

    @staticmethod
    def _upsample(cam: torch.Tensor, size) -> np.ndarray:
        """Đưa feature map (h, w) về đúng kích thước ảnh đầu vào (H, W)."""
        cam = F.interpolate(cam, size=size, mode="bilinear", align_corners=False)
        return cam[0, 0].cpu().numpy()

    @staticmethod
    def _normalize(cam: np.ndarray) -> np.ndarray:
        """Đưa về [0,1] để vẽ.

        KHÔNG dùng (cam - min) / (max - min + 1e-8): khi dải giá trị của cam rất nhỏ
        (ví dụ 1e-13, xảy ra với model chưa train hoặc khi activation gần 0), hằng số
        1e-8 LẤN ÁT mẫu số và nghiền heatmap về 0 — trông như "không có vùng nào quan
        trọng", trong khi thật ra chỉ là lỗi số học. Đây là lỗi ĐÃ GẶP THẬT, xem
        docs/SU_CO.md mục 3.
        """
        cam_min, cam_max = float(cam.min()), float(cam.max())
        span = cam_max - cam_min
        if span > 0.0:
            return (cam - cam_min) / span

        # cam phẳng thật: mọi alpha_k * A^k <= 0, tức layer này không có vùng nào
        # HỖ TRỢ lớp c. Báo rõ thay vì trả về mảng 0 một cách im lặng.
        import warnings
        warnings.warn(
            f"Grad-CAM suy biến: heatmap phẳng (giá trị {cam_min:.3e}). "
            f"Nguyên nhân thường gặp: (a) model chưa được train, "
            f"(b) target layer nằm ngoài đường forward, "
            f"(c) activation của layer gần 0. Kiểm tra lại target layer.",
            RuntimeWarning, stacklevel=3,
        )
        return np.zeros_like(cam)

    @staticmethod
    def overlay(image_uint8: np.ndarray, heatmap: np.ndarray,
                alpha: float = 0.45) -> np.ndarray:
        """Phủ heatmap lên ảnh gốc bằng colormap JET.

        image_uint8: HWC uint8, heatmap: HW float [0,1]  ->  HWC uint8
        """
        import cv2

        if heatmap.shape[:2] != image_uint8.shape[:2]:
            heatmap = cv2.resize(heatmap, (image_uint8.shape[1], image_uint8.shape[0]))

        colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)   # JET ra BGR, đổi về RGB
        blended = (1 - alpha) * image_uint8.astype(np.float32) + \
            alpha * colored.astype(np.float32)
        return np.clip(blended, 0, 255).astype(np.uint8)


def feature_map_size(model: nn.Module, img_size: int,
                     device: torch.device | None = None) -> tuple[int, int]:
    """Kích thước feature map ở target layer — để BÁO CÁO độ phân giải heatmap.

    Đây là con số phải nói ra khi trình bày Grad-CAM: 3x3 (M2 @48) so với
    7x7 (M3 @224) là khác biệt rất lớn về ý nghĩa của bản đồ nhiệt.
    """
    device = device or next(model.parameters()).device
    captured: dict[str, torch.Size] = {}

    def hook(module, inputs, output) -> None:
        """Bắt shape của feature map ở target layer rồi thôi."""
        captured["shape"] = output.shape

    handle = model.gradcam_target_layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model.eval()
            model(torch.zeros(1, 3, img_size, img_size, device=device))
    finally:
        handle.remove()

    shape = captured["shape"]
    return int(shape[-2]), int(shape[-1])
