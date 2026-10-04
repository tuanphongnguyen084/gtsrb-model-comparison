# Outline thuyết trình  (chủ: Phong Nguyễn, mỗi người trình bày phần mình)

Giả định 20 phút trình bày + 10 phút hỏi đáp. Nếu ngắn hơn, cắt mục 5 và 7.

| # | Slide | Phút | Người | Nội dung cốt lõi |
|---|---|---|---|---|
| 1 | Bài toán & dữ liệu | 1,5 | Huy | 43 lớp, 39.209+12.630 ảnh. **Mở bài bằng việc GTSRB đã bão hoà**: người 98,84%, SOTA 99,46%→99,7%. Nên câu hỏi của nhóm không phải "accuracy bao nhiêu" mà là 4 câu khác |
| 2 | ★ Rò rỉ dữ liệu do track | 2,5 | Huy | 30 frame/biển báo vật lý. Hình minh hoạ 2 frame gần giống nhau. **Bảng đối chứng random vs track**. Đây là slide gây ấn tượng nhất — để sớm |
| 3 | Tiền xử lý & augmentation | 2 | Huy | CLAHE vs HE (hình so sánh). **Vì sao cấm flip** — hình lớp 33 lật ngang thành lớp 34, rất thuyết phục bằng hình |
| 4 | M1 LeNet — mốc tham chiếu | 1,5 | Hoàng | Kiến trúc + "96% tham số nằm ở một lớp FC" → dẫn vào GAP ở slide sau |
| 5 | M2 VGG-like — 5 lựa chọn thiết kế | 3 | Hoàng | conv 3×3 ×2 vs 5×5 (18C² vs 25C²); BN; residual (`∂y/∂x = ∂F/∂x + 1`); SpatialDropout vs Dropout; GAP. **Kèm bảng ablation nội bộ** — đây là bằng chứng, không phải lời nói |
| 6 | M3 Transfer learning | 3 | Phong Trần | Phân tầng đặc trưng; vì sao upsample 224 (downsample 32×); discriminative LR; 2 pha freeze→unfreeze. Bảng 3 backbone |
| 7 | Kết quả & kiểm định | 3 | Phong Nguyễn | Bảng so sánh chính. **McNemar p-value** — "chênh lệch này có thật không". Per-class + confusion matrix: lớp nào khó và vì sao |
| 8 | Ablation | 2 | Phong Nguyễn | 5 trục, mỗi trục 1 hình + 1 câu kết luận định lượng |
| 9 | Robustness | 2 | Phong Nguyễn | Đường cong theo severity. **Relative robustness** — model giỏi nhất ≠ model bền nhất |
| 10 | Grad-CAM | 1,5 | Phong Trần | Lưới 3 model. **Nhấn vào ca SAI** — nó thú vị hơn ca đúng. Nêu giới hạn 3×3 |
| 11 | Tốc độ & Pareto | 1,5 | Phong Trần | p50/p95. **FLOPs ≠ latency** (MobileNet). Khuyến nghị triển khai |
| 12 | Kết luận & hướng mở rộng | 1 | Phong Nguyễn | 4 câu hỏi ban đầu → 4 câu trả lời. Hạn chế: đây là phân loại không phải detection, phân phối test giống train. Hướng tiếp: ensemble/TTA, adversarial, quantization, kiểm tra khái quát sang bộ khác |

## Nguyên tắc làm slide

1. **Mỗi slide một thông điệp.** Nếu phải nói "và thêm nữa" thì tách slide.
2. **Hình thắng bảng, bảng thắng chữ.** Slide 2 và 3 nên gần như toàn hình.
3. **Mọi số phải có nguồn.** Chân slide ghi `run_id` hoặc tên file bảng.
4. **Không đọc slide.** Slide là chỗ dựa cho khán giả, không phải cho người nói.
5. **Slide 2 là vũ khí.** Hầu hết bài GTSRB không xử lý rò rỉ track. Đặt nó sớm
   để mọi số liệu phía sau được tin.
6. **Chủ động nêu hạn chế** ở slide 12 trước khi bị hỏi. Tự nêu được hạn chế là
   dấu hiệu hiểu bài sâu nhất.
