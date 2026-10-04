# GTSRB 43 lớp — lệnh tắt cho cả nhóm.  Dùng: make help
#
# PY trỏ tới python của môi trường ảo dùng chung ~/.venv (xem make setup-mac).
# Ghi đè nếu bạn dùng venv khác:  make train-m1 PY=./.venv/bin/python

PY := $(HOME)/.venv/bin/python
PIP := $(HOME)/.venv/bin/pip
RUNS := "artifacts/runs/*"

.PHONY: help
help:  ## Hiện danh sách lệnh
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- môi trường
.PHONY: setup-mac
setup-mac:  ## Cài môi trường trên macOS (Apple Silicon, dùng MPS)
	python3 -m venv $(HOME)/.venv
	$(PIP) install -U pip
	$(PIP) install torch torchvision
	$(PIP) install -r requirements.txt
	$(PY) -m ipykernel install --user --name=gtsrb --display-name "Python (GTSRB)"
	$(PY) -c "import torch; print('torch', torch.__version__, '| MPS:', torch.backends.mps.is_available())"

.PHONY: setup-colab
setup-colab:  ## Cài trên Google Colab — CHỈ gói còn thiếu, KHÔNG cài đè torch
	pip install -q -r requirements-colab.txt
	python -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"

.PHONY: zip
zip:  ## Đóng gói CODE (không gồm dữ liệu) để upload lên Colab
	@rm -f gtsrb_code.zip
	@zip -qr gtsrb_code.zip \
	    src scripts configs tests notebooks docs \
	    Makefile requirements.txt requirements-colab.txt pytest.ini README.md \
	    -x "*__pycache__*" "*.pyc" "*.ipynb_checkpoints*"
	@echo "Đã tạo gtsrb_code.zip ($$(du -h gtsrb_code.zip | cut -f1))"
	@echo "Upload file này ở Ô 3 của notebooks/00_colab_train.ipynb"

# ---------------------------------------------------------------- dữ liệu [A]
.PHONY: data
data:  ## Tải + tiền xử lý + split THEO TRACK  [A]
	$(PY) scripts/prepare_data.py --download --preprocess clahe --img-size 48

.PHONY: data-leaky
data-leaky:  ## ĐỐI CHỨNG: split random (CÓ rò rỉ) — chỉ để đo phần "ảo"  [A]
	$(PY) scripts/prepare_data.py --no-group-by-track --skip-cache

.PHONY: test
test:  ## pytest — PHẢI xanh trước khi push (gồm cổng chặn rò rỉ)
	$(PY) -m pytest tests/ -m "not slow" -v

.PHONY: test-all
test-all:  ## pytest kể cả test chậm (tải trọng số pretrained)
	$(PY) -m pytest tests/ -v

# ---------------------------------------------------------------- huấn luyện
.PHONY: smoke
smoke:  ## Chạy thử 2 epoch / 2000 mẫu để kiểm đường ống không vỡ
	$(PY) scripts/train.py --config configs/smoke.yaml --subset 2000 --tag smoke

.PHONY: train-m1 train-m2 train-m3 train-all
train-m1:  ## Train M1 LeNet (mốc tham chiếu)       [B]  ~12 phút trên M1 Pro
	$(PY) scripts/train.py --config configs/m1_lenet.yaml
train-m2:  ## Train M2 VGG-like + residual           [B]  ~35 phút trên M1 Pro
	$(PY) scripts/train.py --config configs/m2_vggres.yaml
train-m3:  ## Train M3 cả 3 backbone pretrained      [C]  nên chạy trên Colab
	$(PY) scripts/train.py --config configs/m3_resnet18.yaml
	$(PY) scripts/train.py --config configs/m3_mobilenetv2.yaml
	$(PY) scripts/train.py --config configs/m3_effnetb0.yaml
train-all: train-m1 train-m2 train-m3  ## Train cả 3 model

# ---------------------------------------------------------------- đánh giá [D]
.PHONY: eval
eval:  ## Bảng so sánh chính + per-class + confusion + McNemar + ECE  [D]
	$(PY) scripts/evaluate.py --runs $(RUNS) --out reports/tables/main_comparison.csv

.PHONY: ablation ablation-dry ablation-collect
ablation-dry:  ## Xem 30 thực nghiệm ablation sẽ chạy, chưa chạy thật
	$(PY) scripts/run_ablation.py --axes all --dry-run
ablation-budget:  ## XẾP HẠNG 30 biến thể ở 15 epoch (~5h thay vì 16h) — DÙNG CÁI NÀY TRƯỚC
	$(PY) scripts/run_ablation.py --axes all --budget
ablation:  ## Chạy TOÀN BỘ ablation ở độ dài đầy đủ (6 trục, 30 run) — rất lâu
	$(PY) scripts/run_ablation.py --axes all
ablation-collect:  ## Tổng hợp lại bảng ablation từ các run đã có
	$(PY) scripts/run_ablation.py --collect-only

.PHONY: robustness
robustness:  ## 5 nhiễu x 5 mức x N model + relative robustness + mCE  [D]
	$(PY) scripts/run_robustness.py --runs $(RUNS)

# ---------------------------------------------------------------- C
.PHONY: speed gradcam
speed:  ## Latency p50/p95 + FLOPs + biểu đồ Pareto  [C]
	$(PY) scripts/benchmark_speed.py --runs $(RUNS)
gradcam:  ## Lưới Grad-CAM: đúng / SAI / trước-sau nhiễu  [C]

.PHONY: report
report:  ## Tự sinh docs/KET_QUA.md từ mọi result.json  [D]
	$(PY) scripts/make_report.py
	$(PY) scripts/make_gradcam.py --runs $(RUNS)

# ---------------------------------------------------------------- tất cả
.PHONY: all
all: data test train-all eval robustness speed gradcam  ## Toàn bộ đường ống

.PHONY: clean-runs
clean-runs:  ## Xoá mọi checkpoint và kết quả run (KHÔNG xoá dữ liệu)
	rm -rf artifacts/runs/*
