#!/bin/zsh
# run_everything.sh — chạy nốt TOÀN BỘ phần thực nghiệm còn thiếu.
#
# ★ AN TOÀN KHI CHẠY LẠI: mỗi bước tự kiểm đã xong chưa. Máy ngủ hoặc job bị kill
#   thì chỉ cần chạy lại đúng lệnh này, nó làm tiếp từ chỗ dở.
#
#   ./run_everything.sh          Theo dõi:  tail -f logs/_all.log
#
# Thứ tự xếp theo giá trị trên mỗi giờ: việc nhanh và quan trọng chạy trước,
# để nếu có sự cố thì phần mất là phần ít giá trị nhất.

set -u
cd "${0:A:h}"
PY=~/.venv/bin/python
mkdir -p logs

# Chừa 2 lõi cho người dùng. Không đặt thì PyTorch chiếm hết 8 lõi và máy giật
# tới mức không gõ phím được. Ghi đè bằng: GTSRB_THREADS=8 ./run_everything.sh
export GTSRB_THREADS="${GTSRB_THREADS:-6}"
export OMP_NUM_THREADS="$GTSRB_THREADS"
export MKL_NUM_THREADS="$GTSRB_THREADS"

# Chạy ở mức ưu tiên thấp để nhường CPU cho công việc tương tác
renice +10 -p $$ > /dev/null 2>&1 || true

# ---- Khoá: chỉ cho phép MỘT bản chạy ----
# Hai bản chạy song song sẽ tranh GPU (không nhanh hơn, dễ OOM) và cùng ghi vào
# reports/tables/ -> hỏng kết quả. Khoá ghi PID để biết bản nào đang giữ.
LOCK=logs/.running.lock
if [ -f "$LOCK" ] && kill -0 "$(cat $LOCK)" 2>/dev/null; then
  echo "[$(date +%H:%M)] ĐÃ CÓ một bản đang chạy (PID $(cat $LOCK)). Thoát."
  exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT INT TERM
say() { echo "[$(date +%H:%M)] $*" }

# Đếm số run ablation đã xong, để biết còn bao nhiêu
n_abl() { ls artifacts/runs 2>/dev/null | grep -cE "resolution|augmentation|preprocess|scaling|label_smoothing|components" }
n_seed() { ls artifacts/runs 2>/dev/null | grep -c "seed4" }

# Chờ mọi job train/đánh giá khác xong trước, để không tranh GPU.
# Chạy hai job cùng lúc trên MPS/CUDA KHÔNG làm tổng thời gian giảm, mà còn dễ OOM.
# ★ Phải khớp tiến trình PYTHON thật, KHÔNG chỉ khớp chuỗi.
# Lỗi đã gặp: `pgrep -f "scripts/evaluate.py"` khớp luôn vào một shell khác có
# chứa chuỗi đó trong dòng lệnh -> vòng chờ không bao giờ thoát (deadlock tự
# tham chiếu). Thêm "Python" vào pattern để chỉ khớp tiến trình thông dịch.
while ps -eo command= \
      | grep -E "Python.*scripts/(train|evaluate|benchmark_speed|run_robustness)\.py" \
      | grep -qv grep; do
  sleep 15
done

say "================ BẮT ĐẦU ================"
say "Máy phải CẮM SẠC và MỞ NẮP. caffeinate chỉ chặn ngủ khi có sạc."

# ---------------------------------------------------------------- 1. Nén + xuất
if [ -f reports/tables/edge_export.csv ] && [ "$(grep -c , reports/tables/edge_export.csv)" -gt 10 ]; then
  say "1/5 NÉN + XUẤT ONNX — bỏ qua (đã có)"
else
  say "1/5 NÉN int8 + XUẤT TorchScript/ONNX cho 5 model  (~15 phút)"
  $PY -u scripts/export_edge.py --runs "artifacts/runs/m1*" "artifacts/runs/m2*" \
      "artifacts/runs/m3*" --max-samples 3000 > logs/edge.log 2>&1
  say "    -> mã thoát $?"
fi

# ---------------------------------------------------------------- 2. Rò rỉ
if [ -f reports/tables/leakage_experiment.csv ]; then
  say "2/5 THÍ NGHIỆM RÒ RỈ — bỏ qua (đã có)"
else
  say "2/5 THÍ NGHIỆM RÒ RỈ: train M1 hai lần, split sai vs split đúng  (~25 phút)"
  $PY -u scripts/run_leakage_experiment.py > logs/leakage.log 2>&1
  say "    -> mã thoát $?"
fi

# ---------------------------------------------------------------- 3. Ablation
if [ "${SKIP_HEAVY:-0}" = "1" ]; then
  say "3/5 ABLATION — BỎ QUA (SKIP_HEAVY=1, chạy trên Colab)"
else
say "3/5 ABLATION 30 run, chế độ ngân sách 15 epoch  (~5 giờ)"
say "    đã xong $(n_abl)/30 trước khi bắt đầu"
$PY -u scripts/run_ablation.py --axes all --budget >> logs/ablation.log 2>&1
say "    -> mã thoát $?,  giờ có $(n_abl)/30"
fi

# ---------------------------------------------------------------- 4. Nhiều seed
if [ "${SKIP_HEAVY:-0}" = "1" ]; then
  say "4/5 SEED — BỎ QUA (SKIP_HEAVY=1, chạy trên Colab)"
else
say "4/5 NHIỀU SEED cho M2  (~1,2 giờ)"
say "    đã có $(n_seed) run seed"
$PY -u scripts/run_seeds.py --config configs/m2_vggres.yaml --seeds 43 44 \
    >> logs/seeds.log 2>&1
say "    -> mã thoát $?,  giờ có $(n_seed) run seed"
fi

# ---------------------------------------------------------------- 5. Tổng hợp
say "5/5 ĐÁNH GIÁ TẤT CẢ + TỔNG HỢP  (~30 phút)"
$PY -u scripts/evaluate.py --runs "artifacts/runs/*" > logs/evaluate_all.log 2>&1 \
    && say "    evaluate OK"
$PY -u scripts/run_ablation.py --collect-only > logs/ablation_collect.log 2>&1 \
    && say "    bảng + hình ablation OK"
$PY -u scripts/run_seeds.py --collect-only > logs/seeds_collect.log 2>&1 \
    && say "    bảng seed OK"
$PY -u scripts/make_report.py > logs/report.log 2>&1 && say "    KET_QUA.md OK"
$PY -u scripts/make_baocao.py > logs/baocao.log 2>&1 && say "    BAO_CAO.md OK"

say "================ HOÀN TẤT ================"
touch logs/.ALL_DONE
