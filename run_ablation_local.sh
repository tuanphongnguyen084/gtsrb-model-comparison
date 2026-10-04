#!/bin/zsh
# run_ablation_local.sh — chạy 30 run ablation + 2 seed ở MÁY LOCAL.
#
# VÌ SAO KHÔNG DÙNG COLAB: 27/30 job là M1/M2 ở 48px — máy này chạy 9 phút/job.
# Chỉ 3 job là M3. Tổng ~6 giờ, không giới hạn phiên, không cần ai trông, và
# chạy lại được từ chỗ dở. Hai phiên Colab trước mất 200 phút mà không ra job
# ablation nào.
#
# AN TOÀN KHI CHẠY LẠI: run đã có result.json sẽ được bỏ qua.
#   ./run_ablation_local.sh          Theo dõi: tail -f logs/ablation_local.log
set -u
cd "${0:A:h}"
PY=~/.venv/bin/python
LOG=logs/ablation_local.log

# Khoá PID — chạy hai lần cùng lúc sẽ tranh cache và làm hỏng index.csv
# (lỗi đã gặp thật: hai mẻ cùng vào bước thí nghiệm rò rỉ).
KHOA=logs/.ablation_local.pid
if [[ -f $KHOA ]] && kill -0 "$(cat $KHOA)" 2>/dev/null; then
  echo "Đang có mẻ chạy (PID $(cat $KHOA)). Thoát."; exit 1
fi
echo $$ > $KHOA
trap 'rm -f $KHOA' EXIT

noi() { echo "[$(date +%H:%M)] $*" | tee -a $LOG; }

noi "================ BẮT ĐẦU ablation local ================"
noi "Máy phải CẮM SẠC và MỞ NẮP. caffeinate chỉ chặn ngủ khi có sạc."
noi "Ước tính ~6 giờ. Xem tiến độ: tail -f $LOG"

# renice +10: ưu tiên thấp để máy vẫn dùng được bình thường
noi "1/2 ABLATION 30 run (15 epoch mỗi run, chế độ --budget)"
caffeinate -s nice -n 10 $PY -u scripts/run_ablation.py \
    --axes all --budget >> $LOG 2>&1
noi "    -> mã thoát $?"

noi "2/2 SEED 43 và 44 cho M2"
caffeinate -s nice -n 10 $PY -u scripts/run_seeds.py \
    --config configs/m2_vggres.yaml --seeds 43 44 >> $LOG 2>&1
noi "    -> mã thoát $?"

noi "Tổng hợp bảng"
$PY -u scripts/run_ablation.py --collect-only >> $LOG 2>&1 && noi "    bảng ablation OK"
$PY -u scripts/run_seeds.py    --collect-only >> $LOG 2>&1 && noi "    bảng seed OK"

touch logs/.ABLATION_DONE
noi "================ HOÀN TẤT ================"
