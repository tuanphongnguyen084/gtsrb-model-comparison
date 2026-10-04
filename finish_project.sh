#!/bin/zsh
# finish_project.sh — chạy nốt phần còn lại của dự án.
#
# ★ AN TOÀN KHI CHẠY LẠI (idempotent): mỗi bước tự kiểm tra đã xong chưa.
#   Job chết giữa chừng thì chỉ cần chạy lại đúng lệnh này, nó làm tiếp từ chỗ dở.
#
#   ./finish_project.sh
#
# Theo dõi:  tail -f logs/_main.log

set -u
cd "${0:A:h}"
PY=~/.venv/bin/python
mkdir -p logs

say() { echo "[$(date +%H:%M)] $*" }

# ---------------------------------------------------------------- 1. Robustness
# run_robustness.py tự bỏ qua model đã có trong CSV, nên gọi lại là an toàn.
n_done=$($PY -c "
import pandas as pd, pathlib, sys
p = pathlib.Path('reports/tables/robustness.csv')
print(len(pd.read_csv(p)['run_id'].unique()) if p.exists() else 0)
" 2>/dev/null || echo 0)
n_runs=$(ls -d artifacts/runs/*/ 2>/dev/null | wc -l | tr -d ' ')

if [ "$n_done" -ge "$n_runs" ] && [ "$n_runs" -gt 0 ]; then
  say "1/4 ROBUSTNESS — bỏ qua ($n_done/$n_runs model đã xong)"
else
  say "1/4 ROBUSTNESS — $n_done/$n_runs model đã xong, chạy tiếp"
  $PY -u scripts/run_robustness.py --runs "artifacts/runs/*" >> logs/robustness.log 2>&1
  say "    -> mã thoát $?"
fi

# ---------------------------------------------------------------- 2-3. Train M3
# Quy ước: có result.json = đã train xong. Có last.pt mà không có result.json
# = đang train dở -> dùng --resume thay vì train lại từ đầu.
train_if_needed() {
  local name=$1 config=$2
  local done_dir=$(ls -d artifacts/runs/${name}_* 2>/dev/null | while read d; do
      [ -f "$d/result.json" ] && echo "$d"; done | tail -1)
  if [ -n "$done_dir" ]; then
    say "    $name — bỏ qua (đã xong: $(basename $done_dir))"
    return
  fi
  local partial=$(ls -d artifacts/runs/${name}_* 2>/dev/null | while read d; do
      [ -f "$d/last.pt" ] && echo "$d"; done | tail -1)
  if [ -n "$partial" ]; then
    say "    $name — chạy TIẾP từ $(basename $partial)"
    $PY -u scripts/train.py --config "$config" --resume "$partial" >> logs/$name.log 2>&1
  else
    say "    $name — train mới"
    $PY -u scripts/train.py --config "$config" > logs/$name.log 2>&1
  fi
  say "    -> mã thoát $?"
}

say "2/4 TRAIN m3_mobilenetv2"
train_if_needed m3_mobilenetv2 configs/m3_mobilenetv2.yaml
say "3/4 TRAIN m3_effnetb0"
train_if_needed m3_effnetb0 configs/m3_effnetb0.yaml

# ---------------------------------------------------------------- 4. Tổng hợp
# Bốn bước này rẻ (vài phút) và cần chạy lại mỗi khi có model mới -> luôn chạy.
say "4/4 ĐÁNH GIÁ + TỐC ĐỘ + GRAD-CAM + BÁO CÁO"
$PY -u scripts/evaluate.py        --runs "artifacts/runs/*" > logs/evaluate.log 2>&1 && say "    evaluate OK"
$PY -u scripts/benchmark_speed.py --runs "artifacts/runs/*" > logs/speed.log    2>&1 && say "    speed OK"
$PY -u scripts/make_gradcam.py    --runs "artifacts/runs/*" > logs/gradcam.log  2>&1 && say "    gradcam OK"
$PY -u scripts/make_report.py                                > logs/report.log   2>&1 && say "    report OK"

say "HOÀN TẤT — xem docs/KET_QUA.md"
touch logs/.DONE
