#!/bin/zsh
# tien_do.sh — xem mẻ ablation local đang tới đâu. Chạy: ./tien_do.sh
cd "${0:A:h}"
# zsh mặc định BÁO LỖI khi glob không khớp gì — tắt đi, vì lúc
# chưa có run seed nào thì artifacts/runs/*seed4*/ đúng là không khớp.
setopt null_glob
XONG=$(ls artifacts/runs 2>/dev/null | grep -c "budget" )
XONG=$(for d in artifacts/runs/*budget*/; do [[ -f "$d/result.json" ]] && echo x; done | wc -l | tr -d ' ')
SEED=$(for d in artifacts/runs/*seed4*/; do [[ -f "$d/result.json" ]] && echo x; done | wc -l | tr -d ' ')
DANG=$(tail -40 logs/ablation_local.log 2>/dev/null | grep "| RUN " | tail -1 | sed 's/.*RUN //')
EP=$(tail -5 logs/ablation_local.log 2>/dev/null | grep -oE "epoch +[0-9]+/[0-9]+" | tail -1)

printf "\n  ABLATION  %2d/30 run xong" "$XONG"
(( XONG > 0 )) && printf "   [%s%s]" "$(printf '#%.0s' $(seq 1 $XONG))" "$(printf '.%.0s' $(seq 1 $((30-XONG))))"
printf "\n  SEED      %2d/2 run xong\n" "$SEED"
[[ -n "$DANG" ]] && printf "\n  đang chạy  %s\n             %s\n" "$DANG" "$EP"

if [[ -f logs/.ABLATION_DONE ]]; then
  printf "\n  *** HOÀN TẤT — nói với Claude để sinh lại báo cáo ***\n\n"
elif pgrep -qf "scripts/run_ablation.py|scripts/run_seeds.py"; then
  printf "  tình trạng  đang chạy bình thường\n\n"
else
  printf "\n  !!! MẺ ĐÃ DỪNG mà chưa xong — chạy lại: ./run_ablation_local.sh\n\n"
fi
