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
CON=$((30-XONG))
(( XONG > 0 )) && printf "   [%s%s]" \
  "$(printf '#%.0s' $(seq 1 $XONG))" \
  "$( (( CON > 0 )) && printf '.%.0s' $(seq 1 $CON) )"
printf "\n  SEED      %2d/2 run xong\n" "$SEED"
[[ -n "$DANG" ]] && printf "\n  đang chạy  %s\n             %s\n" "$DANG" "$EP"

# "HOÀN TẤT" phải dựa trên ĐẾM ĐƯỢC BAO NHIÊU RUN, không dựa vào file đánh dấu.
#
# LỖI ĐÃ GẶP: mẻ đêm 5/10 báo mã thoát 0 và tạo .ABLATION_DONE, nhưng chỉ có
# 26/30 run (already_done() thiếu tên model nên bỏ qua 4 job). Script này nhìn
# file đánh dấu nên vẫn in "HOÀN TẤT" ở 26/30 — đúng cái lỗi mà nó phải phát
# hiện. Cùng họ với progress.py báo "còn 6h42m" cạnh dòng "HOÀN TẤT".
if (( XONG >= 30 && SEED >= 2 )); then
  printf "\n  *** ĐỦ 30/30 + 2 seed — nói với Claude để sinh lại báo cáo ***\n\n"
elif pgrep -qf "scripts/run_ablation.py|scripts/run_seeds.py"; then
  printf "  tình trạng  đang chạy bình thường\n\n"
else
  printf "\n  !!! MẺ ĐÃ DỪNG mà CHƯA ĐỦ (%d/30 ablation, %d/2 seed)\n" "$XONG" "$SEED"
  if [[ -f logs/.ABLATION_DONE ]]; then
    printf "      Có file .ABLATION_DONE nhưng số run KHÔNG đủ -> có job bị bỏ qua.\n"
  fi
  printf "      Chạy lại:  ./run_ablation_local.sh\n\n"
fi
