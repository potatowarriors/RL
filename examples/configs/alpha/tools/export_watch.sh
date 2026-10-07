#!/usr/bin/env bash
# 본 런 체크포인트를 주기적으로 Pai 호환 HF 로 반출하는 감시 루프 (export_rl_hf.sh 를 감싼다, 2026-10-07).
# keep_top_k 가 지우기 전에 반출해야 한다. 체크포인트 하나는 save_period × keep_top_k 스텝 동안 남는다 (반출은 CPU 로 ≈ 3.5분).
# 완료 판정: NeMo-RL 은 tmp_step_N 에 저장한 뒤 step_N 으로 이름을 바꾼다 (`nemo_rl/utils/checkpoint.py`). step_N 이 보이면 저장이 끝난 것이다.
#
# 사용: export_watch.sh <checkpoint_dir> <out_root> <every_n_steps> [poll_sec=120]
#   <out_root>/hf_step_NNNNN 에 반출한다. 실패하면 hf_step_NNNNN.failed 를 남기고 그 스텝은 다시 시도하지 않는다 (로그 hf_step_NNNNN.log).
set -uo pipefail
CK=${1:?checkpoint dir}; OUT=${2:?out root}; EVERY=${3:?every n steps}; POLL=${4:-120}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
while true; do
    for d in $(ls -d "$CK"/step_* 2>/dev/null | sort -t_ -k2 -n); do
        n=${d##*_}
        [[ "$n" =~ ^[0-9]+$ ]] || continue
        (( n % EVERY == 0 )) || continue
        tgt=$(printf "%s/hf_step_%05d" "$OUT" "$n")
        if [ -e "$tgt" ] || [ -e "$tgt.failed" ]; then continue; fi
        echo "[export_watch $(date '+%F %T')] step $n → $tgt"
        if bash "$HERE/export_rl_hf.sh" "$d" "$tgt" > "$tgt.log" 2>&1; then
            echo "[export_watch $(date '+%F %T')]   ok: $(grep -a 'tensors=' "$tgt.log" | tail -1)"
        else
            echo "[export_watch $(date '+%F %T')]   FAILED → $tgt.log"
            touch "$tgt.failed"
        fi
    done
    sleep "$POLL"
done
