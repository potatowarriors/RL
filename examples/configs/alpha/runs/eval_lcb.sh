#!/bin/bash
# LiveCodeBench v5 코드 벤치 (사용자 지시 2026-10-19 "코드벤치는 지금 바로 준비해"). Pai 벤치 스위트에 코드 벤치가 없어 RL 쪽에서 잰다.
# 데이터: alpha_blends/eval/lcb_v5_322.jsonl — Nemotron-RL competitive_coding validation 322문제 (LCB v5, 2024-07~2025-02, 공식 LCB 프롬프트·
#   공식 runner 테스트, 함수형 124 · 표준입력 198). 학습 풀은 이 322 와 대조해 오염 0 (teacher_pool build_summary). 채점은 학습과 같은 Gym code_gen.
# 측정 조건은 학습과 같다: temperature 1.0 · top_p 1.0, 생성 상한 32K (MAXOUT=49152 면 48K), 문제당 REPEATS 회 (기본 4).
# 순서: 다른 런 없음·GPU 비움 확인 → fleet 16 (main1 serve_fleet 8 + sub1 ray_serve_alpha 8, P0-3/P0-4 와 같은 플래그) → 체크포인트마다
#   run_measure.sh → summarize_lcb.py → 다음 체크포인트는 fleet 를 내렸다 다시 띄운다 → 정지.
# 사용: setsid nohup bash examples/configs/alpha/runs/eval_lcb.sh <tag>=<hf_dir> [<tag>=<hf_dir> ...] > $NRL_ROOT/p0_measure/eval_lcb.out 2>&1 &
#   예: lcb_sft2862=<hfmodel_0002862> lcb_s1_282=<…/teacher_codemath2/hf/hf_step_00282> lcb_s2_175=<…/teacher_codemath_s2/hf/hf_step_00175>
#   환경변수: REPEATS (기본 4) · MAXOUT (기본 32768; 49152 이면 48K 모델 설정과 max_len 53248)
# 결과: $NRL_ROOT/p0_measure/<tag>/{rollouts.jsonl,lcb_summary.json} · 진행 $NRL_ROOT/p0_measure/eval_lcb.log. 실패하면 ABORT 를 남기고 멈춘다.
set -u
NRL=${NRL_ROOT:-/home/work/vidsearch/tools/nemo_rl}
P=$NRL/p0_measure
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
HERE=$REPO/examples/configs/alpha/runs
PAI=/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha
DATA=/home/work/Datasets/LL_datasets/posttraining/RL/alpha_blends/eval/lcb_v5_322.jsonl
REPEATS=${REPEATS:-4}
MAXOUT=${MAXOUT:-32768}
if [ "$MAXOUT" = 49152 ]; then MODEL_CFG=$HERE/alpha_vllm_model_p0_48k.yaml; MAXLEN=53248; else MODEL_CFG=$HERE/alpha_vllm_model_p0_32k.yaml; MAXLEN=40960; fi
LOG=$P/eval_lcb.log
HOSTS="10.0.37.4 10.0.37.2"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
ray_py() { "$NRL/clean_run.sh" /usr/bin/env RAY_ADDRESS=10.0.37.4:6379 "$NRL/venv-driver/bin/python" -I "$@" </dev/null 2>/dev/null; }
abort() { log "ABORT: $*"; exit 1; }
[ $# -ge 1 ] || abort "usage: eval_lcb.sh <tag>=<hf_dir> ..."
log "armed: $* · repeats=$REPEATS maxout=$MAXOUT maxlen=$MAXLEN data=$(wc -l < $DATA) rows"

start_fleet() {
  local ck=$1
  ra=$(ray_py "$HERE/gpu_check.py" | awk '/NRL_RUN_ACTORS/{print $2}')
  [ "$ra" = "0" ] || abort "NRL_RUN_ACTORS=${ra:-?} — another NeMo-RL run is alive"
  gl=$(ray_py "$HERE/gym_cleanup.py" kill | awk '/GYM_LEFTOVER_TOTAL/{print $2}')
  busy=?
  for i in $(seq 1 18); do busy=$(ray_py "$HERE/gpu_check.py" | awk '/BUSY_GPUS/{print $2}'); [ "$busy" = "0" ] && break; sleep 10; done
  [ "$busy" = "0" ] || abort "BUSY_GPUS=$busy (Gym leftovers killed: ${gl:-?})"
  export TOOLS=1 TOOL_PARSER=qwen3_xml REASONING_PARSER=nemotron_v3 VLLM_ENFORCE_STRICT_TOOL_CALLING=0 PREFIX_CACHE=0
  GPUS=0,1,2,3,4,5,6,7 bash "$PAI/eval_sft/serve_fleet.sh" "$ck" "$MAXLEN" 8 8100 >> "$LOG" 2>&1
  ray_py "$P/ray_serve_alpha.py" start 10.0.37.2 "$ck" "$MAXLEN" >> "$LOG" 2>&1 || abort "ray_serve_alpha.py start failed"
  local ready=0
  for i in $(seq 1 120); do
    ready=0
    for h in $HOSTS; do for p in 8000 8001 8002 8003 8004 8005 8006 8007; do
      curl -s -m 3 "http://$h:$p/v1/models" 2>/dev/null | grep -q '"alpha"' && ready=$((ready+1))
    done; done
    [ "$ready" -eq 16 ] && break
    sleep 15
  done
  log "servers ready: $ready/16 ($ck)"
  [ "$ready" -eq 16 ] || abort "only $ready/16 vLLM servers ready after 30 min (fleets left running for diagnosis)"
}
stop_fleet() {
  bash "$PAI/eval_sft/stop_fleet.sh" 0,1,2,3,4,5,6,7 >> "$LOG" 2>&1
  ray_py "$P/ray_serve_alpha.py" stop 10.0.37.2 >> "$LOG" 2>&1
  sleep 30
}

for spec in "$@"; do
  tag=${spec%%=*}; ck=${spec#*=}
  [ -f "$ck/model.safetensors.index.json" ] || abort "HF ckpt missing for $tag: $ck"
  start_fleet "$ck"
  log "$tag: $(wc -l < $DATA) × $REPEATS @ max_tokens $MAXOUT"
  MODEL_CFG=$MODEL_CFG bash "$P/run_measure.sh" "$tag" "$DATA" "$REPEATS" 1024 >> "$LOG" 2>&1
  log "$tag run: $(tail -2 "$P/$tag/run.log" | tr '\n' ' ')"
  "$NRL/venv-driver/bin/python" -I "$REPO/examples/configs/alpha/tools/summarize_lcb.py" "$P/$tag" --cap "$MAXOUT" >> "$LOG" 2>&1
  log "$tag summary: $(cat "$P/$tag/lcb_summary.json" 2>/dev/null | tr -d '\n' | cut -c1-300)"
  stop_fleet
done
log "DONE"
