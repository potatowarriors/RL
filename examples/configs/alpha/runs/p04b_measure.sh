#!/bin/bash
# P0-4 난이도 지도 측정 — 1단계 결과 ckpt(teacher_codemath2 hf_step_00282)로 프롬프트별 pass@8 을 잰다 (사용자 승인 2026-10-18 14:40, RL_PLAN 결정 38).
# 순서: 양 노드 GPU 비움 확인 → fleet 16 기동 (p03_orchestrate.sh 와 같은 플래그) → A) 1단계 블렌드 9,028 × 8 @32K
#   → B) code 풀 나머지(블렌드 제외) × 8 → C) math 풀 나머지 × 8 → fleet 정지. 각 단계는 $NRL_ROOT/p0_measure/run_measure.sh (gym eval run).
# 실행: setsid nohup bash examples/configs/alpha/runs/p04_measure.sh > $NRL_ROOT/p0_measure/p04_measure.out 2>&1 &   진행: $NRL_ROOT/p0_measure/p04_measure.log
# 실패하면 그 자리에서 ABORT (fleet 는 진단용으로 남김). 뒤 단계를 생략하려면 touch $NRL_ROOT/p0_measure/.p04_stop (단계 경계에서 멈춘다).
set -u
NRL=${NRL_ROOT:-/home/work/vidsearch/tools/nemo_rl}
P=$NRL/p0_measure
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
HERE=$REPO/examples/configs/alpha/runs
PAI=/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha
CK=${CK:-$REPO/results/alpha/teacher_codemath2/hf/hf_step_00282}
POOL=/home/work/Datasets/LL_datasets/posttraining/RL/alpha_blends/teacher_pool
BLEND=/home/work/Datasets/LL_datasets/posttraining/RL/alpha_blends/teacher_blends/teacher_codemath_s1_v1.jsonl
REPEATS=${REPEATS:-8}
LOG=$P/p04b_measure.log
MAXLEN=40960
MODEL_CFG=$HERE/alpha_vllm_model_p0_32k.yaml
HOSTS="10.0.37.4 10.0.37.2"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
ray_py() { "$NRL/clean_run.sh" /usr/bin/env RAY_ADDRESS=10.0.37.4:6379 "$NRL/venv-driver/bin/python" -I "$@" </dev/null 2>/dev/null; }
abort() { log "ABORT: $*"; exit 1; }
[ -f "$CK/model.safetensors.index.json" ] || abort "HF ckpt missing: $CK"
log "armed: ckpt=$CK repeats=$REPEATS phases: A blend $(wc -l < $BLEND) · B code_rest $(wc -l < $POOL/p04_code_rest.jsonl) · C math_rest $(wc -l < $POOL/p04_math_rest.jsonl)"

# 1) 다른 런 없음 → Gym 잔여 정리 → 양 노드 GPU 비움
ra=$(ray_py "$HERE/gpu_check.py" | awk '/NRL_RUN_ACTORS/{print $2}')
[ "$ra" = "0" ] || abort "NRL_RUN_ACTORS=${ra:-?} — another NeMo-RL run is alive"
gl=$(ray_py "$HERE/gym_cleanup.py" kill | awk '/GYM_LEFTOVER_TOTAL/{print $2}')
log "leftover Gym servers killed: ${gl:-?}"
busy=?
for i in $(seq 1 18); do busy=$(ray_py "$HERE/gpu_check.py" | awk '/BUSY_GPUS/{print $2}'); [ "$busy" = "0" ] && break; sleep 10; done
[ "$busy" = "0" ] || abort "BUSY_GPUS=$busy after Gym cleanup (3 min)"
log "both nodes idle (BUSY_GPUS=0)"

# 2) fleet 기동 — p03 과 같은 플래그
export TOOLS=1 TOOL_PARSER=qwen3_xml REASONING_PARSER=nemotron_v3 VLLM_ENFORCE_STRICT_TOOL_CALLING=0 PREFIX_CACHE=0
log "starting main1 fleet: serve_fleet.sh $CK $MAXLEN 8 8100"
GPUS=0,1,2,3,4,5,6,7 bash "$PAI/eval_sft/serve_fleet.sh" "$CK" "$MAXLEN" 8 8100 >> "$LOG" 2>&1
log "starting sub1 servers via Ray: ray_serve_alpha.py start 10.0.37.2 $CK $MAXLEN"
ray_py "$P/ray_serve_alpha.py" start 10.0.37.2 "$CK" "$MAXLEN" >> "$LOG" 2>&1 || abort "ray_serve_alpha.py start failed"
ready=0
for i in $(seq 1 120); do
    ready=0
    for h in $HOSTS; do for p in 8000 8001 8002 8003 8004 8005 8006 8007; do
        curl -s -m 3 "http://$h:$p/v1/models" 2>/dev/null | grep -q '"alpha"' && ready=$((ready+1))
    done; done
    [ "$ready" -eq 16 ] && break
    sleep 15
done
log "servers ready: $ready/16"
[ "$ready" -eq 16 ] || abort "only $ready/16 vLLM servers ready after 30 min (fleets left running for diagnosis)"

# 3) 측정 단계 A → B → C (단계 경계에서 .p04_stop 확인)
run_phase() { tag=$1; input=$2
    [ -f "$P/.p04b_stop" ] && { log "stop file found — skipping $tag"; return 0; }
    log "phase $tag: $(wc -l < "$input") × $REPEATS @32K"
    MODEL_CFG=$MODEL_CFG bash "$P/run_measure.sh" "$tag" "$input" "$REPEATS" 1024 >> "$LOG" 2>&1
    log "$tag done: $(tail -2 "$P/$tag/run.log" | tr '\n' ' ')"
}
# (phase A done 2026-10-18 20:30)
run_phase p04_code8 "$POOL/p04_code_rest.jsonl"
# (phase C run by p04_measure.sh)

# 4) fleet 정지
bash "$PAI/eval_sft/stop_fleet.sh" 0,1,2,3,4,5,6,7 >> "$LOG" 2>&1
ray_py "$P/ray_serve_alpha.py" stop 10.0.37.2 >> "$LOG" 2>&1
log "DONE — results: $P/p04_blend8/rollouts.jsonl · $P/p04_code8/rollouts.jsonl · $P/p04_math8/rollouts.jsonl"
