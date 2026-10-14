#!/bin/bash
# P0-3 Code·Math alpha 사전 측정 오케스트레이터 (RL_DATA.md §5.5·§5.6, 사용자 승인 2026-10-13).
# 순서: main2r3 종료(rc 0) 대기 → 최종 step_820 HF 반출 (결정 28) → 남은 Gym 서버(judge 포함) 정리 → 양 노드 GPU 비움 확인
#   → fleet 16 기동 (main1 Pai serve_fleet.sh 8 + sub1 $NRL_ROOT/p0_measure/ray_serve_alpha.py 8, 10-08 과 같은 TOOLS=1·qwen3_xml·nemotron_v3)
#   → 수학 12,000 × 4회 @32K → 코드 12,000 × 4회 @32K (둘 다 runs/alpha_vllm_model_p0_32k.yaml) → fleet 정지.
# 측정 본체는 10-08 과 같은 $NRL_ROOT/p0_measure/run_measure.sh (gym env start → gym eval run --no-serve → gym eval profile).
# 32K 로 잰다 (사용자 2026-10-14: Code·Math 1차 학습 상한을 32K 로 — 24K 측정(p03_math, 20% 진행 뒤 중단)에서 수학 롤아웃 71% 가
# 24K 에 잘려 0/4 의 62% 가 "4회 모두 잘림" 이었다). 48K 참고치는 10-08 code48k 스모크 (끝냄 16~31% · 정답 6%).
# max_len 40960: 렌더 프롬프트 최대 코드 4,448 · 수학 1,484 토큰 + 생성 32,768.
# 실패하면 그 자리에서 멈추고 ABORT 를 남긴다 — 재시도·재기동은 하지 않는다 (사용자 판단).
# 실행: setsid nohup bash examples/configs/alpha/runs/p03_orchestrate.sh [case=main2r3] [final_step=820] > $NRL_ROOT/p0_measure/p03_orchestrate.out 2>&1 &
#       (세션과 무관하게 돈다; 진행은 $NRL_ROOT/p0_measure/p03_orchestrate.log). case 는 종료를 기다릴 STEM 런의 runs.log 태그,
#       final_step 은 반출할 마지막 체크포인트 번호 (결정 29: main2r4 는 735).
set -u
CASE=${1:-main2r3}
FINAL=${2:-820}
HF_FINAL=$(printf 'hf_step_%05d' "$FINAL")
NRL=${NRL_ROOT:-/home/work/vidsearch/tools/nemo_rl}
P=$NRL/p0_measure
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
HERE=$REPO/examples/configs/alpha/runs
PAI=/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha
CK=/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha/outputs/alpha_baseline_48L_sft_128k_agentic_resume_20261001_011900/hfmodel_0002862
POOL=/home/work/Datasets/LL_datasets/posttraining/RL/alpha_blends/teacher_pool
O=$REPO/results/alpha/teacher_stem
LOG=$P/p03_orchestrate.log
MAXLEN=40960
MODEL_CFG=$HERE/alpha_vllm_model_p0_32k.yaml
HOSTS="10.0.37.4 10.0.37.2"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
ray_py() { "$NRL/clean_run.sh" /usr/bin/env RAY_ADDRESS=10.0.37.4:6379 "$NRL/venv-driver/bin/python" -I "$@" </dev/null 2>/dev/null; }
abort() { log "ABORT: $*"; exit 1; }

log "armed: waiting for $CASE exit (runs.log case=$CASE rc=); final ckpt step_$FINAL → hf/$HF_FINAL"
until grep -q "case=$CASE rc=" "$O/runs.log"; do sleep 120; done
rc=$(grep "case=$CASE rc=" "$O/runs.log" | tail -1 | grep -oE 'rc=[0-9]+')
log "$CASE exited: $rc"
[ "$rc" = "rc=0" ] || abort "$CASE did not finish cleanly ($rc) — P0-3 not launched"
# rc 0 이어도 최종 체크포인트가 없으면 진짜 종료가 아니다 (2026-10-13 main2r5: 에폭 클램프로 기동 직후 rc 0 — KNOWN_ISSUES 10-13).
sleep 30
[ -d "$O/ckpt/step_$FINAL" ] || abort "$CASE exited rc=0 but ckpt/step_$FINAL is missing — not a real finish, P0-3 not launched"

# 1) 최종 step_$FINAL HF 반출 (결정 28). 변환은 CPU(gloo). 이미 있으면 건너뛴다.
if [ -d "$O/ckpt/step_$FINAL" ] && [ ! -e "$O/hf/$HF_FINAL" ]; then
    log "exporting step_$FINAL → hf/$HF_FINAL"
    bash "$REPO/examples/configs/alpha/tools/export_rl_hf.sh" "$O/ckpt/step_$FINAL" "$O/hf/$HF_FINAL" > "$O/hf/$HF_FINAL.log" 2>&1
    log "export rc=$? $(grep -a 'tensors=' "$O/hf/$HF_FINAL.log" | tail -1)"
else
    log "export skipped: step_$FINAL=$([ -d "$O/ckpt/step_$FINAL" ] && echo yes || echo no) existing=$([ -e "$O/hf/$HF_FINAL" ] && echo yes || echo no)"
fi

# 2) 남은 런 액터 0 확인 → Gym 잔여(judge 포함) 정리 → GPU 비움 (launch.sh 와 같은 순서)
ra=$(ray_py "$HERE/gpu_check.py" | awk '/NRL_RUN_ACTORS/{print $2}')
[ "$ra" = "0" ] || abort "NRL_RUN_ACTORS=${ra:-?} — another NeMo-RL run is alive"
gl=$(ray_py "$HERE/gym_cleanup.py" kill | awk '/GYM_LEFTOVER_TOTAL/{print $2}')
log "leftover Gym servers killed: ${gl:-?}"
busy=?
for i in $(seq 1 18); do
    busy=$(ray_py "$HERE/gpu_check.py" | awk '/BUSY_GPUS/{print $2}')
    [ "$busy" = "0" ] && break
    sleep 10
done
[ "$busy" = "0" ] || abort "BUSY_GPUS=$busy after Gym cleanup (3 min)"
log "both nodes idle (BUSY_GPUS=0)"

# 3) fleet 기동 — 10-08 과 같은 플래그 (TOOLS=1 qwen3_xml · nemotron_v3 · strict 끔 · prefix cache 끔 · gpu util 0.95)
export TOOLS=1 TOOL_PARSER=qwen3_xml REASONING_PARSER=nemotron_v3 VLLM_ENFORCE_STRICT_TOOL_CALLING=0 PREFIX_CACHE=0
log "starting main1 fleet: serve_fleet.sh $CK $MAXLEN 8 8100"
GPUS=0,1,2,3,4,5,6,7 bash "$PAI/eval_sft/serve_fleet.sh" "$CK" "$MAXLEN" 8 8100 >> "$LOG" 2>&1
log "starting sub1 servers via Ray: ray_serve_alpha.py start 10.0.37.2 $CK $MAXLEN"
ray_py "$P/ray_serve_alpha.py" start 10.0.37.2 "$CK" "$MAXLEN" >> "$LOG" 2>&1 || abort "ray_serve_alpha.py start failed"

ready=0
for i in $(seq 1 120); do   # 최대 30분
    ready=0
    for h in $HOSTS; do for p in 8000 8001 8002 8003 8004 8005 8006 8007; do
        curl -s -m 3 "http://$h:$p/v1/models" 2>/dev/null | grep -q '"alpha"' && ready=$((ready+1))
    done; done
    [ "$ready" -eq 16 ] && break
    sleep 15
done
log "servers ready: $ready/16"
[ "$ready" -eq 16 ] || abort "only $ready/16 vLLM servers ready after 30 min (fleets left running for diagnosis)"

# 4) 측정 — 수학(32K) 먼저, 코드(32K) 다음. run_measure.sh 가 Gym 서버 기동·수집·프로파일을 한다. 태그 *32 (24K 부분 결과 p03_math 는 보존).
log "P0-3 math: 12,000 × 4 @32K"
MODEL_CFG=$MODEL_CFG bash "$P/run_measure.sh" p03_math32 "$POOL/math_measure.jsonl" 4 1024 >> "$LOG" 2>&1
log "math done: $(tail -2 "$P/p03_math32/run.log" | tr '\n' ' ')"
log "P0-3 code: 12,000 × 4 @32K"
MODEL_CFG=$MODEL_CFG bash "$P/run_measure.sh" p03_code32 "$POOL/code_measure.jsonl" 4 1024 >> "$LOG" 2>&1
log "code done: $(tail -2 "$P/p03_code32/run.log" | tr '\n' ' ')"

# 5) fleet 정지 (Pai stop_fleet.sh: SIGTERM → 메모리 회수 확인; sub1 은 Ray 액터 stop)
bash "$PAI/eval_sft/stop_fleet.sh" 0,1,2,3,4,5,6,7 >> "$LOG" 2>&1
ray_py "$P/ray_serve_alpha.py" stop 10.0.37.2 >> "$LOG" 2>&1
log "DONE — results: $P/p03_math32/rollouts.jsonl · $P/p03_code32/rollouts.jsonl"
