#!/bin/bash
# SWE 학생 롤아웃 측정 — 학습 없이 Gym `swe_agents` 로 학생이 과제를 푸는지 잰다 (2026-11-24, docs/E2E_SANDBOX.md §7.7).
# 재는 것: 통과율 · 과제별 보상 갈림 · 종료 사유 · 턴 수 · 롤아웃 시간 · 컨테이너 메모리 · 노드 메모리. 에이전트 설정은 학습 레시피
#   (teacher_swe_e2e_alpha.yaml) 와 같다: 200 턴 · 1,800 s · 턴당 8,192 · CodeAct 프롬프트 (이 Gym 의 학습 설정 기본값).
#   모델은 Pai fleet (main1 GPU 8, 벤치와 같은 서빙) 이다 — 학습 때의 NeMo-RL vLLM 워커가 아니다. 토큰 ID · logprob · refit 은 이 측정이 보지 않는다.
#   eval_swe_student.sh <tag> <hf_ckpt> <input.jsonl> [concurrency=128] [repeats=1]
#   환경변수: TURNS (200) · AGENT_TIMEOUT (1800) · MEM_MB (32768) · MAXLEN (131072) · MEM_STOP_GIB (1400, 노드 메모리가 넘으면 중단)
# 결과: $NRL_ROOT/p0_measure/swe/<tag>/{rollouts.jsonl,rollouts_failures.jsonl,node_samples.tsv,run.log}. GPU 를 잡는다 — 승인 뒤에만 띄운다.
# main1 에서 띄운다. 정답 패치 검증 (run_swe_golden.sh) 과 같은 Gym 서버 디렉토리를 쓰므로 동시에 못 돈다. 실행 중에 편집하지 않는다.
set -u
TAG=$1; CK=$(readlink -f "$2"); INPUT=$(readlink -f "$3"); CONC=${4:-128}; REPEATS=${5:-1}
TURNS=${TURNS:-200}; AGENT_TIMEOUT=${AGENT_TIMEOUT:-1800}; MEM_MB=${MEM_MB:-32768}; MAXLEN=${MAXLEN:-131072}; MEM_STOP_GIB=${MEM_STOP_GIB:-1400}
NRL=${NRL_ROOT:-/home/work/vidsearch/tools/nemo_rl}; P=$NRL/p0_measure; D=$P/swe/$TAG; GYM=/home/work/vidsearch/tools/Gym; SIF=$NRL/sif/swe
REPO=$(cd "$(dirname "$0")/../../../.." && pwd); HERE=$REPO/examples/configs/alpha/runs
PAI=/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha
MODEL_CFG=$HERE/alpha_vllm_model_swe_8k.yaml
PORTS="8000 8001 8002 8003 8004 8005 8006 8007"
mkdir -p $D; LOG=$D/run.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
ray_py() { "$NRL/clean_run.sh" /usr/bin/env RAY_ADDRESS=main1:6379 "$NRL/venv-driver/bin/python" -I "$@" </dev/null 2>/dev/null; }
on_sub1() { ssh -o BatchMode=yes -o ConnectTimeout=10 sub1 "$1" 2>/dev/null | tail -1; }
SERVER_DIRS="$GYM/responses_api_agents/swe_agents $GYM/responses_api_models/vllm_model"
gym_server_pids() {  # Gym 자식 서버 (`python app.py`) 는 명령줄로 구분되지 않는다 — cwd 로 찾는다
  local pid cwd d
  for pid in $(pgrep -f '^python app\.py$' 2>/dev/null); do
    cwd=$(readlink /proc/$pid/cwd 2>/dev/null) || continue
    for d in $SERVER_DIRS; do [ "$cwd" = "$d" ] && echo $pid; done
  done
}
stop_env() {  # 프로세스 그룹째로 내린다 (KNOWN_ISSUES 2026-11-20)
  [ -n "${ENV_PGID:-}" ] || return 0
  kill -INT -- -$ENV_PGID 2>/dev/null
  for i in $(seq 1 30); do pgrep -g $ENV_PGID > /dev/null || break; sleep 1; done
  pgrep -g $ENV_PGID > /dev/null && { kill -TERM -- -$ENV_PGID 2>/dev/null; sleep 5; }
  pgrep -g $ENV_PGID > /dev/null && { kill -KILL -- -$ENV_PGID 2>/dev/null; sleep 2; }
  wait ${ENV_PID:-} 2>/dev/null
  local left="$(pgrep -g $ENV_PGID | tr '\n' ' ')$(gym_server_pids | tr '\n' ' ')"
  [ -z "${left// /}" ] && log "gym env 정리됨 (그룹 $ENV_PGID)" || log "경고: gym 프로세스가 남아 있다: $left"
  ENV_PID=; ENV_PGID=
}
FLEET_UP=0
stop_fleet() {
  [ "$FLEET_UP" = 1 ] || return 0
  FLEET_UP=0; bash "$PAI/eval_sft/stop_fleet.sh" 0,1,2,3,4,5,6,7 >> "$LOG" 2>&1; log "fleet 정지"
}
FUSE_CNT='ps -eo ppid,comm | awk "\$1==1 && (\$2==\"squashfuse_ll\" || \$2==\"fuse-overlayfs\")" | wc -l'
fuse_orphans() {  # 강제 종료된 컨테이너가 남기는 FUSE 데몬 (부모 1) 수 — "main1+sub1". sub1 을 못 읽으면 ? 로 적는다
  local a b; a=$(bash -c "$FUSE_CNT"); b=$(on_sub1 "$FUSE_CNT"); echo "${a:-?}+${b:-?}"
}
DONE=0
cleanup() {
  [ "$DONE" = 1 ] && return 0; DONE=1
  [ -n "${EVAL_PID:-}" ] && kill $EVAL_PID 2>/dev/null
  [ -n "${SAMPLER:-}" ] && kill $SAMPLER 2>/dev/null
  stop_env; stop_fleet
}
abort() { log "ABORT: $*"; cleanup; exit 1; }
trap 'log "signal — 정리"; cleanup; exit 130' INT TERM
trap cleanup EXIT

# 0. 기동 전 점검 — 하나라도 깨지면 GPU 를 잡기 전에 멈춘다
[ "$(hostname)" = main1 ] || { log "ABORT: main1 에서 띄운다 (지금 $(hostname))"; exit 1; }
[ -f "$CK/model.safetensors.index.json" ] || { log "ABORT: HF 체크포인트가 아니다: $CK"; exit 1; }
[ -s "$INPUT" ] || { log "ABORT: 입력이 비었다: $INPUT"; exit 1; }
[ -e $D/rollouts.jsonl ] && { log "ABORT: $D/rollouts.jsonl 이 이미 있다"; exit 1; }
[ -n "$(gym_server_pids)" ] && { log "ABORT: 같은 서버 디렉토리에서 도는 Gym 서버가 이미 있다 (정답 패치 검증이 도는 중인가)"; exit 1; }
SESS='grep -c "^sessiondir max size = 16384" /etc/apptainer/apptainer.conf'
[ "$(bash -c "$SESS")" = 1 ] && [ "$(on_sub1 "$SESS")" = 1 ] || { log "ABORT: 쓰기 공간 16384 MB 설정이 두 노드에 없다 (setup_session.sh --sessiondir-mb 16384)"; exit 1; }
BINDCHK='p=$(pgrep -x raylet | head -1); tr "\0" "\n" < /proc/$p/environ 2>/dev/null | grep -c "^APPTAINER_BIND=/proc$"'
[ "$(bash -c "$BINDCHK")" = 1 ] && [ "$(on_sub1 "$BINDCHK")" = 1 ] || { log "ABORT: raylet 환경에 APPTAINER_BIND=/proc 이 없다 (runs/ray_cluster.sh 로 다시 띄운다)"; exit 1; }
for p in 8100 $PORTS; do
  (exec 3<>/dev/tcp/127.0.0.1/$p) 2>/dev/null && { log "ABORT: 포트 $p 를 이미 누가 듣고 있다 (묵은 fleet · 프록시 · 터널)"; exit 1; }
done
ra=$(ray_py "$HERE/gpu_check.py" | awk '/NRL_RUN_ACTORS/{print $2}')
[ "$ra" = "0" ] || { log "ABORT: NRL_RUN_ACTORS=${ra:-?} — 다른 NeMo-RL 런이 살아 있다"; exit 1; }
busy=$(ray_py "$HERE/gpu_check.py" | awk '/BUSY_GPUS/{print $2}')
[ "$busy" = "0" ] || { log "ABORT: BUSY_GPUS=${busy:-?}"; exit 1; }
MAIN_IP=$(getent hosts main1 | awk '{print $1}')
ROWS=$(wc -l < $INPUT); FUSE0=$(fuse_orphans)
log "armed: tag=$TAG ck=$CK input=$ROWS repeats=$REPEATS conc=$CONC turns=$TURNS agent_timeout=$AGENT_TIMEOUT mem_mb=$MEM_MB maxlen=$MAXLEN repo=$(git -C $REPO rev-parse --short HEAD) gym=$(git -C $GYM rev-parse --short HEAD) fuse_orphans_before=$FUSE0 load=$(cut -d' ' -f1 /proc/loadavg)"

# 1. fleet — eval_lcb.sh 와 같은 플래그 (도구 파서 · 사고 파서 · 제약 디코딩 끔 · prefix cache 끔)
export TOOLS=1 TOOL_PARSER=qwen3_xml REASONING_PARSER=nemotron_v3 VLLM_ENFORCE_STRICT_TOOL_CALLING=0 PREFIX_CACHE=0
FLEET_UP=1
GPUS=0,1,2,3,4,5,6,7 bash "$PAI/eval_sft/serve_fleet.sh" "$CK" "$MAXLEN" 8 8100 >> "$LOG" 2>&1
ready=0
for i in $(seq 1 120); do
  ready=0
  for p in $PORTS; do curl -s -m 3 "http://127.0.0.1:$p/v1/models" 2>/dev/null | grep -q '"alpha"' && ready=$((ready+1)); done
  [ "$ready" -eq 8 ] && break
  sleep 15
done
[ "$ready" -eq 8 ] || abort "vLLM 서버 $ready/8 만 준비됨 (30분)"
# 붙은 대상 확인 (KNOWN_ISSUES 2026-11-11 — 응답이 온다는 것만으로 통과시키지 않는다): 서버가 보고하는 모델 경로가 이 체크포인트여야 한다.
# Gym 이 실제로 부르는 것은 프록시 :8100 이므로 프록시를 거친 응답도 본다 (세션 고정 프록시라 여러 번 불러 여러 백엔드에 닿게 한다).
ident=0; for p in $PORTS; do curl -s -m 5 "http://127.0.0.1:$p/v1/models" | grep -qF "$CK" && ident=$((ident+1)); done
[ "$ident" -eq 8 ] || abort "모델 경로가 체크포인트와 다른 서버가 있다 ($ident/8 일치)"
pident=0; for i in $(seq 1 16); do curl -s -m 5 "http://127.0.0.1:8100/v1/models" | grep -qF "$CK" && pident=$((pident+1)); done
[ "$pident" -eq 16 ] || abort "프록시 :8100 응답의 모델 경로가 체크포인트와 다르다 ($pident/16 일치)"
log "fleet ready: 8/8 · 모델 경로 일치 백엔드 8/8 · 프록시 16/16"

# 2. Gym 서버. use_absolute_ip: 에이전트 컨테이너는 두 노드에 뜨고 (Ray SPREAD) 컨테이너 안에서 Gym 모델 서버를 부른다 —
#    기본값 (127.0.0.1) 이면 sub1 의 컨테이너가 모델에 닿지 못한다. 학습 레시피도 true 다.
cat > $D/overrides.yaml <<YAML
cache_dir: $NRL/gym_cache
use_absolute_ip: true
ray_head_node_address: main1:6379
swe_agents_train:
  responses_api_agents:
    swe_agents:
      agent_max_turns: $TURNS
      concurrency: $CONC
      swebench_agent_timeout: $AGENT_TIMEOUT
      apptainer_memory_limit_mb: $MEM_MB
      dataset_path: $INPUT
      container_formatter:
        - $SIF/r2egym_{instance_id}.sif
        - $SIF/swegym_sweb.eval.x86_64.{instance_id}.sif
        - $SIF/swerebench/{instance_id}.sif
swe_agents_val:
  responses_api_agents:
    swe_agents:
      agent_max_turns: $TURNS
      concurrency: $CONC
      swebench_agent_timeout: $AGENT_TIMEOUT
      apptainer_memory_limit_mb: $MEM_MB
      dataset_path: $INPUT
      container_formatter:
        - $SIF/r2egym_{instance_id}.sif
        - $SIF/swegym_sweb.eval.x86_64.{instance_id}.sif
        - $SIF/swerebench/{instance_id}.sif
YAML
export PATH=/home/work/vidsearch/tools/bin:$PATH UV_CACHE_DIR=/home/work/vidsearch/tools/uv_cache \
       UV_PYTHON_INSTALL_DIR=/home/work/vidsearch/tools/uv_python RAY_TMPDIR=/tmp
cd $GYM && source .venv/bin/activate
setsid gym env start --config responses_api_agents/swe_agents/configs/swebench_openhands_training.yaml \
  --config $MODEL_CFG --config $D/overrides.yaml \
  ++policy_base_url=http://127.0.0.1:8100/v1 ++policy_api_key=EMPTY ++policy_model_name=alpha \
  > $D/env_start.log 2>&1 < /dev/null &
ENV_PID=$!; ENV_PGID=$ENV_PID
./scripts/wait_for_servers.sh $ENV_PID 11000 2400 >> "$LOG" 2>&1 || abort "gym servers failed ($D/env_start.log)"
grep -q "'host': '127.0.0.1'" $D/env_start.log && abort "Gym 서버가 127.0.0.1 에 묶였다 — use_absolute_ip 가 먹지 않았다"
log "gym servers ready (주소 $(grep -o "'host': '[0-9.]*'" $D/env_start.log | sort -u | tr '\n' ' '))"

# 3. 롤아웃. 클라이언트가 멈추면 (/aggregate_metrics 재시도, 2026-11-24 golden_ultra_b2) GPU 를 쥔 채 남으므로 시간 한도를 건다.
WAVES=$(( (ROWS * REPEATS + CONC - 1) / CONC )); BUDGET=$(( WAVES * (AGENT_TIMEOUT + 1800) + 900 ))
t0=$(date +%s)
timeout $BUDGET gym eval run --no-serve --input $INPUT --output $D/rollouts.jsonl --num-repeats $REPEATS --concurrency $CONC > $D/eval_run.log 2>&1 < /dev/null &
EVAL_PID=$!

# 4. 노드 표본 (15 초): 세션 cgroup 의 shmem (컨테이너 쓰기 공간은 tmpfs 라 여기에 잡힌다) · 전체 (GiB) · load. 전체가 한도를 넘으면 롤아웃을 내린다.
( m='echo "$(awk "/^shmem /{printf \"%.2f\", \$2/2^30}" /sys/fs/cgroup/memory.stat) $(awk "{printf \"%.1f\", \$1/2^30}" /sys/fs/cgroup/memory.current) $(cut -d" " -f1 /proc/loadavg)"'
  echo -e "time\tmain1_shmem main1_total main1_load\tsub1_shmem sub1_total sub1_load" > $D/node_samples.tsv
  while kill -0 $EVAL_PID 2>/dev/null; do
    a=$(bash -c "$m"); b=$(on_sub1 "$m")
    echo -e "$(date '+%T')\t$a\t${b:-? ? ?}" >> $D/node_samples.tsv
    for tot in $(echo "$a" | awk '{print $2}') $(echo "${b:-0 0 0}" | awk '{print $2}'); do
      if [ "${tot%.*}" -ge "$MEM_STOP_GIB" ] 2>/dev/null; then echo "$(date '+%T') total ${tot} GiB >= $MEM_STOP_GIB" > $D/MEM_STOP; kill $EVAL_PID 2>/dev/null; fi
    done
    sleep 15
  done ) &
SAMPLER=$!
wait $EVAL_PID; rc=$?; EVAL_PID=
n_ok=$(wc -l < $D/rollouts.jsonl 2>/dev/null || echo 0); n_fail=$(wc -l < $D/rollouts_failures.jsonl 2>/dev/null || echo 0)
log "eval rc=$rc (124 = 시간 한도 $BUDGET s) rollouts=$n_ok failures=$n_fail expected=$((ROWS * REPEATS)) seconds=$(( $(date +%s) - t0 ))$([ -e $D/MEM_STOP ] && echo " MEM_STOP: $(cat $D/MEM_STOP)")"
kill $SAMPLER 2>/dev/null; SAMPLER=
stop_env
stop_fleet
log "fuse_orphans before=$FUSE0 after=$(fuse_orphans) · gym_servers_left=$(gym_server_pids | wc -l) · busy_gpus=$(ray_py "$HERE/gpu_check.py" | awk '/BUSY_GPUS/{print $2}')"
DONE=1
log "DONE"
