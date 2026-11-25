#!/bin/bash
# SWE 학생 롤아웃 측정 — 학습 없이 Gym `swe_agents` 로 학생이 과제를 푸는지 잰다 (2026-11-24, docs/E2E_SANDBOX.md §7.7).
# 재는 것: 통과율 · 과제별 보상 갈림 · 종료 사유 · 턴 수 · 롤아웃 시간 · 컨테이너 메모리 · 노드 메모리. 에이전트 설정은 학습 레시피
#   (teacher_swe_e2e_alpha.yaml) 와 같다: 200 턴 · 1,800 s · 턴당 8,192 · CodeAct 프롬프트 (이 Gym 의 학습 설정 기본값).
#   모델은 Pai fleet (main1 GPU 8, 벤치와 같은 서빙) 이다 — 학습 때의 NeMo-RL vLLM 워커가 아니다. 토큰 ID · logprob · refit 은 이 측정이 보지 않는다.
#   eval_swe_student.sh <tag> <hf_ckpt> <input.jsonl> [concurrency=128] [repeats=1]
#   환경변수: TURNS (200) · AGENT_TIMEOUT (1800) · MEM_MB (32768) · MAXLEN (131072) · MEM_STOP_GIB (1400, 노드의 anon + shmem 이 넘으면 중단 — 페이지 캐시는 세지 않는다)
#            GYM (기본 /home/work/vidsearch/tools/Gym) · MODEL_CFG (기본 alpha_vllm_model_swe_8k.yaml) — 다른 Gym 체크아웃 · 모델 서버 설정으로 돌릴 때
#            STUB_URL=http://127.0.0.1:<포트>/v1 — fleet 대신 이미 떠 있는 모델 주소를 쓴다 (가짜 모델 서버로 고리 전체를 CPU 에서 도는 시험)
#            RAY_HEAD (기본 main1:6379) — `local` 이면 클러스터에 붙지 않고 Gym 이 띄우는 로컬 Ray 를 쓴다 (Gym 과 클러스터의 Ray 버전이 다를 때.
#                    컨테이너가 이 노드에만 뜨고, Apptainer 환경변수는 실행기가 직접 준다)
#            OH_LOG=1 — OpenHands 디버그 로그를 켠다 (진단용, 기본은 Gym 과 같이 CRITICAL)
#            DRY=1 — GPU 없이 기동 경로만 돌린다: fleet 를 띄우지 않고 모델 주소를 죽은 포트로 둔다. Gym 서버 주소 · 두 노드의 컨테이너 기동 ·
#                    클라이언트 접속 · 컨테이너 정리를 CPU 로 확인하는 용도다 (롤아웃은 모델 호출에서 실패한다). 승인 없이 돌릴 수 있다.
# 결과: $NRL_ROOT/p0_measure/swe/<tag>/{rollouts.jsonl,rollouts_failures.jsonl,node_samples.tsv,run.log}. GPU 를 잡는다 — 승인 뒤에만 띄운다.
# main1 에서 띄운다. 정답 패치 검증 (run_swe_golden.sh) 과 같은 Gym 서버 디렉토리를 쓰므로 동시에 못 돈다. 실행 중에 편집하지 않는다.
set -u
TAG=$1; CK=$(readlink -f "$2"); INPUT=$(readlink -f "$3"); CONC=${4:-128}; REPEATS=${5:-1}
TURNS=${TURNS:-200}; AGENT_TIMEOUT=${AGENT_TIMEOUT:-1800}; MEM_MB=${MEM_MB:-32768}; MAXLEN=${MAXLEN:-131072}; MEM_STOP_GIB=${MEM_STOP_GIB:-1400}; DRY=${DRY:-0}; OH_LOG=${OH_LOG:-0}
NRL=${NRL_ROOT:-/home/work/vidsearch/tools/nemo_rl}; P=$NRL/p0_measure; D=$P/swe/$TAG; GYM=${GYM:-/home/work/vidsearch/tools/Gym}; SIF=$NRL/sif/swe
STUB_URL=${STUB_URL:-}; RAY_HEAD=${RAY_HEAD:-main1:6379}
REPO=$(cd "$(dirname "$0")/../../../.." && pwd); HERE=$REPO/examples/configs/alpha/runs
PAI=/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha
MODEL_CFG=${MODEL_CFG:-$HERE/alpha_vllm_model_swe_8k.yaml}
PREFIX_CACHE=${PREFIX_CACHE:-0}
PROMPT_MIX=${PROMPT_MIX:-codeact}   # codeact | codex | opencode | all (세 가지를 롤아웃마다 무작위 — RL_PLAN 결정 81)
PORTS="8000 8001 8002 8003 8004 8005 8006 8007"
mkdir -p $D; LOG=$D/run.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
ray_py() { "$NRL/clean_run.sh" /usr/bin/env RAY_ADDRESS=main1:6379 "$NRL/venv-driver/bin/python" -I "$@" </dev/null 2>/dev/null; }
on_sub1() { timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=10 sub1 "$1" 2>/dev/null | tail -1; }
PLUGINS=${PLUGINS:-$REPO/examples/configs/alpha/gym_plugins}
SERVER_DIRS="$GYM/responses_api_agents/swe_agents $GYM/responses_api_models/vllm_model $PLUGINS/responses_api_models/alpha_vllm_model"
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
kill_containers() {  # 이 실행의 컨테이너를 두 노드에서 내린다 — 서버를 내려도 컨테이너는 자기 세션으로 남는다 (swe_kill_containers.py)
  grep -o "SWE agents results root for this run: [^ ]*" $D/env_start.log 2>/dev/null | awk '{print $NF}' | sort -u > $D/results_roots.txt
  [ -s $D/results_roots.txt ] || { log "컨테이너 정리: 결과 루트를 로그에서 찾지 못했다"; return 0; }
  local c="python3 -I $HERE/swe_kill_containers.py $D/results_roots.txt kill"
  log "컨테이너 정리: main1 $(bash -c "$c" | tail -1) · sub1 $(on_sub1 "$c")"
  # 강제로 죽은 컨테이너가 남긴 FUSE 데몬 — 이 실행 전에 없던 것만 내린다
  log "FUSE 고아 정리: main1 $(python3 -I $HERE/swe_kill_containers.py orphans-kill $D/fuse_before.main1 | tail -1) · sub1 $(on_sub1 "python3 -I $HERE/swe_kill_containers.py orphans-kill $D/fuse_before.sub1")"
}
DONE=0
cleanup() {
  [ "$DONE" = 1 ] && return 0; DONE=1
  [ -n "${EVAL_PID:-}" ] && kill $EVAL_PID 2>/dev/null
  [ -n "${SAMPLER:-}" ] && kill $SAMPLER 2>/dev/null
  stop_env; kill_containers; stop_fleet
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
if [ "$RAY_HEAD" = local ]; then
  export APPTAINER_BIND=/proc APPTAINERENV_USER=$(id -un)   # Gym 이 띄우는 로컬 Ray 의 워커가 이 환경을 물려받는다
else
  [ "$(bash -c "$BINDCHK")" = 1 ] && [ "$(on_sub1 "$BINDCHK")" = 1 ] || { log "ABORT: raylet 환경에 APPTAINER_BIND=/proc 이 없다 (runs/ray_cluster.sh 로 다시 띄운다)"; exit 1; }
fi
if [ "$GYM" = /home/work/vidsearch/tools/Gym ]; then   # 설치된 Gym 이면 upstream 수정 (결정 79) 이 얹혀 있어야 한다 — 빠지면 결과 조립이 500 으로 죽는다
  "$REPO/examples/configs/alpha/tools/apply_gym_patches.sh" check >> "$LOG" 2>&1 || { log "ABORT: Gym 패치가 적용돼 있지 않다 (tools/apply_gym_patches.sh)"; exit 1; }
fi
for p in 8100 $PORTS; do
  (exec 3<>/dev/tcp/127.0.0.1/$p) 2>/dev/null && { log "ABORT: 포트 $p 를 이미 누가 듣고 있다 (묵은 fleet · 프록시 · 터널)"; exit 1; }
done
ra=$(ray_py "$HERE/gpu_check.py" | awk '/NRL_RUN_ACTORS/{print $2}')
[ "$ra" = "0" ] || { log "ABORT: NRL_RUN_ACTORS=${ra:-?} — 다른 NeMo-RL 런이 살아 있다"; exit 1; }
busy=$(ray_py "$HERE/gpu_check.py" | awk '/BUSY_GPUS/{print $2}')
[ "$busy" = "0" ] || { log "ABORT: BUSY_GPUS=${busy:-?}"; exit 1; }
MAIN_IP=$(getent hosts main1 | awk '{print $1}')
python3 -I $HERE/swe_kill_containers.py orphans > $D/fuse_before.main1; on_sub1 "python3 -I $HERE/swe_kill_containers.py orphans > $D/fuse_before.sub1; echo ok" > /dev/null
[ -e $D/fuse_before.sub1 ] || { log "ABORT: sub1 의 FUSE 고아 목록을 적지 못했다 (ssh)"; exit 1; }
ROWS=$(wc -l < $INPUT); FUSE0=$(fuse_orphans)
log "armed: tag=$TAG gym_dir=$GYM ray=$RAY_HEAD model_cfg=$(basename $MODEL_CFG) prefix_cache=$PREFIX_CACHE prompt_mix=$PROMPT_MIX ck=$CK input=$ROWS repeats=$REPEATS conc=$CONC turns=$TURNS agent_timeout=$AGENT_TIMEOUT mem_mb=$MEM_MB maxlen=$MAXLEN repo=$(git -C $REPO rev-parse --short HEAD) gym=$(git -C $GYM rev-parse --short HEAD) fuse_orphans_before=$FUSE0 load=$(cut -d' ' -f1 /proc/loadavg)"

if [ "$DRY" = 1 ]; then
  MODEL_URL=http://127.0.0.1:9/v1; log "DRY=1 — fleet 를 띄우지 않는다 (모델 주소 $MODEL_URL, 롤아웃은 모델 호출에서 실패한다)"
elif [ -n "$STUB_URL" ]; then
  MODEL_URL=$STUB_URL
  curl -s -m 5 "$STUB_URL/models" | grep -q '"alpha"' || { log "ABORT: STUB_URL 이 응답하지 않는다 ($STUB_URL)"; exit 1; }
  log "STUB_URL — fleet 를 띄우지 않는다 (모델 주소 $MODEL_URL)"
else
MODEL_URL=http://127.0.0.1:8100/v1
# 1. fleet — eval_lcb.sh 와 같은 플래그 (도구 파서 · 사고 파서 · 제약 디코딩 끔). prefix cache 는 기본 끔이고 PREFIX_CACHE=1 로 켠다
#    (2026-11-25 — 켬/끔 대조 측정. GDN 상태 fp32 + 캐시는 Pai 게이트 `prefix_cache_check.py --block 1056` 를 통과한 뒤에만 켠다).
export TOOLS=1 TOOL_PARSER=qwen3_xml REASONING_PARSER=nemotron_v3 VLLM_ENFORCE_STRICT_TOOL_CALLING=0 PREFIX_CACHE=$PREFIX_CACHE
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
# 캐시 설정이 요청대로 떴는지 서버 로그의 머리줄로 확인한다 (KNOWN_ISSUES 2026-11-08 — 기본값이 조용히 바뀐 적이 있다)
pc=0; for p in $PORTS; do grep -a -m1 "^\[serve\] " /home/work/vidsearch/tools/fleet_logs/serve_$p.log 2>/dev/null | grep -q "prefix_cache=$PREFIX_CACHE" && pc=$((pc+1)); done
[ "$pc" -eq 8 ] || abort "서버 로그의 prefix_cache 표지가 요청 ($PREFIX_CACHE) 과 다른 서버가 있다 ($pc/8 일치)"
log "fleet ready: 8/8 · 모델 경로 일치 백엔드 8/8 · 프록시 16/16 · prefix_cache=$PREFIX_CACHE 8/8"
fi

# 2. Gym 서버. use_absolute_ip: 에이전트 컨테이너는 두 노드에 뜨고 (Ray SPREAD) 컨테이너 안에서 Gym 모델 서버를 부른다 —
#    기본값 (127.0.0.1) 이면 sub1 의 컨테이너가 모델에 닿지 못한다. 학습 레시피도 true 다.
# 에이전트 프롬프트 (Gym 학습 설정은 CodeAct 만 켜 둔다 — 세 클래스와 프롬프트 틀은 설치본에 다 있다, E2E_SANDBOX.md §7.7 "프롬프트 혼합")
po() { printf '        - user_prompt_template: responses_api_agents/swe_agents/prompts/%s/user_prompt.j2\n          system_prompt_template: responses_api_agents/swe_agents/prompts/%s/system_prompt.j2\n          agent_cls: %s\n          diversify_tool_names: false\n' "$1" "$1" "$2"; }
case "$PROMPT_MIX" in
  codeact)  PO=$(po openhands CodeActAgent); PO_RANDOM=false ;;
  codex)    PO=$(po codex CodexAgent); PO_RANDOM=false ;;
  opencode) PO=$(po opencode OpenCodeAgent); PO_RANDOM=false ;;
  all)      PO=$(po codex CodexAgent; po opencode OpenCodeAgent; po openhands CodeActAgent); PO_RANDOM=true ;;
  *) echo "PROMPT_MIX 는 codeact | codex | opencode | all"; exit 1 ;;
esac
cat > $D/overrides.yaml <<YAML
cache_dir: $NRL/gym_cache
use_absolute_ip: true
$([ "$DRY" = 1 ] && echo "model_endpoint_readiness_timeout_seconds: 0")
$([ "$RAY_HEAD" = local ] || echo "ray_head_node_address: $RAY_HEAD")
swe_agents_train:
  responses_api_agents:
    swe_agents:
      agent_max_turns: $TURNS
      concurrency: $CONC
      swebench_agent_timeout: $AGENT_TIMEOUT
      apptainer_memory_limit_mb: $MEM_MB
      openhands_should_log: $([ "$OH_LOG" = 1 ] && echo true || echo false)
      agent_prompt_override_random: $PO_RANDOM
      agent_prompt_overrides:
$PO
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
      openhands_should_log: $([ "$OH_LOG" = 1 ] && echo true || echo false)
      agent_prompt_override_random: $PO_RANDOM
      agent_prompt_overrides:
$PO
      dataset_path: $INPUT
      container_formatter:
        - $SIF/r2egym_{instance_id}.sif
        - $SIF/swegym_sweb.eval.x86_64.{instance_id}.sif
        - $SIF/swerebench/{instance_id}.sif
YAML
export PATH=/home/work/vidsearch/tools/bin:$PATH UV_CACHE_DIR=/home/work/vidsearch/tools/uv_cache \
       UV_PYTHON_INSTALL_DIR=/home/work/vidsearch/tools/uv_python RAY_TMPDIR=/tmp
cd $GYM && source .venv/bin/activate
if [ "$RAY_HEAD" = local ]; then
  # 주소 없이 ray.init() 을 부르면 이 노드에 이미 떠 있는 클러스터에 붙는다 (버전이 다르면 "Version mismatch" 로 죽는다, 2026-11-25).
  # 새 로컬 Ray 를 강제하고, 임시 디렉토리도 따로 둬 운영 클러스터의 /tmp/ray 를 건드리지 않는다.
  export RAY_ADDRESS=local RAY_TMPDIR=/tmp/gymlocal_ray_$$; mkdir -p $RAY_TMPDIR
fi
export NEMO_GYM_EXTRA_ROOTS=$PLUGINS   # 모델 서버는 alpha 플러그인 (alpha_vllm_model_swe_8k.yaml)
export RAY_DEDUP_LOGS=0   # 워커 로그 줄이 합쳐지면 (repeated Nx across cluster) 어느 노드 · 어느 과제의 줄인지 잃는다
setsid gym env start --config responses_api_agents/swe_agents/configs/swebench_openhands_training.yaml \
  --config $MODEL_CFG --config $D/overrides.yaml \
  ++policy_base_url=$MODEL_URL ++policy_api_key=EMPTY ++policy_model_name=alpha \
  > $D/env_start.log 2>&1 < /dev/null &
ENV_PID=$!; ENV_PGID=$ENV_PID
# Gym 의 scripts/wait_for_servers.sh 는 127.0.0.1 만 본다 — use_absolute_ip 면 head 서버가 노드 주소에만 묶이므로 직접 기다린다.
HEAD=http://$MAIN_IP:11000; gready=0
for i in $(seq 1 480); do
  kill -0 $ENV_PID 2>/dev/null || abort "gym env start 가 죽었다 ($D/env_start.log)"
  urls=$(curl -s -m 5 "$HEAD/server_instances" 2>/dev/null | python3 -I -c "import json,sys
try: print(' '.join(x['url'] for x in json.load(sys.stdin) if x.get('url')))
except Exception: pass" 2>/dev/null)
  if [ -n "$urls" ]; then
    n=0; okc=0; for u in $urls; do n=$((n+1)); curl -sf -m 5 "$u/docs" > /dev/null 2>&1 && okc=$((okc+1)); done
    [ "$okc" -eq "$n" ] && [ "$n" -ge 3 ] && { gready=1; break; }
  fi
  sleep 5
done
[ "$gready" = 1 ] || abort "gym 서버가 40분 안에 준비되지 않았다 ($D/env_start.log)"
# 묶인 주소 확인: head 가 돌려준 서버 주소가 전부 이 노드의 주소여야 한다 (127.0.0.1 이면 sub1 의 컨테이너가 모델 서버에 닿지 못한다).
# 로그의 설정 출력으로 확인하지 않는다 — 그 줄은 서버가 다 뜬 뒤에 찍혀 경쟁이 난다 (2026-11-25 student_stub3 에서 빈 값으로 오판).
hosts=$(for u in $urls; do echo "$u" | sed -E 's#^https?://([^:/]+).*#\1#'; done | sort -u | tr '\n' ' ')
[ "$hosts" = "$MAIN_IP " ] || abort "Gym 서버가 노드 주소 ($MAIN_IP) 가 아닌 곳에 묶였다: $hosts"
log "gym servers ready ($n 개, 주소 $hosts)"

# 3. 롤아웃. 클라이언트가 멈추면 (/aggregate_metrics 재시도, 2026-11-24 golden_ultra_b2) GPU 를 쥔 채 남으므로 시간 한도를 건다.
WAVES=$(( (ROWS * REPEATS + CONC - 1) / CONC )); BUDGET=$(( WAVES * (AGENT_TIMEOUT + 1800) + 900 ))
t0=$(date +%s)
timeout $BUDGET gym eval run --no-serve --input $INPUT --output $D/rollouts.jsonl --num-repeats $REPEATS --concurrency $CONC ++use_absolute_ip=true > $D/eval_run.log 2>&1 < /dev/null &
EVAL_PID=$!

# 4. 노드 표본 (15 초): 세션 cgroup 의 shmem (컨테이너 쓰기 공간은 tmpfs 라 여기에 잡힌다) · anon · 전체 (GiB) · load.
#    중단 기준은 anon + shmem 이다 — 전체 (memory.current) 는 되돌려 받을 수 있는 페이지 캐시 (모델 가중치 · SIF) 를 포함해 부풀어 보인다.
( m='awk "/^shmem /{s=\$2} /^anon /{a=\$2} END{printf \"%.2f %.1f \", s/2^30, a/2^30}" /sys/fs/cgroup/memory.stat; echo "$(awk "{printf \"%.1f\", \$1/2^30}" /sys/fs/cgroup/memory.current) $(cut -d" " -f1 /proc/loadavg)"'
  echo -e "time\tmain1_shmem main1_anon main1_total main1_load\tsub1_shmem sub1_anon sub1_total sub1_load" > $D/node_samples.tsv
  while kill -0 $EVAL_PID 2>/dev/null; do
    a=$(bash -c "$m"); b=$(on_sub1 "$m")
    echo -e "$(date '+%T')\t$a\t${b:-? ? ? ?}" >> $D/node_samples.tsv
    for used in $(echo "$a" | awk '{printf "%d", $1+$2}') $(echo "${b:-0 0 0 0}" | awk '{printf "%d", $1+$2}'); do
      if [ "$used" -ge "$MEM_STOP_GIB" ] 2>/dev/null; then echo "$(date '+%T') anon+shmem ${used} GiB >= $MEM_STOP_GIB" > $D/MEM_STOP; kill $EVAL_PID 2>/dev/null; fi
    done
    sleep 15
  done ) &
SAMPLER=$!
wait $EVAL_PID; rc=$?; EVAL_PID=
n_ok=$(wc -l < $D/rollouts.jsonl 2>/dev/null || echo 0); n_fail=$(wc -l < $D/rollouts_failures.jsonl 2>/dev/null || echo 0)
log "eval rc=$rc (124 = 시간 한도 $BUDGET s) rollouts=$n_ok failures=$n_fail expected=$((ROWS * REPEATS)) seconds=$(( $(date +%s) - t0 ))$([ -e $D/MEM_STOP ] && echo " MEM_STOP: $(cat $D/MEM_STOP)")"
kill $SAMPLER 2>/dev/null; SAMPLER=
stop_env
kill_containers
stop_fleet
cc="python3 -I $HERE/swe_kill_containers.py $D/results_roots.txt"
log "잔재: containers main1 [$(bash -c "$cc" | tail -1)] sub1 [$(on_sub1 "$cc")] · fuse_orphans before=$FUSE0 after=$(fuse_orphans) · gym_servers_left=$(gym_server_pids | wc -l) · busy_gpus=$(ray_py "$HERE/gpu_check.py" | awk '/BUSY_GPUS/{print $2}')"
DONE=1
log "DONE"
