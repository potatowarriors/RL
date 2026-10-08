#!/bin/bash
# alpha RLVR 런 실행기 (student_rlvr1_alpha.yaml, 2노드: 학습·롤아웃 노드는 Ray 가 기동마다 정한다). 2026-10-08.
# 산출물은 리포 워크스페이스 안 results/alpha/<campaign>/ 에 둔다 (gitignored, 사용자 지시 2026-10-08):
#   <tag>/ (logger.log_dir: TensorBoard·덤프·wandb·r3_trace·nemo_gym 로그) · <tag>.log (드라이버 로그) · runs.log (기동·종료 기록) · ckpt/
# wandb 는 RL 전용 프로젝트 alpha-rl 에 올린다 (group = campaign, name = campaign_tag). 키는 Pai 규약 키 파일에서 읽어 환경변수로만 넘긴다
# (clean_run.sh 화이트리스트가 WANDB_API_KEY 를 통과시킨다). 값은 출력·로그하지 않는다.
#   launch.sh <campaign> <tag> [hydra override ...]
#   환경변수: RECIPE (기본: student_rlvr1_alpha.yaml — teacher 는 teacher_*_alpha.yaml) · DATA (기본: judge-free 블렌드) · VAL (기본: DATA)
#            CKPT (기본: results/alpha/<campaign>/ckpt — 같은 CKPT 로 다시 띄우면 최신 step 에서 재개, G7)
#            R3_TRACE (기본 1) · WANDB (기본 1, 0 이면 끈다)
#            EXTRA_ENV (공백으로 나눈 VAR=값 목록 — clean_run.sh 화이트리스트를 거쳐 드라이버에 넘긴다. 예: NRL_REQUEST_PRIORITY_LONG_AGENTS=a,b)
# 기동 전에 GPU 점유(1 GiB)를 검사하고, 지난 런이 남긴 Gym 서버를 정리한다 (KNOWN_ISSUES 2026-10-08).
# 이 스크립트는 실행 중에 편집하지 않는다 — bash 는 스크립트를 오프셋으로 읽는다.
set -u
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
HERE=$REPO/examples/configs/alpha/runs
NRL=/home/work/vidsearch/tools/nemo_rl
campaign=$1; tag=$2; shift 2
OUT=$REPO/results/alpha/$campaign
RECIPE=${RECIPE:-examples/configs/alpha/student_rlvr1_alpha.yaml}
DATA=${DATA:-/home/work/Datasets/LL_datasets/posttraining/RL/alpha_blends/rlvr1_alpha_judgefree.jsonl}
VAL=${VAL:-$DATA}
CKPT=${CKPT:-$OUT/ckpt}
mkdir -p $OUT/$tag
cd $REPO
ray_py() { $NRL/clean_run.sh /usr/bin/env RAY_ADDRESS=10.0.37.4:6379 $NRL/venv-driver/bin/python -I "$@" </dev/null 2>/dev/null; }
busy=$(ray_py $HERE/gpu_check.py | awk '/BUSY_GPUS/{print $2}')
[ "$busy" = "0" ] || { echo "[$(date '+%F %T')] case=$tag GPU 점유 검사 실패 (BUSY_GPUS=$busy) — 기동 중단" | tee -a $OUT/runs.log; exit 1; }
gym_left=$(ray_py $HERE/gym_cleanup.py kill | awk '/GYM_LEFTOVER_TOTAL/{print $2}')
echo "[$(date '+%F %T')] case=$tag 남은 Gym 서버 정리: ${gym_left:-?}개" | tee -a $OUT/runs.log
TRACE_ENV=""
[ "${R3_TRACE:-1}" = "1" ] && TRACE_ENV="NRL_R3_TRACE=1 NRL_R3_TRACE_DIR=$OUT/$tag/r3_trace"
WANDB_ARGS=""
if [ "${WANDB:-1}" = "1" ]; then
  WANDB_KEY_FILE=/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha/scripts/.wandb_key
  [ -s "$WANDB_KEY_FILE" ] || { echo "[$(date '+%F %T')] case=$tag wandb 키 파일 없음 — 기동 중단 (WANDB=0 이면 끈다)" | tee -a $OUT/runs.log; exit 1; }
  export WANDB_API_KEY="$(cat "$WANDB_KEY_FILE")"
  WANDB_ARGS="logger.wandb_enabled=true logger.wandb.project=alpha-rl ++logger.wandb.group=$campaign logger.wandb.name=${campaign}_${tag}"
fi
echo "[$(date '+%F %T')] host=$(hostname) commit=$(git rev-parse --short HEAD) case=$tag recipe=$(basename $RECIPE) data=$(basename $DATA) ckpt=$CKPT r3_trace=${R3_TRACE:-1} wandb=${WANDB:-1} extra_env=${EXTRA_ENV:-} overrides=$*" | tee -a $OUT/runs.log
env -u CUDA_VISIBLE_DEVICES $NRL/clean_run.sh /usr/bin/env -u NVIDIA_VISIBLE_DEVICES -u HOSTNAME \
  $(env | grep -o '^BACKENDAI_[A-Z_]*' | sed 's/^/-u /' | tr '\n' ' ') \
  RAY_ADDRESS=10.0.37.4:6379 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 TP_SOCKET_IFNAME=eth0 \
  NEMO_GYM_EXTRA_ROOTS=$REPO/examples/configs/alpha/gym_plugins NRL_ROUTER_REPLAY_VALIDATE=1 $TRACE_ENV ${EXTRA_ENV:-} \
  $NRL/bin/uv run --locked python examples/nemo_gym/run_grpo_nemo_gym.py \
  --config $RECIPE \
  data.train.data_path=$DATA data.validation.data_path=$VAL \
  logger.log_dir=$OUT/$tag env.nemo_gym.nemo_gym_log_dir=$OUT/$tag/nemo_gym \
  ++env.nemo_gym.uv_venv_dir=$NRL/gates/gym_smoke/gym_venvs ++env.nemo_gym.uv_cache_dir=$NRL/uv-cache \
  checkpointing.checkpoint_dir=$CKPT $WANDB_ARGS \
  "$@" < /dev/null > $OUT/$tag.log 2>&1
rc=$?
echo "[$(date '+%F %T')] case=$tag rc=$rc" | tee -a $OUT/runs.log
exit $rc
