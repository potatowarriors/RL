#!/usr/bin/env bash
# alpha RL 용 2노드 Ray 클러스터를 띄우고 내린다. 세션을 다시 만들 때마다 한 번 (main1 에서 실행).
#
#   ray_cluster.sh start     head = main1, worker = sub1 (ssh). 이미 떠 있으면 아무것도 하지 않는다
#   ray_cluster.sh status    노드 · 자원 · spill 경로
#   ray_cluster.sh stop      두 노드의 Ray 프로세스를 내린다 (런이 돌고 있으면 거부, FORCE=1 로 강제)
#
# 값은 2026-10-07 에 손으로 띄운 클러스터에서 잰 것이다 (기동 명령이 기록돼 있지 않았다, SETUP.md §5.1):
#   head 포트 6379 · 노드마다 CPU 220 · GPU 8 · object store 64 GiB · plasma /dev/shm · temp /tmp/ray.
# 달라진 것은 spill 경로 하나다 — 루트 디스크 (/tmp/ray/session_*) 대신 NFS $NRL_ROOT/ray_spill/<노드> 로 보낸다.
# 루트 디스크가 spill 로 찬 사고가 두 번 있었다 (KNOWN_ISSUES 2026-10-13 · 10-20).
# 주소는 역할 이름에서 푼다 — 세션을 다시 만들면 IP 가 바뀐다 (2026-11-24).
# 전제: 두 노드에 compat 595 가 들어가 있어야 한다 (Pai eval_sft/restore_bench_env.sh, SETUP.md §5.1 의 2).
set -euo pipefail

NRL=${NRL_ROOT:-/home/work/vidsearch/tools/nemo_rl}
PORT=6379
OBJECT_STORE=${OBJECT_STORE:-68719476736}
CPUS=${CPUS:-220}
GPUS=${GPUS:-8}
RAY="$NRL/clean_run.sh $NRL/venv-driver/bin/ray"
# raylet 의 환경은 그 노드의 모든 Ray 워커가 물려받는다. Gym `swe_agents` 는 Ray 워커에서 `apptainer exec --pid` 를 부르는데,
# 이 세션에서는 `--pid` 가 `APPTAINER_BIND=/proc` 없이는 /proc 없는 컨테이너를 만든다 (E2E_SANDBOX.md §5 · §6.1, bindproc).
# 드라이버 쪽 환경변수는 raylet 이 띄우는 워커에 닿지 않으므로 여기서 준다.
RAY_START="$NRL/clean_run.sh /usr/bin/env APPTAINER_BIND=/proc $NRL/venv-driver/bin/ray"
MAIN_IP=$(getent hosts main1 | awk '{print $1}')
SUB_IP=$(getent hosts sub1 | awk '{print $1}')
[ -n "$MAIN_IP" ] && [ -n "$SUB_IP" ] || { echo "main1 · sub1 주소를 풀지 못했다" >&2; exit 1; }

compat_ok() { [ "$(readlink /usr/local/cuda/compat/lib.real/libcuda.so.1 2>/dev/null)" = "libcuda.so.595.91.07" ]; }
node_flags() {  # $1 = 노드 이름, $2 = IP
    echo "--node-ip-address=$2 --object-store-memory=$OBJECT_STORE --object-spilling-directory=$NRL/ray_spill/$1 \
--num-cpus=$CPUS --num-gpus=$GPUS --disable-usage-stats"
}

case "${1:-}" in
start)
    [ "$(hostname)" = main1 ] || { echo "main1 에서 실행한다 (지금 $(hostname))" >&2; exit 1; }
    if $RAY status --address=main1:$PORT </dev/null >/dev/null 2>&1; then
        echo "이미 떠 있다 (main1:$PORT) — status 로 확인"; exit 0
    fi
    compat_ok || { echo "main1 에 compat 595 가 없다 — restore_bench_env.sh 먼저" >&2; exit 1; }
    ssh -o BatchMode=yes sub1 "$(declare -f compat_ok); compat_ok" \
        || { echo "sub1 에 compat 595 가 없다 — restore_bench_env.sh 먼저" >&2; exit 1; }
    mkdir -p "$NRL/ray_spill/main1" "$NRL/ray_spill/sub1"
    # shellcheck disable=SC2046
    $RAY_START start --head --port=$PORT $(node_flags main1 "$MAIN_IP") </dev/null
    # shellcheck disable=SC2029
    ssh -o BatchMode=yes sub1 "$RAY_START start --address=main1:$PORT $(node_flags sub1 "$SUB_IP") </dev/null"
    # 준비 대기. 기동 직후에는 노드마다 미리 띄우는 워커 (CPU 수 = 220) 가 등록 중이다. 그 사이 (기동 18 초 뒤) 에 낸 태스크가
    # 응답 없이 멈춘 적이 있다 — raylet "workers ... have not registered within the timeout" (2026-11-24). 두 노드에서 태스크가
    # 실제로 돌아올 때까지 기다린 뒤에 돌려준다. 한 번의 시도는 60 초로 끊는다.
    sleep 20
    ready=0
    for _ in $(seq 1 20); do
        if timeout 60 "$NRL/clean_run.sh" /usr/bin/env RAY_ADDRESS=main1:$PORT "$NRL/venv-driver/bin/python" -I \
            "$(dirname "${BASH_SOURCE[0]}")/gpu_check.py" </dev/null 2>/dev/null | grep -q '^BUSY_GPUS'; then
            ready=1; break
        fi
        sleep 10
    done
    [ "$ready" = 1 ] || { echo "클러스터가 떴으나 태스크가 돌아오지 않는다 — raylet 로그 /tmp/ray/session_latest/logs/raylet.err" >&2; exit 1; }
    echo "[$(date '+%F %T')] ray_cluster start: head main1($MAIN_IP):$PORT worker sub1($SUB_IP) object_store=$OBJECT_STORE spill=$NRL/ray_spill/<node> (태스크 응답 확인)" \
        | tee -a "$NRL/logs/ray_cluster.log"
    ;;
status)
    $RAY status --address=main1:$PORT </dev/null
    for n in main1 sub1; do
        echo "spill $n: $(find "$NRL/ray_spill/$n" -type f 2>/dev/null | wc -l) files, $(du -sh "$NRL/ray_spill/$n" 2>/dev/null | cut -f1)"
    done
    ;;
stop)
    if [ "${FORCE:-0}" != 1 ]; then
        actors=$($NRL/clean_run.sh /usr/bin/env RAY_ADDRESS=main1:$PORT "$NRL/venv-driver/bin/python" -I \
            "$(dirname "${BASH_SOURCE[0]}")/gpu_check.py" </dev/null 2>/dev/null | awk '/NRL_RUN_ACTORS/{print $2}')
        [ "${actors:-0}" = 0 ] || { echo "NeMo-RL 런 액터 $actors 개가 살아 있다 — 내리지 않는다 (FORCE=1 로 강제)" >&2; exit 1; }
    fi
    # shellcheck disable=SC2029
    ssh -o BatchMode=yes sub1 "$RAY stop </dev/null" || true
    $RAY stop </dev/null || true
    echo "[$(date '+%F %T')] ray_cluster stop" | tee -a "$NRL/logs/ray_cluster.log"
    ;;
*)
    echo "usage: $0 start|status|stop" >&2; exit 2 ;;
esac
