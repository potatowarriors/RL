#!/usr/bin/env bash
# SWE 태스크 이미지 → .sif 본 변환을 세션과 분리해 띄운다 (세션을 꺼도 계속 돈다).
# 노드마다 한 번씩, 서로 다른 샤드로 부른다 — Docker Hub 풀 한도가 출구 IP 마다 따로이기 때문이다.
#
#   main1:  ./launch_build_sifs.sh 0
#   sub1 :  ./launch_build_sifs.sh 1          (main1 에서 `ssh sub1 <이 스크립트> 1` 로도 된다)
#
# 환경변수: BLEND (기본 Super swe2.jsonl) · OUT (기본 $NRL_ROOT/sif/swe) · SHARDS (기본 2)
#           WORKERS (기본 4) · PULLS_PER_HOUR (기본 80) · PULL_COST (기본 1) · TAG (로그 이름, 기본 swe2)
#           EXTRA (build_sifs.py 에 덧붙일 인자, 예: EXTRA="--dry-run" 으로 기동 경로만 시험)
# 진행 확인: tail -f $OUT/logs/<TAG>_shard<K>.*.log · 기록 $OUT/build_log.shard<K>of<N>.jsonl
# 끝난 뒤 : build_sifs.py --blend ... --out $OUT --apptainer ... --verify     (전 파일 내용 확인)
# 멈추기  : kill -- -$(cat $OUT/logs/<TAG>_shard<K>.pgid)    (받던 이미지만 버려지고 다시 띄우면 이어받는다)
set -euo pipefail

K=${1:?샤드 번호가 필요하다 (main1 0, sub1 1)}
NRL_ROOT=${NRL_ROOT:-/home/work/vidsearch/tools/nemo_rl}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BLEND=${BLEND:-/home/work/Datasets/LL_datasets/posttraining/RL/Nemotron-RL-Super-Training-Blends/swe2.jsonl}
OUT=${OUT:-$NRL_ROOT/sif/swe}
SHARDS=${SHARDS:-2}
TAG=${TAG:-swe2}
APPTAINER=${APPTAINER:-$NRL_ROOT/apptainer_test/root145/usr/bin/apptainer}
TMP=${TMP:-/tmp/sifbuild}

mkdir -p "$OUT/logs" "$TMP"
PGID_FILE=$OUT/logs/${TAG}_shard$K.pgid
if [[ -s $PGID_FILE ]] && kill -0 -- "-$(cat "$PGID_FILE")" 2>/dev/null; then
    echo "이미 돌고 있다: 프로세스 그룹 $(cat "$PGID_FILE") ($PGID_FILE)" >&2
    exit 1
fi
LOG=$OUT/logs/${TAG}_shard$K.$(date +%m%d_%H%M).log

setsid nohup python3 "$HERE/build_sifs.py" \
    --blend "$BLEND" --out "$OUT" --apptainer "$APPTAINER" --tmp "$TMP" \
    --shard "$K/$SHARDS" --workers "${WORKERS:-4}" \
    --pulls-per-hour "${PULLS_PER_HOUR:-80}" --pull-cost "${PULL_COST:-1}" ${EXTRA:-} \
    >"$LOG" 2>&1 </dev/null &
echo $! >"$PGID_FILE"
echo "기동: $(hostname) 샤드 $K/$SHARDS · 프로세스 그룹 $(cat "$PGID_FILE") · 로그 $LOG"
