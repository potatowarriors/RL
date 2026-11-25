#!/bin/bash
# 고정한 Gym (v0.6.0 3045a793) 의 두 체크아웃에 examples/configs/alpha/gym_patches/*.patch 를 적용하거나 확인한다 (RL_PLAN.md 결정 79).
#   apply_gym_patches.sh          적용 (이미 적용돼 있으면 그대로 둔다)
#   apply_gym_patches.sh check    확인만 — 하나라도 안 돼 있으면 종료 1
# 대상: NeMo-RL 서브모듈 3rdparty/Gym-workspace/Gym (학습) · 측정용 체크아웃 $GYM_TOOLS (기본 /home/work/vidsearch/tools/Gym).
# 실행 중인 런이 있으면 적용하지 않는다 — Gym 서버가 이 소스를 import 한다.
set -u
MODE=${1:-apply}
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
PATCHES=$REPO/examples/configs/alpha/gym_patches
PIN=3045a793346a31291d7ea4ae6af3f94a35036ce5
rc=0
for G in "$REPO/3rdparty/Gym-workspace/Gym" "${GYM_TOOLS:-/home/work/vidsearch/tools/Gym}"; do
  [ -e "$G/.git" ] || { echo "MISSING  체크아웃이 없다: $G"; rc=1; continue; }
  head=$(git -C "$G" rev-parse HEAD)
  [ "$head" = "$PIN" ] || { echo "FAIL     $G 가 ${head:0:9} 다 — 패치는 ${PIN:0:9} 용이다"; rc=1; continue; }
  for p in "$PATCHES"/*.patch; do
    name=$(basename "$p")
    if git -C "$G" apply --reverse --check "$p" 2>/dev/null; then echo "APPLIED  $name  ($G)"
    elif [ "$MODE" = check ]; then echo "NOT APPLIED  $name  ($G)"; rc=1
    elif git -C "$G" apply --check "$p" 2>/dev/null; then git -C "$G" apply "$p" && echo "APPLIED NOW  $name  ($G)" || { echo "FAIL     $name  ($G)"; rc=1; }
    else echo "FAIL     $name 이 붙지 않는다 ($G) — 그 파일을 손으로 고쳤는가"; rc=1; fi
  done
done
exit $rc
