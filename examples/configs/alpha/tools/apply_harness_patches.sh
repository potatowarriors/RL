#!/bin/bash
# Gym `swe_agents` 가 쓰는 nv-OpenHands 체크아웃 (고정 5f018005) 에 examples/configs/alpha/harness_patches/*.patch 를 적용하거나 확인한다 (RL_PLAN.md 결정 88).
#   apply_harness_patches.sh          적용 (이미 적용돼 있으면 그대로 둔다)
#   apply_harness_patches.sh check    확인만 — 하나라도 안 돼 있으면 종료 1
#   apply_harness_patches.sh revert   되돌린다
# 대상: $NRL_ROOT/gym_cache/swe_agents/swe_openhands_setup/*/<고정 커밋>/OpenHands (학습 · 측정 공용, 두 노드가 보는 NFS).
# 커밋하지 않고 작업 트리만 고친다 — Gym 은 HEAD 가 고정 커밋과 같으면 트리를 건드리지 않고, 다르면 통째로 되돌린다
# (`swe_agents/app.py` `_sync_openhands_to_commit`). 패키지는 editable 설치라 (`.venv/…/openhands_ai.pth`) 소스 수정이 그대로 실행된다.
# 하니스 프로세스가 돌고 있으면 적용 · 되돌리기를 하지 않는다 — 컨테이너가 이 소스를 import 한다. 두 노드를 모두 본다.
set -u
MODE=${1:-apply}
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
PATCHES=$REPO/examples/configs/alpha/harness_patches
PIN=5f0180054732945df08ad2293903e6873f0492b6
ROOT=${NRL_ROOT:-/home/work/vidsearch/tools/nemo_rl}/gym_cache/swe_agents/swe_openhands_setup
rc=0; found=0
if [ "$MODE" != check ]; then
  # 대괄호는 pgrep 이 자기 명령줄에 맞는 것을 막는다.
  HARNESS='pgrep -f "swe_bench/run_infe[r]" | wc -l'
  n1=$(bash -c "$HARNESS"); n2=$(timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=10 sub1 "$HARNESS" 2>/dev/null | tail -1)
  [ "${n1:-1}" = 0 ] && [ "${n2:-1}" = 0 ] || { echo "FAIL     하니스 프로세스가 돌고 있거나 sub1 을 확인하지 못했다 (main1 ${n1:-?} · sub1 ${n2:-?}) — 끝난 뒤에 한다"; exit 1; }
fi
for G in "$ROOT"/*/"$PIN"/OpenHands; do
  [ -e "$G/.git" ] || continue
  found=1
  head=$(git -C "$G" rev-parse HEAD)
  [ "$head" = "$PIN" ] || { echo "FAIL     $G 가 ${head:0:9} 다 — 패치는 ${PIN:0:9} 용이다"; rc=1; continue; }
  for p in "$PATCHES"/*.patch; do
    name=$(basename "$p")
    if git -C "$G" apply --reverse --check "$p" 2>/dev/null; then
      if [ "$MODE" = revert ]; then
        git -C "$G" apply --reverse "$p" && echo "REVERTED  $name  ($G)" || { echo "FAIL     $name  ($G)"; rc=1; }
      else
        echo "APPLIED  $name  ($G)"
      fi
    elif [ "$MODE" = check ]; then echo "NOT APPLIED  $name  ($G)"; rc=1
    elif [ "$MODE" = revert ]; then echo "NOT APPLIED  $name  ($G) — 되돌릴 것이 없다"
    elif git -C "$G" apply --check "$p" 2>/dev/null; then
      git -C "$G" apply "$p" && echo "APPLIED NOW  $name  ($G)" || { echo "FAIL     $name  ($G)"; rc=1; }
    else echo "FAIL     $name 이 붙지 않는다 ($G) — 그 파일을 손으로 고쳤는가"; rc=1; fi
  done
done
[ $found = 1 ] || { echo "MISSING  고정 커밋 ${PIN:0:9} 의 하니스 체크아웃이 없다: $ROOT"; rc=1; }
exit $rc
