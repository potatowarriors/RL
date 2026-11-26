#!/bin/bash
# Gym `swe_agents` 가 쓰는 nv-OpenHands 체크아웃 (고정 5f018005) 에 examples/configs/alpha/harness_patches/*.patch 를 적용하거나 확인한다 (RL_PLAN.md 결정 88).
#   apply_harness_patches.sh          적용 (이미 적용돼 있으면 그대로 둔다)
#   apply_harness_patches.sh check    확인만 — 아래 조건 중 하나라도 안 되면 종료 1
#   apply_harness_patches.sh revert   되돌린다
# 대상은 Gym 이 실제로 고르는 트리 하나다: <cache_dir>/swe_agents/swe_openhands_setup/<저장소 이름>-<sha256(URL) 8 자>/<커밋>/OpenHands
#   (`swe_agents/app.py` `_openhands_setup_target` · `_repo_slug`). cache_dir 기본은 $NRL_ROOT/gym_cache (학습 레시피 · 측정 실행기가 쓰는 값),
#   다른 값이면 GYM_CACHE_DIR 로 준다. 레시피가 가리키는 저장소 · 커밋 · cache_dir 이 이 값과 같은지는 `check_alpha_recipe.py` 가 본다.
# check 가 보는 것: ① 그 트리가 있다 ② HEAD 가 고정 커밋이다 (다르면 Gym 이 트리를 통째로 되돌린다) ③ `.venv/bin/python` 이 있다
#   (없으면 Gym 이 트리를 지우고 새로 받는다 — 패치가 사라진다) ④ 패치가 전부 적용돼 있다.
# 커밋하지 않고 작업 트리만 고친다. 패키지는 editable 설치라 (`.venv/…/openhands_ai.pth`) 소스 수정이 그대로 실행된다.
# 하니스 프로세스가 돌고 있으면 적용 · 되돌리기를 하지 않는다 — 컨테이너가 이 소스를 import 한다. 두 노드를 모두 본다.
set -u
MODE=${1:-apply}
case "$MODE" in apply|check|revert) ;; *) echo "FAIL     모드는 apply · check · revert 중 하나다: $MODE"; exit 1 ;; esac
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
PATCHES=$REPO/examples/configs/alpha/harness_patches
PIN=5f0180054732945df08ad2293903e6873f0492b6
URL=https://github.com/sdevare-nv/nv-OpenHands.git
ROOT=${GYM_CACHE_DIR:-${NRL_ROOT:-/home/work/vidsearch/tools/nemo_rl}/gym_cache}/swe_agents/swe_openhands_setup
SLUG=$(python3 -I -c 'import hashlib, re, sys; u = sys.argv[1]; print(re.sub(r"\.git$", "", u.rstrip("/").rsplit("/", 1)[-1]) + "-" + hashlib.sha256(u.encode()).hexdigest()[:8])' "$URL")
G=$ROOT/$SLUG/$PIN/OpenHands
if [ "$MODE" != check ]; then
  # 대괄호는 pgrep 이 자기 명령줄에 맞는 것을 막는다.
  HARNESS='pgrep -f "swe_bench/run_infe[r]" | wc -l'
  n1=$(bash -c "$HARNESS"); n2=$(timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=10 sub1 "$HARNESS" 2>/dev/null | tail -1)
  [ "${n1:-1}" = 0 ] && [ "${n2:-1}" = 0 ] || { echo "FAIL     하니스 프로세스가 돌고 있거나 sub1 을 확인하지 못했다 (main1 ${n1:-?} · sub1 ${n2:-?}) — 끝난 뒤에 한다"; exit 1; }
fi
[ -e "$G/.git" ] || { echo "MISSING  Gym 이 쓸 하니스 트리가 없다: $G"; exit 1; }
head=$(git -C "$G" rev-parse HEAD)
[ "$head" = "$PIN" ] || { echo "FAIL     $G 의 HEAD 가 ${head:0:9} 다 — 고정 커밋 ${PIN:0:9} 이 아니면 Gym 이 트리를 되돌린다"; exit 1; }
[ -x "$G/.venv/bin/python" ] || { echo "FAIL     $G/.venv/bin/python 이 없다 — Gym 이 트리를 지우고 새로 받는다 (패치가 사라진다)"; exit 1; }
rc=0
for p in "$PATCHES"/*.patch; do
  name=$(basename "$p")
  if git -C "$G" apply --reverse --check "$p" 2>/dev/null; then
    if [ "$MODE" = revert ]; then
      if git -C "$G" apply --reverse "$p"; then echo "REVERTED  $name  ($G)"; else echo "FAIL     $name  ($G)"; rc=1; fi
    else
      echo "APPLIED  $name  ($G)"
    fi
  elif [ "$MODE" = check ]; then echo "NOT APPLIED  $name  ($G)"; rc=1
  elif [ "$MODE" = revert ]; then echo "NOT APPLIED  $name  ($G) — 되돌릴 것이 없다"
  elif git -C "$G" apply --check "$p" 2>/dev/null; then
    if git -C "$G" apply "$p"; then echo "APPLIED NOW  $name  ($G)"; else echo "FAIL     $name  ($G)"; rc=1; fi
  else echo "FAIL     $name 이 붙지 않는다 ($G) — 그 파일을 손으로 고쳤는가"; rc=1; fi
done
# 같은 캐시에 다른 하니스 트리가 있으면 알린다 — 설정이 다른 저장소 · 커밋을 가리키면 그 트리가 쓰이고, 거기에는 패치가 없다.
for other in "$ROOT"/*/*/OpenHands; do
  [ -e "$other/.git" ] && [ "$other" != "$G" ] && echo "NOTE     다른 하니스 트리가 있다 (패치 대상 아님): $other"
done
exit $rc
