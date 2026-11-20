#!/bin/bash
# 인프라 설정 (seccomp · AppArmor · /dev/fuse) 이 적용된 세션에 Apptainer 를 설치하고 우리 쪽 설정을 넣는다.
# 전부 컨테이너 로컬 변경이라 세션을 다시 만들 때마다, 샌드박스가 뜨는 모든 노드 (main1 · sub1) 에서 실행한다.
#
# usage: setup_session.sh [--pid-mode bindproc|default] [--multi-uid] [--skip-install]
#   --pid-mode bindproc  PID namespace 를 유지하고 새 procfs 마운트만 끈다 (`mount proc = no`).
#                        실행 쪽에서 `APPTAINER_BIND=/proc` 를 함께 줘야 컨테이너 안에 /proc 가 생긴다.
#                        기본값 default 는 apptainer.conf 를 건드리지 않는다 (그러면 `--pid` 가 /proc 마스킹 때문에 실패한다).
#   --multi-uid          newuidmap · newgidmap 을 setuid 대신 file capability 로 바꾸고 subuid · subgid 범위를 준다
#                        (`--fakeroot` 의 다중 uid 매핑. SWE 경로에는 필요 없다).
# PID 방식과 다중 uid 는 승인 뒤에 켠다 (docs/E2E_SANDBOX.md §5).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)
PID_MODE=default; MULTI_UID=0; INSTALL=1
while [ $# -gt 0 ]; do
  case "$1" in
    --pid-mode) PID_MODE=$2; shift 2 ;;
    --multi-uid) MULTI_UID=1; shift ;;
    --skip-install) INSTALL=0; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
CONF=/etc/apptainer/apptainer.conf

if [ "$INSTALL" = 1 ] && ! command -v apptainer >/dev/null; then
  # 기준선과 같은 설치기 (Apptainer 1.4.5 deb, setuid 없음)
  sudo -n bash "$REPO/docker/install_apptainer.sh"
fi
command -v apptainer >/dev/null || { echo "apptainer is not installed" >&2; exit 1; }
echo "apptainer: $(apptainer --version)"
[ -e /etc/localtime ] || { echo "/etc/localtime is missing: apptainer exec will fail on its default bind (install tzdata)" >&2; exit 1; }

case "$PID_MODE" in
  default) : ;;
  bindproc)
    sudo -n sed -i -E 's/^allow pid ns = .*/allow pid ns = yes/; s/^mount proc = .*/mount proc = no/' "$CONF"
    echo "pid mode bindproc: $(grep -E '^(allow pid ns|mount proc) =' "$CONF" | tr '\n' ';')"
    echo "  -> the process that runs apptainer must export APPTAINER_BIND=/proc" ;;
  *) echo "unknown --pid-mode: $PID_MODE" >&2; exit 2 ;;
esac

if [ "$MULTI_UID" = 1 ]; then
  command -v setcap >/dev/null || sudo -n sh -c 'apt-get update -qq && apt-get install -y -qq libcap2-bin'
  command -v newuidmap >/dev/null || sudo -n sh -c 'apt-get update -qq && apt-get install -y -qq uidmap'
  sudo -n sh -c 'chmod u-s /usr/bin/newuidmap /usr/bin/newgidmap && setcap cap_setuid=ep /usr/bin/newuidmap && setcap cap_setgid=ep /usr/bin/newgidmap'
  U=$(id -un)
  grep -q "^$U:" /etc/subuid || echo "$U:100000:65536" | sudo -n tee -a /etc/subuid >/dev/null
  grep -q "^$U:" /etc/subgid || echo "$U:100000:65536" | sudo -n tee -a /etc/subgid >/dev/null
  echo "multi-uid: $(getcap /usr/bin/newuidmap /usr/bin/newgidmap | tr '\n' ' ') subuid=$(grep "^$U:" /etc/subuid)"
fi
echo "done. next: tools/apptainer/acceptance.sh <alpine.sif>"
