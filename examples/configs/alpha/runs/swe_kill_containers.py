#!/usr/bin/env python3
"""이 노드에서 한 Gym `swe_agents` 실행이 띄운 컨테이너를 찾아 (kill 이면) 내린다 (2026-11-24, docs/E2E_SANDBOX.md §7.7).
  swe_kill_containers.py <roots_file> [kill]
roots_file: 그 실행의 결과 루트 경로가 한 줄씩 (env_start.log 의 "SWE agents results root for this run: <경로>").
찾는 법: 컨테이너 안의 프로세스는 그 결과 디렉토리를 bind 마운트한 mount namespace 에 있다 → /proc/<pid>/mountinfo 에 결과 루트 이름이 보인다.
  명령줄로는 못 찾는다 — Apptainer starter 가 프로세스 이름을 "Apptainer runtime parent: <이미지>" 로 바꾼다.
내리는 법: 컨테이너는 자기 세션으로 뜬다 (Gym `start_new_session=True`). starter · 컨테이너 프로세스 · FUSE 데몬 (squashfuse_ll · fuse-overlayfs) 이
  세션 ID 를 공유하므로 (실측 11-24) 먼저 세션 리더의 프로세스 그룹에 SIGTERM 을 보내 starter 가 마운트를 풀게 하고, 남은 세션 프로세스는 SIGKILL 한다.
Gym 서버나 실행기를 내려도 컨테이너는 남는다 (채점 컨테이너는 에이전트 출력 파일을 기다리며 끝없이 돈다) — 그래서 실행기가 끝날 때 부른다.
출력: SWE_CONTAINERS sessions=<처음 찾은 컨테이너 세션 수> left=<끝난 뒤 남은 세션 수>"""
import os, signal, sys, time
names = [os.path.basename(l.strip().rstrip("/")) for l in open(sys.argv[1]) if l.strip()]
KILL = len(sys.argv) > 2 and sys.argv[2] == "kill"
MY_SID = os.getsid(0)
def stat(pid):   # (pgid, sid)
    f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
    return int(f[2]), int(f[3])
def sessions():
    s = set()
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            if any(n in open(f"/proc/{pid}/mountinfo").read() for n in names):
                s.add(stat(pid)[1])
        except (OSError, IndexError): pass
    s.discard(MY_SID); s.discard(0)
    return s
def members(sids):
    out = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            if stat(pid)[1] in sids: out.append(int(pid))
        except (OSError, IndexError): pass
    return out
first = sessions() if names else set()
if KILL and first:
    for sid in first:                      # 세션 리더 = starter. 그 그룹 (starter · 컨테이너 프로세스) 에 SIGTERM
        try: os.killpg(sid, signal.SIGTERM)
        except OSError: pass
    for _ in range(10):
        time.sleep(1)
        if not members(first): break
    for pid in members(first):             # 남은 것 (FUSE 데몬 포함) 은 SIGKILL
        try: os.kill(pid, signal.SIGKILL)
        except OSError: pass
    time.sleep(1)
left = {s for s in first if members({s})} | sessions() if names else set()
print(f"SWE_CONTAINERS sessions={len(first)} left={len(left)}")
