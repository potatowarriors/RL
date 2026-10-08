"""Ray 클러스터 두 노드에 남은 NeMo-Gym 서버 프로세스(`python app.py`)를 찾아(인자 kill 이면 종료) 출력한다 (2026-10-08).
Gym 서버는 별도 Ray 작업으로 떠서 NeMo-RL 드라이버가 끝나도 남는다 — 첫 런 전까지 G5·G7 런의 잔여 132개가 쌓였다.
runs/launch.sh 는 GPU 점유 검사(런 없음)를 통과한 뒤에만 이걸 kill 로 부른다."""
import os, signal, subprocess, sys
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
ray.init(address="auto", logging_level="ERROR")
KILL = len(sys.argv) > 1 and sys.argv[1] == "kill"
@ray.remote(num_cpus=0.01)
def scan(kill):
    import socket
    rows = subprocess.run(["ps", "-eo", "pid,etime,args"], capture_output=True, text=True).stdout.splitlines()[1:]
    hits = [r for r in rows if r.split(None, 2)[-1].strip() == "python app.py"]
    killed = 0
    if kill:
        for r in hits:
            try:
                os.kill(int(r.split()[0]), signal.SIGTERM); killed += 1
            except Exception:
                pass
    return socket.gethostname(), len(hits), killed
total = 0
for n in ray.nodes():
    if n["Alive"]:
        host, found, killed = ray.get(scan.options(scheduling_strategy=NodeAffinitySchedulingStrategy(n["NodeID"], soft=False)).remote(KILL))
        total += found
        print(f"GYM_LEFTOVER {host} found={found} killed={killed}")
print(f"GYM_LEFTOVER_TOTAL {total}")
