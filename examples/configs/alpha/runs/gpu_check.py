"""Ray 클러스터 각 노드의 물리 GPU 점유(MiB)를 출력 — 기동 전 점유 검사(1 GiB) 용. sub1 은 ssh 가 막혀 Ray 태스크로 본다."""
import subprocess, ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
ray.init(address="auto", logging_level="ERROR")
@ray.remote(num_cpus=0.01)
def smi():
    import socket
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return socket.gethostname(), [int(l.split(",")[1]) for l in out.strip().splitlines()]
busy = 0
for n in ray.nodes():
    if not n["Alive"]:
        continue
    host, mem = ray.get(smi.options(scheduling_strategy=NodeAffinitySchedulingStrategy(n["NodeID"], soft=False)).remote())
    print(n["NodeManagerAddress"], host, mem)
    busy += sum(m > 1024 for m in mem)
print("BUSY_GPUS", busy)
