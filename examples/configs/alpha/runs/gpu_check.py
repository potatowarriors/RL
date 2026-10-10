"""Ray 클러스터 각 노드의 물리 GPU 점유(MiB)를 출력 — 기동 전 점유 검사(1 GiB) 용. sub1 은 ssh 가 막혀 Ray 태스크로 본다.

NRL_RUN_ACTORS 는 살아 있는 NeMo-RL 런 액터 수다. 0 이면 남은 GPU 점유는 지난 런의 Gym judge(local_vllm_model)이고 정리해도 된다
(2026-10-10 — judge 는 Gym 서버가 띄운 Ray 액터라 드라이버가 끝나도 GPU 를 잡은 채 남는다).
"""
import subprocess, ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.state import list_actors
ray.init(address="auto", logging_level="ERROR")
RUN_CLASSES = {"MegatronPolicyWorker", "DTensorPolicyWorker", "DTensorPolicyWorkerV2", "VllmGenerationWorker",
               "VllmAsyncGenerationWorker", "AsyncTrajectoryCollector", "ReplayBuffer", "NemoGym"}
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
run = [a.class_name for a in list_actors(filters=[("state", "=", "ALIVE")], limit=10000) if a.class_name in RUN_CLASSES]
print("NRL_RUN_ACTORS", len(run), sorted(set(run)))
print("BUSY_GPUS", busy)
