"""런 중 노드별 GPU 배치·메모리 표본 (MS1 ② ⑤, 2026-11-09) — Ray 태스크로 각 노드에서 nvidia-smi 를 읽는다 (sub1 은 ssh 가 막혀 있다).

한 표본 = 노드마다 {GPU 번호: 사용 MiB, 그 GPU 의 프로세스 pid 와 Ray 액터 이름} + 호스트 RAM 사용 GiB. JSONL 한 줄씩 --out 에 붙인다.
액터 이름은 Ray 상태 API(list_actors: pid · node · class_name · name)로 pid 와 맞춘다 — teacher 워커는 이름이 `teacher_<alias>` 로 시작한다
(teacher_worker_group.py RayWorkerGroup name_prefix).

  $NRL_ROOT/clean_run.sh /usr/bin/env RAY_ADDRESS=10.0.37.4:6379 $NRL_ROOT/venv-driver/bin/python -I \
      examples/configs/alpha/runs/sample_gpu_layout.py --out <file.jsonl> [--interval 10] [--count 0] </dev/null
"""

import argparse
import json
import subprocess
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.state import list_actors


@ray.remote(num_cpus=0.01)
def probe():
    import socket

    gpus = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip().splitlines()
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip().splitlines()
    uuid2idx, out = {}, {}
    for line in gpus:
        idx, uuid, mem = [x.strip() for x in line.split(",")]
        uuid2idx[uuid] = idx
        out[idx] = {"mem_mib": int(mem), "procs": []}
    for line in apps:
        if not line.strip():
            continue
        pid, uuid, mem = [x.strip() for x in line.split(",")]
        if uuid in uuid2idx:
            out[uuid2idx[uuid]]["procs"].append({"pid": int(pid), "mem_mib": int(mem)})
    meminfo = dict(line.split(":", 1) for line in open("/proc/meminfo").read().splitlines() if ":" in line)
    used_gib = (int(meminfo["MemTotal"].split()[0]) - int(meminfo["MemAvailable"].split()[0])) / 2**20
    return socket.gethostname(), out, round(used_gib, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--count", type=int, default=0, help="0 = 끝없이")
    a = ap.parse_args()
    ray.init(address="auto", logging_level="ERROR")
    i = 0
    while a.count == 0 or i < a.count:
        i += 1
        try:
            actors = {x.pid: {"class": x.class_name, "name": x.name} for x in list_actors(filters=[("state", "=", "ALIVE")], limit=10000, detail=True)
                      if getattr(x, "pid", None)}
        except Exception as e:  # noqa: BLE001 — 상태 API 가 잠깐 실패해도 표본은 남긴다
            actors = {"error": str(e)[:200]}
        rec = {"t": time.strftime("%F %T"), "nodes": {}}
        for n in ray.nodes():
            if not n["Alive"]:
                continue
            host, gpus, ram = ray.get(probe.options(scheduling_strategy=NodeAffinitySchedulingStrategy(n["NodeID"], soft=False)).remote())
            for g in gpus.values():
                for p in g["procs"]:
                    p.update(actors.get(p["pid"], {}) if isinstance(actors, dict) else {})
            rec["nodes"][n["NodeManagerAddress"]] = {"host": host, "ram_used_gib": ram, "gpus": gpus}
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        time.sleep(a.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
