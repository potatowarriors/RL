"""런 중 Ray 배치 그룹(placement group)의 bundle → 노드를 읽어 역할별 GPU 배치를 판정한다 (MS1 ② 적대적 검토 H3, 2026-11-09).

nvidia-smi pid 는 컨테이너 밖 pid 라 Ray 액터 pid 와 맞지 않는다 (sample_gpu_layout.py 의 한계). 그래서 배치 그룹 이름으로 역할을 가린다:
  grpo_train_cluster*      학습 (Megatron)       — 한 노드에 GPU 8
  grpo_inference_cluster*  생성 (vLLM)           — 한 노드에 GPU --gen-gpus
  teacher_*                OPD teacher           — 그룹마다 GPU 1, 생성과 같은 노드
판정: 학습 노드 ≠ 생성 노드, teacher 전부가 생성 노드, 노드별 GPU 합 ≤ 8. 학습이 main1 이든 sub1 이든 (Ray 가 정한다) 둘 다 허용한다.

--actors-out 를 주면 teacher 액터 (이름 teacher_<alias>-…, pid, 노드 IP) 를 JSON 으로 남긴다 — analyze_mopd_smoke.py --teacher-actors 가
드라이버 로그의 `successfully loaded checkpoint from …` 줄과 (노드, pid) 로 맞춰 teacher 신원을 본다 (MS2 ②, 검토 3차).

  $NRL_ROOT/clean_run.sh /usr/bin/env RAY_ADDRESS=10.0.37.4:6379 $NRL_ROOT/venv-driver/bin/python -I \
      examples/configs/alpha/runs/check_placement.py [--gen-gpus 4] [--teachers 4] [--actors-out <file.json>] </dev/null
종료 코드 0 = PASS.
"""

import argparse
import collections
import json
import sys

import ray
from ray.util.state import list_actors, list_placement_groups


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-gpus", type=int, default=4)
    ap.add_argument("--teachers", type=int, default=4)
    ap.add_argument("--actors-out")
    a = ap.parse_args()
    ray.init(address="auto", logging_level="ERROR")
    ip = {n["NodeID"]: n["NodeManagerAddress"] for n in ray.nodes() if n["Alive"]}
    roles = collections.defaultdict(lambda: collections.Counter())  # role -> node ip -> GPUs
    teacher_groups = []
    for pg in list_placement_groups(detail=True, limit=10000):
        if pg.state != "CREATED":
            continue
        name = pg.name or ""
        role = ("train" if name.startswith("grpo_train_cluster") else "gen" if name.startswith("grpo_inference_cluster")
                else "teacher" if name.startswith("teacher_") else None)
        if role is None:
            continue
        gpus = 0
        for b in pg.bundles or []:
            g = (b.get("unit_resources") or {}).get("GPU", 0)
            node = ip.get(b.get("node_id"), b.get("node_id"))
            roles[role][node] += g
            gpus += g
        if role == "teacher":
            teacher_groups.append((name, gpus))
        print(f"{role:8s} {name:60s} " + ", ".join(f"{ip.get(b.get('node_id'), '?')}:{(b.get('unit_resources') or {}).get('GPU', 0):g}" for b in pg.bundles or []))
    ok = True

    def fail(msg):
        nonlocal ok
        ok = False
        print("FAIL", msg)

    train_nodes = {n for n, g in roles["train"].items() if g}
    gen_nodes = {n for n, g in roles["gen"].items() if g}
    teacher_nodes = {n for n, g in roles["teacher"].items() if g}
    if len(train_nodes) != 1 or sum(roles["train"].values()) != 8:
        fail(f"학습 배치 {dict(roles['train'])} — 한 노드에 GPU 8 이어야 한다")
    if len(gen_nodes) != 1 or sum(roles["gen"].values()) != a.gen_gpus:
        fail(f"생성 배치 {dict(roles['gen'])} — 한 노드에 GPU {a.gen_gpus}")
    if train_nodes & gen_nodes:
        fail("학습과 생성이 같은 노드")
    if len(teacher_groups) != a.teachers or any(g != 1 for _, g in teacher_groups):
        fail(f"teacher 그룹 {teacher_groups} — {a.teachers} 개, 각 GPU 1")
    if teacher_nodes != gen_nodes:
        fail(f"teacher 노드 {teacher_nodes} ≠ 생성 노드 {gen_nodes}")
    per_node = collections.Counter()
    for r in roles.values():
        per_node.update(r)
    if any(g > 8 for g in per_node.values()):
        fail(f"노드별 GPU 합 {dict(per_node)} > 8")
    if a.actors_out:
        actors = [{"name": x.name, "pid": x.pid, "node_ip": ip.get(x.node_id, x.node_id), "class": x.class_name}
                  for x in list_actors(filters=[("state", "=", "ALIVE")], limit=10000, detail=True) if (x.name or "").startswith("teacher_")]
        with open(a.actors_out, "w") as f:
            json.dump(actors, f, indent=1)
        print(f"teacher actors {len(actors)} → {a.actors_out}")
        if len(actors) != a.teachers:
            fail(f"ALIVE teacher 액터 {len(actors)} 개 ≠ {a.teachers}")
    print("placement:", {r: dict(v) for r, v in roles.items()}, "→", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
