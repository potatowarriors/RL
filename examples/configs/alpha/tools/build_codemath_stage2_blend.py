"""Code·Math 2단계 블렌드 생성 (RL_PLAN.md 결정 38, 사용자 승인 2026-10-19).

P0-4 난이도 지도(`teacher_pool/p04_passmap.jsonl`, 1단계 결과 hf_step_00282 로 프롬프트당 8회 @32K)에서
학습 구간(정답 1~7/8)만 고른다. 0/8(못 품)·8/8(다 품)은 advantage 가 0 이라 뺀다.
  - 검증셋: 학습 구간에서 도메인당 --val-per-domain 개를 무작위로 떼어 학습에서 제외한다.
  - code 하위 구간(정답 1~2/8)은 학습 파일에 두 번 넣는다 — 프롬프트당 샘플 32개(G=32)와 같은
    "양성 예시를 얻을 확률" 을 생성 비용 2배로 얻는다 (사용자 2026-10-18 논의, 결정 38).
  - 행 내용은 측정 풀(`teacher_pool/{math,code}_measure.jsonl`)의 행 그대로이고, 측정값을 `stage2_k8` 로 덧붙인다.

사용:
  python examples/configs/alpha/tools/build_codemath_stage2_blend.py \
      --out /home/work/Datasets/LL_datasets/posttraining/RL/alpha_blends/teacher_blends/teacher_codemath_s2_v1.jsonl
결과: <out> · <out>_val.jsonl · <out>_stats.json
"""

import argparse
import collections
import json
import random

POOL = "/home/work/Datasets/LL_datasets/posttraining/RL/alpha_blends/teacher_pool"


def load_pool_rows():
    rows = {}
    for name in ("math", "code"):
        with open(f"{POOL}/{name}_measure.jsonl") as f:
            for line in f:
                r = json.loads(line)
                rows[r["pool_key"]] = r
    return rows


def load_passmap(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--passmap", default=f"{POOL}/p04_passmap.jsonl")
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--val-per-domain", type=int, default=128)
    ap.add_argument("--dup-code-max-k", type=int, default=2, help="code 프롬프트 중 정답 k/8 ≤ 이 값이면 두 번 넣는다")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    pool = load_pool_rows()
    pm = load_passmap(a.passmap)
    assert all(r["n"] == 8 for r in pm), "pass map must have 8 samples per prompt"
    assert all(r["key"] in pool for r in pm), "pass map key missing from the measure pools"

    frontier = collections.defaultdict(list)
    bands = collections.defaultdict(collections.Counter)
    for r in pm:
        k = r["k"]
        band = "zero" if k == 0 else ("all" if k == 8 else "frontier")
        bands[r["agent"]][band] += 1
        if band == "frontier":
            frontier[r["agent"]].append(r)

    train, val = [], []
    dup_count = 0
    for agent in ("math", "code"):
        items = sorted(frontier[agent], key=lambda r: r["key"])  # 결정적 순서 뒤 시드 셔플
        rng.shuffle(items)
        held, rest = items[: a.val_per_domain], items[a.val_per_domain :]
        for r in held:
            row = dict(pool[r["key"]])
            row["stage2_k8"] = r["k"]
            row["stage2_split"] = f"{agent}_val"
            val.append(row)
        for r in rest:
            row = dict(pool[r["key"]])
            row["stage2_k8"] = r["k"]
            row["stage2_split"] = f"{agent}_train"
            train.append(row)
            if agent == "code" and r["k"] <= a.dup_code_max_k:
                dup = dict(row)
                dup["stage2_split"] = f"{agent}_train_dup"
                train.append(dup)
                dup_count += 1

    rng.shuffle(train)
    rng.shuffle(val)
    with open(a.out, "w") as f:
        for row in train:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    val_path = a.out.replace(".jsonl", "_val.jsonl")
    with open(val_path, "w") as f:
        for row in val:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def kdist(rows):
        return dict(sorted(collections.Counter(r["stage2_k8"] for r in rows).items()))

    stats = {
        "seed": a.seed,
        "passmap": a.passmap,
        "bands_by_agent": {k: dict(v) for k, v in bands.items()},
        "train_rows": len(train),
        "train_unique_prompts": len({r["pool_key"] for r in train}),
        "train_by_split": dict(collections.Counter(r["stage2_split"] for r in train)),
        "train_by_agent": dict(collections.Counter(r["agent_ref"]["name"] for r in train)),
        "train_k8_math": kdist([r for r in train if "math" in r["agent_ref"]["name"]]),
        "train_k8_code": kdist([r for r in train if "code" in r["agent_ref"]["name"]]),
        "code_duplicated": dup_count,
        "val_rows": len(val),
        "val_by_agent": dict(collections.Counter(r["agent_ref"]["name"] for r in val)),
        "val_path": val_path,
        "val_train_overlap": len({r["pool_key"] for r in val} & {r["pool_key"] for r in train}),
    }
    with open(a.out.replace(".jsonl", "_stats.json"), "w") as f:
        json.dump(stats, f, indent=1, ensure_ascii=False)
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
