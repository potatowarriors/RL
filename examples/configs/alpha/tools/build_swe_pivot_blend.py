"""SWE 피벗 teacher 블렌드 (RL_PLAN.md 결정 59 · 61, RL_DATA.md §6) — 고유 피벗만, 인스턴스 단위 검증 분리 (CPU, 결정론적).

원천은 Super `swe1.jsonl` 이다. `Nemotron-RL-Agentic-SWE-Pivot-v1/train.jsonl` 과 같은 50,661 행인데 agent 이름이 Gym 설정에 있는
`swe_pivot_single_step_tool_use_with_argument_comparison_agent` 다 (Pivot-v1 원본의 `..._swe` 는 Gym 설정에 없다).
같은 피벗이 평균 12.7 회 반복되고 반복마다 `pass_rate` 만 다르다 — 입력 · 도구 정의 · 기대 행동의 해시로 고유 피벗 하나만 남긴다.
남긴 행의 `pass_rate` 는 반복들의 평균으로 바꾸고 원래 값들은 `pass_rate_repeats` 에 둔다 (학습 경로는 이 필드를 읽지 않는다).
검증셋은 인스턴스(`metadata.instance_id`) 단위로 뗀다 — 같은 저장소 이슈의 다른 harness · 다른 턴이 학습에 남지 않게 한다.
출력 옆에 `.stats.json` 을 남긴다. `--seen` 을 주면 그 블렌드들이 이미 쓴 피벗 수를 통계에 적는다.

  python examples/configs/alpha/tools/build_swe_pivot_blend.py --out <dir>/teacher_swe_pivot_v1.jsonl \
      [--n-val 128] [--seed 20261112] [--seen <teacher_tool_v1.jsonl> <teacher_tool_v1_val.jsonl>]
"""

import argparse
import collections
import hashlib
import json
import os
import random

RL = "/home/work/Datasets/LL_datasets/posttraining/RL"
SOURCE = f"{RL}/Nemotron-RL-Super-Training-Blends/swe1.jsonl"
SWE = "swe_pivot_single_step_tool_use_with_argument_comparison_agent"


def task_key(row: dict) -> str:
    rcp = row.get("responses_create_params") or {}
    blob = json.dumps([rcp.get("input"), rcp.get("tools"), row.get("expected_action")], sort_keys=True)
    return hashlib.md5(blob.encode()).hexdigest()


def seen_keys(paths: list[str]) -> set[str]:
    keys = set()
    for path in paths:
        with open(path) as f:
            for line in f:
                if SWE not in line:
                    continue
                row = json.loads(line)
                if (row.get("agent_ref") or {}).get("name") == SWE:
                    keys.add(task_key(row))
    return keys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-val", type=int, default=128, help="검증 피벗 수의 하한 (인스턴스 단위로 채운다)")
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--seen", nargs="*", default=[])
    a = ap.parse_args()
    out_val = a.out.replace(".jsonl", "_val.jsonl")
    for path in (a.out, out_val):
        if os.path.exists(path):
            print(f"출력이 이미 있다: {path} — 지우고 다시 실행한다")
            return 1

    pool: dict[str, dict] = {}
    repeats = collections.defaultdict(list)
    source_rows = 0
    dropped = collections.Counter()
    with open(SOURCE) as f:
        for line in f:
            row = json.loads(line)
            source_rows += 1
            if (row.get("agent_ref") or {}).get("name") != SWE:
                dropped["other_agent"] += 1
                continue
            if not (row.get("metadata") or {}).get("instance_id"):
                dropped["no_instance_id"] += 1
                continue
            if (row.get("expected_action") or {}).get("type") != "function_call":
                dropped["expected_action_not_function_call"] += 1
                continue
            key = task_key(row)
            pool.setdefault(key, row)
            repeats[key].append(row.get("pass_rate"))

    by_instance = collections.defaultdict(list)
    for key in sorted(pool):
        row = pool[key]
        rates = [r for r in repeats[key] if r is not None]
        row["pass_rate_repeats"] = rates
        if rates:
            row["pass_rate"] = sum(rates) / len(rates)
        by_instance[row["metadata"]["instance_id"]].append(key)

    rng = random.Random(a.seed)
    instances = sorted(by_instance)
    rng.shuffle(instances)
    val_keys: list[str] = []
    val_instances = []
    for inst in instances:
        if len(val_keys) >= a.n_val:
            break
        val_instances.append(inst)
        val_keys += by_instance[inst]
    val_set = set(val_keys)
    train_keys = [k for k in sorted(pool) if k not in val_set]
    rng.shuffle(train_keys)
    rng.shuffle(val_keys)

    seen = seen_keys(a.seen) if a.seen else None
    for path, keys in ((a.out, train_keys), (out_val, val_keys)):
        with open(path, "w") as fo:
            for k in keys:
                fo.write(json.dumps(pool[k], ensure_ascii=False) + "\n")
        rows = [pool[k] for k in keys]
        st = {
            "file": path,
            "rows": len(rows),
            "instances": len({r["metadata"]["instance_id"] for r in rows}),
            "by_harness": collections.Counter(r["metadata"].get("agent_cls") for r in rows),
            "by_expected_tool": collections.Counter(r["expected_action"].get("name") for r in rows),
            "mean_ref_pass_rate": round(sum(r["pass_rate"] for r in rows) / len(rows), 4),
            "source": SOURCE,
            "source_rows": source_rows,
            "unique_pool": len(pool),
            "dropped": dropped,
            "seed": a.seed,
        }
        if seen is not None:
            st["seen_by"] = a.seen
            st["rows_seen"] = sum(k in seen for k in keys)
        with open(path + ".stats.json", "w") as fs:
            json.dump(st, fs, indent=1, ensure_ascii=False)
        print(json.dumps(st, ensure_ascii=False))
    overlap = {pool[k]["metadata"]["instance_id"] for k in train_keys} & set(val_instances)
    print(f"학습-검증 인스턴스 겹침: {len(overlap)}")
    return 1 if overlap else 0


if __name__ == "__main__":
    raise SystemExit(main())
