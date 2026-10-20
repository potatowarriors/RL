"""Code·Math 3단계 블렌드 생성 (RL_PLAN.md 결정 39, 사용자 결정 2026-10-19/20 — DAPO 동적 샘플링으로 전체 후보를 학습).

전체 후보 풀(teacher_pool/{math,code}_{measure,reserve}.jsonl, 53,377문제)을 전부 넣고 다음만 뺀다.
  - 2단계 held-out 검증셋(teacher_codemath_s2_v1_val.jsonl, 256) — 3단계도 같은 검증셋으로 비교한다.
  - 2단계 롤아웃 기록(rollout_groups.jsonl)에서 마지막 방문이 16/16 인 문제 — 다 푼 문제. 0/16 은 뺴지 않는다 (학습 중 풀리기 시작할 수 있다).
code 문제 중 마지막 방문 정답이 1~2/16 인 것은 두 번 넣는다 (G=32 효과, 2단계와 같은 규칙). 측정값 없는 예비 풀은 한 번.
행에 stage3_source (s2_log / p04 / reserve) · stage3_last_correct (마지막 방문 정답 수 또는 null) 을 덧붙인다.
학습 중 전부 정답·전부 오답 그룹은 DAPO 필터(grpo.async_grpo.drop_zero_std_groups)가 버린다.

사용:
  python examples/configs/alpha/tools/build_codemath_stage3_blend.py \
      --out /home/work/Datasets/LL_datasets/posttraining/RL/alpha_blends/teacher_blends/teacher_codemath_s3_v1.jsonl
결과: <out> · <out>_stats.json (검증셋은 2단계 _val.jsonl 을 그대로 쓴다)
"""

import argparse
import collections
import json
import random

POOL = "/home/work/Datasets/LL_datasets/posttraining/RL/alpha_blends/teacher_pool"
BLENDS = "/home/work/Datasets/LL_datasets/posttraining/RL/alpha_blends/teacher_blends"
S2_LOG = "/home/work/vidsearch/repos/project_s/NeMo-RL/results/alpha/teacher_codemath_s2/s1/rollout_groups.jsonl"


def iter_pool():
    for name in ("math_measure", "math_reserve", "code_measure", "code_reserve"):
        with open(f"{POOL}/{name}.jsonl") as f:
            for line in f:
                yield name, json.loads(line)


def last_visit(log_path):
    """pool_key -> (correct, n) of the most recent visit (highest generation weight version)."""
    last = {}
    with open(log_path) as f:
        for line in f:
            r = json.loads(line)
            k = r.get("pool_key")
            if k is None:
                continue
            v = r["generation_weight_version"]
            if k not in last or v >= last[k][0]:
                last[k] = (v, sum(1 for x in r["rewards"] if x > 0), len(r["rewards"]))
    return {k: (c, n) for k, (v, c, n) in last.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--s2-log", default=S2_LOG)
    ap.add_argument("--val", default=f"{BLENDS}/teacher_codemath_s2_v1_val.jsonl")
    ap.add_argument("--passmap", default=f"{POOL}/p04_passmap.jsonl")
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--dup-code-max-correct", type=int, default=2)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    val_keys = {json.loads(l)["pool_key"] for l in open(a.val)}
    visited = last_visit(a.s2_log)
    p04 = {json.loads(l)["key"] for l in open(a.passmap)}

    rows, stats = [], collections.Counter()
    seen = set()
    for name, r in iter_pool():
        k = r["pool_key"]
        if k in seen:
            stats["duplicate_key_skipped"] += 1
            continue
        seen.add(k)
        dom = "code" if "code" in r["agent_ref"]["name"] else "math"
        if k in val_keys:
            stats[f"{dom}_excluded_val"] += 1
            continue
        lv = visited.get(k)
        if lv is not None:
            c, n = lv
            if c == n:
                stats[f"{dom}_excluded_all_correct_16"] += 1
                continue
            src = "s2_log"
        else:
            src = "p04" if k in p04 else "reserve"
        row = dict(r)
        row["stage3_source"] = src
        row["stage3_last_correct"] = lv[0] if lv else None
        rows.append(row)
        stats[f"{dom}_{src}"] += 1
        if dom == "code" and lv is not None and lv[0] <= a.dup_code_max_correct:
            dup = dict(row)
            dup["stage3_source"] = src + "_dup"
            rows.append(dup)
            stats["code_duplicated"] += 1

    rng.shuffle(rows)
    with open(a.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    agents = collections.Counter(r["agent_ref"]["name"] for r in rows)
    out = {
        "seed": a.seed,
        "train_rows": len(rows),
        "train_unique_prompts": len({r["pool_key"] for r in rows}),
        "by_agent": dict(agents),
        "by_source": dict(collections.Counter(r["stage3_source"] for r in rows)),
        "counts": dict(sorted(stats.items())),
        "val_path": a.val,
        "val_rows": len(val_keys),
        "val_train_overlap": len(val_keys & {r["pool_key"] for r in rows}),
        "steps_per_epoch_at_256": len(rows) // 256,
    }
    with open(a.out.replace(".jsonl", "_stats.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
