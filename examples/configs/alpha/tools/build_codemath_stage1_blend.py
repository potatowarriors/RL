"""Code·Math 1단계(32K) 블렌드 — RL_PLAN.md 결정 32 (사용자 2026-10-14).

규칙
  수학: P0-3 32K 측정(12,000 × 4)에서 4회 중 1회 이상 맞힌 문제 전부 + 같은 수의 0/4 문제.
        0/4 는 "4회 모두 상한에 잘림" 에서만 뽑고 (끝냈는데 틀린 문제는 지식 문제라 제외), 원천(dataset) 비율을 유지한 무작위.
        참조 pass_rate 순서는 쓰지 않는다 — alpha 성패를 가르지 못했다 (Nano 4~15/16 전 구간 18~21%).
  코드: 측정을 중간(3,657 롤아웃)에 멈췄다 (사용자 2026-10-14 16:35 — 12시간 더 재서 수백 문제 더 찾는 것은 비효율).
        이미 1회 이상 맞힌 문제 전부 + Nano tier-1 행을 Nano SFT 참조 pass_rate 높은 순으로 채워 수학과 같은 수.
        측정에서 4회 모두 끝내고도 틀린 문제는 채움 후보에서 뺀다.
  수학·코드 행 수를 같게 두고 전체를 섞어 step 마다 draw 가 1:1 이 되게 한다. 검증셋은 예비 풀에서 128 + 128.

실행 (CPU):
  python examples/configs/alpha/tools/build_codemath_stage1_blend.py \
      --out /home/work/Datasets/LL_datasets/posttraining/RL/alpha_blends/teacher_blends/teacher_codemath_s1_v1.jsonl
"""

import argparse
import collections
import json
import os
import random

POOL = "/home/work/Datasets/LL_datasets/posttraining/RL/alpha_blends/teacher_pool"
MEASURE = "/home/work/vidsearch/tools/nemo_rl/p0_measure"
NANO_CODE = "nano_v3_sft_profiled_comp_coding_50tests"


def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def rollout_buckets(path):
    """task index -> list of (reward, status) over the measured rollouts."""
    per = collections.defaultdict(list)
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            resp = d.get("response") or {}
            per[int(d["_ng_task_index"])].append(
                (float(d.get("reward") or 0.0), resp.get("status"))
            )
    return per


def stratified_sample(indices, key, n, rng):
    """Sample n indices keeping the proportion of key(index) groups (largest remainder)."""
    groups = collections.defaultdict(list)
    for i in indices:
        groups[key(i)].append(i)
    total = len(indices)
    quotas = {g: n * len(v) / total for g, v in groups.items()}
    base = {g: int(q) for g, q in quotas.items()}
    rest = n - sum(base.values())
    for g, _ in sorted(quotas.items(), key=lambda kv: -(kv[1] - int(kv[1])))[:rest]:
        base[g] += 1
    out = []
    for g, v in groups.items():
        v = list(v)
        rng.shuffle(v)
        out.extend(v[: base[g]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--val-per-domain", type=int, default=128)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    # ---- math ----
    math_rows = load_rows(f"{POOL}/math_measure.jsonl")
    mb = rollout_buckets(f"{MEASURE}/p03_math32/rollouts.jsonl")
    math_solved, math_zero_trunc = [], []
    for i, v in mb.items():
        if len(v) < 4:
            continue
        correct = sum(1 for r, _ in v if r >= 1)
        if correct >= 1:
            math_solved.append((i, correct))
        elif all(s == "incomplete" for _, s in v):
            math_zero_trunc.append(i)
    n_math = len(math_solved)
    zero_pick = stratified_sample(
        math_zero_trunc, lambda i: math_rows[i].get("dataset"), n_math, rng
    )
    math_out = []
    for i, c in math_solved:
        r = dict(math_rows[i])
        r["stage1_bucket"] = f"math_solved_{c}of4"
        math_out.append(r)
    for i in zero_pick:
        r = dict(math_rows[i])
        r["stage1_bucket"] = "math_zero_all_truncated"
        math_out.append(r)

    # ---- code ----
    code_rows = load_rows(f"{POOL}/code_measure.jsonl")
    cb = rollout_buckets(f"{MEASURE}/p03_code32/rollouts.jsonl")
    code_solved, code_exclude = [], set()
    for i, v in cb.items():
        correct = sum(1 for r, _ in v if r >= 1)
        if correct >= 1:
            code_solved.append((i, correct, len(v)))
        elif len(v) >= 4 and any(s == "completed" for _, s in v):
            code_exclude.add(i)  # 끝내고도 틀림 — 지식 문제
    solved_set = {i for i, _, _ in code_solved}
    fill_pool = [
        i
        for i, r in enumerate(code_rows)
        if r.get("dataset") == NANO_CODE
        and r.get("ref_pass_rate_nano_sft") is not None
        and i not in solved_set
        and i not in code_exclude
    ]
    rng.shuffle(fill_pool)  # 동률 무작위
    fill_pool.sort(key=lambda i: -float(code_rows[i]["ref_pass_rate_nano_sft"]))
    n_fill = len(math_out) - len(code_solved)
    assert n_fill <= len(fill_pool), (n_fill, len(fill_pool))
    code_out = []
    for i, c, n in code_solved:
        r = dict(code_rows[i])
        r["stage1_bucket"] = f"code_solved_{c}of{n}"
        code_out.append(r)
    for i in fill_pool[:n_fill]:
        r = dict(code_rows[i])
        r["stage1_bucket"] = "code_fill_nano_pass_rate"
        code_out.append(r)
    assert len(code_out) == len(math_out)

    rows = math_out + code_out
    rng.shuffle(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- validation set from the reserves (never trained on) ----
    val = []
    for dom in ("math", "code"):
        res = load_rows(f"{POOL}/{dom}_reserve.jsonl")
        idx = list(range(len(res)))
        rng.shuffle(idx)
        for i in idx[: a.val_per_domain]:
            r = dict(res[i])
            r["stage1_bucket"] = f"{dom}_val_reserve"
            val.append(r)
    rng.shuffle(val)
    val_path = a.out.replace(".jsonl", "_val.jsonl")
    with open(val_path, "w") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "seed": a.seed,
        "rows": len(rows),
        "math": len(math_out),
        "code": len(code_out),
        "by_bucket": dict(collections.Counter(r["stage1_bucket"] for r in rows)),
        "by_agent": dict(
            collections.Counter(r["agent_ref"]["name"] for r in rows)
        ),
        "math_by_dataset": dict(
            collections.Counter(r.get("dataset") for r in math_out)
        ),
        "code_by_dataset": dict(
            collections.Counter(r.get("dataset") for r in code_out)
        ),
        "math_zero_trunc_available": len(math_zero_trunc),
        "code_fill_pool_available": len(fill_pool),
        "code_fill_min_nano_pass_rate": (
            float(code_rows[fill_pool[n_fill - 1]]["ref_pass_rate_nano_sft"])
            if n_fill
            else None
        ),
        "code_excluded_completed_wrong": len(code_exclude),
        "code_rollouts_measured": int(sum(len(v) for v in cb.values())),
        "val_rows": len(val),
        "val_path": val_path,
    }
    with open(a.out + ".stats.json", "w") as f:
        json.dump(stats, f, indent=1, ensure_ascii=False)
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
