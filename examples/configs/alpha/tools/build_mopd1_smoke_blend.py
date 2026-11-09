"""MOPD1 스모크 블렌드 (게이트 MS1, RL_PLAN.md 결정 55) — CPU, 결정론적.

본 블렌드 mopd1_v1.jsonl 에서 teacher 몫 비율대로 --n 행을 층화 추출하고, teacher 1장 메모리 탐침용으로
teacher_tool_v1 에서 렌더 프롬프트가 --long-min~--long-max 토큰인 swe_pivot 행 --n-long 개를 더한다
(본 블렌드는 32,768 상한으로 이 행들을 뺐다 — 생성 상한은 vLLM 이 max_model_len − 프롬프트로 줄인다).
검증 파일은 mopd1_v1_val.jsonl 의 앞 --n-val 행.

  $NRL_ROOT/clean_run.sh $NRL_ROOT/venv-driver/bin/python -I examples/configs/alpha/tools/build_mopd1_smoke_blend.py \
      --out <dir>/mopd1_smoke_v1.jsonl [--n 190] [--n-long 2] [--seed 20261109] </dev/null
"""

import argparse
import collections
import json
import os
import random
import sys

import orjson

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_general_blend as G  # noqa: E402
import build_mopd1_blend as B  # noqa: E402
import measure_blend_prompt_lengths as M  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default=f"{G.AB}/mopd_blends/mopd1_v1.jsonl")
    ap.add_argument("--n", type=int, default=190)
    ap.add_argument("--n-long", type=int, default=2)
    ap.add_argument("--long-min", type=int, default=56000)
    ap.add_argument("--long-max", type=int, default=60000)
    ap.add_argument("--n-val", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261109)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    agent_teacher = {}
    for share, (teacher, files, _) in B.SHARES.items():
        for f in files:
            for line in open(f):
                agent_teacher.setdefault(orjson.loads(line)["agent_ref"]["name"], teacher)
    by_teacher = collections.defaultdict(list)
    for line in open(a.src):
        r = orjson.loads(line)
        by_teacher[agent_teacher[r["agent_ref"]["name"]]].append(r)
    total = sum(len(v) for v in by_teacher.values())
    rows = []
    for t, v in sorted(by_teacher.items()):
        k = round(a.n * len(v) / total)
        rows += rng.sample(v, k)
    rows = rows[: a.n]

    M._init(G.SFT_HF)
    long_rows = []
    for line in open(f"{B.TB}/teacher_tool_v1.jsonl"):
        r = orjson.loads(line)
        if r["agent_ref"]["name"] != "swe_pivot_single_step_tool_use_with_argument_comparison_agent":
            continue
        _, n, failed = M._one(line)
        if a.long_min <= n <= a.long_max and not failed:
            r["_smoke_prompt_tokens"] = n
            long_rows.append(r)
        if len(long_rows) >= a.n_long:
            break
    if len(long_rows) < a.n_long:
        raise SystemExit(f"only {len(long_rows)} long rows in [{a.long_min}, {a.long_max}]")
    long_tokens = [r.pop("_smoke_prompt_tokens") for r in long_rows]
    rows += long_rows
    rng.shuffle(rows)
    with open(a.out, "wb") as f:
        for r in rows:
            f.write(orjson.dumps(r) + b"\n")
    val = []
    for line in open(a.src.replace(".jsonl", "_val.jsonl")):
        val.append(line)
        if len(val) >= a.n_val:
            break
    with open(a.out.replace(".jsonl", "_val.jsonl"), "w") as f:
        f.writelines(val)
    st = {"rows": len(rows), "long_rows_prompt_tokens": long_tokens, "val_rows": len(val),
          "by_teacher": dict(collections.Counter(agent_teacher[r["agent_ref"]["name"]] for r in rows)),
          "by_agent": dict(collections.Counter(r["agent_ref"]["name"] for r in rows).most_common())}
    json.dump(st, open(a.out.replace(".jsonl", ".stats.json"), "w"), indent=1)
    print(json.dumps(st, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
