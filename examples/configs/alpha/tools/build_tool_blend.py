"""1차 도구 사용 warm-up teacher 블렌드 (RL_DATA.md §5.8, 결정 19) — 단일 스텝 도구 호출 환경 3종 (CPU, 결정론적).

v1·E1 RLVR 런에서 검증한 judge 불필요 환경만 쓴다: tau 피벗 · toolcall_schema · swe_pivot.
원천은 Ultra 복원 블렌드(rlvr1·rlvr2·mopd)와 Super 블렌드(rlvr1·rlvr2·rlvr3)의 해당 agent 행이다.
중복은 입력 메시지 + 도구 정의 + 기대 행동의 해시로 걸러낸다 (같은 대화라도 도구·정답이 다르면 다른 과제).
참조 pass_rate 가 있으면 (0,1) 인 행만 쓴다 — 0·1 은 참조 정책에서도 advantage 가 0 이다.
검증셋(--n-val)은 agent 비율대로 먼저 떼고 학습 블렌드에서 뺀다. 출력 옆에 .stats.json 을 남긴다.

  python examples/configs/alpha/tools/build_tool_blend.py --out <dir>/teacher_tool_v1.jsonl \
      [--n-train 25600] [--n-val 512] [--seed 20261008]
"""

import argparse
import collections
import json
import os
import random

import orjson
import xxhash

RL = "/home/work/Datasets/LL_datasets/posttraining/RL"
AB = f"{RL}/alpha_blends"
SOURCES = [
    f"{AB}/ultra_restored/rlvr1.jsonl",
    f"{AB}/ultra_restored/rlvr2.jsonl",
    f"{AB}/ultra_restored/mopd.jsonl",
    f"{AB}/super_restored/rlvr1.jsonl",
    f"{AB}/super_restored/rlvr2.jsonl",
    f"{RL}/Nemotron-RL-Super-Training-Blends/rlvr3.jsonl",
]
TAU = "single_step_tool_use_with_argument_comparison_agent"
SCHEMA = "toolcall_schema_single_step_tool_use_with_argument_comparison_agent"
SWE = "swe_pivot_single_step_tool_use_with_argument_comparison_agent"
# 목표 비율: swe_pivot 은 고유 행 전부(약 3.75천)를 쓰고, 나머지를 tau · toolcall_schema 로 채운다
SHARE = {SCHEMA: 0.30, SWE: 0.15}


def task_key(row) -> str:
    rcp = row.get("responses_create_params") or {}
    blob = orjson.dumps(
        [rcp.get("input"), rcp.get("tools"), row.get("expected_action"), row.get("ground_truth")],
        option=orjson.OPT_SORT_KEYS,
    )
    return xxhash.xxh64_hexdigest(blob)


def collect():
    pools = collections.defaultdict(dict)  # agent -> key -> row
    seen_rows = collections.Counter()
    dropped_pr = collections.Counter()
    for path in SOURCES:
        with open(path, "rb") as f:
            for line in f:
                if b"single_step_tool_use_with_argument_comparison" not in line:
                    continue
                row = orjson.loads(line)
                ag = (row.get("agent_ref") or {}).get("name", "")
                if ag not in (TAU, SCHEMA, SWE):
                    continue
                seen_rows[ag] += 1
                pr = row.get("pass_rate")
                if pr is not None and not (0.0 < float(pr) < 1.0):
                    dropped_pr[ag] += 1
                    continue
                pools[ag].setdefault(task_key(row), row)
    return pools, seen_rows, dropped_pr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=25600)
    ap.add_argument("--n-val", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20261008)
    a = ap.parse_args()
    if os.path.exists(a.out):
        print(f"출력이 이미 있다: {a.out} — 지우고 다시 실행한다")
        return 1
    pools, seen_rows, dropped_pr = collect()
    rng = random.Random(a.seed)
    keys = {ag: sorted(p) for ag, p in pools.items()}
    for ag in keys:
        rng.shuffle(keys[ag])
    total = a.n_train + a.n_val
    want = {SCHEMA: round(total * SHARE[SCHEMA]), SWE: min(len(keys[SWE]), round(total * SHARE[SWE]))}
    want[TAU] = total - want[SCHEMA] - want[SWE]
    for ag, n in want.items():
        if n > len(keys[ag]):
            raise SystemExit(f"{ag}: 필요 {n} > 고유 {len(keys[ag])}")
    val, train = [], []
    for ag, n in want.items():
        n_val = round(a.n_val * n / total)
        chosen = keys[ag][:n]
        val += [(ag, k) for k in chosen[:n_val]]
        train += [(ag, k) for k in chosen[n_val:]]
    # 검증셋 크기를 정확히 맞춘다 (반올림 오차는 tau 에서 조정)
    while len(val) > a.n_val:
        train.append(val.pop())
    rng.shuffle(train)
    rng.shuffle(val)
    out_val = a.out.replace(".jsonl", "_val.jsonl")
    for path, items in ((a.out, train), (out_val, val)):
        with open(path, "wb") as fo:
            for ag, k in items:
                fo.write(orjson.dumps(pools[ag][k]) + b"\n")
        st = {
            "file": path,
            "rows": len(items),
            "by_agent": collections.Counter(ag for ag, _ in items),
            "unique_pool_by_agent": {ag: len(p) for ag, p in pools.items()},
            "source_rows_by_agent": seen_rows,
            "dropped_pass_rate_0_or_1": dropped_pr,
            "seed": a.seed,
        }
        with open(path + ".stats.json", "w") as fs:
            json.dump(st, fs, indent=1, ensure_ascii=False)
        print(json.dumps(st, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
