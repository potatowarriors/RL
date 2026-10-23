"""Code·Math 3단계 바퀴 경계 블렌드 갱신 (RL_PLAN.md 결정 39·40, 사용자 승인 2026-10-22).

이전 바퀴의 롤아웃 기록(rollout_groups.jsonl)으로 다음 바퀴 블렌드를 만든다.
  - 마지막 방문이 전부 정답(16/16)인 문제는 뺀다.
  - 마지막 방문이 전부 오답(0/16)인 문제는 --zero-keep (기본 0.25) 비율만 무작위로 남긴다 — 학습 중 풀리기 시작하는 문제를 잡기 위해.
  - 그 밖(1~15/16, 미방문, 과장으로 버린 그룹)은 그대로 넣는다. code 1~2/16 두 번 넣기(G=32 효과)는 결정 43 (사용자 2026-10-23) 으로
    기본 끔 — 이전 블렌드의 *_dup 행도 pool_key 로 접어 한 번만 넣는다. 켜려면 --dup-code-max-correct 2.
  - 검증셋(2단계 _val.jsonl) 은 계속 뺀다. 행 내용은 이전 블렌드 행 그대로이고 round 필드(stage3_round, stage3_last_correct)를 갱신한다.

사용:
  python examples/configs/alpha/tools/build_codemath_round_blend.py --prev <이전 블렌드.jsonl> --log <rollout_groups.jsonl> \
      --out <다음 블렌드.jsonl> [--zero-keep 0.25] [--seed N]
결과: <out> · <out>_stats.json
"""

import argparse
import collections
import json
import random


def last_visit(path):
    last = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            k = r.get("pool_key")
            if k is None:
                continue
            v = r["generation_weight_version"]
            if k not in last or v >= last[k][0]:
                c = sum(1 for x in r["rewards"] if x > 0)
                last[k] = (v, c, len(r["rewards"]), r.get("dropped"))
    return {k: (c, n, d) for k, (v, c, n, d) in last.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prev", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val", default="/home/work/Datasets/LL_datasets/posttraining/RL/alpha_blends/teacher_blends/teacher_codemath_s2_v1_val.jsonl")
    ap.add_argument("--zero-keep", type=float, default=0.25)
    ap.add_argument("--dup-code-max-correct", type=int, default=0, help="code 정답 수가 1~이 값이면 두 번 넣는다. 결정 43 (2026-10-23) 으로 기본 0 = 중복 없음")
    ap.add_argument("--seed", type=int, default=20261022)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    val_keys = {json.loads(l)["pool_key"] for l in open(a.val)}
    lv = last_visit(a.log)

    prev = {}
    for line in open(a.prev):
        r = json.loads(line)
        prev.setdefault(r["pool_key"], r)  # one row per prompt; duplicates re-made below
    rows, stats = [], collections.Counter()
    for k, r in prev.items():
        dom = "code" if "code" in r["agent_ref"]["name"] else "math"
        if k in val_keys:
            stats[f"{dom}_excluded_val"] += 1
            continue
        row = dict(r)
        row["stage3_round"] = int(r.get("stage3_round", 1)) + 1
        visit = lv.get(k)
        if visit is None:
            row["stage3_last_correct"] = r.get("stage3_last_correct")
            band = "unvisited"
        else:
            c, n, d = visit
            row["stage3_last_correct"] = c
            if d == "overlong":
                band = "overlong"
            elif c == n:
                stats[f"{dom}_excluded_all_correct"] += 1
                continue
            elif c == 0:
                if rng.random() >= a.zero_keep:
                    stats[f"{dom}_excluded_zero_sampled_out"] += 1
                    continue
                band = "zero_kept"
            else:
                band = "signal"
        row["stage3_band"] = band
        rows.append(row)
        stats[f"{dom}_{band}"] += 1
        c = row["stage3_last_correct"]
        if dom == "code" and c is not None and 1 <= c <= a.dup_code_max_correct:
            dup = dict(row)
            dup["stage3_band"] = band + "_dup"
            rows.append(dup)
            stats["code_duplicated"] += 1
    rng.shuffle(rows)
    with open(a.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    out = {
        "prev": a.prev, "log": a.log, "seed": a.seed, "zero_keep": a.zero_keep,
        "visited_prompts": len(lv), "train_rows": len(rows),
        "train_unique_prompts": len({r["pool_key"] for r in rows}),
        "by_agent": dict(collections.Counter(r["agent_ref"]["name"] for r in rows)),
        "counts": dict(sorted(stats.items())),
        "val_train_overlap": len(val_keys & {r["pool_key"] for r in rows}),
        "steps_per_pass_at_128_keep0.5": len(rows) * 0.5 // 128,
    }
    with open(a.out.replace(".jsonl", "_stats.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
