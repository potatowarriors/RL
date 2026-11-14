"""terminal 피벗 teacher 블렌드 (RL_PLAN.md 결정 62, RL_DATA.md §6) — 프로파일링 통과율로 거른 피벗, 과제 단위 검증 분리 (CPU, 결정론적).

원천: HF `nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1` 31,111 행 (전부 고유, agent `terminus_judge_string_only_simple_agent`).
프로파일: MOPD1 학생으로 행마다 8회 생성한 결과 요약 (`$NRL_ROOT/p0_measure/term_pivot/term_profile_v1.json` — 유사도 · 스키마/완료 일치).
남기는 행: 임계 `--threshold` 에서 통과 횟수가 1 ~ (반복 − 1) 인 행. 0 회 · 전부 통과는 그룹 advantage 가 0 이라 뺀다 (PivotRL 필터).
작업 완료 선언 행 (`task_complete: true`) 은 학습 블렌드의 `--tc-share` 이하로 줄인다 — 프로파일링에서 신호 있는 행의 26% 라
그대로 두면 완료 선언 학습이 주가 된다. 검증셋은 과제(`task_name`) 단위로 뗀다. 남긴 행에 `pass_rate` (그 임계 기준) 를 적는다.
채점 임계는 행이 아니라 레시피에서 정한다 (`string_similarity_threshold`). 출력 옆에 `.stats.json` 을 남긴다.

  python examples/configs/alpha/tools/build_terminal_pivot_blend.py --out <dir>/teacher_terminal_pivot_v1.jsonl \
      [--threshold 0.7] [--tc-share 0.15] [--n-val-tasks 30] [--seed 20261114]
"""

import argparse
import collections
import json
import os
import random

RL = "/home/work/Datasets/LL_datasets/posttraining/RL"
SOURCE = f"{RL}/Nemotron-RL-Agentic-Terminal-Pivot-v1/atcb_terminal_pivot_release_final_v2.jsonl"
PROFILE = "/home/work/vidsearch/tools/nemo_rl/p0_measure/term_pivot/term_profile_v1.json"
AGENT = "terminus_judge_string_only_simple_agent"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--tc-share", type=float, default=0.15, help="학습 블렌드에서 작업 완료 선언 행의 최대 비율")
    ap.add_argument("--n-val-tasks", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20261114)
    a = ap.parse_args()
    out_val = a.out.replace(".jsonl", "_val.jsonl")
    for path in (a.out, out_val):
        if os.path.exists(path):
            print(f"출력이 이미 있다: {path} — 지우고 다시 실행한다")
            return 1

    with open(PROFILE) as f:
        prof = json.load(f)
    passes = {}
    for uuid, d in prof.items():
        n = len(d["r"])
        k = sum(1 for sim, ok, _ in d["r"] if ok and sim is not None and sim >= a.threshold)
        passes[uuid] = (k, n)
    keep = {u for u, (k, n) in passes.items() if 0 < k < n}

    rng = random.Random(a.seed)
    tasks = sorted({prof[u]["task"] for u in keep})
    rng.shuffle(tasks)
    val_tasks = set(tasks[: a.n_val_tasks])
    train_cmd = sorted(u for u in keep if prof[u]["task"] not in val_tasks and not prof[u]["tc"])
    train_tc = sorted(u for u in keep if prof[u]["task"] not in val_tasks and prof[u]["tc"])
    rng.shuffle(train_tc)
    # tc / (cmd + tc) <= share  →  tc <= cmd * share / (1 - share)
    tc_cap = int(len(train_cmd) * a.tc_share / (1 - a.tc_share))
    tc_dropped = max(0, len(train_tc) - tc_cap)
    train = set(train_cmd) | set(train_tc[:tc_cap])
    val = {u for u in keep if prof[u]["task"] in val_tasks}

    rows = {}
    source_rows = 0
    bad_agent = 0
    with open(SOURCE) as f:
        for line in f:
            source_rows += 1
            row = json.loads(line)
            u = row["uuid"]
            if u not in train and u not in val:
                continue
            if (row.get("agent_ref") or {}).get("name") != AGENT:
                bad_agent += 1
                continue
            k, n = passes[u]
            row["pass_rate"] = k / n
            row["pass_rate_total"] = n
            row["pass_rate_threshold"] = a.threshold
            rows[u] = row
    if bad_agent or len(rows) != len(train) + len(val):
        print(f"행 불일치: agent 다름 {bad_agent} · 찾은 행 {len(rows)} / 기대 {len(train) + len(val)}")
        return 1

    def kind(u: str) -> str:
        d = prof[u]
        if d["tc"]:
            return "task_complete"
        kl = d["klen"]
        return "cmd<=80" if kl <= 80 else "cmd<=200" if kl <= 200 else "cmd<=500" if kl <= 500 else "cmd>500"

    for path, keys in ((a.out, sorted(train)), (out_val, sorted(val))):
        rng.shuffle(keys)
        with open(path, "w") as fo:
            for u in keys:
                fo.write(json.dumps(rows[u], ensure_ascii=False) + "\n")
        st = {
            "file": path,
            "rows": len(keys),
            "tasks": len({prof[u]["task"] for u in keys}),
            "by_kind": collections.Counter(kind(u) for u in keys),
            "task_complete_share": round(sum(prof[u]["tc"] for u in keys) / len(keys), 4),
            "mean_profile_pass_rate": round(sum(rows[u]["pass_rate"] for u in keys) / len(keys), 4),
            "threshold": a.threshold,
            "tc_share_cap": a.tc_share,
            "task_complete_rows_dropped_by_cap": tc_dropped if path == a.out else 0,
            "source": SOURCE,
            "source_rows": source_rows,
            "profile": PROFILE,
            "profiled_rows": len(prof),
            "rows_with_signal": len(keep),
            "seed": a.seed,
        }
        with open(path + ".stats.json", "w") as fs:
            json.dump(st, fs, indent=1, ensure_ascii=False)
        print(json.dumps(st, ensure_ascii=False))
    overlap = {prof[u]["task"] for u in train} & val_tasks
    print(f"학습-검증 과제 겹침: {len(overlap)}")
    return 1 if overlap else 0


if __name__ == "__main__":
    raise SystemExit(main())
