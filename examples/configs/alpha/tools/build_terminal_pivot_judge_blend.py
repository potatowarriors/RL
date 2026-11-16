"""terminal 피벗 teacher 블렌드 — 논문 방식 (RL_PLAN.md 결정 65 · 66 · 67, RL_DATA.md §6). CPU, 결정론적.

원천: HF `nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1` 31,111 행. 프로파일: MOPD1 학생 롤아웃 × 8 을 새 채점기
(alpha_terminus_judge + Gemma judge) 로 다시 채점한 행별 요약 `$NRL_ROOT/p0_measure/term_pivot/rejudge/term_profile_judge_v1.json`
(행마다 `o` = [보상, 출력 종류, 사유] 8개 · `gk` 정답 keystrokes · `g_tc` · `g_wait` · `g_rep` · `ctrl` · `task`).

선별 (PivotRL A.2 의 𝒟adv + 명령 중복 제거에, 사고 2026-11-16 의 경로를 막는 처리를 더함):
  1. 정답에 제어 키가 든 행 제외 (`ctrl`, 결정 66)
  2. 분산 > 0 (통과 1 ~ 반복−1 회) ∧ 평균 통과율 < `--lam` (결정 67: 0.5)
  3. 정답이 직전 명령의 반복인 행 제외 (`g_rep`, 결정 67)
  4. 명령 행: 정답 keystrokes 가 글자 그대로 같은 행은 `--keep` 개만 남긴다 (결정 67: 1). 빈 대기 정답은 한 행이 된다
  5. 완료 선언 행 (`g_tc`): 중복 제거 대상이 아니다 (명령이 전부 비어 한 묶음이 된다). 2 를 통과한 것을 학습 블렌드의 `--tc-share` 까지 (결정 67: 원본 비중 0.088)
검증셋은 과제 단위로 뗀다. 행의 agent_ref 를 `alpha_terminus_judge_simple_agent` 로 바꾸고 `pass_rate` (judge 기준) 를 적는다.
출력 옆 `.stats.json` 에 쏠림 지표를 남긴다 — 출력 종류별 양의 advantage 몫과 들림 (양의 몫 ÷ 음의 몫).

  python examples/configs/alpha/tools/build_terminal_pivot_judge_blend.py --out <dir>/teacher_terminal_pivot_judge_v1.jsonl \
      [--lam 0.5] [--keep 1] [--tc-share 0.088] [--n-val-tasks 30] [--seed 20261116]
종료 코드: 쏠림 기준 (빈 대기 + 직전 반복 출력의 양의 advantage 몫 ≤ 2% ∧ 빈 대기의 들림 ≤ 일하는 명령의 들림) 을 못 넘으면 1.
"""

import argparse
import collections
import json
import os
import random

RL = "/home/work/Datasets/LL_datasets/posttraining/RL"
SOURCE = f"{RL}/Nemotron-RL-Agentic-Terminal-Pivot-v1/atcb_terminal_pivot_release_final_v2.jsonl"
PROFILE = "/home/work/vidsearch/tools/nemo_rl/p0_measure/term_pivot/rejudge/term_profile_judge_v1.json"
AGENT = "alpha_terminus_judge_simple_agent"
MAX_WAIT_REPEAT_SHARE = 0.02


def passes(d: dict) -> int:
    return sum(int(o[0]) for o in d["o"])


def skew(rows: list[dict]) -> dict:
    """출력 종류별 양의 advantage 몫과 들림. advantage = (보상 − 그룹 평균) / 그룹 표준편차 (GRPO 정규화)."""
    pos, neg = collections.Counter(), collections.Counter()
    for d in rows:
        r = [o[0] for o in d["o"]]
        mu = sum(r) / len(r)
        sd = (sum((x - mu) ** 2 for x in r) / len(r)) ** 0.5
        if sd == 0:
            continue
        for o in d["o"]:
            a = (o[0] - mu) / sd
            (pos if a > 0 else neg)[o[1]] += abs(a)
    tot = sum(pos.values()) or 1.0
    cats = ("other", "wait", "repeat", "tc", "invalid", "recall")
    return {"pos_share": {c: round(pos[c] / tot, 4) for c in cats},
            "lift": {c: (round(pos[c] / neg[c], 3) if neg[c] else None) for c in cats}}


def kind(d: dict) -> str:
    if d["g_tc"]:
        return "task_complete"
    if d["g_wait"]:
        return "wait"
    kl = d["klen"]
    return "cmd<=80" if kl <= 80 else "cmd<=200" if kl <= 200 else "cmd<=500" if kl <= 500 else "cmd>500"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--keep", type=int, default=1)
    ap.add_argument("--tc-share", type=float, default=0.088)
    ap.add_argument("--n-val-tasks", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20261116)
    a = ap.parse_args()
    out_val = a.out.replace(".jsonl", "_val.jsonl")
    for path in (a.out, out_val):
        if os.path.exists(path):
            print(f"출력이 이미 있다: {path} — 지우고 다시 실행한다")
            return 1
    with open(PROFILE) as f:
        prof = json.load(f)
    funnel = collections.OrderedDict()
    funnel["source"] = len(prof)
    pool = {u: d for u, d in prof.items() if not d["ctrl"]}
    funnel["not_control_key"] = len(pool)
    pool = {u: d for u, d in pool.items() if 0 < passes(d) < len(d["o"])}
    funnel["variance>0"] = len(pool)
    pool = {u: d for u, d in pool.items() if passes(d) / len(d["o"]) < a.lam}
    funnel[f"mean<{a.lam}"] = len(pool)
    pool = {u: d for u, d in pool.items() if d["g_tc"] or not d["g_rep"]}
    funnel["not_repeat_of_previous"] = len(pool)

    rng = random.Random(a.seed)
    tasks = sorted({d["task"] for d in pool.values()})
    rng.shuffle(tasks)
    val_tasks = set(tasks[: a.n_val_tasks])

    def select(us: list[str], apply_tc_cap: bool) -> tuple[list[str], dict]:
        cmd = sorted(u for u in us if not pool[u]["g_tc"])
        tc = sorted(u for u in us if pool[u]["g_tc"])
        groups = collections.defaultdict(list)
        for u in cmd:
            groups[pool[u]["gk"]].append(u)
        kept = []
        for gk in sorted(groups):
            g = groups[gk]
            rng.shuffle(g)
            kept += g[: a.keep]
        rng.shuffle(tc)
        cap = int(len(kept) * a.tc_share / (1 - a.tc_share)) if apply_tc_cap else len(tc)
        return kept + tc[:cap], {"cmd_before_dedup": len(cmd), "cmd_after_dedup": len(kept), "tc_available": len(tc), "tc_kept": min(len(tc), cap)}

    train, st_train = select([u for u in pool if pool[u]["task"] not in val_tasks], True)
    val, st_val = select([u for u in pool if pool[u]["task"] in val_tasks], True)
    want = set(train) | set(val)
    rows = {}
    with open(SOURCE) as f:
        for line in f:
            r = json.loads(line)
            if r["uuid"] in want:
                d = pool[r["uuid"]]
                r["agent_ref"] = {"type": "responses_api_agents", "name": AGENT}
                r["pass_rate"] = passes(d) / len(d["o"])
                r["pass_rate_total"] = len(d["o"])
                r["pass_rate_scorer"] = "alpha_terminus_judge+gemma-4-31B-it"
                rows[r["uuid"]] = r
    if len(rows) != len(want):
        print(f"행 불일치: 찾은 {len(rows)} / 기대 {len(want)}")
        return 1
    verdicts = []
    for path, keys, st_sel in ((a.out, train, st_train), (out_val, val, st_val)):
        keys = sorted(keys)
        rng.shuffle(keys)
        with open(path, "w") as fo:
            for u in keys:
                fo.write(json.dumps(rows[u], ensure_ascii=False) + "\n")
        sel = [pool[u] for u in keys]
        sk = skew(sel)
        wr = sk["pos_share"]["wait"] + sk["pos_share"]["repeat"]
        ok = wr <= MAX_WAIT_REPEAT_SHARE and (sk["lift"]["wait"] or 0.0) <= (sk["lift"]["other"] or 0.0)
        st = {"file": path, "rows": len(keys), "tasks": len({d["task"] for d in sel}), "by_kind": collections.Counter(kind(d) for d in sel),
              "task_complete_share": round(sum(d["g_tc"] for d in sel) / len(sel), 4),
              "mean_profile_pass_rate": round(sum(passes(d) for d in sel) / sum(len(d["o"]) for d in sel), 4),
              "selection": st_sel, "funnel": funnel if path == a.out else None, "skew": sk,
              "wait_plus_repeat_pos_share": round(wr, 4), "skew_criterion": {"max_wait_repeat_share": MAX_WAIT_REPEAT_SHARE, "pass": ok},
              "lam": a.lam, "keep": a.keep, "tc_share_cap": a.tc_share, "source": SOURCE, "profile": PROFILE, "seed": a.seed}
        with open(path + ".stats.json", "w") as fs:
            json.dump(st, fs, indent=1, ensure_ascii=False)
        print(json.dumps(st, ensure_ascii=False))
        verdicts.append(ok)
    overlap = {pool[u]["task"] for u in train} & val_tasks
    print(f"학습-검증 과제 겹침: {len(overlap)} · 쏠림 기준 학습 {'PASS' if verdicts[0] else 'FAIL'}")
    return 0 if (not overlap and verdicts[0]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
