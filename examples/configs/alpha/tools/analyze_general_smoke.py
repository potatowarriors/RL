r"""General teacher 스모크 판정 (CPU) — W&B 오프라인 표 · 롤아웃 그룹 기록 · Gym 로그로 agent 별 보상과 judge 판정 파싱을 센다.

게이트 정의는 docs/GATES.md JG1. 스모크는 J1 보충(j1c)과 같은 방식으로 Gym 전체 결과 표를 남긴다:
  EXTRA_ENV="WANDB_MODE=offline" WANDB=1 launch.sh ... logger.wandb.metric_allowlist=null ++logger.wandb.log_nemo_gym_full_result_tables=true
  python3 examples/configs/alpha/tools/analyze_general_smoke.py --wandb <log_dir>/exp_*/wandb --groups <log_dir>/rollout_groups.jsonl \
      --gym-logs <log_dir>/nemo_gym [--genrm-stats <stats.json>] [--show 2]

판정 파싱 (Gym v0.6.0 각 서버의 실제 추출 규칙에서 "엄격 경로" 만 성공으로 센다 — 대체 경로는 조용히 NO·FAIL·오답으로 떨어진다):
  - multichallenge: rubric judge_response 에 [[YES]] · [[NO]] 가 있음 (없으면 서버는 마지막 줄 키워드 → 기본 NO)
  - inverse_if: judge_response 에 {"result": "PASS"|"FAIL"} JSON 이 있음 (없으면 키워드 스캔 → 기본 FAIL)
  - abstention: judge_output 이 A·B·C 한 글자이거나 마지막 줄이 그 글자 (없으면 서버는 아무 글자나 찾는다 → 기본 B=오답).
    정책이 거절 토큰을 그대로 내면 judge 를 부르지 않는다 (judge_output None, 판정 대상에서 뺀다)
  - over_refusal · jailbreak: judge_evaluation.verdict_label 이 None 이 아니고 error 가 없음 (jailbreak 은 quality_judge_evaluation 도 같은 기준)
  - GenRM: 응답 표에는 비교별 기록이 없다 → 롤아웃 그룹 보상의 분산(무승부로 떨어지면 그룹 보상이 같아진다)과
    alpha_genrm_model 의 [alpha_genrm] parse_failed 로그 · /alpha_genrm/stats 집계로 본다
"""
import argparse
import collections
import glob
import json
import os
import re
import statistics

JSON_RESULT = re.compile(r"\{[^{}]*\"result\"\s*:\s*\"(PASS|FAIL)\"[^{}]*\}", re.I)
GENRM_AGENTS = ("genrm_simple_agent", "genrm_simple_agent_reasoning_off")


def load_tables(root):
    rows, seen = collections.defaultdict(list), set()
    for f in sorted(glob.glob(os.path.join(root, "**", "*.table.json"), recursive=True)):
        agent = os.path.basename(os.path.dirname(f))
        for row in json.load(open(f))["data"]:
            if (agent, row[0]) in seen:
                continue
            seen.add((agent, row[0]))
            rows[agent].append(json.loads(row[0]))
    return rows


def find(d, key):
    """중첩 dict 에서 key 를 찾는다 (표 행이 verify 응답을 어디에 담는지 판마다 다를 수 있어서)."""
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = find(v, key)
            if r is not None:
                return r
    elif isinstance(d, list):
        for v in d:
            r = find(v, key)
            if r is not None:
                return r
    return None


def abstention_strict(text):
    c = (text or "").strip()
    return c in ("A", "B", "C") or c.rsplit("\n", 1)[-1].strip() in ("A", "B", "C")


def judge_parse(agent, r):
    """(판정 수, 엄격 파싱 실패 수, 예시 실패 원문)."""
    if agent.startswith("multichallenge"):
        evs = find(r, "rubric_evaluations") or []
        bad = [e for e in evs if "[[YES]]" not in e.get("judge_response", "") and "[[NO]]" not in e.get("judge_response", "")]
        return len(evs), len(bad), (bad[0].get("judge_response", "")[-300:] if bad else None)
    if agent.startswith("inverse_if"):
        evs = find(r, "rubric_evaluations") or []
        bad = [e for e in evs if not JSON_RESULT.search(e.get("judge_response", ""))]
        return len(evs), len(bad), (bad[0].get("judge_response", "")[-300:] if bad else None)
    if agent.startswith("abstention"):
        out = find(r, "judge_output")
        if out is None:
            return 0, 0, None
        return 1, int(not abstention_strict(out)), (None if abstention_strict(out) else out[-300:])
    if agent.startswith("over_refusal") or agent.startswith("jailbreak"):
        n = bad = 0
        ex = None
        for key in ("judge_evaluation", "quality_judge_evaluation"):
            ev = find(r, key)
            if ev is None:
                continue
            n += 1
            if ev.get("verdict_label") is None or ev.get("error"):
                bad += 1
                ex = ex or str(ev)[:300]
        return n, bad, ex
    return 0, 0, None


def reward_of(r):
    v = find(r, "reward")
    return float(v) if isinstance(v, (int, float)) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--gym-logs", default=None)
    ap.add_argument("--genrm-stats", default=None)
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    rows = load_tables(a.wandb)
    print("agent | 표본 | 보상 평균 | 0 비율 | 고유값 | 판정 수 | 엄격 파싱 실패")
    judge_tot = collections.Counter()
    for agent in sorted(rows):
        rs = [x for x in (reward_of(r) for r in rows[agent]) if x is not None]
        n_j = n_bad = 0
        examples = []
        for r in rows[agent]:
            j, b, ex = judge_parse(agent, r)
            n_j += j
            n_bad += b
            if ex and len(examples) < a.show:
                examples.append(ex)
        judge_tot["n"] += n_j
        judge_tot["bad"] += n_bad
        zero = sum(1 for x in rs if x == 0) / len(rs) if rs else float("nan")
        print(f"{agent} | {len(rows[agent])} | {statistics.mean(rs) if rs else float('nan'):.3f} | {zero:.2f} | {len(set(rs))} | {n_j} | {n_bad}"
              + (f" ({100 * n_bad / n_j:.1f}%)" if n_j else ""))
        for ex in examples:
            print("    실패 예:", repr(ex))
    print(f"judge 엄격 파싱 실패 합계: {judge_tot['bad']}/{judge_tot['n']}"
          + (f" ({100 * judge_tot['bad'] / judge_tot['n']:.2f}%)" if judge_tot["n"] else ""))
    groups = collections.defaultdict(list)
    for line in open(a.groups):
        g = json.loads(line)
        if g.get("dropped"):
            continue
        groups[g.get("agent") or "?"].append(g["rewards"])
    print("롤아웃 그룹 | 그룹 수 | 보상이 모두 같은 그룹 | 비율")
    for agent in sorted(groups):
        same = sum(1 for rw in groups[agent] if len(set(rw)) <= 1)
        print(f"{agent} | {len(groups[agent])} | {same} | {same / len(groups[agent]):.2f}")
    if a.gym_logs:
        n_fail, last = 0, {}
        stat_re = re.compile(r"\[alpha_genrm\] stats pid=(\d+) requests=(\d+) parsed=(\d+) parse_failed=(\d+)")
        for f in glob.glob(os.path.join(a.gym_logs, "**", "*"), recursive=True):
            if not os.path.isfile(f):
                continue
            try:
                text = open(f, errors="ignore").read()
            except OSError:
                continue
            n_fail += text.count("[alpha_genrm] parse_failed")
            for m in stat_re.finditer(text):
                pid, req, ok, bad = map(int, m.groups())
                if req >= last.get(pid, (0, 0, 0))[0]:
                    last[pid] = (req, ok, bad)
        tot = [sum(v[i] for v in last.values()) for i in range(3)]
        print(f"[alpha_genrm] parse_failed 로그 줄: {n_fail} · 마지막 누적 (워커 {len(last)}개 합, 50회 단위라 실제보다 최대 49×워커 적다): "
              f"requests={tot[0]} parsed={tot[1]} parse_failed={tot[2]}")
    if a.genrm_stats:
        print("alpha_genrm stats:", json.load(open(a.genrm_stats)))
    n_genrm_groups = sum(len(groups.get(x, [])) for x in GENRM_AGENTS)
    print(f"GenRM 그룹 {n_genrm_groups} → 기대 비교 수 ≥ {16 * n_genrm_groups} (circular, 그룹 16)")


if __name__ == "__main__":
    main()
