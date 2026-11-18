r"""conversational tool-use E2E — 시뮬레이터 · judge 검증 롤아웃 분석 (CPU, RL_PLAN.md 결정 69 2단계).

입력: `gym eval run` + `conversational_tool_use_simulation` 이 쓴 rollouts.jsonl (실행기 `$NRL_ROOT/p0_measure/convtool/run_sim_check.sh`).
보상이 0/1 하나로 나오므로, 0 이 된 까닭을 한 대화에 하나씩 배타적으로 귀속한다 (위에서부터 먼저 해당하는 것):
  judge_error        판정 생성 실패 (`verification_generation_error`) — judge 가 형식을 못 맞춰 보상 0. 채점기 고장이지 상담원 실패가 아니다
  user_sim           고객 시뮬레이터 실패 (`user_failure` · 고객 메시지 판정 실패)
  tool_sim           도구 결과 시뮬레이터 실패 (`tool_failure` · `invalid_environment_tool_execution`)
  transfer_mismatch  이관 정답과 어긋남 (결정적 0, judge 생략)
  agent_invalid      상담원의 도구 호출이 스키마 위반 등 (`invalid_agent_*`)
  agent_message      상담원 메시지 · 도구 호출 단위 판정 실패
  agent_conversation 대화 전체 판정 실패
  other
  python examples/configs/alpha/tools/analyze_convtool_sim.py <rollouts.jsonl> [--out summary.json]
"""
import argparse
import collections
import json


def cause(r) -> str:
    if r.get("reward") == 1.0:
        return "pass"
    labels = set(r.get("failure_labels") or [])
    inv = set(r.get("trajectory_invalid_reasons") or [])
    mv = (r.get("judge_diagnostics") or {}).get("message_verification_results") or []
    if "verification_generation_error" in labels | inv or any(m.get("generation_error") for m in mv):
        return "judge_error"
    if "user_failure" in labels or any(m["source"] == "user" and m.get("reward") == 0 for m in mv):
        return "user_sim"
    if "tool_failure" in labels or any(m["source"] == "environment" and m.get("reward") == 0 for m in mv):
        return "tool_sim"
    if r.get("transfer_ground_truth_mismatch"):
        return "transfer_mismatch"
    if any(x.startswith("invalid_agent") for x in inv):
        return "agent_invalid"
    if any(m["source"] == "agent" and m.get("reward") == 0 for m in mv):
        return "agent_message"
    av = ((r.get("result") or {}).get("trajectory") or {}).get("agent_verification_result") or {}
    if (av.get("conversation_verification_result") or {}).get("reward") == 0:
        return "agent_conversation"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rollouts")
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.rollouts)]
    n = len(rows)
    causes = collections.Counter(cause(r) for r in rows)
    by_task = collections.defaultdict(list)
    for r in rows:
        by_task[r["id"]].append(r["reward"])
    k = collections.Counter("all0" if not any(v) else "all1" if all(v) else "mixed" for v in by_task.values())
    q = lambda xs: [round(sorted(xs)[int(p * (len(xs) - 1))], 1) for p in (0.1, 0.5, 0.9, 1.0)]
    # 이관: Gemma 라벨 × 원 궤적의 이관 여부 구분별
    xt = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        key = ("scope" if r.get("outside_policy_scope") else "in") + "/" + ("ref_transfer" if (r.get("metadata") or {}).get("reference_transferred") else "ref_no")
        xt[key]["n"] += 1
        xt[key]["observed_transfer"] += bool(r.get("transferred"))
        xt[key]["mismatch"] += bool(r.get("transfer_ground_truth_mismatch"))
        xt[key]["reward"] += r["reward"]
    # 도구 결과: 반환 스키마 유무별 메시지 단위 판정
    tool_ret = collections.defaultdict(lambda: collections.Counter())
    mv_all = collections.Counter()
    for r in rows:
        has = {t["name"]: bool(t.get("returns")) for t in r.get("tools") or []}
        for m in (r.get("judge_diagnostics") or {}).get("message_verification_results") or []:
            st = "err" if m.get("generation_error") else "unjudged" if m.get("reward") is None else "ok" if m["reward"] else "fail"
            mv_all[m["source"] + "/" + st] += 1
            if m["source"] == "environment":
                tool_ret["with_returns" if has.get(m.get("tool_name")) else "no_returns"][st] += 1
    for r in rows:
        if "invalid_environment_tool_execution" in (r.get("trajectory_invalid_reasons") or []):
            tool_ret["(dialogue)"]["invalid_environment_tool_execution"] += 1
    summ = {
        "dialogues": n, "tasks": len(by_task), "mean_reward": round(sum(r["reward"] for r in rows) / n, 4),
        "tasks_all0_mixed_all1": [k["all0"], k["mixed"], k["all1"]],
        "cause_share": {c: round(v / n, 4) for c, v in causes.most_common()},
        "simulator_or_judge_fault_share": round((causes["judge_error"] + causes["user_sim"] + causes["tool_sim"]) / n, 4),
        "terminal_state": dict(collections.Counter(r.get("terminal_state") for r in rows)),
        "invalid_reasons": dict(collections.Counter(x for r in rows for x in (r.get("trajectory_invalid_reasons") or []))),
        "transfer_by_label": {key: {"n": v["n"], "observed_transfer": round(v["observed_transfer"] / v["n"], 3), "mismatch": round(v["mismatch"] / v["n"], 3),
                                    "mean_reward": round(v["reward"] / v["n"], 3)} for key, v in sorted(xt.items())},
        "message_judgments": dict(sorted(mv_all.items())), "tool_results_by_returns_schema": {k2: dict(v) for k2, v in tool_ret.items()},
        "user_messages_p10_50_90_max": q([r["num_user_messages"] for r in rows]),
        "agent_messages_p10_50_90_max": q([r["num_agent_messages"] for r in rows]),
        "tool_calls_p10_50_90_max": q([r["num_tool_calls"] for r in rows]),
    }
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    if a.out:
        json.dump(summ, open(a.out, "w"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
