"""JG1 분석기 음성 대조 (CPU, docs/GATES.md JG1): 실제 표 구조 (GenRM 행 = failure_reason · response · responses_create_params · reward 4필드,
identity 는 롤아웃 그룹 기록 dataset) 와 서로 다른 행으로 만든 합성 입력. 건강 → PASS, 고장 → 해당 기준 FAIL, 임계 근처 양쪽 확인."""
import json, os, shutil, subprocess, sys
A = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze_general_smoke.py")
BASE = sys.argv[1]  # 합성 입력을 쓸 빈 디렉토리
# 사용: python3 negctl_general_smoke.py <빈 디렉토리> — 마지막 줄 '음성 대조: 14/14 기대대로'
RULE = ["instruction_following_simple_agent", "structured_outputs_simple_agent", "structured_outputs_v3_simple_agent",
        "citation_format_simple_agent", "freeform_formatting_simple_agent"]
GENRM = ["genrm_simple_agent", "genrm_simple_agent_reasoning_off"]
SAFETY = ["jailbreak_detection_simple_agent", "jailbreak_refusal_with_explanation", "jailbreak_hard_refusal_with_helplines",
          "jailbreak_engagement_with_disclaimer", "jailbreak_hard_refusal_no_redirection", "over_refusal_detection_simple_agent"]
N = 100
def resp(text, status="completed"):
    return {"status": status, "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}
def healthy():
    rows = {}
    for ag in RULE:
        rows[ag] = [{"reward": float(i % 2), "response": resp(f"answer {i}")} for i in range(N)]
    for ag in GENRM:
        rows[ag] = [{"failure_reason": None, "reward": 3.0 + (i % 7) / 10, "response": resp(f"a{i}"), "responses_create_params": {"input": [{"role": "user", "content": f"q{i}"}]}} for i in range(N)]
    rows["multichallenge_simple_agent"] = [{"reward": 1.0, "response": resp(f"a{i}"), "rubric_evaluations": [{"judge_response": f"r{i} [[YES]]"}]} for i in range(N)]
    rows["inverse_if_simple_agent"] = [{"reward": 0.0, "response": resp(f"a{i}"), "rubric_evaluations": [{"judge_response": '{"result": "FAIL", "e": %d}' % i}]} for i in range(N)]
    rows["abstention_simple_agent"] = [{"reward": 1.0, "response": resp(f"a{i}"), "extracted_answer": f"x{i}", "verdict": "correct", "judge_output": "A"} for i in range(N)]
    for ag in SAFETY:
        lab = "[[COMPLIED]]" if ag.startswith("over") else "[[SAFE]]"
        rows[ag] = [{"reward": 1.0, "response": resp(f"no {i}"), "model_response_text": f"I can't help {i}",
                     "judge_evaluation": {"verdict_label": lab, "response": resp(f"reason {i}\n{lab}")}} for i in range(N)]
    for i, x in enumerate(rows["jailbreak_refusal_with_explanation"]):
        x.pop("judge_evaluation"); x["response_policy"] = "refusal_with_explanation"; x["verifier_labels"] = {"refusal": "[[REFUSED]]", "explanation": "[[HAS_EXPLANATION]]"}
    groups = [{"agent": ag, "rewards": [1.0, 0.0], "time": 2e9, "dataset": "x"} for ag in rows]
    groups += [{"agent": "genrm_simple_agent", "rewards": [3.1, 2.9], "time": 2e9, "dataset": "identity_nosys.alpha_v1"}] * 2
    groups += [{"agent": "genrm_simple_agent", "rewards": [3.1, 2.9], "time": 2e9, "dataset": "ultra"}] * 2
    # GenRM 그룹 5개 (genrm 4 + reasoning_off 1) → 기대 비교 80, identity 2 그룹 → principle 하한 32
    logs = "[alpha_genrm] stats pid=7 requests=100 parsed=100 parse_failed=0 principle_injected=40 score_tie=20 rank1=10 rank2=20 rank3=20 rank4=20 rank5=20 rank6=10\n"
    return rows, groups, logs
def write(d, rows, groups, logs):
    shutil.rmtree(d, ignore_errors=True)
    for ag, rs in rows.items():
        os.makedirs(f"{d}/wandb/files/media/table/{ag}", exist_ok=True)
        json.dump({"data": [[json.dumps(r)] for r in rs]}, open(f"{d}/wandb/files/media/table/{ag}/t.table.json", "w"))
    os.makedirs(f"{d}/logs", exist_ok=True)
    open(f"{d}/logs/gym.log", "w").write(logs)
    open(f"{d}/groups.jsonl", "w").write("".join(json.dumps(g) + "\n" for g in groups))
def run(name, mut, expect_fail):
    rows, groups, logs = healthy()
    rows, groups, logs = mut(rows, groups, logs)
    d = f"{BASE}/{name}"; write(d, rows, groups, logs)
    out = subprocess.run([sys.executable, "-I", A, "--wandb", f"{d}/wandb", "--groups", f"{d}/groups.jsonl", "--gym-logs", f"{d}/logs"], capture_output=True, text=True)
    failed = [l for l in out.stdout.splitlines() if l.startswith("[FAIL]")]
    hit = any(expect_fail in l for l in failed) if expect_fail else not failed
    print(f"{'OK ' if hit else 'BAD'} {name}: rc={out.returncode} FAIL={[l.split(':')[0][7:] for l in failed]}")
    if not hit: print(out.stdout[-1500:], out.stderr[-800:])
    return hit
def m_judge_fail(r, g, l):
    for x in r["multichallenge_simple_agent"][:10]: x.pop("rubric_evaluations"); x["_ng_failure_class"] = "judge_failed"; x["reward"] = 0.0
    return r, g, l
def m_echo_mismatch(r, g, l):
    for i, x in enumerate(r["jailbreak_detection_simple_agent"][:5]):
        x["judge_evaluation"] = {"verdict_label": "[[SAFE]]", "response": resp(f"Respond with either [[SAFE]] or [[UNSAFE]]. analysis {i}\n[[UNSAFE]]")}
    return r, g, l
def m_policy_none(r, g, l):
    for x in r["jailbreak_refusal_with_explanation"][:10]: x["verifier_labels"] = {"refusal": None, "explanation": None}
    return r, g, l
def m_genrm_fallback(r, g, l):
    return r, g, l + "[GenRM] Error in comparison for pair (0, 1): connection\n" * 5
def m_position_bias(r, g, l):
    return r, g, l.replace("rank1=10 rank2=20 rank3=20 rank4=20 rank5=20 rank6=10", "rank1=90 rank2=10 rank3=0 rank4=0 rank5=0 rank6=0")
def m_rule_constant(r, g, l):
    r["citation_format_simple_agent"] = [{"reward": 1.0, "response": resp(f"a{i}")} for i in range(90)] + [{"reward": 0.0, "response": resp("", "incomplete")} for _ in range(10)]
    return r, g, l
def m_principle_lost(r, g, l):
    return r, g, l.replace("principle_injected=40", "principle_injected=0")
def m_genrm_dead_quiet(r, g, l):
    return r, g, "[alpha_genrm] stats pid=7 requests=10 parsed=10 parse_failed=0 principle_injected=40 score_tie=2 rank1=1 rank2=2 rank3=2 rank4=2 rank5=2 rank6=1\n"
def m_abstention_boxed_idk(r, g, l):
    for i, x in enumerate(r["abstention_simple_agent"][:20]): x.update({"extracted_answer": "IDK", "verdict": "abstain", "judge_output": None, "reward": 0.5})
    return r, g, l
def m_multi_parse(frac):
    def f(r, g, l):
        for x in r["multichallenge_simple_agent"][:int(frac * N)]: x["rubric_evaluations"] = [{"judge_response": "I think it passes"}]
        return r, g, l
    return f
def m_adapter_parse(failed):
    def f(r, g, l):
        return r, g, l.replace("parsed=100 parse_failed=0", f"parsed={100 - failed} parse_failed={failed}")
    return f
ok = [run("healthy", lambda r, g, l: (r, g, l), None),
      run("judge_failed_rows", m_judge_fail, "P5a"),
      run("safety_echo_then_opposite_verdict", m_echo_mismatch, "P5d"),
      run("policy_verifier_none", m_policy_none, "P5e"),
      run("genrm_fallback", m_genrm_fallback, "P6a"),
      run("genrm_position_bias", m_position_bias, "P6c"),
      run("rule_constant_with_penalty_zero", m_rule_constant, "P4 citation_format"),
      run("principle_not_injected", m_principle_lost, "P6e"),
      run("genrm_dead_without_log_lines", m_genrm_dead_quiet, "P6f"),
      run("abstention_boxed_idk_is_not_unjudged", m_abstention_boxed_idk, None),
      run("multichallenge_parse_fail_1pct_passes", m_multi_parse(0.01), None),
      run("multichallenge_parse_fail_5pct_fails", m_multi_parse(0.05), "P5c multichallenge"),
      run("adapter_parse_fail_1pct_passes", m_adapter_parse(1), None),
      run("adapter_parse_fail_5pct_fails", m_adapter_parse(5), "P6b")]
print(f"음성 대조: {sum(ok)}/{len(ok)} 기대대로")
