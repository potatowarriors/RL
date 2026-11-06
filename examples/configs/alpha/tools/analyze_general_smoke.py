r"""General teacher 스모크 JG1 판정 (CPU) — W&B 오프라인 표 · 롤아웃 그룹 기록 · Gym 로그로 기준마다 PASS/FAIL 을 낸다.

게이트 정의는 docs/GATES.md JG1. 스모크는 J1 보충(j1c)과 같은 방식으로 Gym 전체 결과 표를 남긴다 (새 tag 로 — 그룹 기록·Gym 로그는 이어 쓴다):
  EXTRA_ENV="WANDB_MODE=offline" WANDB=1 launch.sh ... logger.wandb.metric_allowlist=null ++logger.wandb.log_nemo_gym_full_result_tables=true
  python3 examples/configs/alpha/tools/analyze_general_smoke.py --wandb <log_dir>/exp_*/wandb --groups <log_dir>/rollout_groups.jsonl \
      --gym-logs <Gym 로그 디렉토리> --since <기동 epoch 초> [--show 2]

조용한 실패 경로 (JG1 적대적 검토 2026-11-06 반영):
  - judge 호출 실패는 Gym judge_failsafe 가 보상 0 + `_ng_failure_class` 행으로 바꾸고 NeMo-RL 은 그대로 학습한다 → agent 별 그 행 수 = 0 을 요구.
  - 판정 기록이 없는 행 (rubric_evaluations · judge_output · judge_evaluation · verifier_labels 없음) 은 "판정 안 됨". 정당한 생략은
    abstention 의 [IDK] 그대로 (judge 를 부르지 않는다) · 안전 agent 의 빈 응답뿐 → 판정 행 ≥ 95% (대상 행 기준), 0 이면 FAIL.
  - 엄격 파싱: multichallenge [[YES]]/[[NO]] · inverse_if JSON result · abstention 한 글자 A/B/C · 안전 judge 원문에 서로 다른 라벨이 정확히 하나
    (지시문의 "[[SAFE]] or [[UNSAFE]]" 를 되풀이하면 서버는 첫 라벨을 읽는다) · 정책별 jailbreak verifier 라벨 None = 실패.
  - GenRM: genrm_compare 의 무승부 대체 로그 ("[GenRM] Parse failed for pair" · "[GenRM] Error in comparison") 를 직접 센다.
    그룹 보상의 분산은 길이 보너스·벌점과 보상 벌점 때문에 고장 난 GenRM 에서도 생긴다 → 판정에 쓰지 않는다.
    alpha_genrm_model 누적 집계로 순위 분포 (위치 편향·상수 판정) · 동점 · principle 주입 수를 본다.
  - 규칙 채점 agent 의 보상 고유값은 잘리지 않고 최종 답이 있는 표본만으로 센다 (벌점 0 이 고유값을 만든다).
"""
import argparse
import collections
import glob
import json
import os
import re
import statistics

JSON_RESULT = re.compile(r"\{[^{}]*\"result\"\s*:\s*\"(PASS|FAIL)\"[^{}]*\}", re.I)
GENRM = ("genrm_simple_agent", "genrm_simple_agent_reasoning_off")
RULE = ("instruction_following_simple_agent", "structured_outputs_simple_agent", "structured_outputs_v3_simple_agent",
        "citation_format_simple_agent", "freeform_formatting_simple_agent")
JUDGE_GROUPS = {
    "multichallenge": ("multichallenge_simple_agent",),
    "inverse_if": ("inverse_if_simple_agent",),
    "abstention": ("abstention_simple_agent",),
    "safety": ("jailbreak_detection_simple_agent", "jailbreak_refusal_with_explanation", "jailbreak_hard_refusal_with_helplines",
               "jailbreak_engagement_with_disclaimer", "jailbreak_hard_refusal_no_redirection", "over_refusal_detection_simple_agent"),
}
SAFETY_LABELS = {"jailbreak": ("[[SAFE]]", "[[UNSAFE]]"), "over_refusal": ("[[COMPLIED]]", "[[REFUSED]]"),
                 "quality": ("[[HAS_EXPLANATION]]", "[[NO_EXPLANATION]]")}


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


def last_text(resp):
    try:
        for item in reversed(resp.get("output") or []):
            if item.get("type") == "message":
                for p in reversed(item.get("content") or []):
                    if p.get("type") == "output_text":
                        return p.get("text") or ""
    except AttributeError:
        pass
    return ""


def policy_response(r):
    resp = r.get("response") if isinstance(r.get("response"), dict) else find(r, "response")
    return resp if isinstance(resp, dict) else {}


def truncated_or_empty(r):
    resp = policy_response(r)
    return resp.get("status") == "incomplete" or bool(resp.get("incomplete_details")) or not last_text(resp).strip()


def top_reward(r):
    v = r.get("reward")
    if isinstance(v, (int, float)):
        return float(v)
    v = find(r, "reward")
    return float(v) if isinstance(v, (int, float)) else None


def strict_one_label(text, labels):
    present = {lab for lab in labels if lab in (text or "")}
    if len(present) == 1:
        return True
    last = (text or "").strip().rsplit("\n", 1)[-1]
    return sum(lab in last for lab in labels) == 1


def judge_record(agent, r):
    """(판정 대상인가, 판정됐는가, 판정 수, 엄격 실패 수, 예시)."""
    if agent.startswith("multichallenge") or agent.startswith("inverse_if"):
        evs = find(r, "rubric_evaluations") or []
        if agent.startswith("multichallenge"):
            bad = [e for e in evs if "[[YES]]" not in e.get("judge_response", "") and "[[NO]]" not in e.get("judge_response", "")]
        else:
            bad = [e for e in evs if not JSON_RESULT.search(e.get("judge_response", ""))]
        return True, bool(evs), len(evs), len(bad), (bad[0].get("judge_response", "")[-300:] if bad else None)
    if agent.startswith("abstention"):
        if str(find(r, "extracted_answer") or "").strip() == "[IDK]":
            return False, False, 0, 0, None
        out = find(r, "judge_output")
        if out is None:
            return True, False, 0, 0, None
        c = out.strip()
        ok = c in ("A", "B", "C") or c.rsplit("\n", 1)[-1].strip() in ("A", "B", "C")
        return True, True, 1, int(not ok), (None if ok else out[-300:])
    # 안전
    if not str(find(r, "model_response_text") or "").strip():
        return False, False, 0, 0, None
    n = bad = 0
    ex = None
    labels_map = find(r, "verifier_labels")
    if isinstance(labels_map, dict) and labels_map:
        for v_name, lab in labels_map.items():
            n += 1
            if lab is None:
                bad += 1
                ex = ex or f"verifier {v_name} label None"
    kind = "over_refusal" if agent.startswith("over_refusal") else "jailbreak"
    for key, labs in (("judge_evaluation", SAFETY_LABELS[kind]), ("quality_judge_evaluation", SAFETY_LABELS["quality"])):
        ev = r.get(key) if isinstance(r.get(key), dict) else find(r, key)
        if not isinstance(ev, dict):
            continue
        n += 1
        text = last_text(ev.get("response") or {})
        if ev.get("verdict_label") is None or ev.get("error") or not strict_one_label(text, labs):
            bad += 1
            ex = ex or (ev.get("error") or text[-300:])
    return True, n > 0, n, bad, ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--gym-logs", required=True)
    ap.add_argument("--since", type=float, default=0.0, help="이 epoch 초 이전 그룹 기록·로그 파일은 무시 (재실행 누적 방지)")
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    verdicts = []

    def check(name, ok, detail):
        verdicts.append((name, ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    rows = load_tables(a.wandb)
    strata = set(rows)
    ident = sum(1 for ag in GENRM for r in rows.get(ag, []) if str(find(r, "dataset") or "").startswith("identity"))
    expected = set(RULE) | set(GENRM) | {x for g in JUDGE_GROUPS.values() for x in g}
    check("P3 agent 16 + identity 가 표에 있음", expected <= strata and ident > 0,
          f"없는 agent {sorted(expected - strata)} · identity 표본 {ident}")

    fails = {ag: sum(1 for r in rows[ag] if find(r, "_ng_failure_class") is not None) for ag in rows}
    check("P5a judge 실패 대체 행 (_ng_failure_class) = 0", sum(fails.values()) == 0, {k: v for k, v in fails.items() if v})

    for ag in RULE:
        vals = {top_reward(r) for r in rows.get(ag, []) if not truncated_or_empty(r)} - {None}
        check(f"P4 {ag} 잘리지 않은 표본의 보상 고유값 ≥ 2", len(vals) >= 2, f"고유값 {sorted(vals)[:6]}")

    for g, agents in JUDGE_GROUPS.items():
        elig = judged = n = bad = 0
        examples = []
        for ag in agents:
            for r in rows.get(ag, []):
                e, j, k, b, ex = judge_record(ag, r)
                elig += e
                judged += j
                n += k
                bad += b
                if ex and len(examples) < a.show:
                    examples.append(ex)
        check(f"P5b {g} 판정된 행 ≥ 95% (대상 {elig})", elig > 0 and judged >= 0.95 * elig, f"판정 {judged}/{elig}")
        check(f"P5c {g} 엄격 파싱 실패 ≤ 2%", n > 0 and bad <= 0.02 * n, f"{bad}/{n}")
        for ex in examples:
            print("      실패 예:", repr(ex))

    groups = collections.defaultdict(list)
    for line in open(a.groups):
        g = json.loads(line)
        if g.get("time", 1e18) < a.since or g.get("dropped"):
            continue
        groups[g.get("agent") or "?"].append(g)
    n_genrm_groups = sum(len(groups.get(x, [])) for x in GENRM)
    log_text = ""
    for f in glob.glob(os.path.join(a.gym_logs, "**", "*"), recursive=True):
        if os.path.isfile(f) and os.path.getmtime(f) >= a.since:
            try:
                log_text += open(f, errors="ignore").read()
            except OSError:
                pass
    fallback = log_text.count("[GenRM] Parse failed for pair") + log_text.count("[GenRM] Error in comparison")
    expect_cmp = 16 * n_genrm_groups
    check("P6a GenRM 무승부 대체 ≤ 2% (genrm_compare 로그 직접 계수)", n_genrm_groups > 0 and fallback <= 0.02 * expect_cmp,
          f"대체 {fallback} / 비교 약 {expect_cmp} (GenRM 그룹 {n_genrm_groups})")
    last = {}
    for m in re.finditer(r"\[alpha_genrm\] stats pid=(\d+) ((?:\w+=\d+ ?)+)", log_text):
        kv = dict(x.split("=") for x in m.group(2).split())
        kv = {k: int(v) for k, v in kv.items()}
        pid = int(m.group(1))
        if kv.get("requests", 0) >= last.get(pid, {}).get("requests", -1):
            last[pid] = kv
    tot = collections.Counter()
    for kv in last.values():
        tot.update(kv)
    req = tot["requests"]
    check("P6b alpha_genrm 파싱 실패 ≤ 2%", req > 0 and tot["parse_failed"] <= 0.02 * req,
          f"{tot['parse_failed']}/{req} (워커 {len(last)}개, 50회 단위 스냅샷)")
    ranks = [tot[f"rank{i}"] for i in range(1, 7)]
    r1 = sum(ranks[:3]) / max(1, sum(ranks))
    check("P6c 순위 분포: response_1 우세 비율 0.2~0.8 (위치 편향·상수 판정)", sum(ranks) > 0 and 0.2 <= r1 <= 0.8,
          f"순위 1~6 = {ranks}, response_1 우세 {r1:.2f}")
    tie = tot["score_tie"] / max(1, tot["parsed"])
    check("P6d 개별 점수 동점 ≤ 50%", tot["parsed"] > 0 and tie <= 0.5, f"{tot['score_tie']}/{tot['parsed']} = {tie:.2f}")
    print(f"[기록] principle 주입 {tot['principle_injected']} (스냅샷 하한) — principle 행 표본 {sum(1 for ag in GENRM for r in rows.get(ag, []) if find(r, 'principle'))}")
    print("[기록] 그룹 보상이 모두 같은 그룹 비율 (판정에 쓰지 않음):",
          {ag: round(sum(1 for g in gs if len(set(g['rewards'])) <= 1) / len(gs), 2) for ag, gs in sorted(groups.items())})
    n_fail = sum(1 for _, ok in verdicts if not ok)
    print(f"JG1 분석기 판정: {'PASS' if n_fail == 0 else 'FAIL'} ({len(verdicts) - n_fail}/{len(verdicts)}) — P1·P2 (rc · 배치 · KL · 마스킹) 는 로그·TensorBoard 로 따로 본다")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
