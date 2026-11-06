r"""General teacher 스모크 JG1 판정 (CPU) — W&B 오프라인 표 · 롤아웃 그룹 기록 · Gym 로그로 기준마다 PASS/FAIL 을 낸다.

게이트 정의는 docs/GATES.md JG1. 스모크는 J1 보충(j1c)과 같은 방식으로 Gym 전체 결과 표를 남긴다 (새 tag 로 — 그룹 기록·Gym 로그는 이어 쓴다):
  EXTRA_ENV="WANDB_MODE=offline" WANDB=1 launch.sh ... logger.wandb.metric_allowlist=null ++logger.wandb.log_nemo_gym_full_result_tables=true
  python3 examples/configs/alpha/tools/analyze_general_smoke.py --wandb <log_dir>/exp_*/wandb --groups <log_dir>/rollout_groups.jsonl \
      --gym-logs <Gym 로그 디렉토리> --since <기동 epoch 초> [--show 2]

조용한 실패 경로 (JG1 적대적 검토 2026-11-06 반영):
  - judge 호출 실패는 Gym judge_failsafe 가 보상 0 + `_ng_failure_class` 행으로 바꾸고 NeMo-RL 은 그대로 학습한다 → agent 별 그 행 수 = 0 을 요구.
  - 판정 기록이 없는 행 (rubric_evaluations · judge_output · judge_evaluation · verifier_labels 없음) 은 "판정 안 됨". 정당한 생략은
    abstention 의 [IDK] 그대로 (judge 를 부르지 않는다) · 안전 agent 의 빈 응답뿐 → 판정 행 ≥ 95% (대상 행 기준), 0 이면 FAIL.
  - 엄격 파싱 (≤ 3%): multichallenge [[YES]]/[[NO]] · inverse_if JSON result · abstention 한 글자 A/B/C · 안전 judge 는 서버가 고른 라벨이 원문의
    마지막 라벨과 같아야 성공 (서버는 처음 나온 라벨을 읽는다 — 지시문 되풀이 뒤 반대 판정이면 오독) · 정책별 jailbreak verifier 라벨 None = 실패.
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
POLICY_AGENTS = ("jailbreak_refusal_with_explanation", "jailbreak_hard_refusal_with_helplines",
                 "jailbreak_engagement_with_disclaimer", "jailbreak_hard_refusal_no_redirection")  # 블렌드 행에 response_policy_mapped 가 있다
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


def last_label(text, labels):
    """judge 의 최종 판정 라벨: 라벨만 단독으로 있는 마지막 줄 → 없으면 라벨이 정확히 하나 있는 마지막 줄 → 없으면 원문 마지막 라벨
    (판정 뒤 "(It is not [[UNSAFE]] because…)" 같은 반대 라벨 언급에 흔들리지 않게). 서버는 처음 나온 라벨을 읽으므로,
    지시문 되풀이 뒤 최종 판정이 반대면 서버 판정과 어긋난다."""
    lines = (text or "").splitlines()
    for line in reversed(lines):  # 1순위: 라벨만 단독으로 있는 줄 (최종 판정 줄의 흔한 모양)
        bare = line.strip().strip("*.:`'\" ").replace("Verdict", "").replace("verdict", "").strip(" :")
        if bare in labels:
            return bare
    for line in reversed(lines):  # 2순위: 라벨이 정확히 하나 있는 마지막 줄
        present = [lab for lab in labels if lab in line]
        if len(present) == 1:
            return present[0]
    pos = {lab: (text or "").rfind(lab) for lab in labels}
    pos = {k: v for k, v in pos.items() if v >= 0}
    return max(pos, key=pos.get) if pos else None


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
        out = find(r, "judge_output")
        if out is None and find(r, "verdict") == "abstain":  # 정규화 뒤 거절 토큰과 같으면 서버가 judge 를 부르지 않는다 (\boxed{IDK} 포함)
            return False, False, 0, 0, None
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
    r["_alpha_verifier"] = (0, 0)
    if isinstance(labels_map, dict) and labels_map:
        nv = sum(1 for _ in labels_map)
        nb = sum(1 for lab in labels_map.values() if lab is None)
        r["_alpha_verifier"] = (nv, nb)
        n += nv
        bad += nb
        if nb:
            ex = ex or f"verifier label None {nb}/{nv}"
    kind = "over_refusal" if agent.startswith("over_refusal") else "jailbreak"
    mismatch = 0
    for key, labs in (("judge_evaluation", SAFETY_LABELS[kind]), ("quality_judge_evaluation", SAFETY_LABELS["quality"])):
        ev = r.get(key) if isinstance(r.get(key), dict) else find(r, key)
        if not isinstance(ev, dict):
            continue
        n += 1
        text = last_text(ev.get("response") or {})
        lab = ev.get("verdict_label")
        if lab is None or ev.get("error") or lab != last_label(text, labs):
            bad += 1
            mismatch += int(lab is not None and not ev.get("error") and lab != last_label(text, labs))
            ex = ex or (ev.get("error") or f"서버 {lab} · 원문 마지막 {last_label(text, labs)} · {text[-200:]}")
    r["_alpha_label_mismatch"] = mismatch
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
    groups = collections.defaultdict(list)
    for line in open(a.groups):
        g = json.loads(line)
        if g.get("time", 1e18) < a.since or g.get("dropped"):
            continue
        groups[g.get("agent") or "?"].append(g)
    # GenRM 표 행은 genrm_compare 가 필드 3개짜리 BaseVerifyResponse 로 돌려줘 dataset · principle 이 없다 → identity 는 그룹 기록 dataset 으로 센다
    ident_groups = sum(1 for ag in GENRM for g in groups.get(ag, []) if str(g.get("dataset") or "").startswith("identity"))
    strata = set(rows)
    expected = set(RULE) | set(GENRM) | {x for g in JUDGE_GROUPS.values() for x in g}
    check("P3 agent 16 + identity 가 기록에 있음", expected <= strata and ident_groups > 0,
          f"표에 없는 agent {sorted(expected - strata)} · identity 그룹 {ident_groups}")

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
        check(f"P5c {g} 엄격 파싱 실패 ≤ 3%", n > 0 and bad <= 0.03 * n, f"{bad}/{n}")
        if g == "safety":
            mm = sum(r.get("_alpha_label_mismatch", 0) for ag in agents for r in rows.get(ag, []))
            legacy_n = sum(1 for ag in agents for r in rows.get(ag, []) if isinstance(find(r, "judge_evaluation"), dict))
            check("P5d 안전 judge 서버 판정 == 최종 판정 라벨, 불일치 ≤ 0.5%", mm <= 0.005 * max(legacy_n, 1),
                  f"불일치 {mm}/{legacy_n} (정책별 verifier 는 원문이 없어 이 대리 지표로 본다)")
            pol_elig = [r for ag in POLICY_AGENTS for r in rows.get(ag, []) if str(find(r, "model_response_text") or "").strip()]
            pol_on = sum(1 for r in pol_elig if find(r, "response_policy") or isinstance(find(r, "verifier_labels"), dict))
            check("P5f 정책별 jailbreak 경로가 실제로 쓰임 (정책 agent 대상 행의 ≥ 95%)", len(pol_elig) > 0 and pol_on >= 0.95 * len(pol_elig),
                  f"{pol_on}/{len(pol_elig)} — templates 를 못 읽거나 정책 이름이 map 에 없으면 서버가 조용히 legacy 판정으로 바꾼다")
            vn = sum(r.get("_alpha_verifier", (0, 0))[0] for ag in agents for r in rows.get(ag, []))
            vb = sum(r.get("_alpha_verifier", (0, 0))[1] for ag in agents for r in rows.get(ag, []))
            pol_rows = sum(1 for ag in agents for r in rows.get(ag, []) if find(r, "response_policy") or find(r, "verifier_labels") is not None)
            check("P5e 정책별 jailbreak verifier 라벨 None ≤ 3% (안전 묶음에 희석되지 않게 따로)", (vn > 0 or pol_rows == 0) and vb <= 0.03 * max(vn, 1),
                  f"None {vb}/{vn}")
        for ex in examples:
            print("      실패 예:", repr(ex))

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
    check("P6b alpha_genrm 파싱 실패 ≤ 3% (부하 측정 기준선 3/270 = 1.1%, 실패는 genrm_compare 가 재시도)", req > 0 and tot["parse_failed"] <= 0.03 * req,
          f"{tot['parse_failed']}/{req} (워커 {len(last)}개, 10회 단위 스냅샷)")
    ranks = [tot[f"rank{i}"] for i in range(1, 7)]
    r1 = sum(ranks[:3]) / max(1, sum(ranks))
    check("P6c 순위 분포: response_1 우세 비율 0.2~0.8 (위치 편향·상수 판정)", sum(ranks) > 0 and 0.2 <= r1 <= 0.8,
          f"순위 1~6 = {ranks}, response_1 우세 {r1:.2f}")
    tie = tot["score_tie"] / max(1, tot["parsed"])
    check("P6d 개별 점수 동점 ≤ 90% (항상 동점만 잡는다 — 템플릿이 동점을 허용하고 기준선이 없다)", tot["parsed"] > 0 and tie <= 0.9,
          f"{tot['score_tie']}/{tot['parsed']} = {tie:.2f}")
    check("P6f GenRM 생존: 어댑터 파싱 성공 ≥ 0.9 × 16 × GenRM 그룹", n_genrm_groups > 0 and tot["parsed"] >= 0.9 * expect_cmp,
          f"파싱 성공 {tot['parsed']} / 기대 비교 {expect_cmp} (스냅샷이 워커마다 최대 9 적게 잡는다)")
    # identity 행은 모두 principle 을 갖는다 → 16 × identity 그룹 이 principle 실린 비교 수의 하한 (Ultra GenRM 행의 기준까지 더하면 더 크다)
    check("P6e principle 주입 ≥ 0.5 × 16 × identity 그룹", ident_groups > 0 and tot["principle_injected"] >= 0.5 * 16 * ident_groups,
          f"주입 {tot['principle_injected']} / 하한 {16 * ident_groups}")
    print("[기록] 그룹 보상이 모두 같은 그룹 비율 (판정에 쓰지 않음):",
          {ag: round(sum(1 for g in gs if len(set(g['rewards'])) <= 1) / len(gs), 2) for ag, gs in sorted(groups.items())})
    n_fail = sum(1 for _, ok in verdicts if not ok)
    print(f"JG1 분석기 판정: {'PASS' if n_fail == 0 else 'FAIL'} ({len(verdicts) - n_fail}/{len(verdicts)}) — P1·P2 (rc · 배치 · KL · 마스킹) 는 로그·TensorBoard 로 따로 본다")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
