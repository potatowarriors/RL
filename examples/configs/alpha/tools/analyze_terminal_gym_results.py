r"""terminal 피벗 스모크 게이트 (TS1) — 학습 경로가 남긴 Gym 전체 결과로 채점기가 의도대로 붙었는지 본다 (CPU, RL_PLAN.md 결정 68).

2026-11-16 블렌드 검토의 지적: judge 가 일부만 실패하면 Gym `judge_failsafe` 가 보상 0 으로 바꾸고 NeMo-RL 은 그대로 학습에 쓴다 —
평균 보상 범위만 보면 통과한다. 플러그인 대신 Gym 원본 서버가 뜨거나 빠른 통과 임계가 다른 값이어도 평균은 거의 같다.
그래서 평균이 아니라 **행 단위 불변식**으로 본다.

입력: W&B 오프라인 표 (`logger.wandb.log_nemo_gym_full_result_tables=true`, tools/analyze_gym_full_results.py 와 같은 형식) 또는 `--jsonl`.
  EXTRA_ENV="WANDB_MODE=offline UV_OFFLINE=1" WANDB=1 launch.sh <campaign> <tag> ... logger.wandb.metric_allowlist=null \
      ++logger.wandb.log_nemo_gym_full_result_tables=true grpo.max_num_steps=3
  python3 examples/configs/alpha/tools/analyze_terminal_gym_results.py <log_dir>/exp_*/wandb --blend <blend.jsonl> [--profile <term_profile_judge_v2.json>]

기준 (하나라도 어기면 종료 코드 1):
  I1  judge 장애 0 — `_ng_failure_class` 가 있거나, 사유가 unknown_error 이거나, `failure_reason` · `model_output` 키가 없는 행
  I2  완료 선언 양방향 — 정답이 완료가 아닌데 후보가 완료를 선언한 행이 `--min-early-tc` (기본 20) 개 이상 있고, 그 행 전부가
      보상 0 · 사유 task_complete_check_failed · judge 판정 0 개 (플러그인만의 서명 — Gym 원본 서버는 judge 로 보낸다)
  I3  빠른 통과는 완전 일치만 — judge 판정 없이 보상 1 인 행의 유사도는 전부 1.0; 스키마 · 완료 검사를 통과하고 유사도 < 1.0 인 행은
      judge 판정이 정확히 1 개이고, 그런 행이 전체의 50% 이상 (검사 대상이 없어서 통과하지 않게)
  I4  judge 모델 — judge 응답의 model 이 `--judge-name` (기본 gemma) 을 담는다 (정책 모델이 채점하지 않는다)
  I5  judge 판정 파싱 실패 ≤ 2% (judge 호출 대비)
  I6  한 번만 호출 — 응답의 reasoning 항목이 2개 이상인 행 0 (`max_steps: 1`)
  R1  사유 분포가 프로파일 기준 (블렌드 행의 재채점 분포) 과 ±`--tol` (기본 0.06) 안
  R2  평균 보상이 기준 평균 × [0.6, 1.4] 안
보고: 넓은 반복류 출력의 양의 advantage 몫 · 들림 (uuid 로 묶어 계산) · `schema_check_passed` 평균 (본 런 매 step 감시의 기준값).
자가 시험: 실제 행을 변조한 5 종 (judge 장애 행 · 조기 완료에 보상 1 · 유사도 0.97 인데 judge 없이 보상 1 · judge 모델 이름 alpha ·
`judge_evaluations` 키 삭제) 이 각각 FAIL 하는지 먼저 본다. 하나라도 PASS 면 도구가 고장이므로 종료 코드 2.
"""
import argparse
import collections
import glob
import importlib.util
import json
import os
import sys
from pathlib import Path


def load_tables(root):
    rows, seen = [], set()
    for f in sorted(glob.glob(os.path.join(root, "**", "*.table.json"), recursive=True)):
        t = json.load(open(f))
        for row in t["data"]:
            if row[0] in seen:
                continue
            seen.add(row[0])
            rows.append(json.loads(row[0]))
    return rows


def parse(text):
    if not isinstance(text, str):
        return None
    text = text.split("</think>")[-1].strip() if "</think>" in text else text
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) else None


def evaluate(rows, a, tbl, base_dist, base_mean):
    fails, n = [], len(rows)
    reasons = collections.Counter()
    c = collections.Counter()
    judge_models = collections.Counter()
    groups = collections.defaultdict(list)
    for r in rows:
        if r.get("_ng_failure_class") or "failure_reason" not in r or "model_output" not in r:
            c["judge_or_server_failure"] += 1
            reasons["(judge 장애 · 키 없음)"] += 1
            groups[r.get("uuid")].append((0.0, False))
            continue
        reason = str(r.get("failure_reason") or "none")
        reasons[reason] += 1
        if reason == "unknown_error":
            c["judge_or_server_failure"] += 1
        exp = parse(r.get("expected_answer"))
        pred = r.get("parsed_output") if isinstance(r.get("parsed_output"), dict) else parse(r.get("model_output"))
        if "judge_evaluations" not in r:
            c["no_judge_evaluations_key"] += 1
        evs = r.get("judge_evaluations") or []
        sim = r.get("similarity_score")
        reward = r.get("reward")
        if exp is not None and pred is not None and pred.get("task_complete") and not exp.get("task_complete"):
            c["early_tc"] += 1
            if reward != 0.0 or reason != "task_complete_check_failed" or evs:
                c["early_tc_not_plugin_signature"] += 1
        if reward == 1.0 and not evs and sim != 1.0:
            c["fast_pass_not_exact"] += 1
        if r.get("schema_check_passed") and r.get("task_complete_check_passed") and isinstance(sim, (int, float)) and sim < 1.0:
            c["should_judge"] += 1
            if len(evs) != 1:
                c["judge_count_not_1"] += 1
        if r.get("schema_check_passed"):
            c["schema_passed"] += 1
        for e in evs:
            c["judge_calls"] += 1
            m = str(((e.get("response") or {}).get("model")) or "")
            judge_models[m] += 1
            if a.judge_name not in m.lower():
                c["judge_wrong_model"] += 1
            if e.get("verdict_label") is None:
                c["judge_parse_fail"] += 1
        out = (r.get("response") or {}).get("output") or []
        if sum(1 for o in out if o.get("type") == "reasoning") > 1:
            c["multi_call"] += 1
        hist = []
        for m in ((r.get("responses_create_params") or {}).get("input") or []):
            if m.get("role") == "assistant" and isinstance(m.get("content"), str):
                h = parse(m["content"])
                hist.append(tbl.ks(h) if h else [])
        rep = bool(pred is not None and not pred.get("task_complete") and tbl.repeat_like(tbl.ks(pred), hist))
        groups[r.get("uuid")].append((float(reward or 0.0), rep))
    if n < a.min_rows:
        fails.append(f"행 {n} < {a.min_rows}")
    for key, msg in (("judge_or_server_failure", "I1 judge 장애 · 키 없는 행"), ("early_tc_not_plugin_signature", "I2 조기 완료 선언이 플러그인 서명과 다름"),
                     ("fast_pass_not_exact", "I3 완전 일치가 아닌 빠른 통과"), ("judge_count_not_1", "I3 judge 판정 수가 1 이 아닌 행"),
                     ("no_judge_evaluations_key", "I3 judge_evaluations 키 없는 행"),
                     ("judge_wrong_model", "I4 judge 응답의 모델이 다름"), ("multi_call", "I6 두 번 이상 호출")):
        if c[key]:
            fails.append(f"{msg} {c[key]}")
    if c["early_tc"] < a.min_early_tc:
        fails.append(f"I2 조기 완료 선언 행 {c['early_tc']} < {a.min_early_tc} — 검사 대상 부족")
    if c["should_judge"] < 0.5 * n:
        fails.append(f"I3 judge 대상 행 {c['should_judge']} < 전체의 50% — 검사 대상 부족")
    if not c["judge_calls"]:
        fails.append("I4 judge 호출 0")
    elif c["judge_parse_fail"] / c["judge_calls"] > 0.02:
        fails.append(f"I5 판정 파싱 실패 {c['judge_parse_fail'] / c['judge_calls']:.3f} > 0.02")
    if len(groups) < 2 or None in groups:
        fails.append("uuid 로 묶을 수 없음")
    dist = {k: round(v / n, 4) for k, v in reasons.most_common()}
    for k in set(dist) | set(base_dist):
        if k != "recall" and abs(dist.get(k, 0.0) - base_dist.get(k, 0.0)) > a.tol:
            fails.append(f"R1 사유 {k}: {dist.get(k, 0.0)} 대 기준 {base_dist.get(k, 0.0)}")
    mean = sum(1 for r in rows if r.get("reward") == 1.0) / n
    if not (0.6 * base_mean <= mean <= 1.4 * base_mean):
        fails.append(f"R2 평균 보상 {mean:.3f} 대 기준 {base_mean:.3f}")
    pos, neg = collections.Counter(), collections.Counter()
    for g in groups.values():
        r_ = [x[0] for x in g]
        mu = sum(r_) / len(r_)
        sd = (sum((x - mu) ** 2 for x in r_) / len(r_)) ** 0.5
        if sd == 0:
            continue
        for rw, rep in g:
            adv = (rw - mu) / sd
            (pos if adv > 0 else neg)["rep" if rep else "other"] += abs(adv)
    tp = sum(pos.values()) or 1.0
    summ = {"rows": n, "prompts": len(groups), "mean_reward": round(mean, 4), "baseline_mean": round(base_mean, 4), "reasons": dist,
            "baseline_reasons": base_dist, "counts": dict(c), "judge_models": dict(judge_models),
            "schema_check_passed_mean": round(c["schema_passed"] / n, 4),
            "repeat_like_pos_share": round(pos["rep"] / tp, 4), "repeat_like_lift": round(pos["rep"] / neg["rep"], 3) if neg["rep"] else None,
            "verdict": "PASS" if not fails else "FAIL", "fails": fails}
    return summ, fails


def tampers(rows):
    """실제 행을 변조한 음성 대조 5 종. 각각 evaluate 가 FAIL 해야 한다."""
    import copy
    out = {}
    t = copy.deepcopy(rows)
    for r in t[: max(1, len(t) // 20)]:
        r.pop("failure_reason", None)
        r.pop("model_output", None)
        r["_ng_failure_class"] = "judge_failed"
        r["reward"] = 0.0
    out["judge 장애 5%"] = t
    t = copy.deepcopy(rows)
    hit = [r for r in t if str(r.get("failure_reason")) == "task_complete_check_failed" and not (r.get("judge_evaluations") or [])]
    for r in hit[:1]:
        r["reward"], r["failure_reason"] = 1.0, "none"
    out["조기 완료에 보상 1"] = t if hit else None
    t = copy.deepcopy(rows)
    hit = [r for r in t if str(r.get("failure_reason")) == "judge_not_equivalent"]
    for r in hit[:1]:
        r["reward"], r["failure_reason"], r["judge_evaluations"], r["similarity_score"] = 1.0, "none", [], 0.97
    out["유사도 0.97 빠른 통과"] = t if hit else None
    t = copy.deepcopy(rows)
    for r in t:
        for e in r.get("judge_evaluations") or []:
            if isinstance(e.get("response"), dict):
                e["response"]["model"] = "alpha"
    out["judge 모델 이름 alpha"] = t
    t = copy.deepcopy(rows)
    for r in t:
        r.pop("judge_evaluations", None)
    out["judge_evaluations 키 삭제"] = t
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?")
    ap.add_argument("--jsonl")
    ap.add_argument("--blend", required=True)
    ap.add_argument("--profile", default="/home/work/vidsearch/tools/nemo_rl/p0_measure/term_pivot/rejudge/term_profile_judge_v2.json")
    ap.add_argument("--judge-name", default="gemma")
    ap.add_argument("--tol", type=float, default=0.06)
    ap.add_argument("--min-rows", type=int, default=1024)
    ap.add_argument("--min-early-tc", type=int, default=20)
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.jsonl)] if a.jsonl else load_tables(a.root)
    if not rows:
        print(json.dumps({"rows": 0, "verdict": "FAIL", "fails": ["행 0"]}, ensure_ascii=False))
        return 1
    spec = importlib.util.spec_from_file_location("tbl", Path(__file__).with_name("build_terminal_profile_table.py"))
    tbl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tbl)
    prof = json.load(open(a.profile))
    base, tot = collections.Counter(), 0
    for line in open(a.blend):
        for o in prof[json.loads(line)["uuid"]]["o"]:
            base[o[1]] += 1
            tot += 1
    base_dist = {k: round(v / tot, 4) for k, v in base.most_common()}
    base_mean = base["none"] / tot
    summ, fails = evaluate(rows, a, tbl, base_dist, base_mean)
    self_ok, self_log = True, {}
    for name, t in tampers(rows).items():
        if t is None:
            self_log[name] = "만들 수 없음 (대상 행 없음)"
            self_ok = False
            continue
        _, f = evaluate(t, a, tbl, base_dist, base_mean)
        new = [x for x in f if x not in fails]
        self_log[name] = "FAIL (정상)" if new else "PASS — 도구가 못 잡음"
        self_ok &= bool(new)
    summ["selfcheck"] = self_log
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    if not self_ok:
        print("자가 시험 실패 — 도구를 신뢰할 수 없다")
        return 2
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
