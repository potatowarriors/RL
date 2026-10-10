r"""Gym 전체 결과(W&B 표)로 judge 판정 파싱 실패 · 재판정 · rdkit 추출 실패를 센다 (CPU, 2026-10-10 — GATES.md J1).

`env.should_log_nemo_gym_responses: true` 인 레시피는 `train_data_step*.jsonl` 을 쓰지 않는다. 전체 결과는
`logger.wandb.log_nemo_gym_full_result_tables=true` 일 때만 W&B 표로 남는다. 게이트는 W&B 에 올리지 않으므로 오프라인으로 남긴다:
  EXTRA_ENV="WANDB_MODE=offline" WANDB=1 launch.sh ... logger.wandb.metric_allowlist=null ++logger.wandb.log_nemo_gym_full_result_tables=true
  python3 examples/configs/alpha/tools/analyze_gym_full_results.py <log_dir>/exp_*/wandb [--show 3]
판정 (Gym equivalence_llm_judge app.py `_generate_judge_evaluation`):
  - 파싱 실패 = judge 출력에 `[[A=B]]` · `[[A!=B]]` 가 둘 다 없음 (`verdict_label` None → 같지 않음으로 처리)
  - 재판정 = 첫 판정이 같지 않고 행에 `output_regex` 가 있어 최종 응답 전체로 한 번 더 판정 (평가 2개). 맞으면 보상 0.5
  - 재판정 종류: 첫 판정에 보낸 추출문이 최종 응답 전체 안에서 닫는 문자(`)` `]` `}` `**` `$` `\` `.`)만 남기고 끝나면 "완전 추출" —
    잘린 답을 살린 것이 아니라 같은 답을 문맥과 함께 다시 물어 판정이 뒤집힌 것이다. 추출문 뒤에 답이 더 이어지면 "잘린 추출" (결정 25 가 겨냥한 경우)
  - ns_tools 행은 `delegated_response` 안의 equivalence_llm_judge 결과를 본다
rdkit (alpha 플러그인): `extracted_answer` None = 지시한 형식(\boxed{N} · ((N)))의 정수를 못 읽음. 최종 답 본문이 비면 생성 상한 잘림으로 본다.
"""
import argparse
import collections
import glob
import json
import os
import re

JUDGE_AGENTS = ("equivalence_llm_judge_simple_agent", "ns_tools_simple_agent")


def load(root):
    rows = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(root, "**", "*.table.json"), recursive=True)):
        agent = os.path.basename(os.path.dirname(f))
        t = json.load(open(f))
        for row in t["data"]:
            rows[agent].append(json.loads(row[0]))
    return rows


def output_text(r):
    out = []
    for item in (r.get("response") or {}).get("output") or []:
        if item.get("type") == "message":
            out += [c.get("text", "") for c in item.get("content") or []]
    return "".join(out)


def bucket(x):
    return {0.0: "0", 0.5: "0.5", 1.0: "1"}.get(round(float(x), 6), "other")


CLOSE_ONLY = re.compile(r"^[\s\)\]\}\.\*\$\\]*$")


def judge_candidate(ev):
    c = ev["responses_create_params"]["input"][-1]["content"]
    return (c if isinstance(c, str) else json.dumps(c)).split("CANDIDATE:")[-1].strip()


def rescue_kind(ev1, ev2):
    ext, full = judge_candidate(ev1), judge_candidate(ev2)
    i = full.rfind(ext)
    if i < 0:
        return "추출문 미발견"
    return "완전 추출(판정 뒤집힘)" if CLOSE_ONLY.match(full[i + len(ext):]) else "잘린 추출"


def judge_view(r):
    j = r.get("delegated_response") if "judge_evaluations" not in r else r
    return j if j and "judge_evaluations" in j else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--show", type=int, default=2)
    a = ap.parse_args()
    rows = load(a.root)
    print("표본:", {k: len(v) for k, v in rows.items()})
    for agent, rs in sorted(rows.items()):
        print(f"\n## {agent}  n={len(rs)}  보상 {dict(collections.Counter(bucket(r['reward']) for r in rs))}")
        if agent in JUDGE_AGENTS:
            calls = fails = no_view = 0
            resc = collections.Counter()
            shown = 0
            for r in rs:
                j = judge_view(r)
                if j is None:
                    no_view += 1
                    continue
                ev = j["judge_evaluations"]
                calls += len(ev)
                fails += sum(e.get("verdict_label") is None for e in ev)
                if len(ev) == 2:
                    ok = bucket(j["reward"]) == "0.5"
                    resc["시도"] += 1
                    resc["성공(0.5)" if ok else "실패"] += 1
                    resc[f"{'성공' if ok else '실패'} · {rescue_kind(*ev)}"] += 1
                    if bucket(j["reward"]) == "0.5" and shown < a.show:
                        shown += 1
                        first = ev[0]["responses_create_params"]["input"][-1]["content"]
                        print(f"  [재판정 성공 표본] 정답={j['expected_answer'][:120]!r}\n    첫 판정 입력 끝: {str(first)[-300:]!r}")
            print(f"  judge 호출 {calls} · 파싱 실패 {fails} ({100 * fails / max(calls, 1):.2f}%) · judge 결과 없음 {no_view}")
            print(f"  재판정 {dict(sorted(resc.items()))}")
        if agent == "rdkit_chemistry_agent":
            kinds = collections.Counter()
            shown = 0
            for r in rs:
                if r.get("extracted_answer") is None:
                    txt = output_text(r)
                    k = "추출 실패 · 최종 답 없음(잘림)" if not txt.strip() else "추출 실패 · 형식 불일치"
                    if k.endswith("불일치") and shown < a.show:
                        shown += 1
                        print(f"  [형식 불일치 표본] box={r.get('use_box_format')} 끝 300자: {txt[-300:]!r}")
                else:
                    k = "정답" if bucket(r["reward"]) == "1" else "오답"
                kinds[k] += 1
            print(f"  rdkit {dict(kinds)}")


if __name__ == "__main__":
    main()
