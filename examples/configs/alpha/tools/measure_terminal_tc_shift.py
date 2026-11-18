r"""terminal 피벗 — 완료 선언 경계가 학습으로 어디서 움직였는지 직접 잰다 (GPU 는 실행기, 이 파일은 CPU). KNOWN_ISSUES 2026-11-17.

배경: 1 에폭 모델 (B) 이 시작점 (A) 보다 TB-2.1 에서 약 10 step 일찍 완료를 선언한다. 원인 후보 — (가) 완료 정답 행의 선별 편향
(나) 완료 정답 행 자체 (다) 반복 · 재실행 정답 제거에 따른 재검증 억제 · 막히면 선언 — 를 가르려고, 같은 입력에 대한 두 모델의 출력 종류를 비교한다.
채점 보상은 쓰지 않는다 (출력 파싱만 — judge 불필요).

층 (학습 블렌드에 든 행은 모두 제외. 학습에 쓰지 않은 과제의 행을 먼저 뽑고, 모자라면 학습 과제의 블렌드 밖 행으로 채워
`heldout_task=false` 로 표시한다. 이력에 이미 완료 선언이 있거나 마지막 메시지가 완료 확인 질문인 행은 T 층에서 뺀다 — 거기서의 선언은
확인 답변이다):
  T1  정답 완료 ∧ 프로파일 통과 1~3/8 (블렌드 선별 기준과 같음, 학습에는 안 쓰인 행)     선별된 완료 행과 닮은 상태
  T2  정답 완료 ∧ 프로파일에서 A 의 완료 출력 ≥ 4/8 ∧ T1 아님                          선별되지 않는 완료 상태 (천장 근처 — 상대 차로 본다)
  C1  정답 명령 ∧ 같은 궤적의 첫 완료 정답까지 1~2 턴 ∧ 그 궤적의 완료 행이 학습에 없음    조기 선언 경계
  C2  정답 명령 ∧ 첫 완료 정답까지 6 턴 이상 ∧ 깊이 (앞선 턴) 8 이상                     깊이를 맞춘 음성 대조
  S   A 의 TB-2.1 궤적에서 직전 명령 반복이 처음 시작된 step 직전까지의 접두 (막힌 상태)
  S0  같은 A 궤적들에서 반복이 시작되지 않는 step 직전까지의 접두 (TB 형식 상태의 대조)
      S · S0 는 궤적의 analysis · plan · tool_calls 로 Terminus-2 JSON 을 다시 조립한다 — 원 응답과 글자 단위로 같지 않고 이력의 추론이 없다.

  build    --out <dir>/input.jsonl [--tb-prefixes <jsonl>] [--seed 20261118]     층별 입력 (원천 행 그대로 + `probe` 필드)
  analyze  --a <rollouts.jsonl> --b <rollouts.jsonl> --input <input.jsonl>      A 는 16회 (앞 8 · 뒤 8 로 잡음 바닥), B 는 8회
출력 종류: recall (모델을 두 번 이상 부름 — string_only agent 는 max_steps 가 없다) · truncated (생성 상한에서 잘림) · invalid (JSON 을
못 뽑음; Terminus-2 처럼 앞뒤 텍스트 · 코드 펜스는 벗겨 본다) · tc (완료 선언) · repeat_like (넓은 반복류) · other.
완료 출력률은 전체 출력 대비와 파싱된 출력 (tc · repeat_like · other) 대비를 둘 다 낸다.

유효성 조건 (하나라도 어기면 종료 코드 1 — 결과를 해석하지 않는다; 2026-11-18 적대적 검토에서 실행 전에 고정):
  V1  모든 층에서 잡음 바닥 (A 뒤 8 − A 앞 8) 의 완료 출력률 99.2% 구간 (6 층 Bonferroni) 이 0 을 포함
  V2  두 모델 모두 recall · truncated 가 각각 3% 미만
  V3  S 재현 대조: A 의 S 층 repeat_like ≥ 20% 이고 C2 의 3 배 이상
  V4  양성 대조: S 층의 repeat_like 차 (B − A) 의 95% 구간 상한 < 0 (TB 에서 반복이 23.8 → 14.1% 로 줄었다)
해석 규칙:
  경계 이동     C1 (학습 밖 과제) 의 완료 차 ≥ +3.5 pt ∧ 구간 하한 > 0 ∧ C2 의 완료 차가 그 1/3 이하. C2 도 같이 오르면 위치 규칙 · 전역 편향
  막히면 선언   완료 차 (S) − 완료 차 (S0) 의 구간 하한 > 0 ∧ S 의 repeat_like 차 < 0
  (가) 대 (나) 는 이 측정으로 가르지 못한다 (B 는 선별된 완료 행으로만 학습했다) — T1 · T2 는 일반화 범위만 보여 준다.
"""
import argparse
import collections
import importlib.util
import json
import random
import re
from pathlib import Path

RL = "/home/work/Datasets/LL_datasets/posttraining/RL"
SOURCE = f"{RL}/Nemotron-RL-Agentic-Terminal-Pivot-v1/atcb_terminal_pivot_release_final_v2.jsonl"
PROFILE = "/home/work/vidsearch/tools/nemo_rl/p0_measure/term_pivot/rejudge/term_profile_judge_v2.json"
TRAIN = f"{RL}/alpha_blends/teacher_blends/teacher_terminal_pivot_judge_v4.jsonl"
SIZES = {"T1": 150, "T2": 150, "C1": 200, "C2": 150, "S": 150, "S0": 150}
CONFIRM = "Are you sure you want to mark the task as complete"


def _tbl():
    spec = importlib.util.spec_from_file_location("tbl", Path(__file__).with_name("build_terminal_profile_table.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build(a) -> int:
    rng = random.Random(a.seed)
    prof = json.load(open(PROFILE))
    train_rows, train_tasks = set(), set()
    for line in open(TRAIN):
        r = json.loads(line)
        train_rows.add(r["uuid"])
        train_tasks.add(r["task_name"])
    meta, by_traj, prior_tc = {}, collections.defaultdict(list), set()
    for line in open(SOURCE):
        r = json.loads(line)
        md = r["metadata"]
        msgs = r["responses_create_params"]["input"]
        if CONFIRM in str(msgs[-1].get("content")) or any(
                m.get("role") == "assistant" and '"task_complete": true' in str(m.get("content")) for m in msgs):
            prior_tc.add(r["uuid"])
        meta[r["uuid"]] = (r["task_name"], md["source_trajectory_uid"], md["pivot_agent_turn_index"])
        by_traj[md["source_trajectory_uid"]].append((md["pivot_agent_turn_index"], r["uuid"]))
    first_tc = {}
    for tid, rows in by_traj.items():
        tcs = [i for i, u in rows if prof[u]["g_tc"]]
        if tcs:
            first_tc[tid] = min(tcs)
    train_tc_traj = {meta[u][1] for u in train_rows if prof[u]["g_tc"]}
    cand = collections.defaultdict(list)
    for u, d in prof.items():
        if d["ctrl"] or u in train_rows:
            continue
        task, tid, idx = meta[u]
        if d["g_tc"]:
            if u in prior_tc:
                continue
            n_pass = sum(1 for o in d["o"] if o[0] == 1.0)
            n_tc = sum(1 for o in d["o"] if o[3] == "tc")
            if 1 <= n_pass <= 3:
                cand["T1"].append(u)
            elif n_tc >= 4:
                cand["T2"].append(u)
        elif tid in first_tc:
            dist = first_tc[tid] - idx
            if 1 <= dist <= 2 and tid not in train_tc_traj and u not in prior_tc:
                cand["C1"].append(u)
            elif dist >= 6 and idx >= 8:
                cand["C2"].append(u)
    chosen = {}
    for layer in ("T1", "T2", "C1", "C2"):
        held = sorted(u for u in cand[layer] if meta[u][0] not in train_tasks)
        rest = sorted(u for u in cand[layer] if meta[u][0] in train_tasks)
        rng.shuffle(held)
        rng.shuffle(rest)
        pick = (held + rest)[: SIZES[layer]]
        for u in pick:
            chosen[u] = {"layer": layer, "heldout_task": meta[u][0] not in train_tasks}
        print(layer, "후보", len(cand[layer]), "학습 밖 과제", len(held), "→ 선택", len(pick), "(학습 밖 과제", sum(chosen[u]["heldout_task"] for u in pick), ")")
    n = 0
    with open(a.out, "w") as fo:
        for line in open(SOURCE):
            r = json.loads(line)
            if r["uuid"] in chosen:
                r["probe"] = chosen[r["uuid"]]
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
        if a.tb_prefixes:
            rows = [json.loads(line) for line in open(a.tb_prefixes)]
            rows.sort(key=lambda x: (x["kind"], x["trial"]))
            rng.shuffle(rows)
            template = json.loads(open(SOURCE).readline())
            for kind in ("S", "S0"):
                seen_task, k = collections.Counter(), 0
                for x in rows:
                    if x["kind"] != kind or k >= SIZES[kind] or seen_task[x["task"]] >= 3:
                        continue
                    seen_task[x["task"]] += 1
                    r = {"schema_version": template["schema_version"], "uuid": f"tbprefix-{kind}-" + x["trial"], "task_name": "tb21/" + x["task"],
                         "tool_name": template["tool_name"], "expected_answer": x["next_action"], "agent_ref": template["agent_ref"],
                         "metadata": {"harness": "terminus_2", "source": "tb21f40571b2", "trial": x["trial"], "prefix_agent_steps": x["n_agent"]},
                         "responses_create_params": {"input": x["input"]}, "probe": {"layer": kind, "heldout_task": True}}
                    fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                    k += 1
                    n += 1
                print(kind, "후보", sum(x["kind"] == kind for x in rows), "→ 선택", k, "과제", len(seen_task))
    print("rows", n, "→", a.out)
    return 0


def parse_action(row):
    """Terminus-2 처럼 응답에서 JSON 객체를 뽑는다: 채점기의 parsed_output → 없으면 model_output 의 첫 `{` ~ 마지막 `}`."""
    po = row.get("parsed_output")
    if isinstance(po, dict):
        return po
    text = row.get("model_output")
    if not isinstance(text, str):
        return None
    text = text.split("</think>")[-1]
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        d = json.loads(text[i:j + 1])
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) else None


def category(tbl, row) -> str:
    resp = row.get("response") or {}
    out = resp.get("output") or []
    if sum(1 for o in out if o.get("type") == "reasoning") > 1:
        return "recall"
    if resp.get("status") == "incomplete" or resp.get("incomplete_details"):
        return "truncated"
    po = parse_action(row)
    if po is None:
        # JSON 이 깨졌어도 완료 선언 문구가 있으면 따로 센다 (보조 지표 — 전체 출력의 약 0.5%)
        return "invalid_tc_text" if re.search(r'"task_complete"\s*:\s*true', str(row.get("model_output"))) else "invalid"
    if po.get("task_complete") is True or po.get("is_task_complete") is True:
        return "tc"
    hist = []
    for m in row["responses_create_params"]["input"]:
        if m.get("role") == "assistant" and isinstance(m.get("content"), str):
            try:
                hist.append(tbl.ks(json.loads(m["content"])))
            except json.JSONDecodeError:
                hist.append([])
    return "repeat_like" if tbl.repeat_like(tbl.ks(po), hist) else "other"


CATS = ("tc", "repeat_like", "other", "invalid", "invalid_tc_text", "recall", "truncated")


def load_rollouts(path, tbl):
    out = collections.defaultdict(list)
    for line in open(path):
        x = json.loads(line)
        out[x["uuid"]].append((x.get("_ng_rollout_index", 0), category(tbl, x)))
    return {u: [c for _, c in sorted(v)] for u, v in out.items()}


def boot(pairs, rng, n=4000, level=0.95):
    """과제 단위 클러스터 부트스트랩 — pairs = [(과제, 값)]. 과제를 재표본하고 그 과제의 행을 모두 넣는다 (같은 과제의 행은 독립이 아니다)."""
    if not pairs:
        return (None, None, None)
    by = collections.defaultdict(list)
    for t, v in pairs:
        by[t].append(v)
    tasks = sorted(by)
    m = sum(v for _, v in pairs) / len(pairs)
    bs = []
    for _ in range(n):
        tot = cnt = 0
        for t in rng.choices(tasks, k=len(tasks)):
            tot += sum(by[t])
            cnt += len(by[t])
        bs.append(tot / cnt)
    bs.sort()
    lo = (1 - level) / 2
    return (round(m, 4), round(bs[int(lo * n)], 4), round(bs[min(n - 1, int((1 - lo) * n))], 4))


def frac(xs, cat):
    return sum(c == cat for c in xs) / len(xs)


def frac_parsed(xs, cat="tc"):
    ok = [c for c in xs if c in ("tc", "repeat_like", "other")]
    return (sum(c == cat for c in ok) / len(ok)) if ok else None


def analyze(a) -> int:
    tbl = _tbl()
    rng = random.Random(0)
    layer, task = {}, {}
    for line in open(a.input):
        r = json.loads(line)
        layer[r["uuid"]] = r["probe"]
        task[r["uuid"]] = r["task_name"]
    A, B = load_rollouts(a.a, tbl), load_rollouts(a.b, tbl)
    bad = {u for u in layer if len(A.get(u, [])) != 16 or len(B.get(u, [])) != 8}
    fails = []
    if len(bad) > 0.02 * len(layer):
        fails.append(f"롤아웃 수가 맞지 않는 행 {len(bad)} > 2%")
    allA = [c for u in layer if u not in bad for c in A[u]]
    allB = [c for u in layer if u not in bad for c in B[u]]
    overall = {m: {c: round(frac(xs, c), 4) for c in CATS} for m, xs in (("A", allA), ("B", allB))}
    for m in ("A", "B"):
        for c in ("recall", "truncated"):
            if overall[m][c] >= 0.03:
                fails.append(f"V2 {m} {c} {overall[m][c]} ≥ 0.03")
    res = {}
    names = ("T1", "T2", "C1", "C2", "S", "S0")
    for name in names:
        for held in (None, True):
            us = [u for u, p in layer.items() if p["layer"] == name and u not in bad and (held is None or p["heldout_task"])]
            if not us or (held and len(us) == sum(1 for u, p in layer.items() if p["layer"] == name and u not in bad)):
                continue
            row = {"rows": len(us)}
            for cat in CATS:
                row[cat] = {"A": round(sum(frac(A[u][:8], cat) for u in us) / len(us), 4), "B": round(sum(frac(B[u], cat) for u in us) / len(us), 4),
                            "B-A": boot([(task[u], frac(B[u], cat) - frac(A[u][:8], cat)) for u in us], rng),
                            "noise(A2-A1)": boot([(task[u], frac(A[u][8:], cat) - frac(A[u][:8], cat)) for u in us], rng, level=0.992)}
            row["tasks"] = len({task[u] for u in us})
            pa = [(task[u], frac_parsed(A[u][:8]), frac_parsed(B[u])) for u in us]
            pa = [(t, x, y) for t, x, y in pa if x is not None and y is not None]
            ma, mb = sum(x for _, x, _ in pa) / len(pa), sum(y for _, _, y in pa) / len(pa)
            row["tc_of_parsed"] = {"A": round(ma, 4), "B": round(mb, 4), "B-A": boot([(t, y - x) for t, x, y in pa], rng),
                                   "relative_to_headroom": round((mb - ma) / (1 - ma), 4) if ma < 1 else None}
            res[name + ("" if held is None else "/heldout")] = row
    for name in names:
        if name in res:
            lo, hi = res[name]["tc"]["noise(A2-A1)"][1:]
            if not (lo <= 0 <= hi):
                fails.append(f"V1 {name} 잡음 바닥 99.2% 구간 [{lo}, {hi}] 이 0 을 포함하지 않음")
    if "S" in res and "C2" in res:
        s_rep, c2_rep = res["S"]["repeat_like"]["A"], res["C2"]["repeat_like"]["A"]
        if not (s_rep >= 0.20 and s_rep >= 3 * c2_rep):
            fails.append(f"V3 S 재현 대조: A repeat_like S {s_rep} · C2 {c2_rep}")
        if not (res["S"]["repeat_like"]["B-A"][2] < 0):
            fails.append(f"V4 양성 대조: S repeat_like 차 구간 상한 {res['S']['repeat_like']['B-A'][2]} ≥ 0")
    else:
        fails.append("S · C2 층 없음")
    if "S" in res and "S0" in res:
        # 대응형 클러스터 부트스트랩: 과제를 재표본하고 그 과제의 S 행과 S0 행을 함께 넣는다 (두 층은 시도 · 과제를 공유한다)
        ds, d0 = collections.defaultdict(list), collections.defaultdict(list)
        for u, p in layer.items():
            if u in bad or p["layer"] not in ("S", "S0"):
                continue
            (ds if p["layer"] == "S" else d0)[task[u]].append(frac(B[u], "tc") - frac(A[u][:8], "tc"))
        tasks = sorted(set(ds) | set(d0))
        mean = lambda d, ts: (sum(sum(d[t]) for t in ts) / max(1, sum(len(d[t]) for t in ts)))
        bs = []
        for _ in range(4000):
            ts = rng.choices(tasks, k=len(tasks))
            if sum(len(ds[t]) for t in ts) and sum(len(d0[t]) for t in ts):
                bs.append(mean(ds, ts) - mean(d0, ts))
        bs.sort()
        res["S_minus_S0_tc_diff"] = {"mean": round(mean(ds, tasks) - mean(d0, tasks), 4),
                                     "ci95": [round(bs[int(0.025 * len(bs))], 4), round(bs[int(0.975 * len(bs))], 4)], "tasks": len(tasks)}
    summ = {"rows": len(layer), "rows_with_wrong_counts": len(bad), "overall": overall, "layers": res,
            "validity": "PASS" if not fails else "FAIL", "fails": fails}
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    if a.out:
        json.dump(summ, open(a.out, "w"), ensure_ascii=False, indent=1)
    return 0 if not fails else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--tb-prefixes")
    b.add_argument("--seed", type=int, default=20261118)
    b.set_defaults(fn=build)
    z = sub.add_parser("analyze")
    z.add_argument("--a", required=True)
    z.add_argument("--b", required=True)
    z.add_argument("--input", required=True)
    z.add_argument("--out")
    z.set_defaults(fn=analyze)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
