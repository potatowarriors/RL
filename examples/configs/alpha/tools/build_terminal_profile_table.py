"""terminal 피벗 프로파일 표 — 재채점 결과와 원 롤아웃을 행별 요약으로 합친다 (CPU, RL_PLAN.md 결정 65~68).

입력: `$NRL_ROOT/p0_measure/term_pivot/{full,rest}/rollouts.jsonl` (MOPD1 학생 × 8) · `rejudge/rejudge.jsonl` (새 채점기 보상) · 원본 데이터 (과제 이름).
출력: `rejudge/term_profile_judge_v2.json` — uuid → {
  task, klen, gk (정답 keystrokes, \\x00 구분),
  g_tc 완료 선언 · g_wait 빈 대기 · ctrl 제어 키 (`C-<글자>` 한 개짜리 keystroke — tmux 키 이름) ·
  g_rep6 정답이 최근 6턴 중 하나와 유사도 ≥ 0.7 · g_near 직전 턴과 유사도 0.5~0.7 · g_rerun 정답의 명령이 전부 앞 턴에 글자 그대로 있음 ·
  g_sleep 정답이 sleep · echo 뿐 · lenient1 통과가 1회뿐이고 그것이 정답과 유사도 < 0.3 인 judge 통과,
  o: [[보상, 사유, 유사도, 출력 종류, 넓은 반복류 여부] × 8] }
출력 종류: recall (두 번 이상 호출) · invalid · tc · wait (빈 keystrokes) · repeat (직전 턴과 유사도 ≥ 0.7) · other.
넓은 반복류 (2026-11-16 블렌드 검토의 정의): 빈 대기 · 빈 commands · 최근 6턴 중 하나와 유사도 ≥ 0.7 · 앞 턴 명령의 글자 그대로 재실행 · sleep/echo 뿐.

  python examples/configs/alpha/tools/build_terminal_profile_table.py [--out <path>]
"""

import argparse
import json
import re
from difflib import SequenceMatcher

T = "/home/work/vidsearch/tools/nemo_rl/p0_measure/term_pivot/"
SOURCE = "/home/work/Datasets/LL_datasets/posttraining/RL/Nemotron-RL-Agentic-Terminal-Pivot-v1/atcb_terminal_pivot_release_final_v2.jsonl"
CTRL = re.compile(r"C-.|M-.")
SLEEP = re.compile(r"(sleep\s+[\d.]+|echo(\s.*)?)")


def ks(d) -> list[str]:
    cm = d.get("commands", []) if isinstance(d, dict) else []
    cm = cm if isinstance(cm, list) else []
    return [str(c.get("keystrokes", "")) for c in cm if isinstance(c, dict) and "keystrokes" in c]


def sim(a, b) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, "".join(a), "".join(b)).ratio()


def is_wait(k) -> bool:
    return bool(k) and all(not x.strip() for x in k)


def is_sleep(k) -> bool:
    body = [x.strip() for x in k if x.strip()]
    return bool(body) and all(SLEEP.fullmatch(x) for x in body)


def repeat_like(k, hist) -> bool:
    """넓은 반복류: 빈 commands · 빈 대기 · 최근 6턴과 유사도 ≥ 0.7 · 앞 턴 명령 재실행 · sleep/echo 뿐."""
    if not k or is_wait(k) or is_sleep(k):
        return True
    if any(sim(k, h) >= 0.7 for h in hist[-6:]):
        return True
    earlier = {x.strip() for h in hist for x in h if x.strip()}
    body = [x.strip() for x in k if x.strip()]
    return bool(body) and all(x in earlier for x in body)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=T + "rejudge/term_profile_judge_v2.json")
    a = ap.parse_args()
    rj = {}
    with open(T + "rejudge/rejudge.jsonl") as f:
        for line in f:
            x = json.loads(line)
            rj[(x["uuid"], x["ri"])] = (x["reward"], x["reason"], x["sim"])
    rows, hists = {}, {}
    for name in ("full/rollouts.jsonl", "rest/rollouts.jsonl"):
        with open(T + name) as f:
            for line in f:
                x = json.loads(line)
                u = x["uuid"]
                rw, rs, sm = rj[(u, x["_ng_rollout_index"])]
                d = rows.get(u)
                if d is None:
                    g = json.loads(x["expected_answer"])
                    gk = ks(g)
                    hist = []
                    for m in x["responses_create_params"]["input"]:
                        if m.get("role") == "assistant" and isinstance(m.get("content"), str):
                            try:
                                hist.append(ks(json.loads(m["content"])))
                            except json.JSONDecodeError:
                                hist.append([])
                    hists[u] = hist
                    earlier = {k.strip() for h in hist for k in h if k.strip()}
                    body = [k.strip() for k in gk if k.strip()]
                    sp = sim(gk, hist[-1]) if hist else 0.0
                    d = rows[u] = {
                        "gk": "\x00".join(gk), "klen": len("".join(gk)), "g_tc": bool(g.get("task_complete")), "g_wait": is_wait(gk),
                        "ctrl": any(CTRL.fullmatch(k.strip()) for k in gk),
                        "g_rep6": bool(hist) and any(sim(gk, h) >= 0.7 for h in hist[-6:]),
                        "g_near": bool(hist) and 0.5 <= sp < 0.7,
                        "g_rerun": bool(body) and all(k in earlier for k in body),
                        "g_sleep": is_sleep(gk), "o": [],
                    }
                hist = hists[u]
                po = x.get("parsed_output")
                if rs == "recall":
                    cat, rep = "recall", False
                elif not isinstance(po, dict):
                    cat, rep = "invalid", False
                else:
                    k = ks(po)
                    rep = repeat_like(k, hist)
                    if po.get("task_complete") or po.get("is_task_complete"):
                        cat, rep = "tc", False
                    elif is_wait(k):
                        cat = "wait"
                    elif hist and sim(k, hist[-1]) >= 0.7:
                        cat = "repeat"
                    else:
                        cat = "other"
                d["o"].append([rw, rs, sm, cat, rep])
    with open(SOURCE) as f:
        for line in f:
            r = json.loads(line)
            rows[r["uuid"]]["task"] = r["task_name"]
    for d in rows.values():
        ps = [o for o in d["o"] if o[0] == 1.0]
        d["lenient1"] = len(ps) == 1 and ps[0][2] is not None and ps[0][2] < 0.3
    with open(a.out, "w") as f:
        json.dump(rows, f)
    n = len(rows)
    flags = ("g_tc", "g_wait", "ctrl", "g_rep6", "g_near", "g_rerun", "g_sleep", "lenient1")
    print(json.dumps({"rows": n, **{k: sum(bool(d[k]) for d in rows.values()) for k in flags}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
