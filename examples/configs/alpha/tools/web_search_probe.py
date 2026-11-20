"""검색 객관식 환경 (alpha_web_search_mcqa) 소규모 측정 — 입력 생성 · 결과 분석 · 자가 시험 (CPU, 표준 라이브러리만).

재는 것: 롤아웃당 검색 횟수 (비용) · 통과율 · 검색이 보상에 기여하는가. 두 갈래를 같은 문항으로 돌린다 —
  tools   : 데이터 행 그대로 (도구 `search` · `browse`)
  notools : 도구 정의를 빼고 지시문에서 도구 문장을 뺀 것 (모델의 지식 + 찍기 바닥)

  build     <mcqa_search.jsonl> <out_dir> [--per-stratum 16]      난이도 표지별로 같은 수를 뽑아 sample · input_tools · input_notools 를 쓴다
  analyze   <out_dir> --tools <rollouts.jsonl> --notools <rollouts.jsonl> [--repeats 4] [--json out.json]
  smoke-check <rollouts.jsonl> <tools|notools>                     본 실행 전 몇 행으로 경로를 본다 (크레딧을 다 쓰기 전에 멈추려고)
  self-test                                                        변조한 합성 결과가 겨냥한 유효성 검사에서 걸리는지 본다

analyze 는 먼저 유효성 (V1~V6) 을 본다. 하나라도 FAIL 이면 수치를 결론으로 쓰지 않는다 (종료 코드 1). 값이 없는 키는 0 으로 읽지 않고 FAIL 이다.
해석 규칙 (실행 전에 고정): "검색이 기여한다" = 문항별 짝 차 (tools − notools) 의 부트스트랩 95% 구간 하한 > 0. 문항 48 개로는 약 0.1 미만의
차이를 가르지 못한다. 구간이 0 을 품으면 "구분 안 됨" 이지 "기여 없음" 이 아니다.
종료 코드: 0 = 유효, 1 = 유효성 FAIL, 2 = 자가 시험 실패.
"""

import argparse
import collections
import copy
import hashlib
import json
import random
import re
import statistics
import sys
from pathlib import Path


AGENT = {"type": "responses_api_agents", "name": "alpha_web_search_mcqa_simple_agent"}
TOOL_INSTRUCTIONS_PREFIX = "Solve the following problem. You are encouraged to use the tools provided to you"
NOTOOLS_INSTRUCTIONS = (
    "Solve the following problem. Return your final answer and put the answer option "
    "(and only the option [A, B, C, etc]) inside \\boxed{{}}."
)
OPTION_LINE = re.compile(r"^\s*\(?([A-Z])[:.)]\s", re.MULTILINE)
COUNT_KEYS = ("num_search_calls", "num_search_errors", "num_search_retries", "num_search_credits", "num_browse_calls", "num_browse_errors")
REQUIRED_KEYS = ("reward", "response", "responses_create_params", "expected_answer", "answer_found", "parsed_option",
                 "_ng_task_index", "_ng_rollout_index", *COUNT_KEYS)
EPOCH_ROLLOUTS = 2929 * 16
USD_PER_CREDIT = 0.008  # Tavily 종량제 (결제 전 확인 필요)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def n_options(row):
    return len(set(OPTION_LINE.findall(row["responses_create_params"]["input"])))


# ---------------------------------------------------------------- build


def build(source, out_dir, per_stratum):
    rows = read_jsonl(source)
    strata = collections.defaultdict(list)
    for row in rows:
        params = row["responses_create_params"]
        assert isinstance(params["input"], str) and params["instructions"].startswith(TOOL_INSTRUCTIONS_PREFIX), "데이터 형식이 바뀌었다"
        assert [t["name"] for t in params["tools"]] == ["search", "browse"], "도구 구성이 바뀌었다"
        strata[row["task_difficulty_qwen3_32b_avg_8"]].append(row)
    sample = []
    for level in sorted(strata):  # 층 안에서는 질문 해시 순 — 파일 순서 · 실행 시점과 무관하게 같은 표본
        ordered = sorted(strata[level], key=lambda r: hashlib.sha256(r["responses_create_params"]["input"].encode()).hexdigest())
        assert len(ordered) >= per_stratum
        sample += ordered[:per_stratum]
    notools = []
    for row in sample:
        r = copy.deepcopy(row)
        params = r["responses_create_params"]
        del params["tools"]
        params.pop("parallel_tool_calls", None)
        params["instructions"] = NOTOOLS_INSTRUCTIONS
        notools.append(r | {"agent_ref": AGENT})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("sample.jsonl", "input_tools.jsonl", "input_notools.jsonl"):
        assert not (out / name).exists(), f"{out / name} 이 이미 있다"
    write_jsonl(out / "sample.jsonl", sample)
    write_jsonl(out / "input_tools.jsonl", [r | {"agent_ref": AGENT} for r in sample])
    write_jsonl(out / "input_notools.jsonl", notools)
    floor = statistics.mean(1 / n_options(r) for r in sample)
    print(f"표본 {len(sample)} 문항 (난이도 표지별 {per_stratum}: {sorted(strata)}) · 선택지 수 {dict(collections.Counter(n_options(r) for r in sample))}"
          f" · 무작위 선택 기대 보상 {floor:.3f} → {out}")


# ---------------------------------------------------------------- 롤아웃 한 줄 읽기


def output_text(item):
    content = item.get("content")
    if isinstance(content, str):
        return content
    return "".join(p.get("text") or "" for p in content or [] if p.get("type") == "output_text")


def rescore(row):
    """플러그인 verify 와 같은 규칙을 결과 행만으로 다시 계산한다: 마지막 항목이 assistant 메시지 → 마지막 </think> 뒤 → 첫 \\boxed{…} 완전 일치."""
    output = row["response"]["output"]
    last = output[-1] if output else {}
    if last.get("type") != "message" or last.get("role") != "assistant":
        return 0.0, None
    m = re.search(r"\\boxed\{(.*?)\}", output_text(last).split("</think>")[-1])
    parsed = m.group(1) if m else None
    return (1.0 if parsed is not None and parsed == row["expected_answer"] else 0.0), parsed


def normalize_option(parsed):
    """형식만 다른 답을 가려내기 위한 느슨한 읽기 (보상에는 쓰지 않는다)."""
    if parsed is None:
        return None
    s = re.sub(r"\\(?:text|textbf|mathrm|mathbf)\s*\{([^{}]*)\}?", r"\1", parsed)
    s = re.sub(r"[\s$().\[\]{}:*]", "", s).upper()
    return s if re.fullmatch(r"[A-J]", s) else None


def tool_events(row):
    """(이름, 결과 종류) 목록. 결과 종류: ok · backend_error · model_error (질의 · 인자 · URL 탓) · fetch_failed · unknown_tool."""
    names = {o["call_id"]: o["name"] for o in row["response"]["output"] if o.get("type") == "function_call"}
    events = []
    for o in row["response"]["output"]:
        if o.get("type") != "function_call_output":
            continue
        name = names.get(o["call_id"], "?")
        try:
            body = json.loads(o["output"])
        except (json.JSONDecodeError, TypeError):
            body = None
        text = body.get("search_results") if name == "search" and isinstance(body, dict) else None
        page = body.get("page_content") if name == "browse" and isinstance(body, dict) else None
        if name not in ("search", "browse"):
            kind = "unknown_tool"
        elif text is None and page is None:
            kind = "model_error"  # 인자 오류 (422 · JSON 파싱 실패) — 서버 핸들러에 닿지 않았다
        elif text is not None:
            kind = "ok" if text.startswith("Search Results") else "backend_error" if text.startswith("Error: search is unavailable") else "model_error"
        else:
            kind = "model_error" if page.startswith("Error: ") and "Unexpected" not in page else \
                   "fetch_failed" if page in ("No HTML found", "No text found") or page.startswith("Error: Unexpected") else "ok"
        events.append((name, kind, text is not None or page is not None))
    return events


# ---------------------------------------------------------------- 유효성


def validity(sample, arms, repeats):
    """(검사 이름, 통과, 설명) 목록. arms = {"tools": rows, "notools": rows}."""
    checks = []

    def add(name, problems):
        checks.append((name, not problems, "; ".join(problems[:4]) + (f" … 외 {len(problems) - 4}" if len(problems) > 4 else "")))

    v1 = []
    for arm, rows in arms.items():
        if len(rows) != len(sample) * repeats:
            v1.append(f"{arm}: 행 {len(rows)} ≠ {len(sample)} × {repeats}")
        missing = collections.Counter(k for r in rows for k in REQUIRED_KEYS if k not in r or (k in COUNT_KEYS and r[k] is None))
        v1 += [f"{arm}: 키 {k} 없는 행 {n}" for k, n in missing.items()]
        if missing:
            continue
        ids = collections.Counter((r["_ng_task_index"], r["_ng_rollout_index"]) for r in rows)
        expected = {(t, i) for t in range(len(sample)) for i in range(repeats)}
        if set(ids) != expected or max(ids.values(), default=0) > 1:
            v1.append(f"{arm}: (문항, 반복) 조합이 {len(expected)} 개와 다름 (고유 {len(ids)} · 중복 {sum(n > 1 for n in ids.values())})")
        if any(r["reward"] not in (0.0, 1.0) for r in rows):
            v1.append(f"{arm}: 보상이 0 · 1 이 아닌 행")
    add("V1 완전성 (행 수 · 키 · 문항 × 반복 · 보상 값)", v1)
    if v1:
        return checks  # 아래 검사는 키가 있어야 한다

    v2 = []
    for arm, rows in arms.items():
        want_tools = arm == "tools"
        bad = sum(bool(r["responses_create_params"].get("tools")) != want_tools for r in rows)
        if bad:
            v2.append(f"{arm}: 도구 정의가 갈래와 다른 행 {bad}")
    called = sum(any(o.get("type") == "function_call" for o in r["response"]["output"]) or r["num_search_calls"] or r["num_browse_calls"]
                 for r in arms["notools"])
    if called:
        v2.append(f"notools: 도구 호출이 있는 행 {called}")
    add("V2 갈래 분리 (tools 만 도구를 가진다 · notools 는 호출 0)", v2)

    v3 = []
    for r in arms["tools"]:
        events = tool_events(r)
        handled = collections.Counter(name for name, _, reached in events if reached)
        if handled["search"] != r["num_search_calls"] or handled["browse"] != r["num_browse_calls"]:
            v3.append(f"문항 {r['_ng_task_index']}/{r['_ng_rollout_index']}: 기록 search {handled['search']} · browse {handled['browse']} ≠ "
                      f"서버 {r['num_search_calls']} · {r['num_browse_calls']}")
    add("V3 세션 통계 = 대화 기록 (서버가 센 호출 수가 결과 행의 도구 결과 수와 같다)", v3)

    errors = [r for r in arms["tools"] if r["num_search_errors"] or r.get("instance_config") or
              any(kind == "backend_error" for _, kind, _ in tool_events(r))]
    add("V4 검색 백엔드 장애 0 (한도 · 인증 · 재시도 뒤 실패)", [f"장애가 있던 롤아웃 {len(errors)} / {len(arms['tools'])}"] if errors else [])

    v5 = [f"{arm} 문항 {r['_ng_task_index']}/{r['_ng_rollout_index']}: 기록 {r['reward']} ≠ 재계산 {rescore(r)[0]}"
          for arm, rows in arms.items() for r in rows if rescore(r)[0] != r["reward"]]
    add("V5 보상 재계산 일치 (결과 행만으로 같은 규칙을 다시 적용)", v5)

    v6 = []
    for arm, rows in arms.items():
        bad = sum(r["expected_answer"] != sample[r["_ng_task_index"]]["expected_answer"] or
                  r["responses_create_params"]["input"] != sample[r["_ng_task_index"]]["responses_create_params"]["input"] for r in rows)
        if bad:
            v6.append(f"{arm}: 표본의 같은 번호 문항과 질문 · 정답이 다른 행 {bad}")
    add("V6 문항 정렬 (결과의 문항 번호가 표본과 같은 질문 · 정답)", v6)
    return checks


def smoke_check(path, arm):
    """몇 행짜리 결과로 경로만 본다: 키 · 갈래 분리 · 세션 통계 = 기록 · 백엔드 장애 0 · 보상 재계산. 하나라도 어긋나면 1."""
    rows = read_jsonl(path)
    problems = []
    if not rows:
        problems.append("행이 없다")
    for r in rows:
        missing = [k for k in REQUIRED_KEYS if k not in r or (k in COUNT_KEYS and r[k] is None)]
        if missing:
            problems.append(f"키 없음 {missing}")
            continue
        events = tool_events(r)
        handled = collections.Counter(name for name, _, reached in events if reached)
        has_calls = any(o.get("type") == "function_call" for o in r["response"]["output"])
        if arm == "notools" and (has_calls or r["responses_create_params"].get("tools") or r["num_search_calls"] or r["num_browse_calls"]):
            problems.append("notools 인데 도구 정의 · 호출이 있다")
        if arm == "tools" and not r["responses_create_params"].get("tools"):
            problems.append("tools 인데 도구 정의가 없다")
        if handled["search"] != r["num_search_calls"] or handled["browse"] != r["num_browse_calls"]:
            problems.append(f"세션 통계 ≠ 기록 (기록 {dict(handled)} · 서버 search {r['num_search_calls']} browse {r['num_browse_calls']})")
        if r["num_search_errors"] or r.get("instance_config") or any(kind == "backend_error" for _, kind, _ in events):
            problems.append("검색 백엔드 장애")
        if rescore(r)[0] != r["reward"]:
            problems.append(f"보상 재계산 불일치 (기록 {r['reward']} · 재계산 {rescore(r)[0]})")
    kinds = collections.Counter(f"{name}:{kind}" for r in rows if "response" in r for name, kind, _ in tool_events(r))
    ended = sum((r["response"]["output"] or [{}])[-1].get("type") == "message" for r in rows if "response" in r)
    print(f"smoke-check {arm}: 행 {len(rows)} · 보상 합 {sum(r.get('reward', 0) for r in rows)} · 메시지로 끝난 행 {ended} · 도구 결과 {dict(kinds)} · "
          f"크레딧 {sum(r.get('num_search_credits') or 0 for r in rows)}")
    for p_ in sorted(set(problems)):
        print(f"  결함: {p_} ({problems.count(p_)} 행)")
    print("smoke-check:", "PASS" if not problems else "FAIL")
    return 0 if not problems else 1


# ---------------------------------------------------------------- 수치


def by_task(rows, value, n_tasks):
    out = [[] for _ in range(n_tasks)]
    for r in rows:
        out[r["_ng_task_index"]].append(value(r))
    return out


def mean_se(per_task):
    means = [statistics.mean(v) for v in per_task]
    return statistics.mean(means), statistics.stdev(means) / len(means) ** 0.5


def paired_ci(a, b, seed=0, draws=10000):
    """문항 단위 짝 차 (a − b) 의 평균과 부트스트랩 95% 구간. a · b 는 문항별 값 목록."""
    diffs = [statistics.mean(x) - statistics.mean(y) for x, y in zip(a, b)]
    rng = random.Random(seed)
    boots = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(draws))
    return statistics.mean(diffs), boots[int(0.025 * draws)], boots[int(0.975 * draws) - 1]


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def metrics(sample, arms):
    n = len(sample)
    reward = {arm: by_task(rows, lambda r: r["reward"], n) for arm, rows in arms.items()}
    out = {"n_tasks": n, "chance_floor": statistics.mean(1 / n_options(r) for r in sample)}
    for arm in arms:
        m, se = mean_se(reward[arm])
        out[f"reward_{arm}"] = {"mean": m, "se": se}
        half = [[v[: len(v) // 2] for v in reward[arm]], [v[len(v) // 2:] for v in reward[arm]]]
        out[f"noise_aa_{arm}"] = dict(zip(("diff", "lo", "hi"), paired_ci(*half)))  # 같은 갈래의 반반 — 잡음 크기
    out["paired_tools_minus_notools"] = dict(zip(("diff", "lo", "hi"), paired_ci(reward["tools"], reward["notools"])))
    out["by_difficulty"] = {}
    for level in sorted({r["task_difficulty_qwen3_32b_avg_8"] for r in sample}):
        idx = [i for i, r in enumerate(sample) if r["task_difficulty_qwen3_32b_avg_8"] == level]
        out["by_difficulty"][str(level)] = {arm: statistics.mean(statistics.mean(reward[arm][i]) for i in idx) for arm in arms}

    tools = arms["tools"]
    events = [tool_events(r) for r in tools]
    searches = [sum(name == "search" for name, _, _ in ev) for ev in events]
    browses = [sum(name == "browse" for name, _, _ in ev) for ev in events]
    kinds = collections.Counter((name, kind) for ev in events for name, kind, _ in ev)
    credits = [r["num_search_credits"] for r in tools]
    used = [any(name == "search" and kind == "ok" for name, kind, _ in ev) for ev in events]
    out["tools_usage"] = {
        "searches_per_rollout": {"mean": statistics.mean(searches), "p50": pct(searches, 0.5), "p90": pct(searches, 0.9), "max": max(searches)},
        "browses_per_rollout": {"mean": statistics.mean(browses), "p50": pct(browses, 0.5), "p90": pct(browses, 0.9), "max": max(browses)},
        "credits_per_rollout": statistics.mean(credits),
        "credits_total": sum(credits),
        "retries_total": sum(r["num_search_retries"] for r in tools),
        "rollouts_with_successful_search": statistics.mean(used),
        "rollouts_without_any_tool_call": statistics.mean(not ev for ev in events),
        "event_kinds": {f"{name}:{kind}": c for (name, kind), c in sorted(kinds.items())},
        "reward_when_search_used": statistics.mean(r["reward"] for r, u in zip(tools, used) if u) if any(used) else None,
        "reward_when_search_not_used": statistics.mean(r["reward"] for r, u in zip(tools, used) if not u) if not all(used) else None,
    }
    out["epoch_cost_usd"] = {depth: statistics.mean(credits) * mult * EPOCH_ROLLOUTS * USD_PER_CREDIT for depth, mult in (("basic", 1), ("advanced", 2))}
    for arm, rows in arms.items():
        parsed = [rescore(r)[1] for r in rows]
        ended = [(r["response"]["output"] or [{}])[-1].get("type") == "message" for r in rows]
        usage = [r["response"].get("usage") or {} for r in rows]
        out[f"answers_{arm}"] = {
            "answer_found": statistics.mean(p is not None for p in parsed),
            "no_final_message": statistics.mean(not e for e in ended),  # 스텝 상한 · 생성 상한 · 닫지 않은 사고
            "incomplete": statistics.mean(bool(r["response"].get("incomplete_details")) for r in rows),
            "format_only_miss": statistics.mean(r["reward"] == 0 and normalize_option(p) == r["expected_answer"] for r, p in zip(rows, parsed)),
            "input_tokens_sum_mean": statistics.mean(u.get("input_tokens") or 0 for u in usage),
            "input_tokens_sum_max": max(u.get("input_tokens") or 0 for u in usage),
            "output_tokens_sum_mean": statistics.mean(u.get("output_tokens") or 0 for u in usage),
            "usage_missing": sum(not u for u in usage),
        }
    return out


def analyze(out_dir, tools_path, notools_path, repeats, json_path):
    sample = read_jsonl(Path(out_dir) / "sample.jsonl")
    arms = {"tools": read_jsonl(tools_path), "notools": read_jsonl(notools_path)}
    checks = validity(sample, arms, repeats)
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}{'' if ok else ' — ' + detail}")
    valid = all(ok for _, ok, _ in checks)
    result = {"valid": valid, "checks": [{"name": n, "pass": ok, "detail": d} for n, ok, d in checks]}
    if valid or all(ok for name, ok, _ in checks if not name.startswith("V4")):  # 장애만 있으면 수치는 보여 주되 결론으로 쓰지 않는다
        m = metrics(sample, arms)
        result["metrics"] = m
        p = m["paired_tools_minus_notools"]
        print(f"\n보상  tools {m['reward_tools']['mean']:.3f} ± {m['reward_tools']['se']:.3f} · notools {m['reward_notools']['mean']:.3f} ± "
              f"{m['reward_notools']['se']:.3f} · 무작위 {m['chance_floor']:.3f}  (문항 {m['n_tasks']} × 반복 {repeats}, ± 는 문항 단위 표준오차)")
        print(f"짝 차 tools − notools = {p['diff']:+.3f}  (95% {p['lo']:+.3f} ~ {p['hi']:+.3f})  → "
              f"{'검색이 기여한다' if p['lo'] > 0 else '검색이 깎는다' if p['hi'] < 0 else '구분 안 됨'}")
        print("잡음 (같은 갈래 반반의 짝 차): " + " · ".join(
            f"{arm} {m[f'noise_aa_{arm}']['diff']:+.3f} ({m[f'noise_aa_{arm}']['lo']:+.3f} ~ {m[f'noise_aa_{arm}']['hi']:+.3f})" for arm in arms))
        print("난이도 표지별: " + " · ".join(f"{k}: tools {v['tools']:.3f} / notools {v['notools']:.3f}" for k, v in m["by_difficulty"].items()))
        u = m["tools_usage"]
        print(f"검색/롤아웃 평균 {u['searches_per_rollout']['mean']:.2f} · p50 {u['searches_per_rollout']['p50']} · p90 {u['searches_per_rollout']['p90']} · "
              f"최대 {u['searches_per_rollout']['max']} | browse 평균 {u['browses_per_rollout']['mean']:.2f} · 최대 {u['browses_per_rollout']['max']}")
        print(f"크레딧/롤아웃 {u['credits_per_rollout']:.2f} (합 {u['credits_total']}) · 재시도 {u['retries_total']} · 검색 성공이 있는 롤아웃 "
              f"{u['rollouts_with_successful_search']:.1%} · 도구 호출 0 인 롤아웃 {u['rollouts_without_any_tool_call']:.1%}")
        print(f"도구 결과 종류: {u['event_kinds']}")
        print(f"검색을 쓴 롤아웃의 보상 {u['reward_when_search_used']} · 안 쓴 롤아웃 {u['reward_when_search_not_used']} (서술용 — 쓴 쪽이 어려운 문항일 수 있다)")
        print(f"1 에폭 ({EPOCH_ROLLOUTS:,} 롤아웃) 검색 비용 추정: basic ${m['epoch_cost_usd']['basic']:,.0f} · advanced ${m['epoch_cost_usd']['advanced']:,.0f}")
        for arm in arms:
            a = m[f"answers_{arm}"]
            print(f"{arm}: 답 찾음 {a['answer_found']:.1%} · 마지막이 메시지 아님 {a['no_final_message']:.1%} · incomplete {a['incomplete']:.1%} · "
                  f"형식만 틀린 0 점 {a['format_only_miss']:.1%} · 입력 토큰 합 평균 {a['input_tokens_sum_mean']:,.0f} (최대 {a['input_tokens_sum_max']:,}) · "
                  f"출력 토큰 합 평균 {a['output_tokens_sum_mean']:,.0f} · usage 없는 행 {a['usage_missing']}")
    print("\n유효성:", "PASS" if valid else "FAIL — 수치를 결론으로 쓰지 않는다")
    if json_path:
        Path(json_path).write_text(json.dumps(result, ensure_ascii=False, indent=1))
    return 0 if valid else 1


# ---------------------------------------------------------------- 자가 시험


def _synthetic(n_tasks=6, repeats=2):
    sample, arms = [], {"tools": [], "notools": []}
    for t in range(n_tasks):
        question = f"Question {t}?\nA: a\nB: b\nC: c\nD: d"
        sample.append({"expected_answer": "ABCD"[t % 4], "task_difficulty_qwen3_32b_avg_8": (0.25, 0.5)[t % 2],
                       "responses_create_params": {"input": question, "instructions": "x", "tools": [{"name": "search"}, {"name": "browse"}]}})
        for i in range(repeats):
            for arm in arms:
                correct = (t + i) % 2 == 0 if arm == "tools" else t % 3 == 0
                output = []
                if arm == "tools":
                    output += [{"type": "function_call", "call_id": "c1", "name": "search", "arguments": '{"query": "q"}'},
                               {"type": "function_call_output", "call_id": "c1", "output": json.dumps({"search_results": "Search Results\n[1] ..."})},
                               {"type": "function_call", "call_id": "c2", "name": "browse", "arguments": '{"url": "https://a.org"}'},
                               {"type": "function_call_output", "call_id": "c2", "output": json.dumps({"page_content": "page"})}]
                letter = sample[t]["expected_answer"] if correct else "ABCD"[(t + 1) % 4]
                output.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": rf"\boxed{{{letter}}}"}]})
                params = copy.deepcopy(sample[t]["responses_create_params"])
                if arm == "notools":
                    del params["tools"]
                arms[arm].append({"_ng_task_index": t, "_ng_rollout_index": i, "reward": 1.0 if correct else 0.0, "expected_answer": sample[t]["expected_answer"],
                                  "answer_found": True, "parsed_option": letter, "responses_create_params": params, "instance_config": None,
                                  "response": {"output": output, "usage": {"input_tokens": 100, "output_tokens": 10}},
                                  "num_search_calls": int(arm == "tools"), "num_search_errors": 0, "num_search_retries": 0,
                                  "num_search_credits": int(arm == "tools"), "num_browse_calls": int(arm == "tools"), "num_browse_errors": 0})
    return sample, arms


def self_test():
    def failed(mutate):
        sample, arms = _synthetic()
        mutate(sample, arms)
        return {name[:2] for name, ok, _ in validity(sample, arms, 2) if not ok}

    def call_in_notools(s, a):
        a["notools"][0]["response"]["output"].insert(0, {"type": "function_call", "call_id": "c", "name": "search", "arguments": "{}"})

    def unhandled_call(s, a):  # 서버에 닿지 않은 호출이 섞여도 닿은 호출 수는 맞아야 한다
        a["tools"][0]["response"]["output"][1]["output"] = json.dumps({"detail": [{"msg": "Field required"}]})

    mutations = {
        "행 하나 누락": (lambda s, a: a["tools"].pop(), "V1"),
        "(문항, 반복) 중복": (lambda s, a: a["tools"][1].update(_ng_rollout_index=0), "V1"),
        "통계 키 누락": (lambda s, a: a["tools"][0].pop("num_search_calls"), "V1"),
        "통계 값 None": (lambda s, a: a["tools"][0].update(num_search_credits=None), "V1"),
        "notools 에 도구 호출": (call_in_notools, "V2"),
        "notools 에 도구 정의": (lambda s, a: a["notools"][0]["responses_create_params"].update(tools=[{"name": "search"}]), "V2"),
        "서버 통계가 0 (세션 끊김)": (lambda s, a: a["tools"][0].update(num_search_calls=0), "V3"),
        "서버에 닿지 않은 호출을 닿은 것으로 셈": (unhandled_call, "V3"),
        "백엔드 장애 표시": (lambda s, a: a["tools"][0].update(num_search_errors=1, instance_config={"mask_sample": True}), "V4"),
        "보상 뒤집힘": (lambda s, a: a["notools"][0].update(reward=1.0 - a["notools"][0]["reward"]), "V5"),
        "문항 번호 어긋남": (lambda s, a: [r.update(_ng_task_index=(r["_ng_task_index"] + 1) % 6) for r in a["tools"]], "V6"),
    }
    problems = []
    if failed(lambda s, a: None):
        problems.append(f"변조 없는 합성 결과가 FAIL: {failed(lambda s, a: None)}")
    for name, (mutate, target) in mutations.items():
        got = failed(mutate)
        print(f"{'ok  ' if target in got else 'MISS'} {name} → {sorted(got) or '통과'} (겨냥 {target})")
        if target not in got:
            problems.append(name)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        def smoke_rc(arm, mutate=None):
            _, synthetic = _synthetic()
            rows = synthetic[arm][:4]
            if mutate:
                mutate(rows)
            write_jsonl(Path(tmp) / "r.jsonl", rows)
            return smoke_check(Path(tmp) / "r.jsonl", arm)

        smoke_cases = {
            "smoke: 정상 tools": (smoke_rc("tools"), 0),
            "smoke: 정상 notools": (smoke_rc("notools"), 0),
            "smoke: 세션 끊김": (smoke_rc("tools", lambda rows: rows[0].update(num_search_calls=0)), 1),
            "smoke: 백엔드 장애": (smoke_rc("tools", lambda rows: rows[0].update(num_search_errors=1)), 1),
            "smoke: 키 없음": (smoke_rc("tools", lambda rows: rows[0].pop("num_browse_calls")), 1),
            "smoke: notools 에 도구 호출": (smoke_rc("notools", lambda rows: call_in_notools(None, {"notools": rows})), 1),
            "smoke: 빈 파일": (smoke_rc("tools", lambda rows: rows.clear()), 1),
        }
    for name, (got, want) in smoke_cases.items():
        print(f"{'ok  ' if got == want else 'MISS'} {name} → rc {got} (기대 {want})")
        if got != want:
            problems.append(name)
    sample, arms = _synthetic()
    m = metrics(sample, arms)
    if abs(m["chance_floor"] - 0.25) > 1e-9 or m["tools_usage"]["credits_per_rollout"] != 1 or m["tools_usage"]["searches_per_rollout"]["mean"] != 1:
        problems.append("합성 결과의 수치가 기대와 다름")
    print("자가 시험:", "PASS" if not problems else f"FAIL {problems}")
    return 0 if not problems else 2


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("source")
    b.add_argument("out_dir")
    b.add_argument("--per-stratum", type=int, default=16)
    a = sub.add_parser("analyze")
    a.add_argument("out_dir")
    a.add_argument("--tools", required=True)
    a.add_argument("--notools", required=True)
    a.add_argument("--repeats", type=int, default=4)
    a.add_argument("--json", default=None)
    sub.add_parser("self-test")
    c = sub.add_parser("smoke-check")
    c.add_argument("rollouts")
    c.add_argument("arm", choices=("tools", "notools"))
    args = parser.parse_args()
    if args.cmd == "build":
        build(args.source, args.out_dir, args.per_stratum)
        return 0
    if args.cmd == "self-test":
        return self_test()
    if args.cmd == "smoke-check":
        return smoke_check(args.rollouts, args.arm)
    return analyze(args.out_dir, args.tools, args.notools, args.repeats, args.json)


if __name__ == "__main__":
    sys.exit(main())
