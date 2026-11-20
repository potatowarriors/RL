"""검색 객관식 환경 (alpha_web_search_mcqa) 소규모 측정 — 입력 생성 · 결과 분석 · 자가 시험 (CPU, 표준 라이브러리만).

재는 것: 롤아웃당 검색 횟수 (비용) · 통과율 · 검색이 준 정보가 보상에 기여하는가. 세 갈래를 같은 문항으로 돌린다 —
  tools   : 데이터 행 그대로 (도구 `search` · `browse`, 실제 검색)
  placebo : tools 와 같은 행 · 같은 도구 선언 · 같은 턴 상한. 서버만 `search_backend: none` (검색 결과 0 건 · browse 본문 없음)
  notools : 도구 정의를 빼고 지시문에서 도구 문장을 뺀 것 (모델의 지식 + 찍기 바닥, 서술용)
주 대조는 tools − placebo 다. 두 갈래는 프롬프트 · 턴 수 · 생성 상한이 같고 검색이 돌려주는 내용만 다르다.
tools − notools 는 지시문 · 턴 수 · 총 생성량이 함께 달라서 "검색의 기여" 로 읽지 않는다 (2026-11-20 적대적 검토).

  build       <mcqa_search.jsonl> <out_dir> [--per-stratum 16]   난이도 표지별로 같은 수를 뽑아 sample · input_tools · input_notools 를 쓴다
  analyze     <out_dir> --tools F --placebo F --notools F [--repeats 4] [--search-cost 1] [--json out.json]
  smoke-check <rollouts.jsonl> <tools|placebo|notools> [--search-cost 1]   본 실행 전 몇 행으로 경로를 본다 (크레딧을 다 쓰기 전에 멈추려고)
  self-test                                                      변조한 합성 결과가 겨냥한 검사에서 걸리는지 본다

analyze 는 먼저 유효성 (V1~V7) 을 본다. 하나라도 FAIL 이면 수치를 결론으로 쓰지 않는다 (종료 코드 1). 값이 없는 키는 0 으로 읽지 않고 FAIL 이다.
판정 규칙 (실행 전에 고정):
  "검색이 기여한다" = tools − placebo 의 문항별 짝 차 부트스트랩 95% 구간 하한 > 0 이고, 끝까지 답한 롤아웃만으로 다시 낸 구간 하한도 > 0.
  두 구간의 결론이 다르면 "판정 보류" (잘림 · 스텝 상한이 차이를 만들었을 수 있다). 검색 성공이 한 번도 없으면 "도구 미사용 — 판정 불가".
  구간이 0 을 품으면 "구분 안 됨" 이지 "기여 없음" 이 아니다. 48 문항 × 4 의 구간 반폭은 약 0.08~0.09 — 80% 로 잡는 차이는 약 0.15 다.
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
import tempfile
from pathlib import Path


AGENT = {"type": "responses_api_agents", "name": "alpha_web_search_mcqa_simple_agent"}
TOOL_INSTRUCTIONS_PREFIX = "Solve the following problem. You are encouraged to use the tools provided to you"
NOTOOLS_INSTRUCTIONS = (
    "Solve the following problem. Return your final answer and put the answer option "
    "(and only the option [A, B, C, etc]) inside \\boxed{{}}."
)
ARMS = ("tools", "placebo", "notools")
TOOL_ARMS = ("tools", "placebo")
OPTION_LINE = re.compile(r"^\s*\(?([A-Z])[:.)]\s", re.MULTILINE)
COUNT_KEYS = ("num_search_calls", "num_search_errors", "num_search_retries", "num_search_credits", "num_search_usage_missing",
              "num_browse_calls", "num_browse_errors")
REQUIRED_KEYS = ("reward", "response", "responses_create_params", "expected_answer", "answer_found", "parsed_option",
                 "_ng_task_index", "_ng_rollout_index", *COUNT_KEYS)
EPOCH_ROLLOUTS = 2929 * 16
USD_PER_CREDIT = 0.008  # Tavily 종량제 (결제 전 확인 필요)
LEAK_SHINGLE = 10  # 질문 본문의 연속 단어 수 — 이만큼이 도구 결과에 그대로 나오면 질문 자체가 실린 페이지로 본다


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
    write_jsonl(out / "input_tools.jsonl", [r | {"agent_ref": AGENT} for r in sample])  # placebo 갈래도 이 파일을 쓴다
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
    """플러그인 verify 와 같은 규칙을 결과 행만으로 다시 계산한다: 마지막 항목이 assistant 메시지 → 마지막 </think> 뒤 → 첫 \\boxed{…} 완전 일치.

    서버가 실제로 이 규칙으로 채점했는지만 본다 (규칙이 옳은지는 못 본다 — 형식만 틀린 답은 format_only_miss 로 따로 센다).
    """
    output = row["response"]["output"]
    last = output[-1] if output else {}
    if last.get("type") != "message" or last.get("role") != "assistant":
        return 0.0, None
    m = re.search(r"\\boxed\{(.*?)\}", output_text(last).split("</think>")[-1])
    parsed = m.group(1) if m else None
    return (1.0 if parsed is not None and parsed == row["expected_answer"] else 0.0), parsed


def normalize_option(parsed):
    """형식만 다른 답을 가려내기 위한 느슨한 읽기 (보상에는 쓰지 않는다). `{C` (지시문의 \\boxed{{}} 를 따라 쓴 답) 도 C 로 읽는다."""
    if parsed is None:
        return None
    s = re.sub(r"\\(?:text|textbf|mathrm|mathbf)\s*\{([^{}]*)\}?", r"\1", parsed)
    s = re.sub(r"[\s$().\[\]{}:*]", "", s).upper()
    return s if re.fullmatch(r"[A-J]", s) else None


def end_state(row):
    """message = 끝까지 답함 · incomplete = 생성 상한 등 · step_cap = 스텝 상한 (마지막이 도구 결과면 그 결과를 모델이 못 봤다) · other."""
    output = row["response"]["output"]
    last = output[-1] if output else {}
    if row["response"].get("incomplete_details"):
        return "incomplete"
    if last.get("type") == "message" and last.get("role") == "assistant":
        return "message"
    return "step_cap" if last.get("type") in ("function_call_output", "function_call", "reasoning") else "other"


def tool_events(row):
    """(도구 이름, 결과 종류, 서버 핸들러에 닿았는가, 결과 글) 목록.

    search: ok (결과 ≥ 1) · ok_empty (결과 0 건) · backend_error · rejected (검색 엔진이 질의를 거절) · model_error (빈 · 긴 질의)
    browse: ok · blocked (내부 주소 · 제외 도메인 · http 아님) · fetch_failed
    공통  : bad_args (인자 오류 — 핸들러에 닿지 않음) · unknown_tool
    """
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
            kind = "bad_args"
        elif text is not None:
            kind = ("ok" if "\n[1] " in text else "ok_empty") if text.startswith("Search Results") else \
                   "backend_error" if text.startswith("Error: search is unavailable") else \
                   "rejected" if text.startswith("Error: the search engine rejected") else "model_error"
        else:
            kind = "fetch_failed" if page in ("No HTML found", "No text found") or page.startswith("Error: Unexpected") else \
                   "blocked" if page.startswith("Error: ") else "ok"
        events.append((name, kind, text is not None or page is not None, text if text is not None else page or ""))
    return events


def words(text):
    return re.findall(r"\w+", text.lower())


def question_leaked(row):
    """질문 본문 (선택지 앞) 의 연속 LEAK_SHINGLE 단어가 도구 결과에 그대로 있는가."""
    stem = words(OPTION_LINE.split(row["responses_create_params"]["input"], maxsplit=1)[0])
    if len(stem) < LEAK_SHINGLE:
        return False
    seen = " " + " ".join(words(" ".join(text for _, kind, _, text in tool_events(row) if kind == "ok"))) + " "
    return any(" " + " ".join(stem[i:i + LEAK_SHINGLE]) + " " in seen for i in range(len(stem) - LEAK_SHINGLE + 1))


# ---------------------------------------------------------------- 유효성


def row_problems(r, arm, search_cost):
    """한 행의 경로 결함 (키가 다 있을 때). validity 와 smoke_check 가 함께 쓴다."""
    problems = []
    events = tool_events(r)
    handled = collections.Counter(name for name, _, reached, _ in events if reached)
    kinds = collections.Counter(f"{name}:{kind}" for name, kind, _, _ in events)
    has_tools = bool(r["responses_create_params"].get("tools"))
    if has_tools != (arm in TOOL_ARMS):
        problems.append("V2 도구 정의가 갈래와 다름")
    if arm == "notools" and (events or any(o.get("type") == "function_call" for o in r["response"]["output"]) or
                             r["num_search_calls"] or r["num_browse_calls"]):
        problems.append("V2 notools 인데 도구 호출이 있음")
    if arm == "placebo" and (kinds["search:ok"] or kinds["browse:ok"] or r["num_search_credits"]):
        problems.append("V2 placebo 인데 검색 결과 · 본문 · 크레딧이 있음")
    if handled["search"] != r["num_search_calls"] or handled["browse"] != r["num_browse_calls"]:
        problems.append("V3 세션 통계 ≠ 대화 기록")
    if r["num_search_errors"] or r.get("instance_config") or kinds["search:backend_error"]:
        problems.append("V4 검색 백엔드 장애")
    if rescore(r)[0] != r["reward"]:
        problems.append("V5 보상 재계산 불일치")
    if arm == "tools":
        if r["num_search_usage_missing"]:
            problems.append("V7 Tavily 응답에 usage 없음")
        if r["num_search_credits"] != (kinds["search:ok"] + kinds["search:ok_empty"]) * search_cost:
            problems.append("V7 크레딧 ≠ 성공한 검색 수 × 정가")
    return problems


def validity(sample, arms, repeats, search_cost=1):
    """(검사 이름, 통과, 설명) 목록. arms = {"tools": rows, "placebo": rows, "notools": rows}."""
    names = {"V1": "완전성 (행 수 · 키 · 문항 × 반복 · 보상 값)", "V2": "갈래 분리 (도구 정의 · notools 호출 0 · placebo 정보 0)",
             "V3": "세션 통계 = 대화 기록 (서버가 센 호출 수 = 결과 행의 도구 결과 수)", "V4": "검색 백엔드 장애 0 (한도 · 인증 · 재시도 뒤 실패)",
             "V5": "보상 재계산 일치", "V6": "문항 정렬 (결과의 문항 번호가 표본과 같은 질문 · 정답)",
             "V7": "검색 경로 생존 · 크레딧 장부 (성공한 검색이 있다 · 크레딧 = 성공 수 × 정가 · usage 누락 0)"}
    found = collections.defaultdict(list)
    for arm, rows in arms.items():
        if len(rows) != len(sample) * repeats:
            found["V1"].append(f"{arm}: 행 {len(rows)} ≠ {len(sample)} × {repeats}")
        missing = collections.Counter(k for r in rows for k in REQUIRED_KEYS if k not in r or (k in COUNT_KEYS and r[k] is None))
        found["V1"] += [f"{arm}: 키 {k} 없는 행 {n}" for k, n in missing.items()]
        if missing:
            continue
        ids = collections.Counter((r["_ng_task_index"], r["_ng_rollout_index"]) for r in rows)
        expected = {(t, i) for t in range(len(sample)) for i in range(repeats)}
        if set(ids) != expected or max(ids.values(), default=0) > 1:
            found["V1"].append(f"{arm}: (문항, 반복) 조합이 {len(expected)} 개와 다름 (고유 {len(ids)} · 중복 {sum(n > 1 for n in ids.values())})")
        if any(r["reward"] not in (0.0, 1.0) for r in rows):
            found["V1"].append(f"{arm}: 보상이 0 · 1 이 아닌 행")
    if not found["V1"]:  # 아래 검사는 키가 있어야 한다
        for arm, rows in arms.items():
            per_check = collections.Counter()
            for r in rows:
                for p in row_problems(r, arm, search_cost):
                    per_check[tuple(p.split(" ", 1))] += 1
            for (check, text), n in sorted(per_check.items()):
                found[check].append(f"{arm}: {text} {n} 행")
            bad = sum(r["expected_answer"] != sample[r["_ng_task_index"]]["expected_answer"] or
                      r["responses_create_params"]["input"] != sample[r["_ng_task_index"]]["responses_create_params"]["input"] for r in rows)
            if bad:
                found["V6"].append(f"{arm}: 표본의 같은 번호 문항과 질문 · 정답이 다른 행 {bad}")
        kinds = collections.Counter(f"{name}:{kind}" for r in arms["tools"] for name, kind, _, _ in tool_events(r))
        calls = sum(n for k, n in kinds.items() if k.startswith("search:"))
        if calls and not kinds["search:ok"]:
            found["V7"].append(f"tools: 검색 호출 {calls} 회 중 결과가 있는 검색 0 회 (거절 {kinds['search:rejected']} · 빈 결과 {kinds['search:ok_empty']})")
    return [(f"{v} {names[v]}", not found[v], "; ".join(found[v][:4]) + (f" … 외 {len(found[v]) - 4}" if len(found[v]) > 4 else ""))
            for v in names if v == "V1" or not found["V1"]]


def smoke_check(path, arm, search_cost):
    """몇 행짜리 결과로 경로만 본다. tools 갈래는 검색을 불렀는데 결과가 있는 검색이 하나도 없으면 FAIL (검색 경로가 죽은 채 크레딧을 쓰지 않게)."""
    rows = read_jsonl(path)
    problems = [] if rows else ["행이 없다"]
    for r in rows:
        missing = [k for k in REQUIRED_KEYS if k not in r or (k in COUNT_KEYS and r[k] is None)]
        problems += [f"V1 키 없음 {missing}"] if missing else row_problems(r, arm, search_cost)
    complete = [r for r in rows if not any(k not in r for k in REQUIRED_KEYS)]
    kinds = collections.Counter(f"{name}:{kind}" for r in complete for name, kind, _, _ in tool_events(r))
    ends = collections.Counter(end_state(r) for r in complete)
    calls = sum(n for k, n in kinds.items() if k.startswith("search:"))
    if arm == "tools" and calls and not kinds["search:ok"]:
        problems.append("V7 검색을 불렀는데 결과가 있는 검색이 없다")
    if complete and not ends["message"]:
        problems.append("끝까지 답한 행이 하나도 없다 (생성 상한 · 스텝 상한 · 사고를 닫지 않음)")
    print(f"smoke-check {arm}: 행 {len(rows)} · 보상 합 {sum(r.get('reward', 0) for r in rows)} · 끝 상태 {dict(ends)} · 도구 결과 {dict(kinds)} · "
          f"크레딧 {sum(r.get('num_search_credits') or 0 for r in rows)}")
    for p_ in sorted(set(problems)):
        print(f"  결함: {p_} ({problems.count(p_)} 행)")
    print("smoke-check:", "PASS" if not problems else "FAIL")
    return 0 if not problems else 1


# ---------------------------------------------------------------- 수치


def by_task(rows, value, n_tasks, keep=lambda r: True):
    """문항별 값 목록 (반복 번호 순 — 결과 파일은 끝난 순서로 쓰여 있다)."""
    out = [[] for _ in range(n_tasks)]
    for r in sorted(rows, key=lambda r: r["_ng_rollout_index"]):
        if keep(r):
            out[r["_ng_task_index"]].append(value(r))
    return out


def mean_se(per_task):
    means = [statistics.mean(v) for v in per_task if v]
    return statistics.mean(means), (statistics.stdev(means) / len(means) ** 0.5 if len(means) > 1 else float("nan"))


def paired_ci(a, b, seed=0, draws=10000):
    """문항 단위 짝 차 (a − b) 의 평균과 부트스트랩 95% 구간. 한쪽에 값이 없는 문항은 뺀다 (n 에 남긴다)."""
    diffs = [statistics.mean(x) - statistics.mean(y) for x, y in zip(a, b) if x and y]
    if len(diffs) < 2:
        return {"diff": None, "lo": None, "hi": None, "n": len(diffs)}
    rng = random.Random(seed)
    boots = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(draws))
    return {"diff": statistics.mean(diffs), "lo": boots[int(0.025 * draws)], "hi": boots[int(0.975 * draws) - 1], "n": len(diffs)}


def sign(ci):
    return None if ci["lo"] is None else 1 if ci["lo"] > 0 else -1 if ci["hi"] < 0 else 0


def verdict(primary, completed_only, search_success_share):
    if not search_success_share:
        return "도구 미사용 — 판정 불가"
    a, b = sign(primary), sign(completed_only)
    if a is None or b is None:
        return "판정 보류 (끝까지 답한 롤아웃이 부족해 다시 낼 수 없다)"
    if a != b:
        return "판정 보류 (잘림 · 스텝 상한 롤아웃을 빼면 결론이 달라진다)"
    return {1: "검색이 기여한다", -1: "검색이 깎는다", 0: "구분 안 됨"}[a]


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def metrics(sample, arms, repeats):
    n = len(sample)
    reward = {arm: by_task(rows, lambda r: r["reward"], n) for arm, rows in arms.items()}
    done = {arm: by_task(rows, lambda r: r["reward"], n, keep=lambda r: end_state(r) == "message") for arm, rows in arms.items()}
    out = {"n_tasks": n, "chance_floor": statistics.mean(1 / n_options(r) for r in sample)}
    for arm in arms:
        m, se = mean_se(reward[arm])
        out[f"reward_{arm}"] = {"mean": m, "se": se}
        half = repeats // 2  # 반복 번호 앞 절반 대 뒤 절반 — 같은 갈래 안의 잡음 크기
        out[f"noise_aa_{arm}"] = paired_ci([v[:half] for v in reward[arm]], [v[half:] for v in reward[arm]])
    out["primary_tools_minus_placebo"] = paired_ci(reward["tools"], reward["placebo"])
    out["primary_completed_only"] = paired_ci(done["tools"], done["placebo"])
    out["tools_minus_notools"] = paired_ci(reward["tools"], reward["notools"])
    out["placebo_minus_notools"] = paired_ci(reward["placebo"], reward["notools"])
    leaked = {(r["_ng_task_index"], r["_ng_rollout_index"]) for r in arms["tools"] if question_leaked(r)}
    out["leak"] = {"rollouts_with_question_text_in_results": len(leaked) / len(arms["tools"]),
                   "primary_without_leaked": paired_ci(
                       by_task(arms["tools"], lambda r: r["reward"], n, keep=lambda r: (r["_ng_task_index"], r["_ng_rollout_index"]) not in leaked),
                       reward["placebo"])}
    out["by_difficulty"] = {}
    for level in sorted({r["task_difficulty_qwen3_32b_avg_8"] for r in sample}):
        idx = [i for i, r in enumerate(sample) if r["task_difficulty_qwen3_32b_avg_8"] == level]
        out["by_difficulty"][str(level)] = {arm: statistics.mean(statistics.mean(reward[arm][i]) for i in idx) for arm in arms}

    for arm in TOOL_ARMS:
        rows = arms[arm]
        events = [tool_events(r) for r in rows]
        searches = [sum(name == "search" for name, *_ in ev) for ev in events]
        browses = [sum(name == "browse" for name, *_ in ev) for ev in events]
        kinds = collections.Counter(f"{name}:{kind}" for ev in events for name, kind, _, _ in ev)
        used = [any(name == "search" and kind == "ok" for name, kind, _, _ in ev) for ev in events]
        credits = [r["num_search_credits"] for r in rows]
        out[f"usage_{arm}"] = {
            "searches_per_rollout": {"mean": statistics.mean(searches), "p50": pct(searches, 0.5), "p90": pct(searches, 0.9), "max": max(searches)},
            "browses_per_rollout": {"mean": statistics.mean(browses), "p50": pct(browses, 0.5), "p90": pct(browses, 0.9), "max": max(browses)},
            "credits_per_rollout": statistics.mean(credits), "credits_total": sum(credits),
            "retries_total": sum(r["num_search_retries"] for r in rows),
            "rollouts_with_successful_search": statistics.mean(used),
            "rollouts_without_any_tool_call": statistics.mean(not ev for ev in events),
            "event_kinds": dict(sorted(kinds.items())),
            "search_rejected_share": kinds["search:rejected"] / max(1, sum(searches)),
            "reward_when_search_used": statistics.mean(r["reward"] for r, u in zip(rows, used) if u) if any(used) else None,
            "reward_when_search_not_used": statistics.mean(r["reward"] for r, u in zip(rows, used) if not u) if not all(used) else None,
        }
    credits_per_rollout = out["usage_tools"]["credits_per_rollout"]
    out["epoch_cost_usd"] = {depth: credits_per_rollout * mult * EPOCH_ROLLOUTS * USD_PER_CREDIT for depth, mult in (("basic", 1), ("advanced", 2))}
    for arm, rows in arms.items():
        parsed = [rescore(r)[1] for r in rows]
        ends = collections.Counter(end_state(r) for r in rows)
        usage = [r["response"].get("usage") or {} for r in rows]
        have = [u for u in usage if u.get("input_tokens") is not None and u.get("output_tokens") is not None]
        out[f"answers_{arm}"] = {
            "answer_found": statistics.mean(p is not None for p in parsed),
            "end_states": {k: ends[k] / len(rows) for k in ("message", "incomplete", "step_cap", "other")},
            "format_only_miss": statistics.mean(r["reward"] == 0 and normalize_option(p) == r["expected_answer"] for r, p in zip(rows, parsed)),
            "usage_missing_rows": len(rows) - len(have),  # 없는 값은 평균에 넣지 않는다
            "input_tokens_sum_mean": statistics.mean(u["input_tokens"] for u in have) if have else None,
            "input_tokens_sum_max": max(u["input_tokens"] for u in have) if have else None,
            "output_tokens_sum_mean": statistics.mean(u["output_tokens"] for u in have) if have else None,
            "output_tokens_sum_max": max(u["output_tokens"] for u in have) if have else None,
        }
    out["verdict"] = verdict(out["primary_tools_minus_placebo"], out["primary_completed_only"], out["usage_tools"]["rollouts_with_successful_search"])
    return out


def fmt_ci(ci):
    return "값 없음" if ci["diff"] is None else f"{ci['diff']:+.3f} (95% {ci['lo']:+.3f} ~ {ci['hi']:+.3f}, 문항 {ci['n']})"


def analyze(out_dir, paths, repeats, search_cost, json_path):
    sample = read_jsonl(Path(out_dir) / "sample.jsonl")
    arms = {arm: read_jsonl(paths[arm]) for arm in ARMS}
    checks = validity(sample, arms, repeats, search_cost)
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}{'' if ok else ' — ' + detail}")
    valid = all(ok for _, ok, _ in checks)
    result = {"valid": valid, "checks": [{"name": n, "pass": ok, "detail": d} for n, ok, d in checks]}
    if all(ok for name, ok, _ in checks if name[:2] in ("V1", "V6")):  # 행이 온전하면 수치는 보여 준다 — 유효성 FAIL 이면 결론으로 쓰지 않는다
        m = metrics(sample, arms, repeats)
        result["metrics"] = m
        print("\n보상  " + " · ".join(f"{arm} {m[f'reward_{arm}']['mean']:.3f} ± {m[f'reward_{arm}']['se']:.3f}" for arm in ARMS) +
              f" · 무작위 {m['chance_floor']:.3f}  (문항 {m['n_tasks']} × 반복 {repeats}, ± 는 문항 단위 표준오차)")
        print(f"주 대조  tools − placebo = {fmt_ci(m['primary_tools_minus_placebo'])}")
        print(f"  끝까지 답한 롤아웃만 = {fmt_ci(m['primary_completed_only'])}")
        print(f"  질문 원문이 결과에 나온 롤아웃 ({m['leak']['rollouts_with_question_text_in_results']:.1%}) 을 뺀 것 = {fmt_ci(m['leak']['primary_without_leaked'])}")
        print(f"  → 판정: {m['verdict']}")
        print(f"서술용  tools − notools = {fmt_ci(m['tools_minus_notools'])} · placebo − notools = {fmt_ci(m['placebo_minus_notools'])}"
              " (지시문 · 턴 수 · 총 생성량이 함께 다르다)")
        print("잡음 (같은 갈래의 반복 앞 절반 − 뒤 절반): " + " · ".join(f"{arm} {fmt_ci(m[f'noise_aa_{arm}'])}" for arm in ARMS))
        print("난이도 표지별: " + " · ".join(f"{k}: " + " / ".join(f"{v[arm]:.3f}" for arm in ARMS) for k, v in m["by_difficulty"].items()) +
              f"  ({' / '.join(ARMS)})")
        for arm in TOOL_ARMS:
            u = m[f"usage_{arm}"]
            print(f"[{arm}] 검색/롤아웃 평균 {u['searches_per_rollout']['mean']:.2f} · p50 {u['searches_per_rollout']['p50']} · p90 {u['searches_per_rollout']['p90']}"
                  f" · 최대 {u['searches_per_rollout']['max']} | browse 평균 {u['browses_per_rollout']['mean']:.2f} · 최대 {u['browses_per_rollout']['max']}"
                  f" | 크레딧/롤아웃 {u['credits_per_rollout']:.2f} (합 {u['credits_total']}) · 재시도 {u['retries_total']}")
            print(f"[{arm}] 결과가 있는 검색을 한 롤아웃 {u['rollouts_with_successful_search']:.1%} · 도구 호출 0 인 롤아웃 {u['rollouts_without_any_tool_call']:.1%}"
                  f" · 검색 거절 비율 {u['search_rejected_share']:.1%} · 도구 결과 종류 {u['event_kinds']}")
        u = m["usage_tools"]
        print(f"[tools] 검색을 쓴 롤아웃의 보상 {u['reward_when_search_used']} · 안 쓴 롤아웃 {u['reward_when_search_not_used']} (서술용 — 쓴 쪽이 어려운 문항일 수 있다)")
        print(f"1 에폭 ({EPOCH_ROLLOUTS:,} 롤아웃) 검색 비용 추정: basic ${m['epoch_cost_usd']['basic']:,.0f} · advanced ${m['epoch_cost_usd']['advanced']:,.0f}"
              " (이 측정의 턴 상한 · 생성 상한에서의 값)")
        for arm in ARMS:
            a = m[f"answers_{arm}"]
            e = a["end_states"]
            tokens = "usage 없음" if a["input_tokens_sum_mean"] is None else (
                f"입력 토큰 합 평균 {a['input_tokens_sum_mean']:,.0f} (최대 {a['input_tokens_sum_max']:,}) · 출력 토큰 합 평균 "
                f"{a['output_tokens_sum_mean']:,.0f} (최대 {a['output_tokens_sum_max']:,})")
            print(f"[{arm}] 답 찾음 {a['answer_found']:.1%} · 끝 상태: 답함 {e['message']:.1%} / 생성 상한 등 {e['incomplete']:.1%} / 스텝 상한 {e['step_cap']:.1%}"
                  f" / 기타 {e['other']:.1%} · 형식만 틀린 0 점 {a['format_only_miss']:.1%} · {tokens} · usage 없는 행 {a['usage_missing_rows']}")
    print("\n유효성:", "PASS" if valid else "FAIL — 수치를 결론으로 쓰지 않는다")
    if json_path:
        Path(json_path).write_text(json.dumps(result, ensure_ascii=False, indent=1))
    return 0 if valid else 1


# ---------------------------------------------------------------- 자가 시험


def _call(call_id, name, args, output):
    return [{"type": "function_call", "call_id": call_id, "name": name, "arguments": json.dumps(args)},
            {"type": "function_call_output", "call_id": call_id, "output": json.dumps(output)}]


def _message(letter):
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": rf"\boxed{{{letter}}}"}]}


def _synthetic(n_tasks=6, repeats=2):
    """tools 는 문항 절반을 맞히고 placebo · notools 는 1/3 을 맞히는 깨끗한 결과."""
    sample, arms = [], {arm: [] for arm in ARMS}
    for t in range(n_tasks):
        question = f"Which mechanism explains observation number {t} in the transplanted organ under these stated conditions today?\nA: a\nB: b\nC: c\nD: d"
        sample.append({"expected_answer": "ABCD"[t % 4], "task_difficulty_qwen3_32b_avg_8": (0.25, 0.5)[t % 2],
                       "responses_create_params": {"input": question, "instructions": "x", "tools": [{"name": "search"}, {"name": "browse"}]}})
        for i in range(repeats):
            for arm in ARMS:
                correct = (t + i) % 2 == 0 if arm == "tools" else t % 3 == 0
                output = []
                if arm == "tools":
                    output += _call("c1", "search", {"query": "q"}, {"search_results": "Search Results\n==============\n[1] T (a.org)\n    URL: https://a.org\n    Summary: s\n\n"})
                    output += _call("c2", "browse", {"url": "https://a.org"}, {"page_content": "page"})
                elif arm == "placebo":
                    output += _call("c1", "search", {"query": "q"}, {"search_results": "Search Results\n==============\nNo results found.\n"})
                    output += _call("c2", "browse", {"url": "https://a.org"}, {"page_content": "No HTML found"})
                letter = sample[t]["expected_answer"] if correct else "ABCD"[(t + 1) % 4]
                output.append(_message(letter))
                params = copy.deepcopy(sample[t]["responses_create_params"])
                if arm == "notools":
                    del params["tools"]
                called = int(arm in TOOL_ARMS)
                arms[arm].append({"_ng_task_index": t, "_ng_rollout_index": i, "reward": 1.0 if correct else 0.0, "expected_answer": sample[t]["expected_answer"],
                                  "answer_found": True, "parsed_option": letter, "responses_create_params": params, "instance_config": None,
                                  "response": {"output": output, "usage": {"input_tokens": 100, "output_tokens": 10}},
                                  "num_search_calls": called, "num_search_errors": 0, "num_search_retries": 0, "num_search_usage_missing": 0,
                                  "num_search_credits": int(arm == "tools"), "num_browse_calls": called, "num_browse_errors": int(arm == "placebo")})
    return sample, arms


def self_test():
    problems = []

    def failed(mutate):
        sample, arms = _synthetic()
        mutate(sample, arms)
        return {name[:2] for name, ok, _ in validity(sample, arms, 2) if not ok}

    def call_in_notools(s, a):
        a["notools"][0]["response"]["output"][:0] = _call("c", "search", {}, {"detail": "x"})

    def reject_all(s, a):  # 검색 엔진이 모든 질의를 거절 — 장애로 세지 않는 경로라 V4 는 못 본다
        for r in a["tools"]:
            r["response"]["output"][1]["output"] = json.dumps({"search_results": "Error: the search engine rejected this query"})
            r["num_search_credits"] = 0

    mutations = {
        "행 하나 누락": (lambda s, a: a["tools"].pop(), "V1"),
        "(문항, 반복) 중복": (lambda s, a: a["placebo"][1].update(_ng_rollout_index=0), "V1"),
        "통계 키 누락": (lambda s, a: a["tools"][0].pop("num_search_calls"), "V1"),
        "통계 값 None": (lambda s, a: a["tools"][0].update(num_search_credits=None), "V1"),
        "notools 에 도구 호출": (call_in_notools, "V2"),
        "notools 에 도구 정의": (lambda s, a: a["notools"][0]["responses_create_params"].update(tools=[{"name": "search"}]), "V2"),
        "placebo 가 실제 검색 결과를 받음": (lambda s, a: a["placebo"][0]["response"]["output"].__setitem__(1, a["tools"][0]["response"]["output"][1]), "V2"),
        "서버 통계가 0 (세션 끊김)": (lambda s, a: a["tools"][0].update(num_search_calls=0), "V3"),
        "서버에 닿지 않은 호출을 닿은 것으로 셈": (lambda s, a: a["tools"][0]["response"]["output"][1].update(output=json.dumps({"detail": "x"})), "V3"),
        "백엔드 장애 표시": (lambda s, a: a["tools"][0].update(num_search_errors=1, instance_config={"mask_sample": True}), "V4"),
        "보상 뒤집힘": (lambda s, a: a["notools"][0].update(reward=1.0 - a["notools"][0]["reward"]), "V5"),
        "문항 번호 어긋남": (lambda s, a: [r.update(_ng_task_index=(r["_ng_task_index"] + 1) % 6) for r in a["tools"]], "V6"),
        "검색 엔진이 전부 거절 (검색 경로 사망)": (reject_all, "V7"),
        "크레딧이 0 으로 기록 (usage 누락을 0 으로 읽음)": (lambda s, a: [r.update(num_search_credits=0) for r in a["tools"]], "V7"),
        "usage 누락 표시": (lambda s, a: a["tools"][0].update(num_search_usage_missing=1), "V7"),
    }
    if failed(lambda s, a: None):
        problems.append(f"변조 없는 합성 결과가 FAIL: {failed(lambda s, a: None)}")
    for name, (mutate, target) in mutations.items():
        got = failed(mutate)
        print(f"{'ok  ' if target in got else 'MISS'} {name} → {sorted(got) or '통과'} (겨냥 {target})")
        if target not in got:
            problems.append(name)

    # 판정 규칙: 유효성은 통과하지만 결론을 내면 안 되는 경우
    def judged(mutate):
        sample, arms = _synthetic(n_tasks=12, repeats=4)
        mutate(sample, arms)
        bad = {name[:2] for name, ok, _ in validity(sample, arms, 4) if not ok}
        return bad, metrics(sample, arms, 4)

    def no_tool_use(s, a):  # tools 갈래가 도구를 한 번도 안 불렀는데 보상만 높다
        for r in a["tools"]:
            r["response"]["output"] = [_message(r["expected_answer"])]
            r.update(reward=1.0, num_search_calls=0, num_browse_calls=0, num_search_credits=0)

    def placebo_truncated(s, a):  # placebo 만 생성 상한에 잘려 0 점 — 차이는 검색이 아니라 잘림에서 나온다
        for r in a["tools"]:
            r["response"]["output"][-1] = _message(r["expected_answer"])
            r["reward"] = 1.0
        for r in a["placebo"]:
            r["response"]["output"] = r["response"]["output"][:-1] + [{"type": "reasoning", "summary": []}]
            r["response"]["incomplete_details"] = {"reason": "max_output_tokens"}
            r.update(reward=0.0)

    def clear_win(s, a):  # tools 가 끝까지 답하며 전부 맞히고 placebo 는 그대로 — 판정이 양성으로 나올 수 있어야 한다
        for r in a["tools"]:
            r["response"]["output"][-1] = _message(r["expected_answer"])
            r["reward"] = 1.0

    def shuffled(s, a):
        for rows in a.values():
            random.Random(1).shuffle(rows)

    def leak(s, a):
        for r in a["tools"]:
            stem = r["responses_create_params"]["input"].split("\n")[0]
            r["response"]["output"][1]["output"] = json.dumps({"search_results": f"Search Results\n==============\n[1] T (a.org)\n    URL: https://a.org\n    Summary: {stem} Answer: A\n\n"})

    bad, base = judged(lambda s, a: None)
    cases = {
        "판정: 도구 호출 0 회인데 보상 차이": (judged(no_tool_use), lambda bad, m: not bad and m["verdict"].startswith("도구 미사용")),
        "판정: placebo 만 잘림": (judged(placebo_truncated), lambda bad, m: not bad and m["verdict"].startswith("판정 보류")),
        "판정: 검색 갈래가 뚜렷이 높음": (judged(clear_win), lambda bad, m: not bad and m["verdict"] == "검색이 기여한다"),
        "잡음 추정이 파일 순서와 무관": (judged(shuffled), lambda bad, m: not bad and m["noise_aa_tools"] == base["noise_aa_tools"]
                                and m["primary_tools_minus_placebo"] == base["primary_tools_minus_placebo"]),
        "질문 원문 누출 표시": (judged(leak), lambda bad, m: m["leak"]["rollouts_with_question_text_in_results"] == 1.0),
        "깨끗한 결과에는 누출 0 · 끝 상태 전부 답함": ((bad, base), lambda bad, m: not bad and m["leak"]["rollouts_with_question_text_in_results"] == 0
                                       and m["answers_tools"]["end_states"]["message"] == 1.0 and m["usage_tools"]["credits_per_rollout"] == 1
                                       and abs(m["chance_floor"] - 0.25) < 1e-9),
    }
    for name, ((bad_, m), ok) in cases.items():
        good = ok(bad_, m)
        print(f"{'ok  ' if good else 'MISS'} {name} → 유효성 FAIL {sorted(bad_) or '없음'} · 판정 '{m['verdict']}'")
        if not good:
            problems.append(name)

    with tempfile.TemporaryDirectory() as tmp:
        def smoke_rc(arm, mutate=None):
            _, synthetic = _synthetic()
            rows = synthetic[arm][:4]
            if mutate:
                mutate(rows)
            write_jsonl(Path(tmp) / "r.jsonl", rows)
            return smoke_check(Path(tmp) / "r.jsonl", arm, 1)

        def all_truncated(rows):
            for r in rows:
                r["response"]["output"] = [{"type": "reasoning", "summary": []}]
                r["response"]["incomplete_details"] = {"reason": "max_output_tokens"}
                r["reward"] = 0.0

        smoke_cases = {
            "smoke: 정상 tools": (smoke_rc("tools"), 0),
            "smoke: 정상 placebo": (smoke_rc("placebo"), 0),
            "smoke: 정상 notools": (smoke_rc("notools"), 0),
            "smoke: 세션 끊김": (smoke_rc("tools", lambda rows: rows[0].update(num_search_calls=0)), 1),
            "smoke: 백엔드 장애": (smoke_rc("tools", lambda rows: rows[0].update(num_search_errors=1)), 1),
            "smoke: 키 없음": (smoke_rc("tools", lambda rows: rows[0].pop("num_browse_calls")), 1),
            "smoke: notools 에 도구 호출": (smoke_rc("notools", lambda rows: call_in_notools(None, {"notools": rows})), 1),
            "smoke: 검색이 전부 거절": (smoke_rc("tools", lambda rows: reject_all(None, {"tools": rows})), 1),
            "smoke: 전부 잘림": (smoke_rc("notools", all_truncated), 1),
            "smoke: placebo 에 실제 결과": (smoke_rc("placebo", lambda rows: rows[0].update(num_search_credits=1)), 1),
            "smoke: 빈 파일": (smoke_rc("tools", lambda rows: rows.clear()), 1),
        }
    for name, (got, want) in smoke_cases.items():
        print(f"{'ok  ' if got == want else 'MISS'} {name} → rc {got} (기대 {want})")
        if got != want:
            problems.append(name)
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
    for arm in ARMS:
        a.add_argument(f"--{arm}", required=True)
    a.add_argument("--repeats", type=int, default=4)
    a.add_argument("--search-cost", type=int, default=1, help="검색 1회의 정가 (basic 1 · advanced 2)")
    a.add_argument("--json", default=None)
    sub.add_parser("self-test")
    c = sub.add_parser("smoke-check")
    c.add_argument("rollouts")
    c.add_argument("arm", choices=ARMS)
    c.add_argument("--search-cost", type=int, default=1)
    args = parser.parse_args()
    if args.cmd == "build":
        build(args.source, args.out_dir, args.per_stratum)
        return 0
    if args.cmd == "self-test":
        return self_test()
    if args.cmd == "smoke-check":
        return smoke_check(args.rollouts, args.arm, args.search_cost)
    return analyze(args.out_dir, {arm: getattr(args, arm) for arm in ARMS}, args.repeats, args.search_cost, args.json)


if __name__ == "__main__":
    sys.exit(main())
