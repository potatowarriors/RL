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
검색 백엔드 장애 · 크레딧 장부 어긋남은 행의 5% (INFRA_TOLERANCE — 인프라 허용치, 효과 판정과 무관한 상수) 까지는 그 롤아웃을 빼고 진행한다.
판정 규칙 (실행 전에 고정). 유효성이 FAIL 이면 판정을 내지 않는다. "검색이 기여한다" 는 아래 구간의 95% 하한이 모두 0 보다 클 때만 낸다:
  ① tools − placebo (주 대조)  ② 끝까지 답한 롤아웃만으로 다시 낸 것 (잘림 · 스텝 상한이 만든 차이를 가른다)
  ③ 백엔드 장애로 뺀 tools 롤아웃을 0 점으로 넣어 다시 낸 것 (장애가 어려운 문항에 몰려 효과가 부풀지 않았는가)
  ④ tools − notools (도구 없이도 되는 수준을 넘는가)
  ①이 양인데 ②~④ 중 하나라도 아니면 "판정 보류". placebo − notools 의 상한 < 0 이면 placebo 가 모델을 망가뜨린 것이라 "판정 보류".
  질문 원문이 도구 결과에 그대로 나온 (또는 제외 도메인의 결과를 받은) tools 롤아웃이 5% 를 넘으면 "판정 보류 (누출)" — 누출은 모델이 답을 모를 때
  몰리므로 그 롤아웃만 빼고 다시 내는 것으로는 가를 수 없다 (뺀 대조는 서술용으로만 낸다). 실패 뒤 다시 뽑힌 롤아웃이 5% 를 넘어도 "판정 보류".
  "검색이 깎는다" 는 ① · ② · ④ 의 상한이 모두 0 보다 작을 때만 낸다. 검색 성공이 한 번도 없으면 "도구 미사용 — 판정 불가".
  결과가 있는 검색을 한 롤아웃이 30% 미만이면 차이가 안 보여도 "판정 불가 (도구 사용 부족)" — 학습 전 모델이 도구를 덜 쓴 것이지 환경의 한계가 아니다.
  구간이 0 을 품으면 "구분 안 됨" 이지 "기여 없음" 이 아니다. 48 문항 × 4 에서 "기여" 가 나올 확률 (모의): 실제 차이 0.10 이면 약 0.4, 0.15 면 약 0.8,
  0.20 이면 약 0.95. 이 측정은 양성 판정에 엄격하다 (거짓 양성을 피하는 쪽).
알려진 한계: 빈 결과가 모델을 조금 망가뜨린 몫과 지시문 · 턴 수의 몫이 비슷하면 ① 과 ④ 가 함께 양이 될 수 있다 — 세 갈래로는 가르지 못한다.
            스텝 상한이 5 라 검색 횟수 분포가 5 에서 잘린다 (비용 · 효과 모두 이 상한에서의 값).
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
INFRA_TOLERANCE = 0.05  # 인프라 탓 행 (백엔드 장애 · 장부 어긋남 · 다시 뽑힘) 의 허용 비율 — 한 갈래에서 이런 행을 합쳐서 센다
LEAK_TOLERANCE = 0.05   # 누출 롤아웃 비율이 이를 넘으면 판정 보류
MIN_SEARCH_USE = 0.30   # 결과가 있는 검색을 한 롤아웃 비율이 이보다 낮으면 "구분 안 됨" 을 근거로 쓰지 않는다
EXCLUDED_DOMAINS = ("huggingface.co", "hf.co", "hf-mirror.com")  # 플러그인 설정과 같아야 한다 (실행기가 기동 설정과 대조)
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
    """message = 끝까지 답함 · generation_cap = 생성 상한에 잘림 · context_overflow = 문맥이 차서 모델이 아무것도 못 냄
    · step_cap = 스텝 상한 (마지막이 도구 결과면 그 결과를 모델이 못 봤다) · other.

    문맥 초과는 Gym vllm_model 이 incomplete_details 가 붙은 빈 assistant 메시지로 바꾼다 — 마지막 도구 결과 뒤에 모델이 낸 것이 없으면 문맥 초과로 본다.
    """
    output = row["response"]["output"]
    last = output[-1] if output else {}
    if row["response"].get("incomplete_details"):
        tail = output[max((i for i, o in enumerate(output) if o.get("type") == "function_call_output"), default=-1) + 1:]
        generated = any(o.get("type") in ("reasoning", "function_call") or (o.get("type") == "message" and output_text(o).strip()) for o in tail)
        return "generation_cap" if generated else "context_overflow"
    if last.get("type") == "message" and last.get("role") == "assistant":
        return "message"
    return "step_cap" if last.get("type") in ("function_call_output", "function_call", "reasoning") else "other"


END_STATES = ("message", "generation_cap", "context_overflow", "step_cap", "other")


def backend_failed(row):
    return bool(row["num_search_errors"] or row.get("instance_config") or any(kind == "backend_error" for _, kind, _, _ in tool_events(row)))


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
        # 어느 핸들러가 답했는지는 응답 본문의 키로 본다. Gym 은 `/{이름}` 으로 보내서 `search? Actually…` 같은 이름도 /search 핸들러에 닿는다.
        text = body.get("search_results") if isinstance(body, dict) and isinstance(body.get("search_results"), str) else None
        page = body.get("page_content") if isinstance(body, dict) and isinstance(body.get("page_content"), str) else None
        name = "search" if text is not None else "browse" if page is not None else re.split(r"[?#]", name, maxsplit=1)[0]
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


def from_excluded_domain(row):
    """검색 결과의 URL 이 제외 도메인인가 (검색 엔진이 제외 목록을 지켰는지의 사후 확인)."""
    for name, kind, _, text in tool_events(row):
        if name == "search" and kind == "ok":
            for host in re.findall(r"^    URL: \S*?://([^/\s:]+)", text, re.MULTILINE):
                host = host.lower().rstrip(".")
                if any(host == d or host.endswith("." + d) for d in EXCLUDED_DOMAINS):
                    return True
    return False


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
    if backend_failed(r):
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
    notes = collections.defaultdict(list)
    if not found["V1"]:  # 아래 검사는 키가 있어야 한다
        for arm, rows in arms.items():
            per_check, infra_rows = collections.Counter(), 0
            for r in rows:
                problems = set(row_problems(r, arm, search_cost))
                infra_rows += any(p[:2] in ("V4", "V7") for p in problems)
                for p in problems:
                    per_check[tuple(p.split(" ", 1))] += 1
            # 인프라 탓 행 (백엔드 장애 · 장부 어긋남) 은 한 갈래에서 합쳐 허용치까지 봐준다 — 한 번뿐인 예산에서 한 행 때문에 전부 버리지 않으려고.
            # 백엔드 장애 행은 보상 계산에서 빠진다 (학습에서도 손실에서 빠진다). 장부가 어긋난 행은 그대로 남아 비용 추정에 들어간다.
            tolerated = infra_rows / len(rows) <= INFRA_TOLERANCE
            for (check, text), n in sorted(per_check.items()):
                line = f"{arm}: {text} {n} 행 ({n / len(rows):.1%})"
                (notes if check in ("V4", "V7") and tolerated else found)[check].append(line)
            bad = sum(r["expected_answer"] != sample[r["_ng_task_index"]]["expected_answer"] or
                      r["responses_create_params"]["input"] != sample[r["_ng_task_index"]]["responses_create_params"]["input"] for r in rows)
            if bad:
                found["V6"].append(f"{arm}: 표본의 같은 번호 문항과 질문 · 정답이 다른 행 {bad}")
        found["V7"] += search_liveness_problems(arms["tools"], min_browses=10, max_rejected=0.1, min_answered=0.8)
    out = []
    for v in names:
        if v != "V1" and found["V1"]:
            continue
        lines = found[v] or [f"주의 (인프라 탓 행이 허용치 {INFRA_TOLERANCE:.0%} 안): " + "; ".join(notes[v])] * bool(notes[v])
        out.append((f"{v} {names[v]}", not found[v], "; ".join(lines[:4]) + (f" … 외 {len(lines) - 4}" if len(lines) > 4 else "")))
    return out


def search_liveness_problems(rows, min_browses, max_rejected, min_answered):
    """tools 갈래의 검색 · browse 가 살아 있는가. 모델 탓 질의 (빈 · 400 자 초과) 는 검색 엔진에 가지 않으므로 분모에서 뺀다."""
    kinds = collections.Counter(f"{name}:{kind}" for r in rows for name, kind, reached, _ in tool_events(r) if reached)
    sent = kinds["search:ok"] + kinds["search:ok_empty"] + kinds["search:rejected"] + kinds["search:backend_error"]
    browses = sum(n for k, n in kinds.items() if k.startswith("browse:"))
    answered = kinds["search:ok"] + kinds["search:ok_empty"]
    problems = []
    if sent and not kinds["search:ok"]:
        problems.append(f"tools: 검색 엔진에 보낸 {sent} 회 중 결과가 있는 검색 0 회")
    if sent and answered / sent < min_answered:
        problems.append(f"tools: 검색 엔진이 답한 검색 {answered}/{sent} ({answered / sent:.1%}) < {min_answered:.0%}")
    if sent and kinds["search:rejected"] / sent > max_rejected:
        problems.append(f"tools: 검색 엔진이 거절한 검색 {kinds['search:rejected']}/{sent} > {max_rejected:.0%}")
    if browses >= min_browses and not kinds["browse:ok"]:
        problems.append(f"tools: browse {browses} 회 중 본문을 얻은 것 0 회")
    return problems


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
    if arm == "tools":  # 본 실행과 같은 검사, 행이 적어 문턱만 느슨하게 (거절 50% · 응답 50% · browse 3 회)
        problems += ["V7 " + p for p in search_liveness_problems(complete, min_browses=3, max_rejected=0.5, min_answered=0.5)]
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


def verdict(m):
    """판정 규칙 (모듈 설명) 을 수치 dict 에 적용한다. 유효성 FAIL 이면 부르지 않는다."""
    use = m["usage_tools"]["rollouts_with_successful_search"]
    if not use:
        return "도구 미사용 — 판정 불가"
    primary = sign(m["primary_tools_minus_placebo"])
    if primary is None:
        return "판정 보류 (주 대조를 낼 수 없다)"
    redrawn = [f"{arm} {share:.1%}" for arm, share in m["redrawn_share"].items() if arm in TOOL_ARMS and share > INFRA_TOLERANCE]
    if redrawn:
        return f"판정 보류 (실패 뒤 다시 뽑힌 롤아웃이 많다: {', '.join(redrawn)})"
    if m["leak"]["leaked_share"] > LEAK_TOLERANCE:
        return f"판정 보류 (누출: tools 롤아웃의 {m['leak']['leaked_share']:.1%} 에서 질문 원문 · 제외 도메인이 도구 결과에 나왔다)"
    if primary == 1:
        conditions = {"끝까지 답한 롤아웃만": m["primary_completed_only"], "장애 롤아웃을 0 점으로": m["primary_failed_as_zero"],
                      "도구 없음 대비": m["tools_minus_notools"]}
        holds = [name for name, ci in conditions.items() if sign(ci) != 1]
        if m["placebo_minus_notools"]["hi"] is not None and m["placebo_minus_notools"]["hi"] < 0:
            holds.append("placebo 가 도구 없음보다 낮다 — 빈 결과가 모델을 망가뜨렸다")
        return "검색이 기여한다" if not holds else f"판정 보류 (주 대조는 양이나: {' · '.join(holds)})"
    if primary == -1:
        conditions = {"끝까지 답한 롤아웃만": m["primary_completed_only"], "도구 없음 대비": m["tools_minus_notools"]}
        holds = [name for name, ci in conditions.items() if sign(ci) != -1]
        return "검색이 깎는다" if not holds else f"판정 보류 (주 대조는 음이나: {' · '.join(holds)})"
    if use < MIN_SEARCH_USE:
        return f"판정 불가 (도구 사용 부족 — 결과가 있는 검색을 한 롤아웃 {use:.1%} < {MIN_SEARCH_USE:.0%}, 차이가 안 보이는 것을 근거로 쓰지 않는다)"
    return "구분 안 됨"


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def metrics(sample, arms, repeats, redrawn=None):
    n = len(sample)
    usable = lambda r: not backend_failed(r)  # noqa: E731 — 백엔드 장애 롤아웃은 학습에서도 손실에서 빠진다
    reward = {arm: by_task(rows, lambda r: r["reward"], n, keep=usable) for arm, rows in arms.items()}
    done = {arm: by_task(rows, lambda r: r["reward"], n, keep=lambda r: usable(r) and end_state(r) == "message") for arm, rows in arms.items()}
    out = {"n_tasks": n, "chance_floor": statistics.mean(1 / n_options(r) for r in sample),
           "excluded_backend_failed": {arm: sum(not usable(r) for r in rows) for arm, rows in arms.items()},
           "redrawn_share": {arm: (redrawn or {}).get(arm, 0) / len(rows) for arm, rows in arms.items()}}
    for arm in arms:
        m, se = mean_se(reward[arm])
        out[f"reward_{arm}"] = {"mean": m, "se": se}
        half = repeats // 2  # 반복 번호 앞 절반 대 뒤 절반 — 같은 갈래 안의 잡음 크기
        out[f"noise_aa_{arm}"] = paired_ci([v[:half] for v in reward[arm]], [v[half:] for v in reward[arm]])
    out["primary_tools_minus_placebo"] = paired_ci(reward["tools"], reward["placebo"])
    # 뺀 (백엔드 장애) tools 롤아웃을 0 점으로 넣은 것 — 장애가 어려운 문항에 몰렸다면 빼는 것이 효과를 부풀린다
    out["primary_failed_as_zero"] = paired_ci(by_task(arms["tools"], lambda r: r["reward"] if usable(r) else 0.0, n), reward["placebo"])
    out["primary_completed_only"] = paired_ci(done["tools"], done["placebo"])
    out["tools_minus_notools"] = paired_ci(reward["tools"], reward["notools"])
    out["placebo_minus_notools"] = paired_ci(reward["placebo"], reward["notools"])
    leaked = {(r["_ng_task_index"], r["_ng_rollout_index"]) for r in arms["tools"] if question_leaked(r) or from_excluded_domain(r)}
    out["leak"] = {"leaked_share": len(leaked) / len(arms["tools"]),
                   "question_text_in_results": statistics.mean(question_leaked(r) for r in arms["tools"]),
                   "results_from_excluded_domains": statistics.mean(from_excluded_domain(r) for r in arms["tools"]),
                   "primary_without_leaked": paired_ci(
                       by_task(arms["tools"], lambda r: r["reward"], n,
                               keep=lambda r: usable(r) and (r["_ng_task_index"], r["_ng_rollout_index"]) not in leaked),
                       reward["placebo"])}
    out["by_difficulty"] = {}
    for level in sorted({r["task_difficulty_qwen3_32b_avg_8"] for r in sample}):
        idx = [i for i, r in enumerate(sample) if r["task_difficulty_qwen3_32b_avg_8"] == level]
        level_means = {arm: [statistics.mean(reward[arm][i]) for i in idx if reward[arm][i]] for arm in arms}  # 쓸 롤아웃이 없는 문항은 뺀다
        out["by_difficulty"][str(level)] = {arm: statistics.mean(v) if v else None for arm, v in level_means.items()}

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
            "search_model_error_share": kinds["search:model_error"] / max(1, sum(searches)),  # 빈 · 400 자 초과 질의 (질문을 통째로 붙여넣음)
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
            "end_states": {k: ends[k] / len(rows) for k in END_STATES},
            "format_only_miss": statistics.mean(r["reward"] == 0 and normalize_option(p) == r["expected_answer"] for r, p in zip(rows, parsed)),
            "usage_missing_rows": len(rows) - len(have),  # 없는 값은 평균에 넣지 않는다
            "input_tokens_sum_mean": statistics.mean(u["input_tokens"] for u in have) if have else None,
            "input_tokens_sum_max": max(u["input_tokens"] for u in have) if have else None,
            "output_tokens_sum_mean": statistics.mean(u["output_tokens"] for u in have) if have else None,
            "output_tokens_sum_max": max(u["output_tokens"] for u in have) if have else None,
        }
    return out


def fmt_ci(ci):
    return "값 없음" if ci["diff"] is None else f"{ci['diff']:+.3f} (95% {ci['lo']:+.3f} ~ {ci['hi']:+.3f}, 문항 {ci['n']})"


def analyze(out_dir, paths, repeats, search_cost, json_path):
    sample = read_jsonl(Path(out_dir) / "sample.jsonl")
    arms = {arm: read_jsonl(paths[arm]) for arm in ARMS}
    sidecars = {arm: Path(paths[arm]).with_name(Path(paths[arm]).stem + "_failures.jsonl") for arm in ARMS}
    redrawn = {arm: len(read_jsonl(f)) if f.exists() else 0 for arm, f in sidecars.items()}  # 실패 기록 1 줄 = 다시 뽑힌 롤아웃 1 개
    checks = validity(sample, arms, repeats, search_cost)
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}{' — ' + detail if detail else ''}")
    valid = all(ok for _, ok, _ in checks)
    result = {"valid": valid, "checks": [{"name": n, "pass": ok, "detail": d} for n, ok, d in checks]}
    if all(ok for name, ok, _ in checks if name[:2] in ("V1", "V6")):  # 행이 온전하면 수치는 보여 준다 — 유효성 FAIL 이면 결론으로 쓰지 않는다
        m = metrics(sample, arms, repeats, redrawn)
        m["verdict"] = verdict(m) if valid else None  # 유효성 FAIL 이면 판정을 내지 않는다
        result["metrics"] = m
        print(f"\n뺀 롤아웃 (백엔드 장애): {m['excluded_backend_failed']} · 실패 뒤 다시 뽑힌 롤아웃: {redrawn}")
        print("보상  " + " · ".join(f"{arm} {m[f'reward_{arm}']['mean']:.3f} ± {m[f'reward_{arm}']['se']:.3f}" for arm in ARMS) +
              f" · 무작위 {m['chance_floor']:.3f}  (문항 {m['n_tasks']} × 반복 {repeats}, ± 는 문항 단위 표준오차)")
        print(f"① 주 대조  tools − placebo = {fmt_ci(m['primary_tools_minus_placebo'])}")
        print(f"② 끝까지 답한 롤아웃만 = {fmt_ci(m['primary_completed_only'])}")
        print(f"③ 장애로 뺀 tools 롤아웃을 0 점으로 = {fmt_ci(m['primary_failed_as_zero'])}")
        print(f"④ tools − notools = {fmt_ci(m['tools_minus_notools'])}  (지시문 · 턴 수 · 총 생성량이 함께 다르다 — 단독으로는 검색의 기여가 아니다)")
        print(f"   placebo − notools = {fmt_ci(m['placebo_minus_notools'])}  (상한 < 0 이면 빈 결과가 모델을 망가뜨린 것)")
        k = m["leak"]
        print(f"   누출: tools 롤아웃의 {k['leaked_share']:.1%} (질문 원문이 결과에 {k['question_text_in_results']:.1%} · 제외 도메인 결과 {k['results_from_excluded_domains']:.1%})"
              f" — 허용 {LEAK_TOLERANCE:.0%}. 그 롤아웃을 뺀 대조 (서술용) = {fmt_ci(k['primary_without_leaked'])}")
        print(f"  → 판정: {m['verdict'] if valid else '내지 않는다 (유효성 FAIL)'}")
        print("잡음 (같은 갈래의 반복 앞 절반 − 뒤 절반): " + " · ".join(f"{arm} {fmt_ci(m[f'noise_aa_{arm}'])}" for arm in ARMS))
        print("난이도 표지별: " + " · ".join(f"{k}: " + " / ".join("값 없음" if v[arm] is None else f"{v[arm]:.3f}" for arm in ARMS)
                                         for k, v in m["by_difficulty"].items()) +
              f"  ({' / '.join(ARMS)})")
        for arm in TOOL_ARMS:
            u = m[f"usage_{arm}"]
            print(f"[{arm}] 검색/롤아웃 평균 {u['searches_per_rollout']['mean']:.2f} · p50 {u['searches_per_rollout']['p50']} · p90 {u['searches_per_rollout']['p90']}"
                  f" · 최대 {u['searches_per_rollout']['max']} | browse 평균 {u['browses_per_rollout']['mean']:.2f} · 최대 {u['browses_per_rollout']['max']}"
                  f" | 크레딧/롤아웃 {u['credits_per_rollout']:.2f} (합 {u['credits_total']}) · 재시도 {u['retries_total']}")
            print(f"[{arm}] 결과가 있는 검색을 한 롤아웃 {u['rollouts_with_successful_search']:.1%} · 도구 호출 0 인 롤아웃 {u['rollouts_without_any_tool_call']:.1%}"
                  f" · 검색 거절 비율 {u['search_rejected_share']:.1%} · 모델 탓 질의 (빈 · 400 자 초과) {u['search_model_error_share']:.1%}"
                  f" · 도구 결과 종류 {u['event_kinds']}")
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
            print(f"[{arm}] 답 찾음 {a['answer_found']:.1%} · 끝 상태: 답함 {e['message']:.1%} / 생성 상한 {e['generation_cap']:.1%} / 문맥 초과 {e['context_overflow']:.1%}"
                  f" / 스텝 상한 {e['step_cap']:.1%} / 기타 {e['other']:.1%} · 형식만 틀린 0 점 {a['format_only_miss']:.1%} · {tokens} · usage 없는 행 {a['usage_missing_rows']}")
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
    def judged(mutate, redrawn=None):
        sample, arms = _synthetic(n_tasks=12, repeats=4)
        mutate(sample, arms)
        bad = {name[:2] for name, ok, _ in validity(sample, arms, 4) if not ok}
        m = metrics(sample, arms, 4, redrawn)
        m["verdict"] = verdict(m) if not bad else None  # analyze 와 같은 순서
        return bad, m

    SEARCH_OK = {"search_results": "Search Results\n==============\n[1] T (a.org)\n    URL: https://a.org\n    Summary: s\n\n"}

    def leak_within_questions(s, a):  # 모델이 아는 롤아웃은 누출 없이 맞히고, 모르는 롤아웃은 질문을 붙여넣어 답안지 페이지를 찾는다 (문항 안에서 섞임)
        for arm in ARMS:
            for r in a[arm]:
                knows = r["_ng_rollout_index"] < 2
                set_reward(r, knows or arm == "tools")
                if arm == "tools" and not knows:
                    stem = r["responses_create_params"]["input"].split("\n")[0]
                    r["response"]["output"][1]["output"] = json.dumps({"search_results": SEARCH_OK["search_results"].replace("Summary: s", f"Summary: {stem} Answer: A")})

    def excluded_domain_result(s, a):
        for r in a["tools"][:8]:
            r["response"]["output"][1]["output"] = json.dumps({"search_results": SEARCH_OK["search_results"].replace("https://a.org", "https://cdn.huggingface.co/x")})

    def odd_tool_name(s, a):  # Gym 은 `/{이름}` 으로 보내 `search? …` 도 /search 핸들러가 받는다 — 서버 통계와 어긋난 것으로 세면 안 된다
        a["tools"][0]["response"]["output"][0]["name"] = "search? Actually, let me search"
        a["placebo"][0]["response"]["output"][0]["name"] = "search#x"

    def long_queries(s, a):  # 롤아웃의 절반이 400 자 넘는 질의를 먼저 보낸다 — 모델 탓이지 검색 경로 장애가 아니다
        for r in a["tools"][::2]:
            r["response"]["output"][:0] = _call("c0", "search", {"query": "x" * 500}, {"search_results": "Query is too long"})
            r["num_search_calls"] += 1

    def low_use_no_difference(s, a):  # tools 의 8/48 롤아웃만 검색하고 나머지는 도구 없이 답한다. 세 갈래 보상은 같다
        for arm in ARMS:
            for r in a[arm]:
                set_reward(r, r["_ng_task_index"] % 2 == 0)
        for r in a["tools"][8:]:
            r["response"]["output"] = r["response"]["output"][-1:]
            r.update(num_search_calls=0, num_browse_calls=0, num_search_credits=0)

    def tools_all_wrong(s, a):
        for r in a["tools"]:
            set_reward(r, False)

    def mixed_infra(s, a):  # 장애 1 행 + usage 누락 2 행 = 3/48 — 범주마다는 5% 아래지만 합치면 넘는다
        backend_failed_rows(1)(s, a)
        for r in a["tools"][5:7]:
            r["num_search_usage_missing"] = 1

    def set_reward(r, correct):
        letter = r["expected_answer"] if correct else "ABCD"[("ABCD".index(r["expected_answer"]) + 1) % 4]
        r["response"]["output"][-1] = _message(letter)
        r.update(reward=1.0 if correct else 0.0, parsed_option=letter)

    def placebo_damaged(s, a):  # tools 와 notools 는 같은 실력, placebo 만 빈 결과에 포기 (상자 없는 답) — 검색 내용의 몫이 아니다
        for r in a["tools"] + a["notools"]:
            set_reward(r, r["_ng_task_index"] % 2 == 0)
        for r in a["placebo"]:
            set_reward(r, False)
            r["response"]["output"][-1] = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "I could not find it."}]}
            r.update(answer_found=False, parsed_option="")

    def leak_only(s, a):  # 검색이 도운 것은 질문 원문 (과 정답) 이 실린 페이지가 걸린 문항뿐
        for r in a["placebo"] + a["notools"]:
            set_reward(r, False)
        for r in a["tools"]:
            leaked = r["_ng_task_index"] < 6
            set_reward(r, leaked)
            if leaked:
                stem = r["responses_create_params"]["input"].split("\n")[0]
                r["response"]["output"][1]["output"] = json.dumps({"search_results": f"Search Results\n==============\n[1] T (a.org)\n    URL: https://a.org\n    Summary: {stem} Answer: A\n\n"})

    def search_mostly_rejected(s, a):
        for r in a["tools"][1:]:
            r["response"]["output"][1]["output"] = json.dumps({"search_results": "Error: the search engine rejected this query"})
            r["num_search_credits"] = 0

    def browse_dead(s, a):
        for r in a["tools"]:
            r["response"]["output"][3]["output"] = json.dumps({"page_content": "No HTML found"})
            r["num_browse_errors"] = 1

    def backend_failed_rows(k):
        def mutate(s, a):
            for r in a["tools"][:k]:
                r.update(num_search_errors=1, instance_config={"mask_sample": True}, reward=0.0)
                r["response"]["output"][-1] = _message("Z")
        return mutate

    def overflow_and_cap(s, a):
        empty = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": ""}]}
        a["tools"][0]["response"].update(incomplete_details={"reason": "max_output_tokens"})
        a["tools"][0]["response"]["output"][-1] = empty                 # Gym 의 문맥 초과 = 도구 결과 뒤에 빈 메시지
        a["tools"][0]["reward"] = 0.0
        a["tools"][1]["response"].update(incomplete_details={"reason": "max_output_tokens"})
        a["tools"][1]["response"]["output"][-1:] = [{"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking…"}]}, empty]
        a["tools"][1]["reward"] = 0.0                                   # 생성 상한 = 사고를 내다가 잘림

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
        "판정: placebo 만 망가짐 (tools = notools)": (judged(placebo_damaged), lambda bad, m: not bad and m["verdict"].startswith("판정 보류")
                                             and sign(m["primary_tools_minus_placebo"]) == 1),
        "판정: 누출된 문항에서만 효과": (judged(leak_only), lambda bad, m: not bad and m["verdict"].startswith("판정 보류 (누출")
                               and sign(m["primary_tools_minus_placebo"]) == 1),
        "판정: 문항 안에서 모르는 롤아웃만 누출": (judged(leak_within_questions), lambda bad, m: not bad and m["verdict"].startswith("판정 보류 (누출")
                                    and sign(m["leak"]["primary_without_leaked"]) is not None),
        "판정: 제외 도메인 결과도 누출로 센다": (judged(excluded_domain_result), lambda bad, m: abs(m["leak"]["results_from_excluded_domains"] - 8 / 48) < 1e-9
                                   and m["verdict"].startswith("판정 보류 (누출")),
        "판정: 도구를 거의 안 썼고 차이 없음": (judged(low_use_no_difference), lambda bad, m: not bad and m["verdict"].startswith("판정 불가 (도구 사용 부족")),
        "판정: 검색 갈래가 뚜렷이 낮음": (judged(tools_all_wrong), lambda bad, m: not bad and m["verdict"] == "검색이 깎는다"),
        "판정: 장애 롤아웃을 0 점으로 넣은 구간": (judged(backend_failed_rows(2)), lambda bad, m: not bad
                                    and m["primary_failed_as_zero"]["diff"] <= m["primary_tools_minus_placebo"]["diff"]),
        "유효성: 이상한 도구 이름도 닿은 핸들러로 센다": (judged(odd_tool_name), lambda bad, m: not bad),
        "유효성: 긴 질의 (모델 탓) 는 검색 장애가 아니다": (judged(long_queries), lambda bad, m: not bad and m["usage_tools"]["search_model_error_share"] > 0.3),
        "유효성: 인프라 탓 행은 합쳐서 센다": (judged(mixed_infra), lambda bad, m: {"V4", "V7"} <= bad and m["verdict"] is None),
        "판정: 다시 뽑힌 롤아웃이 많음": (judged(clear_win, redrawn={"tools": 6}), lambda bad, m: not bad and m["verdict"].startswith("판정 보류")),
        "유효성: 검색이 거의 다 거절": (judged(search_mostly_rejected), lambda bad, m: "V7" in bad),
        "유효성: browse 가 전부 실패": (judged(browse_dead), lambda bad, m: "V7" in bad),
        "유효성: 백엔드 장애 1 행 (2%) 은 빼고 진행": (judged(backend_failed_rows(1)), lambda bad, m: not bad and m["excluded_backend_failed"]["tools"] == 1),
        "유효성: 백엔드 장애 10 행 (21%) 은 FAIL": (judged(backend_failed_rows(10)), lambda bad, m: "V4" in bad),
        "끝 상태: 문맥 초과와 생성 상한을 가른다": (judged(overflow_and_cap), lambda bad, m: abs(m["answers_tools"]["end_states"]["context_overflow"] - 1 / 48) < 1e-9
                                    and abs(m["answers_tools"]["end_states"]["generation_cap"] - 1 / 48) < 1e-9),
        "잡음 추정이 파일 순서와 무관": (judged(shuffled), lambda bad, m: not bad and m["noise_aa_tools"] == base["noise_aa_tools"]
                                and m["primary_tools_minus_placebo"] == base["primary_tools_minus_placebo"]),
        "질문 원문 누출 표시": (judged(leak), lambda bad, m: m["leak"]["leaked_share"] == 1.0 and m["leak"]["question_text_in_results"] == 1.0),
        "깨끗한 결과에는 누출 0 · 끝 상태 전부 답함": ((bad, base), lambda bad, m: not bad and m["leak"]["leaked_share"] == 0
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
