"""alpha_web_search_mcqa 단위 테스트 (CPU · 네트워크 없음 — Tavily 와 페이지 가져오기는 가짜로 바꾼다).

실행 (운영 Ray 클러스터에 붙지 않게 격리):
  NEMO_GYM_EXTRA_ROOTS=<NeMo-RL>/examples/configs/alpha/gym_plugins RAY_ADDRESS=local <이 플러그인 venv>/bin/pytest --noconftest tests/test_app.py
"""

import asyncio
import socket
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import nemo_gym  # noqa: F401  — import 시 플러그인 루트와 Gym 루트를 sys.path 에 올린다
import pytest
import yaml
from fastapi.testclient import TestClient
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from pydantic import ValidationError
from resources_servers.alpha_web_search_mcqa import app as plugin
from resources_servers.alpha_web_search_mcqa.app import (
    AlphaWebSearchMcqaResourcesServer,
    AlphaWebSearchMcqaResourcesServerConfig,
    blocked_url_reason,
    fetch_page_text,
    final_answer_text,
    format_search_results,
    load_tavily_keys,
)
from resources_servers.google_search.app import (
    BaseGetPageContentRequest,
    BaseSearchQueryRequest,
    GoogleSearchResourcesServer,
    GoogleSearchResourcesServerConfig,
    GoogleSearchVerifyRequest,
    _extract_last_assistant_text,
    box_parser,
)
from tavily.errors import BadRequestError, ForbiddenError, InvalidAPIKeyError, UsageLimitExceededError


SECRET = "tvly-UNIT-TEST-SECRET"  # pragma: allowlist secret
HIT = {"title": "Chronic allograft vasculopathy", "url": "https://en.wikipedia.org/wiki/CAV", "content": "CAV is a ..."}


PRIVATE_NAMES = {"intranet.example.com": ["10.0.0.5"], "dual.example.com": ["93.184.216.34", "192.168.1.7"]}


@pytest.fixture(autouse=True)
def no_dns(monkeypatch):
    """이름 풀이를 가짜로 바꾼다 (네트워크 없음). 숫자 표기는 실제 getaddrinfo 가 푼다 — DNS 를 타지 않는다."""

    def fake(host):
        if host in PRIVATE_NAMES:
            return PRIVATE_NAMES[host]
        try:
            socket.inet_aton(host)
        except OSError:
            if ":" not in host:
                return ["93.184.216.34"]
        return sorted({info[4][0] for info in socket.getaddrinfo(host, None)})

    monkeypatch.setattr(plugin, "resolve_addresses", fake)


class FakeTavily:
    """programmed: 호출마다 차례로 쓰는 결과 (dict 는 반환, 예외는 raise). 다 쓰면 마지막 것을 되풀이한다."""

    def __init__(self, *programmed):
        self.programmed = list(programmed) or [{"results": [HIT], "usage": {"credits": 1}}]
        self.calls = []

    async def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        outcome = self.programmed[min(len(self.calls), len(self.programmed)) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def make_server(*fakes, **cfg):
    cfg = {"tavily_api_key": SECRET, "search_retry_backoff_s": 0.0} | cfg
    config = AlphaWebSearchMcqaResourcesServerConfig(host="127.0.0.1", port=1, entrypoint="app.py", name="alpha_web_search_mcqa", **cfg)
    server = AlphaWebSearchMcqaResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    if fakes:
        server._tavily_clients = list(fakes)
    return server


def message(text):
    return {"id": "msg", "role": "assistant", "status": "completed", "type": "message",
            "content": [{"annotations": [], "text": text, "type": "output_text"}]}


CALL = {"id": "fc", "type": "function_call", "call_id": "c1", "name": "search", "arguments": '{"query": "x"}', "status": "completed"}
CALL_OUTPUT = {"type": "function_call_output", "call_id": "c1", "output": '{"search_results": "..."}'}


def verify_body(*output, expected="C"):
    resp = NeMoGymResponse(id="r", created_at=0.0, model="m", object="response", parallel_tool_calls=False, tool_choice="none",
                           tools=[], output=list(output))
    return {"responses_create_params": {"input": "Which option?"}, "response": resp.model_dump(mode="json"),
            "expected_answer": expected, "task_difficulty_qwen3_32b_avg_8": 0.5}


def session(sid="s1"):
    return SimpleNamespace(session={SESSION_ID_KEY: sid})


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- 순수 함수


def test_format_search_results_follows_tavily_search_layout_and_drops_answer():
    text = format_search_results({"answer": "The answer is C.", "results": [HIT, {"title": "t2", "url": "https://a.org/x", "content": "y"}]}, 2000)
    assert text.startswith("Search Results\n==============\n")
    assert "[1] Chronic allograft vasculopathy (en.wikipedia.org)\n    URL: https://en.wikipedia.org/wiki/CAV\n    Summary: CAV is a ..." in text
    assert "[2] t2 (a.org)" in text
    assert "The answer is C." not in text


def test_format_search_results_truncates_and_handles_empty():
    long = {"results": [{"title": "t", "url": "https://a.org", "content": "line one\n" + "x" * 5000}]}
    assert "Summary: line one\n\n" in format_search_results(long, 100)  # 줄 경계에서 자른다
    assert len(format_search_results({"results": [{"title": "t", "url": "https://a.org", "content": "x" * 5000}]}, 100)) < 250
    assert "No results found." in format_search_results({"results": []}, 100)
    assert "No results found." in format_search_results({}, 100)


def test_load_tavily_keys_sources_and_precedence(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=hf_x\nGEMINI_API_KEY=g\nthis line is broken = = \nexport TAVILY_API_KEY='tvly-a, tvly-b'\n")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-env")
    assert load_tavily_keys("tvly-x", str(env_file)) == ["tvly-x"]
    assert load_tavily_keys(["tvly-x", " tvly-y "], None) == ["tvly-x", "tvly-y"]
    assert load_tavily_keys(None, str(env_file)) == ["tvly-a", "tvly-b"]
    assert load_tavily_keys(None, None) == ["tvly-env"]
    (tmp_path / "empty.env").write_text("HF_TOKEN=hf_x\n")
    with pytest.raises(ValueError):  # 파일을 줬는데 키 줄이 없으면 환경변수로 넘어가지 않는다
        load_tavily_keys(None, str(tmp_path / "empty.env"))
    monkeypatch.delenv("TAVILY_API_KEY")
    with pytest.raises(ValueError):
        load_tavily_keys(None, None)


@pytest.mark.parametrize(
    ("url", "blocked"),
    [
        ("https://en.wikipedia.org/wiki/X", False),
        ("http://pubmed.ncbi.nlm.nih.gov/123", False),
        ("https://8.8.8.8/x", False),
        ("file:///etc/passwd", True),
        ("ftp://example.com/x", True),
        ("not a url", True),
        ("http://localhost:8000/v1/models", True),
        ("http://127.0.0.1:8265/", True),
        ("http://10.0.0.5/", True),
        ("http://192.168.1.105:9000/", True),
        ("http://169.254.169.254/latest/meta-data", True),
        ("http://[::1]:8000/", True),
        ("http://sub1:8000/", True),          # 점 없는 내부 호스트 이름
        ("http://ray-head.internal/", True),
        ("http://[bad", True),
        ("http://127.1/", True),                 # 숫자 표기 — libc 는 127.0.0.1 로 푼다
        ("http://0x7f.0.0.1/", True),
        ("http://2130706433/", True),
        ("http://intranet.example.com/", True),  # 이름이 사설 주소로 풀린다
        ("http://dual.example.com/", True),      # 주소 하나라도 내부면 막는다
    ],
)
def test_blocked_url_reason(url, blocked):
    assert (blocked_url_reason(url) is not None) is blocked


def test_excluded_domains_apply_to_browse_urls():
    excluded = ("huggingface.co", "hf.co")
    assert blocked_url_reason("https://huggingface.co/datasets/nvidia/x", excluded) == "URL is in excluded domains"
    assert blocked_url_reason("https://cdn-lfs.hf.co/x", excluded) == "URL is in excluded domains"
    assert blocked_url_reason("https://huggingface.co./datasets/x", excluded) == "URL is in excluded domains"  # 끝의 점
    assert blocked_url_reason("https://nothuggingface.co/x", excluded) is None
    assert blocked_url_reason("https://en.wikipedia.org/wiki/X", excluded) is None


def test_fetch_page_text_mirrors_upstream_truncation(monkeypatch):
    monkeypatch.setattr(plugin.trafilatura, "fetch_url", lambda url: "<html>")
    monkeypatch.setattr(plugin.trafilatura, "extract", lambda html: "word " * 10001)
    ok, text = fetch_page_text("https://a.org")
    assert ok and len(text) == 5000 + 3 + 5000 and "..." in text
    monkeypatch.setattr(plugin.trafilatura, "extract", lambda html: "short page")
    assert fetch_page_text("https://a.org") == (True, "short page")
    monkeypatch.setattr(plugin.trafilatura, "extract", lambda html: None)  # 원본은 여기서 AttributeError
    assert fetch_page_text("https://a.org") == (False, "No text found")
    monkeypatch.setattr(plugin.trafilatura, "fetch_url", lambda url: None)
    assert fetch_page_text("https://a.org") == (False, "No HTML found")


# ---------------------------------------------------------------- 채점


@pytest.mark.parametrize(
    ("output", "reward", "found", "parsed"),
    [
        ([message(r"The answer is \boxed{C}")], 1.0, True, "C"),
        ([CALL, CALL_OUTPUT, message(r"After searching: \boxed{C}")], 1.0, True, "C"),
        ([message(r"\boxed{B}")], 0.0, True, "B"),
        ([message("I think it is C.")], 0.0, False, ""),                         # 원본은 500
        ([message(r"<think>maybe \boxed{A}?</think>Final: \boxed{C}")], 1.0, True, "C"),  # 사고 속 답은 보지 않는다
        ([message(r"\boxed{A} ... no, \boxed{C}")], 0.0, True, "A"),             # 원본과 같이 첫 것
        ([message(r"\boxed{c}")], 0.0, True, "c"),                               # 원본과 같이 완전 일치
        ([message(r"\boxed{C}"), CALL], 0.0, False, ""),                         # 마지막 항목이 메시지가 아님 (스텝 상한)
        ([CALL, CALL_OUTPUT], 0.0, False, ""),
    ],
)
def test_verify_reward(output, reward, found, parsed):
    vr = run(make_server().verify(session(), GoogleSearchVerifyRequest.model_validate(verify_body(*output))))
    assert (vr.reward, vr.answer_found, vr.parsed_option) == (reward, found, parsed)
    assert vr.instance_config is None and vr.failure_reason is None
    assert (vr.num_search_calls, vr.num_search_errors, vr.num_browse_calls, vr.num_browse_errors) == (0, 0, 0, 0)


def upstream_server():
    config = GoogleSearchResourcesServerConfig(host="127.0.0.1", port=1, entrypoint="app.py", name="google_search",
                                               google_api_key="k", google_cx="cx")  # pragma: allowlist secret
    return GoogleSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def test_upstream_verify_raises_without_boxed_answer_and_plugin_does_not():
    """이 플러그인이 verify 를 덮어쓰는 이유. upstream 이 고치면 이 테스트가 알려 준다."""
    body = GoogleSearchVerifyRequest.model_validate(verify_body(message("I think it is C.")))
    with pytest.raises(ValidationError):
        run(upstream_server().verify(body))
    assert run(make_server().verify(session(), body)).reward == 0.0


@pytest.mark.parametrize("text", [r"The answer is \boxed{C}", r"\boxed{B} and \boxed{C}", r"x \boxed{D} y", r"\boxed{C}" + "\nnotes"])
def test_parse_matches_upstream_where_upstream_works(text):
    body = GoogleSearchVerifyRequest.model_validate(verify_body(message(text)))
    assert box_parser(final_answer_text(body)) == box_parser(_extract_last_assistant_text(body))


# ---------------------------------------------------------------- 검색


def test_search_passes_provided_parameters_and_never_asks_for_an_answer():
    fake = FakeTavily()
    out = run(make_server(fake, exclude_domains=[]).search(session(), BaseSearchQueryRequest(query="cav predictor")))
    query, kwargs = fake.calls[0]
    assert query == "cav predictor"
    assert kwargs["search_depth"] == "basic" and kwargs["max_results"] == 10
    assert kwargs["include_answer"] is False and kwargs["exclude_domains"] is None and kwargs["include_usage"] is True
    assert out.search_results.startswith("Search Results")
    fake2 = FakeTavily()
    run(make_server(fake2, search_depth="advanced", exclude_domains=["x.com"]).search(session(), BaseSearchQueryRequest(query="q")))
    assert fake2.calls[0][1]["search_depth"] == "advanced" and fake2.calls[0][1]["exclude_domains"] == ["x.com"]


@pytest.mark.parametrize("query", ["", "   ", "x" * 401])
def test_search_guards_do_not_call_backend(query):
    fake = FakeTavily()
    server = make_server(fake)
    out = run(server.search(session(), BaseSearchQueryRequest(query=query)))
    assert out.search_results in ("Query is empty", "Query is too long") and fake.calls == []
    vr = run(server.verify(session(), GoogleSearchVerifyRequest.model_validate(verify_body(message(r"\boxed{C}")))))
    assert (vr.num_search_calls, vr.num_search_errors, vr.instance_config) == (1, 0, None)


@pytest.mark.parametrize(("error", "kind"), [(ForbiddenError("plan limit " + SECRET), "usage_limit"), (InvalidAPIKeyError("bad " + SECRET), "invalid_api_key")])
def test_backend_failure_is_not_retried_is_counted_and_masks_the_sample(error, kind, capsys):
    fake = FakeTavily(error)
    server = make_server(fake)
    out = run(server.search(session(), BaseSearchQueryRequest(query="q")))
    assert out.search_results == f"Error: search is unavailable ({kind})"
    assert len(fake.calls) == 1
    vr = run(server.verify(session(), GoogleSearchVerifyRequest.model_validate(verify_body(message(r"\boxed{C}")))))
    assert (vr.num_search_calls, vr.num_search_errors) == (1, 1)
    assert vr.instance_config == {"mask_sample": True} and vr.failure_reason == "search_backend_error"
    assert vr.reward == 1.0  # 보상은 그대로 계산한다 — 손실에서만 빠진다
    assert SECRET not in out.search_results and SECRET not in capsys.readouterr().out


def test_mask_can_be_turned_off():
    server = make_server(FakeTavily(ForbiddenError("x")), mask_on_search_error=False)
    run(server.search(session(), BaseSearchQueryRequest(query="q")))
    vr = run(server.verify(session(), GoogleSearchVerifyRequest.model_validate(verify_body(message(r"\boxed{C}")))))
    assert vr.num_search_errors == 1 and vr.instance_config is None and vr.failure_reason == "search_backend_error"


def test_transient_failures_are_retried():
    ok = {"results": [HIT], "usage": {"credits": 1}}
    fake = FakeTavily(UsageLimitExceededError("429"), TimeoutError("t"), ok)
    server = make_server(fake)
    assert run(server.search(session(), BaseSearchQueryRequest(query="q"))).search_results.startswith("Search Results")
    assert len(fake.calls) == 3
    always = FakeTavily(RuntimeError("503 " + SECRET))
    assert AlphaWebSearchMcqaResourcesServerConfig.model_fields["search_max_tries"].default == 6
    server2 = make_server(always, search_max_tries=3)
    assert run(server2.search(session(), BaseSearchQueryRequest(query="q"))).search_results == "Error: search is unavailable (unavailable)"
    assert len(always.calls) == 3
    vr = run(server2.verify(session(), GoogleSearchVerifyRequest.model_validate(verify_body(message("x")))))
    assert vr.num_search_errors == 1 and vr.instance_config == {"mask_sample": True}


def test_rate_limit_429_waits_and_recovers_then_gives_up_after_the_deadline():
    ok = {"results": [HIT], "usage": {"credits": 1}}
    recovering = FakeTavily(*[UsageLimitExceededError("429")] * 6, ok)  # search_max_tries (3) 보다 많이 와도 기다린다
    server = make_server(recovering)
    assert run(server.search(session(), BaseSearchQueryRequest(query="q"))).search_results.startswith("Search Results")
    vr = run(server.verify(session(), GoogleSearchVerifyRequest.model_validate(verify_body(message("x")))))
    assert (len(recovering.calls), vr.num_search_retries, vr.num_search_errors, vr.instance_config) == (7, 6, 0, None)

    stuck = FakeTavily(UsageLimitExceededError("429 " + SECRET))
    server2 = make_server(stuck, search_rate_limit_max_wait_s=0.05)
    out = run(server2.search(session(), BaseSearchQueryRequest(query="q")))
    assert out.search_results == "Error: search is unavailable (rate_limited)"
    vr2 = run(server2.verify(session(), GoogleSearchVerifyRequest.model_validate(verify_body(message("x")))))
    assert vr2.num_search_errors == 1 and vr2.num_search_retries >= 1 and vr2.instance_config == {"mask_sample": True}


def test_search_rate_limiter_spaces_calls_across_sessions():
    class Timed(FakeTavily):
        async def search(self, query, **kwargs):
            self.times = getattr(self, "times", []) + [time.perf_counter()]
            return await super().search(query, **kwargs)

    async def burst(server, n):
        await asyncio.gather(*(server.search(session(f"s{i}"), BaseSearchQueryRequest(query="q")) for i in range(n)))

    limited = Timed()
    run(burst(make_server(limited, search_max_per_minute=600), 5))  # 0.1 초 간격
    gaps = [b - a for a, b in zip(limited.times, limited.times[1:])]
    assert len(limited.times) == 5 and min(gaps) > 0.07 and limited.times[-1] - limited.times[0] > 0.35
    free = Timed()
    run(burst(make_server(free), 5))
    assert free.times[-1] - free.times[0] < 0.05


def test_placebo_backend_returns_nothing_and_calls_nothing(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(plugin, "fetch_page_text", lambda url: pytest.fail("placebo 가 페이지를 가져왔다"))
    config = AlphaWebSearchMcqaResourcesServerConfig(host="127.0.0.1", port=1, entrypoint="app.py", name="alpha_web_search_mcqa",
                                                     search_backend="none")  # 키 없이 뜬다
    server = AlphaWebSearchMcqaResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    assert server._tavily_clients == []
    out = run(server.search(session(), BaseSearchQueryRequest(query="cav predictor")))
    assert out.search_results == "Search Results\n==============\nNo results found.\n"
    assert run(server.browse(session(), BaseGetPageContentRequest(url="https://en.wikipedia.org/wiki/CAV"))).page_content == "No HTML found"
    vr = run(server.verify(session(), GoogleSearchVerifyRequest.model_validate(verify_body(message(r"\boxed{C}")))))
    assert (vr.reward, vr.num_search_calls, vr.num_search_errors, vr.num_search_credits, vr.num_browse_calls, vr.instance_config) == (1.0, 1, 0, 0, 1, None)


def test_credit_budget_stops_searches_before_overspending_even_when_concurrent():
    fake = FakeTavily({"results": [HIT], "usage": {"credits": 1}})
    server = make_server(fake, search_credit_budget=3)

    async def burst():
        return await asyncio.gather(*(server.search(session(f"s{i}"), BaseSearchQueryRequest(query="q")) for i in range(5)))

    outs = [o.search_results for o in run(burst())]
    assert len(fake.calls) == 3 and server._credits_committed == 3
    assert sum(o.startswith("Search Results") for o in outs) == 3
    assert sum(o == "Error: search is unavailable (budget_exhausted)" for o in outs) == 2
    masked = [run(server.verify(session(f"s{i}"), GoogleSearchVerifyRequest.model_validate(verify_body(message("x"))))) for i in range(5)]
    assert sum(v.instance_config == {"mask_sample": True} for v in masked) == 2 and sum(v.num_search_credits for v in masked) == 3

    advanced = FakeTavily({"results": [HIT], "usage": {"credits": 2}})
    server2 = make_server(advanced, search_depth="advanced", search_credit_budget=3)
    run(server2.search(session(), BaseSearchQueryRequest(query="q")))
    assert run(server2.search(session(), BaseSearchQueryRequest(query="q"))).search_results.endswith("(budget_exhausted)") and len(advanced.calls) == 1


def test_budget_ledger_releases_uncharged_rejections_and_keeps_unknown_failures():
    ok = {"results": [HIT], "usage": {"credits": 1}}
    for rejection in (ForbiddenError("limit"), BadRequestError("bad"), InvalidAPIKeyError("key")):  # 과금되지 않는 거절
        fake = FakeTavily(rejection, ok)
        server = make_server(fake, search_credit_budget=1)
        run(server.search(session("a"), BaseSearchQueryRequest(query="q")))
        assert server._credits_committed == 0
        assert run(server.search(session("b"), BaseSearchQueryRequest(query="q"))).search_results.startswith("Search Results")
    rate = make_server(FakeTavily(UsageLimitExceededError("429"), UsageLimitExceededError("429"), ok), search_credit_budget=1)
    assert run(rate.search(session(), BaseSearchQueryRequest(query="q"))).search_results.startswith("Search Results") and rate._credits_committed == 1

    # 시간 초과 · 5xx 는 과금됐을 수 있다 — 장부에서 빼지 않는다. 그래서 재시도가 상한을 넘겨 쓰지 못한다.
    flaky = FakeTavily(TimeoutError("t"), RuntimeError("503"), ok)
    server = make_server(flaky, search_credit_budget=10)
    assert run(server.search(session(), BaseSearchQueryRequest(query="q"))).search_results.startswith("Search Results")
    vr = run(server.verify(session(), GoogleSearchVerifyRequest.model_validate(verify_body(message("x")))))
    assert (server._credits_committed, vr.num_search_credits, vr.num_search_retries) == (3, 1, 2)
    tight = FakeTavily(TimeoutError("t"), ok)
    server2 = make_server(tight, search_credit_budget=1)
    assert run(server2.search(session(), BaseSearchQueryRequest(query="q"))).search_results.endswith("(budget_exhausted)") and len(tight.calls) == 1


def test_missing_usage_is_counted_at_list_price_not_zero():
    server = make_server(FakeTavily({"results": [HIT]}), search_depth="advanced")
    run(server.search(session(), BaseSearchQueryRequest(query="q")))
    vr = run(server.verify(session(), GoogleSearchVerifyRequest.model_validate(verify_body(message("x")))))
    assert (vr.num_search_credits, vr.num_search_usage_missing, vr.instance_config) == (2, 1, None)


def test_rejected_query_is_model_visible_not_a_backend_failure():
    fake = FakeTavily(BadRequestError("invalid query"))
    server = make_server(fake)
    out = run(server.search(session(), BaseSearchQueryRequest(query="q")))
    assert out.search_results == "Error: the search engine rejected this query" and len(fake.calls) == 1
    vr = run(server.verify(session(), GoogleSearchVerifyRequest.model_validate(verify_body(message("x")))))
    assert vr.num_search_errors == 0 and vr.instance_config is None


def test_keys_are_used_in_rotation():
    a, b = FakeTavily(), FakeTavily()
    server = make_server(a, b)
    for _ in range(4):
        run(server.search(session(), BaseSearchQueryRequest(query="q")))
    assert (len(a.calls), len(b.calls)) == (2, 2)
    assert len(make_server(tavily_api_key=["tvly-1", "tvly-2", "tvly-3"])._tavily_clients) == 3


def test_more_than_one_worker_is_refused():
    with pytest.raises(ValueError, match="num_workers"):
        make_server(num_workers=2)


# ---------------------------------------------------------------- browse


def test_browse_blocks_internal_urls_without_fetching(monkeypatch):
    monkeypatch.setattr(plugin, "fetch_page_text", lambda url: pytest.fail("fetched a blocked URL"))
    server = make_server(exclude_domains=["huggingface.co"])
    out = run(server.browse(session(), BaseGetPageContentRequest(url="http://127.0.0.1:8000/v1/models")))
    assert out.page_content == "Error: internal addresses cannot be browsed"
    assert run(server.browse(session("x"), BaseGetPageContentRequest(url="http://intranet.example.com/"))).page_content.startswith("Error: internal")
    assert run(server.browse(session("x"), BaseGetPageContentRequest(url="https://huggingface.co/datasets/x"))).page_content == "Error: URL is in excluded domains"
    vr = run(server.verify(session(), GoogleSearchVerifyRequest.model_validate(verify_body(message("x")))))
    assert (vr.num_browse_calls, vr.num_browse_errors, vr.instance_config) == (1, 1, None)  # browse 실패는 마스크하지 않는다


def test_browse_failures_are_counted_and_do_not_leak_details(monkeypatch):
    server = make_server()
    monkeypatch.setattr(plugin, "fetch_page_text", lambda url: (False, "No HTML found"))
    assert run(server.browse(session(), BaseGetPageContentRequest(url="https://a.org"))).page_content == "No HTML found"

    def boom(url):
        raise RuntimeError("connection to proxy user:pw@host failed")

    monkeypatch.setattr(plugin, "fetch_page_text", boom)
    assert run(server.browse(session(), BaseGetPageContentRequest(url="https://a.org"))).page_content == "Error: Unexpected error = RuntimeError"
    monkeypatch.setattr(plugin, "fetch_page_text", lambda url: (True, "page text"))
    assert run(server.browse(session(), BaseGetPageContentRequest(url="https://a.org"))).page_content == "page text"
    vr = run(server.verify(session(), GoogleSearchVerifyRequest.model_validate(verify_body(message("x")))))
    assert (vr.num_browse_calls, vr.num_browse_errors) == (3, 2)


def test_slow_browse_does_not_block_other_requests(monkeypatch):
    """원본 browse 는 async 핸들러 안의 동기 호출이라 이 테스트의 검색이 페이지를 다 받을 때까지 기다린다."""

    def slow(url):
        time.sleep(0.5)
        return True, "page"

    monkeypatch.setattr(plugin, "fetch_page_text", slow)
    server = make_server(FakeTavily())

    async def scenario():
        started = time.perf_counter()
        browse = asyncio.create_task(server.browse(session("a"), BaseGetPageContentRequest(url="https://a.org")))
        await asyncio.sleep(0.05)  # browse 가 스레드에 들어간 뒤
        await server.search(session("b"), BaseSearchQueryRequest(query="q"))
        search_done = time.perf_counter() - started
        await browse
        return search_done, time.perf_counter() - started

    search_done, total = run(scenario())
    assert search_done < 0.3 and total >= 0.5


# ---------------------------------------------------------------- HTTP 경로 (세션 쿠키 · 라우트 등록 · judge_failsafe 래핑)


def test_http_flow_keeps_per_session_counts_and_clears_them(monkeypatch):
    monkeypatch.setattr(plugin, "fetch_page_text", lambda url: (True, "page text about CAV"))
    fake = FakeTavily({"results": [HIT], "usage": {"credits": 1}})
    server = make_server(fake)
    web = server.setup_webserver()
    with TestClient(web) as one, TestClient(web) as two:
        assert one.post("/seed_session", json={}).status_code == 200
        assert two.post("/seed_session", json={}).status_code == 200
        first = one.post("/search", json={"query": "cav predictor"})
        assert first.status_code == 200 and first.json()["search_results"].startswith("Search Results")
        assert one.post("/search", json={"query": "cav dsa"}).status_code == 200
        assert one.post("/browse", json={"url": "https://en.wikipedia.org/wiki/CAV"}).json() == {"page_content": "page text about CAV"}
        assert two.post("/search", json={"query": "other rollout"}).status_code == 200
        assert one.post("/search", json={}).status_code == 422  # 인자 없는 도구 호출 — 모델에게 보이는 오류, 핸들러에 닿지 않아 세지 않는다

        r1 = one.post("/verify", json=verify_body(message(r"\boxed{C}")))
        assert r1.status_code == 200, r1.text
        v1 = r1.json()
        assert (v1["reward"], v1["answer_found"], v1["parsed_option"]) == (1.0, True, "C")
        assert (v1["num_search_calls"], v1["num_search_errors"], v1["num_search_credits"]) == (2, 0, 2)
        assert (v1["num_browse_calls"], v1["num_browse_errors"]) == (1, 0)
        assert v1["instance_config"] is None and v1["expected_answer"] == "C"

        v2 = two.post("/verify", json=verify_body(message("no box"))).json()
        assert (v2["reward"], v2["answer_found"], v2["num_search_calls"], v2["num_search_credits"]) == (0.0, False, 1, 1)
    assert server._session_stats == {}  # verify 가 세션 통계를 지운다


def test_http_verify_carries_mask_flag_where_nemo_rl_reads_it():
    server = make_server(FakeTavily(ForbiddenError("limit")))
    with TestClient(server.setup_webserver()) as client:
        client.post("/seed_session", json={})
        assert client.post("/search", json={"query": "q"}).json() == {"search_results": "Error: search is unavailable (usage_limit)"}
        full_result = client.post("/verify", json=verify_body(message(r"\boxed{C}"))).json()
    # nemo_rl/experience/rollouts.py `_extract_mask_sample_flags` 와 같은 읽기
    assert bool((full_result.get("instance_config") or {}).get("mask_sample", False)) is True


# ---------------------------------------------------------------- 설정 파일


def test_config_yaml_is_consistent_with_the_server(tmp_path):
    cfg = yaml.safe_load((Path(plugin.__file__).parent / "configs" / "alpha_web_search_mcqa.yaml").read_text())
    server_block = cfg["alpha_web_search_mcqa_resources_server"]["resources_servers"]
    assert list(server_block) == [Path(plugin.__file__).parent.name]  # 서버 이름 = 플러그인 디렉토리
    agent = cfg["alpha_web_search_mcqa_simple_agent"]["responses_api_agents"]["simple_agent"]
    assert agent["resources_server"] == {"type": "resources_servers", "name": "alpha_web_search_mcqa_resources_server"}
    assert agent["max_steps"] >= 2  # 재호출 가드 (KNOWN_ISSUES 2026-11-14). 1 이면 도구를 쓸 수 없다

    fields = dict(server_block["alpha_web_search_mcqa"])
    assert "tavily_api_key" not in fields and fields["tavily_api_key_file"].endswith(".env")  # 키 값은 설정에 두지 않는다
    meta = {k: fields.pop(k) for k in ("domain", "verified", "description")}
    assert meta["verified"] is False
    unknown = set(fields) - set(AlphaWebSearchMcqaResourcesServerConfig.model_fields)
    assert not unknown, f"서버 설정에 없는 키: {unknown}"
    env_file = tmp_path / ".env"
    env_file.write_text("TAVILY_API_KEY=tvly-from-file\n")
    config = AlphaWebSearchMcqaResourcesServerConfig(host="127.0.0.1", port=1, name="alpha_web_search_mcqa",
                                                     **(fields | {"tavily_api_key_file": str(env_file)}))
    assert len(AlphaWebSearchMcqaResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))._tavily_clients) == 1
