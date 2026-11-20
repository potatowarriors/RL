r"""alpha Gym 플러그인 — resources_servers/alpha_web_search_mcqa. Gym 0.6.0 `google_search` 의 검색 백엔드만 Tavily 로 바꾼다 (RL_PLAN.md 결정 72 · 73).

Google Custom Search JSON API 는 신규 고객을 받지 않고 2027-01-01 에 종료된다 (기존 고객도 하루 10k 질의). 데이터
`nvidia/Nemotron-RL-knowledge-web_search-mcqa` 는 도구 이름 `search` · `browse` 와 `\boxed{글자}` 답을 전제로 하므로 도구 이름 · 요청 모델 ·
채점 규칙은 Gym `google_search` 것을 그대로 쓰고 `search` 내부만 바꾼다. 데이터 행은 `agent_ref` 만 붙이면 된다.

Gym `google_search` 와 다른 점:
1. `search` 는 Tavily 를 부른다. 결과 수 (10) 와 결과 글 형식은 Gym `tavily_search` 를 따른다. Tavily 가 지어 주는 요약 답 (`answer`) 은
   요청하지도 돌려주지도 않는다 — 객관식에서는 정답을 그대로 알려 줄 수 있다.
2. `browse` 는 같은 trafilatura 추출 · 같은 자르기지만 스레드 풀에서 돈다. 원본은 async 핸들러 안의 동기 호출이라 페이지 하나가 서버
   전체 (다른 롤아웃의 도구 호출 · 채점) 를 세운다. http(s) 가 아니거나 내부 주소인 URL 은 가져오지 않는다 — 롤아웃 노드에는 인증 없는
   내부 서비스가 떠 있다. 이름이 내부 주소로 풀리는 경우와 리다이렉트는 막지 못한다.
3. `verify` 는 `\boxed{…}` 가 없는 답을 보상 0 으로 돌린다. 원본은 `parsed_option=None` 을 str 필드에 넣어 500 을 내고, NeMo-RL 은 롤아웃
   예외를 그대로 올린다. 채점 규칙 (마지막 assistant 메시지의 첫 `\boxed{…}` 가 정답과 완전히 같으면 1) 은 원본과 같다.
4. 세션별 도구 호출 · 실패 수 · Tavily 크레딧을 verify 응답에 싣는다. 검색 백엔드 장애 (한도 · 인증 · 재시도 뒤에도 실패) 가 있던 롤아웃은
   `instance_config.mask_sample` 로 표시해 NeMo-RL 이 손실에서 뺀다 (`rollouts._extract_mask_sample_flags`, 지표
   `num_mask_sample_filtered`). 원본은 실패를 `Error: …` 도구 결과로만 돌려 할당량이 바닥나도 보상만 떨어진 채 런이 계속된다.

5. Tavily 의 분당 한도 (개발 키 100 · 운영 키 1,000 회) 를 서버가 지킨다. `search_max_per_minute` 를 주면 검색을 고르게 띄워 보내고,
   그래도 429 가 오면 `search_rate_limit_max_wait_s` 까지 기다렸다 다시 부른다. 롤아웃 수백 개가 한꺼번에 검색해도 장애로 세지 않는다.

세션 통계 · 속도 제한은 프로세스 메모리에 있다 — 이 서버는 워커 1개로만 띄운다 (`num_workers` > 1 이면 기동을 거부한다).
"""

import asyncio
import ipaddress
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import urlparse

import trafilatura
from fastapi import Request
from pydantic import Field, PrivateAttr
from tavily import AsyncTavilyClient
from tavily.errors import BadRequestError, ForbiddenError, InvalidAPIKeyError, UsageLimitExceededError

from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.google_search.app import (
    BaseGetPageContentRequest,
    BaseGetPageContentResponse,
    BaseGetSearchQueryResponse,
    BaseSearchQueryRequest,
    GoogleSearchResourcesServer,
    GoogleSearchResourcesServerConfig,
    GoogleSearchVerifyRequest,
    GoogleSearchVerifyResponse,
    box_parser,
)


TAVILY_KEY_NAME = "TAVILY_API_KEY"
MAX_QUERY_CHARS = 400  # Tavily 의 질의 길이 상한 (Gym tavily_search 와 같은 가드)
BROWSE_MAX_WORDS = 10000  # 아래 두 값은 Gym google_search.browse 의 자르기 그대로다
BROWSE_KEEP_CHARS = 5000


class AlphaWebSearchMcqaResourcesServerConfig(GoogleSearchResourcesServerConfig):
    google_api_key: str = ""  # 부모의 필수 필드 — 이 서버는 쓰지 않는다
    google_cx: str = ""
    # 키는 하나만 준다: 값 (목록이면 돌려 쓴다) · 파일 (`TAVILY_API_KEY=…` 줄이 있는 env 형식) · 둘 다 없으면 환경변수.
    # 레시피에는 파일 경로를 쓴다 — 값을 쓰면 설정 덤프와 W&B 에 남는다.
    tavily_api_key: Optional[str | list[str]] = None
    tavily_api_key_file: Optional[str] = None
    search_depth: Literal["basic", "advanced"] = "basic"  # 검색당 1 · 2 크레딧. Gym tavily_search 는 advanced
    max_results: int = 10
    max_result_chars: int = 2000
    exclude_domains: list[str] = Field(default_factory=list)
    search_max_tries: int = 3  # 5xx · 시간 초과 · 연결 오류에 대한 시도 수
    search_retry_backoff_s: float = 0.5
    search_timeout_s: float = 30.0
    search_max_per_minute: Optional[float] = None  # 서버 전체의 검색 호출 속도 상한 (없으면 제한하지 않는다)
    search_rate_limit_max_wait_s: float = 120.0  # 429 를 받은 검색 하나가 다시 부르며 기다리는 최대 시간
    browse_max_concurrency: int = 32
    mask_on_search_error: bool = True


class AlphaWebSearchMcqaVerifyResponse(GoogleSearchVerifyResponse):
    # 원본 응답에는 요청의 이 두 필드가 없다 — 결과 행만으로 재채점 · 난이도별 집계를 하려고 싣는다.
    expected_answer: str
    task_difficulty_qwen3_32b_avg_8: float
    answer_found: bool
    num_search_calls: int
    num_search_errors: int
    num_search_retries: int
    num_search_credits: int
    num_browse_calls: int
    num_browse_errors: int
    instance_config: Optional[dict[str, Any]] = None


class RateLimiter:
    """호출을 60 / per_minute 초 간격으로 고르게 띄운다 (몰아서 보내면 분당 한도 안이어도 429 가 온다)."""

    def __init__(self, per_minute: float):
        self._interval = 60.0 / per_minute
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        loop = asyncio.get_running_loop()
        async with self._lock:
            slot = max(self._next, loop.time())
            self._next = slot + self._interval
        await asyncio.sleep(max(0.0, slot - loop.time()))


class SearchBackendError(Exception):
    """정책 탓이 아닌 검색 실패. `kind` 만 모델에게 보인다 — 원문에는 요청 정보가 섞일 수 있다."""

    def __init__(self, kind: str, cause: BaseException):
        super().__init__(f"{kind}: {type(cause).__name__}")
        self.kind = kind


def load_tavily_keys(explicit: Optional[str | list[str]], key_file: Optional[str]) -> list[str]:
    if explicit:
        raw = [explicit] if isinstance(explicit, str) else list(explicit)
    elif key_file:
        raw = []
        for line in Path(key_file).read_text().splitlines():
            m = re.match(rf"\s*(?:export\s+)?{TAVILY_KEY_NAME}\s*=\s*(.*)$", line)
            if m:
                raw = m.group(1).strip().strip("'\"").split(",")
    else:
        raw = os.environ.get(TAVILY_KEY_NAME, "").split(",")
    keys = [k.strip() for k in raw if k and k.strip()]
    if not keys:
        raise ValueError(
            f"alpha_web_search_mcqa: Tavily 키가 없다 — tavily_api_key · tavily_api_key_file · 환경변수 {TAVILY_KEY_NAME} 중 하나를 준다"
        )
    return keys


def format_search_results(results: dict, max_result_chars: int) -> str:
    """Gym tavily_search 의 결과 글 형식. Tavily 의 `answer` 는 보지 않는다."""
    out = ["Search Results\n==============\n"]
    for i, item in enumerate(results.get("results") or [], 1):
        url = item.get("url") or ""
        snippet = re.sub(r"[​‌‍﻿]", "", item.get("content") or "")
        snippet = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+$", "", snippet, flags=re.MULTILINE)).strip()
        if len(snippet) > max_result_chars:
            cut = snippet.rfind("\n", 0, max_result_chars)
            snippet = snippet[: cut if cut != -1 else max_result_chars]
        out.append(f"[{i}] {item.get('title') or ''} ({urlparse(url).hostname or url})\n    URL: {url}\n    Summary: {snippet}\n\n")
    if len(out) == 1:
        out.append("No results found.\n")
    return "".join(out)


def blocked_url_reason(url: str) -> Optional[str]:
    """가져오면 안 되는 URL 이면 모델에게 보일 사유를 돌려준다."""
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
    except ValueError:
        return "invalid URL"
    if parsed.scheme not in ("http", "https") or not host:
        return "only http(s) URLs can be browsed"
    try:
        internal = not ipaddress.ip_address(host).is_global
    except ValueError:  # 이름: 점이 없거나 내부 전용 접미사면 내부 주소로 본다
        internal = "." not in host or host.endswith((".localhost", ".local", ".internal"))
    return "internal addresses cannot be browsed" if internal else None


def fetch_page_text(url: str) -> tuple[bool, str]:
    """Gym google_search.browse 의 추출 · 자르기 · 문구 그대로 (동기). (본문을 얻었는가, 모델에게 보일 글) 을 돌려준다."""
    html = trafilatura.fetch_url(url)
    if not html:
        return False, "No HTML found"
    text = trafilatura.extract(html)
    if not text:
        return False, "No text found"
    if len(text.split()) > BROWSE_MAX_WORDS:
        text = text[:BROWSE_KEEP_CHARS] + "..." + text[-BROWSE_KEEP_CHARS:]
    return True, text


def final_answer_text(body: GoogleSearchVerifyRequest) -> Optional[str]:
    """마지막 출력 항목이 assistant 메시지면 그 글. 사고 태그가 섞였으면 마지막 `</think>` 뒤만 본다 (alpha_terminus_judge 와 같다)."""
    output = body.response.output
    last = output[-1] if output else None
    if getattr(last, "type", None) != "message" or getattr(last, "role", None) != "assistant":
        return None
    content = last.content
    if not isinstance(content, str):
        content = "".join(getattr(part, "text", None) or "" for part in content if getattr(part, "type", None) == "output_text")
    return content.split("</think>")[-1]


class AlphaWebSearchMcqaResourcesServer(GoogleSearchResourcesServer):
    config: AlphaWebSearchMcqaResourcesServerConfig

    _tavily_clients: list[AsyncTavilyClient] = PrivateAttr(default_factory=list)
    _num_search_requests: int = PrivateAttr(default=0)
    _session_stats: dict[str, Counter] = PrivateAttr(default_factory=dict)
    _browse_pool: Optional[ThreadPoolExecutor] = PrivateAttr(default=None)
    _limiter: Optional[RateLimiter] = PrivateAttr(default=None)

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if (self.config.num_workers or 1) > 1:
            raise ValueError("alpha_web_search_mcqa: 세션 통계가 프로세스 메모리에 있다 — num_workers 는 1 이어야 한다")
        keys = load_tavily_keys(self.config.tavily_api_key, self.config.tavily_api_key_file)
        self._tavily_clients = [AsyncTavilyClient(api_key=k) for k in keys]
        self._browse_pool = ThreadPoolExecutor(max_workers=self.config.browse_max_concurrency, thread_name_prefix="browse")
        if self.config.search_max_per_minute:
            self._limiter = RateLimiter(self.config.search_max_per_minute)
        print(
            f"alpha_web_search_mcqa: Tavily 키 {len(keys)} 개 · search_depth={self.config.search_depth} · max_results={self.config.max_results}"
            f" · search_max_per_minute={self.config.search_max_per_minute}"
        )

    def _stats(self, request: Request) -> Counter:
        return self._session_stats.setdefault(request.session[SESSION_ID_KEY], Counter())

    async def _tavily_search(self, query: str, stats: Counter) -> dict:
        loop = asyncio.get_running_loop()
        failures, rate_hits, rate_deadline = 0, 0, None
        while True:
            if self._limiter is not None:
                await self._limiter.wait()
            client = self._tavily_clients[self._num_search_requests % len(self._tavily_clients)]
            self._num_search_requests += 1
            try:
                return await client.search(
                    query,
                    search_depth=self.config.search_depth,
                    max_results=self.config.max_results,
                    exclude_domains=self.config.exclude_domains or None,
                    include_answer=False,
                    include_usage=True,
                    timeout=self.config.search_timeout_s,
                )
            except BadRequestError:
                raise  # 질의 탓 — 모델에게 보이고 장애로 세지 않는다
            except InvalidAPIKeyError as e:
                raise SearchBackendError("invalid_api_key", e) from e
            except ForbiddenError as e:  # 403 · 432 (요금제 한도) · 433 (종량제 한도) — 다시 불러도 같다
                raise SearchBackendError("usage_limit", e) from e
            except UsageLimitExceededError as e:  # 429 분당 한도 — 기다리면 풀린다
                rate_deadline = rate_deadline or loop.time() + self.config.search_rate_limit_max_wait_s
                if loop.time() >= rate_deadline:
                    raise SearchBackendError("rate_limited", e) from e
                delay = min(self.config.search_retry_backoff_s * 2 ** min(rate_hits, 6), 30.0)
                rate_hits += 1
            except Exception as e:  # 5xx · 시간 초과 · 연결 오류
                failures += 1
                if failures >= self.config.search_max_tries:
                    raise SearchBackendError("unavailable", e) from e
                delay = self.config.search_retry_backoff_s * 2 ** (failures - 1)
            stats["search_retries"] += 1
            await asyncio.sleep(delay)

    async def search(self, request: Request, body: BaseSearchQueryRequest) -> BaseGetSearchQueryResponse:
        stats = self._stats(request)
        stats["search_calls"] += 1
        if not body.query.strip():
            return BaseGetSearchQueryResponse(search_results="Query is empty")
        if len(body.query) > MAX_QUERY_CHARS:
            return BaseGetSearchQueryResponse(search_results="Query is too long")
        try:
            results = await self._tavily_search(body.query, stats)
        except BadRequestError:
            return BaseGetSearchQueryResponse(search_results="Error: the search engine rejected this query")
        except SearchBackendError as e:
            stats["search_errors"] += 1
            print(f"alpha_web_search_mcqa: 검색 백엔드 실패 — {e}")
            return BaseGetSearchQueryResponse(search_results=f"Error: search is unavailable ({e.kind})")
        stats["search_credits"] += int((results.get("usage") or {}).get("credits") or 0)
        return BaseGetSearchQueryResponse(search_results=format_search_results(results, self.config.max_result_chars))

    async def browse(self, request: Request, body: BaseGetPageContentRequest) -> BaseGetPageContentResponse:
        stats = self._stats(request)
        stats["browse_calls"] += 1
        reason = blocked_url_reason(body.url)
        if reason:
            stats["browse_errors"] += 1
            return BaseGetPageContentResponse(page_content=f"Error: {reason}")
        try:
            ok, text = await asyncio.get_running_loop().run_in_executor(self._browse_pool, fetch_page_text, body.url)
        except Exception as e:
            ok, text = False, f"Error: Unexpected error = {type(e).__name__}"
        if not ok:
            stats["browse_errors"] += 1
        return BaseGetPageContentResponse(page_content=text)

    async def verify(self, request: Request, body: GoogleSearchVerifyRequest) -> AlphaWebSearchMcqaVerifyResponse:
        stats = self._session_stats.pop(request.session[SESSION_ID_KEY], Counter())
        text = final_answer_text(body)
        parsed_option = box_parser(text) if text is not None else None
        backend_failed = stats["search_errors"] > 0
        return AlphaWebSearchMcqaVerifyResponse(
            **body.model_dump(),
            reward=1.0 if parsed_option is not None and parsed_option == body.expected_answer else 0.0,
            parsed_option=parsed_option or "",
            answer_found=parsed_option is not None,
            num_search_calls=stats["search_calls"],
            num_search_errors=stats["search_errors"],
            num_search_retries=stats["search_retries"],
            num_search_credits=stats["search_credits"],
            num_browse_calls=stats["browse_calls"],
            num_browse_errors=stats["browse_errors"],
            failure_reason="search_backend_error" if backend_failed else None,
            instance_config={"mask_sample": True} if backend_failed and self.config.mask_on_search_error else None,
        )


if __name__ == "__main__":
    AlphaWebSearchMcqaResourcesServer.run_webserver()
