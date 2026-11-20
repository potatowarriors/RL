"""alpha_web_search_mcqa 사전 검사 (CPU). 데이터 행이 플러그인 서버와 맞는지 보고, `--live` 면 실제 Tavily 로 한 문항을 끝까지 돌린다.

데이터 검사: 행이 서버 요청 모델에 맞는가 · 도구 이름과 인자가 서버 라우트와 같은가 · 정답 글자가 문제의 선택지에 있는가 · agent_ref.
`--live`: 플러그인 설정 파일 그대로 서버를 프로세스 안에 띄워 검색 1회 → 첫 결과 페이지 browse → verify. Tavily 크레딧을 쓴다
(basic 1 · advanced 2) — 사용량 조회 (`GET /usage`) 는 크레딧을 쓰지 않는다. 키는 출력하지 않는다.

실행 (플러그인 venv, Ray 격리):
  NEMO_GYM_EXTRA_ROOTS=<NeMo-RL>/examples/configs/alpha/gym_plugins RAY_ADDRESS=local <venv>/bin/python \
    examples/configs/alpha/tools/check_web_search_mcqa.py <rows.jsonl> [--expect-agent-ref NAME] [--live]
종료 코드: 0 = 결함 0, 1 = 결함 있음.
"""

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import nemo_gym  # noqa: F401  — import 시 플러그인 루트와 Gym 루트를 sys.path 에 올린다
import yaml
from fastapi.testclient import TestClient
from nemo_gym.server_utils import ServerClient
from resources_servers.alpha_web_search_mcqa import app as plugin
from resources_servers.google_search.app import BaseGetPageContentRequest, BaseSearchQueryRequest, GoogleSearchRunRequest


AGENT_NAME = "alpha_web_search_mcqa_simple_agent"
TOOL_ARGS = {"search": set(BaseSearchQueryRequest.model_fields), "browse": set(BaseGetPageContentRequest.model_fields)}
OPTION_LINE = re.compile(r"^\s*\(?([A-Z])[:.)]\s", re.MULTILINE)


def make_server():
    """운영과 같은 설정 파일로 서버를 만든다 (키 파일 경로 포함)."""
    cfg = yaml.safe_load((Path(plugin.__file__).parent / "configs" / "alpha_web_search_mcqa.yaml").read_text())
    fields = dict(cfg["alpha_web_search_mcqa_resources_server"]["resources_servers"]["alpha_web_search_mcqa"])
    for meta in ("domain", "verified", "description"):
        fields.pop(meta, None)
    config = plugin.AlphaWebSearchMcqaResourcesServerConfig(host="127.0.0.1", port=1, name="alpha_web_search_mcqa", **fields)
    return plugin.AlphaWebSearchMcqaResourcesServer(config=config, server_client=MagicMock(spec=ServerClient)), fields


def prompt_text(params: dict) -> str:
    inp = params["input"]
    return inp if isinstance(inp, str) else "\n".join(m["content"] for m in inp if isinstance(m.get("content"), str))


def check_rows(path: Path, expect_agent_ref: str | None, routes: set[str]) -> tuple[list[dict], int]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    defects = collections.Counter()
    options_per_row, floor = collections.Counter(), 0.0
    for row in rows:
        try:
            GoogleSearchRunRequest.model_validate(row)
        except Exception:
            defects["요청 모델 검증 실패"] += 1
            continue
        tools = {t.get("name"): set((t.get("parameters") or {}).get("properties") or {}) for t in row["responses_create_params"].get("tools") or []}
        if set(tools) != set(TOOL_ARGS) or not {f"/{name}" for name in tools} <= routes:
            defects["도구 이름이 서버 라우트와 다름"] += 1
        elif tools != TOOL_ARGS:
            defects["도구 인자가 서버 요청 모델과 다름"] += 1
        labels = sorted(set(OPTION_LINE.findall(prompt_text(row["responses_create_params"]))))
        options_per_row[len(labels)] += 1
        floor += 1.0 / len(labels) if labels else 0.0
        if row["expected_answer"] not in labels:
            defects["정답 글자가 문제의 선택지에 없음"] += 1
        agent = (row.get("agent_ref") or {}).get("name")
        if expect_agent_ref is None:
            defects["agent_ref 없음 (블렌드에서 붙여야 함 — 결함으로 세지 않음)"] += agent is None
        elif agent != expect_agent_ref:
            defects[f"agent_ref 가 {expect_agent_ref} 가 아님"] += 1
    print(f"행 {len(rows)} · 선택지 수 분포 {dict(sorted(options_per_row.items()))} · 무작위 선택 기대 보상 {floor / max(len(rows), 1):.3f}")
    hard = 0
    for name, n in sorted(defects.items()):
        counted = "결함으로 세지 않음" not in name
        hard += n if counted else 0
        print(f"  {'결함' if counted and n else '참고'}: {name} = {n}")
    return rows, hard


def tavily_usage(key: str) -> dict:
    data = httpx.get("https://api.tavily.com/usage", headers={"Authorization": f"Bearer {key}"}, timeout=30).raise_for_status().json()
    return {"plan": data["account"]["current_plan"], "used": data["account"]["plan_usage"], "limit": data["account"]["plan_limit"]}


def live_check(server, fields: dict, row: dict) -> int:
    key = plugin.load_tavily_keys(fields.get("tavily_api_key"), fields.get("tavily_api_key_file"))[0]
    before = tavily_usage(key)
    print(f"Tavily 요금제 {before['plan']} · 사용 {before['used']} / {before['limit']}")
    query = prompt_text(row["responses_create_params"]).splitlines()[0][: plugin.MAX_QUERY_CHARS]
    with TestClient(server.setup_webserver()) as client:
        client.post("/seed_session", json={}).raise_for_status()
        results = client.post("/search", json={"query": query}).raise_for_status().json()["search_results"]
        urls = re.findall(r"^    URL: (\S+)$", results, re.MULTILINE)
        print(f"검색: 질의 {len(query)} 자 → 결과 {len(urls)} 건 · 글 {len(results)} 자\n{results[:600]}")
        page = client.post("/browse", json={"url": urls[0]}).raise_for_status().json()["page_content"] if urls else ""
        print(f"browse: {urls[0] if urls else '-'} → {len(page)} 자 · 앞 200 자: {page[:200]!r}")
        answer = {"id": "msg", "role": "assistant", "status": "completed", "type": "message",
                  "content": [{"annotations": [], "text": rf"\boxed{{{row['expected_answer']}}}", "type": "output_text"}]}
        response = {"id": "r", "created_at": 0.0, "model": "m", "object": "response", "parallel_tool_calls": False,
                    "tool_choice": "none", "tools": [], "output": [answer]}
        verdict = client.post("/verify", json=row | {"response": response}).raise_for_status().json()
    after = tavily_usage(key)
    shown = {k: verdict[k] for k in ("reward", "answer_found", "num_search_calls", "num_search_errors", "num_search_credits",
                                     "num_browse_calls", "num_browse_errors", "instance_config", "failure_reason")}
    print(f"verify: {shown}")
    print(f"Tavily 사용량 {before['used']} → {after['used']} (응답이 알린 크레딧 {verdict['num_search_credits']})")
    problems = []
    if not urls or results.startswith("Error"):
        problems.append("검색 결과 없음")
    if verdict["num_search_errors"] or verdict["instance_config"]:
        problems.append("검색 백엔드 장애로 기록됨")
    if verdict["reward"] != 1.0:
        problems.append("정답 글자를 넣었는데 보상이 1 이 아님")
    if verdict["num_browse_errors"] or len(page) < 200:
        problems.append("browse 가 본문을 얻지 못함")
    if key in results or key in page or key in json.dumps(verdict):
        problems.append("키가 출력에 섞임")
    for p in problems:
        print(f"  결함: {p}")
    return len(problems)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rows", type=Path)
    parser.add_argument("--expect-agent-ref", default=None, help=f"블렌드 파일이면 {AGENT_NAME}")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    server, fields = make_server()
    routes = {r.path for r in server.setup_webserver().routes}
    rows, defects = check_rows(args.rows, args.expect_agent_ref, routes)
    if args.live:
        defects += live_check(server, fields, rows[0])
    print("결과:", "결함 0" if defects == 0 else f"결함 {defects}")
    return 0 if defects == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
