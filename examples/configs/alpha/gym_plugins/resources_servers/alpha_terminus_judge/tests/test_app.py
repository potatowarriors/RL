"""alpha_terminus_judge 단위 테스트 (CPU, judge 끔 — 유사도 경로만).

실행: NEMO_GYM_EXTRA_ROOTS=<NeMo-RL>/examples/configs/alpha/gym_plugins 를 주고 terminus_judge 서버 venv 의 pytest 로 이 파일을 돌린다.
"""

import asyncio
import json
from unittest.mock import MagicMock

import nemo_gym  # noqa: F401  — import 시 플러그인 루트와 Gym 루트를 sys.path 에 올린다
import pytest
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.alpha_terminus_judge.app import (
    AlphaTerminusJudgeResourcesServer,
    AlphaTerminusJudgeResourcesServerConfig,
    early_completion,
)
from resources_servers.terminus_judge.app import TerminusJudgeVerifyRequest


def action(keystrokes, complete=False, key="task_complete"):
    return json.dumps({"analysis": "a", "plan": "p", "commands": [{"keystrokes": k, "duration": 0.1} for k in keystrokes], key: complete})


@pytest.mark.parametrize(
    ("expected", "generated", "want"),
    [
        (action(["ls\n"]), action(["ls\n"], True), True),            # 조기 완료 선언
        (action(["ls\n"]), "<think>x</think>" + action(["ls\n"], True), True),
        (action(["ls\n"]), action(["ls\n"]), False),
        (action([], True), action([], True), False),                  # 정답도 완료
        (action([], True), action(["ls\n"]), False),                  # 반대 방향은 Gym 검사가 잡는다
        (action(["ls\n"]), "not json", False),
        (action(["ls\n"]), action(["ls\n"], True, key="is_task_complete"), True),
    ],
)
def test_early_completion(expected, generated, want):
    assert early_completion(expected, generated)[0] is want


def make_server():
    cfg = AlphaTerminusJudgeResourcesServerConfig(host="127.0.0.1", port=1, entrypoint="app.py", name="alpha_terminus_judge",
                                                  enable_string_similarity=True, string_similarity_threshold=1.0, enable_llm_judge=False)
    return AlphaTerminusJudgeResourcesServer(config=cfg, server_client=MagicMock(spec=ServerClient))


def verify(server, expected, generated):
    resp = NeMoGymResponse(id="r", created_at=0.0, model="m", object="response", parallel_tool_calls=False, tool_choice="none", tools=[],
                           output=[{"id": "msg", "role": "assistant", "status": "completed", "type": "message",
                                    "content": [{"annotations": [], "text": generated, "type": "output_text"}]}])
    req = TerminusJudgeVerifyRequest.model_validate({"responses_create_params": {"input": [{"role": "user", "content": "x"}]},
                                                     "response": resp.model_dump(), "expected_answer": expected,
                                                     "metadata": {"harness": "terminus_2"}})
    return asyncio.run(server.verify(req))


def test_early_completion_gets_zero():
    vr = verify(make_server(), action(["ls -la\n"]), action(["ls -la\n"], True))
    assert vr.reward == 0.0 and vr.failure_reason.value == "task_complete_check_failed" and vr.task_complete_check_passed is False


def test_exact_match_still_passes():
    vr = verify(make_server(), action(["ls -la\n"]), action(["ls -la\n"]))
    assert vr.reward == 1.0 and vr.failure_reason.value == "none"


def test_near_match_does_not_fast_pass_at_threshold_one():
    gold = "sed -i 's/timeout = 30/timeout = 300/' /app/a/very/long/path/to/some/configuration/file/settings.ini\n"
    vr = verify(make_server(), action([gold]), action([gold.replace("300", "003")]))
    assert vr.similarity_score is not None and 0.95 < vr.similarity_score < 1.0 and vr.reward == 0.0   # judge 끔이라 0 — 0.95 임계였다면 1


def test_gold_complete_candidate_complete_passes():
    vr = verify(make_server(), action([], True), action([], True))
    assert vr.reward == 1.0
