"""alpha_vllm_model 의 strict 처리 · chat 요청의 하니스 전용 키 처리 단위 테스트 (CPU). Gym 내장 동작과의 차이만 확인한다.

실행: NEMO_GYM_EXTRA_ROOTS=<NeMo-RL>/examples/configs/alpha/gym_plugins 를 주고 nemo_gym 이 있는 venv 의 pytest 로 이 파일을 돌린다.
"""

import asyncio

import pytest
from fastapi.exceptions import RequestValidationError

import nemo_gym  # noqa: F401  — import 시 플러그인 루트와 Gym 루트를 sys.path 에 올려 responses_api_models 를 합친다
from responses_api_models.alpha_vllm_model.app import AlphaVLLMModel
from responses_api_models.vllm_model.app import VLLMModel


def _chat_body(**extra):
    return {"model": "alpha", "messages": [{"role": "user", "content": "hi"}], "temperature": 1.0, **extra}


def _dispatch(cls, body):
    """서버를 띄우지 않고 dispatch 만 돈다 — 검증을 통과한 요청이 `chat_completions` 로 넘어가는 지점에서 받아 본다."""
    server = cls.__new__(cls)   # __init__ (서버 설정 · 클라이언트) 없이 메서드만 쓴다
    seen = {}

    async def fake_invoke(request, params):
        seen["params"] = params
        return {"ok": True}

    server._invoke_chat_completions = fake_invoke
    asyncio.run(cls.chat_completions_dispatch(server, None, body))
    return seen["params"]


def test_builtin_rejects_openhands_client_field():
    # 음성 대조: Gym 0.6.0 내장 서버는 nv-OpenHands 가 늘 보내는 키를 422 로 거부한다 (이 결함이 고쳐지면 이 테스트가 깨져 알려 준다)
    with pytest.raises(RequestValidationError) as e:
        _dispatch(VLLMModel, _chat_body(aws_region_name=None))
    assert [err["loc"] for err in e.value.errors()] == [("body", "aws_region_name")]


def test_alpha_drops_openhands_client_field_and_does_not_forward_it():
    params = _dispatch(AlphaVLLMModel, _chat_body(aws_region_name=None))
    dumped = params.model_dump(exclude_unset=True)
    assert "aws_region_name" not in dumped
    assert dumped["messages"][0]["content"] == "hi" and dumped["temperature"] == 1.0


def test_alpha_still_rejects_other_unknown_fields():
    with pytest.raises(RequestValidationError) as e:
        _dispatch(AlphaVLLMModel, _chat_body(aws_region_name=None, not_a_real_field=1))
    assert [err["loc"] for err in e.value.errors()] == [("body", "not_a_real_field")]


def _body():
    return {
        "tools": [
            {"type": "function", "function": {"name": "a", "parameters": {}, "strict": True}},
            {"type": "function", "function": {"name": "b", "parameters": {}, "strict": False}},
            {"type": "function", "function": {"name": "c", "parameters": {}, "strict": None}},
            {"type": "function", "function": {"name": "d", "parameters": {}}},
        ]
    }


def test_alpha_keeps_bool_strict_and_drops_none():
    body = _body()
    AlphaVLLMModel._strip_hosted_only_tool_fields(body)
    fns = [t["function"] for t in body["tools"]]
    assert fns[0]["strict"] is True
    assert fns[1]["strict"] is False
    assert "strict" not in fns[2]
    assert "strict" not in fns[3]


def test_builtin_still_strips_everything():
    body = _body()
    VLLMModel._strip_hosted_only_tool_fields(body)
    assert all("strict" not in t["function"] for t in body["tools"])


def test_no_tools_is_noop():
    body = {"messages": []}
    AlphaVLLMModel._strip_hosted_only_tool_fields(body)
    assert body == {"messages": []}
