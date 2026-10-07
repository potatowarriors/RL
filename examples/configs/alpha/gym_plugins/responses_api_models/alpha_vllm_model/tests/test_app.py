"""alpha_vllm_model 의 strict 처리 단위 테스트 (CPU). Gym 내장 동작과의 차이만 확인한다.

실행: NEMO_GYM_EXTRA_ROOTS=<NeMo-RL>/examples/configs/alpha/gym_plugins 를 주고 nemo_gym 이 있는 venv 의 pytest 로 이 파일을 돌린다.
"""

import nemo_gym  # noqa: F401  — import 시 플러그인 루트와 Gym 루트를 sys.path 에 올려 responses_api_models 를 합친다
from responses_api_models.alpha_vllm_model.app import AlphaVLLMModel
from responses_api_models.vllm_model.app import VLLMModel


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
