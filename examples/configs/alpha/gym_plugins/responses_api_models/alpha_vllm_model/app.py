"""alpha vLLM 모델 서버 — Gym `VLLMModel` 과 같고, 도구 정의의 `strict` 만 남긴다 (RL_PLAN.md 결정 13, G3).

Gym 0.6.0 `VLLMModel._strip_hosted_only_tool_fields` 는 `strict` 를 무조건 지운다. alpha SFT(Agentic-v2 interactive)는
`<strict>True</strict>` 를 렌더한 형식으로 학습했고, RL 블렌드 도구 행 13,190/13,190 이 `strict: true` 를 담는다.
그래서 bool 이면 남기고, None 처럼 값이 없으면 지운다 (`<strict>None</strict>` 렌더 방지).

Gym 플러그인 루트(`NEMO_GYM_EXTRA_ROOTS=<이 리포>/examples/configs/alpha/gym_plugins`)로 붙는다.
`responses_api_models` 는 namespace 패키지라 아래 import 는 Gym 내장 `vllm_model` 을 가리킨다.
그 외 동작(토큰 ID 전달·R3 routed_experts 전달·추론 파서 처리)은 상속 그대로다.

둘째 차이 (RL_PLAN.md 결정 78 · 79, KNOWN_ISSUES 2026-11-25): `/v1/chat/completions` 요청에서 하니스 전용 키를 검증 전에 버린다.
Gym 0.6.0 (`3045a793`) 의 chat 요청 모델은 모르는 키를 422 로 거부하는데 (`extra="forbid"`), 그 Gym 이 `swe_agents` 학습 설정에 고정한
nv-OpenHands (`5f018005`) 는 최상위 `aws_region_name` (값 null) 과 도구 결과 메시지의 `name` 을 요청마다 넣어 보낸다 — 에이전트의 모델 호출이 422 로 끝난다.
upstream Gym 은 뒤에 고쳤다 (#4011 `bd755562d` · #4483 `5c14c66b4`, `nemo_gym/openai_utils.py`). 그 수정을 Gym 소스에 이식하지 않고 여기서 하는 이유:
이 서버의 venv 는 Gym 라이브러리를 공개 패키지 (0.6.0) 로 받는다 (Gym 이 플러그인 서버의 venv 를 그렇게 만든다) — Gym 체크아웃을 고쳐도 닿지 않는다.
다른 모르는 키는 지금처럼 422 다.
"""

from typing import Any, Dict

from fastapi import Body, Request
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from responses_api_models.vllm_model.app import VLLMModel


# 하니스가 보내지만 모델 서버 · 백엔드가 쓰지 않는 요청 키. upstream Gym 의 같은 이름 상수와 같은 값이다.
CHAT_REQUEST_IGNORED_CLIENT_FIELDS = frozenset({"aws_region_name"})


def _drop_harness_only_fields(body: Dict[str, Any]) -> None:
    """chat 요청에서 하니스 전용 키를 검증 전에 버린다 (upstream Gym #4011 · #4483 과 같은 대상).

    - 최상위 `aws_region_name` (nv-OpenHands 가 값 null 로 늘 보낸다)
    - 도구 결과 메시지의 `name` (OpenHands 의 메시지 직렬화가 `tool_call_id` 옆에 넣는다). upstream 은 받아서 넘기지만
      alpha 채팅 템플릿은 도구 결과를 내용만으로 렌더하고 `name` 을 쓰지 않으므로 버려도 모델 입력이 같다.
    - 메시지의 `provider_specific_fields` (litellm 의 `Message.model_dump()` 가 assistant 턴마다 넣는다)
    """
    for field in CHAT_REQUEST_IGNORED_CLIENT_FIELDS:
        body.pop(field, None)
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            message.pop("provider_specific_fields", None)
            if message.get("role") == "tool":
                message.pop("name", None)


class AlphaVLLMModel(VLLMModel):
    """`strict` 가 bool 인 도구 정의는 그대로 두고, chat 요청의 하니스 전용 키는 검증 전에 버린다."""

    async def chat_completions_dispatch(self, request: Request, body: dict = Body()):
        _drop_harness_only_fields(body)
        return await super().chat_completions_dispatch(request, body)

    @staticmethod
    def _strip_hosted_only_tool_fields(body_dict: Dict[str, Any]) -> None:
        for tool_dict in body_dict.get("tools") or []:
            if tool_dict.get("type") != "function":
                continue
            function = tool_dict.get("function") or {}
            if "strict" in function and not isinstance(function["strict"], bool):
                function.pop("strict")


if __name__ == "__main__":
    AlphaVLLMModel.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = AlphaVLLMModel.run_webserver()  # noqa: F401
