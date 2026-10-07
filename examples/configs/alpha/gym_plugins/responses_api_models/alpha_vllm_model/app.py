"""alpha vLLM 모델 서버 — Gym `VLLMModel` 과 같고, 도구 정의의 `strict` 만 남긴다 (RL_PLAN.md 결정 13, G3).

Gym 0.6.0 `VLLMModel._strip_hosted_only_tool_fields` 는 `strict` 를 무조건 지운다. alpha SFT(Agentic-v2 interactive)는
`<strict>True</strict>` 를 렌더한 형식으로 학습했고, RL 블렌드 도구 행 13,190/13,190 이 `strict: true` 를 담는다.
그래서 bool 이면 남기고, None 처럼 값이 없으면 지운다 (`<strict>None</strict>` 렌더 방지).

Gym 플러그인 루트(`NEMO_GYM_EXTRA_ROOTS=<이 리포>/examples/configs/alpha/gym_plugins`)로 붙는다.
`responses_api_models` 는 namespace 패키지라 아래 import 는 Gym 내장 `vllm_model` 을 가리킨다.
그 외 동작(토큰 ID 전달·R3 routed_experts 전달·추론 파서 처리)은 상속 그대로다.
"""

from typing import Any, Dict

from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from responses_api_models.vllm_model.app import VLLMModel


class AlphaVLLMModel(VLLMModel):
    """`strict` 가 bool 인 도구 정의는 그대로 둔다."""

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
