"""alpha GenRM 모델 서버 — 이미 떠 있는 외부 vLLM endpoint 의 GenRM 을 Gym `genrm_compare` 에 붙인다 (RL_PLAN.md 결정 49).

Gym 0.6.0 의 `genrm_model` 은 vLLM 을 직접 띄우는 경로(`LocalVLLMModel`)만 있고, 그 모듈은 import 만으로 vllm 을 요구한다
(플러그인 서버 venv 에는 vllm 이 없다). 그래서 `GenRMModelMixin._preprocess_chat_completion_create_params` 의 역할 주입
(metadata 의 response_1 · response_2 · principle 을 대화 이력 뒤 메시지로 붙인다, Gym `responses_api_models/genrm_model/app.py`
v0.6.0 47~80행)을 `inject_genrm_roles` 로 옮겨 외부 base_url 을 받는 `VLLMModel` 에 얹는다.

`nvidia/Llama-3_3-Nemotron-Super-49B-GenRM` 은 chat template 이 점수를 `\\boxed{x, y}` (개별 1~5) · `\\boxed{z}` (순위 1~6) 로
내라고 지시하고, reasoning parser 없이 띄우면 `<think>…</think>` 가 본문에 섞인다. 반면 `genrm_compare.parse_genrm_output` 은
JSON `{"score_1", "score_2", "ranking"}` 만 읽고, 못 읽으면 재시도 뒤 기본값(순위 3.5 = 무승부)을 돌려준다 — 보상이 조용히 사라진다.
그래서 응답의 최종 텍스트를 `</think>` 뒤의 boxed 점수로 만든 JSON 으로 바꾼다. 못 읽으면 텍스트를 그대로 두고(채점기가 재시도한다)
실패를 센다. 집계는 `GET /alpha_genrm/stats` 와 로그 `[alpha_genrm]` 로 본다.

Gym 플러그인 루트(`NEMO_GYM_EXTRA_ROOTS=<이 리포>/examples/configs/alpha/gym_plugins`)로 붙는다.
"""

import json
import logging
import re
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request
from pydantic import PrivateAttr

from nemo_gym.base_responses_api_model import Body
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig

logger = logging.getLogger(__name__)

_THINK_OPEN, _THINK_CLOSE = "<think>", "</think>"
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_INDIVIDUAL = re.compile(r"\[The Begin of Individual Scores\](.*?)\[The End of Individual Scores\]", re.S)
_RANKING = re.compile(r"\[The Begin of Ranking Score\](.*?)\[The End of Ranking Score\]", re.S)


def _ints(s: str) -> List[int]:
    return [int(x) for x in re.findall(r"-?\d+", s)]


def parse_boxed_scores(text: str) -> Optional[Tuple[int, int, int]]:
    """GenRM 49B 출력에서 (score_1, score_2, ranking) 을 꺼낸다. 못 꺼내면 None.

    - `</think>` 가 있으면 그 뒤만 본다 (사고 안의 boxed 는 쓰지 않는다). `<think>` 만 있고 닫히지 않았으면 None.
    - 먼저 "[The Begin of Individual Scores]" · "[The Begin of Ranking Score]" 구간 안의 boxed 를, 없으면 사고 뒤 boxed 중
      숫자 2개짜리의 마지막과 1개짜리의 마지막을 쓴다.
    - 범위 밖 (개별 1~5, 순위 1~6) 이면 None.
    """
    if _THINK_CLOSE in text:
        tail = text.rsplit(_THINK_CLOSE, 1)[1]
    elif _THINK_OPEN in text:
        return None
    else:
        tail = text
    pair = rank = None
    m = _INDIVIDUAL.search(tail)
    if m:
        boxes = [_ints(b) for b in _BOXED.findall(m.group(1))]
        boxes = [b for b in boxes if len(b) == 2]
        pair = boxes[-1] if boxes else None
    m = _RANKING.search(tail)
    if m:
        boxes = [_ints(b) for b in _BOXED.findall(m.group(1))]
        boxes = [b for b in boxes if len(b) == 1]
        rank = boxes[-1][0] if boxes else None
    if pair is None or rank is None:
        all_boxes = [_ints(b) for b in _BOXED.findall(tail)]
        if pair is None:
            twos = [b for b in all_boxes if len(b) == 2]
            pair = twos[-1] if twos else None
        if rank is None:
            ones = [b for b in all_boxes if len(b) == 1]
            rank = ones[-1][0] if ones else None
    if pair is None or rank is None:
        return None
    s1, s2 = pair
    if not (1 <= s1 <= 5 and 1 <= s2 <= 5 and 1 <= rank <= 6):
        return None
    return s1, s2, rank


def normalize_response_dict(resp: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Responses API dict 의 최종 output_text 를 점수 JSON 으로 바꾼다. (새 dict, 성공 여부)."""
    texts = []
    for item in resp.get("output") or []:
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if part.get("type") == "output_text":
                texts.append(part)
    if not texts:
        return resp, False
    parsed = parse_boxed_scores("".join(p.get("text") or "" for p in texts))
    if parsed is None:
        return resp, False
    s1, s2, rank = parsed
    texts[-1]["text"] = json.dumps({"score_1": s1, "score_2": s2, "ranking": rank})
    for p in texts[:-1]:
        p["text"] = ""
    return resp, True


def inject_genrm_roles(body_dict: Dict[str, Any], supports_principle_role: bool) -> Dict[str, Any]:
    """Gym `GenRMModelMixin` 과 같은 역할 주입. metadata 는 소비하고 vLLM 으로 넘기지 않는다."""
    metadata = body_dict.pop("metadata", None) or {}
    messages = body_dict["messages"]
    principle = metadata.get("principle")
    if supports_principle_role and principle:
        messages.append({"role": "principle", "content": principle})
    if metadata.get("response_1") is not None:
        messages.append({"role": "response_1", "content": metadata["response_1"]})
    if metadata.get("response_2") is not None:
        messages.append({"role": "response_2", "content": metadata["response_2"]})
    return body_dict


class AlphaGenRMModelConfig(VLLMModelConfig):
    """외부 GenRM endpoint 설정. principle 역할은 49B GenRM 이 지원하지 않는다."""

    supports_principle_role: bool = False


class AlphaGenRMModel(VLLMModel):
    """외부 vLLM GenRM + boxed 점수 → JSON 변환."""

    config: AlphaGenRMModelConfig

    _stats_lock: Any = PrivateAttr(default_factory=Lock)
    _stats: Dict[str, int] = PrivateAttr(default_factory=lambda: {"requests": 0, "parsed": 0, "parse_failed": 0})

    def _preprocess_chat_completion_create_params(self, request: Request, body_dict: Dict[str, Any]) -> Dict[str, Any]:
        body_dict = super()._preprocess_chat_completion_create_params(request, body_dict)
        return inject_genrm_roles(body_dict, self.config.supports_principle_role)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.get("/alpha_genrm/stats")(self.stats)
        return app

    async def stats(self) -> Dict[str, int]:
        with self._stats_lock:
            return dict(self._stats)

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        response = await super().responses(request, body)
        resp_dict, ok = normalize_response_dict(response.model_dump(mode="json"))
        with self._stats_lock:
            self._stats["requests"] += 1
            self._stats["parsed" if ok else "parse_failed"] += 1
            failed, total = self._stats["parse_failed"], self._stats["requests"]
        if not ok:
            logger.warning(f"[alpha_genrm] parse_failed ({failed}/{total}); output text left unchanged for genrm_compare retry")
            return response
        return NeMoGymResponse.model_validate(resp_dict)


if __name__ == "__main__":
    AlphaGenRMModel.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = AlphaGenRMModel.run_webserver()  # noqa: F401
