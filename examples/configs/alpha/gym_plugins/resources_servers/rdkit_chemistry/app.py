r"""alpha Gym 플러그인 — resources_servers/rdkit_chemistry. Gym 0.6.0 에는 이 서버가 없다 (Ultra 블렌드에만 행이 있다).

행: SMILES 하나의 RDKit 성질(작용기 개수 · 원자·결합 개수 · 0/1 판정)을 도구 없이 답한다 (method "direct").
정답 expected_answer 는 모두 정수다 ("13.0" 처럼 소수점 표기). 답 형식은 프롬프트가 정한다 —
use_box_format 이면 \boxed{N}, 아니면 ((N)).
채점: 최종 답 텍스트에서 지시한 형식의 마지막 값을 정수로 읽어 정답과 같으면 1, 아니면 0. 형식을 어기면 0 이다.
"""

import re
from typing import Any, Optional

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)

DOUBLE_PAREN = re.compile(r"\(\(\s*([^()]*?)\s*\)\)")
LATEX_WRAP = re.compile(r"\\(?:text|textbf|mathbf|mathrm)\{([^{}]*)\}")
INTEGER = re.compile(r"[-+]?\d+(?:\.0*)?")


class RDKitChemistryResourcesServerConfig(BaseResourcesServerConfig):
    pass


class RDKitChemistryRunRequest(BaseRunRequest):
    expected_answer: str
    use_box_format: bool
    property: Optional[str] = None
    property_type: Optional[str] = None
    smiles: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class RDKitChemistryVerifyRequest(RDKitChemistryRunRequest, BaseVerifyRequest):
    pass


class RDKitChemistryVerifyResponse(BaseVerifyResponse):
    expected_answer: str
    extracted_answer: Optional[str]


def _boxed_contents(text: str) -> list[str]:
    r"""\boxed{...} 의 내용을 중괄호 짝을 맞춰 모두 꺼낸다 (\boxed{\text{3}} 처럼 중첩돼도 된다)."""
    out = []
    for m in re.finditer(r"\\boxed\{", text):
        depth, i = 1, m.end()
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        if depth == 0:
            out.append(text[m.end() : i - 1])
    return out


def extract_answer(text: str, use_box_format: bool) -> Optional[int]:
    """지시한 형식의 마지막 값을 정수로 읽는다. 값이 없거나 정수 하나가 아니면 None."""
    found = _boxed_contents(text) if use_box_format else DOUBLE_PAREN.findall(text)
    if not found:
        return None
    value = found[-1]
    while LATEX_WRAP.search(value):
        value = LATEX_WRAP.sub(r"\1", value)
    value = value.strip().strip("$").strip()
    if not INTEGER.fullmatch(value):
        return None
    return int(float(value))


class RDKitChemistryResourcesServer(SimpleResourcesServer):
    config: RDKitChemistryResourcesServerConfig

    async def verify(
        self, body: RDKitChemistryVerifyRequest
    ) -> RDKitChemistryVerifyResponse:
        pred = extract_answer(body.response.output_text, body.use_box_format)
        gold = int(float(body.expected_answer))
        return RDKitChemistryVerifyResponse(
            **body.model_dump(),
            reward=float(pred == gold),
            extracted_answer=None if pred is None else str(pred),
        )


if __name__ == "__main__":
    RDKitChemistryResourcesServer.run_webserver()
