"""rdkit_chemistry 채점기 단위 테스트 (CPU).

실행: NEMO_GYM_EXTRA_ROOTS=<NeMo-RL>/examples/configs/alpha/gym_plugins 를 주고 nemo_gym 이 있는 venv 의 pytest 로 이 파일을 돌린다.
"""

from unittest.mock import MagicMock

import nemo_gym  # noqa: F401  — import 시 플러그인 루트와 Gym 루트를 sys.path 에 올려 resources_servers 를 합친다
import pytest
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.rdkit_chemistry.app import (
    RDKitChemistryResourcesServer,
    RDKitChemistryResourcesServerConfig,
    RDKitChemistryVerifyRequest,
    extract_answer,
)


@pytest.mark.parametrize(
    ("text", "use_box_format", "expected"),
    [
        ("There are 13 amide bonds.\n\n((13))", False, 13),
        ("((13.0))", False, 13),
        ("First guess ((2)), corrected: (( 3 ))", False, 3),
        ("((1 = yes))", False, None),
        ("((2.5))", False, None),
        ("\\boxed{13}", False, None),  # 지시는 ((N)) — 다른 형식은 0 점
        ("The count is \\boxed{6}.", True, 6),
        ("$\\boxed{\\text{4}}$", True, 4),
        ("\\boxed{\\textbf{12}}", True, 12),
        ("((6))", True, None),  # 지시는 \boxed{N}
        ("\\boxed{0} then \\boxed{1}", True, 1),
        ("no answer", True, None),
        ("", False, None),
    ],
)
def test_extract_answer(text, use_box_format, expected):
    assert extract_answer(text, use_box_format) == expected


@pytest.mark.parametrize(
    ("text", "use_box_format", "expected"),
    [
        ("there is no C(=O)N group.\n\n((No))", False, 0),
        ("((NO))", False, 0),
        ("((yes.))", False, 1),
        ("\\boxed{Yes}", True, 1),
        ("\\boxed{\\text{False}}", True, 0),
        ("((**No**))", False, 0),
        ("((1))", False, 1),  # 정수 0/1 도 그대로 읽는다
        ("((maybe))", False, None),
        ("\\boxed{No}", False, None),  # 지시는 ((N))
        ("No.", False, None),  # 형식 없음
    ],
)
def test_extract_answer_yes_no(text, use_box_format, expected):
    assert extract_answer(text, use_box_format, yes_no=True) == expected


def test_yes_no_only_for_yes_no_rows():
    assert extract_answer("((No))", False) is None


def _request(
    text: str,
    expected_answer: str,
    use_box_format: bool,
    property_type: str = "fragment",
) -> RDKitChemistryVerifyRequest:
    response = NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            {
                "id": "msg_test",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )
    return RDKitChemistryVerifyRequest(
        responses_create_params={
            "input": [{"role": "user", "content": "How many amide bonds?"}]
        },
        response=response,
        expected_answer=expected_answer,
        use_box_format=use_box_format,
        property="fr_amide",
        property_type=property_type,
        smiles="CC(=O)NC",
    )


@pytest.mark.parametrize(
    ("text", "expected_answer", "use_box_format", "reward", "property_type"),
    [
        ("((1))", "1.0", False, 1.0, "fragment"),
        ("((2))", "1.0", False, 0.0, "fragment"),
        ("\\boxed{0}", "0.0", True, 1.0, "fragment"),
        ("((0))", "0.0", True, 0.0, "fragment"),
        ("((No))", "0.0", False, 1.0, "bool"),
        ("\\boxed{Yes}", "1.0", True, 1.0, "presence"),
        ("\\boxed{Yes}", "0.0", True, 0.0, "presence"),
        ("((Yes))", "1.0", False, 0.0, "count"),  # 개수 행은 정수만
    ],
)
async def test_verify(text, expected_answer, use_box_format, reward, property_type):
    server = RDKitChemistryResourcesServer(
        config=RDKitChemistryResourcesServerConfig(
            host="0.0.0.0", port=8080, entrypoint="", name=""
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    result = await server.verify(
        _request(text, expected_answer, use_box_format, property_type)
    )
    assert result.reward == reward
    assert result.expected_answer == expected_answer
