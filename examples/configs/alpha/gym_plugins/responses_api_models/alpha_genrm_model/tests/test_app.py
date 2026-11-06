"""alpha_genrm_model 단위 테스트 (CPU). boxed 점수 → JSON 변환과 Gym genrm_compare 파서와의 연결을 확인한다.

실행: NEMO_GYM_EXTRA_ROOTS=<NeMo-RL>/examples/configs/alpha/gym_plugins 를 주고 nemo_gym 이 있는 venv 의 pytest 로 이 파일을 돌린다.
고정 자료 data/genrm49b_sky_blue.txt 는 2026-11-06 외부 endpoint 실측 응답 원문이다 (응답 1 우세, \\boxed{4, 1} · \\boxed{1}).
"""

import json
from pathlib import Path

import pytest

import nemo_gym  # noqa: F401  — import 시 플러그인 루트와 Gym 루트를 sys.path 에 올린다
from resources_servers.genrm_compare.utils import GenRMOutputParseError, parse_genrm_output
from responses_api_models.alpha_genrm_model.app import (
    AlphaGenRMModel,
    _last_scores,
    inject_genrm_roles,
    normalize_response_dict,
    parse_boxed_scores,
)
from responses_api_models.vllm_model.app import VLLMModel

REAL = (Path(__file__).parent / "data" / "genrm49b_sky_blue.txt").read_text()

SECTIONS = (
    "[The Begin of Individual Scores]\n\\boxed{{{a}, {b}}}\n[The End of Individual Scores]\n\n"
    "[The Begin of Ranking Score]\n\\boxed{{{r}}}\n[The End of Ranking Score]"
)


def _resp(text):
    return {
        "id": "r",
        "object": "response",
        "output": [
            {"type": "message", "role": "assistant", "id": "m", "status": "completed",
             "content": [{"type": "output_text", "text": text, "annotations": []}]}
        ],
    }


def test_real_output_parses():
    assert parse_boxed_scores(REAL) == (4, 1, 1)


def test_real_output_breaks_gym_parser_without_adapter():
    # 음성 대조: 변환 없이는 Gym 파서가 실패한다 (genrm_compare 는 이때 재시도 뒤 무승부 3.5 로 떨어진다)
    with pytest.raises(GenRMOutputParseError):
        parse_genrm_output(REAL, 3.0, 3.5, raise_on_fail=True)


def test_normalized_output_reaches_gym_parser():
    resp, ok = normalize_response_dict(_resp(REAL))
    assert ok
    text = resp["output"][0]["content"][0]["text"]
    assert json.loads(text) == {"score_1": 4, "score_2": 1, "ranking": 1}
    assert parse_genrm_output(text, 3.0, 3.5, raise_on_fail=True) == (4.0, 1.0, 1.0)


def test_boxed_inside_think_is_ignored():
    text = "<think>maybe \\boxed{5, 5} and \\boxed{6}</think>\n" + SECTIONS.format(a=2, b=4, r=5)
    assert parse_boxed_scores(text) == (2, 4, 5)


def test_unterminated_think_fails():
    assert parse_boxed_scores("<think>long reasoning \\boxed{4, 2} \\boxed{2}") is None


def test_out_of_range_fails():
    assert parse_boxed_scores("</think>" + SECTIONS.format(a=7, b=2, r=2)) is None
    assert parse_boxed_scores("</think>" + SECTIONS.format(a=3, b=2, r=0)) is None


def test_missing_ranking_fails():
    assert parse_boxed_scores("</think>[The Begin of Individual Scores]\\boxed{3, 2}[The End of Individual Scores]") is None


def test_fallback_without_section_markers():
    assert parse_boxed_scores("</think> scores \\boxed{3,5} ranking \\boxed{6}") == (3, 5, 6)


def test_failed_parse_leaves_text_unchanged():
    raw = "</think> no scores here"
    resp, ok = normalize_response_dict(_resp(raw))
    assert not ok and resp["output"][0]["content"][0]["text"] == raw


def test_role_injection_matches_gym_mixin():
    # Gym GenRMModelMixin (v0.6.0) 과 같은 결과: metadata 를 소비하고 대화 뒤에 response_1 · response_2 를 붙인다
    body = {"messages": [{"role": "user", "content": "Q"}], "metadata": {"response_1": "A", "response_2": "B", "principle": "P"}}
    out = inject_genrm_roles(body, supports_principle_role=False)
    assert "metadata" not in out
    assert out["messages"] == [{"role": "user", "content": "Q"}, {"role": "response_1", "content": "A"}, {"role": "response_2", "content": "B"}]
    body = {"messages": [], "metadata": {"response_1": "A", "principle": "P"}}
    assert inject_genrm_roles(body, supports_principle_role=True)["messages"] == [
        {"role": "principle", "content": "P"}, {"role": "response_1", "content": "A"}]


def test_principle_as_user_suffix_for_genrm_without_principle_role():
    body = {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"},
                         {"role": "user", "content": "Q2"}], "metadata": {"response_1": "A", "response_2": "B", "principle": "Criteria: say alpha"}}
    out = inject_genrm_roles(body, supports_principle_role=False, principle_as_user_suffix=True)
    assert out["messages"][3]["content"].startswith("Q2\n\n[Evaluation criteria for judging the responses]\nCriteria: say alpha")
    assert out["messages"][1]["content"] == "Q1" and out["messages"][0] == {"role": "system", "content": "S"}
    assert [m["role"] for m in out["messages"]][-2:] == ["response_1", "response_2"]
    # default_principle "" (일반 대화 행) 이면 아무것도 붙지 않는다
    body = {"messages": [{"role": "user", "content": "Q"}], "metadata": {"response_1": "A", "response_2": "B", "principle": ""}}
    assert inject_genrm_roles(body, False, True)["messages"][0]["content"] == "Q"


def test_genrm_template_drops_system_messages():
    # 49B GenRM chat template (리비전 3160979) 은 대화에서 user · assistant 만 그린다 — system 으로 principle 을 넣으면 안 되는 이유
    tpl = (Path(__file__).parent / "data" / "genrm49b_chat_template.jinja").read_text()
    assert "set system_message = 'detailed thinking on'" in tpl
    loop = tpl[tpl.index("#### Conversation Context"):tpl.index("#### Responses to be Scored")]
    assert "'user'" in loop and "'assistant'" in loop and "'system'" not in loop


def test_role_injection_matches_gym_source():
    # Gym 원본 소스와 같은 순서·조건인지 원문으로 대조한다 (원본은 import 하면 vllm 을 요구해 직접 부를 수 없다)
    src = (Path(nemo_gym.__file__).parent.parent / "responses_api_models" / "genrm_model" / "app.py").read_text()
    for frag in ('body_dict.pop("metadata", None) or {}', 'if self.config.supports_principle_role and principle:',
                 '{"role": "principle", "content": principle}', '{"role": "response_1", "content": response_1}',
                 '{"role": "response_2", "content": response_2}'):
        assert frag in src, frag
    assert src.index('"role": "principle"') < src.index('"role": "response_1"') < src.index('"role": "response_2"')


def test_server_overrides_preprocess():
    assert AlphaGenRMModel._preprocess_chat_completion_create_params is not VLLMModel._preprocess_chat_completion_create_params


def test_last_scores_reads_normalized_json():
    resp, ok = normalize_response_dict(_resp(REAL))
    assert ok and _last_scores(resp) == {"score_1": 4, "score_2": 1, "ranking": 1}
