r"""alpha Gym 플러그인 — resources_servers/alpha_terminus_judge. Gym 0.6.0 `terminus_judge` 에 완료 선언 검사 하나를 더한다 (RL_PLAN.md 결정 66).

Gym 의 `check_task_complete` 는 한 방향뿐이다: 정답이 task_complete 일 때 후보도 선언했는지만 본다. 정답이 "진행 중" 인데 후보가 정답과 같은
명령을 내면서 task_complete 를 선언하면 유사도 1.0 으로 보상 1 을 받는다 (게이트 TJ1 의 K_earlytc 100/100 통과, MOPD1 학생 프로파일링에서
조기 완료 선언 1,096 출력). Terminal-Bench 에서 섣부른 완료 선언은 과제를 끝내 버려 점수를 깎는다.
이 서버는 그 경우를 보상 0 (`task_complete_check_failed`) 으로 돌리고, 나머지는 전부 Gym 서버의 verify 에 맡긴다 —
스키마 · 유사도 빠른 통과 · LLM judge (rubrics_v4) · 판정 파싱은 Gym 코드 그대로다.
빠른 통과 임계 (`string_similarity_threshold`) 를 1.0 으로 두는 것은 설정이다 (configs/alpha_terminus_judge.yaml) — 0.95 면 긴 명령에서
`sed -i` 치환값 하나만 바뀐 후보가 judge 없이 통과한다 (TJ1 K_fastsed 100/100).
"""

import json
from pathlib import Path

import resources_servers.terminus_judge.app as _upstream_app
from resources_servers.terminus_judge.app import (
    FailureCode,
    TerminusJudgeResourcesServer,
    TerminusJudgeResourcesServerConfig,
    TerminusJudgeVerifyRequest,
    TerminusJudgeVerifyResponse,
    _extract_expected_answer,
    _extract_last_assistant_text,
    _sanitize_for_json,
    _sanitize_pydantic_model,
    _sanitize_unicode_string,
)


class AlphaTerminusJudgeResourcesServerConfig(TerminusJudgeResourcesServerConfig):
    pass


def _declares_complete(d: dict) -> bool:
    return bool(d.get("task_complete") or d.get("is_task_complete"))


def early_completion(expected: str | None, generated: str | None) -> tuple[bool, str, dict | None]:
    """정답은 완료가 아닌데 후보가 완료를 선언했는가. (판정, 사고 태그를 뗀 후보 텍스트, 파싱한 후보) 를 돌려준다."""
    if not expected or not generated:
        return False, generated or "", None
    text = generated.split("</think>")[-1].strip() if "</think>" in generated else generated
    try:
        exp, pred = json.loads(expected), json.loads(text)
    except json.JSONDecodeError:
        return False, text, None
    if not isinstance(exp, dict) or not isinstance(pred, dict):
        return False, text, None
    return (_declares_complete(pred) and not _declares_complete(exp)), text, pred


class AlphaTerminusJudgeResourcesServer(TerminusJudgeResourcesServer):
    config: AlphaTerminusJudgeResourcesServerConfig

    def __init__(self, *args, **kwargs):
        cfg = kwargs.get("config")
        fpath = getattr(cfg, "judge_prompt_template_fpath", None)
        if fpath and not Path(fpath).is_absolute() and not Path(fpath).exists():
            upstream = Path(_upstream_app.__file__).parent / fpath   # 프롬프트는 Gym 의 것을 그대로 쓴다 (사본을 두지 않는다)
            if upstream.exists():
                cfg.judge_prompt_template_fpath = str(upstream)
        super().__init__(*args, **kwargs)

    async def verify(self, body: TerminusJudgeVerifyRequest) -> TerminusJudgeVerifyResponse:
        expected = _extract_expected_answer(body)
        early, text, pred = early_completion(expected, _extract_last_assistant_text(body))
        if not early:
            return await super().verify(body)
        return TerminusJudgeVerifyResponse(
            responses_create_params=_sanitize_pydantic_model(body.responses_create_params),
            response=_sanitize_pydantic_model(body.response),
            reward=0.0,
            uuid=body.uuid,
            expected_answer=_sanitize_unicode_string(expected),
            model_output=_sanitize_unicode_string(text),
            parsed_output=_sanitize_for_json(pred),
            similarity_score=None,
            schema_check_passed=None,
            task_complete_check_passed=False,
            string_similarity_passed=None,
            judge_passed=None,
            failure_reason=FailureCode.TASK_COMPLETE_CHECK_FAILED,
            judge_evaluations=[],
            metadata=_sanitize_for_json(body.metadata),
            threshold=body.threshold,
        )


if __name__ == "__main__":
    AlphaTerminusJudgeResourcesServer.run_webserver()
