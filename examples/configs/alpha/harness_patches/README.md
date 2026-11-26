# harness_patches — nv-OpenHands (Gym `swe_agents` 하니스) 에 얹는 수정

고정 커밋 `5f0180054732945df08ad2293903e6873f0492b6` (`sdevare-nv/nv-OpenHands`) 용이다. 적용 · 확인 · 되돌리기는 `tools/apply_harness_patches.sh` 가 한다.
대상은 `$NRL_ROOT/gym_cache/swe_agents/swe_openhands_setup/*/<커밋>/OpenHands` 의 작업 트리다 (커밋하지 않는다 — Gym 은 HEAD 가 고정 커밋이면 트리를 건드리지 않는다).
SWE 런 · 측정을 띄우기 전에 `apply_harness_patches.sh check` 가 0 이어야 한다.

| 패치 | 무엇 | 왜 |
|---|---|---|
| `0001-conversation-memory-keep-rejected-tool-calls.patch` | 대화 조립 (`openhands/memory/conversation_memory.py` `_process_action`) 이 `ValidationFailureAction` · `FunctionCallNotExistsAction` 도 assistant 메시지로 만든다 | 하니스가 거부한 도구 호출 (필수 인자 누락 · 없는 도구 이름) 의 응답과 오류 문구가 다음 프롬프트와 학습 궤적에서 모두 빠졌다 — 모델은 오류를 못 보고 같은 프롬프트로 다시 불리고, 학습기는 그 생성을 받지 못한다 (`docs/E2E_SANDBOX.md` §7.11, `RL_PLAN.md` 결정 88). 패치 뒤에는 그 턴이 토큰 ID 와 함께 남고 오류 문구가 바로 뒤 `tool` 응답으로 나간다 |

학습기 쪽 짝: 레시피의 `env.nemo_gym.invalid_tool_call_output_patterns` (NeMo-RL 본체 패치) 가 그 오류 응답을 보고 턴에 잘못된 도구 호출 표지를 붙인다.
확인: fork 의 단위 테스트 (`tests/unit/memory/test_conversation_memory.py` · `test_tool_error_conversation_flow.py`) 는 패치 전후로 결과가 같다 (같은 13 개가 원래 실패한다).
가짜 모델 시험은 `$NRL_ROOT/gates/swe_e2e_s3/run_stub_reject.sh` (음성 대조 = `reverted`).
