---
paths:
  - "3rdparty/**"
  - "pyproject.toml"
  - "uv.lock"
---

# 서브모듈·의존성 규칙 (alpha)

- **Megatron-Bridge** (`3rdparty/Megatron-Bridge-workspace/Megatron-Bridge`): AlphaBridge 는 fork `potatowarriors/Megatron-Bridge`
  브랜치 **`alpha/bridge`** 에 있다 (`91b88c8f` AlphaBridge, `bcc4e415` FlashQLA opt-in). 워킹트리는 이 브랜치를 체크아웃한다.
  superproject 에 기록된 포인터는 upstream `0c565c9a` 다 → `git status` 의 `(new commits)` 표시는 현재 정상이다.
  **포인터를 fork 커밋으로 고정할지는 열린 결정이다** (`docs/STATUS.md`). 결정 전에는 포인터를 staging 하지 않는다.
  새 클론은 AlphaBridge 가 없으므로 `git -C <bridge> fetch fork && git checkout alpha/bridge` 가 필요하다.
- 브리지 코드를 고친 커밋은 **fork 에 먼저 push** 한다. 로컬 전용 SHA 를 superproject 에 기록하면 클론이 깨진다.
  브리지 수정 중에는 레시피에 `megatron_cfg.force_reconvert_from_hf: true` 를 두고(변환 캐시에 버전 검사 없음),
  끝나면 `docs/GATES.md` 게이트 1~3 을 재실행한다.
- **Gym** (`3rdparty/Gym-workspace/Gym`): **v0.6.0 `3045a793` 고정** (사용자 결정 2026-09-15, 머지 `0fcf0d88b` 2026-10-06).
  버전을 올리려면 먼저 제안 보고 후 승인을 받는다.
- **vLLM**: fork 를 lock 에 넣지 않는다 — `nemo-gym[vllm]` 의 flashinfer 핀과 충돌한다(`KNOWN_ISSUES` #10).
  alpha 모델은 플러그인(`examples/configs/alpha/vllm_alpha_plugin/`)으로만 등록한다. `3rdparty/vllm/` 은 커스텀 빌드용
  untracked 체크아웃이며 커밋하지 않는다.
- **uv lock 재생성은 uv 0.11.28 로만 한다** (`docker/Dockerfile` 의 `UV_VERSION` 핀). 신버전은 editable 메타데이터를 지워
  Ray 워커 venv 가 깨진다(`KNOWN_ISSUES` #9). pyproject 를 바꿨으면 다음 실행에 `NRL_FORCE_REBUILD_VENVS=true`.
