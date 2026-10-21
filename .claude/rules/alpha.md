# alpha_v2 RL 작업 규칙 (이 fork 의 alpha 작업 전용)

이 파일은 `paths` 가 없어 **매 세션 로드된다.** 루트 `CLAUDE.md`(= upstream `AGENTS.md`)는 upstream 소유라 수정하지 않고,
alpha 지침은 여기와 [`examples/configs/alpha/CLAUDE.md`](../../examples/configs/alpha/CLAUDE.md) 에 둔다 (2026-10-07 이관).

## 이 리포의 위치

- `potatowarriors/RL`(public) = upstream `NVIDIA-NeMo/RL` 의 fork. alpha 작업 브랜치는 **`alpha/post-train`**.
- alpha_v2(15.08B GDN+Attention+MoE 하이브리드)의 **RL 단계 전부**를 여기서 한다 (사용자 지시 2026-10-06).
  pre-train·LC·SFT·벤치 스위트는 Pai 스택에 남는다. 두 스택의 인터페이스는 **HF 체크포인트(`hfmodel_*`) 하나**뿐이다.
- 경로 표기: `project_s` = `/home/work/vidsearch/repos/project_s`, Pai = `project_s/Pai-Megatron-Patch`,
  `$NRL_ROOT` = `/home/work/vidsearch/tools/nemo_rl` (NFS 영속 설치·게이트 산출물).

## 문서 지도 — 어디에 무엇을 쓰나

| 무엇 | 정본 |
|---|---|
| 지금 상태 · 다음 할 일 · 열린 결정 | [`examples/configs/alpha/docs/STATUS.md`](../../examples/configs/alpha/docs/STATUS.md) |
| 문서 색인 (새 문서는 여기 한 줄 등록) | [`examples/configs/alpha/docs/README.md`](../../examples/configs/alpha/docs/README.md) |
| 사고·수정 서사 | [`examples/configs/alpha/docs/KNOWN_ISSUES.md`](../../examples/configs/alpha/docs/KNOWN_ISSUES.md) |
| 모델 불변량 · 레시피 기본값 · 함정 표 | [`examples/configs/alpha/CLAUDE.md`](../../examples/configs/alpha/CLAUDE.md) |
| 검증 게이트 정의·결과 | [`examples/configs/alpha/docs/GATES.md`](../../examples/configs/alpha/docs/GATES.md) |
| 작업 유형별 문서 지도 (훅이 요구하는 skill) | `alpha-rl` skill — [`examples/configs/alpha/skills/alpha-rl/SKILL.md`](../../examples/configs/alpha/skills/alpha-rl/SKILL.md) |
| upstream 문서·서브모듈 skill 링크 | [`examples/configs/alpha/docs/README.md`](../../examples/configs/alpha/docs/README.md) "upstream 참조" |
| Pai 쪽 (SFT·벤치·모델 정본) | `project_s/Pai-Megatron-Patch/examples/alpha/docs/README.md` |

## upstream 경계

- **upstream 문서·설정의 기존 내용은 고치지 않는다**: `AGENTS.md`·`CLAUDE.md`·`docs/`·`.claude/settings.json`·`.claude/skills/` 기존 항목.
  alpha 문서는 `examples/configs/alpha/` 와 `.claude/rules/alpha*.md` 에만 둔다 — rebase 충돌을 0 으로 유지. 예외는 둘이다.
  - **upstream 커밋 이식은 `docs/` 변경까지 원형대로 가져온다.** 문서만 빼면 코드와 문서가 어긋나고, 문서를 지운 커밋이
    rebase 뒤에도 남아 upstream 문서를 지운다 (2026-10-08 `0caa1696f` 를 `2579140ab` 로 되돌림).
  - **`.claude/skills/` 에는 `alpha-*` 이름의 새 링크만 추가한다.** 본문은 `examples/configs/alpha/skills/` 에 둔다.
- **본체(`nemo_rl/`·`tools/`·`tests/`)는 최소 수정.** 먼저 "수정을 소유한 레이어"(Megatron-Bridge fork·vLLM 플러그인·
  레시피 yaml)에 둘 수 있는지 본다. 본체여야 하면 alpha 문서·레시피와 **별도 커밋**으로 나누고 메시지에 rebase 주의를 적는다
  (선례 `1eca2f383` refit_verifier). v0.7/r0.8 rebase 때 이 커밋들이 충돌 후보다.
- upstream 훅(UserPromptSubmit)이 매 프롬프트 "skill 먼저 호출"을 요구한다. alpha 작업이면 **`alpha-rl` skill** 을 호출한다 —
  작업 유형별로 읽을 alpha 문서·upstream 문서·서브모듈 skill 의 지도다. upstream skill 은 아래 적용표를 따른다.

## upstream skill 적용표 (2026-10-08 · 끄기 설정 2026-10-21)

| 분류 | skill | alpha 에서 |
|---|---|---|
| 적용 | `config-conventions`·`error-handling`·`linting-and-formatting`·`testing`(Ray pragma)·`review-pr`(로컬 브랜치 모드) | 본체 수정·upstream 이식·본체 패치 검토 때 |
| 적용 | `contributing` | Conventional Commits + `git commit -s`. `/ok to test`(NVIDIA CI)만 무관 |
| 일부 | `nemo-rl-docs` | docstring 규칙만. "새 기능 문서는 `docs/` 에"는 alpha 기능에 적용하지 않는다 |
| 일부 | `build-and-dependency` | 참고만. uv 0.11.28 고정·`clean_run.sh` 가 우선 (`alpha-submodules.md`·alpha `docs/SETUP.md`) |
| 쓰지 않음 | `copyright` | alpha 고유 파일(`examples/configs/alpha/**`)에는 NVIDIA 저작권 헤더를 달지 않는다 (사용자 결정 2026-10-08) |
| 쓰지 않음 | `nemo-rl-session-memory`·`nemo-rl-auto-research` | 상태는 `STATUS.md`, 실험 브랜치는 커밋 규칙 3 이 대신한다. `session/` 디렉토리를 만들지 않는다. 설정 `off` |
| 무관 | `launch-nemo-rl`(k8s)·`nemo-rl-brev-etiquette`(Brev)·`cicd`(NVIDIA CI)·`review-pr-team` | 설정 `user-invocable-only` — 모델은 고르지 않고 사용자가 `/이름` 으로만 부른다 |
| 설정 | `.claude/settings.local.json` 의 `skillOverrides` (git 미추적, 이 표가 정본) | rebase 로 들어온 새 upstream skill 은 기본으로 켜진다 → 이 표에 분류하고 설정에 반영한다 |

## 승인·검증·보고 — 공통 원칙 (2026-10-21 이관)

승인 · 독립 검토 · 검증 · 공유 워킹트리 git · 기록 · 보고 문체는 사용자 공통 원칙 `/home/work/vidsearch/.claude-config/CLAUDE.md` 가 정본이다.
여기에는 alpha 고유 적용만 둔다.
- 체크포인트·레시피·브리지·플러그인을 바꾸면 `GATES.md` 의 재실행 조건을 따른다. 런 기동·재개·재시작은 `alpha-launch` skill 순서대로 한다.
- **세션 점검**: 사용자 수준 CLAUDE.md(`/home/work/vidsearch/.claude-config/<노드>/CLAUDE.md`)의 내용이 이 세션 컨텍스트에 없으면 `CLAUDE_CONFIG_DIR` 없이 뜬 세션이다. 작업 전에 사용자에게 알리고 `/home/work/vidsearch/.claude-config/CLAUDE.md` 를 Read 로 읽는다.

## 커밋·이력 규칙

1. 즉시 커밋·push 와 공유 워킹트리 규율은 공통 원칙을 따른다. 메시지는 `type(alpha): summary`
   (feat/fix/docs/chore/perf/experiment/test), `git commit -s` (upstream `contributing`).
2. **서브모듈 포인터**는 `.claude/rules/alpha-submodules.md` 를 따른다. 결정 전에는 포인터를 staging 하지 않는다.
3. 실험 브랜치는 `alpha/<topic>` 으로 만든다. 결론이 나면 — 기각이어도 — 기록 커밋으로 `alpha/post-train` 에 흡수하고 브랜치는 지운다
   (선례 `0fcf0d88b` Merge alpha/gym-0.6.0).

## 메모리·문서 규칙

1. **한 사실은 한 곳에** (공통 원칙). Pai 문서와 겹치는 사실(모델 상수·벤치 규약)은 Pai 가 정본이고, 여기는 RL 에 필요한 한 줄과 링크만 둔다.
2. **CLAUDE.md·rules 는 지침만.** 이 파일 ≤100줄, alpha `CLAUDE.md` ≤200줄. 사고 서사는 `KNOWN_ISSUES.md` 에 쓰고 함정 표에는 한 줄.
3. **상태는 `examples/configs/alpha/docs/STATUS.md` 에 커밋한다.**
4. **새 문서는 `docs/README.md` 에 한 줄 등록.** 새 "단일 진입점" 문서를 만들기 전에 기존 문서에 절을 추가할 수 있는지 본다.
5. **스테이지 경계마다 정리.** 끝난 트랙은 STATUS "완료" 절로 내리고, 두 스테이지 지난 사고는 `docs/archive/` 로 옮긴다.
