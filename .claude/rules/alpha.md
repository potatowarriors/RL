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
| Pai 쪽 (SFT·벤치·모델 정본) | `project_s/Pai-Megatron-Patch/examples/alpha/docs/README.md` |

## upstream 경계

- **수정 금지 (upstream 문서·설정)**: `AGENTS.md`·`CLAUDE.md`·`docs/`·`.claude/settings.json`·`.claude/skills/`.
  alpha 문서는 `examples/configs/alpha/` 와 `.claude/rules/alpha*.md` 에만 둔다 — rebase 충돌을 0 으로 유지.
- **본체(`nemo_rl/`·`tools/`·`tests/`)는 최소 수정.** 먼저 "수정을 소유한 레이어"(Megatron-Bridge fork·vLLM 플러그인·
  레시피 yaml)에 둘 수 있는지 본다. 본체여야 하면 alpha 문서·레시피와 **별도 커밋**으로 나누고 메시지에 rebase 주의를 적는다
  (선례 `1eca2f383` refit_verifier). v0.7/r0.8 rebase 때 이 커밋들이 충돌 후보다.
- upstream `.claude/settings.json` 의 UserPromptSubmit 훅이 매 프롬프트 "skill 먼저 호출"을 요구한다. upstream skill 은
  upstream 기여 작업용이다. alpha 작업에 맞는 skill 이 없으면 이 파일과 alpha `CLAUDE.md` 를 그 역할로 본다.

## 검증 규칙

검증을 거치지 않은 결과는 믿을 수 없다. 검증을 건너뛴 "성공"은 잠재 문제를 다음 단계로 넘기는 것이다.

1. **검증이 깨지면 검증을 고친다 — 절대 건너뛰지 않는다.** 환경 문제로 게이트가 실패하면 원인을 제거해 복원한다.
   우회 플래그·게이트 단계 주석 처리·임계 완화로 "해결"하지 않는다. 우회는 **이미 통과한 산출물 재사용**에만 쓴다.
2. **모든 작업은 검증을 완료한 뒤 진행한다.** 체크포인트·레시피·브리지·플러그인을 바꾸면 `GATES.md` 의 재실행 조건을 따른다.
   검증 수단이 없으면 먼저 방법(ON/OFF 통제 대조, 스모크, 수치 게이트)을 제시하고 완료한 뒤 다음으로 간다.
3. 보고에는 무엇을 어떻게 검증했고 어떤 수치가 나왔는지 그대로 적는다 (예: `14,181/14,181`, `KL 0.0015/0.0013/0.0014 PASS`).
   실패·부분 통과·미실행을 통과처럼 쓰지 않는다.

## 커밋·이력 규칙

1. **작업 단위 즉시 커밋 + 즉시 push** (`origin` = 유일한 백업). 메시지는 `type(alpha): summary`
   (feat/fix/docs/chore/perf/experiment/test).
2. **공유 워킹트리.** 여러 세션이 같은 NFS 워킹트리에서 동시에 일한다. 커밋은 항상 **명시 pathspec**(`git add <파일>`).
   `git add .`/`-A`/`commit -a` 금지. 커밋 전 `git status` 로 staged 에 내 파일만 있는지 확인한다.
   `git reset` 류 HEAD 이동 금지 — 잘못 커밋했으면 revert 나 수정 커밋으로 고친다. 남의 untracked 파일은 건드리지 않는다.
3. **서브모듈 포인터**는 `.claude/rules/alpha-submodules.md` 를 따른다. 결정 전에는 포인터를 staging 하지 않는다.
4. 실험 브랜치는 `alpha/<topic>` 으로 만든다. 결론이 나면 — 기각이어도 — 기록 커밋으로 `alpha/post-train` 에 흡수하고 브랜치는 지운다
   (선례 `0fcf0d88b` Merge alpha/gym-0.6.0).

## 메모리·문서 규칙

1. **한 사실은 한 곳에.** 우선순위는 docs > alpha `CLAUDE.md`·rules > auto-memory 다. 다른 곳에는 링크만 둔다.
   Pai 문서와 겹치는 사실(모델 상수·벤치 규약)은 Pai 가 정본이고, 여기는 RL 에 필요한 한 줄과 링크만 둔다.
2. **CLAUDE.md·rules 는 지침만.** 이 파일 ≤100줄, alpha `CLAUDE.md` ≤200줄. 사고 서사는 `KNOWN_ISSUES.md` 에 쓰고 함정 표에는 한 줄.
   상대 시점("이번 세션") 금지, 날짜는 절대 표기.
3. **상태는 `STATUS.md` 에 커밋한다.** auto-memory 에 진행 상태를 쓰지 않는다 — 메모리는 노드별이라 다른 세션이 못 본다.
4. **새 문서는 `docs/README.md` 에 한 줄 등록.** 새 "단일 진입점" 문서를 만들기 전에 기존 문서에 절을 추가할 수 있는지 본다.
5. **스테이지 경계마다 정리.** 끝난 트랙은 STATUS "완료" 절로 내리고, 두 스테이지 지난 사고는 `docs/archive/` 로 옮긴다.

## 보고 문체

- 결론을 먼저 쓰고 근거는 뒤에 쓴다. 한 문장에는 한 주장만 담는다. 괄호 속 부연은 최소화하고 수치 나열은 표로 만든다.
- 긴 합성 명사구는 풀어 쓴다. 보고는 의사결정 자료다.
