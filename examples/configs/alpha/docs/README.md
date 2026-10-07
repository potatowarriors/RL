# alpha RL 문서 색인

**규칙**: 새 문서는 여기 **한 줄**로 등록한다. `CLAUDE.md` 에는 문서 요약을 쓰지 않는다.
현재 상태는 [`STATUS.md`](STATUS.md) 한 곳에만 쓴다. 사고 서사는 [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) 에 쓰고 `../CLAUDE.md` 함정 표에는 한 줄만.
다른 리포 경로는 `project_s/…` 로 적는다 (`project_s` = `/home/work/vidsearch/repos/project_s`).
2026-10-07 문서 체계 신설 — 워크스페이스 `NEMO_RL_SETUP.md`·`ALPHA_POSTTRAIN_PROGRESS.md` 와 Pai 의 RL 관련 절을 이관했다.

## 지금 볼 것

| 문서 | 한 줄 |
|---|---|
| [STATUS.md](STATUS.md) | RL 트랙 상태 · 다음 할 일 · 열린 결정 · 선행 의존(Pai). 세션 종료 시 갱신·커밋 |
| [KNOWN_ISSUES.md](KNOWN_ISSUES.md) | 사고·수정 기록 — 장문맥 OOM(단편화), Gym 멀티턴 턴 경계, Gym 렌더 차이(`strict` 등), 엔진 동등성 판정법, GDN 상태 bf16(KL 게이트), zero-centered RMSNorm, 환경·운영 원장 #1~#27 |
| [GATES.md](GATES.md) | 검증 게이트 정본 — 재실행 조건 · M1~M5 모델 정합성(M5 = SFT↔RL 엔진 동등성) · R1~R4 RL 실행(R3 렌더 패리티, R4 학습 메모리) · K1·D1 · E1~E4 환경 |

## 가이드·설계 (현행)

| 문서 | 한 줄 |
|---|---|
| [SETUP.md](SETUP.md) | 환경 구축·운영 — Pai 컨테이너 공존 설치(기본), 셋업 재구성 제안(§1.1, 승인 대기), 환경 구조, 실행, RL 전용 세션 설치(구), 재생성 복원 |
| [RL_PLAN.md](RL_PLAN.md) | 단계(PivotRL → RLVR → 교사 RL → MOPD) · 설계 결정 13건 · SFT→RL 승계 범위 · 2노드 분리 토폴로지 · alpha 규모 적용 설계 · 평가 |
| [RL_DATA.md](RL_DATA.md) | RL 데이터 원천 26종·Ultra 블렌드 · 준비된 alpha 블렌드(identity 0.70%) · 주의 · 미결 |

## 포팅 명세 (2026-08-13 동결)

| 문서 | 한 줄 |
|---|---|
| [SPEC_pai_alpha_conversion.md](SPEC_pai_alpha_conversion.md) | Pai HF↔Megatron 변환·검증 방법 — 아키텍처 상수, 전체 가중치 매핑, 브리지로 옮길 함정 |
| [SPEC_megatron_bridge_surface.md](SPEC_megatron_bridge_surface.md) | Megatron-Bridge 등록 메커니즘·Qwen3-Next 브리지·DSV3 라우터 매핑 — AlphaBridge 구현 표면 |
| [SPEC_nemo_rl_env_wiring.md](SPEC_nemo_rl_env_wiring.md) | NeMo-RL 보상 환경 → GRPO 배선 · 데이터 파이프라인 · Gym 통합 · Ultra 레시피 형태 |
| [SPEC_rl_dataset_inventory.md](SPEC_rl_dataset_inventory.md) | RL 데이터셋 전수 인벤토리 — 스키마 샘플·스테이지 매핑·라이선스·identity 셋 |

## study/ · archive/

| 문서 | 한 줄 |
|---|---|
| [study/pivotrl_study.md](study/pivotrl_study.md) | PivotRL 스터디 — 방법·Ultra 사용처·공개 자산 실사·agentic SFT 이후 적용 설계·게이트 (2026-09-22, Pai 에서 이관) |
| [archive/POSTTRAIN_PREP_2026-08.md](archive/POSTTRAIN_PREP_2026-08.md) | [동결] RL 인프라 준비 종합 보고 — 타임라인·결정·게이트·산출물 맵 (2026-08-12~21) |

## 이 리포 밖 (Pai 스택 — 링크만)

| 문서 | 한 줄 |
|---|---|
| `project_s/Pai-Megatron-Patch/examples/alpha/docs/README.md` | Pai alpha 문서 색인 (pre-train·LC·SFT·벤치) |
| `project_s/Pai-Megatron-Patch/examples/alpha/docs/STATUS.md` | Pai 상태판 — agentic SFT·벤치 체인 (RL 시작점 ckpt 의 출처) |
| `project_s/Pai-Megatron-Patch/examples/alpha/docs/SFT_BENCHMARKS.md` | 벤치 스위트 정본 — RL ckpt 평가에도 그대로 사용 |
| `project_s/Pai-Megatron-Patch/examples/alpha/docs/SFT_RL_DATASETS.md` | SFT 데이터 자산·블렌드 (§2). RL 절은 여기 `RL_DATA.md`·`RL_PLAN.md` 로 이관됨 |
| `project_s/Pai-Megatron-Patch/examples/alpha/study/nemo_gym_assessment.md` · `examples/alpha/gym/README.md` | NeMo-Gym 채택 분석(2026-09-15)과 Pai 쪽 Gym 설정·게이트 |
| `project_s/RESTORE_AFTER_REBOOT.md` | 컨테이너 재생성 후 전체 복원 runbook (노드·GPU·GitHub 인증) |
