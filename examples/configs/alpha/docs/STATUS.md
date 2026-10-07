# alpha RL 현재 상태판

**규칙**: 세션 종료 시 자기 트랙의 행을 갱신하고 **커밋·push** 한다. 상태는 여기에만 쓴다 — auto-memory 에 쓰지 않는다
(메모리는 노드별이라 다른 세션이 못 본다). 날짜는 절대 표기. 끝난 트랙은 "완료" 절로 내리고 정본 링크만 남긴다.
게이트 수치는 [`GATES.md`](GATES.md) 에 쓰고 여기에는 판정과 링크만 쓴다. Pai 쪽(SFT·벤치) 상태는 Pai `examples/alpha/docs/STATUS.md`.

_마지막 갱신: 2026-10-07 12:50 (상태판 신설 — Pai `STATUS.md` 의 "NeMo-RL RL 준비" 행을 이관, 문서 체계 정리)_

## 진행 중

| 트랙 | 상태 | 다음 할 일 | 정본 |
|---|---|---|---|
| **RL 준비 (main1, 2026-10-06~, 사용자 지시: RL 관련 작업은 전부 이 트랙)** | **환경**: Pai 컨테이너 안에서 시스템 무변경으로 구동한다(원장 #15~#21). 영속 설치는 NFS `$NRL_ROOT`(셋업 `project_s/setup_nemo_rl_env_noninvasive.sh`). 사용자 결정 10-06: **원본 NFS·venv 로컬**. **게이트 전부 PASS (agentic iter2400)**: M1 라운드트립 · M2 forward 패리티 · M3 refit · R1 KL(기본 FAIL → R3 + GDN 상태 fp32 로 PASS, Adam·Muon 둘 다) · R2 Muon 실적용 · E4 Gym v0.6.0 머지·push(`0fcf0d88b`). RL 레시피 기본값 = fp32 상태 + R3. 버전: v0.7.0 미채택, r0.8.0 정식 릴리스 시 재검토 | 셋업 재구성 승인 → 새 lock 으로 스냅샷 빌드·검증(Gym GPU 테스트 2건 포함) → 2노드 Ray 스모크(sub1 해제 후) → PivotRL | `GATES.md`, `SETUP.md`, `KNOWN_ISSUES.md` 맨 위 |

## 선행 의존 (Pai 스택)

| 항목 | 상태 (Pai STATUS 기준) | RL 쪽 영향 |
|---|---|---|
| agentic SFT 본 런 → 최종 ckpt iter2862 · 벤치 체인 | Pai `STATUS.md` "agentic 최종 iter2862 벤치" 행 | 완주 뒤 짧은 decay 여부 결정 → agentic base 확정 → PivotRL 시작점. 새 ckpt 는 `GATES.md` 재실행 조건(M1~M3 → R1)을 따른다 |

## 열린 사용자 결정

| 결정 | 선택지 · 권고 | 정본 |
|---|---|---|
| Megatron-Bridge 포인터를 fork(`bcc4e415`)로 고정할지 | 현재 superproject 기록은 upstream `0c565c9a` — 새 클론은 AlphaBridge 가 없다 | `.claude/rules/alpha-submodules.md` |
| 셋업 재구성 제안 | 로컬 venv 스냅샷 · 인터프리터 로컬 · 바이트코드 사전 컴파일 · JIT 캐시 로컬 (2026-10-06 제안, 승인 대기) | `SETUP.md` §1 |
| 본 RL 레시피의 lr·wd·clip·router bias rate | 스모크 레시피 값은 KL 게이트용이다 | `grpo_alpha_smoke_muon.yaml` 헤더 |
| MOPD 재현 범위 · 장문 컨텍스트 캡 | 교사 슬롯 2~3 vs 5 · SWE 교사·MOPD 192k → 128k 캡 또는 SWE 슬롯 축소 · NeMo RL/Gym 스택 포팅 vs 자체 구현(verl/ChatLearn 백엔드 검토) | `RL_PLAN.md` §4 |
| ARC-AGI-v1 사용 | 라이선스 `pending-legal-review` — 블렌드 내 nvarc 행(각 ~2%) 사용 전 법무 확인 | `RL_DATA.md` §3 |

## 보류·추적

| 항목 | 판정 | 재평가 시점 |
|---|---|---|
| FlashQLA E2E 스텝 이득 | 커널 단위 검증만 완료(K1·M2). 레시피 opt-in | 첫 장문 RL 런 |
| vLLM 롤아웃측 FlashQLA | prefill 한정 이득, vllm 코드 변경 필요 — 후순위 | 롤아웃이 병목일 때 |
| fla 0.4.2 네이티브 GQA NaN upstream 보고 | 2026-08-18 "보고 예정" 기록 뒤 실행 여부 미기록 (mcore 는 MHA 확장으로 우회 중, 원장 #14) | — |
| RL 데이터 후속 4건 (Gym 프리페치·Math-v2 복원·SWE 상한·litmus-bench) | 미착수 | `RL_DATA.md` §4 |

## 완료

| 트랙 | 결과 | 정본 |
|---|---|---|
| RL 인프라 준비 (2026-08-12 ~ 08-21) | fork 3종 · AlphaBridge · vLLM 플러그인 · 게이트 M1~M3·K1·D1 · RL 블렌드 2종 · FlashQLA opt-in 통합 · 2노드 분리 토폴로지 확정 | `archive/POSTTRAIN_PREP_2026-08.md` |
| vLLM 단독 서빙 정량 게이트 (2026-08-24) | M4 PASS | `GATES.md` M4 |
| Pai 컨테이너 공존 설치 (2026-10-06) | 시스템 무변경 구동, 원장 #15~#20 해결, E1~E3 PASS | `SETUP.md` §1 |
