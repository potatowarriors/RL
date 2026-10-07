# alpha RL 현재 상태판

**규칙**: 세션 종료 시 자기 트랙의 행을 갱신하고 **커밋·push** 한다. 상태는 여기에만 쓴다 — auto-memory 에 쓰지 않는다
(메모리는 노드별이라 다른 세션이 못 본다). 날짜는 절대 표기. 끝난 트랙은 "완료" 절로 내리고 정본 링크만 남긴다.
게이트 수치는 [`GATES.md`](GATES.md) 에 쓰고 여기에는 판정과 링크만 쓴다. Pai 쪽(SFT·벤치) 상태는 Pai `examples/alpha/docs/STATUS.md`.

_마지막 갱신: 2026-10-07 13:39 (RL 준비 — M5·R3·R4 게이트 결과, Gym 멀티턴 턴 경계 수정, 메모리 상한·ES 결정 대기)_

## 진행 중

| 트랙 | 상태 | 다음 할 일 | 정본 |
|---|---|---|---|
| **RL 준비 (main1·sub1, 2026-10-06~, 사용자 지시: RL 관련 작업은 전부 이 트랙)** | **환경**: Pai 컨테이너 안에서 시스템 무변경으로 구동한다(원장 #15~#27). 영속 설치는 NFS `$NRL_ROOT`(셋업 `project_s/setup_nemo_rl_env_noninvasive.sh`). 사용자 결정 10-06: **원본 NFS·venv 로컬**. **게이트 (agentic iter2400)**: M1 라운드트립 · M2 forward 패리티 · M3 refit · R1 KL(R3 + GDN 상태 fp32, Adam·Muon) · R2 Muon 실적용 · E4 Gym v0.6.0 전부 PASS. **10-07 추가**: M5 SFT↔RL 엔진 동등성 **PASS** · R3 렌더 — 네이티브 GRPO PASS, Gym 멀티턴 턴 경계 결함 수정(`turn_end_token_id: 3`), `strict` 유지 미구현 · R4 메모리 — 1노드 **128K/CP8 OK**(`expandable_segments` 조건), Muon 오프로드 불필요. 학습 엔진 방향 = NeMo-RL mcore + Pai 기능 포팅(`RL_PLAN.md` 결정 11). 버전: v0.7.0 미채택, r0.8.0 정식 릴리스 시 재검토 | ① `strict` 유지 구현(Gym 플러그인 서버) → R3 P6 실경로 재실행 ② ES 결정(아래) 뒤 sub1 GRPO 스모크로 refit 시간·R1 재확인 ③ packing+CP 경로 KL 게이트(장문맥 수치, CP 는 packing 필수) ④ 셋업 재구성 승인 → 스냅샷 빌드·검증(Gym GPU 테스트 2건 포함) ⑤ 2노드 Ray 스모크 ⑥ PivotRL | `GATES.md`, `SETUP.md`, `KNOWN_ISSUES.md` 2026-10-07 항목 |

## 선행 의존 (Pai 스택)

| 항목 | 상태 (Pai STATUS 기준) | RL 쪽 영향 |
|---|---|---|
| agentic SFT 본 런 → 최종 ckpt iter2862 · 벤치 체인 | Pai `STATUS.md` "agentic 최종 iter2862 벤치" 행 | 완주 뒤 짧은 decay 여부 결정 → agentic base 확정 → PivotRL 시작점. 새 ckpt 는 `GATES.md` 재실행 조건(M1~M3 → R1)을 따른다 |

## 열린 사용자 결정

| 결정 | 선택지 · 권고 | 정본 |
|---|---|---|
| Megatron-Bridge 포인터를 fork(`bcc4e415`)로 고정할지 | 현재 superproject 기록은 upstream `0c565c9a` — 새 클론은 AlphaBridge 가 없다 | `.claude/rules/alpha-submodules.md` |
| 셋업 재구성 제안 | 로컬 venv 스냅샷 · 인터프리터 로컬 · 바이트코드 사전 컴파일 · JIT 캐시 로컬 (2026-10-06 제안, 승인 대기) | `SETUP.md` §1 |
| RL 학습 워커에 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 를 기본값으로 둘지 | 권고: 정책 워커에만 켠다. 켜기 전에 sub1 GRPO 스모크로 refit 시간과 R1 KL 을 확인한다. 끄면 1노드 상한은 rank 당 12K 토큰(96K/CP8)이다 | `KNOWN_ISSUES.md` 2026-10-07 메모리 항목 |
| 본 RL 레시피의 lr·wd·clip·라우터 동결 유지 | 스모크 레시피 값은 KL 게이트용이다. SFT 의 clip 8.0·lr 1e-5 는 loss 정규화가 달라 그대로 못 옮긴다 | `grpo_alpha_smoke_muon.yaml` 헤더, `RL_PLAN.md` §2 승계 범위 |
| MOPD 재현 범위 · 장문 컨텍스트 캡 | 교사 슬롯 2~3 vs 5 · SWE 교사·MOPD 192k → 128k 캡 또는 SWE 슬롯 축소 · NeMo RL/Gym 스택 포팅 vs 자체 구현(verl/ChatLearn 백엔드 검토) | `RL_PLAN.md` §4 |
| ARC-AGI-v1 사용 | 라이선스 `pending-legal-review` — 블렌드 내 nvarc 행(각 ~2%) 사용 전 법무 확인 | `RL_DATA.md` §3 |

## 보류·추적

| 항목 | 판정 | 재평가 시점 |
|---|---|---|
| FlashQLA E2E 스텝 이득 | 커널 단위 검증만 완료(K1·M2). 레시피 opt-in | 첫 장문 RL 런 |
| vLLM 롤아웃측 FlashQLA | prefill 한정 이득, vllm 코드 변경 필요 — 후순위 | 롤아웃이 병목일 때 |
| fla 0.4.2 네이티브 GQA NaN upstream 보고 | 2026-08-18 "보고 예정" 기록 뒤 실행 여부 미기록 (mcore 는 MHA 확장으로 우회 중, 원장 #14) | — |
| RL 데이터 후속 4건 (Gym 프리페치·Math-v2 복원·SWE 상한·litmus-bench) | 미착수 | `RL_DATA.md` §4 |
| Muon optimizer-state offload 포팅 (Pai 기능 #4) | R4 실측상 1노드 128K/CP8 에 불필요 (ES 조건) — 보류 | 컨텍스트 > 128K 이거나 ES 기각 시 |
| logprob backward 메모리 절감 (int64 `one_hot` → scatter, chunk 축소, CP 호환 fused linear logprobs) | ES 로 128K 가 들어가 불필요 — 보류. chunk 1024 는 peak −3.8 GB 실측 | ES 기각 시 |
| RL 환경 데이터의 도구 `description` 필수 검사 | 현 블렌드 0건이라 미착수 (R3 렌더 차이 #2) | 새 RL 환경 추가 시 |

## 완료

| 트랙 | 결과 | 정본 |
|---|---|---|
| RL 인프라 준비 (2026-08-12 ~ 08-21) | fork 3종 · AlphaBridge · vLLM 플러그인 · 게이트 M1~M3·K1·D1 · RL 블렌드 2종 · FlashQLA opt-in 통합 · 2노드 분리 토폴로지 확정 | `archive/POSTTRAIN_PREP_2026-08.md` |
| vLLM 단독 서빙 정량 게이트 (2026-08-24) | M4 PASS | `GATES.md` M4 |
| Pai 컨테이너 공존 설치 (2026-10-06) | 시스템 무변경 구동, 원장 #15~#20 해결, E1~E3 PASS | `SETUP.md` §1 |
