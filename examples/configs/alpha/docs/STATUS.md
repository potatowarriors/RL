# alpha RL 현재 상태판

**규칙**: 세션 종료 시 자기 트랙의 행을 갱신하고 **커밋·push** 한다. 상태는 여기에만 쓴다 — auto-memory 에 쓰지 않는다
(메모리는 노드별이라 다른 세션이 못 본다). 날짜는 절대 표기. 끝난 트랙은 "완료" 절로 내리고 정본 링크만 남긴다.
게이트 수치는 [`GATES.md`](GATES.md) 에 쓰고 여기에는 판정과 링크만 쓴다. Pai 쪽(SFT·벤치) 상태는 Pai `examples/alpha/docs/STATUS.md`.

_마지막 갱신: 2026-10-07 (계획 재편 — 정확한 RLVR 런이 첫 목표, 속도 최적화는 그 뒤. G0 완료·G1 PASS·R5 부분 실측)_

## 진행 중

| 트랙 | 상태 | 다음 할 일 | 정본 |
|---|---|---|---|
| **RL 준비 (main1·sub1, 2026-10-06~, 사용자 지시: RL 관련 작업은 전부 이 트랙)** | **환경**: Pai 컨테이너 안에서 시스템 무변경으로 구동한다(원장 #15~#27). 영속 설치는 NFS `$NRL_ROOT`(셋업 `project_s/setup_nemo_rl_env_noninvasive.sh`). 사용자 결정 10-06: **원본 NFS·venv 로컬**. **게이트 (agentic iter2400)**: M1 라운드트립 · M2 forward 패리티 · M3 refit · R1 KL(R3 + GDN 상태 fp32, Adam·Muon) · R2 Muon 실적용 · E4 Gym v0.6.0 전부 PASS. **10-07 추가**: M5 SFT↔RL 엔진 동등성 **PASS** · R3 렌더 — 네이티브 GRPO PASS, Gym 멀티턴 턴 경계 결함 수정(`turn_end_token_id: 3`), `strict` 유지 미구현 · R4 메모리 — 1노드 **128K/CP8 OK**(`expandable_segments` 조건), Muon 오프로드 불필요. 학습 엔진 방향 = NeMo-RL mcore + Pai 기능 포팅(`RL_PLAN.md` 결정 11). 버전: v0.7.0 미채택, r0.8.0 정식 릴리스 시 재검토 | **RLVR — 1단계 정확성 → 2단계 첫 RLVR 런(병목 측정) → 3단계 속도 (사용자 지시 2026-10-07 재편, 정본 `RLVR_READINESS.md` §6)**: ✅ G0 레시피 골격·검사 도구 · ✅ G1 packing 상태 격리 PASS · ⏸ R5 recompute 실측은 기준선만 남기고 연기(128K full 33.4K tok/s, selective OOM). 다음: ① G2 packing+CP8 경로 R1 + R3 trace ② G3 Gym 파서 스모크 → `strict` 유지 구현 → R3 P6 ③ D1 결정 → G5 2노드 실레시피 스모크 → G6·G7 ④ 첫 RLVR 런 — 정확성 지표 + `exposed_generation`·`policy_training` 등으로 병목 판정 ⑤ 병목 쪽 속도 최적화. 기존 항목: 셋업 재구성 승인 대기 · PivotRL 위치 재검토 (`RL_PLAN.md` 결정 14) |

## 선행 의존 (Pai 스택)

| 항목 | 상태 (Pai STATUS 기준) | RL 쪽 영향 |
|---|---|---|
| agentic SFT 본 런 → 최종 ckpt iter2862 · 벤치 체인 | Pai `STATUS.md` "agentic 최종 iter2862 벤치" 행 | 완주 뒤 짧은 decay 여부 결정 → agentic base 확정 → PivotRL 시작점. 새 ckpt 는 `GATES.md` 재실행 조건(M1~M3 → R1)을 따른다 |

## 열린 사용자 결정

| 결정 | 선택지 · 권고 | 정본 |
|---|---|---|
| Megatron-Bridge 포인터를 fork(`bcc4e415`)로 고정할지 | 현재 superproject 기록은 upstream `0c565c9a` — 새 클론은 AlphaBridge 가 없다 | `.claude/rules/alpha-submodules.md` |
| 셋업 재구성 제안 | 로컬 venv 스냅샷 · 인터프리터 로컬 · 바이트코드 사전 컴파일 · JIT 캐시 로컬 (2026-10-06 제안, 승인 대기) | `SETUP.md` §1 |
| RL 학습 워커에 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 를 기본값으로 둘지 (D6) | 권고: 정책 워커에만 켠다. Pai SFT 는 항상 켰고 "refit 5배" 근거는 측정 없는 FP8 docstring 이다. G5 에서 refit 시간을 확인한다. 끄면 1노드 상한은 rank 당 12K 토큰(96K/CP8)이다 | `KNOWN_ISSUES.md` 2026-10-07 메모리 항목 |
| RLVR1 블렌드의 judge·sandbox 의존 행 23.4% + nvarc 4.2% (D1) | 블렌드에서 제외 / node1 일부를 judge 로 / 외부 API. identity 689행도 GenRM 채점 | `RLVR_READINESS.md` H4 |
| vLLM prefix caching (D2) | 끈다(upstream R3 레시피) / 켜고 R3 route 누락을 따로 검증. NeMo-RL 은 키가 없으면 켠다 | `RLVR_READINESS.md` M1 |
| RL 부하 균형 (D3) · KL 계수 (D4) · 길이 단계 (D7) | bias 갱신 0 유지 vs 1e-3(Ultra) · KL 0 + `seq_logprob_error_threshold: 2`(Ultra) vs 0.01(상속) · 128K 바로 vs 단계 상향 | `RLVR_READINESS.md` §7 |
| 본 RL 레시피의 lr·warmup·라우터 동결 유지 (D5) | 스모크 레시피 값은 KL 게이트용이다. clip 은 결정됨에 가깝다: SFT 8.0@CP8 = NeMo-RL 1.0 (Pai grad norm 은 CP 배수, NeMo-RL 은 CP 무관 — `RLVR_READINESS.md` §2). 스케줄 단위는 GRPO step | `grpo_alpha_smoke_muon.yaml` 헤더, `RL_PLAN.md` §2 승계 범위 |
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
