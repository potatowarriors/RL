# alpha RL 현재 상태판

**규칙**: 세션 종료 시 자기 트랙의 행을 갱신하고 **커밋·push** 한다. 상태는 여기에만 쓴다 — auto-memory 에 쓰지 않는다
(메모리는 노드별이라 다른 세션이 못 본다). 날짜는 절대 표기. 끝난 트랙은 "완료" 절로 내리고 정본 링크만 남긴다.
게이트 수치는 [`GATES.md`](GATES.md) 에 쓰고 여기에는 판정과 링크만 쓴다. Pai 쪽(SFT·벤치) 상태는 Pai `examples/alpha/docs/STATUS.md`.

_마지막 갱신: 2026-10-08 (첫 RLVR 런 병목 판정 · 가속 검토 · 정확성 결함 4건 이식 · 재시작 결정)_

## 진행 중

| 트랙 | 상태 | 다음 할 일 | 정본 |
|---|---|---|---|
| **RL 준비 (main1·sub1, 2026-10-06~, 사용자 지시: RL 관련 작업은 전부 이 트랙)** | **환경**: Pai 컨테이너 안에서 시스템 무변경으로 구동한다(원장 #15~#27). 영속 설치는 NFS `$NRL_ROOT`(셋업 `project_s/setup_nemo_rl_env_noninvasive.sh`). 사용자 결정 10-06: **원본 NFS·venv 로컬**. **게이트 (agentic iter2400)**: M1 라운드트립 · M2 forward 패리티 · M3 refit · R1 KL(R3 + GDN 상태 fp32, Adam·Muon) · R2 Muon 실적용 · E4 Gym v0.6.0 전부 PASS. **10-07 추가**: M5 SFT↔RL 엔진 동등성 **PASS** · R3 렌더 — 네이티브 GRPO PASS, Gym 멀티턴 턴 경계 결함 수정(`turn_end_token_id: 3`), `strict` 유지 미구현 · R4 메모리 — 1노드 **128K/CP8 OK**(`expandable_segments` 조건), Muon 오프로드 불필요. 학습 엔진 방향 = NeMo-RL mcore + Pai 기능 포팅(`RL_PLAN.md` 결정 11). 버전: v0.7.0 미채택, r0.8.0 정식 릴리스 시 재검토 | **RLVR — 1단계 정확성 → 2단계 첫 RLVR 런(병목 측정) → 3단계 속도 (사용자 지시 2026-10-07 재편, 정본 `RLVR_READINESS.md` §6)**: **1단계 정확성 게이트 완료 (2026-10-07)**: ✅ G0 · ✅ G1 · ✅ G2 (4K·32K) · ✅ G3 (수정 5건) · ✅ G5 (2노드, 동기 저장) · ✅ G6 (반출은 `tools/export_rl_hf.sh` 로만) · ✅ G7 (런 연장 재개, `scheduler.max_steps` 수정) — 수치는 `GATES.md`. ⏸ R5 recompute 실측은 기준선만 남기고 연기(128K full 33.4K tok/s, selective OOM). **2단계 첫 RLVR 런 진행 중 (2026-10-07~)**: `$NRL_ROOT/runs/rlvr1_alpha/` (`run_rlvr1.sh`, 체크포인트 `ckpt/`, HF 반출 `hf/`). 구간 1 = 10스텝(학습 데이터 덤프·R3 trace) → 구간 2 = 재개로 1에폭까지(덤프 끔). **중간 결과 (1~4스텝, 2026-10-08 00:05)**: 정합 — KL 0.0019/0.0019/0.0019/0.0017, 마스킹 0, 위치별 k3 [0,256) 0.0013~0.0014 → [32K,64K) 0.0020 (서서히 오름, 지켜볼 항목), 잘림 3.1~3.4%, 페널티로 보상 0 = think 형식 7.4~9.4% · 빈 답 2.7~3.2%. **병목 판정 (5스텝 뒤 수정)**: 정상 상태(3~5스텝) 스텝 ≈ 31분 중 롤아웃 대기 46% · 학습 35% · logprob 16% · 동기화 2%. 매 배치에 128K 생성 샘플이 있어 배치 완성이 45~60분에 묶이고, `max_trajectory_age_steps 1` 이라 대기가 격스텝으로 생긴다 (5스텝 29분). 생성 노드는 반대쪽 스텝에서 10~20분씩 유휴. 이 속도면 1에폭 ≈ 24일. (1~4스텝만 본 첫 판정은 '학습 74%' 였다) 수치는 `RLVR_READINESS.md` §5. **가속 검토·재시작 결정 (2026-10-08, `RL_PLAN.md` 결정 16)**: 검토 중 첫 런의 정확성 결함 4건을 찾았다 (effort shaping 누락 등). 구간 1 은 7스텝에서 오류 없이 6.5시간 멈춰(in-flight 요청 67개가 vLLM 엔진 1개에서 진척 0) 07:40 사용자 결정으로 정지했다 (`KNOWN_ISSUES.md` 2026-10-08). upstream 이식 7건을 `alpha/post-train` 에 병합했다 (`4768471eb`): 결함 4건 + #4171(loss 안 마스킹) + vLLM HTTP 견고성 2건(멈춤 원인 후보). 단위 테스트·ruff 통과. 런마다 쌓이던 Gym 서버 132개를 정리했고 v2 실행기가 기동 전에 정리한다. **E0 = G8 PASS (08:18)**: 이식·64K·loss 안 마스킹·두 레버의 기능 확인 — KL 0.0020/0.0020/0.0019, 마스킹 0, logprob 단계 0 s (`GATES.md` G8). **진행 중**: E1 레버 A/B (`runs/rlvr1_alpha_v2/run_e1.sh`, 08:29~) — 본 런 규모(64 × 16, 64K)로 5스텝씩, e1a 레버 켬 → e1b 끔 (arm 당 상한 85분). 다음: 레버 판정 → v2 본 런 처음부터. 기존 항목: 셋업 재구성 승인 대기 · PivotRL 위치 재검토 (`RL_PLAN.md` 결정 14) |

### 첫 RLVR 런 저장 용량 (2026-10-07 결정, `$NRL_ROOT/runs/rlvr1_alpha/`)

| 항목 | 주기 | 1회 크기 | 보존 | 사용량 |
|---|---|---|---|---|
| 학습 체크포인트 (모델 + 옵티마이저, `ckpt/`) | 10스텝 | 155 GB (실측) | 최근 2개 | 평소 310 GB · 저장 중 465 GB (새 것을 다 쓴 뒤 오래된 것을 지운다) |
| HF 반출본 (평가·롤백, `hf/`) | 100스텝 | 30 GB (실측) | 모두 | 1에폭(≈ 1,120스텝)에 11개, 330 GB |
| 구간 1 학습 데이터 덤프 (`seg1/exp_*/`) | 첫 10스텝 | 스텝당 5~9 GB (추정) | 분석 기록 뒤 삭제 | 일시 50~90 GB |
| **합계** | | | | **1에폭 끝 ≈ 640 GB · 저장 중 최대 ≈ 795 GB** |

그 밖의 NFS 사용: HF→mcore 변환 캐시 31 GB(`mcore_ckpt/`, 런 기동에 필요), 게이트 15 GB(그중 13 GB 는 런이 쓰는 Gym venv).

## 선행 의존 (Pai 스택)

| 항목 | 상태 (Pai STATUS 기준) | RL 쪽 영향 |
|---|---|---|
| agentic SFT 본 런 → 최종 ckpt iter2862 · 벤치 체인 | Pai `STATUS.md` "agentic 최종 iter2862 벤치" 행 | 완주 뒤 짧은 decay 여부 결정 → agentic base 확정 → PivotRL 시작점. 새 ckpt 는 `GATES.md` 재실행 조건(M1~M3 → R1)을 따른다 |

## 열린 사용자 결정

| 결정 | 선택지 · 권고 | 정본 |
|---|---|---|
| ~~reward_penalties (첫 런)~~ **결정됨 2026-10-07 — 4종 켬** — Ultra 는 4종(중복 reasoning·빈 최종 답·금지 토큰·think 태그 형식)을 모두 켠다. 발동하면 보상 0 | 근거: G5 롤아웃 384개에 오프라인 적용 시 보상 > 0 인 83개 중 깎이는 것 0 — 발동(형식 4a 78·빈 답 65·형식 4b 11)은 전부 이미 보상 0 인 잘림·퇴화 응답 (`tools/analyze_reward_penalties.py`). alpha token id 는 레시피에 반영됨 | `RLVR_READINESS.md` H5 |
| ~~체크포인트 보존 (첫 런)~~ **결정됨 2026-10-07 (같은 날 축소)** — `keep_top_k 2`(최근 2개) + `tools/export_watch.sh` 로 100스텝마다 HF 반출(1회 30 GB, 모두 보존). 학습 데이터 덤프는 첫 10스텝만 켜고, 분석을 기록한 뒤 지운다. 게이트 산출물 삭제: 스모크 체크포인트 6개 930 GB · G6 HF 반출본 90 GB · M5 랭크별 덤프 9.4 GB. 용량 예산은 아래 "첫 RLVR 런 저장 용량" | 저장 1회 155 GB, 1에폭 ≈ 1,120 스텝, 전부 보존이면 ≈ 17 TB (NFS 여유 13 TB, 공용) | 이 표 |
| ~~D1·D2·D3·D4·D5·D7~~ | **결정됨 2026-10-07** — 첫 RLVR 런: judge 불필요 10개 환경 71,730행 · prefix caching 끔 · bias 갱신 0 · KL 0 + threshold 2 · lr 1e-6 warmup 10 · 128K | `RL_PLAN.md` 결정 15 |
| Megatron-Bridge 포인터를 fork(`bcc4e415`)로 고정할지 | 현재 superproject 기록은 upstream `0c565c9a` — 새 클론은 AlphaBridge 가 없다 | `.claude/rules/alpha-submodules.md` |
| 셋업 재구성 제안 | 로컬 venv 스냅샷 · 인터프리터 로컬 · 바이트코드 사전 컴파일 · JIT 캐시 로컬 (2026-10-06 제안, 승인 대기) | `SETUP.md` §1 |
| RL 학습 워커에 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 를 기본값으로 둘지 (D6) — **첫 런은 권고값(켬)으로 간다 (사용자 2026-10-07)** | 권고: 정책 워커에만 켠다. Pai SFT 는 항상 켰고 "refit 5배" 근거는 측정 없는 FP8 docstring 이다. G5 에서 refit 시간을 확인한다. 끄면 1노드 상한은 rank 당 12K 토큰(96K/CP8)이다 | `KNOWN_ISSUES.md` 2026-10-07 메모리 항목 |
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
