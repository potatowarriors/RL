# alpha RL 현재 상태판

**규칙**: 세션 종료 시 자기 트랙의 행을 갱신하고 **커밋·push** 한다. 상태는 여기에만 쓴다 — auto-memory 에 쓰지 않는다
(메모리는 노드별이라 다른 세션이 못 본다). 날짜는 절대 표기. 끝난 트랙은 "완료" 절로 내리고 정본 링크만 남긴다.
게이트 수치는 [`GATES.md`](GATES.md) 에 쓰고 여기에는 판정과 링크만 쓴다. Pai 쪽(SFT·벤치) 상태는 Pai `examples/alpha/docs/STATUS.md`.

_마지막 갱신: 2026-10-10 11:05 (도구 사용 teacher 완료 — 400/400 rc 0, 보상 150스텝 뒤 평탄, teacher 체크포인트 선택 미정)_

## 진행 중

| 트랙 | 상태 | 다음 할 일 | 정본 |
|---|---|---|---|
| **1차 teacher RLVR 4종 (2026-10-08~, `RL_PLAN.md` 결정 18·19)** | 사용자 결정 10-08: 속도 최적화를 멈추고 현 자원에 맞게 학습 규모를 줄인다. 순서는 SFT ckpt → teacher 3종(General = chat·IF · STEM · Code·Math) → MOPD1 → MOPD1 에서 agentic teacher(SWE·search·terminal 등) → MOPD2 (General·STEM teacher 재사용). **블렌드안 보고 (`RL_DATA.md` §5, 승인 대기)**: teacher 당 250 스텝 × 64 × 16, 생성 상한 STEM 32K · Code·Math 24K→48K · General 16K, P0 포함 4.6~6.6일 무중단 (추정). Ultra 블렌드는 alpha 에게 너무 어렵다 — 보상 > 0 code_gen 0.7% · math 5.1% (IF 17.7% · mcqa 43.8%) | **도구 사용 warm-up teacher 본 런 — ✅ 완료 2026-10-10 08:11 (400/400, rc 0, 학습 32.5 h; 시작 10-08 23:24 `lr3e6`, iter2862, lr 3e-6 `RL_PLAN.md` 결정 22)**: **최종 점검 (10-10)**: 보상 0.487 (1~50스텝) → 0.546 (101~150) → 0.558 (351~400) — 150스텝 이후 평탄하다 (151~400 추세 100스텝당 +0.007 ± 0.005). 정책은 끝까지 굳었다: 리플레이 버퍼 192그룹 기준 혼합 80% → 50% (step 210) → 37% (step 400), 항상 실패 9% → 23%, pass@16 0.91 → 0.77, entropy 0.765 → 0.672. gen KL 0.00118 · 마스킹 0 으로 정합은 끝까지 정상. HF 반출 100·200·300·400 모두 PASS (`hf/hf_step_00N00`, 텐서 14,181 · 오류 0 · 동결 변화 0), 반출 감시 루프는 멈췄다. 체크포인트 `ckpt/step_390`·`step_400` 이 남아 있다. GPU 16장은 08:11 부터 비어 있다 (완료 감시를 걸어 두지 않았다). **teacher 로 쓸 체크포인트는 미정** — 학습 보상은 150스텝 뒤 평탄하고 다양성은 계속 줄었다. 도구 벤치 (Pai BFCL v4 · τ³) 로 SFT 2862 와 step 200·400 을 비교해 고르는 안을 보고했다 (승인 대기, iter2862 의 τ³ 는 아직 없다). 이하 진행 기록: `RECIPE=teacher_tool_alpha.yaml` (`b3115c226`) · `teacher_tool_v1.jsonl` 25,600 (400 스텝) · 산출 `results/alpha/teacher_tool/` (`lr3e6/` · `ckpt/` · `hf/` 100 스텝마다 반출 `export_watch.sh`) · wandb `alpha-rl`/`teacher_tool_lr3e6`. 스텝 약 315 s 면 완료 ≈ 10-10 11:00. 두 런은 seed 42 로 프롬프트 순서가 같다 (1·2스텝 평균 프롬프트 길이 7360.55 · 7575.41 일치) — 보상은 1e-6 런의 같은 스텝과 짝지어 비교한다 (스텝당 차이의 잡음 약 0.02, 10스텝 평균 약 0.006). **step 50 검증 (10-09 03:43, `KNOWN_ISSUES.md` 2026-10-08 검증 결과)**: bf16 반영 계수 0.30~0.36 (예측 0.34, 1e-6 런 0.10~0.13) — lr 변경은 의도대로 작동한다. 정합은 1e-6 런과 같다 (gen KL 0.00135 · 마스킹 0 · entropy 0.764). 짝지은 보상 차이 (3e-6 − 1e-6) 11~50스텝 **+0.0088 ± 0.0028** — 유의하지만 작고 (상대 약 2%), 구간별 +0.012 · +0.001 · +0.012 · +0.010 으로 커지는 추세는 아직 없다. 환경별: swe +0.016 · tau +0.010 · schema +0.002. 1e-6 런 기록은 55스텝까지라 이후는 절대 보상 추세와 100스텝 반출 HF 로 본다. 스텝 평균 294 s (31~50). **200스텝 점검 (10-09 17:30, step 219)**: 보상 0.479 (1~25스텝) → 0.548 (176~218스텝), 추세 100스텝당 +0.032 ± 0.006 — 1 에폭이라 매 스텝 처음 보는 프롬프트에서 잰 값이다. 오름폭이 작은 이유는 정책이 프롬프트별로 굳는 것이다: 그룹 안 보상 표준편차 0.37 → 0.18 (TensorBoard), 리플레이 버퍼 192그룹 비교 (1e-6 런 step 50 → 이 런 step 210) 에서 혼합 그룹 80% → 50% · 항상 실패 9% → 20% · 항상 성공 11% → 30% · pass@16 0.91 → 0.80, entropy 0.77 → 0.71. 항상 실패 그룹은 대부분 정답 행동 하나와 다른 행동을 일관되게 고른다 (tau 21개 중 도구 없이 응답 12 · 다른 도구 5, swe_pivot 6개 모두 다른 도구). 정합·학습 경로 이상은 아니다. 남은 스텝은 학습 신호(혼합 그룹)가 절반이라 오름폭이 더 작을 것이다. step 100·200 HF 반출 PASS (텐서 14,181 · 오류 0). **lr 1e-6 런 `main` (18:28 ~ 23:23, 55스텝 완료 후 정지)**: 정합은 정상(KL 0.00134 · 마스킹 0)인데 보상이 평탄했다 (1~10스텝 = 31~40스텝 0.481). 학습 경로 점검은 모두 정상이고, 50스텝 누적 갱신의 약 12% 만 bf16 가중치에 반영됐다 (`KNOWN_ISSUES.md` 2026-10-08). 체크포인트 `ckpt_lr1e6_main/` (310 GB) 는 step 50 검증 뒤 지웠다 (사용자 승인 2026-10-09), W&B `teacher_tool_main` 은 남아 있다. 1~2스텝 처리량: 스텝 330~336 s (학습 82%, 학습 쪽이 병목). vLLM 생성 타임라인 그래프 8개는 `log_plot` 경로라 W&B 필터를 거치지 않는다 — `logger.py` 수정은 이번 재기동에도 들어가지 않았다 (실행 중 import 코드 편집 금지 #26). **W&B 정리 (사용자 승인 2026-10-08)**: 시험 런 5개를 지웠다. **judge 배치 A (사용자 결정 2026-10-08, `RL_PLAN.md` 결정 21)**: 롤아웃 노드 = 롤아웃 6장 + Gemma judge 2장. 본체 수정 `alpha/judge-split` `2807ef2be` — 새 단위 테스트 4건 PASS, `test_grpo.py` 전체는 변경 전후 같은 환경 실패 22건 (`KNOWN_ISSUES.md` 단위 테스트 항목). 도구 teacher 가 끝나면 (≈ 10-10 11:00, lr 3e-6 재시작으로 약 4시간 늦어짐) GPU 창에서 Gemma judge 검증(P0-4) · 2노드 스모크(자원 배치 · judge 보상 · 스텝 시간) · 코드 난이도 측정(P0-3) 을 하고, 스모크 PASS 뒤 병합해 STEM 을 시작한다. **STEM 준비 (2026-10-08, CPU 완료)**: 레시피 `teacher_stem_alpha.yaml` (lr 5e-6, 결정 23 `6d8f6de46` — 검사기 ERROR 0) · rdkit 채점 플러그인 (Gym 0.6.0 에 서버 없음) · 검사기 보강 (judge 참조 · judge GPU 예산 · 데이터 agent · ns_tools verifier) — `alpha/judge-split` `a38ad04bf`·`ab3bfe518`. 최종 후보 블렌드 52,500 + 검증 666 (D1 OK, **비율 승인 대기** — python 도구 과학 10% 를 넣었다, `RL_DATA.md` §5.4) · J1 스모크 256행 · P0-4 에 Science-v1 서술형 판정 검증 200쌍 추가. **P0 진행 (2026-10-08)**: ✅ iter2862 게이트 5종 PASS (R1 0.0014/0.0013/0.0013, `GATES.md`) · ✅ 수학 마스킹 복원 · ✅ Gemma-4-31B-it 내려받기 · ✅ Code·Math 후보 풀 (코드 23,971 · 수학 29,406, 측정 목록 각 12,000) · ✅ calendar 원인 (모델 능력) · ✅ gpu06 NeMo-Skills sandbox (`vedas` 줄을 빼고 재빌드 · 기동 · 터널, `RL_DATA.md` §5.9). 다음: 도구 사용 warm-up (진행 중) → GPU 창 (judge 검증 · 2노드 스모크 · 코드 측정) → STEM → Code·Math → General | `RL_DATA.md` §5 · `RL_PLAN.md` §1 |
| **RL 준비 (main1·sub1, 2026-10-06~, 사용자 지시: RL 관련 작업은 전부 이 트랙)** | **환경**: Pai 컨테이너 안에서 시스템 무변경으로 구동한다(원장 #15~#27). 영속 설치는 NFS `$NRL_ROOT`(셋업 `project_s/setup_nemo_rl_env_noninvasive.sh`). 사용자 결정 10-06: 원본 NFS·venv 로컬. 버전: v0.7.0 미채택, r0.8.0 정식 릴리스 시 재검토. **검증 완료**: 모델·엔진 게이트 M1~M5 · R1~R4 · E4, RLVR 경로 게이트 G0~G8 · Z1 (수치는 `GATES.md`). 학습 엔진 = NeMo-RL mcore + Pai 기능 포팅 (`RL_PLAN.md` 결정 11). **실행 기반**: 실행기 `runs/launch.sh` — 산출물은 리포 `results/alpha/<campaign>/`, wandb 는 RL 전용 프로젝트 `alpha-rl`. 레시피가 `student_rlvr1_alpha.yaml` 로 고정돼 있어 teacher 레시피에 쓰려면 `--config` 를 받게 고친다. 레시피 기본값에 E1 레버(chunked prefill 16K · `max_trajectory_age_steps` 2)가 들어갔다. 기본 꺼짐으로 병합한 기능(advantage 0 제외 · 요청 우선순위 · 요청별 계측)과 켜는 조건은 `RLVR_READINESS.md` §5.6. 첫 RLVR 런·v2·속도 최적화(2026-10-07~08)는 아래 "완료" 절 | ⏸ R5 recompute 실측 연기 (128K full 33.4K tok/s 기준선만, selective OOM) · 셋업 재구성 승인 대기 · PivotRL 위치 재검토 (`RL_PLAN.md` 결정 14 — 결정 18 로 2차 agentic teacher 단계의 방법 후보) | `RLVR_READINESS.md` §5·§6 · `GATES.md` |

### 첫 RLVR 런 저장 용량 (2026-10-07 결정, `$NRL_ROOT/runs/rlvr1_alpha/`)

첫 런·v2 는 2026-10-08 에 끝났고 체크포인트·HF 반출본은 남기지 않았다 (남은 로그·덤프는 `RLVR_READINESS.md` §5.6). 아래 실측 크기는 teacher 런 용량 계획의 기준값으로 남긴다.

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
| **코드 능력 끌어올리기 — 1차 최대 과제 (사용자 2026-10-08, `RL_PLAN.md` 결정 20)** | 24K 에서 alpha 는 코드 답을 거의 끝내지 못한다 (32회 중 1회, 정답 0). 미루지 않고 48K 로 다시 재고 커리큘럼으로 올린다. Code·Math teacher 시작(세 번째) 전까지 방법을 찾는다 | `RL_DATA.md` §5.5 |
| ~~1차 teacher 블렌드안~~ **결정됨 (사용자 2026-10-08, `RL_PLAN.md` 결정 19)** | 시작 ckpt agentic iter2862 · judge Gemma-4-31B-it · P0 승인 · 규모 B · 도구 사용 warm-up teacher 추가 · sandbox gpu06 · 한국어 보강 안 함 · Code·Math teacher 의 MOPD2 재사용은 학습 경향을 보고 결정. **남은 것**: STEM 비율안 (2026-10-08 보고: 객관식 50.7 · 주관식 20 · 퍼즐 15 · python 도구 과학 10 · 화학 4.3%, `RL_DATA.md` §5.4) 승인 · Code·Math 의 ns_tools·Lean 비율 | `RL_DATA.md` §5.0 |
| ~~reward_penalties (첫 런)~~ **결정됨 2026-10-07 — 4종 켬** — Ultra 는 4종(중복 reasoning·빈 최종 답·금지 토큰·think 태그 형식)을 모두 켠다. 발동하면 보상 0 | 근거: G5 롤아웃 384개에 오프라인 적용 시 보상 > 0 인 83개 중 깎이는 것 0 — 발동(형식 4a 78·빈 답 65·형식 4b 11)은 전부 이미 보상 0 인 잘림·퇴화 응답 (`tools/analyze_reward_penalties.py`). alpha token id 는 레시피에 반영됨 | `RLVR_READINESS.md` H5 |
| ~~체크포인트 보존 (첫 런)~~ **결정됨 2026-10-07 (같은 날 축소)** — `keep_top_k 2`(최근 2개) + `tools/export_watch.sh` 로 100스텝마다 HF 반출(1회 30 GB, 모두 보존). 학습 데이터 덤프는 첫 10스텝만 켜고, 분석을 기록한 뒤 지운다. 게이트 산출물 삭제: 스모크 체크포인트 6개 930 GB · G6 HF 반출본 90 GB · M5 랭크별 덤프 9.4 GB. 용량 예산은 아래 "첫 RLVR 런 저장 용량" | 저장 1회 155 GB, 1에폭 ≈ 1,120 스텝, 전부 보존이면 ≈ 17 TB (NFS 여유 13 TB, 공용) | 이 표 |
| **GBS·lr** (2026-10-08 지적, teacher 레시피 작성 때 결정) — **lr 결정됨: 도구 사용 teacher 3e-6 (`RL_PLAN.md` 결정 22) · 나머지 1차 teacher 5e-6 (사용자 결정 2026-10-09, 결정 23). 부모 `student_rlvr1_alpha.yaml` 의 1e-6 은 student 용으로 남겼다 (미정). GBS 는 아직 열려 있다** (근거: lr 1e-6 은 bf16 반영률 약 12%, 3e-6 은 step 50 에서 0.33 확인 · 짝 비교 보상 +0.009 ± 0.003, `KNOWN_ISSUES.md` 2026-10-08) | alpha 골격은 64 × 16 = GBS 1,024 · Muon lr 1e-6 · warmup 10 이다 (`student_rlvr1_alpha.yaml`, 64 는 `# 결정 대기` 임시값). 1차 teacher 규모안도 64 × 16 을 쓴다 (`RL_DATA.md` §5). Ultra 참조 (Adam, warmup 10): RLVR1·2 512 × 16 · lr 4e-6 / ifbench·rlhf teacher 128 × 16 · 2.5e-6 / reasoning teacher 128 × 16 · 3e-6 / swe teacher 32 × 16 · 3e-6 / MOPD 1024 × 1 · 2e-6. 배치 크기와 처리량: 정상 상태의 생성 노드는 최대의 약 1/3 로 돌고 64K 응답의 디코딩이 스텝을 정한다 (`RLVR_READINESS.md` §5.4). 배치를 바꿨을 때 시간당 샘플 수가 어떻게 변하는지는 실측이 없다 (앞서 쓴 "최대 처리량의 91%" 는 틀린 판단이라 §5.4 에서 정정했다) | 선택지(64·128·256)와 lr 조정안을 teacher 레시피와 함께 보고 → 사용자 결정 |
| ~~D1·D2·D3·D4·D5·D7~~ | **결정됨 2026-10-07** — 첫 RLVR 런: judge 불필요 10개 환경 71,730행 · prefix caching 끔 · bias 갱신 0 · KL 0 + threshold 2 · lr 1e-6 warmup 10 · 128K | `RL_PLAN.md` 결정 15 |
| Megatron-Bridge 포인터를 fork(`bcc4e415`)로 고정할지 | 현재 superproject 기록은 upstream `0c565c9a` — 새 클론은 AlphaBridge 가 없다 | `.claude/rules/alpha-submodules.md` |
| 셋업 재구성 제안 | 로컬 venv 스냅샷 · 인터프리터 로컬 · 바이트코드 사전 컴파일 · JIT 캐시 로컬 (2026-10-06 제안, 승인 대기) | `SETUP.md` §1 |
| RL 학습 워커에 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 를 기본값으로 둘지 (D6) — **첫 런은 권고값(켬)으로 간다 (사용자 2026-10-07)** | 권고: 정책 워커에만 켠다. Pai SFT 는 항상 켰고 "refit 5배" 근거는 측정 없는 FP8 docstring 이다. G5 에서 refit 시간을 확인한다. 끄면 1노드 상한은 rank 당 12K 토큰(96K/CP8)이다 | `KNOWN_ISSUES.md` 2026-10-07 메모리 항목 |
| MOPD 재현 범위 · 장문 컨텍스트 캡 | 교사 슬롯은 결정 18 로 정함 (MOPD1 = teacher 3종) · SWE 교사·MOPD 192k → 128k 캡 또는 SWE 슬롯 축소 · NeMo RL/Gym 스택 포팅 vs 자체 구현(verl/ChatLearn 백엔드 검토) | `RL_PLAN.md` §4 |
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
| 첫 RLVR 런 · v2 · 속도 최적화 (2026-10-07 ~ 10-08, `RL_PLAN.md` 결정 15~18) | v1 첫 런은 정확성 결함 4건과 7스텝 멈춤으로 정지 → upstream 이식 7건 뒤 v2 설정 G8 PASS. 속도 레버 E1 채택 (정상 상태 15.8 → 12.3 분/스텝, KL 같음). rollout 긴 꼬리 계측 T0: 긴 요청 대기 0~3 s, 64K 디코딩 17~37분이 임계 경로 → 우선순위 시험 생략, 투기 디코딩 기각. advantage 0 제외는 Z1 PASS 뒤 기본 꺼짐 병합. 현 속도 E2E 1,650~1,830 tok/s/GPU 는 NeMo-RL H100 기준 범위이고 학습 MFU 는 약 5% 다. Ultra 형태 일정 42~44일 → 결정 18 로 규모 축소 (1차 teacher 트랙). 모두 1~10스텝 시험 런이라 학습된 모델은 없다 | `RLVR_READINESS.md` §5.1~§5.6 · `GATES.md` G8·Z1 |
| RL 인프라 준비 (2026-08-12 ~ 08-21) | fork 3종 · AlphaBridge · vLLM 플러그인 · 게이트 M1~M3·K1·D1 · RL 블렌드 2종 · FlashQLA opt-in 통합 · 2노드 분리 토폴로지 확정 | `archive/POSTTRAIN_PREP_2026-08.md` |
| vLLM 단독 서빙 정량 게이트 (2026-08-24) | M4 PASS | `GATES.md` M4 |
| Pai 컨테이너 공존 설치 (2026-10-06) | 시스템 무변경 구동, 원장 #15~#20 해결, E1~E3 PASS | `SETUP.md` §1 |
