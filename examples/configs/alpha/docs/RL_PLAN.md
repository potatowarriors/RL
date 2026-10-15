# alpha RL 계획 — 단계·설계 결정·토폴로지

RL 단계의 설계 정본이다. 진행 상태와 열린 결정은 [`STATUS.md`](STATUS.md), 데이터는 [`RL_DATA.md`](RL_DATA.md) 에 쓴다.
2026-10-07 이관 출처: Pai `SFT_RL_DATASETS.md` §4·§5-6·§6, `ALPHA_POSTTRAIN_PROGRESS.md` §3, `NEMO_RL_SETUP.md` §5, alpha `README.md` 전제 절.

## 1. 단계 (결정 18, 2026-10-08)

alpha 의 순서는 Ultra 와 다르다. teacher 를 SFT ckpt 에서 바로 만들고 MOPD 를 두 번 한다.
Ultra 원형(SFT → Student-RLVR → 전문 교사 RL → MOPD)의 레시피는 이 리포의 `examples/nemo_gym/nemotron-3-ultra/` 에 있고, 설정 참조로만 쓴다.

```
[Pai] LC → general SFT(iter2862) → agentic SFT ──evaluate.sh──▶ hfmodel_*
[NeMo-RL] SFT ckpt ─┬─ RLVR teacher: General (chat·IF) ──┐
 (agentic iter2862)  ├─ RLVR teacher: STEM ───────────────┤
                    ├─ RLVR teacher: Code·Math ──────────┼─▶ MOPD1 ─▶ agentic teacher (SWE·search·terminal 등) ─┐
                    └─ RLVR teacher: 도구 사용 warm-up ───┘                                                     ├─▶ MOPD2
                       General·STEM teacher 재사용 (Code·Math 는 학습 경향을 보고 결정) ──────────────────────────┘
```

- 1차 teacher 블렌드·규모·sandbox 는 [`RL_DATA.md`](RL_DATA.md) §5 (결정 현황 §5.0).
- PivotRL 은 2차 agentic teacher 단계의 방법 후보로 다시 본다 — 방법·검증기 매핑은 [`study/pivotrl_study.md`](study/pivotrl_study.md) §5.
- **2차 agentic 보상 설계 과제 (2026-10-10, 도구 teacher τ³ 에서)**: 단일 스텝 피벗(정답 이력 + 다음 행동 일치)은 행동 정확도를 올리지만 "사용자가 조를 때 정책대로 거절"을 가르치지 못한다 — 금지 쓰기가 학습 후반에 늘었다 (airline 35 → 48/96, `STATUS.md`). 2차 단계에는 (1) 사용자 시뮬레이터 다중 턴 환경 + 최종 DB 상태 채점, (2) 정책 위반 쓰기에 다른 실수보다 큰 음의 보상, (3) 피벗을 쓰면 '거절 뒤에도 조르는' 상태를 넣고 메시지 보상이 실제 거절인지 검사 (현재 Gym 은 아무 메시지나 1, `verification_utils.py:162-175`) 를 넣는다. τ³ 의 airline · retail 은 학습에 쓰지 않는다 (벤치 오염). 이득 상한: 거절 과제가 2862 수준이면 τ³ 평균 +2.5, 2862 의 금지 쓰기까지 없애면 약 +6.
- 레시피 파일과 작성 여부는 [`../README.md`](../README.md) "레시피" 표.

## 2. 설계 결정 (현행)

| # | 결정 | 근거 | 일자 |
|---|---|---|---|
| 1 | RL 프레임워크 = NeMo-RL, **mcore 학습 + vLLM 롤아웃** | Ultra 실전 선례(하이브리드+MoE 동형), 공개 레시피, Megatron 자산 연속성. DTensor+HF 는 `modeling_alpha.py` 의 transformers 5.x 비호환으로 사실상 배제 | 2026-08-13 |
| 2 | mcore 측 **24층 1:1 매핑** | HF 24층(믹서+MLP) ≡ mcore 24 TransformerLayer. Pai 의 48층 분할은 내부 표현일 뿐 | 2026-08-13 |
| 3 | vLLM 통합 = **플러그인 패키지** (`vllm.general_plugins`) | stock 휠 유지로 lock 충돌 제거, 버전 업그레이드 rebase 무비용. in-tree 구현은 `potatowarriors/vllm` `alpha/model` 에 기록용으로만 | 2026-08-13 |
| 4 | **2노드 분리 토폴로지 + async GRPO** (§3) | long-context 메모리 요구와 IB 부재 제약의 최적해 (사용자 정정으로 colocate 권고 폐기) | 2026-08-14 |
| 5 | RLVR 환경 = **NeMo Gym** (Ultra 패턴) | 데이터가 이미 Gym 행 형식(행별 `agent_ref` 라우팅). 1노드 스모크는 judge-free 서버 구성(nanov3 템플릿) | 2026-08-13 |
| 6 | FlashQLA = **opt-in** (`ALPHA_GDN_BACKEND=flashqla`) | 검증된 커널 이득(49k 에서 fwd+bwd 3.5×), 실패 시 시끄럽게 중단 | 2026-08-18 |
| 7 | RL 기본값 = **GDN 재귀 상태 fp32 + R3** | KL 게이트 0.0042 → 0.0014 (`KNOWN_ISSUES.md` GDN 재귀 상태 항목) | 2026-10-06 |
| 8 | RL 옵티마이저도 **Muon** (`dist_muon`, SFT 동역학 정렬) | 사용자 결정. 필드명·기본값이 SFT 와 달라 nesterov·extra_scale_factor 를 명시 (`grpo_alpha_smoke_muon.yaml` 헤더) | 2026-10-06 |
| 9 | NeMo-RL **v0.7.0 미채택**, r0.8.0 정식 릴리스 시 재검토 | v0.7.0 은 06-27 분기라 현 브랜치보다 오래됨 | 2026-10-06 |
| 10 | Gym **v0.6.0 고정** | Pai SFT 평가와 같은 태그 (Pai `study/nemo_gym_assessment.md`) | 2026-09-15 |
| 11 | 학습 엔진 = **NeMo-RL mcore + Pai 기능 포팅** (B안). Pai Megatron-LM-251125 를 NeMo-RL 에 넣는 A안은 불가, RL 프레임워크 교체 C안은 비권고 | A안: NeMo-RL·Bridge 가 import 하는 mcore 모듈 157개 중 23개가 251125 에 없다(GDN `ssm.gated_delta_net`, R3 `moe.router_replay`, refit `resharding.refit` 등). 모델 클래스(MambaModel 48층 vs GPTModel 24층)와 스택(torch 2.8/TE 2.9 vs 2.11/2.14)도 다르다. C안은 검증 자산(브리지·플러그인·게이트)을 버린다 | 권고 2026-10-06, 사용자가 검증 게이트 3종 진행 승인 2026-10-07 |
| 12 | 멀티턴 경계 토큰 = **`<\|im_end\|>`(3)** (`vllm_cfg.turn_end_token_id`) | alpha EOS(0)는 문서 경계라 Gym 멀티턴이 2번째 호출부터 깨진다 (`KNOWN_ISSUES.md` 2026-10-07 턴 경계) | 2026-10-07 |
| 13 | Gym 도구 정의의 **`strict` 유지** (구현 대기) | SFT Agentic-v2 와 RL 블렌드 도구 행 13,190/13,190 이 strict 를 담는다 (`KNOWN_ISSUES.md` 2026-10-07 렌더 차이) | 사용자 결정 2026-10-07 |
| 15 | **첫 RLVR 런 설정**: judge·sandbox·nvarc 를 뺀 10개 환경 71,730행(`rlvr1_alpha_judgefree.jsonl`) · 최대 128K · KL 0 + `seq_logprob_error_threshold 2` · expert bias 갱신 0 · Muon lr 1e-6 warmup 10 GRPO step. prefix caching 끔·ES 켬은 권고값 그대로. 정확성 확인 뒤 학습·롤아웃 병목을 재고 속도 최적화를 정한다. **게이트 G0~G7 뒤 추가(2026-10-07)**: reward_penalties 4종 켬 · `keep_top_k 2` + 100스텝마다 HF 반출(모두 보존) · 학습 데이터 덤프는 첫 10스텝만, 분석 기록 뒤 삭제 | `RLVR_READINESS.md` D1·D3·D4·D5·D7, `STATUS.md` 첫 런 결정. identity(GenRM 채점)·GenRM·judge 환경은 judge 배치를 정한 뒤 추가한다 | 사용자 결정 2026-10-07 |
| 16 | **첫 RLVR 런 재시작**: 생성 상한 64K (학습 길이 128K 유지) · 시퀀스 마스킹(threshold 2)은 학습 forward 안에서 평가 (upstream #4171) · upstream 정확성 수정 4건 이식 뒤 처음부터 재시작 · 구간 1 종료 뒤 3~4시간 실험(속도 레버 chunked prefill·`max_trajectory_age_steps` 2)을 거쳐 본 런 | 근거: 첫 런 병목 판정(`RLVR_READINESS.md` §5.1)·가속 검토(§5.2)·`KNOWN_ISSUES.md` 2026-10-08 | 사용자 결정 2026-10-08 |
| 17 | **v2 본 런 세부**: code_gen 유지 (64K 상한에서 생성 토큰 65%·학습 토큰 47%·보상>0 0% 임을 알고 유지) · E1 레버 A/B 는 계획대로 완주 · 본 런 첫 10스텝(덤프 구간)은 loss 안 마스킹을 꺼 덤프에 학습측 logprob 을 남긴다(위치별·agent 별 KL) · wandb 로깅 (`alpha-posttraining`, group `rlvr1_v2`) | 근거: E1a 1스텝 agent 별 통계(`RLVR_READINESS.md` §5.3), loss 안 마스킹 덤프의 prev_logprobs 는 0 (G8·E1a) | 사용자 결정 2026-10-08 |
| 18 | **RL 단계 재편 (§1)**: SFT ckpt → 1차 RLVR teacher 3종(General = chat·IF · STEM · Code·Math) → MOPD1 → MOPD1 에서 agentic teacher(SWE·search·terminal 등) → MOPD2 (General·STEM teacher 재사용). Ultra 형태의 Student-RLVR1·2 는 두지 않는다. 속도 최적화 대신 현 자원(H100 16장)에 맞게 학습 규모를 줄인다 | Ultra 형태 일정 42~44일 (`RLVR_READINESS.md` §5.5). 설정 튜닝으로 몇 배 빨라질 근거가 실측상 없다 | 사용자 결정 2026-10-08 |
| 19 | **1차 teacher 세부**: 시작 ckpt = agentic 최종 iter2862 · teacher 4종 (도구 사용 warm-up teacher 추가 — MOPD1 에서 도구 사용 능력 망각 방지) · 규모 B = Ultra teacher 와 같은 샘플 수 (General 640 · STEM 820 · Code·Math 600 스텝, 블렌드 1 에폭) · judge = Gemma-4-31B-it · sandbox = gpu06 `alpha-eval` · 한국어 보강 안 함 · Code·Math teacher 의 MOPD2 재사용은 학습 경향을 보고 결정 | `RL_DATA.md` §5.0·§5.2·§5.8·§5.9 | 사용자 결정 2026-10-08 |
| 20 | **Code·Math 의 코드는 미루지 않는다 (B 안)** — 생성 상한을 48K 로 올리고, 코드 능력을 확인하며 점진적 커리큘럼으로 끌어올린다. 코드 능력 향상이 1차 최대 과제다. Code·Math 는 teacher 순서상 세 번째라 지금은 앞 작업(도구 사용 warm-up → STEM)에 집중하고 코드 해법은 열린 과제로 둔다. 순서·lr(Muon 1e-6, warmup 10)·GBS 1,024 는 권고대로 | P0-3 스모크: 24K 에서 코드 1/32 만 끝냄, 수학은 끝내면 대부분 정답 (`RL_DATA.md` §5.5) | 사용자 결정 2026-10-08 |
| 21 | **judge 배치 = A (롤아웃 노드 분할)**: 롤아웃 노드 GPU 8장을 vLLM 롤아웃 6장 + Gym judge(Gemma-4-31B-it TP2) 2장으로 나눈다. NeMo-RL 은 2노드 비-colocated 에서 롤아웃 GPU 수가 노드 GPU 수와 같아야 해서 (`grpo.py` assert — upstream main `9046a5979` 도 같다) 본체를 고친다: 1~8장을 허용하고, 남는 GPU 가 있으면 학습 placement group 을 먼저 잡는다. STEM 은 judge 를 포함한 원래 비율(`RL_DATA.md` §5.4)로 시작한다 | C(STEM 을 judge 없이)는 대체 풀이 작다 — Science-v1 에서 math-verify 로 채점할 수 있는 답은 1,409 개뿐 (`RL_DATA.md` §5.6). 롤아웃 GPU 가 8 → 6 이라 생성 처리량은 최대 25% 준다 — 2노드 스모크에서 잰다 | 사용자 결정 2026-10-08 |
| 22 | **도구 사용 teacher lr 3e-6** (Muon, warmup 10 은 그대로): lr 1e-6 런(55스텝 완료)을 멈추고 처음부터 다시 시작한다 (`teacher_tool_alpha.yaml`, `b3115c226`). 부모 `student_rlvr1_alpha.yaml` 의 1e-6 은 바꾸지 않았다 — STEM·Code·Math teacher 의 lr 은 따로 정한다 | 1e-6 런은 보상 평탄 (1~10스텝 = 31~40스텝 0.481). 학습 경로는 정상인데 50스텝 누적 갱신의 약 12% 만 bf16 가중치에 반영됐다 — 스텝당 갱신 2e-7 이 반올림 경계 6e-5 보다 작다. 3배면 같은 스텝 반영량 약 8.5배 (1차 추정, `KNOWN_ISSUES.md` 2026-10-08) | 사용자 결정 2026-10-08 |
| 23 | **1차 teacher lr 5e-6** (도구 사용 teacher 를 뺀 STEM · Code·Math · General, Muon, warmup 10 은 그대로): 각 teacher 레시피가 `megatron_cfg.optimizer.lr`·`min_lr` 를 덮어쓴다 (STEM `alpha/judge-split` `6d8f6de46`). 부모 `student_rlvr1_alpha.yaml` 의 1e-6 은 바꾸지 않았다 | lr 1e-6 은 50스텝 갱신의 12%, 3e-6 은 33% 만 bf16 가중치에 반영됐다. 3e-6 의 짝 비교 보상 이득은 +0.009 ± 0.003 으로 작았고 정합 지표는 그대로였다 — 사용자가 더 공격적인 값을 택했다. step 50 에서 반영 계수·gen KL·entropy 를 3e-6 런과 대조한다 (`KNOWN_ISSUES.md` 2026-10-08) | 사용자 결정 2026-10-09 |
| 24 | **STEM 블렌드 비율 승인 · judge 분할을 J1 전에 병합**: `teacher_stem_v1.jsonl` 52,500행 (python 도구 과학 10% 포함, `RL_DATA.md` §5.4) 로 STEM teacher 를 학습한다. `alpha/judge-split` 를 J1 2노드 스모크 전에 `alpha/post-train` 에 병합했다 (`8a8544dcc`) — J1 이 FAIL 이면 병합을 revert 한다 | 런은 작업 트리의 `nemo_rl` 을 editable 로 import 하므로 judge 분할 수정은 병합 뒤에만 실경로로 잴 수 있다. 수정은 노드 전체를 쓰는 기존 설정에서 동작이 같고 단위 테스트 4건 PASS (`2807ef2be`), 병합 결과 코드는 브랜치와 같다 (diff 0) | 사용자 결정 2026-10-10 |
| 25 | **Science-v1 정답 추출 결함 636행 교체 + judge 재판정 켬** (재판정은 결정 27 로 철회): (1) 정답이 자기 `output_regex` 에서 잘리는 행을 같은 agent · 분야 · 정답 형식의 결함 없는 Science-v1 행으로 바꾼다 — `teacher_stem_v2.jsonl` (636행) · `v2_val` (6) · `smoke_v3` (16), 행 수 · 비율 · 순서 그대로 (`tools/fix_stem_sci_regex.py`). (2) `teacher_stem_alpha.yaml` 의 equivalence judge 에 `check_full_generation_on_fail: true` · `reward_if_full_generation_succeeds: 0.5` — 추출한 답이 틀리면 최종 응답 전체로 다시 판정해 맞으면 0.5 | 비탐욕 정규식이 정답 속 닫는 문자(문장 괄호 · LaTeX `\)` · `f(x)` · `]` · `))`)에서 끊어, 정답을 요청 형식대로 쓴 탐침 60개 중 14개만 보상을 받았다. Gym 코드에는 구제 장치가 있지만 Gym 설정 파일이 끈다. 처음 집계 662행 중 26행은 결함이 아니었다 (정규식 불일치 → 응답 전체 판정 25 · `**` 표기 차이 1). 교체로는 모델 답 쪽 잘림을 막지 못해 재판정을 함께 켠다 (`KNOWN_ISSUES.md` 2026-10-10) | 사용자 결정 2026-10-10 |
| 26 | **도구 사용 teacher = step 400** (`teacher_tool` lr3e6, `hf/hf_step_00400`) — MOPD 도구 프롬프트의 teacher 로 쓴다 | τ³ (retail 114 + airline 50 × 4) pass^1 은 SFT 2862 · step 100~400 모두 잡음 안이다. step 400 은 기대 행동 일치율이 가장 높다 (retail +10.9 [+6.9, +14.9]) — 대신 airline 거절 과제의 금지 쓰기가 가장 많다 (48/96, 2862 35). 권고는 step 100 이었다 (`STATUS.md` 1차 teacher 행) | 사용자 결정 2026-10-10 |
| 28 | **STEM teacher 는 평가 없이 마감한다**: `main2r3` 최종 `ckpt/step_820` 을 HF 로 반출 (`hf/hf_step_00820`, 100스텝 반출 감시는 800 까지만 담당) 하고 Pai 벤치 없이 다음 teacher (순서상 Code·Math, 결정 19) 로 넘어간다. 종료 뒤 반출은 승인됨 — 런 종료(rc 0) 를 기다려 자동 실행 | 학습 보상 0.27 → 0.48 · gen KL 0.00125 · HF 반출 100~600 모두 PASS 로 파이프라인은 검증됐고, 벤치 GPU 시간보다 다음 teacher 착수가 우선 (`STATUS.md` 10-13 진행 보고) | 사용자 결정 2026-10-13 |
| 29 | **STEM `main2r3` 를 step 717 에서 멈추고 `ckpt/step_710` 에서 `main2r4` 로 재개 — step 당 프롬프트 64 → 256 (GBS 1,024 → 4,096, 옵티마이저는 step 당 1회 그대로), 최종 step 735 (25 step × 256 = 6,400 프롬프트 = 원래 남은 100 step × 64 와 같은 데이터, 52,500 행 안), 체크포인트 주기 10 → 5, lr 5e-6 유지.** 711~717 의 7 step 은 버린다 — 체크포인트 경계(720)를 기다리지 않은 것은 64 배치의 711~717 궤적과 256 배치의 같은 구간을 짝지어 비교하기 위해서다. 최종 반출은 `hf/hf_step_00735`, P0-3 오케스트레이터는 `runs/p03_orchestrate.sh main2r6 735` 로 재장전 (`main2r4`·`main2r5` 는 재개 플래그 문제로 즉시 종료 — 실제 런은 `main2r6`, 추가 오버라이드 `++policy.megatron_cfg.scheduler.use_checkpoint_opt_param_scheduler=true grpo.max_num_epochs=4`, `KNOWN_ISSUES.md` 10-13). 결정 28 의 "step_820" 은 "step_735" 로 읽는다 | 보상 곡선의 step 간 표준편차 0.050 (step 500~713, 평균 0.481, 20-step 이동평균 잔차 0.051) 이 사전학습에서 작은 배치로 본 잡음과 같다는 사용자 판단 — 256 이면 추출 잡음이 절반. 16 GPU 에서는 step 시간이 생성에 묶여 샘플당 wall time 은 같다 (245 s/1,024 궤적 → 약 980 s/4,096). 선택지 ①(25 step)·②(50)·③(100) 중 ① — 같은 데이터·같은 lr 로 배치 효과만 비교하고 다음 teacher 일정을 지킨다 (`STATUS.md` 10-13 19:05) | 사용자 결정 2026-10-13 19:05 |
| 30 | **Code·Math 1차 teacher 의 생성 상한은 32K 로 시작하고, P0-3 사전 측정도 32K 로 다시 잰다** (`runs/p03_orchestrate.sh main2r8 735`, fleet max_len 40,960, 모델 설정 `runs/alpha_vllm_model_p0_32k.yaml`, 태그 `p03_math32`·`p03_code32`). 결정 20 의 "24K 시작 → 48K" 와 10-13 의 "24K 측정" 을 대체한다. 24K 수학 측정은 20% (8,763 롤아웃) 에서 중단, 부분 결과는 `$NRL_ROOT/p0_measure/p03_math/` 에 보존 | 24K 측정 중간 집계: 롤아웃 71.5% 가 24,576 에 잘림, 끝낸 응답의 정답률 32%, 문제별 0/4 의 62% 가 "4회 모두 잘림" — 상한이 능력보다 먼저 막는다. 정답 응답 길이 중앙 2,708 · p90 17,907 토큰, 끝냈지만 오답은 중앙 7,404 · p90 21,180 이라 32K 면 대부분 끝낸다. STEM 32K 런에서 잘림 11.6% → 0.1% 로 "끝내는 법" 을 빨리 배운 것도 근거 (`STATUS.md` 10-14 09:40 보고) | 사용자 결정 2026-10-14 09:42 |
| 31 | **낙오 정체 대책 = 롤아웃 초과 할당 + 이월** (사용자 제안): `grpo.async_grpo.rollout_overprovision = k` 면 수집기가 목표 step 마다 N+k 그룹을 띄우고, 버퍼는 N 그룹이 모이는 즉시 학습에 내보낸다 (현재 step 용 우선, 부족하면 이전 step 용 미사용 그룹). 남은 그룹은 버리지 않고 다음 step 으로 재라벨해 이월하고, 나이 초과분은 기존 규칙으로 폐기한다. 배치의 미사용 꼬리는 다음 목표로 넘겨 프롬프트 손실 0. 기본값 0 = 기존 동작. 본체 수정 (브랜치 `alpha/async-overprovision` → `alpha/post-train` 병합, REBASE NOTE 있음). Code·Math 레시피는 k = 8 (256 기준 +3%) 로 쓸 예정이며, 적용 전 GPU 창에서 10 step 스모크로 "이월 발생 · GBS 일정 · 프롬프트 손실 0" 을 로그로 확인한다 | 10-13 (1 h) · 10-14 (6 h) 낙오 정체 — 버퍼가 "현재 step 그룹 N개 전부" 를 무한 대기. 시간 상한 방식은 GBS 정확성을 깨고, 느린 그룹 폐기는 긴 응답·어려운 문제를 체계적으로 빼는 편향이라 이월을 택했다. 단위 테스트 신규 9 + 기존 버퍼·수집 루프 28 PASS | 사용자 승인 2026-10-14 10:30 |
| 32 | **Code·Math 단계 정의와 1단계(32K) 블렌드 규칙**: 단계 = 생성 상한 (1단계 32K → 2단계 48K). 단계 안의 프롬프트 수·GBS 조정은 부차적 문제로 둔다. 1단계 블렌드는 P0-3 32K 측정에서 **"4회 중 1회 이상 맞힌 문제 전부" + "같은 수의 0/4 문제" (1:1)** — 0/4 는 "4회 모두 잘림" 에서만 뽑고 (끝냈는데 틀린 문제는 제외), Nano SFT 참조 pass_rate (`ref_pass_rate_nano_sft`) 높은 순, 동률 무작위. 수학은 2,257 (1/4 1,137 · 2/4 551 · 3/4 334 · 4/4 235) + 0/4-잘림 2,257 = 4,514. 코드는 측정 완료 뒤 같은 규칙. **수학과 코드는 draw 수를 1:1 로 맞춘다** — 한쪽만 올려 보상이 오르는 것을 막기 위해. 코드 고유 수가 적으면 코드를 2에폭(필요 시 3)으로 돌리고 수학은 그에 맞춰 1~1.3에폭. 1단계가 끝나면 그 모델로 예비 풀(수학 17,406 · 코드 11,971)을 4회 재측정해 2단계(48K) 블렌드를 새로 뽑는다 | 32K 측정: 수학 잘림 68.1% · 0/4 의 54% 가 4회 모두 잘림 (`STATUS.md` 10-14 15:50). 1단계는 "짧게 맺고 맞히기" 신호에 집중, 2단계는 끝내는 법을 배운 모델 기준으로 더 어려운 문제를 고른다 | 사용자 결정 2026-10-14 16:00 (코드 수치는 측정 뒤 보고) |
| 33 | **코드 사전 측정 중단 + 1단계 블렌드 확정 + 초과 할당은 본 런에서 검증**: (1) 코드 32K 측정을 4,909 롤아웃(10%)에서 멈춘다 — 12시간 더 재서 "1회 이상 맞힌" 수백 문제를 더 찾는 것은 무작위 채움 대비 이득이 작고(4,000 중 신호 문제 약 1,200 vs 약 500), 학습이 맺는 법을 배우면 그 차이가 사라진다. 1단계 학습 로그가 문제당 16회의 정답률을 공짜로 주므로 2단계 선별은 그것으로 한다. (2) 블렌드 `teacher_codemath_s1_v1.jsonl` 9,028 행 (`tools/build_codemath_stage1_blend.py`, seed 20261014): 수학 4,514 = 1회 이상 맞힌 2,257 전부 + 0/4-모두-잘림 2,257 (원천 비율 유지 무작위 — 참조 pass_rate 순서는 정보가 없어 쓰지 않음, 결정 32 세부 변경); 코드 4,514 = 측정에서 1회 이상 맞힌 170 + Nano tier-1 을 Nano 참조 pass_rate 높은 순 4,344 (끝내고도 틀린 202 제외; 채움 하한 pass_rate 0.25). 검증셋 예비 풀 128 + 128. 구조 게이트 OK · 렌더 프롬프트 최대 코드 4,448 · 수학 1,357 토큰 · 검사기 ERROR 0. (3) 결정 31 의 초과 할당·이월은 별도 스모크 없이 본 런 초반 로그로 검증한다 — 통과면 그대로 계속, 실패면 중단·수정 | 사용자 2026-10-14 16:35 "2node 12시간 태워서 800개 찾고 나머지 random 이면 pass_rate 로 채우는 것과 뭐가 다른가" — 맞는 지적. 사전 측정은 이번 수학 한 번으로 끝내고 이후는 학습 로그·동적 샘플링으로 | 사용자 승인 2026-10-14 16:40 |
| 34 | **Code·Math 1단계는 시퀀스 단위 손실** (`loss_fn.token_level_loss: false`, 원래 GRPO 방식). 같이 `seq_logprob_error_in_loss: false` 와 `grpo.seq_logprob_error_threshold: null` (in-loss 필터는 token_level 전용이고, threshold 를 남기면 prev_logprobs forward 가 하나 더 생김; 마스킹 발동은 STEM 676 step 에 6 시퀀스, 이 런 19 step 에 1). `ckpt/step_25` 뒤 `s1r1` 을 멈추고 `s1r2` 로 재개. **생성 길이와 맺음이 안정되면 토큰 단위(DAPO)로 복귀한다 (사용자 2026-10-14 23:35)** — 기준: 잘림 비율이 10% 아래로 내려가 유지되고 생성 평균 길이가 평탄해지는 시점, 체크포인트 경계에서 전환 |
| 35 | **시퀀스 단위 손실 롤백 → 토큰 단위 (부모 D4 설정 복귀: `token_level_loss: true` · `seq_logprob_error_in_loss: true` · threshold 2)**. `ckpt/step_60` (07:39) 뒤 `s1r2` 중단 → `s1r3` 재개 | 시퀀스 단위 26 step (26~51): grad norm 0.01 → 0.05~0.08 (STEM 수준)로 커졌지만 행동 변화 없음 — 잘림 0.675 → 0.713 (최근 8 step 평균), 수학 보상 0.226 → 0.219, 생성 평균 25.5K → 26.4K. 가중치 재배분만으로는 "맺기" 가 학습되지 않았다. 남은 선택지: 잘림 벌점(상한 도달 샘플 보상 −1, 비동기 경로 `rollouts.py` 그룹 후처리 본체 수정) · 16K 등 짧은 상한 · 쉬운 수학 원천 — 결정 대기 | 사용자 결정 2026-10-15 07:40 | 토큰 단위(Ultra 원형 승계, DAPO)는 샘플의 75% 가 32K 에 잘린 조건에서 4K 정답 1개가 32K 오답 1개의 1/8 무게라 "맺고 맞히기" 신호가 묻힘 — step 1~21 잘림 0.79 → 0.74 정체 · 보상 0.09~0.16 평탄 · grad norm 0.01 (STEM 0.05~0.10). 시퀀스 단위는 샘플마다 같은 무게 | 사용자 결정 2026-10-14 23:30 |
| 36 | **DAPO 길이 감점(soft overlong punishment)을 async 경로에 넣는다 — 본체 패치 + Code·Math 1단계 레시피 `grpo.reward_shaping` (buffer 4,096 · penalty 1 · 상한 32,768 = `max_new_tokens`). 동적 샘플링은 켜지 않는다.** 구현: `nemo_rl/algorithms/grpo.py` `apply_async_reward_shaping` 이 리플레이 버퍼에서 뽑은 학습 배치에 sync 루프와 같은 위치(롤아웃 지표 뒤 · advantage 앞)에서 upstream `apply_reward_shaping` 을 적용하고 원보상을 `unshaped_total_reward` 로 보존한다. 응답 길이는 턴별 최대 assistant 토큰 (생성 상한이 호출 단위). 지표 `train/reward_shaping/(penalized_rate|mean_penalty)`. 브랜치 `alpha/async-dapo` (`bb0835bd2` 본체 · `2aa81125a` 레시피), 단위 테스트 4 신규 + 기존 29 = 33/33, 검사기 ERROR 0 · WARN 2. `ckpt/step_N` 저장 뒤 `s1r3` 중단 → 병합 → `s1r4` 재개 (승인 대기) | 사용자 2026-10-15 10:00 "생각보다 작은 작업이고 이미 구현된, 검증된 코드를 사용하여 간단히 구현이 가능하다면 바로 적용". 효과: 28,672 토큰까지 감점 0, 32,768 에서 −1 → 잘린 오답 −1 · 잘린 정답 0 · 맺은 오답 0 · 맺은 정답 1. 전부 오답인 그룹(추정 그룹의 약 절반: 코드 p≈0.05 → 4개 모두 0 일 확률 0.81, 수학 p≈0.3 → 0.25)에도 "맺기 > 잘림" 기울기가 생긴다 — 잘림 75% 의 직접 대책. DAPO 논문은 동적 샘플링으로 그런 그룹을 버리지만 여기서는 쓴다. **동적 샘플링을 미루는 근거**: (1) async 수집기는 enqueue 때 그룹을 버리면 보충 생성(gap-fill)이 그 배치 워커가 끝난 뒤에야 시작돼 step 이 보충 라운드마다 32K 생성 지연만큼 멈춘다 — k=4 로는 zero-std 약 50% 를 못 메운다. (2) 멈춤 없이 쓰려면 N+k ≈ 2N(128 그룹)을 띄워야 하는데 생성이 이미 한계(`exposed_generation` 107~172 s = step 의 16~23%, 버퍼 76~77/204)라 step 시간 약 2배, 신호/시간은 같다. (3) 128 그룹 × 3 목표 = 1,536 궤적 in-flight → 10-13 Ray spill 디스크 사고 재발 위험. 학습 연산만 줄이는 대안은 `grpo.skip_zero_advantage_samples` (생성 추가 없음) — 별도 결정. **후속 (사용자 2026-10-15 11:00)**: 감점은 "그룹 안에서 잘리지 않으면 적어도 감점은 받지 않는다" 는 신호를 먼저 주는 단계다. 지금은 버려야 할 zero-std 그룹이 너무 많아 동적 샘플링이 학습 시간만 늘리고 신호를 많이 못 주므로, 감점으로 어느 정도 학습된 뒤에 동적 샘플링을 넣는다. 적용 시점의 기준: 잘림률이 내려가 전부 같은 보상인 그룹 비율이 낮아져(목표 20% 이하) `rollout_overprovision` 을 조금만 올려도 N 개 신호 그룹이 한 라운드에 모일 때 — 실측은 step 로그의 `Rewards stats` 와 그룹별 보상 분포로. 그때 하드 드롭 구현 + spill 경로 NFS 이전을 함께 본다 | 2026-10-15 |
| 27 | **J1 보충 j1c 후속 5건**: (1) rdkit 채점기가 예/아니오 행(`bool` · `presence`)에서 yes/no 도 읽는다 — 채택 (`6e1e88827`) (2) judge 재판정 끔 — 결정 25 의 (2) 철회, `teacher_stem_alpha.yaml` `check_full_generation_on_fail: false` (3) STEM 본 런 `main` (10-10 17:32 기동, 재판정 켬 상태) 유지 (4) 실행기 순서 변경 채택 — 런 액터 0 이면 Gym 서버 정리 뒤 GPU 검사 (`bf379bc96`) (5) j1c 를 J1 판정 근거로 인정 | (1) 맞는 답 `((No))` 가 0 점이었다 — j1c 출력 재채점 정답 4 → 28 /48, 정수 행 그대로. (2) 0.5 보상 39 중 잘린 추출 복구 13, 26 은 같은 답의 판정 뒤집힘 (일부 오답), judge 표본의 3.6%. (4) judge vLLM 이 런 뒤 GPU 2장을 쥐고 남아 다음 기동이 막힌다. (5) judge 파싱 실패 0/1,129 · 재판정 발생을 실제 런에서 쟀다. 다섯 건 모두 승인 없이 먼저 진행했고 사후 승인받았다 — 이후 `.claude/rules/alpha.md` 승인 규칙을 따른다 | 사용자 결정 2026-10-10 |
| 14 | **RLVR 준비를 먼저** 진행한다. Pai 의 중요한 학습 설정을 유지하고 SFT 학습 최적화를 이식해 최대 128K 로 학습한다. 첫 작업은 이식 위험 보고·게이트(`RLVR_READINESS.md`) | §1 의 PivotRL 선행 순서는 재검토 대상 — Ultra RLVR1 블렌드의 38% 가 이미 단일 스텝 도구 호출(피벗형) 환경이다 | 사용자 지시 2026-10-07 |

**검토 예정 — step 당 프롬프트 수 2단계 (사용자 2026-10-13 20:15, 결정 아님, Code·Math 레시피 때 결정)**: 처음 100 step 은 64, 나머지는 256 으로 돈다. 뜻은 "작은 배치로 업데이트 횟수를 벌어 초반 상승을 빨리 얻고, 평탄해지는 구간은 큰 배치로 잡음을 줄인다" 이다. NeMo-RL 에는 런 중 프롬프트 수를 바꾸는 옵션이 없으므로 (`num_prompts_per_step` 은 기동 때 고정) 구현은 체크포인트 경계에서 멈추고 재개하는 2단계 런이다 — 재기동 35~40분, 재개 오버라이드 `++policy.megatron_cfg.scheduler.use_checkpoint_opt_param_scheduler=true` 와 `grpo.max_num_epochs` 클램프 회피가 필수 (`KNOWN_ISSUES.md` 10-13, 결정 29 의 `main2r4`·`main2r5`). 판단 근거는 STEM `main2r3`(64, step 1~717) 대 `main2r6`(256, step 711~735) 의 보상 잔차·상승 기울기 비교다 (`STATUS.md`). 샘플당 wall time 은 두 배치가 같으므로 (생성 한계) 시간 손해는 재기동 1회뿐이다.

**검토 예정 — 장문(128K 생성) 확장 때 staleness 4 + 클립 상한 5.0 (사용자 2026-10-14, 결정 아님, MOPD2 레시피 때 재검토)**: 지금 비동기 설정은 `max_trajectory_age_steps: 2` (궤적이 시작한 뒤 가중치 갱신 2번까지 허용, 초과분 폐기) · 클립 0.2/0.28 이다. 생성 상한을 128K 로 올리면 한 궤적이 step 2개 안에 못 끝나는 경우가 늘어 폐기와 엔진 유휴가 커진다. MiMo-V2.6 은 1M 문맥에서 staleness 4 의 partial rollout(가중치 갱신 뒤 이어서 생성, 확률 재계산 없음 — 우리 `in_flight_weight_updates: true` · `recompute_kv_cache_after_weight_updates: false` 와 같은 방식)을 쓰고, 그만큼 커지는 off-policy 이탈을 "양·음 advantage 별 4개 클립 경계, 초기 [0.2, 5.0], 정책 엔트로피로 런타임 조절" 로 받아낸다 (arXiv 2610.11959). 우리에게는 age 4 + 양의 advantage 상한 5.0 조합이 후보이며, 판단 지표는 age 초과 폐기 비율 · 클립 발동률 · gen KL. NeMo-RL 의 클립은 부호 공통의 `ratio_clip_min/max` 와 음의 advantage 쪽 dual-clip `ratio_clip_c` (`loss_functions.py:128`, 지금 null) 뿐이라, 양의 advantage 상한만 따로 여는 것은 본체 수정이 필요하다.

### SFT → RL 승계 범위 (2026-10-07)

B안에서 SFT 가 만든 특성이 어디까지 이어지는지 정리한다. 게이트 수치는 [`GATES.md`](GATES.md).

| 구분 | 항목 | 근거 |
|---|---|---|
| 이어짐 (검증 완료) | 가중치 (M1 14,181/14,181 비트 동일) · forward 함수 (M2 cos ≥ 0.99984) · forward·gradient 동등성 (M5, bf16 노이즈 바닥 이내) · 롤아웃↔학습 정합 (R1 KL 0.0014) · Muon 동역학 (R2: momentum 0.95·nesterov·extra_scale 0.2·NS 5 quintic·spectral·wd 0.1·Adam β2 0.95) · QGKV 4-way split (upstream 동등, Pai 구현 대비 상대오차 6e-7) · 라우터 정밀도 (연산 fp32 이상, `expert_bias` fp32) · 단일 턴 chat 렌더 (R3 P1 12/12) | — |
| 의도적으로 달라짐 | optimizer 상태는 새로 시작한다 (SFT 스테이지 전환 관례와 같음) · 라우터는 동결한다 (NeMo-RL RL 기본) · lr 은 별도 결정한다. grad clip 은 SFT 8.0@CP8 이 NeMo-RL 1.0 과 같은 크기다 — Pai 는 grad norm 이 CP 배수로 기록되고 NeMo-RL 은 CP 무관 (2026-10-07, `RLVR_READINESS.md` §2) · GDN 구현 코드는 다르다 (Pai megatron_patch vs mcore — 같은 함수임은 M2·M5 로 확인) | `STATUS.md` 열린 결정 |
| 미확인 | Gym 멀티턴 도구 렌더 (`strict` 구현 후 R3 P6) · packing+CP 경로 수치 (장문맥) · RL 후 일반 능력 (벤치 비하락 — Pai `SFT_BENCHMARKS.md`) | `STATUS.md` 다음 할 일 |

Pai 기능 중 포팅 후보였던 Muon optimizer-state offload(Pai 기능 #4, upstream PR #6244 는 Megatron-LM `dev` 에만 있음)는 R4 실측상 1노드 128K/CP8 에 필요 없다.
단 `expandable_segments` 조건이다 (`KNOWN_ISSUES.md` 2026-10-07 메모리 항목). 컨텍스트가 그 이상으로 늘면 다시 본다.

## 3. 토폴로지

클러스터는 Backend.AI 다 (Slurm 없음, `ray start` 수동 기동). 노드 간 InfiniBand 가 없다 (TCP ~9.1 Gbit/s).

- **node0 = 학습 전용, node1 = vLLM 롤아웃 전용** (`generation.colocated.enabled: false`) + **async GRPO + `in_flight_weight_updates`**.
- long-context RLVR 에서는 학습 상태만으로 node0 이 꽉 차므로 분리가 필수다. async 로 두 노드가 동시 가동된다.
- 노드 간 가중치 refit ~28초는 스텝 시간(수십 분) 대비 무시 가능하고 in-flight 로 생성과 오버랩된다.
- colocate(1노드)는 짧은 컨텍스트 스모크 전용이다 (R1 게이트가 이 구성).
- LLM-judge·teacher 가 필요한 스테이지(RLHF teacher, MOPD)에서는 node1 GPU 를 rollout 과 judge 가 분할한다 — 해당 스테이지 설계 시 재배분.

## 4. alpha 규모 적용 설계 (2026-08-01 초안, Pai `SFT_RL_DATASETS.md` §4 에서 이관)

1. **컨텍스트 정합**: Ultra 의 RLVR ctx 49k→65k 는 우리 SFT max 64k·LC 32k~64k 계획(2026-08-01 당시)과 자연스럽게 맞는다. 충돌 지점은 **SWE 교사·MOPD 의 192k** —
   alpha LC 학습 상한이 128k 이므로 **128k 로 캡**(SWE rollout 축소) 또는 SWE 슬롯 축소가 필요하다.
2. **교사 패널 현실화** (Ultra 는 550B 학생 + 전문 교사들; alpha 는 15B-A3B) — **결정 18(§1)로 대체됐다**. 아래는 2026-08 초안 기록이다:
   - general 교사 = alpha Student-RLVR 자신 (레시피 그대로, 추가 자원 불요)
   - 전문 교사 = alpha 체크포인트에서 각각 소규모 RL (교사 RL 은 GBS 2048·수백 step 규모라 우리 클러스터로 가능;
     교사 수를 2~3종으로 축소 검토: Reasoning/IF 우선)
   - 외부 교사 보강: LongBlocks 의 응답 3열(Qwen3-Next-80B 등)은 **오프라인 증류** 소재로 즉시 사용 가능 — on-policy 전 워밍업
   - 교사는 반드시 reasoning 모델이어야 한다 (Pai `KNOWN_ISSUES` 2026-09-09 ko_chat 폐기 교훈)
3. **LC 능력 유지**: RL 은 rollout 비용상 long-context 환경 입력 ≤32k (Nemotron Nano 관행). 각 단계 통과 시
   Pai 벤치 스위트의 RULER 로 회귀를 측정한다.
4. **MOPD 재현 범위** (결정 필요 — `STATUS.md` 열린 결정): 교사 슬롯 수(2~3 vs 5), 192k→128k 캡,
   NeMo RL/Gym 스택 포팅 vs 자체 구현(verl/ChatLearn 백엔드 검토).
5. **effort/budget**: SFT 단계의 effort 렌더(Pai `SFT_RL_DATASETS.md` §2.6)는 Ultra 재현이다. NeMo-RL `effort_levels` 계수는
   Ultra 레시피에 공개돼 있다 — `low_weight 0.1`·`low_penalty 1`·`low_ub 15000` (`examples/nemo_gym/nemotron-3-ultra/student_rlvr1.yaml:393-397`,
   2026-10-07 확인. 이전 "미공개" 기록은 틀렸다). `low_ub` 는 alpha medium-effort 응답 길이 실측으로 다시 본다.
   RLVR1 블렌드 3,429행(3.5%)이 alpha 템플릿과 같은 형식의 마커(`\n\n{reasoning effort: efficient}`)를 담는다.

## 5. 평가

RL 체크포인트 평가는 **Pai 벤치 스위트를 그대로 쓴다** (HF 변환 후). 정본은 Pai `examples/alpha/docs/SFT_BENCHMARKS.md`,
수치는 Pai `examples/alpha/eval_sft/results/TRACKING.md`. 기준선은 agentic SFT 최종 ckpt 다.
