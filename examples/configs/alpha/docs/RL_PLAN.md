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
| 25 | **Science-v1 정답 추출 결함 636행 교체 + judge 재판정 켬**: (1) 정답이 자기 `output_regex` 에서 잘리는 행을 같은 agent · 분야 · 정답 형식의 결함 없는 Science-v1 행으로 바꾼다 — `teacher_stem_v2.jsonl` (636행) · `v2_val` (6) · `smoke_v3` (16), 행 수 · 비율 · 순서 그대로 (`tools/fix_stem_sci_regex.py`). (2) `teacher_stem_alpha.yaml` 의 equivalence judge 에 `check_full_generation_on_fail: true` · `reward_if_full_generation_succeeds: 0.5` — 추출한 답이 틀리면 최종 응답 전체로 다시 판정해 맞으면 0.5 | 비탐욕 정규식이 정답 속 닫는 문자(문장 괄호 · LaTeX `\)` · `f(x)` · `]` · `))`)에서 끊어, 정답을 요청 형식대로 쓴 탐침 60개 중 14개만 보상을 받았다. Gym 코드에는 구제 장치가 있지만 Gym 설정 파일이 끈다. 처음 집계 662행 중 26행은 결함이 아니었다 (정규식 불일치 → 응답 전체 판정 25 · `**` 표기 차이 1). 교체로는 모델 답 쪽 잘림을 막지 못해 재판정을 함께 켠다 (`KNOWN_ISSUES.md` 2026-10-10) | 사용자 결정 2026-10-10 |
| 14 | **RLVR 준비를 먼저** 진행한다. Pai 의 중요한 학습 설정을 유지하고 SFT 학습 최적화를 이식해 최대 128K 로 학습한다. 첫 작업은 이식 위험 보고·게이트(`RLVR_READINESS.md`) | §1 의 PivotRL 선행 순서는 재검토 대상 — Ultra RLVR1 블렌드의 38% 가 이미 단일 스텝 도구 호출(피벗형) 환경이다 | 사용자 지시 2026-10-07 |

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
