# alpha RL 계획 — 단계·설계 결정·토폴로지

RL 단계의 설계 정본이다. 진행 상태와 열린 결정은 [`STATUS.md`](STATUS.md), 데이터는 [`RL_DATA.md`](RL_DATA.md) 에 쓴다.
2026-10-07 이관 출처: Pai `SFT_RL_DATASETS.md` §4·§5-6·§6, `ALPHA_POSTTRAIN_PROGRESS.md` §3, `NEMO_RL_SETUP.md` §5, alpha `README.md` 전제 절.

## 1. 단계

재현 대상은 Nemotron 3 Ultra post-training 이다: SFT → Student-RLVR → 전문 교사 RL → **MOPD**(멀티 교사 on-policy 증류).
레시피 원형은 이 리포의 `examples/nemo_gym/nemotron-3-ultra/` (구 `ultra-v3` 브랜치 `examples/configs/ultra/`).

```
[Pai] LC → general SFT(iter2862) → agentic SFT ──evaluate.sh──▶ hfmodel_*
[NeMo-RL] PivotRL(agentic) → RLVR(GRPO) → 전문 교사 RL(2~3종) → MOPD
```

- agentic SFT 완주 뒤 첫 RL 은 **PivotRL** 이다 — 방법·검증기 매핑·게이트는 [`study/pivotrl_study.md`](study/pivotrl_study.md) §5.
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

### SFT → RL 승계 범위 (2026-10-07)

B안에서 SFT 가 만든 특성이 어디까지 이어지는지 정리한다. 게이트 수치는 [`GATES.md`](GATES.md).

| 구분 | 항목 | 근거 |
|---|---|---|
| 이어짐 (검증 완료) | 가중치 (M1 14,181/14,181 비트 동일) · forward 함수 (M2 cos ≥ 0.99984) · forward·gradient 동등성 (M5, bf16 노이즈 바닥 이내) · 롤아웃↔학습 정합 (R1 KL 0.0014) · Muon 동역학 (R2: momentum 0.95·nesterov·extra_scale 0.2·NS 5 quintic·spectral·wd 0.1·Adam β2 0.95) · QGKV 4-way split (upstream 동등, Pai 구현 대비 상대오차 6e-7) · 라우터 정밀도 (연산 fp32 이상, `expert_bias` fp32) · 단일 턴 chat 렌더 (R3 P1 12/12) | — |
| 의도적으로 달라짐 | optimizer 상태는 새로 시작한다 (SFT 스테이지 전환 관례와 같음) · 라우터는 동결한다 (NeMo-RL RL 기본) · grad clip·lr 은 별도 결정한다 (SFT 8.0·1e-5 는 loss 정규화가 달라 그대로 못 옮김) · GDN 구현 코드는 다르다 (Pai megatron_patch vs mcore — 같은 함수임은 M2·M5 로 확인) | `STATUS.md` 열린 결정 |
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
2. **교사 패널 현실화** (Ultra 는 550B 학생 + 전문 교사들; alpha 는 15B-A3B):
   - general 교사 = alpha Student-RLVR 자신 (레시피 그대로, 추가 자원 불요)
   - 전문 교사 = alpha 체크포인트에서 각각 소규모 RL (교사 RL 은 GBS 2048·수백 step 규모라 우리 클러스터로 가능;
     교사 수를 2~3종으로 축소 검토: Reasoning/IF 우선)
   - 외부 교사 보강: LongBlocks 의 응답 3열(Qwen3-Next-80B 등)은 **오프라인 증류** 소재로 즉시 사용 가능 — on-policy 전 워밍업
   - 교사는 반드시 reasoning 모델이어야 한다 (Pai `KNOWN_ISSUES` 2026-09-09 ko_chat 폐기 교훈)
3. **LC 능력 유지**: RL 은 rollout 비용상 long-context 환경 입력 ≤32k (Nemotron Nano 관행). 각 단계 통과 시
   Pai 벤치 스위트의 RULER 로 회귀를 측정한다.
4. **MOPD 재현 범위** (결정 필요 — `STATUS.md` 열린 결정): 교사 슬롯 수(2~3 vs 5), 192k→128k 캡,
   NeMo RL/Gym 스택 포팅 vs 자체 구현(verl/ChatLearn 백엔드 검토).
5. **effort/budget**: SFT 단계의 effort 렌더(Pai `SFT_RL_DATASETS.md` §2.6)는 Ultra 재현이다. NeMo-RL `effort_levels` 계수
   (`low_weight`/`low_ub`/`low_penalty`)는 미공개 — RL 착수 시 medium-effort 응답 길이 실측으로 정한다.

## 5. 평가

RL 체크포인트 평가는 **Pai 벤치 스위트를 그대로 쓴다** (HF 변환 후). 정본은 Pai `examples/alpha/docs/SFT_BENCHMARKS.md`,
수치는 Pai `examples/alpha/eval_sft/results/TRACKING.md`. 기준선은 agentic SFT 최종 ckpt 다.
