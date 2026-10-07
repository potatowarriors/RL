# alpha post-training 진행 종합 보고 (2026-08-12 ~ 08-21)

> **아카이브 (동결)** — 2026-10-07 워크스페이스 `project_s/ALPHA_POSTTRAIN_PROGRESS.md` 에서 이관. 본문은 작성 시점(2026-08-21) 그대로다.
> 현행 정본: 게이트 [`../GATES.md`](../GATES.md) · 결정 [`../RL_PLAN.md`](../RL_PLAN.md) · 데이터 [`../RL_DATA.md`](../RL_DATA.md) · 상태 [`../STATUS.md`](../STATUS.md).
> 본문의 `NEMO_RL_SETUP.md` 는 [`../SETUP.md`](../SETUP.md)(설치)·[`../KNOWN_ISSUES.md`](../KNOWN_ISSUES.md)(원장)로, `NeMo-RL/examples/configs/alpha/README.md` 는 [`../../README.md`](../../README.md) 로 나뉘었다.

pre-train(DiLoCo stage2) 종료 예정 ~08-27 이전에 NeMo-RL 기반 post-training(SFT→RLVR→teacher RL→MOPD)
인프라를 준비하는 작업의 전 과정 기록. 상세 운영 문서는 `NEMO_RL_SETUP.md`,
컴포넌트별 상세는 `NeMo-RL/examples/configs/alpha/README.md` + `docs/SPEC_*.md` 4종 참조.

## 0. 요약

**RL 착수에 필요한 모든 전제 조건이 완료·검증됐다.** alpha_v2(15.08B hybrid GDN+MoE)가
NeMo-RL의 Megatron-Core 백엔드와 vLLM 롤아웃 양쪽에 통합되어 수치 검증을 통과했고,
RLVR 데이터·환경·토폴로지·커널 가속까지 준비 완료. 남은 것은 GRPO 레시피 yaml 작성과
8-GPU 노드에서의 최종 KL 게이트 확인뿐이다.

## 1. 검증 게이트 총괄 (전부 통과)

| # | 게이트 | 기준 | 결과 | 일자 |
|---|---|---|---|---|
| 1 | NeMo-RL 환경: GRPO 퀵스타트 (DTensor+vLLM) | 정상 학습 스텝 | 57s/step | 08-13 |
| 2 | NeMo-RL 환경: GRPO 퀵스타트 (**mcore**+vLLM) | 〃 | 44–50s/step, KL err ~7e-4 | 08-13 |
| 3 | AlphaBridge 가중치 라운드트립 (HF→mcore→HF) | 텐서별 `max_diff==0.0` + 양방향 커버리지 | **14,181/14,181 정확 일치** | 08-13 |
| 4 | forward 패리티 (mcore vs Pai검증 HF 참조) | argmax 일치 + cos ≥ 0.99 | 영/한/887tok 3종, cos ≥ 0.99988 | 08-13 |
| 5 | vLLM 플러그인 등록 체인 | 아키텍처 해석 | 통과 (stock 0.25.1 휠) | 08-13 |
| 6 | refit_verifier (mcore→vLLM 토큰별 logprob) | mult_prob_err < 1.05 | mean diff 0.020 (≈1.02) | 08-13 |
| 7 | RL 블렌드 구조 검증 | 전행 JSON/키/마스킹 잔여 0 | rlvr1/2_alpha 양쪽 OK | 08-13 |
| 8 | FlashQLA 커널 정합성 (fla naive 3자 대조) | 수치 일치 | fwd/bwd cos ≥ 0.9969 | 08-18 |
| 9 | forward 패리티 재실행 (**FlashQLA 활성**) | #4와 동일 | 3종 PASS, TileLang 실행 확인 | 08-18 |
| — | GRPO E2E KL 게이트 (Generation KL < 0.002) | **미실행** | 8-GPU 노드 첫 구동 시 | (예정) |

## 2. 타임라인

**08-12 — 조사·결정.** multinode 설치 스크립트의 FA3 빌드 OOM 해결(컨테이너 cgroup 232GiB vs
`nproc` 병렬 nvcc → `MAX_JOBS=10` 스크립트 반영). NeMo-RL 조사 후 프레임워크 결정 보고:
알고리즘 스택(GRPO/RM/MOPD)이 계획서(`SFT_RL_DATASETS.md`)와 정확히 일치.

**08-13 (오전) — 방향 검증·구조 확정.** 저장소 코드 레벨 검증: ① Nemotron-3-Ultra는
**Megatron-Core 학습 + 커스텀 vLLM 포크** 조합으로 학습됨(`ultra-v3` 브랜치에 전체 레시피 공개,
"SKIP_SGLANG_BUILD=1" 명시) — alpha가 복제하려는 레시피의 실전 선례. ② Qwen3-Next는 NeMo-RL
main에 미병합(로드맵 v0.7)이나 하부 스택(Megatron-Bridge, vLLM 0.25.1)은 이미 지원. ③ vLLM이
압도적 주류(예제 143 vs SGLang 3). → **트랙 B(mcore+vLLM) 주 경로 확정**, SGLang 배제.
프로젝트 구조 확정: Pai-Megatron-Patch 패턴 그대로 NeMo-RL fork + `examples/configs/alpha/` 허브,
커스텀 코드는 소유 레이어에(Megatron-Bridge fork, vLLM fork), HF 체크포인트가 두 스택의 유일한 인터페이스.

**08-13 (오후) — 환경 구축.** RL 전용 H100 세션에 NeMo-RL 환경 구축. 베이스 이미지(CUDA 12.8,
드라이버 535)와 NeMo-RL(torch cu130) 간 충돌 8건을 순차 해결 — 전부 `setup_nemo_rl_env.sh`에
인코딩(§NEMO_RL_SETUP.md 트러블슈팅 원장 12건). GRPO 퀵스타트 양 백엔드 통과로 환경 검증 완료.
GitHub 연동(gh 디바이스 로그인, fork 3종 remote, 브랜치 push).

**08-13 (저녁) — alpha 통합 + 검증.** 병렬 조사 3건(Pai 변환기 명세 / Megatron-Bridge 표면 /
NeMo-RL 통합 지점) 후 구현: **AlphaBridge**(qwen3_next 브리지 클론 + 델타 5개 — 문자열 소스 등록,
표준 RMSNorm, DSV3 라우터, expert_bias, MTP 제거; **24층 1:1 매핑**으로 Pai의 48층 분할 포팅 회피),
**vLLM `AlphaForCausalLM`**(플러그인 패키지 — fork-in-lock은 nemo-gym flashinfer 핀과 충돌해 기각;
표준 RMSNorm 전면 교체 + 융합 QK-norm 커널의 zero-centered 하드코딩 우회 + FusedMoE DSV3 인자).
게이트 3~6 전부 통과. RLVR 데이터 준비(§4)까지 완료.

**08-14 — 토폴로지 정정.** 사용자 지적으로 colocate 권고 폐기: **node0=학습 전용,
node1=vLLM 롤아웃 전용** + async GRPO + in-flight weight updates. long-context 학습 메모리와
롤아웃 KV를 각 노드가 온전히 사용, TCP refit ~28초는 스텝 시간 대비 무시 가능(오버랩).

**08-18 — FlashQLA 통합.** QwenLM/FlashQLA(TileLang GDN 커널) 평가:
alpha 기하 실측 **fwd 1.8~4.2×, fwd+bwd 1.6~3.7×** (49k에서 f+b 3.5×), 정합성 3중 검증 후
opt-in 통합(`ALPHA_GDN_BACKEND=flashqla`). 부수 발견: **fla 0.4.2 네이티브 GQA는 NaN**
(mcore의 repeat_interleave 확장이 우회 중 — upstream 보고 가치).

## 3. 주요 결정 기록 (결정 → 근거)

1. **RL 프레임워크 = NeMo-RL, 트랙 B(mcore 학습 + vLLM 롤아웃)** — Nemotron-3-Ultra 실전 선례
   (하이브리드+MoE 동형), 공개 레시피 존재, 기존 Megatron 자산 연속성. DTensor+HF는 폴백
   (modeling_alpha.py의 transformers 5.x 비호환으로 사실상 배제됨).
2. **mcore 측 24층 1:1 매핑** — HF 24층(믹서+MLP)≡mcore 24 TransformerLayer. Pai의 48층
   분할은 내부 표현일 뿐임을 규명, 브리지 복잡도 대폭 축소.
3. **vLLM 통합 = 플러그인 패키지** (`vllm.general_plugins`) — stock 휠 유지로 lock 충돌 제거,
   버전 업그레이드 rebase 무비용. in-tree 구현은 upstream 기여용으로만 유지(potatowarriors/vllm).
4. **2노드 분리 토폴로지 + async GRPO** — long-context 메모리 요구와 IB 부재 제약의 최적해
   (08-14 사용자 정정으로 확정).
5. **RLVR 환경 = NeMo Gym (Ultra 패턴)** — 데이터가 이미 Gym 행 형식(행별 `agent_ref` 라우팅),
   레시피 yaml이 로컬 존재. 1노드 스모크는 judge-free 서버 구성(nanov3 템플릿).
6. **FlashQLA = opt-in 기본 채택** — 검증된 3.5× 커널 이득, 시끄러운 실패 설계로 위험 차단.

## 4. RL 데이터 준비 상태

- 원천: Nemotron-Post-Training-v3 RL 26종(62GB) + 자체 `alpha-RL-Identity-Following-v1`(16,510행)
  — 전수 인벤토리·스키마·라이선스는 `docs/SPEC_rl_dataset_inventory.md`
- **Ultra 블렌드 7종이 이미 실행 형식**: 마스킹 수학 행 복원 완료(rlvr1/rlvr2/mopd 각 6,181행,
  DAPO/Skywork 소스), **identity 0.70% 주입**(시드 고정) → `alpha_blends/rlvr{1,2}_alpha.jsonl`
  (99,113 / 99,810행), 구조 검증 통과
- 주의: identity RL은 SFT identity 선행 필수 / GenRM-v1·Safety-v1은 RM 학습용(RL 뱅크 아님) /
  ARC-AGI 라이선스 pending-legal-review(블렌드 내 ~2%) / SWE·MOPD 196k vs alpha 128k 상한 미결

## 5. 산출물 맵

| 저장소·브랜치 | 핵심 커밋 | 내용 |
|---|---|---|
| `potatowarriors/RL` `alpha/post-train` | 6356314e6→0eb8e5579 | alpha 허브(레시피 스캐폴드·검증 도구 6종·SPEC 문서 4종·vLLM 플러그인·RL 데이터 도구·FlashQLA 통합·refit_verifier 수정) |
| `potatowarriors/Megatron-Bridge` `alpha/bridge` | 91b88c8f, bcc4e415 | AlphaBridge + FlashQLA opt-in 스왑 |
| `potatowarriors/vllm` `alpha/model` | (v0.25.1 베이스) | in-tree AlphaForCausalLM (upstream 기여용 기록) |
| 워크스페이스 루트 | — | `setup_nemo_rl_env.sh`(전 해법 인코딩)·`nemo_rl_env.sh`·`NEMO_RL_SETUP.md`·`RESTORE_AFTER_REBOOT.md` §6 |
| 데이터 (NFS) | — | `/home/work/Datasets/LL_datasets/posttraining/RL/alpha_blends/` |

검증 도구 (재사용 게이트): `verify_bridge_roundtrip.py` / `verify_forward_parity.py`(+`gen_hf_reference_logits.py`) /
`verify_rl_blend.py` / `inject_identity_blend.py` / `bench_flashqla.py` / NeMo-RL `tools/refit_verifier.py`(수정판)

## 6. 축적된 핵심 운영 지식 (재발 방지 상위 항목)

전체 12건은 `NEMO_RL_SETUP.md` §4 원장 참조. 특히 중요한 것:

- **zero-centered RMSNorm 함정 (2회 조우)**: Qwen3-Next 계열 코드는 mcore·vLLM 모두 `(1+w)`
  norm이 기본 — alpha는 표준 `w`. 가중치 검증은 통과하면서 forward만 조용히 깨지는 유형이라
  **forward 패리티 게이트가 필수** (Pai의 1p 사건과 동일 클래스)
- **uv 0.11.28 핀**: 신버전 uv로 lock 재생성 시 editable 메타데이터 소실 → Ray 워커 venv 실패
- **드라이버 535 + CUDA 13 스택**: forward-compat(595)을 Backend.AI 훅 경로(`compat/lib.real`)에
  배치해야 유효 — NGC 25.08 전환 시 자동 스킵되도록 스크립트가 프로브 방식
- **변환 캐시 비원자성**: 크래시 후 부분 캐시는 수동 삭제, 브리지 수정 중 `force_reconvert_from_hf`
- **fla 0.4.2 GQA NaN**: mcore의 MHA 확장이 유일한 방어선 — 커널 직접 호출 코드 작성 시 주의

## 7. 남은 작업

**D-day(~08-27) 전:**
1. `grpo_alpha_smoke.yaml` + RLVR 레시피 yaml 작성 (nanov3/ultra 템플릿, FlashQLA·CUDNN_HOME
   env_vars, 분리 토폴로지, async GRPO) — 로드맵 5번
2. Gym venv 프리페치 (실행 노드에서), litmus-bench 모니터링 연결 검토

**D-day 후 (8-GPU/2노드 확보 시):**
3. 최종 ckpt HF 변환(기존 evaluate.sh) → LC 단계 → SFT (Pai 스택, identity 주입 포함)
4. **alpha GRPO 첫 구동: Generation KL < 0.002 게이트** (마지막 미실행 검증) + FlashQLA E2E 스텝 이득 실측
5. RLVR P1(49k) → P2(65k) → teacher RL(축소 패널) → MOPD — 컨텍스트 상한(196k vs 128k) 결정 필요

**추적 항목:** NeMo-RL v0.7 릴리스(Qwen3-Next 지원 — 유지보수 부담 축소), fla GQA 버그 upstream 보고,
ARC-AGI 라이선스 확인, SGLang→vLLM 전환에 따른 기존 `sglang_alpha_model.py` 처분
