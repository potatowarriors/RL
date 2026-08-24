# alpha post-training recipes

alpha_v2 (15.08B hybrid GatedDeltaNet+MoE, `model_type: "alpha"`) post-training on NeMo-RL.
Nemotron-3-Ultra 레시피(`ultra-v3` 브랜치의 `examples/configs/ultra/`)를 참조 원형으로 삼는다.
환경·트러블슈팅은 워크스페이스 루트의 `NEMO_RL_SETUP.md` 참조.

## 통합 상태 (2026-08-13)

| 구성 요소 | 상태 | 검증 |
|---|---|---|
| Megatron-Bridge `AlphaBridge` (fork `alpha/bridge`) | ✅ | 라운드트립 14,181/14,181 텐서 정확 일치 |
| mcore forward 패리티 (vs Pai검증 HF 참조) | ✅ | 영/한/887토큰 3종, argmax·top5 일치, cos ≥ 0.99988 |
| vLLM `AlphaForCausalLM` (`vllm_alpha_plugin/`) | ✅ | 등록 체인 + refit_verifier 통과 (logprob mean diff 0.020) |
| GRPO E2E (Generation KL < 0.002) | ⬜ | 8-GPU 노드 첫 구동 시 확인 |
| vLLM **단독 서빙** (디스크 직접 로드) | ✅ (08-24) | `load_format=auto`로 hfmodel 로드 + greedy 생성 — 첫 토큰이 HF 참조와 일치(" Paris"/" 도쿄") |

## 디렉토리

| 경로 | 내용 |
|---|---|
| `vllm_alpha_plugin/` | vLLM 플러그인 패키지 (pyproject의 vllm extra + uv source로 연결됨) |
| `tools/verify_bridge_roundtrip.py` | HF→mcore→HF 텐서별 정확 일치 게이트 |
| `tools/verify_forward_parity.py` | mcore training-forward vs HF 참조 로짓 게이트 |
| `docs/SPEC_*.md` | Pai 변환기·Megatron-Bridge 분석 명세 (포팅 근거 문서) |

## RL 데이터 (2026-08-13 준비 완료)

Ultra RL 블렌드는 **이미 NeMo Gym 실행 형식**(행별 `agent_ref` 라우팅)으로 배포되어 있고,
레시피 yaml 7종도 트리에 존재한다 (`examples/nemo_gym/nemotron-3-ultra/`). 수행한 준비:

| 산출물 | 경로 (`/home/work/Datasets/LL_datasets/posttraining/RL/alpha_blends/`) | 내용 |
|---|---|---|
| `ultra_restored/*.jsonl` (7종) | 마스킹 수학 행 복원본 | `fill_placeholders.py`로 rlvr1/rlvr2/mopd 각 6,181행 복원 (DAPO/Skywork 소스) |
| `rlvr1_alpha.jsonl` | 99,113행 | 복원본 + **identity 689행(0.70%)** 주입, 시드 20260813 |
| `rlvr2_alpha.jsonl` | 99,810행 | 복원본 + identity 694행(0.70%), 시드 20260814 |

- 도구: `tools/inject_identity_blend.py`(비율 0.3~1.0% 강제, 시드 고정), `tools/verify_rl_blend.py`(전행 JSON/키/잔여 마스킹 검사 + agent 분포) — 두 블렌드 모두 **구조 검증 통과**
- 상세 인벤토리·스키마·라이선스: `docs/SPEC_rl_dataset_inventory.md` / 환경 배선: `docs/SPEC_nemo_rl_env_wiring.md`
- 주의: identity RL은 **SFT identity 주입 이후에만** 보상 신호가 생김 (콜드 정책은 alpha-banana를 모름); `Nemotron-RLHF-GenRM-v1`·`Safety-v1`은 RL 프롬프트 뱅크가 아니라 RM 학습용; `Nemotron-RL-ARC-AGI-v1`은 라이선스 `pending-legal-review` — 블렌드 내 nvarc 행(각 ~2%) 사용 전 법무 확인 필요
- 미결(후속 단계로 이관): Gym venv 프리페치(`examples/nemo_gym/prefetch_venvs.py`, 첫 Gym 실행 노드에서), Math-v2 복원(reasoning teacher), SWE 196k vs alpha 128k 컨텍스트 상한 결정, litmus-bench 모니터링 연결

## FlashQLA (GDN 커널 가속, 2026-08-18 검증·통합)

Qwen 팀의 TileLang 기반 GDN 커널(`flash-qla==0.1.2`, mcore extra에 포함). alpha 기하 실측:
**fwd 1.8~4.2×, fwd+bwd 1.6~3.7×** (시퀀스가 길수록 커짐 — 49k에서 f+b 3.5×). 정합성은
fla naive 참조·확장 MHA 경로·15B 실모델 forward 패리티 3중으로 검증됨.

- **활성화**: 레시피의 `policy.megatron_cfg.env_vars`에 `ALPHA_GDN_BACKEND: "flashqla"`
  (브리지 fork의 `alpha/__init__.py`가 임포트 시 커널 스왑; 실패 시 시끄럽게 중단, 무설정 시 기존 fla)
- 벤치 도구: `tools/bench_flashqla.py` (3구성: fla-MHA / qla-MHA / qla-네이티브GQA)
- ⚠️ **fla 0.4.2 자체 버그 발견**: 네이티브 GQA(Hk≠Hv) 호출은 NaN — mcore는 호출 전
  repeat_interleave 확장으로 우회하고 있어 실경로는 안전. fla upstream 보고 예정
- vLLM 롤아웃측(prefill 한정 이득)은 후순위 — vllm 코드 변경 필요, 별도 검토

## 레시피 (계획)

| 파일 | 단계 | 상태 |
|---|---|---|
| `grpo_alpha_smoke.yaml` | 8-GPU 노드 GRPO 드라이런 + KL 게이트 | TODO (다음 단계) |
| `student_rlvr1.yaml`, `student_rlvr2.yaml` | RLVR (GRPO, SFT 체크포인트에서 시작) | TODO |
| `ifbench_teacher.yaml` 등 | 전문 teacher RL (2~3개로 축소 예정) | TODO |
| `mopd.yaml` | 멀티 teacher on-policy distillation | TODO |

## 전제·주의

- 학습 백엔드: **Megatron-Core** / 롤아웃: **vLLM(플러그인)**. mcore 네이티브 생성 백엔드는
  GDN 추론 미지원으로 alpha에 사용 불가
- 모델 입력: Pai-Megatron-Patch `evaluate.sh` 경로로 변환된 HF 체크포인트 (`hfmodel_*`)
- SFT/LC단계는 기존 Pai-Megatron 스택에서 수행 (프로젝트 결정, 2026-08-13)
- 레시피 작성 시 `policy.megatron_cfg.env_vars`에 `CUDNN_HOME`(mcore 워커 venv의 pip cuDNN 경로) 필수
- 브리지/플러그인 수정 중에는 `megatron_cfg.force_reconvert_from_hf: true` (변환 캐시에 버전 검사 없음)
- **아키텍처 주의**: alpha는 표준 RMSNorm — Qwen3-Next 계열 코드(mcore·vLLM 모두)의
  zero-centered 기본값을 재사용하면 가중치 검증은 통과하고 forward만 조용히 깨진다.
  체크포인트·코드 변경 시 반드시 `tools/`의 두 게이트를 재실행할 것
- 클러스터: Backend.AI (Slurm 없음) — `ray start` 수동 기동. **2노드 토폴로지 (2026-08-14 확정)**:
  **node0 = 학습 전용, node1 = vLLM 롤아웃 전용** (`generation.colocated.enabled: false`) +
  **async GRPO + `in_flight_weight_updates`**. long-context RLVR에서는 학습 상태만으로 node0이
  꽉 차므로 분리가 필수이고, async로 두 노드가 동시 가동된다. 노드 간 TCP(~9.1Gbit/s)를 건너는
  가중치 refit ~28초는 스텝 시간(수십 분) 대비 무시 가능하며 in-flight로 생성과 오버랩됨.
  colocate(1노드)는 짧은 컨텍스트 스모크 전용. LLM-judge/teacher가 필요한 스테이지(RLHF
  teacher, MOPD)에서는 node1 GPU를 rollout과 judge가 분할 — 해당 스테이지 설계 시 재배분
