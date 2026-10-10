# alpha RL — Claude Code Guide

alpha_v2 (15.08B GatedDeltaNet + Attention + MoE 하이브리드, `model_type: "alpha"`) 의 RL 단계 허브다.
이 파일은 **지침만** 담는다. 상태는 [`docs/STATUS.md`](docs/STATUS.md), 문서 색인은 [`docs/README.md`](docs/README.md),
사고 서사는 [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md), 게이트는 [`docs/GATES.md`](docs/GATES.md).
작업 규칙(검증·커밋·문서·보고)은 `.claude/rules/alpha.md`. 새 사실은 docs 에 쓰고 여기에는 한 줄만 둔다.

## 모델 불변량 (RL 에 필요한 것만)

모델 상수의 정본은 `project_s/Pai-Megatron-Patch/examples/alpha/CLAUDE.md` "Baseline 수치" 표다.

| 항목 | 값 | RL 에서의 의미 |
|---|---|---|
| Total / Active params | 15.08B / 1.79B | — |
| 층 매핑 | HF 24층 = mcore 24 TransformerLayer (1:1) | Pai 내부의 48층 분할은 브리지와 무관 (`RL_PLAN.md` 결정 2) |
| TP / EP | **TP=1** (GDN 은 TP>1 미지원) / EP=8 (192 experts → 24/GPU) | 레시피에 그대로 |
| Norm | **표준 RMSNorm** (`w*x`) | Qwen3-Next 계열 코드는 zero-centered 기본 — 재사용 시 forward 만 조용히 깨짐 |
| Vocab (padded) | 163,968 | — |
| 종료 토큰 | `<\|im_end\|>` id 3 (chat 끝) · `<\|endoftext\|>` id 0 (EOD). HF `generation_config` eos = [3, 0] | 롤아웃 stop 조건 |
| Chat template | Pai `examples/alpha/tokenizer_v5/chat_template.jinja` (think·tool 규약은 Pai `docs/INTERLEAVED_THINKING.md`) | 보상·파서가 이 렌더를 가정 |
| 롤아웃 엔진 | **vLLM 플러그인만** — upstream mcore GDN 은 추론 미지원(`GDN does not support inference for now`) | mcore 네이티브 생성 백엔드 사용 불가. 학습측 logprob(teacher-forced)은 무관 |
| sequence packing | 스모크는 끔. **128K(CP>1)는 필수** | "GDN 이라 끔"은 과거 사정 — mcore GDN 은 cu_seqlens 로 상태를 리셋한다. alpha 실측은 G1·G2 (`docs/RLVR_READINESS.md` H2) |

## 레시피 규약 (모든 alpha GRPO 레시피)

- **`grpo_alpha_smoke.yaml` 을 상속한다** (`defaults:`). 여기에 alpha RL 기본값이 있다:
  `policy.generation.vllm_kwargs.mamba_ssm_cache_dtype: float32` + `policy.router_replay.enabled: true`. 둘을 끄면 R1 게이트 FAIL.
  `policy.generation.vllm_cfg.turn_end_token_id: 3` 을 끄면 Gym 멀티턴이 2번째 호출부터 깨진다 (R3).
- Gym 레시피는 `chat_template_kwargs: null`. Ultra 의 `truncate_history_thinking: false` 를 복사하지 않는다 (비도구 이력 reasoning 복원, R3).
- CP>1 은 sequence packing 필수, `use_fused_linear_logprobs` 와 비호환이다 → `logprob_chunk_size` + `fuse_loss` (+ `defer_fp32_logits: true`).
  packing 의 시퀀스 간 상태 격리는 G1 PASS(비트 동일). CP>1 경로의 수치는 미검증이다 — 장문맥 레시피 전에 R1 을 그 경로로 돌린다 (G2).
- **R3 를 켠 레시피에서 `moe_router_fusion` 금지.** mcore fused top-k 는 replay 전에 반환해 R3 가 에러 없이 꺼진다. Pai SFT 는 켰다 — 복사하지 않는다.
- **Gym 레시피는 `vllm_cfg.env_vars.VLLM_ENFORCE_STRICT_TOOL_CALLING: "0"` 필수.** 기본값이면 vLLM 이 `strict:true` 도구에 제약 디코딩을 걸어 롤아웃이 off-policy 가 된다 (G3).
- Ultra 의 `reward_penalties.token_ids`(2·12·13)는 Nemotron id 다 — alpha 에서는 `<|im_start|>`·`<tool_response>`·`</tool_response>`. alpha think 는 14·15.
- **레시피를 돌리기 전에 `tools/check_alpha_recipe.py <recipe>` 가 ERROR 0 이어야 한다** (H1·H3·H5·GBS·CP 패딩·eos 등 자동 검사).
- **W&B 에는 실제 학습 런만 올린다.** 스모크·게이트·속도 시험은 `WANDB=0`. 지표는 `logger.wandb.metric_allowlist`(부모 레시피)로 핵심만 보내고, 전체는 TensorBoard 에 남는다 (사용자 2026-10-08).
- Pai·Ultra 노브를 옮길 때는 NeMo-RL 이 실제로 읽는지 확인한다. `megatron_cfg` 최상위의 모르는 키와 Ultra 의 loss 정규화 키 4개는 조용히 무시된다.
  위험 목록과 게이트 계획은 [`docs/RLVR_READINESS.md`](docs/RLVR_READINESS.md).
- `policy.megatron_cfg.env_vars` 에 `CUDNN_HOME`(Megatron 워커 venv 의 pip cuDNN 경로) 필수.
- 브리지·플러그인 수정 중에는 `megatron_cfg.force_reconvert_from_hf: true` (변환 캐시에 버전 검사 없음).
- FlashQLA 는 opt-in: `policy.megatron_cfg.env_vars` 에 `ALPHA_GDN_BACKEND: "flashqla"`. 실패 시 시끄럽게 중단, 무설정이면 fla.
- GPU 장수는 `cluster.gpus_per_node`·Ray 리소스로 제한한다. **`CUDA_VISIBLE_DEVICES` 금지** (원장 #19).
- 모델 입력은 Pai `evaluate.sh` 경로로 변환·검증된 HF 체크포인트(`hfmodel_*`)다. 새 ckpt 는 `GATES.md` 재실행 조건을 따른다.
- **RL 체크포인트 반출은 `tools/export_rl_hf.sh` 로만 한다.** 변환기 출력 그대로는 Pai(transformers 4.57)가 토크나이저를 못 읽는다 (G6).
- 저장하는 레시피는 `megatron_cfg.scheduler.max_steps` 를 둔다 (≥ 실효 train_iters). 없으면 스텝 수를 바꿔 재개할 때 스케줄러 assert 로 멈춘다 (G7).
- Muon(`dist_muon`): 필드명·기본값이 Pai SFT 와 다르다 — `muon_nesterov`(기본 False), `muon_extra_scale_factor`(기본 1.0 = SFT 의 5배)를
  명시한다. "필수 off" 4개를 끄지 않으면 셋업이 실패한다 (`grpo_alpha_smoke_muon.yaml` 헤더).
- **lr: 1차 teacher 레시피는 `megatron_cfg.optimizer.lr`·`min_lr` 를 5e-6 으로 덮어쓴다** (결정 23, 도구 사용 teacher 만 3e-6).
  부모 `student_rlvr1_alpha.yaml` 의 1e-6 은 bf16 가중치에 갱신의 12% 만 반영된다. 새 lr 런은 step 50 에서 `tools/measure_bf16_delivery.py` 로 잰다.

## Quick Commands

```bash
cd /home/work/vidsearch/repos/project_s/NeMo-RL      # NRL_ROOT=/home/work/vidsearch/tools/nemo_rl
$NRL_ROOT/clean_run.sh $NRL_ROOT/bin/uv run --locked python examples/run_grpo.py \
  --config examples/configs/alpha/<recipe>.yaml policy.model_name=<hfmodel_00NNNNN> logger.log_dir=<dir> </dev/null
python examples/configs/alpha/tools/analyze_rollout_logprob_gap.py <dir>/exp_*/train_data_step*.jsonl   # KL 분해, CPU
```

게이트 명령은 [`docs/GATES.md`](docs/GATES.md), 설치·복원은 [`docs/SETUP.md`](docs/SETUP.md), 디렉토리 지도는 [`README.md`](README.md).

## 함정 표 — Known Issues 한 줄 요약

서사는 [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) (`#N` = 원장 번호). 새 사고는 거기에 쓰고 여기엔 한 줄.

| 날짜 | 증상 | 원인 → 대응 |
|---|---|---|
| 10-10 | rdkit 예/아니오 행(`bool` · `presence`)의 맞는 답 `((No))` 가 0 점 | 정답은 1/0 인데 형식 예시가 정수라 모델이 단어로 답함 → 플러그인이 그 행만 yes/no 도 읽는다. 새 채점기는 실제 롤아웃 출력으로 재채점해 본다 (단위 테스트로는 안 드러났다) |
| 10-10 | judge 를 쓴 런 뒤 GPU 2장이 judge vLLM 에 잡힌 채 남아 다음 기동이 GPU 점유 검사에서 멈춤 · `should_log_nemo_gym_responses: true` 게이트는 judge 판정 기록을 남기지 않음 | judge vLLM 은 Gym 서버 소유 Ray 액터 → `launch.sh` 가 런 액터 0 일 때 Gym 정리 뒤 GPU 검사. 판정 기록은 `WANDB_MODE=offline` + 표 켬 → `tools/analyze_gym_full_results.py` |
| 10-10 | J1 이 Gym 기동에서 rc=1 — `ns_tools finished unexpectedly` (`sandbox_port` 정수), 같은 기동에서 judge Gym venv 의 `import vllm` 이 `libcudart.so.13` 실패 | Gym 서버 설정은 서버 venv 의 pydantic 이 검증한다 → 문자열 필드는 따옴표 (검사기 ERROR). `local_vllm_model` venv 는 `torch-backend=auto` 가 cu129 를 골라 vLLM(cu13)과 어긋난다 → `UV_TORCH_BACKEND=cu130` 으로 따로 빌드 (`docs/KNOWN_ISSUES.md` 2026-10-10) |
| 10-10 | STEM 블렌드 Science-v1 636행은 완벽한 답도 보상 0 이 대부분 (탐침 14/60) | 비탐욕 `output_regex` 가 정답 속 닫는 문자(괄호 · LaTeX `\)` · `]`)에서 끊고, Gym 설정 파일이 재판정을 끈다 → 교체(v2) + 레시피 `check_full_generation_on_fail: true` (결정 25). judge 블렌드는 `check_alpha_recipe.py <recipe> data.train.data_path=<blend>` 로 정답 잘림을 검사한다 |
| 10-08 | 도구 사용 teacher(lr 1e-6)가 42스텝 동안 보상 평탄 — 정합 지표·학습 경로는 정상 | 스텝당 갱신 2e-7 이 bf16 반올림 경계(6e-5)보다 작아 50스텝 누적의 12% 만 bf16 에 반영 → lr 3e-6 (결정 22). 보상이 평탄하면 `tools/measure_bf16_delivery.py` 로 반영률을 잰다 |
| 10-08 | STEM 블렌드의 rdkit 행은 Gym 0.6.0 에 서버가 없고, Science-v1 도구 행은 ns_tools 에 채점기(`verifier_type`)가 등록돼 있지 않음 | 둘 다 학습을 띄워야 드러난다 → rdkit alpha 플러그인 · 레시피 `ns_tools.verifiers` 등록. `check_alpha_recipe.py` 가 데이터의 agent·verifier_type 을 Gym 설정과 대조한다 (블렌드를 바꾸면 다시 돌린다) |
| 10-08 | 레시피에 `grpo.reward_shaping`(DAPO 길이 감점)을 켜도 효과 없음 | async 경로가 호출하지 않는다 → 길이 보상은 `rollouts.py` 그룹 후처리에 (`docs/KNOWN_ISSUES.md`) |
| 10-08 | alpha vLLM `--data-parallel-size 8` 이 NCCL 초기화에서 `double free` 로 죽음 | vLLM DP 모드 문제 (Pai 문서에 이미 있음) → Pai `serve_fleet.sh`(GPU 당 단일 서버 + lb_proxy)·`stop_fleet.sh`. 서빙·벤치 경로는 Pai `SFT_BENCHMARKS.md` §2.5~2.6 이 정본 |
| 10-08 | NeMo-Skills sandbox 빌드 `vedas==0.0.1` 없음 | PyPI 에서 사라짐 → `sandbox.lock` 에서 그 줄 제거, `GITHUB_CI=1` (gpu06 `alpha-eval`, `docs/RL_DATA.md` §5.9) |
| 10-08 | vLLM `speculative_config.method=ngram` 이 엔진 기동에서 `Numba needs NumPy 2.4 or less` 로 죽음 | CPU 제안기의 numba 가 venv NumPy 2.5 미지원 → `method=ngram_gpu` 사용 |
| 10-08 | 같은 배치·같은 가중치로 학습 스텝을 두 번 돌려도 기울기가 1~1.6% 어긋남 (Z1) | backward 비결정성(MoE·bf16) → 기울기 비교 게이트는 같은 배치 반복으로 잡음 바닥을 재고 그 배수로 판정 |
| 10-08 | Gym 런의 `truncation_rate` 가 0 인데 실제로는 생성 상한에서 12~17% 잘림 | 지표가 `max_model_len` 을 채운 샘플만 셈 → 요청별 출력 상한 잘림도 세게 수정 (`78568c330`) |
| 10-08 | v2 덤프로 KL 분석하면 k3 8559 같은 무의미한 값 | loss 안 마스킹 런은 prev_logprob 패스를 건너뛰어 덤프 `prev_logprobs` 가 0 → 분석 도구 2종이 건너뛴다. 위치별 KL 이 필요한 구간만 `seq_logprob_error_in_loss=false` |
| 10-08 | 첫 RLVR 런 7스텝이 오류 없이 6.5시간 멈춤 — in-flight 요청 67개가 vLLM 엔진 1개에서 진척 0 | 원인 후보(교차 스레드 AsyncLLM 사용·문맥 초과 무응답) upstream 수정 이식, v2 는 진척 없음 감시 |
| 10-08 | 런이 끝나도 Gym 서버(`python app.py`)가 남아 132개 누적 | 별도 Ray 작업이라 드라이버와 함께 안 죽음 → v2 `launch.sh` 가 기동 전 `gym_cleanup.py` 로 정리 |
| 10-08 | 첫 RLVR 런이 레시피와 다른 보상으로 학습 — async 경로에서 effort shaping 누락 등 upstream 결함 4건 | upstream 수정 이식(`alpha/perf-fixes`) 뒤 재시작. 큰 런 전에 분기점 이후 upstream `fix(` 커밋을 훑는다 |
| 10-08 | 클러스터 노드에서 돌린 NeMo-RL 단위 테스트가 운영 클러스터에 붙음 | conftest 의 autouse `init_ray_cluster` 가 세션마다 연결 (`RAY_ADDRESS` 무관) → 클러스터가 빌 때만 pytest, CPU 테스트는 `CUDA_VISIBLE_DEVICES=""`. Ray 없는 순수 함수 테스트는 `--noconftest` 로 클러스터에 안 붙는다. 모듈 안 fixture 가 Ray 액터를 만드는 파일(`test_grpo.py`)은 그래도 붙는다 (10-08 재발) → `RAY_ADDRESS=local RAY_TMPDIR=$(mktemp -d /tmp/rut.XXXX)` 로 격리, 상태 API 실패는 변경 전후 비교 |
| 10-07 | 런 연장 재개에서 `OptimizerParamScheduler ... total number of weight decay iterations do not match` (G7) | 스케줄 길이가 train_iters(= max_num_steps)를 따라감 → 레시피 `scheduler.max_steps: 100000` |
| 10-07 | RL 반출 HF 를 Pai 가 못 읽음 `Tokenizer class TokenizersBackend does not exist` (G6) | 변환기가 transformers 5.8 로 메타데이터를 다시 씀 → `tools/export_rl_hf.sh` 로만 반출 (메타데이터는 시작점 복사, 가중치 대조) |
| 10-07 | 2노드 학습이 2스텝째 저장에서 멈춤 (`pidfd_getfd` · writer 사망, G5) | ES 메모리의 CUDA IPC 를 비동기 writer 가 못 받음 → `megatron_cfg.checkpoint.async_save: false` |
| 10-07 | Gym 도구 환경 KL 0.0045, 도구 시퀀스 1/3 이 mult_prob_error > 2 (G3) | vLLM 이 `strict:true` 도구에 xgrammar 제약 디코딩 → `VLLM_ENFORCE_STRICT_TOOL_CALLING=0` (strict 렌더는 유지) → KL 0.0017 |
| 10-07 | vLLM 기동 실패 `max_num_batched_tokens ... smaller than max_model_len` · 첫 턴 프롬프트 > max_model_len 이면 런 중단 | chunked prefill 끔이면 batched tokens = 최대 길이 · 블렌드 프롬프트 길이 사전 측정(첫 런 최대 39.3K, 32K 초과 1,144행) |
| 10-07 | colocated refit `pidfd_getfd: Operation not permitted` (G3) | ES 메모리의 CUDA IPC 를 컨테이너(ptrace_scope 1·CAP_SYS_PTRACE 없음)가 막음 → colocated 런은 ES 끔, 128K 는 2노드 분리(NCCL refit) |
| 10-07 | R4 결과 JSON 의 loss 가 전부 NaN · 128K 처리량 "15K tok/s" | 하네스가 `loss` 키를 읽음(반환은 `global_loss`) · 워밍업 스텝을 잼 → 수정, 정상 상태 33.4K tok/s (R5) |
| 10-07 | 장문맥 학습 스텝 `Triton Error [CUDA]: out of memory` (rank 당 16K 토큰) | PyTorch 캐시 단편화(reserved−alloc 13.6 GB)가 Triton 할당을 막음 → `expandable_segments:True` 로 128K/CP8 OK. D6 결정: 2노드 레시피의 학습 워커에 켠다 |
| 10-07 | Gym 멀티턴 `AssertionError: EOS token #0 not found in template_token_ids` | 턴 경계를 EOS(0)로 찾음 → `vllm_cfg.turn_end_token_id: 3` (본체 패치) |
| 10-07 | Gym 경로 렌더가 SFT 와 다름 (도구 정의) | Gym 이 `strict` 를 지움 → 유지 결정, Gym 플러그인 서버(`gym_plugins/`)로 구현 (G3) · `description` None 은 데이터 게이트 |
| 10-07 | 엔진 동등성 gradient cos 0.98 에서 "불일치"로 보임 | MoE·bf16 포화 — 절대 임계 대신 HF·섭동 기준선과 비교 (M5). 하네스엔 SFT forward 플래그(router fp32 등)를 다 준다 |
| 10-07 | `clean_run.sh` 아래서 환경변수가 안 먹음 (#23) | whitelist `env -i` → `clean_run.sh /usr/bin/env VAR=값 <cmd>` |
| 10-07 | 바쁜 GPU 위에 게이트 기동 (#25) · 실행 중 import 코드 편집으로 잡 사망 (#26) | 런처에 기동 직전 GPU 점유 검사(1 GiB) · 실행 중인 잡이 import 하는 코드는 편집 금지 · `pkill -f`·`pgrep -f` 는 자기 셸까지 맞힌다(exit 144, 10-08 재발) → 이름으로 정리 (`stop_fleet.sh`·Ray actor 이름) |
| 10-06 | GRPO KL 게이트 FAIL 0.0042 — 오차가 생성 위치를 따라 커짐 (#21) | vLLM GDN 재귀 상태 bf16 누적 + MoE 경계 라우팅 뒤집힘 → 레시피 기본값 fp32 상태 + R3 → 0.0014 |
| 10-06 | GRPO 로그로 Muon 적용이 안 보임 | mcore 가 Muon 경로 로그를 기본 숨김 → `verify_muon_optimizer.py` (R2) 로 옵티마이저 직접 검사 |
| 10-06 | 워커 `invalid device ordinal` (#19) | Ray 가 물리 GPU 번호로 set_device → `CUDA_VISIBLE_DEVICES` 쓰지 않음 |
| 10-06 | 워커가 Pai Megatron-LM 을 import (#18) | Pai `PYTHONPATH`·`~/.bashrc`(stdin 소켓) 누출 → `clean_run.sh` + `</dev/null` |
| 10-06 | Ray 워커에서만 libcudart 12/13 이중 로드 (#17) | flashinfer 가 import 시 절대경로 선로딩 → `CUDA_LIB_PATH=$CUDA_HOME/targets/x86_64-linux/lib` |
| 10-06 | #15 뒤에도 fused attn libcudart 이중 로드 (#16) | cudnn-frontend shim 이 soname 둘 다 dlopen → `LD_LIBRARY_PATH` 맨 앞 차단 디렉토리(ldblock) |
| 10-06 | TE 가 cu12 에 링크됨 (#15) | NGC 전역 `CUDACXX` → `CUDACXX=$CUDA_HOME/bin/nvcc` + uv 캐시 checkout 의 `CMakeCache.txt` 삭제 후 재빌드 |
| 10-06 | `uv sync --extra vllm` deep-ep 빌드 실패 (#20) | vllm extra 도 nvcc 13 필요 → 사설 CUDA 13 툴킷 |
| 08-18 | fla 0.4.2 네이티브 GQA(Hk≠Hv) NaN (#14) | fla 버그. mcore 는 MHA 확장으로 우회 → 커널 직접 호출 시 반드시 확장 후 |
| 08-13 | 가중치 검증 통과인데 forward 전멸 | Qwen3-Next 계열 zero-centered RMSNorm 기본값 → 표준 RMSNorm, M1·M2 게이트 함께 |
| 08-13 | `run_config.yaml not found` (#11) | 크래시가 남긴 부분 변환 캐시 → `$NRL_MEGATRON_CHECKPOINT_DIR/model_*` 삭제, 수정 중엔 `force_reconvert_from_hf` |
| 08-13 | lock 에 vLLM fork 연결 시 flashinfer 충돌 (#10) | 오버라이드가 Gym 분기까지 전파 → fork 는 lock 에 넣지 않고 플러그인 |
| 08-13 | Ray 워커 venv `build_editable` 실패 (#9) | 신버전 uv 가 editable 메타데이터 소실 → uv 0.11.28 고정 |
| 08-13 | HOME 49G 포화 (#2·#13) | uv·HF 캐시 기본 경로가 루프 디바이스 → 캐시를 `/opt`·NFS 로 |
| 08-13 | `refit_verifier.py` KeyError (#12) | 도구가 설정 스키마보다 낡음 → 우리 수정 `1eca2f383` |
| 08-13 | `--extra mcore --extra vllm` 충돌 (#6) | 의도된 설계(역할별 venv) → extra 별로 sync |
| 08-13 | 드라이버 535·CUDA 13 빌드 연쇄 (#1·#3~#5·#7·#8) | RL 전용 세션 경로의 해법은 `setup_nemo_rl_env.sh` 에 인코딩. 공존 설치는 `SETUP.md` §1 |
| 08-13 | `from vllm import LLM` 실패 | lock 의 openai 2.6.1 과 vllm 0.25.1 tool_parsers 불일치 — NeMo-RL 실경로 무관, 이 import 로 스모크하지 않음 |

## 환경 불변량·운영 함정

- **uv 는 0.11.28 만.** lock 재생성도 이 버전으로 (`.claude/rules/alpha-submodules.md`).
- **NFS venv 는 느리다** — GRPO setup 371 s vs 로컬 156 s (사용자 결정 2026-10-06: 원본 NFS·venv 로컬).
- **`main1`·`sub1` 은 역할 이름이지 물리 호스트가 아니다.** 호스트는 GPU UUID 로 식별 (Pai `CLAUDE.md` 환경 불변량).
  git push 는 main1 에서 동작한다 (sub1 은 자격 없음). 이 리포는 저장소 설정에 credential helper 가 없어
  `git -c credential.helper=store push origin alpha/post-train` 로 push 한다 (Pai 리포는 `.git/config` 에 `store` 설정, 2026-10-07 확인).
- 노드 간 InfiniBand 가 없다 (TCP ~9.1 Gbit/s). 2노드는 학습/롤아웃 분리 + async GRPO (`RL_PLAN.md` §3).
  **어느 노드가 학습·롤아웃을 맡는지는 Ray 가 기동마다 정한다** (v1·E0 학습 main1, E1a·E1b 학습 sub1). 로그의 워커 `ip=` 로 확인한다.
- Gym 은 CPU 전용이지만 Ray 를 띄운다. GPU 드라이버가 깨진 노드에서는 Backend.AI libcudahook 때문에 `ray.init()` 이 즉사한다.
- HOME(`/home/work`)은 49 GB 루프 볼륨이다. 큰 캐시·다운로드는 `/opt` 나 NFS 로.
- **비밀키**(HF·Gemini·Tavily)는 Pai `examples/alpha/.env`(gitignored). 묻지 말고 `set -a; source <그 파일>; set +a`. 값은 출력·커밋 금지.
