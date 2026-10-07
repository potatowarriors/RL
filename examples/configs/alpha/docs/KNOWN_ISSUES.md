# alpha RL — Known Issues & Fixes

alpha RL 단계(NeMo-RL)의 사고·수정 기록 전문이다 (최신순). [`../CLAUDE.md`](../CLAUDE.md) 의 "함정 표"가 이 문서의 한 줄 요약이다.
새 사고는 **여기에 서사를 쓰고 함정 표에는 한 줄만** 추가한다. 날짜는 절대 표기한다.
2026-10-07 이관: 워크스페이스 `project_s/NEMO_RL_SETUP.md` §4 원장 21건과 Pai `KNOWN_ISSUES.md` 10-06 항목의 RL 측 서사를 옮겼다.
pre-train·SFT·벤치 쪽 사고는 Pai `examples/alpha/docs/KNOWN_ISSUES.md` 가 정본이다.

## 첫 RLVR 런이 레시피와 다른 보상으로 학습됐다 — async 경로의 upstream 결함 4건 (2026-10-08 ✅ 이식, 브랜치 `alpha/perf-fixes`)

**발견 경위**: 첫 RLVR 런 구간 1 의 병목 판정 뒤 최신 NeMo-RL(upstream main 8bf6bd4ba, 분기점 이후 268커밋)을 검토하다 찾았다.
네 건 모두 upstream 에 수정이 있고, 우리 브랜치(분기점 2026-08-12)에는 없었다. 코드로 직접 확인했다.

| # | 결함 | 영향 | upstream 수정 |
|---|---|---|---|
| 1 | async 수집기가 `run_async_nemo_gym_rollout` 에 `effort_config` 를 넘기지 않는다 (`trajectory_collector.py:818-840`, 동기 경로 `grpo.py:2957` 은 넘김) | 레시피의 `env.nemo_gym.effort_levels`(Ultra 계수)가 조용히 무시된다. 마커 프롬프트(블렌드 3,429행)의 긴 응답에 감점이 없었다 — 4스텝 "efficient" math 44,735 토큰 무감점 | 4c6e5c84e (#3885) |
| 2 | async 루프가 `max_num_steps` 만 본다 (`grpo.py:4499`) | 1에폭 런은 데이터가 떨어질 때 "dataloader exhausted" RuntimeError 로 끝나 마지막 저장 뒤 최대 9스텝을 잃는다. `training_info.json` 의 `total_steps` 도 0 이었다 | f49e41dda (#3248) |
| 3 | LOO + `normalize_rewards` 가 std 를 E[r²]−E[r]² 로 구한다 (`algorithms/utils.py:189-195`) | 그룹 보상이 모두 같은 비이진 값이면 fp32 잡음 std 로 나눠 advantage 가 튄다. #1 을 고치면 보상이 비이진이 된다 | 8535caefc (#4158) |
| 4 | in-flight refit 중 vLLM 을 멈추지 않는다 | upstream 은 refit 과 forward 의 경쟁 조건이 NaN logprob·KL 튐을 낸다고 설명한다. 우리 KL 은 정상이었다 (미검증) | 4f653c4dc (#3839) |

**대응**: 네 건을 `alpha/perf-fixes` 에 이식했다 (각 커밋에 REBASE NOTE). #2 는 upstream 이 기대는 재개 lookahead 인프라 없이 핵심만 옮겼다.
첫 런은 warmup 10스텝(lr 1e-7~1e-6) 뒤 처음부터 다시 시작한다 (사용자 결정 2026-10-08). 같은 검토에서 upstream #4171(loss 안 시퀀스 마스킹)도 이식했다.
충돌을 풀며 깨끗이 합쳐진 파일이 이 브랜치에 없는 모듈을 import 하는 경우를 찾았다 (`draft_config.coerce_draft_config` — 그대로면 `grpo.py` 전체 ImportError).
그래서 이식 뒤에는 바뀐 파일 전체를 `ruff --select F821,F401` 로 검사한다.

**교훈**: fork 가 upstream 에서 두 달 떨어지면 정확성 수정도 함께 놓친다. 큰 런 전에 분기점 이후 upstream 의 `fix(` 커밋을 우리 경로(async·Gym·Megatron) 기준으로 훑는다.

## NeMo-RL 단위 테스트가 실행 중인 운영 Ray 클러스터에 붙는다 (2026-10-08)

**발견 경위**: 이식 검증으로 main1 에서 `test_vllm_generation.py` 를 돌렸다. 통합 테스트 하나가 placement group 대기 시간 초과로 실패했는데,
`ray list placement-groups` 에 그 테스트의 `vllm-test-policy-cluster-separate-node0` 가 운영 클러스터 기록으로 남아 있었다 (REMOVED).
원인은 `tests/unit/conftest.py:403` 의 session autouse fixture `init_ray_cluster`(`init_ray()`)다. Ray 를 쓰지 않는 테스트를 골라 돌려도
세션마다 같은 노드의 실행 중 클러스터에 연결하고(로그 "Connecting to existing Ray cluster at address: 10.0.37.4:6379"), GPU 모니터(`ray_gpu_monitor`)도 띄운다.
`RAY_ADDRESS` 를 지워도 같다. 이번에는 GPU 가 모두 런에 잡혀 있어 테스트가 자원을 못 얻고 끝났다. 런이 끝나 GPU 가 비는 순간에 돌았다면 테스트가 GPU 를 가로챌 수 있었다.

**대응**: 운영 클러스터가 떠 있는 노드에서는 NeMo-RL pytest 를 돌리지 않는다 — 클러스터가 비어 있을 때만 돌린다. CPU 단위 테스트는 `CUDA_VISIBLE_DEVICES=""` 로 GPU 를 가린다.
운영 런이 쓰는 venv 도 건드리지 않는다 — pytest 는 scratchpad 의 별도 venv 에 드라이버 venv 의 site-packages 를 `.pth` 로 이어 붙여 쓴다.

## 런을 연장해 재개하면 Megatron 스케줄러 assert 로 멈춘다 — 스케줄 길이가 max_num_steps 를 따라간다 (2026-10-07 ✅ `scheduler.max_steps`)

**발견 경위**: G7(저장 → 재개) 첫 시도에서 G5 의 step_3 를 `grpo.max_num_steps` 3 → 5 로 재개했다 (런 연장 시나리오).
MegatronPolicyWorker 8개가 초기화에서 `AssertionError: OptimizerParamScheduler: class input value 640 and checkpointvalue 384 for total number of weight decay iterations do not match` 로 죽었다.

**원인**: GRPO 는 `megatron_cfg.train_iters = min(max_num_steps, max_num_epochs × 에폭당 스텝)` 을 넣는다 (`nemo_rl/algorithms/grpo.py:1055`).
`scheduler.max_steps` 가 없으면 Bridge 는 `wd_incr_steps = train_iters × GBS` 로 계산한다 (`megatron/bridge/training/config.py` `_calculate_scheduler_steps`).
mcore `OptimizerParamScheduler.load_state_dict` 는 기본값(`override_opt_param_scheduler`·`use_checkpoint_opt_param_scheduler` 모두 false)에서 체크포인트 값과 다르면 assert 한다.
384 = 3 × 128, 640 = 5 × 128 이다. 스텝 수·에폭·블렌드 크기 중 하나라도 바꾸고 재개하면 같은 assert 가 난다.

**대응**: 레시피에 `policy.megatron_cfg.scheduler.max_steps: 100000` 을 둔다. Bridge 는 이 값으로 wd 증가·lr 감쇠 길이를 정해 train_iters 와 떼어 놓는다 (≥ train_iters 여야 한다).
lr·wd 가 상수(lr = min_lr = 1e-6, wd 상수)라 수치는 같다. lr·warmup·wd 값을 바꾸고 재개하면 여전히 assert 한다 — 의도된 보호다.
`check_alpha_recipe.py` 는 미설정을 WARN 으로, 실효 train_iters 보다 작은 값을 ERROR 로 막는다. 이 수정 전에 저장한 스모크 체크포인트(G5)는 감쇠 길이가 달라 이 레시피로 재개할 수 없다.

**교훈**: 저장 → 재개 게이트는 같은 설정 재개만이 아니라 런 연장 시나리오로 돌린다.

## RL 체크포인트를 HF 로 반출하면 Pai 가 토크나이저를 못 읽는다 — 변환기가 메타데이터를 transformers 5 형식으로 다시 쓴다 (2026-10-07 ✅ `tools/export_rl_hf.sh`)

**발견 경위**: G6(RL 체크포인트 → HF → Pai)에서 G5 step_3 를 NeMo-RL 변환기(`examples/converters/convert_megatron_to_hf.py`)로 반출했다.
가중치는 정상이었다. 텐서 14,181개의 이름·shape 가 일치했고, 동결된 라우터·expert bias 는 비트 동일했다. 그런데 Pai 환경(transformers 4.57)에서 반출본의 토크나이저를 열면
`ValueError: Tokenizer class TokenizersBackend does not exist or is not currently imported.` 로 실패했다. Pai 의 forward_sanity·벤치·다음 단계 입력이 모두 여기서 멈춘다.

**원인**: 변환기는 Megatron 워커 venv 의 transformers 5.8.1 로 tokenizer·config 를 다시 저장한다.
- `tokenizer_config.json`: `tokenizer_class` 가 `PreTrainedTokenizerFast` 에서 transformers 5 전용 이름 `TokenizersBackend` 로 바뀐다. `added_tokens_decoder`·`additional_special_tokens` 가 빠진다.
- chat template 이 `tokenizer_config.json` 밖의 `chat_template.jinja` 로 빠진다 (끝 개행만 다르다). `special_tokens_map.json`·`tokenizer_metadata.json` 은 쓰지 않는다.
- `config.json` 은 키가 바뀐다 (`_full_attention_interval`, `rope_parameters`·`layer_types` 추가, `dtype`). 그래도 Pai 의 `configuration_alpha.py` 는 같은 값으로 해석한다 (`compare_hf_forward.py --raw-config` 의 속성 차이는 `rope_parameters` 추가뿐이다).
- `A_log` 18개가 fp32 대신 bf16 로 저장된다. 값은 같다. Pai 도 bf16 로 학습한 값을 fp32 로 저장했을 뿐이라 bf16 왕복이 18/18 무손실이다. HF `modeling_alpha.py` 는 `A_log.float()` 로 쓴다.
- 문서의 변환 명령(`uv run --locked --extra mcore`)은 driver venv 의 패키지 4개를 바꾸고 81개를 설치한다. driver 를 쓰는 런이 돌 때 실행하면 그 런의 환경이 바뀐다.

**대응**: 반출은 `tools/export_rl_hf.sh <ckpt>/step_N <out>` 로만 한다. 이 스크립트는 세 단계로 돈다.
① 변환기를 Megatron 워커 venv 의 python 으로 직접 실행한다 (CPU·gloo, step_3 에서 2분 17초).
② 가중치가 아닌 파일은 시작점 hfmodel(`policy.model_name`)에서 그대로 복사하고, 시작점에 없는 파일(`chat_template.jinja`)은 지운다.
③ `compare_hf_weights.py` 로 시작점과 대조한다. 출처(step·가중치 경로·커밋)는 `rl_export.json` 에 남는다.

**교훈**: 두 스택의 인터페이스는 HF 체크포인트 하나다. 가중치 검증만으로는 부족하다. 소비자(Pai) 환경에서 토크나이저와 config 까지 열어 본다.

## 2노드 학습이 2스텝째 체크포인트 저장에서 멈춘다 — 비동기 writer 가 ES 메모리의 CUDA IPC 를 받지 못한다 (2026-10-07 ✅ `async_save: false`)

**발견 경위**: G5(실레시피 2노드 스모크) 1스텝은 정상이었다(KL 0.0016). 2스텝째 저장(`save_period 2`)에서 학습 워커 8개의 비동기 writer
(nvidia-resiliency-ext `PersistentAsyncCaller`)가 `Failed to receive CUDA IPC handle from the training process (pidfd_getfd: Operation not permitted)` 로 죽었다.
학습 프로세스는 저장 마무리를 기다리며 8분 넘게 GPU 0% 로 멈췄다.

**원인**: colocated refit 사고(아래)와 같다. 상주 writer 는 학습 텐서를 CUDA IPC 로 받는데, 128K 학습에 필요한 `expandable_segments` 메모리는
`pidfd_getfd` 가 있어야 열린다. 이 컨테이너는 `ptrace_scope=1`·`CAP_SYS_PTRACE` 없음이다. `async_save: true` 는 `grpo_math_1B.yaml` 상속값이었다.

**대응**: 레시피 `policy.megatron_cfg.checkpoint.async_save: false` (동기 저장 — 학습이 저장 동안 멈추지만 IPC 를 쓰지 않는다).
`check_alpha_recipe.py` 가 ES + 비동기 저장 조합을 ERROR 로 막는다. 대안(호스트에서 `kernel.yama.ptrace_scope=0`)은 컨테이너 밖 권한이라 쓰지 않는다.

**교훈**: ES 를 켜면 같은 노드 안의 프로세스 간 CUDA 메모리 공유 경로(colocated refit, 비동기 저장)가 전부 막힌다. 새 기능을 켤 때 IPC 를 쓰는지 먼저 본다.

## Gym 도구 환경의 롤아웃이 off-policy 다 — vLLM 이 `strict: true` 도구에 제약 디코딩을 건다 (2026-10-07 ✅ `VLLM_ENFORCE_STRICT_TOOL_CALLING=0`)

**발견 경위**: G3(Gym 1노드 스모크, judge 불필요 10개 환경 138행)의 Generation KL 이 0.0043·0.0045 로 기준 0.002 의 2배를 넘었다.
같은 모델의 네이티브 경로(R1·G2)는 0.0015 였다. `seq_logprob_error_threshold 2` 가 64개 중 17~21개 시퀀스를 마스킹했다.

**진단** (`tools/analyze_gym_logprob_gap.py`, 토큰 덤프 = `env.should_log_nemo_gym_responses=false`):

| 환경 | 시퀀스 | mult_prob_error > 2 | 중앙값 (강제 켬 → 끔) |
|---|---|---|---|
| toolcall_schema 단일 스텝 도구 | 20 | 14 → 0 | 6.53 → 1.015 |
| swe_pivot | 8 | 2 → 0 | 1.24 → 1.007 |
| single_step_tool_use | 8 | 1 → 0 | 1.11 → 1.022 |
| 도구 없는 6개 환경 | 28 | 0 → 0 | 1.02~1.03 (변화 없음) |

- 어긋남은 도구를 선언한 행에만 있었고, 특수 토큰이 아니라 추론 텍스트 중간에서 시작했다.
- R3 는 원인이 아니다 — forward 검증 3,840건 불일치 0, CP 토큰 일치 256,662행.
- 로그에 xgrammar 마스크 커널(`apply_token_bitmask_inplace_kernel`) 컴파일이 있었다 (강제 끔 런에서는 0회).

**원인**: vLLM 0.25.1 `tool_parsers/structural_tag_registry.py::get_model_structural_tag` 는 `tool_choice` 가 auto 여도 도구 하나라도
`strict: true` 면 구조 태그(xgrammar)를 만든다. 환경변수 `VLLM_ENFORCE_STRICT_TOOL_CALLING` 기본값이 True 다 (`vllm/envs.py`).
결정 13 으로 남긴 `strict: true`(RL 블렌드 도구 행 전부)가 롤아웃을 제약된 분포에서 샘플링하게 만들었고, vLLM 이 돌려준 logprob 도 그 분포 기준이었다.
학습측은 제약 없는 분포로 계산하므로 importance ratio 가 틀리고, threshold 마스킹이 도구 샘플을 대량으로 버린다.

**대응**: `policy.generation.vllm_cfg.env_vars.VLLM_ENFORCE_STRICT_TOOL_CALLING: "0"` (레시피 기본값). `strict` 는 프롬프트 렌더에 그대로 남는다 —
도구 행 프롬프트 36/36 에 `<strict>True</strict>` 확인. 결과 KL 0.0017, 마스킹 0, 도구 환경 k3 0.0006~0.0011. `check_alpha_recipe.py` 가 누락을 ERROR 로 막는다.

**교훈**: ① 서빙 계층의 "편의 기능"(구조 태그·제약 디코딩·파서)이 RL 에서는 샘플링 분포를 바꾼다. 롤아웃 경로를 바꾸면 R1 식 KL 을 그 경로에서 다시 잰다.
② KL 평균만 보지 말고 시퀀스·환경별로 쪼갠다 — 이번 결함은 전체의 1/3 인 도구 시퀀스에만 있었다.

## vLLM 엔진이 `max_num_batched_tokens (16384) is smaller than max_model_len` 으로 기동하지 못한다 — chunked prefill 을 끈 레시피 (2026-10-07 ✅ 레시피 수정)

**발견 경위**: G3 를 32K 로 올리자 vLLM async 워커가 `SchedulerConfig` 검증에서 죽었다.
**원인**: `student_rlvr1_alpha.yaml` 은 R3 route 누락을 피하려고 chunked prefill 을 끈다(D2, upstream R3 레시피와 같음). chunked prefill 이 꺼지면
vLLM 은 프롬프트 하나를 한 스텝에 넣어야 하므로 `max_num_batched_tokens >= max_model_len` 을 요구한다. 레시피 값 16384 는 128K 본 런에서도 실패한다.
**대응**: `vllm_kwargs.max_num_batched_tokens: ${policy.max_total_sequence_length}`. `check_alpha_recipe.py` 가 이 조합을 ERROR 로 막는다.

## NeMo-RL 은 첫 턴 프롬프트가 `max_model_len` 을 넘는 Gym 행을 건너뛰지 않고 런을 멈춘다 (2026-10-07, 데이터 게이트)

**발견 경위**: G3 를 8K 로 돌렸더니 프롬프트가 8K 를 넘는 행(스모크 144행 중 15행)에서 vLLM 이 400 을 돌려줬다. NeMo-RL 은
`ValueError: NeMo Gym returned a result with no generation data ... the prompt for the first turn already exceeds the vLLM max_model_len` 로 런 전체를 멈췄다 (`nemo_rl/environments/nemo_gym.py:826`).
**함의**: 블렌드의 렌더 프롬프트 길이를 런 전에 잰다. `rlvr1_alpha_judgefree.jsonl` 은 최대 39,266 토큰(중앙값 3,381, p99 33,972, 32K 초과 1,144행 — 전부 SWE 피벗, 64K 초과 0행)이라 128K 첫 런에는 해당하지 않는다.
측정 함정: Responses API 입력의 `function_call_output` 본문은 `output` 필드, 이전 추론은 `reasoning.summary` 다. 첫 측정은 이 둘을 빠뜨려 최대 22,503 으로 과소 측정했다.
도구 호출 `arguments` 는 JSON 문자열을 dict 로 바꿔야 alpha 템플릿이 렌더된다 (`tools/measure_blend_prompt_lengths.py`).
멀티턴 환경은 턴이 쌓이며 넘을 수 있으므로 환경을 추가할 때 다시 본다.

## colocated refit 이 `pidfd_getfd: Operation not permitted` 로 실패한다 — `expandable_segments` 메모리의 CUDA IPC 를 컨테이너가 막는다 (2026-10-07 ✅ colocated 에서는 ES 끔)

**발견 경위**: G3(Gym 1노드 colocated 스모크)가 첫 refit 에서 죽었다. vLLM async 워커(EngineCore)가 학습 워커의 CUDA IPC 핸들을 열다가
`weight load failed: RuntimeError: pidfd_getfd: Operation not permitted; missing keys (14181)` 를 냈다. 같은 1노드 colocated 인 R1·G2 는 통과했다.

**원인**: 레시피 `student_rlvr1_alpha.yaml` 이 정책 워커에 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`(결정 D6 권고값)를 켠다.
ES 로 잡은 메모리를 다른 프로세스가 CUDA IPC 로 열려면 수신 측이 `pidfd_getfd` 로 내보낸 쪽의 파일 기술자를 가져와야 한다.
이 컨테이너는 `kernel.yama.ptrace_scope=1` 이고 `CAP_SYS_PTRACE` 가 없다. vLLM EngineCore 는 학습 워커의 자손이 아니라서 거부된다.
R1·G2 는 ES 를 켜지 않아 일반 cudaMalloc IPC 를 썼다.

**대응**: colocated(IPC refit) 런에서는 ES 를 끈다 (`policy.megatron_cfg.env_vars=null`). 2노드 분리 토폴로지는 NCCL broadcast 로 refit 하므로 IPC 를 쓰지 않는다 — G5 에서 확인한다.
`tools/check_alpha_recipe.py` 가 colocated + ES 조합을 ERROR 로 막는다. 128K 학습(R4: ES 없으면 rank 당 12K 토큰 상한)은 분리 토폴로지 전제다.

**교훈**: 할당자 설정은 학습 메모리만이 아니라 프로세스 간 메모리 공유(refit) 경로까지 바꾼다. NeMo-RL 주석의 "ES 는 weight transfer 5배 느림"보다 이 환경에서는 "colocated IPC 불가"가 실제 제약이다.

## R4 결과 JSON 의 loss 가 전부 NaN 이고 처리량도 과소 측정됐다 — 하네스가 없는 키를 읽고 워밍업 스텝을 쟀다 (2026-10-07 ✅ 수정)

**발견 경위**: 이식 위험 보고(`RLVR_READINESS.md`)를 쓰며 R4 산출물을 다시 읽었다. 5개 조건(16K/CP2 ~ 128K/CP8) 전 rank·전 스텝의 `loss` 가 NaN 이었다.

**원인**: `tools/measure_train_memory.py` 가 `res.get("loss", nan)` 으로 읽는다. `MegatronPolicyWorkerImpl.train` 은 `global_loss`·`grad_norm` 을 반환한다
(`megatron_policy_worker.py:1004-1011`). 키가 없어 기본값 NaN 이 찍혔다. 수치 문제가 아니다.

**함의**: R4 는 메모리·실행 가능성만 증명했다. 128K/CP8 에서 loss·grad norm 이 유한한지는 본 적이 없다.

**같은 하네스의 두 번째 오류 — 처리량**: R4 의 2번째 스텝 시간(8.5~8.7 s)을 정상 상태로 읽어 128K 처리량을 약 15K tok/s 로 보고했다.
R5(2026-10-07)에서 3 스텝을 재니 정상 상태는 샘플 1개/스텝 5.47 s, 4개/스텝 15.7 s(33.4K tok/s)였다. 2번째 스텝에도 커널 autotune 등 워밍업이 남아 있었고,
샘플 1개 스텝은 옵티마이저·동기화 고정비가 샘플당으로 붙는다.

**교훈**: 게이트 하네스는 읽는 키가 실제로 있는지 assert 한다. 기본값으로 덮인 지표는 "측정 안 함"과 구별되지 않는다.
처리량은 워밍업이 끝난 스텝(3번째 이후)과 여러 마이크로배치 스텝으로 잰다.

## 장문맥 학습 스텝 OOM 은 총량이 아니라 단편화다 — PyTorch 캐시가 쥔 빈 블록 때문에 Triton 할당이 실패한다 (2026-10-07, ES 로 128K/CP8 통과 · 기본값 채택은 결정 대기)

**발견 경위**: R4 메모리 실측(sub1, Muon 레시피, 학습 전용 노드 가정)에서 rank 당 16K 토큰 조건이 길이·CP 와 무관하게 전부 OOM 이었다(32K/CP2 · 64K/CP4 · 128K/CP8).
rank 당 12K 토큰(96K/CP8)까지는 통과했다. 수치표는 [`GATES.md`](GATES.md) R4.

**진단**:
- 예외가 `RuntimeError: Triton Error [CUDA]: out of memory` 였다. PyTorch OOM 이 아니다. GDN 커널(fla Triton·FlashQLA TileLang)은 PyTorch 캐싱 할당자 밖에서 메모리를 잡는다.
  PyTorch 자신의 할당은 OOM 직전에 캐시를 비우고 재시도하지만, 바깥 할당에는 그 기회가 없다.
- 96K/CP8 에서 이미 reserved − alloc 이 13.6 GB 였다(67.6 − 54.0). 캐시가 쥔 빈 블록이 Triton 몫을 막는다.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`(ES) 하나로 128K/CP8 이 통과했다. reserved − alloc 은 2.3 GB 로 줄었다.
  `logprob_chunk_size` 를 2048 → 1024 로 줄이면 peak 가 3.8 GB 더 내려가지만 필수는 아니다.
- 메모리 구성(`torch.cuda.memory._record_memory_history` 스냅샷, 16K/CP2 rank 0): 고정 바닥 27.0 GB 가 가중치·grad·Muon 상태다.
  나머지 대부분은 RL loss 의 `ChunkedDistributedLogprob.backward`(`nemo_rl/distributed/model_utils.py:267`)가 차지한다.
  `grad_input = zeros_like(logits)`(:347, T×V), 저장된 logits, chunk 별 fp32 softmax, int64 `one_hot`(:365) 이다. Muon 상태는 지배 항이 아니다.

**함의**:
- Muon optimizer-state offload(Pai 기능 #4) 포팅은 1노드 128K/CP8 에 필요 없다. 2026-10-06 추정(바닥 ≈45 GB, 오프로드 필요 가능성)을 실측 27 GB 가 대체한다.
- ES 에는 대가가 있다. NeMo-RL 주석(`megatron_policy_worker.py:2742`)은 ES 가 weight transfer 를 약 5배 느리게 한다고 적는다. upstream Megatron 레시피 다수는 `expandable_segments:False` 를 명시한다.
  alpha 구성에서의 refit 비용은 미측정이다. RL 기본값 채택은 결정 대기다 (`STATUS.md`).
  - 2026-10-07 재확인: "5배" 근거는 `_clear_fp8_caches` docstring 한 줄이다(FP8 단편화 맥락, 측정값·전송 방식 없음). 2노드 분리 refit 은 NCCL broadcast 라 TCP 대역이 지배한다.
    Pai SFT 는 `train.sh:82` 에서 ES 를 항상 켰다. 같은 docstring 은 `max_split_size_mb:512` 를 대안으로 든다 (G4 에서 대조).

**같이 드러난 NeMo-RL 제약** (전부 `nemo_rl/models/megatron/setup.py`):
1. CP>1 은 sequence packing 이 필수다 (:839-840 assert). alpha 레시피 불변량 "packing 끔"(GDN)과 충돌한다.
   장문맥 RL 은 packing 을 켜야 하고, packing+CP 경로의 GDN 수치(R1 KL·M5)는 미검증이다.
2. CP>1 은 `use_fused_linear_logprobs` 와 함께 쓸 수 없다 (:843-844). 대신 `logprob_chunk_size` + `sequence_packing.fuse_loss` 를 쓴다 (Ultra 레시피 값).
3. `logprob_chunk_size` 는 `defer_fp32_logits: true` 를 요구한다 (:1130).

**교훈**: ① PyTorch 밖에서 할당하는 커널이 있는 모델은 `max_memory_allocated` 가 여유로워도 OOM 이 난다. reserved 와 alloc 의 차이를 같이 본다.
② 메모리 추정은 실측으로 대체한다. 오프로드가 필요하다는 추정은 틀렸다.

## NeMo Gym 멀티턴이 2번째 모델 호출부터 전부 실패한다 — 턴 경계를 EOS(id 0)로 찾았다 (2026-10-07 ✅ `turn_end_token_id`)

**발견 경위**: R3 렌더 게이트의 Gym 경로 P4 가 0/40 이었다. 이전 턴 토큰을 이어 붙이는 2번째 호출부터 전부
`AssertionError: EOS token #0 not found in template_token_ids` 로 실패했다.
R1 KL 게이트는 단일 턴 네이티브 경로라 이 경로를 지나지 않는다.

**원인**: NeMo-RL `replace_prefix_tokens`(`nemo_rl/models/generation/openai_server_utils.py`)는 다음 턴 요청의 템플릿 토큰에서 이전 턴 끝을 `tokenizer.eos_token_id` 로 찾는다.
그 자리에 모델이 실제로 낸 토큰을 이어 붙인다. alpha 의 `eos_token` 은 `<|endoftext|>`(0, 사전학습 문서 경계)이고 chat 턴 끝은 `<|im_end|>`(3)다.
템플릿 렌더에는 0 이 없으므로 경계를 찾지 못한다. 롤아웃 정지 토큰 {0, 3} 은 맞았다 — 정지와 경계는 다른 문제다.

**수정 (본체, rebase 충돌 후보)**: `replace_prefix_tokens(..., turn_end_token_id=None)` 인자를 추가했다. `None` 이면 기존 동작이다.
vLLM 워커가 `policy.generation.vllm_cfg.turn_end_token_id` 를 넘긴다 (`vllm_worker_async.py`, 설정 키 `vllm/config.py`).
alpha 기본값은 `grpo_alpha_smoke.yaml` 의 `turn_end_token_id: 3` 이다. 단위 테스트 2건을 추가했고 `test_openai_server_utils.py` 7/7 pass 다.
R3 P5(실제 패치 호출)에서 경계 문제는 사라졌고, 남은 차이는 도구 정의뿐이다.

**교훈**: EOS 와 턴 끝이 다른 토크나이저에서는 NeMo-RL 의 "EOS = 턴 끝" 가정을 의심한다. 정지 토큰 확인만으로는 멀티턴 경계를 보장하지 못한다.

## Gym → vLLM 렌더 경로가 SFT 형식과 3군데 다르다 — `strict` 제거 · `description` None · 비도구 이력 reasoning 복원 (2026-10-07, `strict` 유지 결정 · 구현 대기)

R3 렌더 게이트가 찾았다. 턴 경계(위 항목)를 고친 뒤 남는 차이들이다.

| # | 차이 | 영향 | 처리 |
|---|---|---|---|
| 1 | Gym `VLLMModel._strip_hosted_only_tool_fields`(`responses_api_models/vllm_model/app.py:471`)가 도구 정의의 `strict` 를 무조건 지운다 | SFT Agentic-v2 interactive 셋은 `<strict>True</strict>` 를 렌더한 형식으로 학습했다. RL 블렌드 도구 행 13,190/13,190 이 `strict: true` 다. 다른 SFT 도구 셋(kotool 등)에는 strict 가 없다 | **사용자 결정 2026-10-07: 유지.** 미구현. 방법: Gym 0.6.0 플러그인 루트(`NEMO_GYM_EXTRA_ROOTS`)에 `VLLMModel` 서브클래스 서버를 두고 `strict` 가 bool 이면 남기고 null 이면 지운다 — Gym fork 불필요. 선례는 Gym 의 `responses_api_models/vllm_model_with_compaction`. 구현 뒤 R3 P6 을 실경로로 재실행 |
| 2 | description 없는 도구를 vLLM 이 `<description>None</description>` 으로 렌더한다 (`FunctionDefinition.model_dump()` 가 None 을 남김) | RL 블렌드 0건 | 데이터 게이트: RL 환경 데이터에 description 필수 검사 |
| 3 | 도구 없는 멀티턴 롤아웃(c3b)에서 이전 턴 reasoning 이 이력에 복원된다. SFT 는 비도구 이력의 reasoning 을 strip 해 학습했다 | 현 블렌드의 해당 환경 영향 없음 | 구조적. 멀티턴 비도구 환경을 추가할 때 R3 재실행 |

**설정 권고**: alpha Gym 레시피는 `chat_template_kwargs: null` 로 둔다. Ultra 레시피의 `truncate_history_thinking: false` 를 복사하면 도구 없는 이력의 reasoning 까지 복원돼 SFT 규약과 어긋난다 (R3 의 `dataset/ultra` 케이스).

## 엔진 동등성은 절대 임계가 아니라 노이즈 기준선으로 판정한다 — 하네스의 라우터 bf16 누락도 기준선이 잡았다 (2026-10-07 ✅ M5 PASS)

**경위**: M5(SFT 엔진 Pai ↔ RL 엔진 NeMo-RL) 첫 하네스에서 Pai 측 라우터가 bf16 으로 계산됐다.
Pai `tools/alpha_config.py emit-megatron-flags` 는 가중치 검증용이라 forward 설정(`--moe-router-dtype fp32`, router·permute fusion, dropout 0)을 내보내지 않는다.
SFT 구성(Pai `configs/model/baseline_48L.yaml`)대로 플래그를 더해 다시 쟀다. 잘못된 런은 버리지 않고 섭동 기준선으로 썼다 (`$NRL_ROOT/gates/engine_parity_iter2400/pai_router_bf16_harness_bug/`).

**판정 원리**: MoE·bf16 에서는 같은 엔진끼리도 gradient cos 중앙값이 0.98 근처에서 포화한다. 토큰이 적게 몰린 expert 는 라우팅 하나만 뒤집혀도 gradient 가 크게 바뀐다.
0.999 같은 절대 임계를 쓰면 같은 엔진끼리도 FAIL 이다. 그래서 HF(제3 구현)와 섭동 런을 기준선으로 두고, Pai↔NeMo-RL 거리가 그 안에 있는지 본다. 수치는 [`GATES.md`](GATES.md) M5.

**교훈**: ① 동등성 게이트에는 같은 측정을 구현만 바꾼 기준선을 둔다. ② 가중치 검증용 설정 도구를 forward·학습 하네스에 그대로 쓰지 않는다.
③ GDN `A_log` 는 SFT 에서도 bf16 으로 계산했으므로 RL 과 차이가 없다 (확인 완료).

## GDN 재귀 상태가 vLLM 에서 bf16 으로 저장된다 — 디코드 토큰마다 반올림이 누적돼 rollout-vs-train logprob 이 생성 위치에 따라 벌어진다 (2026-10-06 ✅ RL 레시피 기본값 fp32)

원장 #21. 벤치 fleet·채팅·SDG 교사 서빙(전부 bf16 상태)의 처리 결정은 Pai 쪽 미결이다 — Pai `KNOWN_ISSUES.md` 10-06 항목과 Pai `STATUS.md` "열린 사용자 결정".

**발견 경위**: NeMo-RL alpha 8-GPU GRPO KL 게이트(agentic iter2400, `grpo_alpha_smoke.yaml`, main1, EP8·colocated·16×8 롤아웃·최대 4096 토큰)가 Generation KL 0.0042 / 0.0037 / 0.0040 으로 기준 0.002 FAIL.
같은 환경의 1.5B dense 스모크는 0.0007, 같은 ckpt 의 refit 검증(10 토큰)은 mean(exp|Δ|) 1.0293 PASS 였다.

**진단 방법**: NeMo-RL 이 step 마다 남기는 `train_data_step*.jsonl` 의 토큰별 `generation_logprobs`(vLLM)·`prev_logprobs`(mcore)를 오프라인 분해(`tools/analyze_rollout_logprob_gap.py`, CPU). 재계산 k3 KL 이 게이트 출력과 일치(0.00423 = 0.0042).
seed 가 같아 런 사이 롤아웃이 동일하다(생성 토큰 440,577 · Avg Reward 0.1953 동일) → 설정 하나만 바꾼 통제 대조. 산출 `$NRL_ROOT/gates/agentic_iter2400/`.

| 런 (Adam) | KL step 1 / 2 / 3 | \|Δ\|>0.5 토큰 | 위치 0–256 | 1k–2k | 3k–4k | mult_prob_err |
|---|---|---|---|---|---|---|
| ① 기본 | 0.0042 / 0.0037 / 0.0040 | 0.248% | 0.0024 | 0.0044 | 0.0050 | 1.050 |
| ② + R3 (Router Replay) | 0.0026 / 0.0022 / 0.0024 | 0.056% | 0.0014 | 0.0026 | 0.0032 | 1.036 |
| ③ + R3 + GDN 상태 fp32 | **0.0015 / 0.0013 / 0.0014** | 0.011% | 0.0013 | 0.0015 | **0.0015** | 1.024 |

**원인은 둘이다.**
1. **MoE 라우팅 뒤집힘 (꼬리 오차)** — vLLM 과 mcore 의 hidden state 가 앞단 커널 차로 bf16 수준 달라 top-8 경계 expert 가 갈린다. 라우터 자체 정밀도는 문제가 아니다: 연산은 mcore fp64(NeMo-RL 기본 `moe_router_dtype`)·vLLM 플러그인 fp32, `expert_bias` 는 양쪽 fp32, refit 은 dtype 무변환 전송(라운드트립 게이트가 dtype 변화를 FAIL 로 잡는데 14,181/14,181 통과). R3 로 해결.
2. **GDN 재귀 상태 bf16 저장 (위치 누적)** — vLLM `mamba_ssm_cache_dtype` 기본 `"auto"` 는 재귀 상태에 conv/KV 캐시 dtype = 모델 dtype(bf16)을 쓴다(vLLM 0.25.1 `model_executor/layers/mamba/mamba_utils.py:91-92`). 디코드는 토큰마다 상태를 읽고·갱신하고·bf16 으로 다시 쓰므로 반올림이 다음 토큰으로 넘어가 쌓인다. 학습(mcore fla chunked)은 한 forward 안에서 상태를 fp32 로 들고 간다. fp32 로 바꾸자 위치 프로파일이 평탄해졌다(③, step 2·3 도 0.0011~0.0015).

**적용 (사용자 지시 2026-10-06)**: alpha RL 레시피 기본값에 `policy.generation.vllm_kwargs.mamba_ssm_cache_dtype: float32` + `policy.router_replay.enabled: true` (`grpo_alpha_smoke.yaml`, 모든 alpha GRPO 레시피가 상속). NeMo-RL upstream 의 Mamba 계열 레시피(Nemotron nano·super·ultra 등 12종)는 전부 float32 를 지정한다 — alpha 레시피에만 빠져 있었다.
Muon 레시피(`grpo_alpha_smoke_muon.yaml`)도 같은 기본값으로 0.0015 / 0.0013 / 0.0013 PASS.

**교훈**: ① 시간축으로 누적되는 상태(GDN·Mamba·KDA)는 **저장 dtype** 이 정확도를 좌우한다. 토큰마다 새로 계산하는 값(라우터 logits)과 달리 오차가 다음 토큰으로 넘어간다.
② 짧은 검증은 이 누적을 못 본다 — forward 패리티는 teacher-forced prefill(디코드 없음), refit 검증은 10 토큰 디코드였다. 디코드 길이에 따른 **위치별 지표**로 본다.
③ 오차의 원인이 하나라고 가정하지 않는다 — R3 만 켰을 때 0.0026 이라 "거의 됐다"로 오판할 수 있었다. 꼬리(라우팅)와 위치 누적(상태 dtype)은 분해해야 갈린다.

## Qwen3-Next 계열 코드의 zero-centered RMSNorm 기본값 — 가중치 검증은 통과하고 forward 만 조용히 깨진다 (2026-08-13 ✅, 2회 조우)

alpha 는 **표준 RMSNorm**(`w*x`)인데 Qwen3-Next 계열 구현은 Megatron-Bridge(mcore)와 vLLM 양쪽 모두 zero-centered(`(1+w)x`)가 기본이다.
이 델타를 놓치면 **가중치 검증은 100% 통과하면서 forward 가 전멸**한다 (Pai 의 1p 사건 2026-05-26 과 동일 클래스).
AlphaBridge 와 vLLM 플러그인 포팅에서 각각 한 번씩 조우했다. 플러그인은 표준 RMSNorm 으로 전면 교체하고,
융합 QK-norm 커널은 zero-centered +1.0 이 하드코딩돼 있어 비활성화했다.
**대응**: 체크포인트·브리지·플러그인을 바꾸면 라운드트립과 forward 패리티 게이트를 **함께** 돌린다 (`GATES.md` 1·2).

## 환경 구축·운영 원장 #1~#27 (2026-08-13 RL 전용 세션 · 2026-10-06~07 Pai 컨테이너 공존)

증상으로 검색한다. #1~#14 의 해법은 `project_s/setup_nemo_rl_env.sh`(RL 전용 세션용)에, #15~#20 은
`project_s/setup_nemo_rl_env_noninvasive.sh`(Pai 컨테이너 공존, 현행 기본)에 인코딩돼 있다. 설치 절차는 [`SETUP.md`](SETUP.md).

| # | 증상 | 원인 | 해법 |
|---|---|---|---|
| 1 | `The NVIDIA driver on your system is too old (found version 12080/12020)` | 드라이버 535는 CUDA 12.x 세대, NeMo-RL은 torch cu130 | `cuda-compat-13-2` 설치. 단 Backend.AI `libcudahook`(/etc/ld.so.preload)이 **하드코딩 경로 `/usr/local/cuda/compat/lib.real`** 에서 libcuda를 선점 로드하므로 LD_LIBRARY_PATH는 무효 — `compat/lib.real/`에 595 라이브러리 + `lib→lib.real` 심볼릭 링크 + `.checked` 마커 구조를 재현해야 함 |
| 2 | uv 설치 중 `No space left on device` | `/home/work`는 49G 루프 디바이스, uv 캐시 45G+ | `UV_CACHE_DIR=/opt/uv-cache` (오버레이 로컬 디스크) |
| 3 | deep-ep 빌드: `detected CUDA version (12.8) mismatches ... (13.0)` | 시스템 nvcc 12.8로 cu130 대상 빌드 | CUDA 13.0 툴킷 병행 설치, `CUDA_HOME=/usr/local/cuda-13.0` |
| 4 | `fatal error: cuda/std/array: No such file` | CUDA 13에서 CCCL(libcu++) 헤더가 `include/cccl/`로 이동 | `CPATH=$CUDA_HOME/include/cccl` |
| 5 | `ptxas error: Feature 'elect' requires .target sm_90` (compute_75로 빌드됨) | uv 빌드 샌드박스에서 GPU 미감지 → 구형 아키텍처 폴백 | `TORCH_CUDA_ARCH_LIST=9.0a`, `NVTE_CUDA_ARCHS=90` 명시 |
| 6 | `uv sync --extra mcore --extra vllm` → `incompatible with the declared conflicts` | 의도된 설계 (역할별 venv) | extra별로 따로 sync |
| 7 | TE fused attn: `Multiple libcudart libraries found: .so.12 and .so.13` | cu12 빌드 휠(cudnn frontend 등)이 ldconfig의 CUDA 12 경로에서 libcudart.so.12를 로드 | `/usr/local/cuda`→13.0 전환 + `988_cuda-12.conf`/`gds-12-8.conf`를 `/etc/ld.so.conf.d/disabled/`로 이동 + ldconfig (NGC와 동일 상태 재현) |
| 8 | TE fused attn: `cuDNN Error: No valid engine configs` | 시스템 cuDNN 9.8(cu12)이 로드됨 (TE는 9.20 기준) | `CUDNN_HOME`/`LD_LIBRARY_PATH`를 워커 venv의 pip cuDNN으로 고정 (`nemo_rl_env.sh`에 포함) |
| 9 | Ray 워커 venv 구성: `nemo-gym build_editable` 실패 (`egg_base 'cache' does not exist`) | **uv 버전 불일치.** 신버전 uv(0.12.x)로 `uv lock` 재생성 시 editable 패키지의 `[package.metadata]`가 소실됨 | **uv는 반드시 0.11.28** (NeMo-RL `docker/Dockerfile`의 `UV_VERSION` 핀). 이 버전으로 재잠금하면 diff가 최소화됨. 스크립트가 핀 설치함 |
| 10 | 로컬 vLLM fork를 lock에 연결 시 `flashinfer-python` 충돌 (`nemo-gym[vllm]`은 0.6.12 핀) | 소스 오버라이드가 워크스페이스 전체(gym 분기 포함)에 전파 | **fork를 lock에 넣지 말 것.** vLLM 커스텀 모델은 플러그인 패키지로 통합 (`../README.md` 디렉토리 표) |
| 11 | mcore 변환 캐시 로드: `run_config.yaml not found` | 이전 실행 크래시가 남긴 **부분 변환 캐시** (캐시 검사는 디렉토리 존재만 확인) | `$NRL_MEGATRON_CHECKPOINT_DIR/model_*` 삭제 후 재실행. 브리지 코드 수정 중엔 `force_reconvert_from_hf: true` |
| 12 | `tools/refit_verifier.py` 각종 KeyError/AssertionError | 도구가 최신 설정 스키마 대비 낡음 | 수정 커밋 1eca2f383 (스키마 키 5종 + trust_remote_code + 정책 오프로드 + 단일GPU 메모리 설정) — 우리 브랜치에 반영됨 |
| 13 | `/home/work` 49G 재포화 (HF 모델/데이터셋 캐시) | HF 캐시 기본 경로가 소용량 루프 디바이스 | `~/.cache/huggingface` → `/opt/hf_cache/huggingface` 이전 + 심볼릭 링크 (세션 재생성 시 재수행 필요) |
| 14 | fla 0.4.2 `chunk_gated_delta_rule` 네이티브 GQA(Hk≠Hv) **NaN** | fla 자체 버그 (배치·varlen 모두) | mcore는 호출 전 repeat_interleave MHA 확장으로 우회 중이라 실경로 안전. 커널 직접 호출 코드에서는 반드시 확장 후 호출. upstream 보고 예정 |
| 15 | TE fused attn `Multiple libcudart libraries found: libcudart.so.12 and libcudart.so.13` (Pai 컨테이너) — `libtransformer_engine.so` 가 `libcudart.so.12`·`libcublas.so.12` 를 NEEDED | NGC 이미지 전역 `CUDACXX=/usr/local/cuda/bin/nvcc` → TE 의 CMake 단계가 12.9 nvcc·라이브러리를 씀(파이썬 단계는 `CUDA_HOME` 을 따르므로 착시). torch cpp_extension 빌드(DeepEP·mamba 등)는 무관 | `CUDACXX=$CUDA_HOME/bin/nvcc`·`LIBRARY_PATH=$CUDA_HOME/lib64/stubs`. **재빌드 시 uv 캐시 git checkout 의 `build/cmake/CMakeCache.txt` 를 지워야 한다** — 휠 캐시만 지우면 11 초 만에 같은 cu12 링크로 다시 나온다 |
| 16 | #15 해결 뒤에도 fused attn 에서 같은 예외 | cudnn-frontend shim(`cudnn_frontend_shim.h::load_cudart_so`)이 `libcudart.so.12`·`.13` 을 soname 으로 둘 다 dlopen, 둘 다 열리면 예외. `.so.12` 는 시스템 `ld.so.cache`(`000_cuda`·`cuda`·`988_cuda-12.conf`)로 열림 — LD_LIBRARY_PATH 정리로는 못 막음 | `LD_LIBRARY_PATH` 맨 앞에 **차단 디렉토리**(내용이 ELF 가 아닌 `libcudart.so.12`·`libcudart.so` 파일)를 둔다 → 해당 이름 dlopen 이 실패, 캐시로 넘어가지 않음. 프로세스 한정 |
| 17 | Ray Megatron 워커에서만 #16 재발 (단독 프로브는 통과) | `flashinfer/jit/__init__.py:102-105` 가 import 시 `$CUDA_LIB_PATH/libcudart.so.12`(기본 `/usr/local/cuda/targets/x86_64-linux/lib/`)를 **절대경로**·`RTLD_GLOBAL` 로 선로딩 | `CUDA_LIB_PATH=$CUDA_HOME/targets/x86_64-linux/lib` (cu13 툴킷엔 `.so.12` 가 없어 조건 거짓) |
| 18 | 워커 `ImportError: ... experimental_attention_variant_module_specs (…/Pai-Megatron-Patch/backends/megatron/Megatron-LM-251125/…)` | Pai 의 `PYTHONPATH` 가 Ray 워커로 누출. 또 **bash 는 stdin 이 소켓이면 비대화형이어도 `~/.bashrc` 를 source** 하고, 그 10행이 `~/.pai_megatron_alpha_env` 를 무조건 로드 → `env -i` 후에도 자식 bash 에서 되살아남 | 화이트리스트 `env -i`(BACKENDAI_*·NVIDIA_*·LD_PRELOAD·NCCL_CUDA_PATH·HOME 등) + **`</dev/null`** + `unset PYTHONPATH`·`PYTHONNOUSERSITE=1`. 같이 새던 것: `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`(logprob 정밀도), `OMP_NUM_THREADS=220`, py3.12 user site |
| 19 | 워커 `CUDA error: invalid device ordinal` (`torch.cuda.set_device(local_rank)`) | NeMo-RL 은 `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` 로 워커에 전 GPU 를 보이고 `ray.get_gpu_ids()[0]`(물리 번호)로 set_device. 드라이버에 `CUDA_VISIBLE_DEVICES=7` 을 주면 Ray 는 7 을 주는데 워커엔 0 번만 보임 | **`CUDA_VISIBLE_DEVICES` 로 GPU 부분집합을 지정하지 않는다.** 장수 제한은 `cluster.gpus_per_node`·Ray 리소스로 |
| 20 | `uv sync --extra vllm` 이 `deep-ep` 빌드 실패(`detected CUDA version 12.9 mismatches 13.0`) | vllm extra 도 deep-ep 소스 빌드를 포함 — 롤아웃 venv 에도 nvcc 13 필요 | `SETUP.md` §1 의 사설 CUDA 13 툴킷 |
| 21 | alpha 8-GPU GRPO KL 게이트 FAIL(Generation KL 0.0042, 기준 0.002) — 오차가 생성 위치를 따라 커짐 | ① vLLM `mamba_ssm_cache_dtype` 기본 auto = GDN 재귀 상태 bf16 → 디코드마다 반올림 누적 ② MoE top-8 경계 라우팅 뒤집힘 | alpha 레시피 기본값 `policy.generation.vllm_kwargs.mamba_ssm_cache_dtype: float32` + `policy.router_replay.enabled: true` → 0.0015/0.0013/0.0014 PASS. 진단·대조표는 이 문서의 GDN 재귀 상태 항목 |
| 22 | `verify_muon_optimizer.py` 초판이 OmegaConf `${mul:...}` 해석 실패 · 8 rank 가 전부 GPU 0 · rank 0 만 보고 PASS | 도구 결함 3건: resolver 미등록, `set_device` 누락, layer-wise optimizer 는 dense 파라미터를 rank 별로 나누는데 rank 0 만 검사함 | `register_omegaconf_resolvers()` · `torch.cuda.set_device(LOCAL_RANK)` · all-gather 로 전 rank 합집합 판정 (R2, 2026-10-06) |
| 23 | `clean_run.sh` 아래에서 `MEM_SNAPSHOT_DIR`·`PYTORCH_CUDA_ALLOC_CONF` 같은 환경변수가 무시된다 | `clean_run.sh` 는 whitelist `env -i`(#18)라 호출 측 export 가 지워진다 | `clean_run.sh /usr/bin/env VAR=값 <cmd>` 로 whitelist 안쪽에서 주입한다 (2026-10-07) |
| 24 | `pytest` 를 어느 venv 에서도 못 찾는다 | lock 의 런타임 venv 에 test 의존성이 없다 | `$NRL_ROOT/clean_run.sh $NRL_ROOT/bin/uv run --locked --with pytest python -m pytest <test> </dev/null` (2026-10-07) |
| 25 | 2026-10-07 11:57 GPU 게이트를 벤치 fleet 이 쓰던 main1 GPU 위에 기동했다. 약 20 초 뒤 중단했고 fleet 은 재기동 없이 health 8/8 이었다 | 기동 전 점유 확인 누락. 정리 중 `pgrep -f` 가 자기 셸까지 매칭했다(exit 144) | GPU 런처는 기동 직전 점유를 검사해 한 장이라도 1 GiB 이상이면 중단한다 (`$NRL_ROOT/gates/*/run_*_sub1.sh` 패턴). 프로세스 정리는 PID 를 확인한 뒤 한다 |
| 26 | 실행 중인 게이트가 `SyntaxError` 로 죽었다 | 그 잡이 import 하는 NeMo-RL 소스(`openai_server_utils.py`)를 실행 도중 편집했다. NFS 공유 워킹트리라 즉시 반영된다 | 실행 중인 잡이 import 하는 코드는 편집하지 않는다 (2026-10-07) |
| 27 | 셋업 1차 실행 뒤 HOME 에 uv 캐시 24 GB 가 생겼다 (HOME 여유 43 → 20 GB) | 셋업 스크립트가 생성한 `clean_run.sh` 의 heredoc 인용 실수로 `NRL_ROOT` 가 빈 값이 됐고, 캐시가 기본 경로(HOME)로 갔다 | 생성 시점에 절대경로를 박는다. 스크립트 수정 후 처음부터 재실행해 검증했다(1151 s, rc=0). HOME 산출물은 삭제해 복구 (2026-10-06) |

기타: `from vllm import LLM`은 lock의 openai 2.6.1과 vllm 0.25.1 tool_parsers 불일치로
깨져 있으나 **NeMo-RL의 실제 vLLM 사용 경로는 무관** (GRPO 스모크로 확인). vLLM 단독
스모크 테스트를 이 임포트로 하지 말 것.
