# alpha RL — Known Issues & Fixes

alpha RL 단계(NeMo-RL)의 사고·수정 기록 전문이다 (최신순). [`../CLAUDE.md`](../CLAUDE.md) 의 "함정 표"가 이 문서의 한 줄 요약이다.
새 사고는 **여기에 서사를 쓰고 함정 표에는 한 줄만** 추가한다. 날짜는 절대 표기한다.
2026-10-07 이관: 워크스페이스 `project_s/NEMO_RL_SETUP.md` §4 원장 21건과 Pai `KNOWN_ISSUES.md` 10-06 항목의 RL 측 서사를 옮겼다.
pre-train·SFT·벤치 쪽 사고는 Pai `examples/alpha/docs/KNOWN_ISSUES.md` 가 정본이다.

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

## 환경 구축 원장 #1~#21 (2026-08-13 RL 전용 세션 · 2026-10-06 Pai 컨테이너 공존)

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
| 21 | alpha 8-GPU GRPO KL 게이트 FAIL(Generation KL 0.0042, 기준 0.002) — 오차가 생성 위치를 따라 커짐 | ① vLLM `mamba_ssm_cache_dtype` 기본 auto = GDN 재귀 상태 bf16 → 디코드마다 반올림 누적 ② MoE top-8 경계 라우팅 뒤집힘 | alpha 레시피 기본값 `policy.generation.vllm_kwargs.mamba_ssm_cache_dtype: float32` + `policy.router_replay.enabled: true` → 0.0015/0.0013/0.0014 PASS. 진단·대조표는 이 문서 맨 위 항목 |

기타: `from vllm import LLM`은 lock의 openai 2.6.1과 vllm 0.25.1 tool_parsers 불일치로
깨져 있으나 **NeMo-RL의 실제 vLLM 사용 경로는 무관** (GRPO 스모크로 확인). vLLM 단독
스모크 테스트를 이 임포트로 하지 말 것.
