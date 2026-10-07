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
| sequence packing | 끔 (GDN, Qwen3.5 레시피와 동일) | — |

## 레시피 규약 (모든 alpha GRPO 레시피)

- **`grpo_alpha_smoke.yaml` 을 상속한다** (`defaults:`). 여기에 alpha RL 기본값이 있다:
  `policy.generation.vllm_kwargs.mamba_ssm_cache_dtype: float32` + `policy.router_replay.enabled: true`. 둘을 끄면 R1 게이트 FAIL.
- `policy.megatron_cfg.env_vars` 에 `CUDNN_HOME`(Megatron 워커 venv 의 pip cuDNN 경로) 필수.
- 브리지·플러그인 수정 중에는 `megatron_cfg.force_reconvert_from_hf: true` (변환 캐시에 버전 검사 없음).
- FlashQLA 는 opt-in: `policy.megatron_cfg.env_vars` 에 `ALPHA_GDN_BACKEND: "flashqla"`. 실패 시 시끄럽게 중단, 무설정이면 fla.
- GPU 장수는 `cluster.gpus_per_node`·Ray 리소스로 제한한다. **`CUDA_VISIBLE_DEVICES` 금지** (원장 #19).
- 모델 입력은 Pai `evaluate.sh` 경로로 변환·검증된 HF 체크포인트(`hfmodel_*`)다. 새 ckpt 는 `GATES.md` 재실행 조건을 따른다.
- Muon(`dist_muon`): 필드명·기본값이 Pai SFT 와 다르다 — `muon_nesterov`(기본 False), `muon_extra_scale_factor`(기본 1.0 = SFT 의 5배)를
  명시한다. "필수 off" 4개를 끄지 않으면 셋업이 실패한다 (`grpo_alpha_smoke_muon.yaml` 헤더).

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
  git push 는 main1 에서 동작한다 (sub1 은 자격 없음).
- 노드 간 InfiniBand 가 없다 (TCP ~9.1 Gbit/s). 2노드는 학습/롤아웃 분리 + async GRPO (`RL_PLAN.md` §3).
- Gym 은 CPU 전용이지만 Ray 를 띄운다. GPU 드라이버가 깨진 노드에서는 Backend.AI libcudahook 때문에 `ray.init()` 이 즉사한다.
- HOME(`/home/work`)은 49 GB 루프 볼륨이다. 큰 캐시·다운로드는 `/opt` 나 NFS 로.
- **비밀키**(HF·Gemini·Tavily)는 Pai `examples/alpha/.env`(gitignored). 묻지 말고 `set -a; source <그 파일>; set +a`. 값은 출력·커밋 금지.
