# NeMo-RL 환경 세팅 및 운영 가이드 (alpha post-training)

작성: 2026-08-13, 최종 갱신 2026-10-06. 2026-10-07 워크스페이스 `project_s/NEMO_RL_SETUP.md` 에서 이관했다.
트러블슈팅 원장(#1~#21)은 [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md), 검증 게이트는 [`GATES.md`](GATES.md), 현재 상태는 [`STATUS.md`](STATUS.md).

alpha_v2 post-training(RLVR/GRPO → teacher RL → MOPD)을 위한 NeMo-RL 실행 환경이다.
Nemotron-3-Ultra 레시피(NeMo-RL `ultra-v3` 브랜치)를 참조 원형으로 하며,
**학습 백엔드 = Megatron-Core, 롤아웃 = vLLM(alpha 플러그인)** 조합을 주 경로로 한다.

## 1. 기본 경로 — Pai 컨테이너 공존 설치, 시스템 무변경 (2026-10-06 검증, main1)

sudo·apt·`/usr/local/cuda`·ld.so.conf 를 건드리지 않는다. 같은 컨테이너의 Pai 학습·벤치 fleet 과 공존한다.
스크립트: `project_s/setup_nemo_rl_env_noninvasive.sh` (멱등, 전부 `$NRL_ROOT` = `/home/work/vidsearch/tools/nemo_rl` 에 설치).
실행은 항상 `$NRL_ROOT/clean_run.sh <cmd...> </dev/null` (cwd = NeMo-RL).
레시피·프로브·로그 원본: `project_s/reboot_restore/nemo_rl_noninvasive_20261006/` (`env.sh`·`clean_run.sh`·`ldblock/`). 그 `env.sh` 의 경로는 검증 당시 scratch 경로이므로 이식 시 바꾼다.

**사용자 결정 2026-10-06**: 원본(클론·툴킷)은 NFS, **venv 는 로컬**. NFS venv 는 GRPO setup 371 s, 로컬은 156 s 였다.
셋업 재구성 제안(로컬 venv 스냅샷·인터프리터 로컬·바이트코드 사전 컴파일·JIT 캐시 로컬)은 승인 대기다 — §1.1, `STATUS.md`.

| 구성 | 값 |
|---|---|
| CUDA 13 툴킷 | runfile `cuda_13.0.2_580.95.05_linux.run --silent --toolkit --toolkitpath=<사설경로>` (비root, 다운로드 45 s + 설치 120 s) |
| uv | 0.11.28 을 사설 `bin/` 에 (`UV_INSTALL_DIR`), 캐시·python·venv 전부 사설 경로 (`UV_CACHE_DIR`·`UV_PYTHON_INSTALL_DIR`·`NEMO_RL_VENV_DIR`) |
| 빌드 env | `CUDA_HOME`·`CUDACXX`(#15)·`LIBRARY_PATH`·`CPATH=$CUDA_HOME/include/cccl`·`TORCH_CUDA_ARCH_LIST=9.0a`·`NVTE_CUDA_ARCHS=90`·`MAX_JOBS=10` |
| 런타임 env | `LD_LIBRARY_PATH=<ldblock>:$CUDNN_HOME/lib:/usr/local/cuda/compat/lib.real`(#16)·`CUDNN_HOME`=Megatron 워커 venv 의 pip cuDNN·`CUDA_LIB_PATH`(#17)·`unset PYTHONPATH`·`PYTHONNOUSERSITE=1`(#18) |
| 실행 | `clean_run.sh <cmd> </dev/null` (#18). `CUDA_VISIBLE_DEVICES` 금지(#19). whitelist 밖 환경변수는 `clean_run.sh /usr/bin/env VAR=값 <cmd>` 로 넘긴다(#23) |

`#N` 은 [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) 원장 번호다. 설치 검증 결과(venv 빌드·TE fused attn·GRPO 퀵스타트)는 [`GATES.md`](GATES.md) "환경" 절.

### 1.1 셋업 재구성 제안 — 공식 컨테이너의 속도를 컨테이너 없이 재현 (2026-10-06 제안, 승인 대기)

공식 컨테이너가 빠른 이유는 워커 venv 가 이미지 안 로컬 디스크에 미리 빌드돼 있어서다 (`docker/Dockerfile:122,391`).
이미지를 바꿀 수 없으므로 **로컬 venv 스냅샷(tar)을 NFS 에 두고 세션마다 로컬로 복원**한다.
NFS 순차 읽기는 빠르다(835 MB/s). 복원 추정 1~2 분(하한 46 s, 미측정), 전체 재빌드 폴백은 약 11 분이다.
v0.7 의 기동 최적화 9건(#2739·#2920·#3089·#3158·#2741·#2750·#2771·#2793·#2767)은 이미 우리 브랜치에 있다.

| 위치 | 내용 |
|---|---|
| NFS `$NRL_ROOT` | uv · CUDA 13 툴킷 · uv 캐시(폴백용) · mcore 변환 캐시 · HF_HOME · **`snapshots/<lock 지문>.tar.zst`** (신규) |
| 로컬 `/opt/nrl/` 또는 `/tmp/nrl/` (두 노드 같은 경로) | Python 인터프리터 · driver venv · 워커 venv · Gym venv · JIT 캐시 · ldblock — **전부 NFS 에서 이동** |

| # | 변경 | 이유 |
|---|---|---|
| 1 | Python 인터프리터도 로컬로 | venv 의 `bin/python` 이 NFS 의 `uv-python` 을 가리켜 표준 라이브러리를 매번 NFS 에서 읽는다 |
| 2 | 스냅샷 빌드 때 바이트코드 사전 컴파일 | 실측 `import torch,transformers,vllm` 11.6 s (pyc 없음) → 약 5 s |
| 3 | 런타임 패치 사전 적용 | NeMo-RL 이 첫 실행에 TE·vLLM 파일을 제자리 수정한다. 워커 8개가 동시에 하면 경합이 생긴다 |
| 4 | JIT 캐시(vLLM·Triton·FlashInfer·CUDA PTX)를 로컬로 | 기본이 HOME(49 GB)이다. `~/.cache/vllm_0..7`·`~/.triton` 이 이미 쌓이는 중이다 |
| 5 | 별도 `venv-mcore`·`venv-vllm` 제거 | 31 GB 중복이다. 게이트 도구는 워커 venv 로 직접 실행한다 |
| 6 | driver 를 `venv-driver/bin/python` 으로 실행 | `uv run` 의 매회 lock 검사·sync 를 건너뛴다. 단 `UV_NO_SYNC` 를 전역으로 두면 워커 venv 생성이 깨진다 |
| 7 | Gym venv 로컬 + `skip_venv_if_present: true` | Gym 은 기본으로 매 실행 재설치한다 |

운영 모델: lock 이 바뀔 때만 로컬 빌드 → 사전 컴파일·패치 → TE·cudart·fused attn 게이트 → 스냅샷 저장(약 15 분).
세션 재생성 때는 지문 비교 후 복원한다. 매 실행은 `clean_run venv-driver/bin/python run_grpo.py ...` 이고 setup 은 156 s 다 (로컬 venv 실측, NFS venv 는 371 s).

2노드 준비(검증 필요): 두 노드에 같은 절대경로 · `ray start --head`/`--address` 수동 기동(`clean_run` 경유, 포트·ulimit 은 공식 `ray.sub` 따름) ·
`NCCL_IB_DISABLE=1`·`NCCL_SOCKET_IFNAME=eth0`·`GLOO_SOCKET_IFNAME=eth0` · Ray object store 크기 명시(`/dev/shm` 을 Pai 와 공유).
미검증 위험: NeMo-RL 은 driver 의 환경 전체를 모든 노드의 actor 에 넘긴다. main1 고유 값(`NVIDIA_VISIBLE_DEVICES` GPU UUID·`HOSTNAME`·`BACKENDAI_CLUSTER_*`)이 sub1 워커에도 간다.
Backend.AI GPU 훅이 이를 읽는지 모르므로 driver 를 그 값 없이 띄워 2노드 스모크로 확인한다.

결정 필요: 로컬 경로 `/opt/nrl`(세션마다 `sudo mkdir`·`chown` 1회, Pai 무영향) 또는 `/tmp/nrl`(sudo 불필요).
승인 뒤 순서: 셋업 스크립트를 `--build-snapshot`(lock 변경 시)·기본 복원 두 모드로 나눈다 → main1 에서 복원 시간·import·TE fused 프로브·1.5B 스모크·alpha Muon KL 게이트·Gym GPU 테스트 2건 → 2노드 Ray 스모크.

## 2. 환경 구조 (중요 개념)

- **uv 기반 완전 격리.** 시스템 Python(Pai 환경)을 건드리지 않는다. 모든 실행은
  `NeMo-RL/` 디렉토리에서 `uv run --locked ...` 형태.
- **역할별 워커 venv.** `mcore`와 `vllm` extra는 pyproject에 **충돌로 선언**되어 한 venv에
  공존하지 않는다. Ray 워커가 `NeMo-RL/venvs/<worker>/` 아래 역할별 venv를 자동 생성한다.
  수동 prewarm은 `uv sync --locked --extra mcore` 와 `--extra vllm`을 **각각** 실행.
- **pyproject를 수정했으면** 다음 실행 시 `NRL_FORCE_REBUILD_VENVS=true` 필요.

## 3. 프로젝트 구조·실행

```
Pai-Megatron-Patch/examples/alpha/   # pre-train + LC단계 + SFT + 벤치 (기존 스택 유지)
        │  evaluate.sh (MG→HF 변환·검증)
        ▼  hfmodel_* (HF 체크포인트)  ←── 두 스택의 유일한 인터페이스
NeMo-RL/                             # RL 전 과정 (이 환경)
├── examples/configs/alpha/          # ★ 레시피 허브: 스테이지별 yaml + 도구 + 문서
├── 3rdparty/Megatron-Bridge-.../    # ★ alpha bridge 작업 위치 (HF↔mcore 매핑, fork alpha/bridge)
└── 3rdparty/vllm/                   # (untracked) 커스텀 vLLM 빌드 체크아웃 — 통합은 플러그인
```

- 일상 실행: `cd NeMo-RL && $NRL_ROOT/clean_run.sh $NRL_ROOT/bin/uv run --locked python examples/run_grpo.py --config examples/configs/alpha/<stage>.yaml </dev/null`
- 2노드: node0에서 `uv run ray start --head --port=6379`, node1에서 `uv run ray start --address=<node0>:6379` 후 node0에서 동일 명령. 토폴로지 원칙은 [`RL_PLAN.md`](RL_PLAN.md) §2. 2노드 Ray 스모크는 미검증이다.
- 커스텀 코드는 "수정을 소유한 레이어에": NeMo-RL 본체는 최대한 무수정 (rebase 대비).

## 4. RL 전용 세션 설치 (2026-08-13, 구 경로)

Pai 작업이 없는 RL 전용 H100 세션에서만 쓴다. Pai 작업이 도는 컨테이너에서는 §1 을 쓴다.

```bash
cd /home/work/vidsearch/repos/project_s
./setup_nemo_rl_env.sh              # 멱등(idempotent), 재실행 안전
# 이후 매 셸에서:
source nemo_rl_env.sh
```

- **NeMo-RL 전용 세션에서만 실행할 것.** 스크립트가 `/usr/local/cuda` 링크를 13.0으로
  전환하고 ld.so.conf 일부를 비활성화하므로 Pai-Megatron 학습 세션에서 실행하면 안 된다.
- **NGC 25.08(CUDA 13.0) 베이스 이미지에서는** 드라이버/툴킷 우회 단계가 기능 프로브로
  자동 스킵된다. 스크립트 수정 없이 그대로 실행하면 된다.
- 일상 실행: `source nemo_rl_env.sh && cd NeMo-RL && uv run --locked python examples/run_grpo.py --config examples/configs/alpha/<stage>.yaml`
- 검증 환경(2026-08-13): Backend.AI H100 세션, 드라이버 535.261.03, 베이스 이미지 CUDA 12.8.

## 5. 세션 재생성 후 복원

§1 경로는 전부 NFS(`$NRL_ROOT`)라 재설치가 필요 없다. 로컬 venv 를 쓰는 구성이면 venv 만 다시 만든다.
스크립트를 다시 돌려도 멱등이다. 노드·GPU 쪽 공통 복원은 `project_s/RESTORE_AFTER_REBOOT.md` 가 정본이다.

§4 경로(RL 전용 세션)가 재생성된 경우:

- 유지: `NeMo-RL/` 클론·서브모듈·`venvs/`·설치 스크립트 (모두 NFS)
- 소멸: `/opt/uv-cache`(45G), uv 바이너리, CUDA 13 툴킷·compat, ld.so.conf 조정, apt 패키지

```bash
cd /home/work/vidsearch/repos/project_s
./setup_nemo_rl_env.sh        # 멱등 — 소멸분만 재설치 (TE 재빌드 ~30분 포함, uv는 0.11.28 핀 설치)
source nemo_rl_env.sh
# 검증: 두 백엔드 GRPO 스모크 (각 ~15분)
cd NeMo-RL
uv run --locked python examples/run_grpo.py --config examples/configs/grpo_math_1B.yaml grpo.max_num_steps=2 checkpointing.enabled=false
uv run --locked python examples/run_grpo.py --config examples/configs/grpo_math_1B_megatron.yaml grpo.max_num_steps=2 checkpointing.enabled=false
```

GitHub 인증 복원은 `project_s/RESTORE_AFTER_REBOOT.md` §6.1. remote 구성은 다음과 같다 (NFS 클론에 이미 설정돼 있음):
- `NeMo-RL/`: origin=potatowarriors/RL, upstream=NVIDIA-NeMo/RL
- `Megatron-Bridge` 서브모듈: origin=NVIDIA-NeMo(기본), fork=potatowarriors/Megatron-Bridge
- vllm fork: potatowarriors/vllm (`alpha/model` 브랜치, in-tree 구현 기록용)
