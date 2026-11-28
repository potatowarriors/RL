# Gym 패치 — 고정한 Gym (v0.6.0 `3045a793`) 위에 얹는 upstream 수정

Gym 서브모듈은 v0.6.0 에 고정한다 (`.claude/rules/alpha-submodules.md`). 그 버전에 없고 우리 경로에 필요한 upstream 수정은
**upstream 커밋을 그대로** 여기에 패치 파일로 두고 두 체크아웃에 적용한다 (upstream 에 없는 우리 수정은 표에 "없음" 으로 적는다) (`RL_PLAN.md` 결정 79, 사용자 2026-11-25).

| 패치 | upstream 커밋 | 왜 필요한가 |
|---|---|---|
| `0001-…normalize-response-tools-for-replay-3190.patch` | `7bfccecc1` (2026-09-09, #3190) | `swe_agents` 가 롤아웃 결과를 조립할 때 모델 응답에서 되돌려 쓴 도구 정의에 `defer_loading: None` 이 들어가 `SWEBenchVerifyResponse` 검증이 실패한다 (`/run` 500, `KNOWN_ISSUES.md` 2026-11-25) |
| `0002-swe-agents-do-not-mount-the-launch-directory.patch` | 없음 (우리 수정, 2026-11-25) | `swe_agents` 의 컨테이너 옵션 `--no-mount home,tmp,bind-paths` 에 `cwd` 를 더한다. Apptainer 는 실행 위치를 컨테이너에 붙이고 Ray 워커는 raylet 의 현재 디렉토리를 물려받아, 과제 컨테이너가 호스트 홈 (sub1) · 리포 (main1) 를 쓰기 가능하게 봤다 — sub1 홈 볼륨이 과제 캐시로 가득 참 (`KNOWN_ISSUES.md` 2026-11-25). upstream `main` 에도 없는 옵션이라 fork 를 올려도 남긴다 |
| `0003-swe-agents-unfinished-trajectory-penalty.patch` | 없음 (우리 수정, 2026-11-27 — `RL_PLAN.md` 결정 95) | 설정 `unfinished_trajectory_reward` 를 더한다. 값이 있으면 턴 한도 · 문맥 한도로 끝나고 과제를 풀지 못한 롤아웃이 그 보상을 받고 마스킹되지 않는다 (`unfinished_penalized`). 기본 None 은 원래 동작 (채점한 보상 + 마스킹). 근거 · 영향 계산은 `docs/E2E_SANDBOX.md` §7.13 |
| `0004-swe-agents-per-run-writable-mounts-and-private-shm.patch` | 없음 (우리 수정, 2026-11-28 — upstream `origin/main` `363af5e3e` 도 같은 구조다) | 컨테이너 격리. Gym 은 하나뿐인 OpenHands 체크아웃의 `evaluation/oh` · `logs` · `.eval_sessions` 와 프롬프트 템플릿 파일을 모든 에이전트 · 채점 컨테이너에 읽기 · 쓰기로 붙이고, `apptainer.conf` 의 `mount dev = yes` 로 노드의 `/dev/shm` 도 그대로 보인다 — 한 컨테이너의 삭제 명령이 돌고 있던 롤아웃 전부의 결과 파일과 호스트의 공유 메모리 파일을 지웠다 (`s6smoke`, `docs/KNOWN_ISSUES.md` 2026-11-27). 패치: ① 쓰기 가능한 디렉토리를 롤아웃별로 (`evaluation/oh/<run id>` 만 쓰기 가능 · 이웃 런은 빈 읽기 전용 디렉토리로 가림 · `logs` · `.eval_sessions` 는 롤아웃 디렉토리 아래) ② 프롬프트 템플릿 읽기 전용 ③ SWE-rebench 채점 컨테이너의 `swebench_setup` 읽기 전용 (그 하니스는 거기에 쓰지 않는다) ④ 실행 인자에 `--contain` — `/dev` 가 최소 구성이 되고 `/dev/shm` 이 컨테이너 전용 tmpfs 가 된다 (다른 디렉토리를 `/dev/shm` 위에 붙이는 방법은 Apptainer 가 거부한다: "source and destination must be identical when binding to /dev"). `--no-mount home,tmp` 는 그대로라 홈 · `/tmp` 동작은 전과 같다. 달라지는 것: `/dev/fuse` · `/dev/full` · GPU 장치 노드가 컨테이너에 없다 · 전용 `/dev/shm` 의 상한은 tmpfs 기본값 (이 노드에서 약 1 TB — 전에는 노드 공용 128 GiB) ⑤ 설정 `empty_trajectory_reruns` (기본 0) — 모델 호출 기록 없이 돌아온 롤아웃을 그 수만큼 다시 돈다. 손대지 않은 것: `agent_framework: opencode` 경로의 공유 `evaluation/oh` (우리는 쓰지 않는다) · 호스트 `/proc` 과 네트워크 (`docs/E2E_SANDBOX.md`). 확인: 지우지 않는 탐침 `$NRL_ROOT/gates/swe_e2e_s6/run_stub_probe.sh` + `summarize_probe.py --expect isolated` |

- 적용 · 확인: `examples/configs/alpha/tools/apply_gym_patches.sh` (인자 없이 = 적용, `check` = 확인만). 대상은 NeMo-RL 서브모듈
  `3rdparty/Gym-workspace/Gym` (학습) 과 측정용 체크아웃 `/home/work/vidsearch/tools/Gym`. 두 곳 모두 NFS 라 세션을 다시 만들어도 남는다 — 새로 클론했을 때만 다시 적용한다.
- 적용하면 서브모듈이 superproject 의 `git status` 에 수정됨으로 보인다. 포인터는 그대로 `3045a793` 이다 — 서브모듈 안에서 커밋하지 않는다.
- 요청 검증 쪽 수정 (upstream #4011 · #4483: `aws_region_name` · 도구 결과 `name` · `provider_specific_fields`) 은 여기 두지 않는다.
  모델 서버는 우리 플러그인 (`gym_plugins/responses_api_models/alpha_vllm_model`) 이고 그 서버 venv 는 Gym 라이브러리를 공개 패키지로 받기 때문에
  Gym 소스를 고쳐도 닿지 않는다 → 같은 동작을 플러그인이 한다.
- Gym 을 upstream 쪽으로 올리지 않는 이유: 세 수정이 모두 든 Gym 은 자기 쪽 vLLM 을 0.29.0 으로 고정하고 (torchvision 0.28 · transformers ≥ 5.10.4 · Ray ≥ 2.58),
  우리 NeMo-RL 의 고정 (torch 2.11 · torchvision 0.26 · transformers < 5.9 · Ray 2.56.1) 과 한 잠금 파일에서 풀리지 않는다 (2026-11-25 사본에서 `uv lock` 실패 확인).
- fork 를 upstream 에 맞춰 올리면 이 디렉토리는 비운다.
