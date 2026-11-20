# SWE · Terminal E2E 샌드박스 (Apptainer) — 인프라 요청 · 검증 · 착수 준비

2026-11-20 작성. SWE · Terminal 의 E2E 학습은 과제마다 컨테이너를 띄워야 하는데 학습 세션에서 Apptainer 가 돌지 않았다.
이 문서는 그 원인, 인프라에 보낸 요청, 요청의 근거, 적용된 뒤 학습을 시작하기까지의 절차와 준비 상태를 한 곳에 둔다.
상태 한 줄은 [`STATUS.md`](STATUS.md), 사고 서사는 [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) 2026-11-20 항목이 정본이다.

경로 줄임: `$NRL_ROOT` = `/home/work/vidsearch/tools/nemo_rl`, `tools/apptainer/` = `examples/configs/alpha/tools/apptainer/`,
`RL/` = `/home/work/Datasets/LL_datasets/posttraining/RL/`, 리허설 산출물 = `$NRL_ROOT/apptainer_test/rehearsal_20261120/`.

## 1. 지금 상태

| 단계 | 상태 |
|---|---|
| 원인 파악 | 완료 — 세션 컨테이너의 seccomp · AppArmor · `/dev/fuse` 세 층 (§2) |
| 인프라 요청 | **전달됨 (사용자 2026-11-20), 적용 대기** (§3) |
| 요청 사양 검증 | 별도 Docker 호스트 리허설 2회 + 독립 검증 2회, 재판정 = 확인 (§4) |
| 시험 세션 수용 시험 | 대기 — `tools/apptainer/acceptance.sh` (지금의 막힌 세션에서 5 단계 FAIL 을 보고함) |
| 우리 쪽 Apptainer 설정 | 후보 측정 완료, 선택은 승인 대기 (§5) |
| 태스크 이미지 → `.sif` | **Super `swe2` 1,444 개 변환 완료 (2026-11-21)** — 1.36 TiB, 내용 확인 1,444/1,444. 실행 검증은 시험 세션 몫. Ultra `swe` 는 결정 대기 (§7.2 · §9) |
| SWE E2E 레시피 | 제공 경로 · 프론티어 조사 완료, 설정 결정 대기 (§7) |
| Terminal E2E | 제공 경로 조사 완료 — 환경은 `harbor_agent` 하나, 레시피 · 이미지 · 실패 마스킹은 만들어야 함 (§8) |

## 2. 왜 막혔나

세션은 Backend.AI 가 만든 Docker 컨테이너다. Apptainer (setuid 없는 방식) 는 user namespace 를 만들고, 그 안에서 마운트하고, FUSE 로 이미지를 읽는다. 세 단계가 각각 막혀 있다.

| 층 | 세션 실측 (main1, 2026-11-20 — seccomp · AppArmor · capability · `/dev/fuse` 는 sub1 동일) | 결과 |
|---|---|---|
| seccomp | `unshare(0)` · `setns` EPERM, `clone3` ENOSYS (root 도 같음). 에이전트가 넘기는 프로파일은 Docker 기본과 syscall 426 개 · 조건이 같다 | user namespace 생성 불가 |
| AppArmor | `docker-default (enforce)` — 템플릿에 `deny mount`, `pivot_root` 규칙 없음 | seccomp 만 풀면 mount 에서 막힘 (집행은 미측정) |
| 장치 | `/dev/fuse` 없음. root 로 `mknod` 뒤 open 이 EPERM (device cgroup) | SIF 마운트 불가 |
| `/proc` | Docker 마스킹 10곳 + NVIDIA 런타임의 `/proc/driver/nvidia` tmpfs | `--pid` 가 요구하는 새 procfs 마운트를 커널이 거부 |

세션 안에서는 root 로도 셋 다 바꿀 수 없다 (CapBnd `a88465fb` 에 CAP_SYS_ADMIN 없음, seccomp 필터는 조일 수만 있음).
호스트 sysctl 은 이미 충족한다 (`max_user_namespaces` 8,253,991 · `unprivileged_userns_clone` 1 · `apparmor_restrict_unprivileged_userns` 0, 커널 6.5.0-15).

## 3. 인프라 요청 (전달 2026-11-20)

| # | 요청 | 내용 | 줄일 수 있는 부분 |
|---|---|---|---|
| 1 | seccomp | 에이전트가 컨테이너에 넘기는 프로파일에 규칙 둘 추가 (`tools/apptainer/seccomp_rules.json`): `setns` `mount` `umount2` `pivot_root` 허용, `unshare` · `clone` 은 `CLONE_NEWNET` (0x40000000) 이 없을 때만 허용. 기존 규칙 (`clone3` ENOSYS 포함) 은 그대로 | `pivot_root` 는 빼도 통과 (대체 경로). `clone` 은 `--fakeroot` 에만 필요 — SWE 만이면 `unshare` `setns` `mount` `umount2` 가 최소 |
| 2 | AppArmor | 프로파일 `docker-apptainer` (`tools/apptainer/docker-apptainer`, parser 4.x + userns 중재면 `.userns` 판) 를 호스트에 적재하고 세션에 `apparmor=docker-apptainer`. Docker 템플릿에서 바꾼 것: 이름 · `deny mount,` 삭제 · `mount,` `pivot_root,` 추가 · dockerd 라벨 줄 삭제 | `pivot_root,` 규칙 |
| 3 | 장치 | `/dev/fuse` 를 cgroup 권한 `rwm` 과 함께 | 없음 |

요청하지 않은 것: `--privileged` · capability 추가 · docker.sock · `/proc` 마스킹 해제 · cgroup 위임 · seccomp/AppArmor `unconfined` · `sandbox-type=jail`.

**적용 지점** (Backend.AI 공개 코드 main 기준, 배포 버전 미확인 · 적용 미시험 — 안내문 `tools/apptainer/infra_worksheet.md`):
- seccomp 는 `<site-packages>/ai/backend/runner/default-seccomp.json` 이다. 에이전트가 컨테이너를 만들 때마다 읽는다. 노드의 모든 세션에 적용되고 에이전트 업그레이드 때 덮어써진다. dockerd `daemon.json` 을 바꾸면 효과가 없다. 플러그인의 "추가 허용 syscall" 은 이름만 받아 인자 조건을 못 담는다.
- AppArmor 와 장치를 넣는 설정 항목은 `agent.toml` 에 없다. compute 플러그인의 `generate_docker_args()` (그 슬롯을 받은 세션만) 또는 `agent/docker/agent.py` 수정 (`/dev/infiniband` 블록이 선례, 노드 전체) 이다.

**새 network namespace 를 막은 이유**: 호스트 커널 6.5.0-15 는 CVE-2024-1086 (nf_tables) 수정 (6.5.0-26) 이전이고, 이 취약점은 user namespace + 새 network namespace 로 닿는다. 제공 경로 (`swe_agents`) 는 `--net` 을 쓰지 않는다.
대가: Apptainer 의 `--net --network none` 격리를 쓸 수 없다. 나중에 필요하면 커널 갱신과 함께 다시 요청해야 한다.

## 4. 요청의 근거 — 리허설과 독립 검증

장소: gpu06 `alpha-eval` (privileged DinD) 안쪽 dockerd 29.1.3 — 커널 5.4 · cgroup v1 · AppArmor 미적용.
세션을 흉내 낸 컨테이너: `ubuntu:24.04` + `docker/install_apptainer.sh` (Apptainer 1.4.5) · uid 1100 · CapBnd `a88465fb` · Backend.AI 기본 seccomp.
시험 스크립트 · 프로파일 · 로그 · 검증 자료는 리허설 산출물 디렉토리에 있다 (`scripts/t.sh` · `round2/scripts/t2.sh` · `round2/run_{A..F}.log` · `review/01~06`).

| 시험 | 결과 |
|---|---|
| 음성 대조: Backend.AI 기본 프로파일 그대로 | FAIL `Failed to create user namespace` — 세션과 같은 증상 |
| 요청 규칙 + `/dev/fuse` | `unshare -Ur` · tmpfs mount · `pivot_root` · `apptainer exec` · `--writable-tmpfs` · `apptainer build` PASS |
| 새 network namespace | `unshare -Urn` · `clone(NEWUSER\|NEWNET)` 둘 다 EPERM. 이름만 추가한 대조 프로파일에서는 생성됨 |
| 한 개씩 뺀 절제 | `setns` · `umount2` 를 빼면 FAIL. `pivot_root` · `clone3` · 신형 mount API 7종은 빼도 PASS |
| `/dev/fuse` 없음 | FAIL `fuse: device not found` — 대체 경로 없음 |
| 실제 이미지 `sweb.eval.x86_64.django_1776_django-11099` | `swe_agents` 와 같은 플래그로 `/etc/hosts` 쓰기 · 저장소 수정 · 결과 새로 생성 · 테스트 22 개 `tests_rc=0` (uid 1100 과 컨테이너 안 root 양쪽) |
| `--fakeroot --containall --pid` (Terminal 형태) | PASS (root-mapped, `APPTAINER_IGNORE_FAKEROOT_COMMAND=1`) |

**독립 검증** (Opus 5.5 · xhigh, 블라인드 2단계, 같은 검증자가 2회):

| 회 | 판정 | 반론 요지 | 조치 |
|---|---|---|---|
| 1 | 증거 부족 | 요청이 dockerd `daemon.json` 에 적용되면 무효 (적용 계층) · AppArmor 미측정이고 `deny mount` 는 지워야 함 · 커널이 CVE-2024-1086 미수정 · `allow pid ns = no` 는 샌드박스가 세션 프로세스에 신호를 보낼 수 있음 · 하네스의 PASS 조건과 결과 파일 재사용 | 2차 리허설 |
| 2 | **확인** (필수 보완 반영 문안 기준) | seccomp-bpf 방식 syscall 조사는 다른 필터가 막은 호출을 못 봄 → 그 결론 철회 · 바깥 `/proc` 로 세션 프로세스의 `oom_score_adj` 에 쓸 수 있음 (실측 확인) · `CLONE_NEWNET` 차단은 `--net` 격리도 막음 · AppArmor 영속화와 적용 경로를 문안에 | 문안 보완, §5 에 위험 기록 |

**시험 세션에서만 닫히는 것**: AppArmor 집행 · NVIDIA 의 `/proc` 덮임 · 커널 6.5 · cgroup v2 · Backend.AI 의 실제 적용 경로. 리허설 장소가 커널 5.4 이고 AppArmor 가 적용되지 않는 곳이라 우리 쪽에서 더 줄일 수 없다.
`acceptance.sh` 가 단계마다 원인을 하나로 가리킨다 (0 적용된 설정 → 1 seccomp → 2 user namespace → 3 mount → 4 pivot_root → 5 프로세스 간 신호 → 6 `/dev/fuse` → 7 이후 Apptainer).

## 5. 우리 쪽 Apptainer 설정 (인프라와 무관, 선택은 승인 대상)

**실행 신원**: uid 1100 그대로 띄운다. `sudo` 로 real root 실행은 실패한다 (작업 디렉토리 · 결과 디렉토리가 매핑되지 않은 uid 소유가 된다). 컨테이너 안에서는 이미지 파일이 실행 사용자 소유로 보여 `/etc/hosts` · `/root` · 저장소에 쓸 수 있다.

**`--pid`**: 제공 명령은 `--pid` 를 쓰는데 새 procfs 마운트가 거부된다. 설계대로의 동작 (PID namespace + 그 namespace 의 procfs) 은 `/proc` 마스킹을 풀지 않고는 커널 6.5 에서 얻을 수 없다. 선택지 실측 (2차 리허설):

| 측정 | `allow pid ns = no` | namespace 유지 + `mount proc = no` | namespace 유지 + `mount proc = no` + `APPTAINER_BIND=/proc` | 별도 uid 로 실행 |
|---|---|---|---|---|
| 샌드박스 안의 `kill` 이 세션 프로세스에 닿는가 | **죽음** | 안 닿음 | 안 닿음 | 안 닿음 |
| 타임아웃 (트리 kill) 뒤 고아 데몬 | **남음** | 정리됨 | 정리됨 | 학습 쪽 kill 이 **실패** |
| `<(…)` · `/dev/stdout` | 됨 | **안 됨** | 됨 | — |
| 자기 배경 프로세스를 `pkill` · `pgrep` · `ps -p` 로 다루기 | 됨 | 안 됨 | **안 됨** (`/proc` 의 pid 가 바깥 번호) | — |
| `kill $!` · django `--parallel 2` · multiprocessing | 됨 | 됨 | 됨 | — |

후보는 셋째 열이다 (`setup_session.sh --pid-mode bindproc`). 첫째 열은 모델이 `pkill python` 한 번으로 학습을 죽일 수 있어 기각한다.
남는 위험 (PID 방식과 무관하게 바깥 `/proc` 가 보이는 한): 샌드박스 코드가 같은 uid 인 세션 프로세스의 `oom_score_adj` 를 올릴 수 있다 (0 → 1000 실측). environ · cwd · mem 은 거부된다.
미시험: `APPTAINER_BIND` 가 Gym 의 subprocess 환경을 거쳐 Apptainer 까지 전달되는지 (스모크에서 확인), 동시에 도는 샌드박스 사이.

**fakeroot** (Terminal 경로): 호스트의 `fakeroot` 명령이 이미지 glibc 와 안 맞아 실패한다 → `APPTAINER_IGNORE_FAKEROOT_COMMAND=1` (root-mapped 만).
다중 uid 매핑은 Ubuntu 24.04 의 `newuidmap` 이 권한을 낮추지 않고 map 을 열어 CAP_SYS_ADMIN 을 요구한다 → setuid 대신 file capability 로 바꾸면 된다 (`setup_session.sh --multi-uid`, 실측 PASS).
fakeroot instance 는 `instance start --fakeroot` 뒤 `exec` 를 `--fakeroot` 없이 부른다 (실측 PASS, 붙이면 실패).

**그 밖** (미측정): 샌드박스가 세션 network namespace 를 공유한다 (Ray · vLLM 포트에 닿음) · 기본 `mount dev = yes` 가 세션 `/dev` 를 통째로 넘긴다 · `--writable-tmpfs` 크기는 `sessiondir max size` 기본 64 MB 다.
시험 도구 주의: FUSE 를 쓰는 실행을 `strace -f` 로 통째로 추적하지 않는다 (교착, `KNOWN_ISSUES.md` 2026-11-20).

## 6. 적용된 뒤의 순서

| # | 단계 | 잡는 결함 | 소요 | 승인 |
|---|---|---|---|---|
| 1 | 시험 세션에서 `tools/apptainer/acceptance.sh` | 세 설정의 누락 · 과다 부여 · AppArmor 문구 오류 | 분 | — (GPU 미사용) |
| 2 | 시험 세션에 `setup_session.sh` → 실제 이미지 시험 (`$NRL_ROOT/apptainer_test/sif_samples/`) | 커널 6.5 · NVIDIA 덮임 아래의 동작, PID 방식 | 분 | PID 방식 |
| 3 | 본 세션 재생성 · 복원 (`SETUP.md` §5) → 두 노드에 `setup_session.sh` | — | 세션 복원 시간 | 세션 재생성 시점 |
| 4 | CPU: 레시피 검사 `tools/check_alpha_recipe.py` ERROR 0 · 블렌드의 모든 행에 `.sif` 가 있는지 (`build_sifs.py --dry-run`) | 설정 · 데이터 누락 | 분 | — |
| 5 | 무작업 에이전트로 소수 과제 0 점 확인 · 정답 패치 채점 (지원 데이터셋만) | 채점기가 항상 통과 · 항상 실패하는 결함 | 분~수십 분 | GPU 없이 가능한 범위만 |
| 6 | 스모크 3 step (작은 배치, W&B 끔) | 다중 턴 logprob 정합 (GATES G3 형태) · 빈 롤아웃 · `APPTAINER_BIND` 전달 · 잔류 프로세스 · 롤아웃당 CPU · RAM · 시간 | 수십 분 | GPU |
| 7 | 본 런의 첫 step 들을 게이트로 판정하고 통과하면 그대로 학습 | 처리량 · 정체 | 런 길이 | GPU |

## 7. SWE E2E 준비 상태

시작점은 MOPD1 학생 `mopd1/hf/hf_step_00120` (결정 59), SWE-V 12.0% (60/500, 빈 패치 43.6%) 이다. SWE 피벗 teacher 는 8.4 로 내려가 기각됐다 (`study/benchmark_results.md` §0.6).

### 7.1 제공되는 경로 (NeMo-RL 공개 레시피 — 전부 Gym `swe_agents` + Apptainer `.sif`)

| 레시피 | 모델 | 배치 (프롬프트 × 생성) | lr | 문맥 | 턴 · 에이전트 타임아웃 | 메모리 상한 | 데이터 | 한 에폭 |
|---|---|---|---|---|---|---|---|---|
| `nemotron-3-ultra/swe_teacher.yaml` | Ultra 550B | 32 × 16 | 3e-6 상수 | 196,608 | 200 · 3,600 s | 64 GB | Ultra `swe.jsonl` 7,816 (SWE-rebench-V2 7,610 + SWE-Gym 206), 4 에폭 | 244 step |
| `nemotron-3-super/stage2_swe2.yaml` | Super | 16 × 32 | 1e-6 | 196,608 | 200 · 3,600 s | — | Super `swe2.jsonl` 1,444 (R2E-Gym 1,172 + SWE-Gym 272) | 90 step |
| `grpo_qwen3_30ba3b_thinking_swe2.yaml` | Qwen3-30B-A3B | 8 × 8 | 1e-6 | 131,072 | 200 · 1,800 s | — | `swe2.jsonl` | 180 step |

공통: async GRPO · 궤적 나이 1 · `force_on_policy_ratio` · advantage clip ±100 · 잘못된 도구 호출 advantage −5 · `run_with_mixed_prompts` (Codex + OpenCode + CodeAct 프롬프트) · 동시 실행 768~1,024.
Gym 기본 학습 설정 (`swebench_openhands_training.yaml`) 은 OpenHands CodeAct · 100 턴 · 1,800 s · 테스트 1,200 s · 32 GB 다.
Gym v0.6.0 은 두 블렌드의 데이터셋을 모두 처리한다 (`SWERebenchDatasetProcessor` · `R2EGymDatasetProcessor` · 기본 처리기 = SWE-Gym).

Qwen3-30B-A3B 가이드의 공개 결과 (SWE-V pass@1, 500): 원본 23.6 → 피벗 30.4 → E2E 26.8 ~ 31.2 (최고 step 118). E2E 단계의 이득은 측정마다 오르내린다.

### 7.2 데이터와 이미지 (로컬 실측)

| 블렌드 | 위치 | 행 | 이미지 | 언어 | 이미지 크기 (압축, 표본) | 합계 추정 |
|---|---|---|---|---|---|---|
| Super `swe2.jsonl` | `RL/Nemotron-RL-Super-Training-Blends/` | 1,444 | `namanjain12/…` 1,172 · `xingyaoww/sweb.eval.x86_64.…` 272 | Python | R2E-Gym 중앙 0.63 GiB (n=24) · SWE-Gym 약 1.1 GiB | 약 1.1 TiB |
| Ultra `swe.jsonl` | `RL/Nemotron-RL-Ultra-Training-Blends/` | 7,816 | `swerebenchv2/…` 7,610 · SWE-Gym 206 | Python 1,496 · TS 1,407 · JS 1,338 · Go 1,049 · Rust 781 · PHP 643 · Julia 366 · Java 291 외 | SWE-rebench-V2 중앙 0.81 GiB (n=48) | 약 6.8 TiB |

- 표본 85 개가 전부 amd64 다. Ultra 문서의 ARM 재빌드는 우리에게 필요 없다.
- **변환은 지금 세션에서도 된다** — `apptainer build` 는 namespace 를 쓰지 않는다. `$NRL_ROOT/apptainer_test/` 의 1.5.5 로 실측: R2E-Gym 42 s · 0.56 GB, SWE-rebench-V2 78 s · 0.43 GB, SWE-Gym 115 s · 1.1 GB (`sif_samples/`).
- 도구 `tools/apptainer/build_sifs.py`: 블렌드가 쓰는 행만 받고, Gym 이 찾는 이름으로 저장하고, 이어받고, 풀 속도를 제한한다. 이름 검산은 두 블렌드 9,234 개 (SWE-Gym 26 개 중복 제외) 에서 어긋남 0.
- 풀 한도: 이 노드의 익명 한도는 시간당 100 이다 (레지스트리 응답 헤더 실측). 그 한도면 Super 약 18 시간 · Ultra 약 98 시간 (시간당 80 기준). 3 건을 받는 동안 잔여 수가 줄지 않아 실제로 걸리는지는 미확정이다.
- **파일럿 (2026-11-20 17:05, `swe2` · 노드마다 20 개 · 워커 4)**: 40/40 변환 · 실패 0 · `--verify` 재확인 40/40. main1 626 s · sub1 482 s, 임시 공간 최대 22 · 25 GiB. R2E-Gym (n=37) 중앙 59 s · 평균 0.65 GiB, SWE-Gym (n=3) 중앙 188 s · 평균 3.28 GiB (최대 5.46) → `swe2` 전체 추정 약 1.6 TiB (위 표의 1.1 TiB 는 압축 크기 표본에서 낸 값, SWE-Gym 표본이 3 개라 불확실).
- 풀 한도는 출구 IP 마다 따로다 — main1 · sub1 의 `docker-ratelimit-source` 가 다르고 둘 다 100/시간. 그래서 목록을 샤드로 갈라 두 노드에서 받는다 (`--shard K/N`, 파일 이름의 CRC). 파일럿 40 개를 받는 동안 잔여 수는 두 노드 모두 100 → 100 이었다 (원인 미확인 — 한도가 실제로 걸리는지는 여전히 미확정, 걸리면 도구가 429 를 보고 10분 쉬고 재시도한다).
- 변환한 파일은 내용 확인을 통과해야 제 이름을 얻는다: squashfs 목록이 끝까지 읽히고 Gym 이 쓰는 저장소 경로 (`/testbed`, SWE-rebench-V2 는 `/{repo 이름}`) 가 있어야 한다. 자가 시험의 음성 대조: 없는 경로 · 절반으로 자른 파일 · SIF 아닌 파일을 모두 거부. 데이터 블록 손상과 실행 가능 여부는 이 확인으로 잡지 못한다 (시험 세션의 표본 실행 · 정답 패치 채점).
- 저장 위치 `$NRL_ROOT/sif/swe/` (두 노드 같은 경로, NFS 여유 36 TiB). 레시피의 `container_formatter` 는 `…/sif/swe/r2egym_{instance_id}.sif` · `…/sif/swe/swegym_sweb.eval.x86_64.{instance_id}.sif` · `…/sif/swe/swerebench/{instance_id}.sif` 꼴이 된다 (Gym `_find_container` 가 이름을 바꿔 가며 찾는다).
- **본 변환 결과 (`swe2`, 2026-11-20 17:28 → 11-21 02:44)**: 1,444/1,444 변환 · 실패 0 · 429 없음 · 합계 1.36 TiB. R2E-Gym 1,172 개 — 중앙 0.55 GiB · 최대 1.74 · 합 741 GiB, SWE-Gym 272 개 — 중앙 1.45 GiB · 평균 2.41 · 최대 7.54 · 합 656 GiB. 노드마다 워커 4 · 시간당 80 으로 main1 742 개 9 h 16 m · sub1 662 개 8 h 9 m — 처리량이 상한에 묶였다. 풀 잔여 수는 끝까지 100 이었다. 완료 검증: `--dry-run` 받을 것 0 · `--verify` 1,444/1,444 (88 s, 워커 16) · 임시 파일 0. 압축 크기 표본으로 낸 추정 (1.1 TiB) 보다 실제가 1.24 배 컸다.
- Ultra `swe` 로 넓히면 새로 받을 이미지는 7,790 개다 (두 블렌드 고유 9,234 − 받은 1,444). 같은 설정 (노드당 80/시간) 이면 약 49 시간, 크기는 표의 6.8 TiB 에 같은 배율을 곱하면 약 8.4 TiB (SWE-rebench-V2 의 `.sif` 실측은 1 건뿐이라 불확실). 상한을 풀었을 때의 속도는 파일럿의 워커 4 에서 노드당 시간당 115 ~ 150 개였다 (20 개 482 ~ 626 s, 짧은 표본).
- NFS 여유 17 TB (90% 사용). 변환 임시 공간은 로컬 디스크 (여유 435 GB) 를 쓴다.
- 변환은 실행 쪽과 같은 1.4.5 로 한다 — `$NRL_ROOT/apptainer_test/root145/` (공식 deb 를 sha256 확인 뒤 `dpkg-deb -x`, 시스템 무변경). 도구로 1건 실측 (0.56 GB), 1.5.5 로 만든 파일도 1.4.5 가 구조를 읽는다. 실행 (`exec`) 호환은 시험 세션에서 확인한다.

### 7.3 프론티어 비교 (대장 `FRONTIER.md` "agentic 전문가 RL" · 조사 2026-11-12, 새 조사 없음)

| 모델 | 값 | 원천 | 규모 · 구조 차이 | alpha 적용 |
|---|---|---|---|---|
| Nemotron 3 Ultra (기준선) | 32 × 16 · lr 3e-6 · 4 에폭 약 964 step · 196,608 · 200 턴 · 3,600 s · adv clip ±100 | `swe_teacher.yaml` · 보고서 p22 | 550B, 128 노드 | 문맥 · 배치 · 동시 실행은 그대로 못 씀 (alpha 상한 128K, 2 노드) |
| Nemotron 3 Super | 16 × 32 · lr 1e-6 · 196,608 · 200 턴 | `stage2_swe2.yaml` (논문은 배치 · lr · step 공개 안 됨) | 대형, 32 + 노드 | 위와 같음 |
| Qwen3-30B-A3B (NeMo-RL 가이드) | 8 × 8 · lr 1e-6 · 131,072 · 200 턴 · 1,800 s | `grpo_qwen3_30ba3b_thinking_swe2.yaml` | 30B MoE, 같은 프레임워크 | 문맥 · 규모가 가장 가까운 제공 레시피 |
| Xiaomi MOPD SWE teacher (Qwen3-30B-A3B) | 80 × 8 · lr 3e-6 · 65,536 · 50 턴 · R2E-Gym-Lite · 약 150K 시퀀스. SWE-V 35.8 → 51.2 | arXiv 2606.30406 부록 A · 표 2 | 활성 3B, 규모 최근접 | 64K · 50 턴으로도 큰 이득을 보고. 타임아웃은 공개 안 됨 |
| Nemotron-Cascade 2 (30B-A3B) | 16 × 64 · 256K · 200 턴 (OpenHands). 16회 전부 통과 과제 제거 · 전부 실패 90% 폐기. lr · step 본문 미기재 | arXiv 2603.19220 §4.3 | NeMo-RL · 하이브리드 MoE | 과제 선별 방식의 선례 |

### 7.4 alpha 에서 정해야 하는 것 (전부 승인 대상 — 학습 결과를 바꾼다)

| # | 항목 | 제공 경로의 값 | alpha 의 제약 · 선택지 |
|---|---|---|---|
| 1 | 데이터 | Super `swe2` (Python 1,444) 또는 Ultra `swe` (다국어 7,816) | 이미지 준비 시간과 용량이 6 배 차이. `swe2` 는 Qwen3 가이드와 같은 데이터 |
| 2 | 문맥 · 턴 | 131,072 ~ 196,608 · 200 턴 | alpha 상한 128K (`RL_PLAN.md` §4). 학생의 SWE-V 실패에 단계 한도 127 건 · 문맥 초과 14 건이 있다 |
| 3 | 배치 · lr | 8 × 8 ~ 32 × 16 · 1e-6 ~ 3e-6 | 호스트 메모리 (다중 턴 궤적 × CP8 복제, `KNOWN_ISSUES.md` 2026-10-20) 와 bf16 반영률 (결정 23) |
| 4 | 동시 실행 · 메모리 상한 | 768 ~ 1,024 · 32 ~ 64 GB | 2 노드 440 코어 · RAM 4 TB 를 학습 · vLLM 과 나눠 쓴다. 롤아웃당 CPU · RAM · 시간은 미측정 — 스모크가 잰다 |
| 5 | 도구 형식 | OpenHands CodeAct · OpenCode · Codex 프롬프트 혼합 | alpha SFT 의 도구 형식과 얼마나 맞는지 미확인. SWE 피벗 데이터는 세 하니스 형식을 모두 담고 있다 |
| 6 | 빈 롤아웃 | 실패한 롤아웃은 `swe_agents` 가 `instance_config.mask_sample` 로 표시하고 (NeMo-RL 이 읽는 위치) GRPO 가 loss 에서 뺀다 (`grpo.py` `_apply_mask_sample_filter`, 이 fork 에 있음) | 생성 토큰이 아예 없는 결과가 오면 `nemo_rl/environments/nemo_gym.py` 가 `ValueError` 로 step 을 죽인다 (831행). upstream main 도 같은 예외를 던진다 (2026-11-20 확인). 컨테이너 기동 실패가 여기에 걸리는지는 미측정 — 스모크에서 없는 `.sif` 로 일부러 일으켜 본다. 고쳐야 하면 본체 패치 (별도 커밋 · 독립 검토) |
| 7 | 과제 선별 | Cascade 2 는 전부 통과 · 전부 실패 과제를 걸렀다 | 학생 통과율 사전 측정 (결정 40 의 5) 이 선별 근거가 된다 |
| 8 | 판정 벤치 | SWE-V | 기존 경로 (gpu06 docker) 그대로 |

## 8. Terminal E2E — 제공되는 것 조사 (2026-11-20, 읽기 전용)

조사 에이전트의 보고 (파일 경로 근거) 를 옮긴 것이다. 실패 마스킹을 읽는 위치 · `harbor_agent` 의 실패 처리 · 로컬 데이터 존재는 주 세션이 다시 확인했다.

- 학습 기능 (token ID · 다중 턴 연속성 · "NeMo RL Training" 문서) 을 갖춘 terminal E2E 경로는 Gym `harbor_agent` + `Terminus2NemoGym` **하나** 다.
- 그 행동 형식 (본문의 Terminus-2 JSON) 은 alpha terminal SFT 데이터 (NTC 365,705 궤적) 와 **같다**.
- 이 환경을 쓰는 NeMo-RL 레시피는 **없다**. 공개 Ultra · Super 블렌드 13 개에도 해당 행이 0 이다.
- 실패한 롤아웃을 마스킹하는 장치가 없다. 실패하면 빈 output + 보상 0 을 돌려주고, NeMo-RL 은 빈 결과에서 step 을 죽인다.

| 환경 | 행동 형식 | 컨테이너 명령 | 학습 지원 | alpha 에 걸리는 점 |
|---|---|---|---|---|
| `harbor_agent` + Singularity 환경 (기본 설정) | 본문 JSON (Terminus-2) | `singularity exec --no-mount … --pwd … --writable-tmpfs --fakeroot --containall --pid` (`custom_envs/singularity/singularity.py:364-377`) — 리허설에서 root-mapped 로 PASS | 예 — token ID · logprob 수집, 직전 턴 ID 를 다음 요청에 실음. `mask_sample` 없음 (0 건) | 빈 결과 → `ValueError` · 이미지 안에 python3 + uvicorn 필요 · 실행 파일 이름 `singularity` 고정 · 세션 network 공유 |
| `harbor_agent` + Gym sandbox 환경 | 같음 | Gym Apptainer provider | 제공 설정은 평가용 (token 수집 끔) | pull 가능한 이미지 참조만 받음 · 기동 사용자로 실행 |
| `anyterminal_agent` + `terminus_2_agent` | 본문 JSON (Terminus-2) | `apptainer instance start` 뒤 매 exec 가 `--fakeroot instance://` | 부분 — 기본값이 token 수집 끔 · 요약 켬, 학습 설정 · 문서 없음. `mask_sample` 을 응답 최상위에 두는데 NeMo-RL 은 `instance_config.mask_sample` 만 읽는다 (`nemo_rl/experience/rollouts.py` `_extract_mask_sample_flags`) | 리허설에서 실패한 exec 형태 (§5 의 우회를 Gym 설정으로 옮길 수 있는지 미시험) |
| `anyterminal_agent` + `hermes_agent` | 함수 호출 | 같음 | token ID 수집 패치 있음 | alpha terminal SFT 형식과 다름 |
| `terminal_bench_2_1` 서버 + OpenCode | 함수 호출 | PTY 필요 (opensandbox 전용) | 아니오 — 정책 롤아웃의 verify 가 미구현 | 판정 벤치 자체 |

| 데이터 | 규모 | 로컬 | 주의 |
|---|---|---|---|
| Nemotron-Terminal-Synthetic-Tasks (Harbor 형식, `harbor_agent` README 의 예시 데이터) | `task.toml` 수 — easy 44,969 · medium 203,749 (두 샤드) · mixed 5,984 | 있음 `SFT/Nemotron-Terminal-Synthetic-Tasks/skill_based/` (압축 약 1 GB, 미해제) | 과제 이미지가 NVIDIA 내부 레지스트리를 가리킨다 (표본 2) — Dockerfile 로 다시 빌드해야 한다 (docker + 레지스트리 필요) · 정답 풀이 없음 · 채점이 실행 중 `pip install pytest` 를 한다 |
| Terminal-Bench 2.0 · 2.1 | 각 89 | gpu06 에만 | 판정 벤치 — 학습에 쓰지 않는다 |
| Terminal-Pivot-v1 (단일 스텝) | 31,111 | 있음 | 과제 환경은 들어 있지 않다 |

alpha 에 걸리는 제약:
- 문맥 128K. TB 궤적의 최대 prompt 는 중앙 107K · p90 183K 다 (Pai `SFT_BENCHMARKS.md`).
- 이전 턴 사고: alpha 는 에이전트 궤적에서 이전 턴 사고를 **보존한다** (Pai 설계 규칙 2026-09-14 — 도구를 쓰는 궤적은 보존, 도구 없는 일반 대화는 DSV4 방식으로 제거. terminal SFT 인 NTC 는 `--keep-history-think` 로 구웠다). 템플릿은 도구가 선언됐거나 도구 호출 · 도구 턴이 있으면 자동으로 보존한다. Terminus-2 는 도구를 선언하지 않고 터미널 출력을 user 메시지로 보내므로 템플릿이 일반 대화로 보고 자른다. 그래서 이 환경의 레시피는 `chat_template_kwargs: {truncate_history_thinking: false}` 를 **명시해야 한다** — `harbor_agent` README 의 학습 요구 (`README.md:153-175`) 와 같고, 평가 (Pai `SFT_BENCHMARKS.md` "TB-2 추론 복원") 에서 이미 쓰는 설정이다. 빠지면 SFT 와 다른 렌더가 되고 다중 턴 토큰 연속성도 깨진다. alpha 레시피 규약의 `chat_template_kwargs: null` 은 템플릿이 시나리오를 스스로 가리는 환경용이다 (전역으로 켜면 일반 대화 이력까지 복원된다). SWE 의 `swe_agents` 는 도구를 선언하므로 `null` 그대로 보존된다. 이 경로의 다중 턴 토큰 연속성은 RL 쪽에서 아직 재지 않았다 (GATES G3 미실행).
- NTC SFT 의 시스템 프롬프트와 Harbor 평가 프롬프트의 md5 가 다르다. Gym 이 고정한 Harbor 판의 프롬프트가 어느 쪽인지 미확인.

제공되지 않아 만들어야 하는 것: 레시피 · `harbor_agent` 의 실패 마스킹 · 과제 이미지 빌드 경로 · 학습/검증 분할과 TB · NTC 겹침 검사 · 128K 에 맞춘 턴 수와 타임아웃. SWE 와 달리 "받아서 바로 돌리는" 경로가 아니다.

## 9. 열린 결정

| # | 결정 | 권고 | 시점 |
|---|---|---|---|
| 1 | 태스크 이미지 변환 범위 | **`swe2` 1,444 개 완료 (2026-11-21, §7.2)**. 남은 결정: Ultra `swe` 7,790 개를 더 받을지와 풀 상한 — 80/시간 유지면 약 49 시간, 1,444 개를 받는 동안 한도는 한 번도 소모 · 거부되지 않았다 | 레시피의 데이터 결정과 함께 |
| 2 | SWE 데이터 | 이미지는 `swe2` 부터 받는다 (결정 1). 학습 데이터로의 확정과 Ultra `swe` 확대는 `swe2` 변환 뒤 | 레시피 전 |
| 3 | 학생 통과율 사전 측정 (결정 40 의 5) | gpu06 docker 평가 경로로 지금 잴 수 있다. 과제 선별의 근거가 된다 | 지금 (GPU 승인) |
| 4 | `--pid` 처리 방식 | PID namespace 유지 + 바깥 `/proc` bind (§5) | 시험 세션 |
| 5 | 레시피 값 (문맥 · 턴 · 배치 · lr · 동시 실행 · 메모리 상한) | Qwen3 가이드를 기준안으로, 스모크 실측으로 조정 — 별도 보고 | 스모크 전 |
| 6 | 빈 롤아웃 처리 | upstream 수정 여부 확인 → 없으면 본체 패치 (별도 커밋 · 독립 검토) | 스모크 전 |
| 7 | 세션 단위 적용이 안 될 때 노드 전체 적용 수용 | 인프라 답을 본 뒤 | 인프라 답 |
| 8 | gpu06 `alpha-eval` 의 리허설 산출물 삭제 | 시험 세션 결과와 대조한 뒤 | 시험 세션 뒤 |
| 9 | Terminal E2E 착수 여부 · 환경 | 제공 경로는 `harbor_agent` 하나이고 레시피 · 이미지 · 실패 마스킹을 만들어야 한다 (§8). SWE 를 먼저 하고 그 뒤에 설계 보고 | SWE 뒤 |

## 10. 산출물

| 위치 | 내용 |
|---|---|
| `tools/apptainer/acceptance.sh` | 시험 세션 수용 시험 (단계마다 원인 하나) |
| `tools/apptainer/setup_session.sh` | 세션에 Apptainer 설치 · 우리 쪽 설정 (`--pid-mode bindproc` · `--multi-uid`). 세션을 다시 만들 때마다, 두 노드에서 |
| `tools/apptainer/build_sifs.py` | 블렌드가 참조하는 이미지를 `.sif` 로 (이어받기 · 풀 속도 제한 · 이름 검산 · 샤드 분할 · 내용 확인 `--verify` · 자가 시험 `--self-test`) |
| `tools/apptainer/launch_build_sifs.sh` | 본 변환을 세션과 분리해 띄운다 (노드마다 샤드 하나, 프로세스 그룹 기록) |
| `$NRL_ROOT/sif/swe/` | 변환된 `.sif` · 기록 `build_log.shard*of2.jsonl` · 로그 `logs/` |
| `tools/apptainer/seccomp_rules.json` · `docker-apptainer` · `docker-apptainer.userns` · `infra_worksheet.md` | 인프라에 넘긴 요청 사양과 작업 안내 |
| `$NRL_ROOT/apptainer_test/rehearsal_20261120/` | 리허설 스크립트 · 프로파일 · 로그 (`final_run.log` · `round2/run_{A..F}.log`) · 검증 자료 (`review/01~06`) |
| `$NRL_ROOT/apptainer_test/sif_samples/` · `alpine.sif` | 수용 시험용 `.sif` (R2E-Gym · SWE-rebench-V2 · SWE-Gym 각 1) |
| `$NRL_ROOT/apptainer_test/root145/` (1.4.5) · `root/` (1.5.5) | 권한 없이 푼 Apptainer — `LD_LIBRARY_PATH=$NRL_ROOT/apptainer_test/root/usr/lib/x86_64-linux-gnu` 로 실행한다. 지금 세션에서는 변환 (`build`) 만 된다 |
| gpu06 `alpha-eval` | 이미지 `apptainer-rehearsal:1` 434 MB · `ubuntu:24.04` 117 MB · 볼륨 `apptainer_rehearsal` 2.3 GB · `/root/apptainer_rehearsal` |
