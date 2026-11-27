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
| 인프라 요청 | **적용 완료 (2026-11-24)** — 인프라가 세 설정을 넣었고 12:47 에 세션을 다시 만들었다 (§6.1). 요청서는 `tools/apptainer/infra_worksheet.md` (11-24 개정) |
| 요청 사양 검증 | 별도 Docker 호스트 리허설 2회 + 독립 검증 2회, 재판정 = 확인 (§4) |
| 수용 시험 | **PASS (2026-11-24, 두 노드)** — 0 ~ 8 단계 실패 0. 같은 스크립트가 재생성 전 세션에서는 5 단계 FAIL 이었다 (§6.1) |
| 우리 쪽 Apptainer 설정 | Apptainer 1.4.5 + `--pid-mode bindproc` 를 두 노드에 적용 (사용자 승인 2026-11-24, §6.1). 실행 쪽이 `APPTAINER_BIND=/proc` 를 넘겨야 한다 |
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

**역할 구분**: 인프라가 할 일은 위 세 건뿐이다. 호스트에 Apptainer 를 설치할 필요는 없다 — 설치는 요청자가 세션 안에서 한다 (`setup_session.sh`, 세션에서 `sudo` 가능 2026-11-24 확인). 세 건은 컨테이너 생성 때 정해져 세션 안에서는 root 로도 못 바꾼다.

**적용 지점** (Backend.AI 공개 코드 main 기준, 배포 버전 미확인 · 적용 미시험 — 요청서 (2026-11-24 개정) `tools/apptainer/infra_worksheet.md`):
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

### 6.1 적용 결과 (2026-11-24, 새 세션 — GPU 미사용)

세션 재생성 12:47 (이미지 동일 NGC 25.05 · 커널 6.5.0-15). **역할 이름과 호스트가 뒤바뀌었다** — 새 main1 의 GPU UUID 8/8 이 이전 sub1 과, 새 sub1 이 이전 main1 과 일치한다. 주소도 바뀌었다 (main1 `10.0.41.2` · sub1 `10.0.41.4`, 이전 `10.0.37.4` · `10.0.37.2`). 기록 `project_s/reboot_restore/logs/gpu_identity_{main1,sub1}_20261124.txt`.

| 확인 | main1 | sub1 |
|---|---|---|
| 수용 시험 0 ~ 6 (세 설정) | 실패 0 — seccomp 활성 · capability 그대로 · AppArmor `docker-apptainer (enforce)` · user namespace 생성 · 그 안의 tmpfs mount · `pivot_root` · `/dev/fuse` 읽기쓰기 | 같음 |
| 과다 부여가 없는지 | 새 network namespace 거부 (`unshare -Urn` · `clone(NEWUSER\|NEWNET)` EPERM) · `keyctl` · `io_uring_setup` EPERM · `clone3` ENOSYS | 같음 |
| Apptainer 설치 (`setup_session.sh`, 기본 설정) | 1.4.5 · apt 신규 4 · 업그레이드 0 | 같음 |
| 수용 시험 7 ~ 8 (`alpine_145.sif`) | `apptainer exec` (FUSE 로 SIF 마운트) · `--writable-tmpfs` PASS | 같음 |
| 수용 시험 9 (`--pid`, 기본 설정) | 예상대로 실패 — `can't mount proc filesystem to /proc` | 같음 |
| 실제 과제 이미지 (`r2egym_datalad_final_71d543ec….sif`), `swe_agents` 플래그 (`--writable-tmpfs --cleanenv --pid --no-mount home,tmp,bind-paths`) + bindproc | 실행 rc 0 · 컨테이너 안 uid 1100 · 바깥 세션 프로세스에 `kill` 이 닿지 않음 · 프로세스 치환 동작 | 같음 |
| 채점 음성 대조 — 저장소를 고치지 않고 `run_tests.sh` | 3 failed · 1 passed · 1 error (데이터의 "수정 전" 기록과 같음, 기대 결과 표와 불일치 → 보상 0) | 같음 |
| 채점 양성 대조 — 정답 한 줄 (`kwargs = kwargs.copy()`) 을 넣고 `run_tests.sh` | 2 failed · 2 passed · 1 error — **`expected_output_json` 과 다섯 테스트 전부 일치** (→ 보상 1) | 같음 |

- bindproc 시험은 시스템 설정을 바꾸지 않고 했다: `/etc/apptainer/apptainer.conf` 의 사본에서 `mount proc = no` 로 바꿔 `apptainer --config <사본>` 과 `APPTAINER_BIND=/proc` 로 실행 (사본 `$NRL_ROOT/apptainer_test/session_20261124/bindproc.conf`, 스크립트 `inside2.sh`). 시스템 설정은 기본값 그대로다.
- 이것으로 §4 의 "시험 세션에서만 닫히는 것" 가운데 AppArmor 집행 · 커널 6.5 · NVIDIA `/proc` 덮임 · Backend.AI 의 실제 적용 경로가 닫혔다. 남은 것: Gym `swe_agents` 가 띄우는 경로에서 `APPTAINER_BIND` 가 전달되는지, 동시에 여러 샌드박스가 돌 때, `--writable-tmpfs` 64 MB 한도 — 스모크에서 본다.
- 위 대조는 과제 한 개다. R2E-Gym 전체에 대한 채점기 대조는 Gym 의 채점 코드 (`gold` · `empty` 입력) 로 따로 한다.
- 로그: `$NRL_ROOT/apptainer_test/session_20261124/{setup,accept}_{main1,sub1}.log`.
- **bindproc 을 시스템 설정에 적용했다** (사용자 승인 11-24, `setup_session.sh --pid-mode bindproc`, 두 노드). 시스템 설정 + `APPTAINER_BIND=/proc` 로 같은 결과 (무작업 불일치 · 정답 5/5 일치). `APPTAINER_BIND` 가 빠지면 컨테이너 안 `/proc` 가 비어 있다 — Gym 경로에서 이 값이 전달되는지는 스모크 항목.
- 세션을 한 번 더 재생성했다 (14:03, Pai 셋업 사고 때문 — `RESTORE_AFTER_REBOOT.md` §7.11). **세 설정은 재생성 뒤에도 유지됐다** (수용 시험 두 노드 18 PASS · 0 FAIL). 로그 `…/session_20261124b/`.

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
- **alpha 가 이 언어들을 얼마나 봤나 (2026-11-27 ~ 28 측정, CPU 표본 스캔)**: 사전학습 코드 말뭉치 (Nemotron-Pretraining-Code-v1 Synthetic-Code, 3.13 억 행) 의 언어는 Python · SQL · Shell · R · Java · PHP · C · JavaScript · Makefile · C# · C++ 뿐이다. 본 런 데이터 (`teacher_swe_e2e_v1.jsonl` 7,002 과제) 의 49.7% 가 거기 없는 언어 (TS · Go · Rust · Julia · Kotlin · Clojure · Dart) 이고, `s4` 에서 한 번이라도 성공한 과제의 비율은 그 언어들 3.1% · 나머지 8.2% 였다. 다른 단계에서의 노출 (문서 · 궤적 안에 그 언어의 파일 확장자나 코드 블록 표기가 **나오는** 비율 — 과제의 언어가 아니다. 도구 설명 같은 상용구도 잡혀 Python · TS 는 부풀 수 있다):

  | 데이터 (표본) | TS | Go | Rust | Julia | Kotlin | Clojure | Dart | 참고: Python · JS |
  |---|---|---|---|---|---|---|---|---|
  | 사전학습 후반 웹 코드 말뭉치 `cc_code` (100,000) | 0.6% | 0.5% | 0.4% | 0.1% | 0.4% | 0.6% | 0.4% | 9.7% · 11.4% |
  | SFT SWE-v3 궤적 (4,500) | 13.9% | 8.4% | 4.4% | 1.7% | 0.7% | 0.0% | 0.0% | 80.7% · 18.4% |
  | SFT OpenCode-v1 궤적 (3,600) | 100% | 1.9% | 0 | 0 | 0 | 0 | 0 | 100% · 33.6% |
  | SFT Terminal-Corpus 궤적 (1,500) | 0.2% | 0.1% | 0 | 0 | 0 | 0 | 0 | 92.5% · 0.7% |
  | RL Code·Math 블렌드 (6,000) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100% · 0 |
  | RL terminal 피벗 블렌드 (4,000) | 1.9% | 3.7% | 2.1% | 1.7% | 0.1% | 0 | 0 | 72.0% · 5.4% |

  사전학습의 synthetic 하위 묶음 다섯 (question-answering · code-review · rewriting · student-teacher · transpilation) 은 언어 필드가 없어 이 스캔으로 재지 못했다 (미측정). 읽는 법: 일곱 언어는 SFT 의 SWE 궤적에서 TS · Go · Rust 정도만 눈에 띄게 나오고 Clojure · Dart 는 SFT 에 사실상 없다.
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

### 7.5 학생 통과율 사전 측정 — 경로 확인 (2026-11-23, 읽기 전용 · GPU 미사용)

§9 의 3 번이 "gpu06 docker 평가 경로로 지금 잴 수 있다" 고 적었으나 **그 경로로는 `swe2` 과제를 채점할 수 없다** (2026-11-20 작성 때 확인하지 않은 문장이었다).

| 항목 | 확인한 사실 |
|---|---|
| 기존 gpu06 경로 | mini-swe-agent 2.4.6 + swebench 5.0.2, SWE-bench Verified 전용. swebench 5.x 는 신 스키마 · 사전 빌드 이미지 데이터셋만 읽는다 (Pai `EVAL_DOCKER_NODE.md` §6.5) — R2E-Gym 1,172 개는 형식이 다르고 SWE-Gym 272 개는 구 스키마다 (실행 시험은 안 함) |
| R2E-Gym 행의 형식 (`swe2` 1,172 행 전수) | `FAIL_TO_PASS` · `PASS_TO_PASS` 가 전부 빈 목록이고 `patch` · `test_patch` 가 없다 (0 행). 정답은 `expected_output_json` — 테스트별 기대 결과 표 (행당 중앙 55 개, 579 행은 PASSED 아닌 기대값 포함) — 이고, 테스트는 이미지 안에 들어 있다 (`/r2e_tests/test_1.py` · `/testbed/run_tests.sh`, `.sif` 목록 실측). 이미지는 `docker_image` 필드 (`namanjain12/<repo>_final:<commit>`). 원 커밋의 변경은 `parsed_commit_content` 에 파일 내용으로 들어 있다. swebench 5.0.2 가 인스턴스에서 읽는 `image` · `eval_script` · `log_parser` · `test_patch` 가 하나도 없다. SWE-Gym 행은 `FAIL_TO_PASS` · `PASS_TO_PASS` · `test_patch` · `patch` 가 있고 `image` · `eval_script` · `log_parser` 만 없다 |
| 제공되는 경로 | Gym `swe_agents` 서버 + `.sif` + `gym eval run --no-serve --agent swe_agents` (README "Batch evaluation"). 인스턴스마다 Apptainer 컨테이너 둘 (에이전트 · 채점) 을 띄운다 → Apptainer 실행이 필요하다 |
| 채점기 | R2E-Gym = `r2egym…run_local_evaluation` (`setup_scripts/r2e_gym.sh`) · SWE-Gym = 기본 처리기의 SWE-bench fork (`swebench.sh`) · 에이전트 런타임 = OpenHands (`openhands.sh`, miniforge). 셋업 스크립트는 내려받기 · venv 뿐이라 Apptainer 없이 돈다 (미실행) |
| 정답 패치 검증 | `verify_golden_patch=true` 는 SWE-bench 계열 · SWE-rebench 만 지원한다. R2E-Gym 은 목록에 없다 |
| gpu06 `alpha-eval` | Apptainer 실행 조건은 충족 (user namespace 생성됨 · `/dev/fuse` 있음). Apptainer · Gym 은 설치돼 있지 않다. 64 코어 (공용, 부하 약 5) · RAM 가용 267 GB · 디스크 여유 3.3 TB · GitHub · PyPI 접속됨 |
| 전송 | main1 → gpu06 약 116 MiB/s (300 MiB 2.6 s) — `.sif` 100 개 (약 100 GiB) 에 약 15 분 |
| 학습 세션 | main1 · sub1 모두 여전히 Apptainer 실행 불가 (2026-11-23 13:56 재확인) |

그래서 "지금" 가능한 것은 gpu06 에 Gym `swe_agents` 환경을 새로 만들어 표본 (예: 100 과제 × 4) 을 재는 것이고, 과제 선별에 쓸 전수 측정 (1,444 × 4) 은 gpu06 의 공용 64 코어로는 수십 시간이라 학습 세션 (인프라 적용 뒤) 의 몫이다. 선택은 사용자 결정 (§9 의 3).

### 7.6 Gym 하니스 셋업 · 정답 패치 검증 (2026-11-24, GPU 미사용)

실행기 `$NRL_ROOT/p0_measure/swe/run_swe_golden.sh` — 측정용 Gym (`tools/Gym`, v0.6.0) 으로 `swe_agents` 서버를 `verify_golden_patch=true` 로 띄우고 `gym eval run` 으로 표본을 채점한다. Ray 는 alpha 클러스터 (`main1:6379`) 를 쓴다.

| 확인 | 결과 |
|---|---|
| 하니스 셋업 | 첫 기동 726 s 에 다섯 트리가 `$NRL_ROOT/gym_cache/swe_agents/` 에 설치됐다 (OpenHands · R2E-Gym · SWE-bench · SWE-bench Multilingual · SWE-rebench, 합 8.2 GB). `cache_dir` 를 이 경로로 주면 측정용 · 학습용 Gym 이 함께 쓴다. 두 번째 기동은 14 s |
| Ray 경로 | 채점 컨테이너가 Ray 워커에서 뜬다 (sub1 워커 확인). raylet 환경의 `APPTAINER_BIND=/proc` 로 `--pid` 가 동작 — `runs/ray_cluster.sh` 가 `ray start` 에 넣는다 (드라이버 환경은 워커에 닿지 않는다) |
| **정답 패치 검증의 지원 범위** | 코드 (`_run_golden_patch_verification`) 는 데이터셋 이름에 `SWE-bench` · `SWE-rebench` 가 들어 있거나 `swe-bench-ext` · `deepswe` · `denovoswe` 일 때만 받는다. **Ultra `swe.jsonl` 의 SWE-rebench-V2 7,610 행 (97.4%) 은 지원, SWE-Gym 206 행은 미지원. Super `swe2.jsonl` 은 R2E-Gym 1,172 · SWE-Gym 272 모두 미지원** — `swe2` 표본 10 개는 전부 `NotImplementedError` |
| SWE-rebench 표본 5 개 (python · go · php · rust · java, `rebench_sample.jsonl`) | 38 s 에 5 개 채점: **4 개 보상 1.0**, rust `sharkdp__bat-3206` 만 0 |
| rust 실패의 원인 | **Apptainer `--writable-tmpfs` 의 쓰기 공간이 64 MB** (`apptainer.conf` `sessiondir max size = 64` 기본값). `cargo build` 가 `/bat/target` 에 쓰다 `No space left on device` 로 링크 실패. 같은 이미지에서 직접 확인: 기본 설정이면 덧씌움 64M 가 100% 차고 빌드 실패, 설정 사본에서 16,384 로 올리면 31.96 s 에 빌드 완료 (덧씌움 사용 460 MB) |

- **쓰기 공간을 16,384 MB 로 올렸다** (사용자 승인 11-24, 결정 75 — `setup_session.sh --sessiondir-mb 16384`, 두 노드). 같은 표본을 다시 돌려 **5/5 보상 1.0** (rust 채점 45 s). 64 MB 한도는 `swe2` 에도 걸리던 것이다.
- 이 공간은 tmpfs 라 쓴 만큼 세션 메모리에 잡힌다 — 실측: 컨테이너가 3 GB 를 쓰는 동안 세션 cgroup 의 shmem 0.00 → 2.93 GiB, 끝나면 0.00. 세션 컨테이너의 메모리 한도는 노드마다 1,772 GiB (cgroup `memory.max`), 유휴 사용 8 ~ 36 GiB.
- **정답 패치 검증 — Ultra SWE-rebench 687 개 (11-24, 변환과 같은 노드 · 동시 16, `golden_ultra_b2` · `b3`)**: 통과 648 (94.3%). 언어별 python 131/134 · ts 121/129 · js 118/121 · go 87/94 · rust 66/70 · php 51/58 · julia 29/32 · java 23/25 · kotlin 9/9 · r 5/7 · clojure 6/6 · c 1/1 · cpp 1/1. 채점 시간 중앙값 29 s · p90 145 s · 최대 1,200 s.
  실패 39 건의 분류 (`p0_measure/swe/summarize_golden.py` — 사유는 워커 로그의 `Golden-patch eval failed for …` 줄과 과제별 `report.json` 에서 읽는다. 정답 패치 모드의 결과 행은 시간 초과 · 메모리 종료 필드가 항상 비어 있다): ① FAIL_TO_PASS 는 다 통과하고 PASS_TO_PASS 일부만 실패 14 ② 통과한 테스트가 하나도 없음 12 (테스트가 못 돈 경우 — 외부 내려받기 404 · vendor 파일 없음 · `dpkg` 가 superuser 요구 · R 패키지 설치 실패 · php 4 건은 `su` · `chown` 거부) ③ FAIL_TO_PASS 일부 실패 5 ④ **메모리 감시 32 GB 초과로 채점 컨테이너 종료 4** (최대 33 ~ 36 GB) ⑤ 둘 다 일부 실패 2 ⑥ 채점 명령 시간 초과 (1,200 s) 2.
  정정 (11-24 독립 검토): 처음 분류는 로그 문자열 맞추기로 "시간 초과 흔적 6 · 네트워크 1 · 메모리 감시 1" 이라 적었으나 테스트 이름 · 컴파일 출력에 맞은 오분류였다 (메모리 감시는 4 건). 어느 실패가 변환 부하 탓인지는 이 자료로 가를 수 없다 — 다시 돌려서 잰다 (§7.7).
  실행기 결함 (수정됨): 모델 서버 없이 띄우면 Gym 이 600 초 뒤 "model endpoint never answered" 로 서버를 내려, 그때 돌던 과제가 500 으로 끝나고 클라이언트가 `/aggregate_metrics` 재시도로 멈춘다 → `model_endpoint_readiness_timeout_seconds: 0`.
- **Gym 실행이 강제로 죽으면 컨테이너의 FUSE 데몬이 남는다** (11-24 실측): 정답 패치 검증 도중 실행기가 죽자 `squashfuse_ll` · `fuse-overlayfs` 가 부모 1 로 컨테이너마다 한 쌍씩 남았다 (main1 8 쌍 · sub1 2 쌍, 안에서 도는 프로세스는 없음). 쓰기 공간 (tmpfs) 을 쥐고 있으므로 쌓이면 세션 메모리를 먹는다. 정리는 `comm` 이 그 둘이고 부모가 1 인 프로세스를 PID 로 내린다 (이미지 변환 등 다른 Apptainer 작업의 것은 부모가 살아 있다). 학습 런을 내린 뒤 · 다음 기동 전에 두 노드에서 확인한다 — `launch.sh` 의 기동 전 정리에 넣을지는 스모크 계획에서 정한다.
- 컨테이너는 두 노드에 뜬다: Gym 서버 프로세스는 한 노드에 있지만 컨테이너는 Ray 태스크 (`runner_ray_remote`, `scheduling_strategy="SPREAD"` · `num_cpus=0.1`) 가 띄우고, 학습 때 NeMo-RL 이 Gym 에 클러스터의 GCS 주소를 넘긴다 (`nemo_gym.py:487-490`). 같은 옵션의 탐침으로 실측: 128 개 → main1 64 · sub1 64, 64 개 → 32 · 32, 8 개 → 4 · 4.
- 상한 계산: 롤아웃 하나는 에이전트 컨테이너가 끝난 뒤에 채점 컨테이너가 쓰기 시작하므로 가득 찬 컨테이너는 롤아웃당 하나로 본다. 동시 롤아웃 128 이 전부 16 GB 를 채우는 최악이 2,048 GiB — 실측대로 반씩 나뉘면 노드당 1,024 GiB 로 한도 (1,772 GiB) 안이다. 한 노드에 몰린다면 넘는다 (SPREAD 는 보장이 아니라 실측된 동작). 예상 사용은 훨씬 작다 (rust 빌드 0.46 GB). 스모크에서 노드별 shmem 최대치를 재고, 동시 실행을 128 넘게 올릴 때는 이 계산을 다시 한다. 메모리 감시 (`apptainer_memory_limit_mb`) 는 프로세스 RSS 만 세고 이 공간은 세지 않는다.
- `swe2` 의 채점기 대조는 Gym 의 정답 패치 검증으로는 할 수 없다. R2E-Gym 은 채점 코드의 `gold` · `empty` 입력을 직접 불러야 한다 (미실시; 과제 한 개의 손 대조는 §6.1).
- SWE-rebench 표본 이미지 5 개는 `$NRL_ROOT/sif/swe/swerebench/` 에 있다 (1.4.5 로 변환, 합 3.6 GB).

### 7.7 스모크 계획 — 선별 게이트와 학생 롤아웃 측정 (2026-11-24, 승인 전 · GPU 미사용 상태)

단계는 빠른 것부터다. S3 · S4 는 S2 의 실측으로 값을 정한 뒤 따로 보고한다. 초안은 독립 검토 (Opus · xhigh, 읽기 전용) 에서 선별 게이트 "보완 필요" · 학생 측정 "신뢰 불가" 를 받았고 아래는 그 지적을 반영한 판이다 (지적과 조치는 이 절 끝의 표). 재검토 판정: 선별 게이트 "신뢰 가능 (채점 컨테이너 범위에서)" · 학생 측정 "보완 필요" — 그 보완까지 반영했다.

| 단계 | 내용 | GPU | 잡는 결함 | 시간 |
|---|---|---|---|---|
| S1a | 정답 패치 검증 (`run_swe_golden.sh`, Gym `verify_golden_patch`) | 0 | 우리 환경에서 정답도 0 점인 과제 | 687 개 완료, 변환을 따라 계속 |
| S1b | 채점기 음성 대조 — 통과 과제의 정답 패치를 아무것도 안 고치는 패치 (새 파일 하나) 로 바꿔 재채점 (`make_noop_input.py`) | 0 | 안 고쳐도 통과하는 과제 · 채점기가 패치를 보지 않는 결함 · 고치지 않은 저장소에서 흔들리는 테스트 | 648 개 약 1 시간 |
| S1c | 반복성 — 게이트 통과 과제 100 개 (무작위) 와 실패 39 개의 정답 패치를 다시 채점 | 0 | 한 번의 결과가 뒤집히는 비율 (채점 잡음 바닥) · 부하 탓 실패 | 약 30분 |
| S2-dry | 측정 실행기의 기동 경로를 GPU 없이 돌린다 (`DRY=1` — fleet 없이 모델 주소를 죽은 포트로, 과제 4 개) | 0 | 실행기 결함: Gym 서버 주소 · 두 노드의 컨테이너 기동 · 클라이언트 접속 · 시간 한도 · 컨테이너 정리 | 약 10분 |
| S2-stub | 가짜 모델로 에이전트 고리 전체를 돈다 (`STUB_URL` — `$NRL_ROOT/gym_trial/stub_model.py` 가 "정답 패치 적용 → 종료" 도구 호출을 돌려준다, 과제 4 개). 기대 결과는 전부 보상 1 | 0 | 모델 호출 뒤의 결함: 요청 검증 (도구 결과 메시지 포함) · 도구 실행 · 패치 추출 · 채점 · 결과 조립. 패치 전달의 양성 대조. 모델의 실력과 vLLM 의 도구 호출 파싱은 보지 않는다 | 약 3분 |
| S2a | 학생 롤아웃 소량 — 게이트 통과 과제 8 개 × 1 회 (`runs/eval_swe_student.sh`) | main1 8 | 에이전트 컨테이너 기동 · 두 노드에서 모델 호출 · 도구 실행 · 결과 스키마 | 약 30분 + fleet 기동 10분 |
| S2b | 학생 롤아웃 — 게이트 통과 과제 64 개 × 8 회 (언어 비율대로, `make_student_input.py`) · 동시 128 | main1 8 | 학습 신호가 있는가 (과제별 보상 갈림) · 턴 · 시간 · 메모리 · 종료 사유 | 약 2 ~ 2.5 시간 |
| S2c | 학생 패치 재채점 — S2b 의 빈 패치가 아닌 패치를 정답 패치 모드로 다시 채점 (CPU) | 0 | 에이전트 → 패치 → 채점 전달 결함 · 실제 패치에서의 채점 잡음 | 약 30분 |
| S3 | 학습 스모크 (4 × 4 · 3 step · 짧은 턴, `launch.sh` `WANDB=0`) | 16 | 학습 경로 — 다중 턴 토큰 연속성 · KL · 마스킹 · 빈 롤아웃 · refit | 별도 보고 |
| S4 | 본 런의 첫 step 들로 최종 판정 | 16 | — | 별도 보고 |

**선별 게이트 (학습 데이터에 넣는 조건, `p0_measure/swe/golden_lib.py` `gate`)**
1. 그 과제가 들어간 **모든** 정답 패치 실행에서 보상 1. 다시 돌려 통과한 과제는 들이지 않는다.
2. no-op 실행의 채점 보고서가 있고 FAIL_TO_PASS 중 통과하지 못한 것이 있다. 보상 0 만으로는 부족하다 — 정답 패치 모드는 채점이 죽어도 (시간 초과 · 메모리 종료) 보상 0 을 낸다.
3. no-op 실행에서 PASS_TO_PASS 가 전부 통과. 고치지 않은 저장소에서 기존 테스트가 흔들리면 보상에 잡음이 낀다.
4. 정확한 이름의 `.sif` 가 있다. Gym 은 이름이 없으면 그 이름으로 **시작하는** 다른 파일을 조용히 쓴다 (`app.py` `_find_container` — `…-8` 이 `…-89.sif` 에 붙는다). 학습 데이터 파일에는 이 조건을 통과한 행만 둔다.

Gym 의 정답 패치 검증이 받지 않는 SWE-Gym 206 행은 게이트를 통과한 것으로 치지 않는다 (제외 여부는 본 런 보고에서 결정). 정답 패치 모드와 실제 롤아웃은 같은 채점 명령 · 마운트 · 결과 해석을 쓴다 (`app.py` `_setup_params`) — 그래서 정답 · no-op 쌍이 채점 컨테이너의 진짜 대조다. 에이전트 컨테이너와 패치 추출은 이 게이트가 보지 않는다 (S2 · S2c 의 몫).

**S2 의 설정** — 학습 레시피 초안과 같은 에이전트 설정: 200 턴 · 에이전트 1,800 s · 턴당 생성 8,192 · temperature 1.0 · top_p 1.0 · 메모리 감시 32,768 MB · 문맥 131,072 · 동시 128 · `use_absolute_ip: true`. 프롬프트는 CodeAct 하나다 — 이 Gym (`3045a793`) 의 학습 설정이 그렇다 (아래 "프롬프트 혼합" 참조). 모델은 `mopd1/hf/hf_step_00120` 을 Pai fleet (main1 GPU 8) 로 서빙한다 (`eval_lcb.sh` 와 같은 플래그: `TOOLS=1 TOOL_PARSER=qwen3_xml REASONING_PARSER=nemotron_v3 VLLM_ENFORCE_STRICT_TOOL_CALLING=0 PREFIX_CACHE=0`). 동시 128 / GPU 8 은 학습 때의 롤아웃 노드와 같은 비율이다.
이것은 **대리 경로**다 — 벤치와 같은 서빙이고 학습 때의 NeMo-RL vLLM 워커가 아니다. 토큰 ID · logprob · refit · R3 는 S2 가 보지 않는다 (S3 의 몫). 에이전트 시간 한도는 벽시계라 시간 초과 · 턴 · 시간 분포는 서빙 처리량에 달려 있다 — S2 의 값으로 레시피의 한도를 바로 정하지 않는다. 이미지 변환이 같은 노드에서 돌면 부하가 섞인다 → 15 초마다 load 를 함께 적는다.

**S2 의 판정 기준** (하니스가 제대로 돌았는가. 통과율은 기준이 아니라 측정값이다)

| # | 기준 | 지키는 것 |
|---|---|---|
| 1 | 결과 수: `rollouts` + `rollouts_failures` = 입력 행 × 반복. `agent_run_error` 가 2% 이하면 통과, 2 ~ 10% 는 원인을 확인한 뒤 판단, 10% 초과는 중단 | 조용히 빠진 행 |
| 2 | 붙은 대상: 기동 전 포트 8100 · 8000 ~ 8007 이 비어 있고, 백엔드 8/8 과 프록시 :8100 (16 회) 의 `/v1/models` 가 이 체크포인트 경로를 보고 (실행기가 강제) | 다른 모델 · 묵은 서버 · 묵은 터널 (`KNOWN_ISSUES.md` 11-11) |
| 3 | 두 노드에서 모델에 닿았다: Gym 서버가 전부 main1 의 노드 주소에 묶이고 (실행기가 강제), 모델 응답 항목이 1 개 이상인 롤아웃이 95% 이상이며, 워커 로그에서 sub1 이 돌린 롤아웃이 전체의 30% 이상 (컨테이너는 두 노드에 반씩 뜬다 — 절반이 닿지 못하면 95% 가 될 수 없다). 결과 행에는 워커 노드가 없어 노드별 비율은 워커 로그로만 본다 (`RAY_DEDUP_LOGS=0`) | 컨테이너가 모델 서버에 접속 못 함 |
| 4 | 도구가 실제로 실행됐다: 롤아웃의 `response.output` 에서 도구 호출 항목과 오류가 아닌 도구 결과 항목을 세어, 실행된 명령 · 편집이 1 개 이상인 롤아웃이 90% 이상. 도구 결과의 `status` 는 늘 completed 라 오류는 내용 문자열로 가른다. 파싱 실패 (도구 호출 표기가 든 글만 있고 도구 호출 항목이 없는 턴) 비율을 함께 보고 | 도구 호출이 파싱되지 않아 대화만 함 (명령 실행 시간 · 턴 수로는 못 잡는다 — OpenHands 가 모든 Action 에 시간을 더하고 글만 답해도 "계속하라" 로 턴이 돈다) |
| 5 | 패치 전달: ① 채점 로그의 `Checking patch <경로>` 묶음을 모델 패치 · 테스트 패치의 파일 목록으로 갈라, 모델 패치 쪽에 `Rejected hunk` · `patch failed` · `does not apply` 가 없다 (테스트 패치가 거부되는 것은 학생이 테스트 파일을 고친 정당한 결과일 수 있다) ② S2c 재채점의 보상이 원래 보상과 같다 — 불일치 비율이 S1c 에서 잰 뒤집힘 비율 수준 (S2c 는 패치 → 채점만 본다. 추출이 틀려도 같은 패치를 다시 채점하므로 일치한다) ③ 추출: 성공한 편집 도구 호출이 고친 파일이 (되돌린 것을 빼고) 패치의 파일 목록에 들어 있다 · 편집이 성공했는데 패치가 빈 롤아웃의 비율 | 패치 추출 · 적용 실패가 "학생이 못 풂" 으로 보이는 것 (채점은 `git apply … \|\| true` 라 적용 실패를 삼킨다. 보상 = 보고서 `resolved` 라 둘의 일치는 대조가 못 된다) |
| 6 | 잔재 없음: 이 실행의 컨테이너 0 (`runs/swe_kill_containers.py` — 결과 디렉토리를 마운트한 프로세스를 두 노드에서 찾는다) · FUSE 고아 수가 기동 전과 같다 · Gym 서버 0 · GPU 반납 | 다음 기동 방해 · 메모리 누적 (컨테이너는 자기 세션으로 떠서 Gym 서버를 내려도 남고, 채점 컨테이너는 에이전트 출력 파일을 끝없이 기다린다) |

기준 3 ~ 5 의 필드 이름은 S2a 결과에서 확정한다 (정답 패치 모드의 행에는 그 필드가 비어 있어 미리 확인할 수 없다). S2a 가 끝나면 요약 스크립트를 실제 행에 맞추고, 없는 값은 0 이 아니라 결측으로 센 뒤에 S2b 를 돌린다.

**S2b 가 재는 값과 미리 정한 판정** — 과제 64 개를 8 회씩 돌려 과제별 보상 8 개를 얻는다. 학습의 한 프롬프트 (8 회 생성) 가 기울기를 내려면 그 8 회의 보상이 갈려야 한다. 과제당 1 회로는 그 비율의 상한만 나온다 (평균 통과율 5% 가 전 과제 균일이면 34%, 5% 의 과제만 항상 통과면 0).
핵심 값은 **q = 64 개 중, 8 회의 보상이 전부 같지 않고 마스킹되지 않은 롤아웃이 하나 이상 있는 과제의 수 / 64** 다. 마스킹된 롤아웃도 보상 비교에 넣는다 — 학습의 기준선 (`advantage_estimator.py`) 이 마스킹 여부와 상관없이 8 회 전부로 계산되고, 마스킹은 loss 만 0 으로 만든다 (`grpo.py` `_apply_mask_sample_filter`). 분모는 64 로 고정하고 `agent_run_error` 가 있는 과제 수 · 마스킹 비율 · 과제별 비마스킹 수 분포를 따로 보고한다. 잘못된 도구 호출 · 사고 태그의 −5 벌점은 보상과 무관하게 기울기를 내므로 q 에 넣지 않는다.
한 step 의 프롬프트 8 개가 전부 무신호일 확률은 (1 − q)^8 이다: q 15% 면 27%, 5% 면 66%. 제안하는 판정 (임계는 이 식에서 고른 제안값이고 잡음에서 나온 값이 아니다 — 승인 대상): **보상이 갈린 과제가 10 개 이상 (15.6%) 이면 S3 로 · 3 개 이하 (4.7%) 면 중단 보고 · 4 ~ 9 개면 선택지 (프롬프트 수 확대 · 과제 선별) 를 보고**. 64 개 표본의 95% 구간은 넓다 — 3/64 는 1.6 ~ 12.9%, 6/64 는 4.4 ~ 19.0%, 10/64 는 8.7 ~ 26.4%. 참 q 가 10% 일 때 "진행" 으로 나올 확률 0.10 · "중단" 0.11, 참 q 가 15% 면 "진행" 0.50 이다. 그래서 가운데 구간 (선택지 보고) 이 기본 결과이고, 5% 를 믿을 만하게 가르려면 과제 200 개 이상이 필요하다.
함께 보고: 통과율 (95% 구간) · 종료 사유 분포 · 턴 수와 롤아웃 시간 분포 · 컨테이너 최대 메모리 · 노드 메모리 · load. 언어별 값은 표본이 작아 참고로만 적는다.

**S2-dry 결과 (11-24, GPU 미사용)** — 실행기의 서버 기동 · 클라이언트 접속 · 두 노드 배치 · 정리는 통과했다 (`student_dry1`: Gym 서버 21 초 · 4/4 행 반환 · main1 2 · sub1 2 · 합쳐진 로그 줄 0 · 잔재 0). **그러나 에이전트 컨테이너가 기동하지 못하는 결함이 드러났다** — OpenHands 가 셸을 `su root -` 로 열고 uid 1100 이라 실패한다 (`KNOWN_ISSUES.md` 2026-11-24, 독립 검증 확인). 대조 실험에서 `USER` 를 넘기면 15 초에 기동한다. 대응이 정해질 때까지 S2 는 띄우지 않는다.
S2-dry 의 통과 기준은 종료 코드가 아니라 단계 표지다: 모델이 없으면 고친 뒤에도 클라이언트가 끝없이 재시도해 시간 한도로 끝난다 → 과제별 `nemo_gym_metrics.json` 에 `connect_to_runtime_time` · `initialize_runtime_time` 이 있고 에이전트 로그에 `GLOBAL STEP 0` 이 있어야 한다, 두 노드 모두.
강제 중단 시험 (`student_dry2` 에 SIGTERM): 컨테이너 0 · Gym 서버 0 으로 정리됐으나 FUSE 데몬 4 개가 고아로 남았다 → 실행기가 "실행 전에 없던 고아" 만 내린다 (`swe_kill_containers.py orphans-kill`).

**`USER` 전달 뒤의 S2-dry (11-25 09:15, `student_dry3`, 결정 77)** — 두 노드 모두 에이전트가 기동했다 (main1 2 · sub1 2, 연결 11 ~ 17 s · 초기화 2 ~ 5 s · `GLOBAL STEP 0` · 셸 `/bin/bash` · 사용자 `work`, 잔재 0). **다음 결함**: 첫 모델 호출이 Gym 모델 서버에서 422 로 거부된다 (`aws_region_name`, `KNOWN_ISSUES.md` 2026-11-25) — 대응이 정해질 때까지 S2a 는 띄우지 않는다. 선별 통과 628 개 재채점 (`golden_ultra_user1`) 은 09:21 에 시작했다.

**플러그인 수정 뒤의 S2-dry (11-25 10:06, `student_dry4`, 결정 78)** — 단계 표지 4/4 (main1 2 · sub1 2), 모델 서버의 검증 거부 0, 죽은 모델 포트로의 재시도 9 줄 (요청이 검증을 지나 백엔드 호출까지 감), 모델이 없어 시간 한도로 종료 · 마스킹 (기대한 결과), 잔재 0. 선별 통과 628 개 재채점 (`golden_ultra_user1`, `USER` 전달 상태): 626 재통과 · 2 실패 (PASS_TO_PASS 1/309 · 2/21 — S1c 의 뒤집힘 수준) → 선별 통과 626.

**S2-stub 결과 (11-25 11:19, `student_stub7`, 결정 79 적용 뒤 · 설치된 구성)** — 4/4 보상 1 (js · ts · go · rust, 채점 10 ~ 20 s), 요청 8 건 (적용 4 · 종료 4), 모델 서버 검증 거부 0, 두 노드 · 잔재 0, 75 초. 오전의 GPU 시도 (`student_s2a`) 가 깨진 두 지점 — 도구 결과가 든 둘째 호출 · 결과 조립 — 을 지나갔다. 에이전트 경로로 넣은 정답 패치가 보상 1 을 받았으므로 패치 추출 → 채점 전달의 양성 대조도 된다.

**S2a 결과 (11-25 11:21 ~ 11:40, `student_s2a2`, 과제 8 × 1 · main1 GPU 8)** — 기준 1 ~ 6 전부 통과 (`tools/summarize_swe_student.py`): 결과 8 + 실패 0 = 8 · fleet 8/8 · 프록시 16/16 · 모델 응답이 있는 롤아웃 8/8 (main1 4 · sub1 4) · 명령 · 편집 실행 8/8 · 패치 있는 6 건 중 모델 패치 거부 0 · 잔재 0. S2c (학생 패치 6 건을 정답 패치 모드로 재채점, `regrade_s2a2`): 보상 6/6 일치.
  측정값 (표본 8 이라 참고): 통과 1/8 (php 1) · 마스킹 2 (문맥 초과 1 · 턴 한도 1) · 턴당 8,192 잘림 0 · 모델 호출 수 중앙 88 · 최대 200 · 롤아웃 시간 중앙 369 s · 최대 889 s (한도 1,800 s 안) · 에이전트 최대 메모리 14.1 GB · 채점 11.7 GB (node 계열 과제). 모델이 낸 깨진 도구 호출 표기가 글로 남은 롤아웃 2/8 (메시지 3 개 — 파서 실패가 아니라 모델 출력이 깨진 것).
  fleet 적재는 페이지 캐시가 식어 있으면 19.5 분 (10:11), 데워져 있으면 2 분 (11:21). 노드 메모리의 중단 기준은 anon + shmem 으로 바꿨다 — 전체 (`memory.current`) 는 페이지 캐시로 650 ~ 820 GiB 까지 부풀고 유휴 anon 은 1 ~ 12 GiB 다.

**S2b 결과 (11-25 11:50 ~ 13:39, `student_s2b`, 과제 64 × 8 = 512 롤아웃 · 동시 128 · main1 GPU 8, commit `c0f24a40a`)**

| 판정 기준 | 결과 |
|---|---|
| 1 결과 수 | 512 + 실패 0 = 512, `agent_run_error` 0 |
| 2 붙은 대상 | fleet 8/8 · 프록시 16/16 |
| 3 모델 도달 | 512/512, main1 256 · sub1 256 |
| 4 도구 실행 | 512/512. 모델이 낸 깨진 도구 호출 표기가 글로 남은 롤아웃 81 (15.8%) |
| 5 패치 전달 | 패치 있는 261 건 중 모델 패치 거부 0. `package-lock.json` 12 건은 이미지에 원래 있던 추적 안 된 파일이 "새 파일" 로 실려 그 파일만 건너뛴 것 (나머지는 적용, 그중 2 건 보상 1). 편집했는데 패치가 빈 비마스킹 롤아웃 22 건의 궤적: 스스로 되돌림 (`git checkout` · `stash` 등) 13 · 임시 파일만 5 · 순 변화 없음 1 · 빌드 산출물 디렉토리 편집 1 · 미분류 2 (그중 1 건은 편집 직후의 `git status` 가 변경 없음) — 추출 결함의 흔적은 없다. ② 재채점 `regrade_s2b` (11-25 13:42 ~ 14:00): 학생 패치 261 건을 정답 패치 자리에 넣어 다시 채점 — 261 건 전부 같은 보상 (다름 0 · 결과 없음 0) |
| 6 잔재 | 컨테이너 0 · FUSE 고아 0 · Gym 서버 0 · GPU 반납 |

**미리 정한 판정: 진행.** 8 회의 보상이 갈린 과제 q = **12/64 (18.8%, 95% 구간 11.1 ~ 30.0%)** — 기준 10 개 이상. 한 step (프롬프트 8) 이 전부 무신호일 확률 19%. 과제별 통과 수 분포 {0: 51 · 1: 3 · 2: 2 · 5: 3 · 6: 3 · 7: 1 · 8: 1} — 한 번이라도 푼 과제 13 개.

| 측정값 | 값 |
|---|---|
| 통과율 | 55/512 (10.7%, 95% 구간 8.3 ~ 13.7%). 마스킹 제외 55/276 (19.9%) |
| 종료 사유 | 패치 제출 239 · 패치 없음 34 · **시간 초과 205 (40.0%, 마스킹)** · 턴 한도 16 · 문맥 초과 9 · 메모리 감시 6 (마스킹) · 반복 고리 2 · 기타 1 |
| 마스킹 | 236/512 (46.1%). 과제별 학습에 쓰일 롤아웃 수 분포 {0: 5 · 1: 8 · 2: 5 · 3: 8 · 4: 10 · 5: 4 · 6: 5 · 7: 9 · 8: 10} |
| 시간 | 롤아웃 중앙 1,534 s · p90 1,801 s (한도 1,800 s). 모델 응답 대기가 롤아웃 시간의 77% (중앙). S2a (동시 8) 는 중앙 369 s · 최대 889 s 였다 |
| 처리량 | 512 롤아웃에 6,420 s = 분당 4.79 개. 학습 한 step (롤아웃 64) 이 13.4 분, 1 에폭 977 step 이면 롤아웃만 약 9.1 일 |
| 모델 호출 수 | 끝난 롤아웃 중앙 78 · 시간 초과 롤아웃 중앙 106 · 최대 200 |
| 메모리 | 에이전트 중앙 1.4 GB · p90 12.0 GB · 최대 35.4 GB (`paritytech__parity-common-572` 가 8 회 중 7 회 30 GB 이상) · 채점 최대 17.6 GB · 노드 anon + shmem 최대 164 GiB · tmpfs 최대 13.8 GiB · load 최대 217 |
| 언어별 (참고) | php 14/40 · ts 13/96 · go 11/72 · julia 7/24 · python 5/104 · js 4/88 · rust 1/48 · java · clojure · r · kotlin 0 |

읽는 법: 시간 초과 40% 는 학생의 한계가 아니라 서빙 처리량이다 — 동시 128 을 GPU 8 장이 받으면 모델 응답 대기가 롤아웃의 4분의 3 을 차지한다. 이 비율은 학습의 롤아웃 노드 구성과 같으므로 학습에서도 같은 일이 난다. S3 의 값 (동시 실행 · 에이전트 시간 한도 · 메모리 상한 · 턴 수) 을 이 실측으로 정한다. 턴당 8,192 잘림은 지표가 있는 297 건 중 1 건 (시간 초과 롤아웃 215 건은 턴 지표가 없다).

**롤아웃 병목 진단 (11-25, `problem-diagnosis`) — 독립 검증 판정 "확인 (수정 포함)"**

- **원인**: 병목은 GPU 추론이다. 환경 컨테이너 수 상한은 없다. 엔진이 처리한 토큰 22.4억 중 생성은 1,060만 (211:1) 이고 97.9 ~ 98.4% 가 앞 턴에 이미 보낸 접두사를 다시 읽는 prefill 이다 — prefix caching 을 껐다 (`eval_swe_student.sh` `PREFIX_CACHE=0`, 엔진 로그 적중률 0.0% 4,340/4,340 줄). GPU 한 장의 프롬프트 처리량은 요청 1 개 31K → 4 ~ 7 개 49K → 16 개 이상 60 ~ 65K tok/s 로 포화한다. 호출당 모델 대기는 동시 8 의 1.3 s 에서 동시 128 의 5.3 ~ 10.2 s 로 4 ~ 7 배가 됐다.
- **배제**: 컨테이너 자리 대기 (정상 상태 `ray_queue_time` 중앙 0.0 s) · 도구 실행 (명령당 중앙 0.21 s 로 두 런이 같음) · 프록시와 Gym 모델 서버 (모델 호출 시간의 97% 가 엔진 안) · KV 압박 (평균 11 ~ 19%, 선점 0).
- **부수 요인 2 개 (검증자가 찾음)**: ① 첫 물결 기동 폭주 — 128 개가 한꺼번에 뜰 때 컨테이너 시작부터 첫 모델 요청까지 중앙 485 s (이후 물결 37 s) 이고 이 시간이 에이전트 한도 1,800 s 안에 들어간다. 시간 초과율 첫 물결 61% · 이후 47% · 16%. ② fleet 프록시 (`lb_proxy.py`) 의 세션 키가 system + 첫 user 메시지라 한 과제의 8 회가 GPU 한 장에 몰린다 — 서버별 평균 실행 요청 4.5 ~ 20.8.
- **시간 초과 205 건의 분해**: GPU 경합이 없는 호출 지연 (동시 8 실측) 이면 203 건 이상이 한도 안에 끝난다. 기동 폭주만 없앴다면 약 47 건 (6 ~ 78) 이 줄어 127 ~ 199 건이 남는다.
- **쓰지 않는 문장 (검증에서 반박 · 증거 부족)**: "동시 실행을 낮춰도 분당 롤아웃 수는 그대로" (128 미만은 미측정) · "프롬프트 처리량이 처리량을 정한다" (정상 상태에서만 — 런 전체로는 용량의 72%) · "캐시를 켜면 87 배" (GDN 상태는 1,056 토큰 블록 경계에서만 캐시되어 프롬프트 토큰 기준 36 ~ 43 배가 상한이고 GPU 시간 기준 이득 · 캐시 축출은 미측정).
- **선행 사례**: Pai `SFT_BENCHMARKS.md` §2.5 (2026-09-17, 같은 모델 · SWE 에이전트: 처리 토큰의 99% 가 턴마다 되풀이되는 prefill, 16 세션 2 턴 시험에서 캐시 적중 턴 13.2 → 2.2 s — bf16 상태 · 블록 544 기준) · upstream 제품 레시피는 prefix caching 켬 / R3 예제 레시피는 끔 · MiMo-V2.6 은 R3 와 컨텍스트 캐시를 함께 쓴다 (`FRONTIER.md` "학습 중 문제와 해결").
- **독립 검증**: 블라인드 2 단계 (Opus 5.5 · xhigh). 1 단계는 증상과 원자료만으로 같은 주원인에 도달했고 부수 요인 2 개를 추가했다. 2 단계는 핵심 원인을 확인하고 위 세 문장을 반박했다. 검증 스크립트 `scratchpad/verify_s2b/scripts/` (세션 한정).
- **prefix caching 을 끈 경위**: 학습 엔진의 결정 D2 (`RLVR_READINESS.md` §7, 2026-10-07 — R3 를 켠 upstream 예제 레시피 2 종이 끄고, 켜면 route 행이 드물게 누락된다는 upstream 주석) 를 측정 fleet 에도 맞췄고, GDN 상태 fp32 + prefix caching 은 Pai 정확도 게이트 (`prefix_cache_check.py --block 1056`) 를 아직 돌리지 않았다. D2 는 턴이 짧은 RLVR 런을 두고 정한 값이라 SWE (롤아웃당 80 ~ 100 턴 · 문맥 58K) 의 비용을 따지지 않았다 — 재검토 대상.

**중단하고 보고하는 조건**: S2-dry 가 깨짐 (GPU 를 잡기 전) · S2a 에서 기준 2 ~ 4 가 깨짐 · 노드 메모리가 1,400 GiB 를 넘음 (실행기가 롤아웃을 내린다) · `agent_run_error` 가 10% 를 넘음 · 롤아웃이 시간 한도를 넘김 (실행기가 내린다).

**남는 한계 (재검토가 낮은 위험으로 분류)**: ① PASS_TO_PASS 가 빈 과제 (게이트 통과분의 약 11%) 는 no-op 에서 테스트가 실제로 돌았다는 증거가 FAIL_TO_PASS 0/N 뿐이다 — 설치 · 컴파일이 깨져도 같은 값이 나온다 (새는 경우는 "안 고쳐도 통과" 이면서 no-op 채점이 죽은 과제뿐) ② no-op 의 새 파일은 git 상태를 보는 테스트에 중립이 아니다 — "PASS_TO_PASS 불안정" 으로 빠진 과제 일부는 잡음이 아니라 그 파일에 반응한 것이다 (제외는 보수적) ③ 선별 게이트는 동시 16 에서 잰 것이다 — 동시 128 의 채점은 S2b · S2c 가 본다 ④ S1c 재채점 결과는 뒤집힘 비율을 재는 데만 쓰고 게이트 입력에는 넣지 않는다 (넣으면 표본에 든 100 개만 두 번 시험받는다). 비율이 높으면 전수를 다시 채점한다 ⑤ 학습 데이터 파일을 만드는 도구 (게이트 + 정확한 SIF) 는 아직 없다 — S3 보고 전에 만든다.

**프롬프트 혼합** — 레시피 초안의 `run_with_mixed_prompts: true` 는 이 Gym 에 없는 키다 (저장소 전체에 0 건, 설정 모델이 모르는 키를 조용히 받는다). NVIDIA 공개 레시피 넷 (Ultra `swe_teacher` · `mopd` · Super `stage2_swe2` · Qwen3 가이드) 에는 그대로 남아 있다. 지금 Gym 의 학습 설정 (`swebench_openhands_training.yaml`) 은 CodeAct 프롬프트만 켜 두고 Codex · OpenCode 는 주석이다. 세 가지를 섞으려면 `agent_prompt_overrides` 에 세 항목 + `agent_prompt_override_random: true` 를 쓰고 (없으면 과제 ID 로 고정돼 8 회가 같은 프롬프트) OpenCode · Codex 하니스 셋업을 더 깔아야 한다 (지금은 OpenHands 만 설치). CodeAct 단독 대 세 가지 혼합은 승인 대상이다.

**정정 · 결정 (2026-11-25 16:35)**: 추가 설치는 필요 없다 — 세 가지는 서로 다른 하니스가 아니라 설치된 nv-OpenHands (`5f018005`) 안의 에이전트 클래스다 (`openhands/agenthub/codeact_agent` · `codex_agent` · `opencode_agent`), 프롬프트 틀도 설치된 Gym 에 다 있다 (`prompts/openhands` · `codex` · `opencode`). 따로 까는 `swe_opencode_setup` 은 `agent_framework: opencode` (실제 opencode 포크) 용이고 이 혼합과 무관하다. Gym 의 학습 설정 `swebench_openhands_training.yaml` 은 Codex · OpenCode 항목을 주석으로 두고 CodeAct 만 켜 둔 것이다. 도구 집합: CodeAct `execute_bash` · `str_replace_editor` · `think` · `task_tracker` · `finish` / Codex `shell_command` · `apply_patch` · `read_file` · `list_dir` · `grep_files` · `update_plan` · `finish` / OpenCode `bash` · `edit` · `write` · `read` · `glob` · `grep` · `list_dir` · `apply_patch` · `todo` · `question` · `think` · `finish`. **사용자 결정: 세 가지를 모두 쓴다 (`RL_PLAN.md` 결정 81)** — `agent_prompt_overrides` 세 항목 + `agent_prompt_override_random: true` (없으면 과제 id 로 고정돼 8 회가 같은 틀이 된다, `app.py:3762-3768`). Codex · OpenCode 클래스는 우리 환경에서 돌려 본 적이 없으므로 가짜 모델 고리 시험 → 소량 학생 롤아웃으로 먼저 확인한다.

**이미 나온 수치**: S1a 648/687 (94.3%, §7.6). S1b 진행 중 — 350 개 시점에 보상 1 이 7 건 (전부 php: laravel · symfony · roave · n98 — 안 고쳐도 통과), PASS_TO_PASS 불안정 4 건, 보고서 없음 1 건.

**독립 검토 (11-24, 판정: 선별 게이트 보완 필요 · 학생 측정 신뢰 불가) 의 지적과 조치**

| 지적 | 확인 | 조치 |
|---|---|---|
| Gym 서버가 127.0.0.1 에 묶여 sub1 의 컨테이너가 모델에 닿지 못한다 | `global_config.py` 기본값 False · 정답 패치 실행 로그의 `host` 127.0.0.1 | `use_absolute_ip: true` + 실행기가 묶인 주소를 확인. 재검토: 그러면 Gym 의 `wait_for_servers.sh` (127.0.0.1 고정) 와 클라이언트가 head 서버를 못 찾는다 → 실행기가 노드 주소로 직접 기다리고 클라이언트에 `++use_absolute_ip=true` |
| "명령 실행 시간 > 0" · "턴 ≥ 2" 는 도구가 안 돌아도 통과한다 | 검토자가 OpenHands 코드로 제시 | 기준 4 를 도구 호출 · 결과 항목 수로 바꿈 |
| 과제당 1 회로는 보상이 갈리는 비율을 못 잰다 | 식으로 확인 | 64 × 8 로 바꾸고 q 를 직접 잼, 판정을 q 로 미리 정함 |
| no-op 의 보상 0 은 채점이 죽어도 나온다 | 정답 패치 모드 행의 시간 초과 · 메모리 필드가 574/574 비어 있음 | 게이트 2 — 보고서와 FAIL_TO_PASS 로 판정 |
| 실패 분류가 틀렸다 (메모리 감시 4 건) | 워커 로그: 메모리 감시 4 · 시간 초과 2 | 분류를 워커 로그 · 보고서 기반으로 다시 씀 (§7.6 정정) |
| 다시 돌린 결과가 마지막 것으로 덮인다 | `make_student_input.py` | 게이트 1 — 모든 실행에서 통과, S1c 로 뒤집힘 비율 측정 |
| 동시 16 의 채점이 동시 128 에서도 되는지 모른다 | — | S2b 를 동시 128 로, S2c 재채점 |
| 에이전트 → 패치 → 채점 전달을 아무것도 안 본다 · 보상과 보고서 일치는 순환 | `app.py` 보상 = `resolved` | 기준 5 를 세 가지 검사로 바꿈 |
| `run_with_mixed_prompts` 가 무효 | Gym 저장소 0 건 | 위 "프롬프트 혼합" — 결정 대기 |
| 프록시 :8100 의 신원을 안 본다 | — | 기준 2 — 포트 사전 점검 + 프록시 16 회 |
| 변환 부하 · 서빙 비율이 섞인다 | — | load 기록 · 동시 128 / GPU 8 · 한도를 S2 값으로 바로 정하지 않음 |
| 실행기 견고성 (시간 한도 · EXIT 정리 · 메모리 중단 · FUSE 비교 · ssh 실패 · main1 확인 · 세션 설정 점검) | — | 실행기에 전부 반영. 재검토: 중단 · 시간 한도 · 메모리 중단 때 컨테이너가 남는다 → `swe_kill_containers.py` (명령줄이 아니라 mount namespace 로 찾는다 — starter 가 프로세스 이름을 바꾼다, 실측) |
| (재검토) 워커 노드가 결과 행에 없고 Ray 가 로그 줄을 합친다 | no-op 로그에 합쳐진 줄 184 | `RAY_DEDUP_LOGS=0` · 기준 3 을 전체 비율 + 로그의 sub1 비율로 |
| (재검토) q 가 학습의 이점 계산과 다르게 정의됐다 · 64 개로는 임계가 거칠다 | `advantage_estimator.py` 의 기준선은 마스킹과 무관 | q 를 다시 정의, 임계를 과제 수 (10 · 3) 로, 구간을 함께 적음 |
| (재검토) 채점 로그로는 어느 패치가 거부됐는지 모른다 · S2c 는 추출을 못 본다 | — | 기준 5 ① · ③ 을 고침 |
| SIF 없는 행 탐침은 다른 실패를 본다 · 이름이 비슷한 SIF 에 잘못 붙는다 | `_find_container` 의 glob | 탐침을 뺌 (S3 의 몫) · 게이트 4 |

### 7.8 캐시 켬 학생 측정 (2026-11-25 ~ 26, `RL_PLAN.md` 결정 81 · 사용자 승인 11-25)

**조건**: fleet 에 prefix caching 을 켜고 (`PREFIX_CACHE=1`, 정확도 게이트 `GATES.md` "PC-S" 통과 뒤), 세 틀을 롤아웃마다 무작위로 섞고 (`PROMPT_MIX=all`), 엔진 배정은 Gym 의 세션 해시다 (`runs/alpha_vllm_model_swe_8k_direct.yaml` — 프록시를 거치지 않는다). 나머지는 S2b 와 같다 (과제 64 × 8 · 동시 128 · 200 턴 · 1,800 s · 문맥 131,072).

**틀별 소량 런 (11-25 21:50 ~ 22:34, `student_pc1s_{codeact,codex,opencode}`, 12 과제 × 1 · 동시 12)** — 틀마다 접두사 성질과 세션 고정을 따로 확인한다.

| 틀 | fleet 적중률 H | 그 런의 예측 P_lo ~ P_hi | 시간 초과 | 호출당 모델 응답 (중앙) | 편집 도구 실패 |
|---|---|---|---|---|---|
| CodeAct | 0.9833 | 0.9796 ~ 0.9835 | 0/12 | 1.03 s | 29/63 |
| Codex | 0.9815 | 0.9776 ~ 0.9814 | 0/12 | 0.91 s | 8/10 |
| OpenCode | 0.9839 | 0.9812 ~ 0.9840 | 0/12 | 1.06 s | 8/21 |

동시 12 (엔진당 약 1.5 세션 · 턴 기록이 엔진 프롬프트 토큰의 100.0% 를 덮음) 에서 세 틀 모두 H 가 P_hi 와 0.0002 안에서 같다. 같은 기록으로 계산하면 엔진이 롤아웃마다 한 번 바뀔 때 H 는 약 0.975, 호출마다 바뀔 때 0.90 ~ 0.91 이다. 따라서 프롬프트는 직전 프롬프트와 생성을 그대로 잇고 롤아웃은 엔진을 바꾸지 않았다 — 축출이 없는 부하에서 과제 12 개에 대한 결론이고, 적중 경로의 정확성은 말하지 않는다 (그것은 PC-S 의 몫이다). 통과 0/12 씩은 과제 선택의 결과다 (이 12 과제는 S2b 에서도 0/96).

**본 측정 (11-25 22:53 ~ 23:59, `student_s2b_pc1`, commit `482727728`)** — 512/512 롤아웃 · 실패 0 · 기준 1 ~ 6 통과 · 판정 "진행" (보상이 갈린 과제 10/64, 15.6% [8.7, 26.4]).

| 항목 | 캐시 켬 · 세 틀 · 세션 해시 (`student_s2b_pc1`) | S2b (캐시 끔 · CodeAct · 프록시) |
|---|---|---|
| eval 시간 · 처리량 | 3,856 s · 분당 7.97 롤아웃 | 6,420 s · 분당 4.79 롤아웃 |
| 시간 초과 | 1/512 (0.2%) | 205/512 (40.0%) |
| fleet 적중률 · 판정 | 0.9823 — 통과 (예측 0.9780 ~ 0.9819, 기록이 토큰의 97.6% 를 덮음) | 캐시 끔 |
| 프롬프트 토큰 · 실제 계산한 prefill | 34.0 억 · 0.60 억 (1.8%) | 22.4 억 · 전부 |
| 엔진 | 정지 · 느린 엔진 없음 (만부하 구간 2,835 s) · 대기 0 · 선점 0 · KV 최대 13 ~ 27% | — |
| 롤아웃 시간 중앙 · p90 | 666 s · 1,097 s | 1,534 s · 1,801 s |
| 모델 턴 수 중앙 (모든 롤아웃) | 127 | 93 |
| 통과 | 38/512 (7.4% [5.5, 10.0]) | 55/512 (10.7% [8.3, 13.7]) |
| 끝난 사유 | 최대 턴 76 · 문맥 한도 72 · oom 7 · 반복 정지 3 | 최대 턴 16 · 문맥 한도 9 · oom 6 |

| 틀 (본 측정) | 롤아웃 | 통과 | 편집 도구 실패 | 최대 턴 · 문맥 한도 |
|---|---|---|---|---|
| CodeAct | 177 | 17 (9.6% [6.1, 14.8]) | 376/1,073 (35.0%) | 26 · 26 |
| Codex | 177 | 7 (4.0% [1.9, 7.9]) | 217/239 (90.8%) — `apply_patch` 형식 오류 | 31 · 22 |
| OpenCode | 158 | 14 (8.9% [5.4, 14.3]) | 135/483 (28.0%) | 19 · 24 |

**S2b 와의 비교 (CodeAct 롤아웃만 · 공통 과제 62 개 · 과제마다 같은 가중)**: 통과율 0.079 대 0.097 · 시간 초과율 0.016 대 0.401 · 모델 턴 수 119 대 98 · 롤아웃 시간 706 s 대 1,333 s · 도구 호출 있는 턴 0.997 대 0.996 · 추론 길이 403 자 대 422 자. 호출당 응답 시간 (끝까지 간 롤아웃끼리): 턴 1 ~ 20 2.06 s 대 6.88 s · 21 ~ 50 2.39 대 9.56 · 51 ~ 100 2.46 대 10.55 · 101 ~ 200 2.33 대 8.68.

**읽는 법 (분석 도구 검토에서 정한 규칙)**
- 두 런은 캐시 · 엔진 배정 방식 · 틀 구성이 함께 다르다. 차이를 캐시 하나의 효과로 읽지 않고, "N 배" 라고 쓰지 않는다. 처리량은 fleet 수준만 낸다.
- 통과율로 정확도를 판단하지 않는다. CodeAct 끼리 17/177 대 55/512 는 구간이 겹친다. 시간 압박이 사라져 롤아웃이 시간 초과 대신 최대 턴 · 문맥 한도에서 끝난다 (29%).
- Codex 의 낮은 통과율은 캐시와 무관한 틀 간 차이다 — 전용 편집 도구 `apply_patch` 가 239 번 중 217 번 실패했고 (패치 형식 오류), 모델은 셸로 편집한다.
- 혼합 런의 적중률은 큰 고장만 잡는다. 틀별 적중률은 혼합 런에서 볼 수 없고 (엔진 카운터가 틀을 가리지 못한다) 소량 런이 대신한다. 축출과 엔진 바뀜은 부족분이 나와도 가를 수 없다 (이번에는 부족분이 없다).
- 분석 도구 `tools/analyze_swe_cache_run.py` 는 적대적 검토 4 회 (보완 필요 3 → 신뢰 가능) 를 거쳤다. 남은 한계: 진행 중인 롤아웃이 64 개 미만인 꼬리에서 게이지가 0 으로 굳은 정지는 놓친다.

**Codex `apply_patch` 실패의 원인 (2026-11-26 진단 · 블라인드 독립 검증 "확인")** — 사용자 지시 "실패 원인이 모델인지 서빙 파서인지 CPU 로 먼저 가른다".

| 실패 유형 (239 호출 중 실패 217) | 건수 |
|---|---|
| 봉투: 첫 줄이 `*** Begin Patch` 가 아님 39 · 끝 줄이 `*** End Patch` 가 아님 31 | 70 |
| 본문 줄에 ` ` · `+` · `-` 접두사가 없음 (오류 문구는 "`@@` 기대" 32 · "예상 밖의 줄" 24 로 갈리지만 같은 오류다) | 56 |
| 파일 연산 머리줄 오류 (통합 diff 머리줄 · 오타) | 6 |
| 문맥 줄 · `@@` 기준 줄이 파일에 없음 | 81 |
| 파일 없음 (상대 경로 앞에 저장소 디렉토리를 붙임) | 4 |

- **원인은 모델의 출력이다.** 형식 오류 132 건은 받은 인자 글만으로 도구 구현의 파서에서 같은 오류가 재현된다 (접두사 없는 줄 56 건 중 52 건은 공백 하나를 붙이면 파싱된다). 내용 오류 중 모델이 그 파일을 읽은 적이 있는 64 건은 구현의 느슨한 대조로도 0 건이 맞는다.
- **서빙 · 하니스 경로는 원인이 아니다 — 실패 217 건 중 203 건에 대해 보였다.** 턴별 생성 토큰 수와 파싱된 추론 · 본문 · 호출의 토큰 수가 5 토큰 안에서 맞고 (203/217), 인자는 롤아웃 기록 · OpenHands 응답 · 실행된 패치 · 다시 보낸 이력에서 같다 (228/228). 설치된 `qwen3_xml` 파서는 탭 · NBSP · `<` · `&` · 따옴표 · CRLF · 400 줄 패치를 그대로 돌려준다. 패치 안에 `</parameter>` · `</function>` · `</think>` 가 있으면 깨지지만 이번 자료에는 없었다. 이력이 없는 14 건은 확인하지 못했다. 턴 잘림 없음 (Codex 최대 6,232 토큰 · `length` 종료 0). 구현은 유효한 패치 11/11 을 적용했다.
- **Codex 안에서 패치 실패와 통과율은 같이 움직이지 않는다**: 패치가 한 번이라도 적용된 롤아웃 1/18 · 도구를 썼지만 한 번도 적용되지 않은 롤아웃 4/69 · 도구를 쓰지 않은 롤아웃 2/90. Codex 롤아웃의 84% 는 diff 를 남기고 끝난다 (주로 셸로 쓴다). Codex 대 CodeAct 통과율 차이는 Fisher p = 0.055 다 — 이 표본에서는 유의하지 않고 원인도 보이지 않았다.
- **학습에서 일어나는 일**: 실패한 호출의 오류 문구가 다음 관측이 되고, 그 턴의 토큰은 롤아웃 전체의 advantage (최종 테스트 보상) 만 받는다. `grpo.invalid_tool_call_advantage` 의 벌점은 도구 호출 표기가 본문 · 추론에 남은 턴에만 걸린다 (`nemo_rl/environments/nemo_gym.py` `_detect_invalid_tool_call_and_malformed_thinking`) — 파싱은 됐지만 패치가 틀린 호출은 벌점 대상이 아니다. 레시피의 `penalize_invalid_tool_call` 키는 이 fork 의 `nemo_rl/` 어디서도 읽지 않는다. 형식에 대한 직접 신호가 없으므로 "RL 이 형식을 가르친다" 는 이 자료로 뒷받침되지 않는다.
- **프롬프트 쪽 결함 둘 (실패의 약 2%)**: 도구 설명 "경로는 상대, 절대 금지" 가 무엇 기준인지 말하지 않는다 (nv-OpenHands `codex_agent/tools/apply_patch.py:64`, 구현은 절대 경로도 받는다) · 시스템 프롬프트의 `{"command":["apply_patch",…]}` 예시가 `input` 인자와 어긋난다 (Gym `prompts/codex/system_prompt.j2:132`).
- **선행 사례**: 공개 E2E SWE 레시피 셋 (Qwen3-30B-A3B · Super · Ultra) 모두 `run_with_mixed_prompts: true` 를 적지만, 여기 고정된 Gym (`3045a793`) 은 그 키를 구현하지 않아 어떤 틀이 섞이는지 확인하지 못했다.

**학습 설정으로 가져갈 관찰 (결정은 `STATUS.md` 열린 결정)**: 동시 128 에서 엔진 여유가 크다 (대기 0 · KV 최대 27%) · 시간 한도 1,800 s 는 p90 1,097 s 보다 넉넉하다 · 최대 턴 200 과 문맥 131,072 에 걸려 끝나는 롤아웃이 29% 다 · Codex 틀의 편집 도구는 지금 모델에서 거의 쓰이지 못한다 · 메모리 한도 초과 7 건.

### 7.9 S3 계획 전 조사 — 막힌 곳과 알게 된 것 (2026-11-26, 읽기 · CPU 만)

**R3 (라우터 재생) 는 지금 핀에서 `swe_agents` 다중 턴 롤아웃과 함께 쓸 수 없다 (코드로 확인, 실행으로는 미확인).**

| 고리 | 근거 |
|---|---|
| R3 가 켜지면 NeMo-RL 은 Gym 결과의 학습 가능한 항목마다 `routed_experts` 를 요구하고, 없으면 예외를 던진다 | `nemo_rl/environments/nemo_gym.py` 728 ~ 735 (`require_routed_experts`), `nemo_rl/algorithms/grpo.py` 818 ~ 828 |
| 항목의 경로는 그 턴 요청의 **전체 시퀀스** 행이어야 한다 (프롬프트 + 생성 이상) | `nemo_gym.py` 720 ~ 727 · 743 ~ 790 |
| 에이전트 하니스 (컨테이너 안의 nv-OpenHands `5f018005`) 는 턴마다 토큰 필드 셋만 남긴다 — `routed_experts` 는 0 건 | `openhands/agenthub/nemo_gym_client.py` 149 ~ 156 · `openhands/core/message.py` 70 ~ 72 |
| 고정된 Gym (`3045a793`) 의 `swe_agents` 는 그 메시지와 마지막 응답으로 궤적을 다시 짓고 경로를 싣지 않는다 | `responses_api_agents/swe_agents/app.py` 3043 ~ 3065 |
| 앞선 다중 턴 R3 런 (workplace `w1r` 등) 은 Gym 서버 안에서 도는 에이전트라 모델 응답 항목을 그대로 넘겼다 | Gym `responses_api_models/vllm_model/app.py` 943 · 993 |

- **그대로 실어 나르는 것도 답이 아니다.** 턴마다 전체 시퀀스의 경로를 실으면 호출당 평균 10.6 MB (프롬프트 평균 55,063 토큰 × 24 층 × 8), `student_s2b_pc1` 규모 (512 롤아웃 · 61,835 호출) 에 약 654 GB 다. **정정 (11-26, §7.10)**: expert id 는 int16 이라 (192 experts) 행당 384 B — 호출당 21.1 MB, 같은 규모에 약 1.3 TB 다. 턴마다 새 토큰의 행만 싣는 델타 방식이면 롤아웃당 최대 50 MB 다.
- **선행 사례**: 공개 E2E SWE 레시피 셋 (Qwen3-30B-A3B · Super · Ultra) 은 모두 prefix caching 켬 · **R3 없음** 이다. upstream 의 R3 + Gym 레시피는 단일 스텝 피벗뿐이다. upstream `main` (`61bc275da`, 우리 fork 보다 512 커밋 앞 · 분기점 2026-08-12) 에는 다중 턴용 **token capture** 경로가 있다 — `token_capture.defer_routed_experts_to_policy`, SingleController Gym 경로, 경로를 델타로 맞춘다 (`fe1d97ce2` #4610 · `681ca0463` #4345 — 둘 다 우리 브랜치에 없다) [**정정 (§7.10)**: 이 둘은 뒤에 얹힌 수정이고 기반 커밋은 `7036e5d16` #3837 이다], Gym `main` (`363af5e3e`) 의 `swe_agents` 가 token capture 와 `routed_experts` 를 다룬다. upstream 가이드 (`docs/guides/router-replay.md`, main) 는 R3 레시피에서 prefix caching 을 끈다고 적는다.
- R3 없이 잰 KL 은 0.0015 다 (R1 2 × 2 · R1-L B, 단일 턴 수학 프롬프트) — 기준 0.001 (결정 80) 을 넘는다. SWE 다중 턴에서의 값은 미측정이다.

**같이 알게 된 것 (S3 계획에 들어간다)**

| 항목 | 사실 | 근거 |
|---|---|---|
| 가중치 갱신과 캐시 | 초안 설정 (async · `in_flight_weight_updates: true` · `recompute_kv_cache_after_weight_updates: false` · 궤적 나이 1) 에서는 갱신 때 KV · GDN 상태 · 경로 슬롯을 아무것도 비우지 않는다. 궤적은 갱신을 넘어 이어지고, 첫 step 만 on-policy 다. `true` 면 진행 중 요청을 선점해 새 가중치로 다시 prefill 한다. 공개 SWE 레시피 셋은 모두 `false` 다 | `algorithms/async_utils/trajectory_collector.py` 876 ~ 943 · 337 ~ 353 · `vllm_worker_async.py` 1648 ~ 1660 |
| 빈 롤아웃 | 생성 토큰이 없는 결과가 하나라도 오면 그 배치 (64 롤아웃) 의 흐름이 끊긴다. 설정으로 끌 수 없고 upstream main 도 같다. 수집기가 배치를 3 번 더 돌리고 네 번째 연속 실패는 치명이다. 두 학생 측정 (1,024 롤아웃) 에서는 0 건이었다 | `nemo_rl/environments/nemo_gym.py` 817 ~ 841 · `trajectory_collector.py` 1547 ~ 1616 |
| 마스킹과 KL | 마스킹된 롤아웃은 KL 에서 빠진다. 턴 상한을 짧게 잡은 스모크는 전부 마스킹돼 KL 0 으로 "통과" 할 수 있다 → `num_mask_sample_filtered` · 유효 시퀀스 수를 함께 본다 | `algorithms/utils.py` 290 · Gym `app.py` 3852 ~ 3859 |
| 실행기 | `launch.sh` 에는 SWE 기동 전 점검 (raylet 실행 위치 · 홈 여유 · Gym 패치 · 남은 컨테이너 · 메모리 감시) 이 없다. `uv_venv_dir` 가 가리키는 곳에 `swe_agents` 서버 venv 가 없다 | `runs/launch.sh` 29 ~ 57 · `runs/eval_swe_student.sh` 84 ~ 120 |
| 죽은 키 | `run_with_mixed_prompts` · `penalize_invalid_tool_call` · `penalize_malformed_thinking` · `invalid_tool_call_strategy` · `max_rollout_turns` (Gym 경로) · `swe_agents.dataset_path` (읽지 않지만 값은 필요) | 조사 보고 (11-26) |
| 토큰 연속성 | 하니스가 직전 턴의 토큰을 다시 보내고 엔진이 그 토큰을 그대로 이어 붙인다 (`required_prefix_token_ids`). 학습 엔진에서 `swe_agents` 로 이 경로가 돈 적은 없다 (S2 는 토큰 정보를 끄고 돌았다) | `vllm_worker_async.py` 526 ~ 690 · `nemo_gym.py` 691 ~ 698 |
| 호스트 메모리 (계산) | 8 × 8 × 131,072: 배치 객체 3.49 GB · CP 8 랭크 27.9 GB · 워커 정점 55.8 GB. 컨테이너 쓰기 공간은 최악 128 × 16 GiB | `KNOWN_ISSUES.md` 10-20 의 식 |

### 7.10 S3 — R3 없이 학습 경로를 돌려 KL 을 쟀다 (2026-11-26, 사용자 승인 "R3 없이 … KL 측정을 바로 진행해")

**결과: R3 없이 SWE 다중 턴의 Generation KL 은 0.0016 이다 (on-policy step 0.00156 · 세 step 합산 0.00162, 95% 구간 0.0015 ~ 0.0017). 기준 0.001 을 넘고, 단일 턴에서 R3 없이 잰 0.0015 와 같은 수준이다.** 수치 · 분해 · 한계 · 검토 기록의 정본은 `GATES.md` "S3" 다. 여기에는 경과와 함께 알게 된 것을 적는다.

| 런 | 구성 | 결과 |
|---|---|---|
| `s3a` (09:40 ~ 10:02) | 과제 2 × 2 회 · 1 step | rc 0 — 처음 도는 경로 통과 (토큰 연속성 · 빈 롤아웃 예외 0), KL 0.0018 (4 시퀀스) |
| `s3b` (10:05 ~ 11:13) | 과제 12 · 4 × 8 · 3 step · 실제 상한 | rc 0 — KL 0.001555 / 0.001637 / 0.001658, 유효 81/96 시퀀스, 배치 재시도 0, 평균 보상 0.31 → 0.41 |

**함께 알게 된 것**

| 항목 | 사실 | 근거 |
|---|---|---|
| 가중치 갱신과 캐시 | 갱신 때 엔진이 prefix cache 를 비우지 않는다. `Successfully reset prefix cache` 는 기동 직후 첫 refit 에만 8 대 모두 찍히고, step 뒤 갱신에는 없다 (§7.9 의 코드 읽기와 같다). 낡은 캐시의 영향은 이 런에서 미측정 — lr 이 워밍업 구간이라 갱신이 거의 없다 | `results/alpha/teacher_swe_e2e/s3b.log` |
| 형식이 틀린 도구 호출 | 하니스가 그 응답을 버리고 같은 프롬프트로 다시 부른다 — 학습에 들어간 96 롤아웃의 8,498 호출 중 97 (1.1%) (처음 적은 58/8,348 은 분석 도구의 짝짓기가 롤아웃 하나를 빼서 적게 센 값이다 — 진단과 정정은 §7.11). 예: `bash` 호출에 필수 인자 `command` 가 없는 응답. 버린 응답은 학습 궤적에 없어 **`invalid_tool_call_advantage: -5` 벌점이 걸리지 않는다** (궤적에 남은 것 기준 Invalid tool call rate 는 step 1 에서 3/3,230). KL 에는 영향 없다 | 덤프의 턴 ↔ 하니스 호출 대조 (`analyze_s3.py`), nv-OpenHands `agenthub/codeact_agent/function_calling.py` 328 ~ 334 (`ValidationFailureAction`) |
| 덤프 키의 방향 | `env.should_log_nemo_gym_responses: true` 가 step 덤프 (`train_data_step*.jsonl`) 를 **건너뛴다**. 초안 레시피의 `false` 는 주석과 반대로 덤프를 쓴다 (32 행에 step 당 약 160 MB). 스모크에는 필요했지만 본 런 (64 행 × 977 step) 에서는 약 300 GB 가 쌓인다 → 본 런 전에 `true` 로 바꾼다 (레시피 변경, 승인 대상) | `nemo_rl/algorithms/grpo.py` `_should_log_nemo_gym_responses` |
| 전부 마스킹된 step | `seq_logprob_error_in_loss=false` 에서는 런이 죽지 않고 KL 0 이 기록된다 (`true` 면 예외로 죽는다) → KL 은 유효 시퀀스 수와 함께 읽는다 | `megatron_policy_worker.py` 735 부근 (독립 검토) |
| 마스킹 비율 | 96 롤아웃 중 15 (16%) 가 상한에 걸려 손실에서 빠졌다 (턴 상한 10 · 문맥 상한 5). 학생 측정 (`student_s2b_pc1`) 의 29% 보다 낮다 — 과제 12 개를 고른 표본이라 비교하지 않는다 | `s3b/analysis.json` |
| 마지막 배치 | 수집기는 마지막 갱신 뒤에도 배치를 하나 더 띄운다. 런이 끝날 때 컨테이너 세션 68 개가 돌고 있었고 래퍼가 정리했다 (남은 것 0) | `gates/swe_e2e_s3/s3b/run.log` |
| Gym 서버의 노드 | Ray 가 기동마다 정한다 (s3a main1 · s3b sub1). `swe_agents` 서버 venv 는 NFS (`gates/gym_smoke/gym_venvs`) 에 첫 기동 때 만들어졌다 | 같은 로그 |
| 벽시계 | 4 × 8 · 3 step 에 68 분 — 첫 배치 32 롤아웃 수집 20 분 47 초, 학습 step 281 ~ 423 s (첫 step 은 고정 비용 포함), step 3 은 롤아웃 대기 포함 1,376 s | 같은 로그 |

**R3 를 SWE 에 쓰려면 — upstream token capture 경로의 범위 (읽기 전용 조사, 11-26)**

| | upstream token capture 를 올린다 | 우리가 전달 경로를 만든다 |
|---|---|---|
| 무엇 | 엔진 워커가 토큰 · logprob · 경로를 TransferQueue 에 두고 Gym 은 좌표만 넘긴다. 에이전트 하니스는 아무것도 나르지 않는다 | alpha 모델 서버 플러그인이 턴마다 새 토큰의 경로 행을 노드 저장소에 두고, NeMo-RL 이 궤적을 조립할 때 가져온다 |
| 규모 | NeMo-RL 본체 약 40 파일 · +15,000 줄 (기반 `7036e5d16` #3837 는 58 파일 · +9,007 줄, 단독 cherry-pick 은 28 경로 충돌) + Gym 커밋 3 개 약 +6,100 줄 | 파일 2 개 · 약 200 ~ 300 줄 (`nemo_rl/environments/nemo_gym.py` 700 ~ 792 · `gym_plugins/…/alpha_vllm_model/app.py`) |
| 전제 | 학습 런타임을 SingleController 로 바꿔야 한다 — token capture 는 지금 쓰는 async GRPO 경로 (`grpo.py` · `trajectory_collector.py`) 가 읽지 않는다 | 런타임 그대로 |
| 지금 핀 | vLLM 0.25.1 · nv-OpenHands 는 가능. Gym 0.6.0 은 그대로는 안 되고 커밋 3 개 (#2794 · #2996 · #2872) 를 패치로 얹어야 한다 (글자 충돌 0, 동작은 미확인). `uv lock` 미확인 | 가능 |
| 잃는 것 | SingleController 는 검증 step (`val_period`) · `reward_shaping` · dynamic sampling 을 거부하고, token capture 와 메시지 단위 advantage 덮어쓰기 (`invalid_tool_call_advantage`) 를 함께 쓰지 못한다 | 없음 (upstream 의 대장 · 다이제스트 검증 · 고아 정리는 없다) |
| 다시 돌릴 게이트 | E1 ~ E3 · R1 · R3 · G2 · G3 · G5 · G7 · PC-S | R3 · G3 · 새 CPU 조립 게이트 (음성 대조 포함) → 이 절의 KL 측정 |
| 남는 문제 (둘 다) | prefix caching 과 R3 — upstream 은 R3 를 켤 때 캐시를 끄라고 권한다 (경로 행이 드물게 빠짐, `docs/guides/router-replay.md` 34 ~ 37). 우리 PC-S 는 순차 요청에서 경로가 비트 동일함을 확인했고 동시 요청은 보지 않았다. 가중치 갱신을 넘는 궤적은 표시만 되고 처리되지 않는다 | 같음 |

- 공개 E2E SWE 레시피 셋 (Ultra `swe_teacher.yaml` · Super `stage2_swe2.yaml` · Qwen3 `swe2`) 은 upstream `main` 에서도 **R3 · token capture 없이** 기존 async GRPO 로 학습한다. token capture 와 R3 를 함께 쓰는 레시피는 시험용 하나 (문맥 8,192 · 6 턴) 다.
- 우리 SWE 레시피의 off-policy 대응은 Ultra `swe_teacher.yaml` 과 같다: 시퀀스 필터 2 · IS 보정 (TIS 5 / 0.2) · `force_on_policy_ratio` · `freeze_moe_router: true` · 궤적 나이 1 · 갱신 뒤 KV 재계산 안 함.
- 확인하지 못한 것: Gym 패치가 0.6.0 에서 도는지, SingleController 가 AlphaBridge 와 도는지, OpenHands 의 이력 압축이 token capture 의 계보를 끊는지.

### 7.11 형식이 틀린 도구 호출이 벌점 없이 사라진다 — 진단 (2026-11-26, 독립 검증 "확인")

사용자 지시 2026-11-26 "하네스가 벌점이 안 걸리게 하는데 이건 수정해야겠어". 수정 방법은 보고 뒤 결정한다 (STATUS "열린 사용자 결정").

**원인 (한 문장)**: 모델이 형식이 틀린 도구 호출을 생성하면 nv-OpenHands (`5f018005`) 가 그것을 `ValidationFailureAction` · `FunctionCallNotExistsAction` 으로 바꾸는데, 대화 조립 코드가 이 두 종류를 assistant 메시지로 만들지 않아 실패한 응답과 오류 문구가 다음 프롬프트와 학습 궤적에서 모두 빠진다. 모델은 오류를 보지 못한 채 토큰까지 같은 프롬프트로 다시 호출된다. 궤적에 남더라도 NeMo-RL 의 탐지기는 이런 호출을 잡지 못한다.

| 고리 | 근거 |
|---|---|
| 틀린 것은 모델이 생성한 원문이다 | s3b 의 롤아웃 160 개 중 이력이 있는 127 개에서 거부 123 건 (55 롤아웃). 이력에 남은 `generation_token_ids` 를 복원: 필수 인자의 태그를 빼고 값만 씀 47 (그중 43 은 한 롤아웃의 반복) · 태그가 깨짐 46 (`<parameter Gupta>` · `<parameter permutation_description>`) · 없는 도구 이름 18 (`shellcommand` · `execute-bash` · `evaluate_bash`) · 그 밖의 스키마 오류 · 문맥 한도에서 잘림 2 (P 130,928 + 144 = 131,072). 호출된 함수의 블록 안에 필수 인자가 제대로 있던 경우는 누락 100 건 중 0. 틀 Codex 70 · CodeAct 39 · OpenCode 14. 샘플링 temperature 1.0 · top_p 1.0 |
| 검증 실패가 예외가 아니라 action 이 된다 | `openhands/agenthub/codeact_agent/function_calling.py` 328 ~ 345 (codex 204 ~ 230 · opencode 278 ~ 295), 모델 응답 전체가 `tool_call_metadata` 로 붙는다 |
| runtime 이 오류 문구를 observation 으로 돌려준다 | `runtime/base.py` 988 ~ 999 · 453 ~ 454 |
| 대화 조립이 둘 다 버린다 | `memory/conversation_memory.py`: `_process_action` 254 ~ 285 의 목록에 두 action 이 없어 421 의 `return []` 로 빠진다 → 대기 중 assistant 메시지가 안 생긴다. observation 은 602 ~ 607 에서 문구가 만들어지지만 801 ~ 811 에서 `tool` 응답으로 보관되고, 내보내는 곳 (171 ~ 182) 은 대기 중 assistant 메시지가 있어야 돈다 |
| 다음 프롬프트가 같다 | 다시 부른 쌍 120 개의 `prompt_token_ids` 가 토큰 단위로 같다. 거부 123 건 = 사라지고 다시 불린 호출 122 + 롤아웃의 마지막 호출 1 (궤적에 남았으나 마스킹된 표본, 표지 없음) |
| 탐지기는 남아도 못 잡는다 | `nemo_rl/environments/nemo_gym.py` 169 ~ 231 은 마지막 출력 항목이 본문 · reasoning 텍스트이고 거기에 `<tool_call>` 류 문자열이 있을 때만 표시한다. 거부된 호출은 파서가 `function_call` 항목으로 만든 것이라 텍스트가 없다. 세 step 에서 표시된 14 건 (3 · 5 · 6) 은 함수 태그가 깨져 파싱되지 않은 종류다 (`<function:shell_command>`) |

- **배제한 대안**: 서빙 파서가 잘 쓴 인자를 잃음 (원문 복원으로 배제 — 다만 2 건은 모델의 깨진 첫 블록 뒤에 온 멀쩡한 블록을 파서가 합치거나 살리지 못했다) · 전송 재시도 (호출마다 응답 id 와 이력 이벤트가 따로 있고 로그에 재시도 0 줄) · 이력 압축 (프롬프트 길이가 그대로) · Gym 의 궤적 조립 (그 전에 이미 프롬프트에 없다).
- **결과**: ① 학습에 들어간 96 롤아웃에서 호출 8,498 중 97 이 사라졌다 (그 밖에 문맥 초과 때 Gym 이 남기는 0 토큰 기록 5) ② 오류를 못 본 모델이 같은 실수를 되풀이한다 — 같은 프롬프트에서 29 · 7 · 5 번 연속 재호출. 29 번 반복한 롤아웃은 재호출 44 번에 모델 시간 354 초 중 122 초 · 생성 23,362 토큰을 썼다 ③ 다시 부른 호출도 200 턴 상한과 1,800 초에 들어간다 — 턴 상한에 걸린 18 롤아웃 중 11 개에 재호출이 있었다 ④ `invalid_tool_call_advantage: -5` 는 이 호출들에 한 번도 걸리지 않는다.
- **선행 사례**: 하니스 fork 이력 — `d6bd12edd` (2026-02-02) 가 action 을, `cdb815028` (03-22) 가 observation 문구와 테스트를 넣었으나 `_process_action` 목록은 그대로다. 코드 주석 ("Send error as a user message while preserving the assistant message") 과 fork 의 테스트 (`tests/unit/memory/test_tool_error_conversation_flow.py`) 는 assistant 메시지와 오류 응답이 함께 남기를 기대한다 — 의도가 아니라 미완성 구현으로 읽힌다 (작성자 확인은 아님). Gym main 이 고정한 `54d2a95e` 도 같다. upstream 이슈 · PR 검색 (`repo:NVIDIA-NeMo/Gym ValidationFailure` 0 건 · `repo:NVIDIA-NeMo/RL invalid_tool_call_advantage` 는 token capture 와의 병용 PR #5022 뿐) 에 이 문제는 없다. 프론티어는 `FRONTIER.md` "보상 · 벌점 · 길이 제어" — RL 처리를 공개한 계열 (Ultra · Qwen3-Coder-Next · MiMo-V2.6 · GLM-4.5) 은 모두 벌을 주고, MiMo 공개 하니스는 오류를 모델에 돌려주고 턴을 남겨 표시한다.
- **독립 검증**: 새 검증자 (증상과 원자료만 받은 1 단계) 가 같은 원인에 도달 → 2 단계 판정 **확인**. 반론 요지 (반영함) — 사라진 호출 수는 58 이 아니라 97 (짝짓기 상한이 롤아웃 하나를 뺐다) · "모두 같은 프롬프트로 다시 불렸다" 는 0 토큰 기록 5 건에는 틀리다 · "학습기에 도달하지 않는다" 는 롤아웃의 마지막 호출에는 틀리다 (1 건) · 144 토큰에서 끝난 2 건은 조기 종료가 아니라 문맥 한도 · 반증 시험 "원문에 멀쩡한 `<parameter=command>` 가 0 건" 은 틀렸고 (다른 블록에 20 건) 호출된 함수의 블록 안으로 좁혀야 한다 (0/100).
- **궤적에 남기면 학습기가 보는 것**: 토큰 ID 가 붙은 `function_call` 항목과 오류 문구 (`Validation failure for <도구>: …` · `Tool <이름> is not registered. …`) 의 `function_call_output` 항목. 메시지 단위로 "거부됨" 을 실어 보낼 기존 필드는 없다 — 표지는 NeMo-RL 본체가 붙여야 한다.
- **부수 발견 (별개 문제, 미결)**: 서빙 파서가 모델 출력을 바꾸는 턴이 8,766 중 117 — 인자 값 앞뒤 공백 제거 37 (공백 · 탭 손실 18, 편집 인자의 선행 공백 손실 1) · `</tool_call>` 뒤 텍스트 누락 34 · 중복 `</think>` 제거 30 · `</think>` 없는 턴 6. 학습 토큰은 원문 그대로이고, 하니스가 실행 · 표시한 내용만 다르다.

**수정 (결정 88, 2026-11-26) 과 CPU 검증** — 턴을 궤적에 남기고 기존 벌점 (턴 전체 advantage −5) 을 건다.

| 부분 | 내용 | 검증 |
|---|---|---|
| 하니스 패치 | `harness_patches/0001-…patch` — `_process_action` 이 두 action 도 assistant 메시지로 만든다. 캐시된 체크아웃의 작업 트리에 적용한다 (`tools/apply_harness_patches.sh` apply · check · revert) | 단위: 패치 전 누락 → 패치 뒤 거부된 턴 (토큰 ID 포함) + 바로 뒤 `tool` 오류 응답. fork 테스트는 전후 같은 13 개 실패. s3b 실제 이력 재생 (독립 검토): 패치 전 0/123 → 패치 뒤 123/123, 짝 없는 tool 메시지 0 |
| 학습기 표지 | 본체 패치 `15ac221d9` · `5f7d9105a` — `env.nemo_gym.invalid_tool_call_output_patterns` (정규식, 도구 출력의 맨 앞 + `name` 그룹이 호출한 도구 이름과 같을 때) 에 맞는 `function_call_output` 이 직전 assistant 메시지의 호출에 답한 것이면 그 메시지를 `is_invalid_tool_call` 로 표시한다. 롤아웃마다 `num_harness_rejected_tool_calls` 를 남겨 본문 탐지와 따로 센다 | 단위 테스트 31 (음성 대조: 같은 문구가 출력 중간에 있음 · 다른 도구 이름 · 앞선 메시지의 호출 · 패턴 없음) · 기존 조립 테스트 8. 실제 이력 재생 → Gym 변환 → 조립 코드: 표지 109/109 · 잘못 붙은 표지 0 |
| 레시피 · 검사기 | 레시피에 정규식 둘. 검사기는 swe_agents 레시피가 가리키는 하니스 트리 (`<cache_dir>/swe_agents/swe_openhands_setup/<slug>/<commit>/OpenHands`) 를 계산해 패치 적용 · HEAD · venv 를 본다 | 음성 대조: 키 없음 · 틀린 정규식 · 빈 문자열에 맞는 정규식 · 패치 되돌림 · 다른 cache_dir · 다른 커밋 · cache_dir 없음 → 전부 ERROR |
| 가짜 모델 시험 (CPU, 실제 컨테이너 · 두 노드, `$NRL_ROOT/gates/swe_e2e_s3/run_stub_reject.sh` · `summarize_stub_reject.py`) | 과제 12 개 × 틀 셋, 첫 응답이 거부될 호출 (필수 인자 누락 6 · 없는 도구 이름 6) | 패치 적용: 틀마다 오류 문구가 다음 요청에 보임 12/12 · 같은 프롬프트 재호출 0/12 · Gym 결과에 거부 응답 항목 12/12 · 학습기 표지 12/12 · 보상 1 12/12. 패치 되돌림 (음성 대조): 0/12 · 12/12 · 0/12 · 0/12 |

- **독립 검토 (수정 · 시험 설계)**: 1 차 "수정 필요" — 수정 자체는 실제 이력 재생으로 확인, 구멍은 전달 경로 (기동 전 점검이 강제되지 않음 · 점검이 아무 트리나 통과) 와 시험 판정 (적용 실패가 묻힘 · 틀 × 종류 조합 미보장 · 런 중 신호가 두 표지를 합침) → 반영 → 2 차 "GPU 스모크 검증기 2 건만 고치면 신뢰 가능" (다시 빠진 거부 턴을 못 잡음 · 벌점 줄 집계가 사고 태그를 빼먹음) → 반영 (`verify_rejected_flags.py`, 수정 전 런 `s3b` 에서 다시 빠진 97 건을 잡는다).
- **남는 한계**: 롤아웃의 마지막 호출이 거부되면 Gym 이 출력 없이 붙여 표지가 없다 (117 중 1) · 문맥 상한에서 잘린 생성도 인자 누락으로 거부된다 (123 중 2 — 지금은 그 롤아웃이 마스킹돼서만 무효) · vLLM 은 이력의 도구 호출 인자를 `json.loads` 한다 — 지금 파서 (qwen3_xml) 는 늘 JSON 을 내지만 파서를 바꾸면 다시 본다 · 토큰 연속성 · 실제 벌점 구간 · 벌점 비중은 GPU 스모크가 본다 · 공용 실행기 (`launch.sh` · `eval_swe_student.sh`) 에는 하니스 점검이 아직 없다 (레시피 검사기와 스모크 래퍼가 본다).

**GPU 스모크 `s3c` (2026-11-26 14:07 ~ 14:47, 4 × 8 · 1 step, commit `80cd0b4cb`, 사용자 승인 "GPU 스모크 진행해") — 판정 통과** (`$NRL_ROOT/gates/swe_e2e_s3/verify_rejected_flags.py` · `s3c/verify.txt` · `analysis.txt`)

| 확인 | 결과 |
|---|---|
| 거부된 턴에 벌점 | 덤프에 들어온 거부된 턴 61 개 전부 구간 전체가 −5 (누락 0) |
| 다시 빠진 거부 턴 | 0 (덤프에 들어온 롤아웃 32 개). 하니스가 부르고 궤적에 넣지 않은 호출 0/3,844 (수정 전 `s3b` 는 97) |
| 건수 대조 | 하니스 거부 수 — TensorBoard 61 = 덤프 61. 벌점 구간 69 = 로그의 벌점 줄 69 (거부 61 + 본문 탐지 8, 사고 태그 0) |
| 토큰 연속성 | 거부된 턴 뒤에 모델 호출이 이어진 사례 61 · 연속성 · 빈 롤아웃 예외 0 · 배치 재시도 0 |
| Generation KL | 0.001770 (유효 23 시퀀스 · 353,504 토큰, TensorBoard 와 0.001%). 거부된 턴의 mult_prob_error 중앙 1.019 · 최대 1.38 (그 밖의 턴 1.018 · 2.19) |
| advantage 그룹 (결정 89) | 과제마다 첫 프롬프트 1 종 — 그룹 8 유지. 값이 leave-one-out 식과 같다 (성공 1 개인 과제: +1.0 / −0.378) |

- **벌점의 무게 (유효 행 실측)**: 벌점 턴 43 개 · 7,317 토큰 = 생성 토큰의 2.07%, advantage 절대량의 **22.4%**. 같은 턴에 다른 형태를 적용하면 — 고정 −2: 10.3% · −1.5: 8.0% · −1: 5.4% · MiMo 부호별 (음수 궤적 × 2, 양수 0): 43 턴 중 28 턴에만 걸리고 (14 턴은 8 개가 전부 실패해 advantage 0 인 과제 · 1 턴은 성공 궤적) 2.5%, 세기 0.756. 레시피의 `reference_policy_kl_penalty` 는 0 이고 엔트로피 항은 없다.
- **관찰**: 호출의 1.6% 가 거부됐다 (61/3,844). 32 롤아웃 중 17 개에 거부된 턴이 있고, 한 롤아웃은 27 번 (연속 19 번) 거부됐다 — 오류 문구가 보여도 같은 실수를 되풀이하는 경우가 있다. 턴 상한에 걸려 마스킹된 롤아웃은 9/32 (28%, `s3b` 는 16% — 과제 4 개라 참고만). 마스킹된 행의 벌점 18 턴은 기울기에 들어가지 않는다.

### 7.12 세 틀 혼합이 advantage 그룹을 쪼갠다 (2026-11-26, 덤프 재현으로 확인)

- NeMo-RL 은 baseline 과 표준편차를 **첫 프롬프트의 토큰이 같은 롤아웃끼리** 계산한다 (`algorithms/utils.py` `calculate_baseline_and_std_per_prompt` 의 `torch.unique(prompts)`, 키는 `grpo.py` `prompt_ids_for_adv`). leave-one-out 이라 자기를 뺀 나머지로 baseline 과 표준편차를 내고, 나머지의 보상이 전부 같으면 정규화하지 않는다.
- 세 틀은 시스템 프롬프트가 달라, `agent_prompt_override_random: true` (롤아웃마다 무작위) 에서는 한 과제의 8 개가 틀별로 갈린다. 이 규칙으로 다시 계산한 값이 `s3b` 덤프의 advantage 와 유효 시퀀스 81/81 에서 일치했다.

| 항목 | `s3b` |
|---|---|
| baseline 그룹 | 과제 12 개 → 33 그룹 (크기 1 이 6 · 2 가 6 · 3 이 10 · 4 가 8 · 5 가 2 · 6 이 1) |
| advantage 가 정확히 0 인 유효 시퀀스 | 29/81 (혼자 남은 롤아웃 · 보상이 전부 같은 그룹) |
| advantage 범위 · 절대값 평균 | −1.5 ~ +1.0 · 0.53 |
| 같은 보상을 과제 단위 (8 개) 로 묶으면 (추정) | 0 인 시퀀스 13/81 · 절대값 평균 0.72 |

- **제공되는 대안**: Gym 의 틀 선택은 `agent_prompt_override_random: false` 면 과제 ID 를 시드로 고른다 (`swe_agents/app.py` 3762 ~ 3768) — 한 과제의 롤아웃이 모두 같은 틀이고 과제 사이에서는 세 틀이 섞인다. 그룹 8 이 유지된다.
- 정규화된 advantage 의 크기 (이 코드의 식): 그룹에 성공이 하나뿐이면 성공 궤적은 +1.0 (정규화 안 됨), 가장 큰 값은 성공이 둘일 때 — G = 8 에서 2.27 · 16 에서 3.61 · 32 에서 5.39. Ultra (G = 16) 에서는 −5 가 더 크고 Super (G = 32) 에서는 아니다.
- Ultra 레시피에도 프롬프트 혼합 키 (`run_with_mixed_prompts: true`) 가 있다. 그쪽에서 그룹이 같은 식으로 갈리는지는 확인하지 못했다.

### 7.13 미완 궤적 벌점 — 한도에 걸린 롤아웃을 마스킹하지 않고 보상 −1 을 준다 (2026-11-27, `RL_PLAN.md` 결정 95)

**왜**: 턴 한도 · 문맥 한도에 걸린 롤아웃은 손실에서 빠진다 (Gym `swe_agents/app.py` 가 `mask_sample` 을 붙이고 학습기가 손실 가중치를 0 으로 만든다, `FRONTIER.md`). 그래서 턴이 길어져 한도에 걸리는 쪽으로 모델이 변해도 손실은 보지 못한다. `s4` 의 step 49 부터 실제로 그랬다.

| 구간 | 턴당 토큰 중앙 · 평균 | 롤아웃당 생성 토큰 | 한도 · 시간 초과로 마스킹 |
|---|---|---|---|
| `s4` step 1 ~ 8 | 111 · 212 | 28.5k | 33% |
| `s4` step 37 ~ 48 | 116 · 213 | 27.9k | 33 ~ 36% |
| `s4` step 49 ~ 54 | 132 · 234 | 31.5k | 37 ~ 40% |
| `s4` step 55 ~ 60 | 142 · 249 | 33.8k | 42 ~ 44% |
| `s5` step 1 ~ 2 (`s4` step 60 가중치, `s4` step 1 ~ 8 과 같은 과제) | 148 · 262 | 35.1k · 36.3k | 44% · 50% |

턴 수는 그대로 (약 130) 이고 평범한 턴 하나하나가 길어졌다. 턴당 상한 8,192 에 걸린 턴은 0.04% 이하다. 길어진 원인은 진단하지 않았다.

**영향 계산 (`s5` step 1 · 2 의 512 롤아웃에 학습기의 advantage 식을 적용 — 그룹 안 leave-one-out, 표준편차 > 0 이면 정규화)**

| | 지금 (마스킹) | 미완 보상 −1 | −0.2 | −0.05 |
|---|---|---|---|---|
| advantage 총량 | 66 만 | 1,460 만 | 1,419 만 | 1,410 만 |
| 과제 보상의 몫 | 100% | 6% | 6% | 6% |
| advantage 가 0 이 아닌 토큰 | 8% | 92% | 92% | 92% |
| 한도에 걸린 롤아웃 · 끝까지 간 실패의 평균 advantage | — · −0.06 | −1.04 · +0.54 | −1.00 · +0.52 | −0.99 · +0.51 |

- 벌점의 크기는 결과를 바꾸지 않는다 (그룹 정규화). 형식 벌점 −5 의 몫은 0.5% 가 된다.
- 한도에 걸리는 비율이 줄면 신호도 같이 줄어든다: 41% → 1,460 만 (과제 몫 6%) · 21% → 1,200 만 (7%) · 10% → 722 만 (11%) · 4% → 350 만 (23%) · 1.5% → 168 만 (47%) · 0% → 78 만 (100%). 다시 길어지면 다시 켜진다.
- 끝까지 간 실패 260 개: 85% 가 패치를 냈고 턴 중앙 103 (20 턴 미만 2.3%). 성공 12 개는 턴 중앙 48, 한도에 걸린 192 개는 196. 이 학생의 성공은 사전 측정에서도 전부 132 턴 이하였다 (§7.8).

**구현**: Gym 패치 `gym_patches/0003-swe-agents-unfinished-trajectory-penalty.patch` — 설정 `unfinished_trajectory_reward` (기본 None = 원래 동작). 턴 한도 · 문맥 한도로 끝나고 (`agent_error_kind`), 풀지 못했고, 시간 초과 · 메모리 초과 · 채점 시간 초과가 아니면 `unfinished_penalized` 를 참으로 하고 마스킹하지 않으며 보상을 그 값으로 낸다. 그 밖의 표지 대상 (시간 초과 등, 한도에 걸렸지만 푼 것) 은 그대로 마스킹한다. NeMo-RL 본체는 고치지 않았다. 지표 `train/swe_agents_train/unfinished_penalized/mean`.

**검증 순서와 결과**: `GATES.md` "S6".

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
| 2 | SWE 데이터 | **결정됨 (사용자 2026-11-24, `RL_PLAN.md` 결정 76)**: Ultra 블렌드 `swe.jsonl` 7,816 행. `swe2` 는 쓰지 않는다. 남은 결정: Ultra 이미지 7,785 개 변환 착수 · 풀 상한 (시간당 80 유지면 노드당 약 49 시간, 크기는 표본 5 개 평균으로 약 5.3 TiB) | 지금 |
| 3 | 학생 통과율 사전 측정 (결정 40 의 5) | **정정 (2026-11-23, §7.5)**: 기존 gpu06 경로로는 못 잰다. ① gpu06 에 Gym `swe_agents` + Apptainer 환경을 만들어 표본 측정 (학습과 같은 하니스, 셋업 필요 · main1 GPU 8장) ② 인프라 적용 뒤 학습 세션에서 전수 측정 (그사이 CPU 로 하니스 셋업 · 레시피 초안) ③ 둘 다 — 사용자 결정 대기 | 지금 |
| 4 | `--pid` 처리 방식 · 쓰기 공간 | **결정됨 (2026-11-24, 결정 75)**: bindproc + `sessiondir max size` 16,384 MB, 두 노드 적용 | — |
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
