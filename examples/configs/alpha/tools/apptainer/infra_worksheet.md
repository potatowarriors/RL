# 인프라 작업 요청서 — 학습 세션에서 Apptainer 를 실행하기 위한 호스트 설정 3건

2026-11-24 개정. 2026-11-20 에 드린 작업 안내문을 대신합니다. 요청 내용은 같고, 역할 구분 · 배경 · 예상 출력을 보탰습니다.

## 1. 요약

- **요청**: 학습 세션 컨테이너에 보안 설정 3건을 적용해 주세요. ① seccomp 규칙 2개 추가 ② AppArmor 프로파일 교체 ③ `/dev/fuse` 장치 전달.
- **대상**: 세션의 두 노드 `main1` · `sub1` 이 도는 호스트 전부.
- **호스트에 Apptainer 를 설치할 필요는 없습니다.** Apptainer 는 요청자가 세션 컨테이너 안에 직접 설치합니다.
- 세 설정은 컨테이너를 만들 때 정해집니다. 그래서 적용 뒤 요청자가 세션을 새로 만듭니다. 떠 있는 세션은 바뀌지 않습니다.

## 2. 왜 필요한가

요청자는 세션 안에서 Apptainer (컨테이너 실행기) 를 돌려야 합니다. 권한 없는 방식 (setuid 없음) 의 Apptainer 는 아래 세 단계를 거치는데, 지금 세션에서는 단계마다 막혀 있습니다.

| 단계 | Apptainer 가 하는 일 | 지금 막는 것 | 세션에서 보이는 증상 | 푸는 설정 |
|---|---|---|---|---|
| 1 | user namespace 를 만든다 | seccomp 프로파일 | `unshare: Operation not permitted` (root 도 같음) | 작업 1 |
| 2 | 그 안에서 mount 한다 | AppArmor `docker-default` 의 `deny mount` | (1 단계에서 먼저 막혀 아직 닿지 않음) | 작업 2 |
| 3 | FUSE 로 이미지 파일을 읽는다 | `/dev/fuse` 장치 없음 | `/dev/fuse: No such file or directory` | 작업 3 |

세션 안에서는 root 로도 셋 다 바꿀 수 없습니다. 컨테이너에 `CAP_SYS_ADMIN` 이 없고, seccomp 는 안에서 더 조일 수만 있기 때문입니다.

## 3. 역할 구분

| 할 일 | 담당 | 장소 |
|---|---|---|
| 작업 1 — seccomp 규칙 2개 추가 | **인프라** | 호스트 (Backend.AI 에이전트 패키지 파일) |
| 작업 2 — AppArmor 프로파일 적재, 세션 컨테이너에 지정 | **인프라** | 호스트 `/etc/apparmor.d/`, 컨테이너 생성 옵션 |
| 작업 3 — `/dev/fuse` 전달 | **인프라** | 컨테이너 생성 옵션 |
| 세션 재생성 | 요청자 | Backend.AI 세션 |
| 적용 확인 (수용 시험) | 요청자 | 세션 안 |
| Apptainer 설치 · 설정 | 요청자 | 세션 안 (`sudo` 로 deb 설치) |

**요청하지 않는 것**: 호스트의 Apptainer 설치 · `--privileged` · capability 추가 · docker.sock 전달 · seccomp 나 AppArmor 의 `unconfined` · `/proc` 마스킹 해제 · cgroup 위임 · 새 network namespace 허용.

## 4. 작업 전 확인 (호스트에서, 두 노드 각각)

`<컨테이너>` 는 세션 컨테이너의 ID 입니다. kernel id 앞자리로 찾습니다 (main1 = `6bf8ef67`, sub1 = `dfdc05b2`. 세션 이름과 전체 ID 는 요청자가 따로 전달합니다).

```bash
docker ps --format '{{.ID}} {{.Names}}' | grep -E '6bf8ef67|dfdc05b2'
# 이름으로 안 잡히면 라벨에서 찾는다
docker ps -q | xargs docker inspect --format '{{.Id}} {{.Name}} {{json .Config.Labels}}' | grep -E '6bf8ef67|dfdc05b2' | cut -c1-120
```

| # | 명령 | 알려 주는 것 | 예상 출력 |
|---|---|---|---|
| 가 | `docker inspect <컨테이너> --format '{{json .HostConfig.SecurityOpt}}' \| cut -c1-120` | seccomp 를 누가 넘기는가 | `["seccomp={...` 로 시작 → Backend.AI 에이전트가 넘긴다 (작업 1 그대로) |
| 나 | `docker inspect <컨테이너> --format '{{.AppArmorProfile}} {{json .HostConfig.Devices}}'` | 지금의 AppArmor 와 장치 | `docker-default` 이고 장치 목록에 `/dev/fuse` 없음 |
| 다 | Backend.AI 에이전트를 실행하는 python 을 찾는다. 설치 방식마다 다르다 (예: `ps -eo args \| grep -i 'backend.ai' \| grep -i agent`, 또는 에이전트 서비스 정의) | 에이전트 패키지가 깔린 python | 그 python 으로 다음 줄을 실행 |
| 라 | `<그 python> -c 'import ai.backend.runner as r, os; print(os.path.dirname(r.__file__))'` | seccomp 파일이 있는 디렉토리 | `…/site-packages/ai/backend/runner` |
| 마 | `apparmor_parser --version; cat /proc/$(pidof dockerd)/attr/current; ls -l /dev/fuse` | AppArmor 판 · dockerd 라벨 · 호스트의 FUSE 장치 | 작업 2 · 3 의 갈림에 쓴다 |

가 의 출력이 비어 있으면 (`[]` · `null`) dockerd 기본 프로파일이 쓰이는 것입니다. 그 경우 작업 1 의 대상이 에이전트 파일이 아니라 dockerd 의 `seccomp-profile` 설정이 되므로, 진행하기 전에 요청자에게 알려 주세요.

## 5. 작업 1 — seccomp 규칙 2개 추가

**대상 파일**: 4-라 에서 나온 디렉토리의 `default-seccomp.json`.

**하는 일**: `"syscalls"` 배열 끝에 아래 두 규칙을 더합니다. 기존 규칙은 건드리지 않습니다.

```json
{"names": ["setns", "mount", "umount2", "pivot_root"], "action": "SCMP_ACT_ALLOW"},
{"names": ["unshare", "clone"], "action": "SCMP_ACT_ALLOW",
 "args": [{"index": 0, "value": 1073741824, "valueTwo": 0, "op": "SCMP_CMP_MASKED_EQ"}]}
```

- 첫째 규칙: `setns` · `mount` · `umount2` · `pivot_root` 를 허용합니다.
- 둘째 규칙: `unshare` · `clone` 을 **플래그에 `CLONE_NEWNET` 이 없을 때만** 허용합니다. `1073741824` 는 `0x40000000` (`CLONE_NEWNET`) 이고, 조건은 "첫째 인자와 이 값을 AND 한 결과가 0" 입니다. 즉 새 network namespace 는 계속 막힙니다.

**적용 명령** (원본을 `.orig` 로 남깁니다):

```bash
python3 - <<'EOF'
import json, shutil
p = "<4-라 의 디렉토리>/default-seccomp.json"
shutil.copy(p, p + ".orig")
d = json.load(open(p))
d["syscalls"] += [
  {"names": ["setns", "mount", "umount2", "pivot_root"], "action": "SCMP_ACT_ALLOW"},
  {"names": ["unshare", "clone"], "action": "SCMP_ACT_ALLOW",
   "args": [{"index": 0, "value": 1073741824, "valueTwo": 0, "op": "SCMP_CMP_MASKED_EQ"}]},
]
json.dump(d, open(p, "w"), indent=2)
EOF
```

**알아 둘 점**
- 에이전트는 컨테이너를 만들 때마다 이 파일을 읽습니다. 새로 만드는 컨테이너부터 적용됩니다.
- **그 노드의 에이전트가 만드는 모든 세션에 적용됩니다.** 요청자 세션만 고를 수 없습니다.
- 패키지 안의 파일이라 에이전트를 업그레이드하면 원래대로 돌아갑니다. 업그레이드 뒤 다시 적용해야 합니다.
- 플러그인의 "추가 허용 syscall" 기능은 이름만 받습니다. 둘째 규칙의 인자 조건을 담지 못하므로 쓰지 않습니다.
- 컨테이너에는 여전히 `CAP_SYS_ADMIN` 이 없습니다. 그래서 `mount` 등은 새로 만든 user namespace 안에서만 동작합니다.

## 6. 작업 2 — AppArmor 프로파일

### 6.1 호스트에 프로파일 적재

부록 A 의 내용을 `/etc/apparmor.d/docker-apptainer` 로 저장하고 적재합니다.

```bash
install -m 644 docker-apptainer /etc/apparmor.d/docker-apptainer
apparmor_parser -r /etc/apparmor.d/docker-apptainer
grep docker-apptainer /sys/kernel/security/apparmor/profiles     # 예상: docker-apptainer (enforce)
```

- 이 프로파일은 Docker 의 `docker-default` 템플릿 (moby `profiles/apparmor/template.go`) 에서 네 군데만 바꾼 것입니다: 이름, `deny mount,` 삭제, `mount,` · `pivot_root,` 추가, dockerd 신호 줄 삭제 (아래 셋째 항목). `/proc` · `/sys` 쓰기 금지 같은 나머지 규칙은 그대로입니다.
- 프로파일을 적재하는 것만으로는 어떤 컨테이너도 바뀌지 않습니다. 6.2 에서 지정한 컨테이너에만 쓰입니다.
- 4-마 의 `apparmor_parser` 가 4.x 이상이고 user namespace 를 중재하면, `pivot_root,` 다음 줄에 `userns,` 를 한 줄 더합니다.
- 4-마 의 dockerd 라벨이 `unconfined` 가 아니면, `signal (receive) peer=<그 라벨>,` 을 한 줄 더합니다.
- `/etc/apparmor.d/` 에 두면 부팅 때 apparmor 서비스가 다시 적재합니다.

### 6.2 세션 컨테이너에 지정

컨테이너의 `HostConfig.SecurityOpt` 에 `apparmor=docker-apptainer` 가 들어가야 합니다. 넣는 방법은 7 절입니다.

**순서 주의**: 6.1 의 적재가 먼저입니다. 적재되지 않은 프로파일을 지정하면 컨테이너 생성이 실패합니다.

## 7. 작업 3 — `/dev/fuse` 전달, 그리고 작업 2 · 3 을 넣는 방법

컨테이너의 `HostConfig.Devices` 에 아래 항목이 들어가야 합니다. 호스트에 `/dev/fuse` 가 없으면 먼저 `modprobe fuse` 를 합니다.

```json
{"PathOnHost": "/dev/fuse", "PathInContainer": "/dev/fuse", "CgroupPermissions": "rwm"}
```

작업 2 의 지정과 작업 3 은 둘 다 "컨테이너 생성 옵션" 입니다. Backend.AI 공개 코드에서 찾은 길은 둘입니다.

| 길 | 위치 | 적용 범위 |
|---|---|---|
| compute 플러그인 | `AbstractComputePlugin.generate_docker_args()` 가 돌려주는 dict 가 Docker 생성 옵션에 병합된다 | 그 플러그인의 자원 슬롯을 받은 세션만 |
| 에이전트 코드 수정 | `agent/docker/agent.py` — `/dev/infiniband` 를 `Devices` 에 넣는 기존 블록과 같은 형태로 `/dev/fuse` 를, seccomp 를 넣는 `SecurityOpt` 목록에 `apparmor=docker-apptainer` 를 추가 | 노드의 모든 세션. 업그레이드 때 사라짐 |

어느 길이든 더해질 내용은 같습니다.

```json
{"HostConfig": {"SecurityOpt": ["apparmor=docker-apptainer"],
                "Devices": [{"PathOnHost": "/dev/fuse", "PathInContainer": "/dev/fuse", "CgroupPermissions": "rwm"}]}}
```

운영 환경에 더 맞는 방법이 있으면 그 방법으로 해 주세요. 8 절의 확인 결과만 같으면 됩니다.

## 8. 권장 순서와 확인

1. 4 절의 확인을 합니다.
2. 6.1 로 AppArmor 프로파일을 적재합니다.
3. 5 절로 seccomp 파일을 고칩니다.
4. 7 절의 방법으로 `apparmor=docker-apptainer` 와 `/dev/fuse` 를 넣습니다.
5. 시험용 세션을 하나 만들어 아래를 확인합니다.
6. 요청자에게 알려 주세요. 요청자가 수용 시험을 돌려 결과를 회신하고, 본 세션을 다시 만듭니다.

**호스트에서**

```bash
docker inspect <새 컨테이너> --format '{{.AppArmorProfile}} {{json .HostConfig.Devices}}'
# 예상: docker-apptainer 이고 장치 목록에 /dev/fuse (rwm)
```

**새 세션 안에서** (요청자의 수용 시험이 같은 것을 더 자세히 봅니다)

| 명령 | 예상 | 확인하는 설정 |
|---|---|---|
| `unshare -Ur true && echo OK` | `OK` | 작업 1 |
| `unshare -Urm sh -c 'mount -t tmpfs none /mnt && echo OK'` | `OK` | 작업 1 + 작업 2 |
| `cat /proc/self/attr/current` | `docker-apptainer (enforce)` | 작업 2 |
| `ls -l /dev/fuse` | 문자 장치 `10, 229` | 작업 3 |
| `unshare -Urn true` | **실패해야 정상** (`Operation not permitted`) | 새 network namespace 가 계속 막혀 있는지 |

## 9. 되돌리기

- seccomp: `default-seccomp.json.orig` 를 원래 이름으로 복원합니다.
- AppArmor: 7 절의 지정을 먼저 빼고, `apparmor_parser -R /etc/apparmor.d/docker-apptainer` 뒤 파일을 지웁니다.
- `/dev/fuse`: 7 절의 플러그인 또는 코드 수정을 제거합니다.
- 되돌린 내용도 새로 만드는 컨테이너부터 적용됩니다.

## 10. 보안 영향

| 구분 | 내용 |
|---|---|
| 새로 열리는 것 | 세션 안에서 user namespace 생성 · 그 안에서의 mount 와 `pivot_root` · FUSE 마운트 |
| 그대로인 것 | 컨테이너의 capability 집합 · `/proc` 마스킹 · privileged 아님 · docker.sock 없음 · **새 network namespace 금지** |
| 범위 | 작업 1 은 노드의 모든 세션, 작업 2 · 3 은 7 절에서 고른 길에 따라 다름 |

새 network namespace 를 막아 둔 이유: 호스트 커널 `6.5.0-15` 는 CVE-2024-1086 (nf_tables) 수정판 `6.5.0-26` 이전이고, 이 취약점은 user namespace 와 새 network namespace 를 함께 써서 닿습니다. 요청자의 작업에는 새 network namespace 가 필요 없습니다.

user namespace 를 여는 것 자체가 커널 공격면을 넓힌다는 점은 남습니다. 커널을 수정판 이상으로 올릴 수 있으면 함께 권합니다.

## 11. 확인하지 못한 것

- 5 절과 7 절의 적용 위치는 **Backend.AI 공개 저장소 (main) 의 코드를 읽어 찾은 것**입니다. 운영 중인 버전에서 파일 이름 · 위치가 다를 수 있습니다. 다르면 알려 주세요.
- 요청 사양은 별도 Docker 호스트에서 시험했습니다. Backend.AI 기본 seccomp 에 위 규칙 2개와 `/dev/fuse` 를 더하면 Apptainer 실행이 통과했고, 규칙을 빼면 실패했으며, 새 network namespace 는 막혔습니다.
- 그 시험 호스트에는 AppArmor 가 적용돼 있지 않았습니다. **부록 A 의 프로파일은 실제 집행 상태에서 시험하지 못했습니다.** 시험용 세션에서 처음 확인됩니다. 8 절의 둘째 명령이 실패하면 AppArmor 쪽입니다.

## 부록 A — `/etc/apparmor.d/docker-apptainer`

```
#include <tunables/global>

profile docker-apptainer flags=(attach_disconnected,mediate_deleted) {
  #include <abstractions/base>

  network,
  capability,
  file,
  umount,
  mount,
  pivot_root,
  # Host (privileged) processes may send signals to container processes.
  signal (receive) peer=unconfined,
  # runc may send signals to container processes (for "docker stop").
  signal (receive) peer=runc,
  # crun may send signals to container processes (for "docker stop" when used with crun OCI runtime).
  signal (receive) peer=crun,
  # Container processes may send signals amongst themselves.
  signal (send,receive) peer=docker-apptainer,

  deny @{PROC}/* w,   # deny write for all files directly in /proc (not in a subdir)
  # deny write to files not in /proc/<number>/** or /proc/sys/**
  deny @{PROC}/{[^1-9],[^1-9][^0-9],[^1-9s][^0-9y][^0-9s],[^1-9][^0-9][^0-9][^0-9/]*}/** w,
  deny @{PROC}/sys/[^k]** w,  # deny /proc/sys except /proc/sys/k* (effectively /proc/sys/kernel)
  deny @{PROC}/sys/kernel/{?,??,[^s][^h][^m]**} w,  # deny everything except shm* in /proc/sys/kernel/
  deny @{PROC}/sysrq-trigger rwklx,
  deny @{PROC}/kcore rwklx,

  deny /sys/[^f]*/** wklx,
  deny /sys/f[^s]*/** wklx,
  deny /sys/fs/[^c]*/** wklx,
  deny /sys/fs/c[^g]*/** wklx,
  deny /sys/fs/cg[^r]*/** wklx,
  deny /sys/firmware/** rwklx,
  deny /sys/devices/virtual/powercap/** rwklx,
  deny /sys/kernel/security/** rwklx,

  # suppress ptrace denials when using 'docker ps' or using 'ps' inside a container
  ptrace (trace,read,tracedby,readby) peer=docker-apptainer,
}
```

같은 내용이 저장소의 `docker-apptainer` 파일에 있고, `userns,` 를 더한 판은 `docker-apptainer.userns` 입니다. seccomp 규칙만 담은 파일은 `seccomp_rules.json` 입니다.
