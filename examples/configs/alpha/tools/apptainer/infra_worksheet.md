# 인프라 작업 안내 — Apptainer 실행을 위한 세 설정 (2026-11-20)


## 0. 먼저 확인 (호스트에서, main1 · sub1 각각)

    docker ps --format '{{.Names}}' | grep 6bf8ef67            # main1 세션 컨테이너 (kernel id 6bf8ef67-8349-4a80-b8c5-559cadd7f4f0)
    docker inspect <컨테이너> --format '{{json .HostConfig.SecurityOpt}}' | cut -c1-120
    docker inspect <컨테이너> --format '{{.AppArmorProfile}} {{json .HostConfig.Devices}}'
    <에이전트의 python> -c 'import ai.backend.runner as r, os; print(os.path.dirname(r.__file__))'
    apparmor_parser --version; cat /proc/$(pidof dockerd)/attr/current; ls -l /dev/fuse

- SecurityOpt 가 `["seccomp={...JSON...}"]` 이면 seccomp 는 Backend.AI 에이전트가 넘긴다 → 1번은 에이전트의 JSON 파일을 고친다.
- SecurityOpt 가 비어 있으면 dockerd 기본 프로파일이 적용된 것이다 → 1번은 dockerd 의 `--seccomp-profile` (daemon.json `seccomp-profile`) 쪽이다.

## 1. seccomp — 파일 `<site-packages>/ai/backend/runner/default-seccomp.json`

`"syscalls"` 배열 끝에 규칙 둘을 추가한다. 기존 규칙은 건드리지 않는다.

    python3 - <<'EOF'
    import json, shutil
    p = "<위에서 확인한 경로>/default-seccomp.json"
    shutil.copy(p, p + ".orig")
    d = json.load(open(p))
    d["syscalls"] += [
      {"names": ["setns", "mount", "umount2", "pivot_root"], "action": "SCMP_ACT_ALLOW"},
      {"names": ["unshare", "clone"], "action": "SCMP_ACT_ALLOW",
       "args": [{"index": 0, "value": 1073741824, "valueTwo": 0, "op": "SCMP_CMP_MASKED_EQ"}]},
    ]
    json.dump(d, open(p, "w"), indent=2)
    EOF

- 에이전트는 컨테이너를 만들 때마다 이 파일을 읽는다 (`agent/docker/agent.py` `_apply_seccomp_profile`). 새로 만드는 컨테이너부터 적용된다.
- 이 파일은 그 에이전트가 만드는 모든 세션에 적용된다 (노드 단위).
- 패키지 안의 파일이라 에이전트 업그레이드 때 덮어써진다 (upstream 은 `scripts/update-default-seccomp.sh` 로 moby 기본값을 그대로 받아 온다).
- compute 플러그인의 `get_additional_allowed_syscalls()` 는 이름만 받는다. 둘째 규칙의 인자 조건을 담지 못하므로 쓰지 않는다.

## 2. AppArmor — 파일 `/etc/apparmor.d/docker-apptainer` (호스트)

    install -m 644 docker-apptainer /etc/apparmor.d/docker-apptainer
    apparmor_parser -r /etc/apparmor.d/docker-apptainer
    grep docker-apptainer /sys/kernel/security/apparmor/profiles     # "docker-apptainer (enforce)"

- `/etc/apparmor.d/` 의 프로파일은 부팅 때 apparmor 서비스가 적재한다.
- parser 가 4.x 이상이고 user namespace 를 중재하면 `docker-apptainer.userns` 판을 쓴다.
- dockerd 라벨이 unconfined 가 아니면 `signal (receive) peer=<그 라벨>,` 한 줄을 더한다.

세션 컨테이너에 붙이려면 `HostConfig.SecurityOpt` 에 `apparmor=docker-apptainer` 가 들어가야 한다.
공개 코드에는 이 설정 항목이 없다 (`agent.toml` 에 없고 저장소 전체에 apparmor 문자열이 없음). 넣는 길은 3번과 같다.

## 3. `/dev/fuse` — `HostConfig.Devices`

    {"PathOnHost": "/dev/fuse", "PathInContainer": "/dev/fuse", "CgroupPermissions": "rwm"}

호스트에 `/dev/fuse` 가 없으면 `modprobe fuse`.

## 2 · 3 을 컨테이너 생성 옵션에 넣는 길 (공개 코드 기준)

| 길 | 위치 | 범위 |
|---|---|---|
| compute 플러그인 | `AbstractComputePlugin.generate_docker_args()` (entry point 그룹 `backendai_accelerator_v21`) 가 돌려주는 dict 가 Docker create 옵션에 병합된다. 목록은 확장된다 (`agent/utils.py` `update_nested_dict`) | 그 플러그인의 자원 슬롯을 받은 세션만 |
| 에이전트 코드 수정 | `agent/docker/agent.py` — `/dev/infiniband` 를 `Devices` 에 넣는 기존 블록과 같은 형태로 `/dev/fuse` 를, `_apply_seccomp_profile` 의 `SecurityOpt` 목록에 `apparmor=docker-apptainer` 를 추가 | 노드의 모든 세션. 업그레이드 때 사라짐 |

돌려줄 dict:

    {"HostConfig": {"SecurityOpt": ["apparmor=docker-apptainer"],
                    "Devices": [{"PathOnHost": "/dev/fuse", "PathInContainer": "/dev/fuse", "CgroupPermissions": "rwm"}]}}

`apparmor=docker-apptainer` 를 넣었는데 그 호스트에 프로파일이 적재돼 있지 않으면 컨테이너 생성이 실패한다. 2번의 적재가 먼저다.

## 적용 뒤 확인

    docker inspect <새 컨테이너> --format '{{.AppArmorProfile}} {{json .HostConfig.Devices}}'    # docker-apptainer, /dev/fuse rwm
    # 세션 안에서: accept/acceptance.sh

되돌리기: `default-seccomp.json.orig` 복원 · `apparmor_parser -R /etc/apparmor.d/docker-apptainer` · 플러그인/코드 수정 제거.
