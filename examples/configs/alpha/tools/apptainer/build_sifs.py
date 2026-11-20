#!/usr/bin/env python3
"""SWE E2E 블렌드가 참조하는 태스크 이미지를 Apptainer `.sif` 로 변환한다.

제공 도구 `examples/nemo_gym/download_swe_images.py` 와 같은 방식 (`apptainer build <tmp>.sif docker://<image>`,
이미지마다 캐시 · 임시 디렉토리 분리) 이다. 그 도구를 그대로 쓰지 않는 이유는 셋이다.
  1. HF 데이터셋 전체를 훑는다 (블렌드가 쓰는 행만 받을 수 없다).
  2. 레지스트리 풀 한도를 지키지 않고 이어받기 기록이 없다.
  3. SWE-rebench-V2 (Ultra `swe.jsonl` 의 7,610 행) 를 다루지 않는다.

파일 이름은 Gym `swe_agents` 의 `_find_container` 가 찾는 이름을 따른다 (레시피 `container_formatter` 와 짝):
  SWE-rebench-V2  <out>/swerebench/{instance_id}.sif
  R2E-Gym         <out>/r2egym_{image tag, ':' -> '_'}.sif          (제공 도구와 같은 이름)
  SWE-Gym         <out>/swegym_sweb.eval.x86_64.{instance_id, '__' -> '_s_', 소문자}.sif

user namespace 가 막힌 세션에서도 돈다 (변환은 namespace 를 쓰지 않는다 — 2026-11-20 실측).

변환한 파일은 마운트 없이 내용을 확인한 뒤에만 제 이름으로 옮긴다 (`check_sif`): squashfs 목록이 끝까지 읽히고
Gym 이 작업 디렉토리로 쓰는 저장소 경로 (`/testbed`, SWE-rebench-V2 는 `/{repo 이름}`) 가 들어 있어야 한다.
데이터 블록 손상은 이 확인으로 잡지 못한다 — 실행 검증 (시험 세션) 의 몫이다.

Docker Hub 풀 한도는 출구 IP 마다 따로다. 노드를 나눠 돌릴 때는 `--shard K/N` 으로 목록을 겹치지 않게 가른다
(파일 이름의 CRC 로 나눠 목록 순서와 무관하다). 기록 파일은 샤드마다 따로 쓴다.

사용:
  build_sifs.py --blend <swe.jsonl> [--blend <swe2.jsonl>] --out <sif_dir> --dry-run
  build_sifs.py --blend ... --out <sif_dir> --apptainer <bin> [--shard K/N] [--pulls-per-hour 80] [--workers 4] [--limit N]
  build_sifs.py --blend ... --out <sif_dir> --apptainer <bin> --verify        # 이미 있는 파일의 내용만 확인
  build_sifs.py --apptainer <bin> --self-test <정상 .sif>                      # 분할 · 내용 확인의 음성 대조
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import zlib


def tasks_from_blend(path: str):
    """블렌드 한 줄마다 (instance_id, dataset_name, image, sif 상대 경로, 저장소 경로) 를 낸다.

    저장소 경로는 Gym `swe_agents` 의 `_resolve_opencode_workspace_path` 와 같은 규칙이다.
    """
    with open(path) as f:
        for line in f:
            md = json.loads(line)["responses_create_params"]["metadata"]
            d = md.get("instance_dict")
            d = json.loads(d) if isinstance(d, str) else (d or {})
            ds = md.get("dataset_name") or d.get("dataset_name") or ""
            iid = md["instance_id"]
            repo_dir = "/testbed"
            if "SWE-rebench" in ds:
                image, rel = d["image_name"], f"swerebench/{iid}.sif"
                repo = str(d.get("repo") or md.get("repo") or "")
                repo_dir = "/" + (repo.split("/", 1)[1] if "/" in repo else repo)
            elif "R2E-Gym" in ds:
                image = d["docker_image"]
                rel = "r2egym_" + image.split("/")[-1].replace(":", "_") + ".sif"
            elif "SWE-Gym" in ds:
                name = "sweb.eval.x86_64." + iid.replace("__", "_s_").lower()
                image, rel = f"xingyaoww/{name}", f"swegym_{name}.sif"
            else:
                yield iid, ds, None, None, None
                continue
            yield iid, ds, image.removeprefix("docker.io/"), rel, repo_dir


def gym_lookup_stem(iid: str, ds: str) -> str | None:
    """Gym `_find_container` 가 R2E-Gym 행에서 만드는 이름 (검산용)."""
    if "R2E-Gym" in ds:
        return re.sub(r"[^_]+__([^-]+)-", lambda m: m.group(1).lower() + "_final_", iid)
    return None


def in_shard(rel: str, shard: tuple[int, int]) -> bool:
    """파일 이름의 CRC 로 샤드를 정한다 (목록 순서 · 다른 블렌드 추가와 무관)."""
    k, n = shard
    return zlib.crc32(rel.encode()) % n == k


def parse_shard(text: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+)/(\d+)", text)
    if not m or not 0 <= int(m.group(1)) < int(m.group(2)):
        raise argparse.ArgumentTypeError("K/N 꼴이어야 한다 (0 <= K < N)")
    return int(m.group(1)), int(m.group(2))


def pull_quota() -> dict | None:
    """Docker Hub 익명 풀 한도와 잔여 수. HEAD 요청이라 한도를 쓰지 않는다. 못 읽으면 None."""
    try:
        with urllib.request.urlopen(
            "https://auth.docker.io/token?service=registry.docker.io"
            "&scope=repository:ratelimitpreview/test:pull",
            timeout=20,
        ) as r:
            token = json.load(r)["token"]
        req = urllib.request.Request(
            "https://registry-1.docker.io/v2/ratelimitpreview/test/manifests/latest",
            method="HEAD",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            limit, remaining = (
                r.headers["ratelimit-limit"],
                r.headers["ratelimit-remaining"],
            )
        return {
            "limit": int(limit.split(";")[0]),
            "remaining": int(remaining.split(";")[0]),
            "time": round(time.time()),
        }
    except (OSError, KeyError, ValueError, AttributeError):
        return None


def find_unsquashfs(apptainer: str) -> str | None:
    """Apptainer 가 함께 싣는 unsquashfs (`<prefix>/libexec/apptainer/bin/`), 없으면 PATH."""
    exe = shutil.which(apptainer) or apptainer
    prefix = os.path.dirname(os.path.dirname(os.path.realpath(exe)))
    bundled = os.path.join(prefix, "libexec", "apptainer", "bin", "unsquashfs")
    return bundled if os.access(bundled, os.X_OK) else shutil.which("unsquashfs")


def check_sif(apptainer: str, unsquashfs: str, sif: str, repo_dir: str) -> str | None:
    """마운트 없이 `.sif` 의 내용을 확인한다. 문제가 있으면 사유를, 없으면 None 을 돌려준다."""
    v = subprocess.run([apptainer, "sif", "list", sif], capture_output=True, text=True)
    if v.returncode != 0:
        return "sif list failed: " + (v.stderr or "").strip()[-200:]
    span = None
    for line in v.stdout.splitlines():
        cols = [c.strip() for c in line.split("|")]
        if len(cols) >= 5 and cols[4].startswith("FS") and "Squashfs" in cols[4]:
            m = re.fullmatch(r"(\d+)-(\d+)", cols[3])
            if m:
                span = int(m.group(1)), int(m.group(2))
    if span is None:
        return "no squashfs partition in sif list"
    if span[1] > os.path.getsize(sif):
        return f"file is shorter than the squashfs partition ({os.path.getsize(sif)} < {span[1]})"
    ls = subprocess.run(
        [unsquashfs, "-o", str(span[0]), "-l", "-d", "", sif],
        capture_output=True,
        text=True,
        errors="replace",
    )
    if ls.returncode != 0:
        return "squashfs listing failed: " + (ls.stderr or "").strip()[-200:]
    entries = set(ls.stdout.splitlines())
    if repo_dir not in entries:
        return f"repository directory {repo_dir} is not in the image ({len(entries)} entries)"
    return None


class PullBudget:
    """시간당 풀 수 상한. 한 번의 `apptainer build` 가 쓰는 풀 수는 `cost` 로 넘긴다."""

    def __init__(self, per_hour: int):
        self.per_hour = per_hour
        self.stamps: list[float] = []
        self.lock = threading.Lock()

    def acquire(self, cost: int = 1) -> None:
        while True:
            with self.lock:
                now = time.time()
                self.stamps = [t for t in self.stamps if now - t < 3600]
                if len(self.stamps) + cost <= self.per_hour:
                    self.stamps += [now] * cost
                    return
                wait = 3600 - (now - self.stamps[0]) + 1
            time.sleep(min(wait, 60))


def build_one(args, budget: PullBudget, task) -> dict:
    iid, ds, image, rel, repo_dir = task
    dst = os.path.join(args.out, rel)
    rec = {"instance_id": iid, "dataset": ds, "image": image, "sif": rel}
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return rec | {"status": "exists"}
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    work = tempfile.mkdtemp(prefix="sifbuild_", dir=args.tmp)
    tmp_sif = dst + ".tmp"
    env = dict(
        os.environ,
        APPTAINER_CACHEDIR=os.path.join(work, "cache"),
        APPTAINER_TMPDIR=os.path.join(work, "tmp"),
    )
    os.makedirs(env["APPTAINER_CACHEDIR"])
    os.makedirs(env["APPTAINER_TMPDIR"])
    t0 = time.time()
    try:
        for attempt in range(1, args.retries + 1):
            budget.acquire(args.pull_cost)
            p = subprocess.run(
                [args.apptainer, "build", "--force", tmp_sif, f"docker://{image}"],
                env=env,
                capture_output=True,
                text=True,
            )
            if p.returncode == 0:
                break
            err = (p.stderr or "")[-400:]
            limited = "toomanyrequests" in err.lower() or "429" in err
            if attempt == args.retries or not (
                limited or "timeout" in err.lower() or "TLS" in err
            ):
                return rec | {
                    "status": "failed",
                    "seconds": round(time.time() - t0),
                    "error": err.strip().splitlines()[-1][:300] if err.strip() else "",
                }
            time.sleep(600 if limited else 30)
        problem = check_sif(args.apptainer, args.unsquashfs, tmp_sif, repo_dir)
        if problem:
            return rec | {
                "status": "failed",
                "seconds": round(time.time() - t0),
                "error": "content check: " + problem,
            }
        os.replace(tmp_sif, dst)
        return rec | {
            "status": "built",
            "seconds": round(time.time() - t0),
            "bytes": os.path.getsize(dst),
        }
    finally:
        if os.path.exists(tmp_sif):
            os.remove(tmp_sif)
        shutil.rmtree(work, ignore_errors=True)


def self_test(args) -> int:
    """분할과 내용 확인이 고장을 잡는지 본다 (음성 대조 포함). 정상 `.sif` 하나가 필요하다."""
    good, fails = args.self_test, []

    def expect(name: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip())
        if not ok:
            fails.append(name)

    names = [f"swerebench/repo__{i}-{i * 7}.sif" for i in range(2000)]
    for n in (2, 3):
        parts = [{x for x in names if in_shard(x, (k, n))} for k in range(n)]
        expect(
            f"shard N={n} covers every name once",
            set().union(*parts) == set(names) and sum(map(len, parts)) == len(names),
            str([len(p) for p in parts]),
        )
    expect(
        "known-good sif passes",
        check_sif(args.apptainer, args.unsquashfs, good, "/testbed") is None,
    )
    r = check_sif(args.apptainer, args.unsquashfs, good, "/no_such_repo_dir")
    expect("missing repository directory is rejected", r is not None, f"({r})")
    with tempfile.TemporaryDirectory(dir=args.tmp) as d:
        cut = os.path.join(d, "truncated.sif")
        with open(good, "rb") as src, open(cut, "wb") as dst:
            shutil.copyfileobj(src, dst)
            dst.truncate(os.path.getsize(good) // 2)
        r = check_sif(args.apptainer, args.unsquashfs, cut, "/testbed")
        expect("truncated sif is rejected", r is not None, f"({r})")
        junk = os.path.join(d, "not_a_sif.sif")
        with open(junk, "wb") as dst:
            dst.write(b"\0" * (1 << 20))
        r = check_sif(args.apptainer, args.unsquashfs, junk, "/testbed")
        expect("non-sif file is rejected", r is not None, f"({r})")
    q = pull_quota()
    expect("pull quota is readable", q is not None, str(q))
    print(f"self-test: {'FAIL ' + str(fails) if fails else 'PASS'}")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--blend",
        action="append",
        help="블렌드 jsonl (여러 번 줄 수 있다)",
    )
    ap.add_argument("--out", help="sif 디렉토리 (레시피의 sif_dir)")
    ap.add_argument("--apptainer", default="apptainer")
    ap.add_argument(
        "--unsquashfs",
        default=None,
        help="내용 확인에 쓰는 unsquashfs (기본은 Apptainer 가 함께 싣는 것)",
    )
    ap.add_argument(
        "--tmp",
        default=None,
        help="변환 임시 공간 (로컬 디스크). 기본은 시스템 임시 디렉토리",
    )
    ap.add_argument(
        "--shard",
        type=parse_shard,
        default=(0, 1),
        metavar="K/N",
        help="목록을 N 으로 나눈 K 번째만 (노드마다 다른 K)",
    )
    ap.add_argument(
        "--pulls-per-hour",
        type=int,
        default=80,
        help="레지스트리 풀 상한 (노드마다 실측 한도 100/시간)",
    )
    ap.add_argument(
        "--pull-cost",
        type=int,
        default=1,
        help="이미지 하나가 쓰는 풀 수 (실측값으로 준다)",
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="앞에서부터 N 개만 (시험용)")
    ap.add_argument(
        "--log",
        default=None,
        help="결과 기록 jsonl (기본 <out>/build_log.jsonl, 샤드를 주면 build_log.shardKofN.jsonl)",
    )
    ap.add_argument("--dry-run", action="store_true", help="받지 않고 목록 · 검산만")
    ap.add_argument(
        "--verify", action="store_true", help="받지 않고 이미 있는 파일의 내용만 확인"
    )
    ap.add_argument(
        "--self-test", metavar="SIF", help="정상 .sif 하나로 분할 · 내용 확인을 시험"
    )
    args = ap.parse_args()

    if not args.dry_run:
        args.unsquashfs = args.unsquashfs or find_unsquashfs(args.apptainer)
        if not args.unsquashfs:
            print("ERROR: unsquashfs 를 찾지 못했다 (--unsquashfs)", file=sys.stderr)
            return 2
    if args.self_test:
        return self_test(args)
    if not args.blend or not args.out:
        ap.error("--blend 와 --out 이 필요하다")

    tasks, unknown, seen = [], 0, {}
    for b in args.blend:
        for iid, ds, image, rel, repo_dir in tasks_from_blend(b):
            if image is None:
                unknown += 1
                continue
            if rel in seen:
                if seen[rel] != image:
                    print(
                        f"ERROR: 같은 파일 이름에 다른 이미지: {rel}: {seen[rel]} vs {image}",
                        file=sys.stderr,
                    )
                    return 2
                continue
            seen[rel] = image
            tasks.append((iid, ds, image, rel, repo_dir))

    by_ds: dict[str, int] = {}
    mismatch = 0
    for iid, ds, image, rel, _ in tasks:
        by_ds[ds] = by_ds.get(ds, 0) + 1
        stem = gym_lookup_stem(iid, ds)
        if stem is not None and rel != f"r2egym_{stem}.sif":
            mismatch += 1
    have = sum(1 for t in tasks if os.path.exists(os.path.join(args.out, t[3])))
    print(
        f"이미지 {len(tasks)} (데이터셋별 {by_ds}) · 이미 있음 {have} · 처리기 없는 행 {unknown} · Gym 조회 이름과 어긋남 {mismatch}"
    )
    if mismatch or unknown:
        print(
            "ERROR: 이름 검산 실패 또는 모르는 데이터셋 — 받지 않는다", file=sys.stderr
        )
        return 2
    mine = [t for t in tasks if in_shard(t[3], args.shard)]
    if args.shard != (0, 1):
        print(f"샤드 {args.shard[0]}/{args.shard[1]}: {len(mine)} 개")

    if args.verify:
        present = [t for t in mine if os.path.exists(os.path.join(args.out, t[3]))]
        with concurrent.futures.ThreadPoolExecutor(args.workers) as ex:
            problems = list(
                ex.map(
                    lambda t: check_sif(
                        args.apptainer,
                        args.unsquashfs,
                        os.path.join(args.out, t[3]),
                        t[4],
                    ),
                    present,
                )
            )
        bad = [(t[3], p) for t, p in zip(present, problems) if p]
        for rel, p in bad:
            print(f"BAD {rel}: {p}", file=sys.stderr)
        print(
            f"내용 확인: 있음 {len(present)} / {len(mine)} · 통과 {len(present) - len(bad)} · 실패 {len(bad)} · 없음 {len(mine) - len(present)}"
        )
        return 1 if bad or len(present) < len(mine) else 0

    todo = [t for t in mine if not os.path.exists(os.path.join(args.out, t[3]))]
    if args.limit:
        todo = todo[: args.limit]
    hours = len(todo) * args.pull_cost / max(args.pulls_per_hour, 1)
    print(
        f"받을 것 {len(todo)} · 풀 한도 기준 최소 {hours:.1f} 시간 (시간당 {args.pulls_per_hour}, 이미지당 풀 {args.pull_cost})"
    )
    if args.dry_run or not todo:
        return 0

    os.makedirs(args.out, exist_ok=True)
    if args.tmp:
        os.makedirs(args.tmp, exist_ok=True)
    default_log = (
        "build_log.jsonl"
        if args.shard == (0, 1)
        else f"build_log.shard{args.shard[0]}of{args.shard[1]}.jsonl"
    )
    log_path = args.log or os.path.join(args.out, default_log)
    budget = PullBudget(args.pulls_per_hour)
    counts: dict[str, int] = {}
    quota_start = pull_quota()
    print(f"풀 잔여 (시작): {quota_start}", flush=True)
    with (
        open(log_path, "a") as log,
        concurrent.futures.ThreadPoolExecutor(args.workers) as ex,
    ):
        log.write(
            json.dumps({"event": "start", "quota": quota_start, "todo": len(todo)})
            + "\n"
        )
        for n, rec in enumerate(ex.map(lambda t: build_one(args, budget, t), todo), 1):
            counts[rec["status"]] = counts.get(rec["status"], 0) + 1
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log.flush()
            if rec["status"] == "failed":
                print(f"FAILED {rec['image']}: {rec.get('error', '')}", file=sys.stderr)
            if n % 20 == 0:
                print(
                    f"진행 {n}/{len(todo)} {counts} 풀 잔여 {pull_quota()}", flush=True
                )
        quota_end = pull_quota()
        log.write(
            json.dumps({"event": "end", "quota": quota_end, "counts": counts}) + "\n"
        )
    print(f"풀 잔여 (끝): {quota_end}")
    print(f"결과 {counts} · 기록 {log_path}")
    return 1 if counts.get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
