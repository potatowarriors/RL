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

사용:
  build_sifs.py --blend <swe.jsonl> [--blend <swe2.jsonl>] --out <sif_dir> --dry-run
  build_sifs.py --blend ... --out <sif_dir> --apptainer <bin> [--pulls-per-hour 80] [--workers 4] [--limit N]
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


def tasks_from_blend(path: str):
    """블렌드 한 줄마다 (instance_id, dataset_name, image, sif 상대 경로) 를 낸다."""
    with open(path) as f:
        for line in f:
            md = json.loads(line)["responses_create_params"]["metadata"]
            d = md.get("instance_dict")
            d = json.loads(d) if isinstance(d, str) else (d or {})
            ds = md.get("dataset_name") or d.get("dataset_name") or ""
            iid = md["instance_id"]
            if "SWE-rebench" in ds:
                image, rel = d["image_name"], f"swerebench/{iid}.sif"
            elif "R2E-Gym" in ds:
                image = d["docker_image"]
                rel = "r2egym_" + image.split("/")[-1].replace(":", "_") + ".sif"
            elif "SWE-Gym" in ds:
                name = "sweb.eval.x86_64." + iid.replace("__", "_s_").lower()
                image, rel = f"xingyaoww/{name}", f"swegym_{name}.sif"
            else:
                yield iid, ds, None, None
                continue
            yield iid, ds, image.removeprefix("docker.io/"), rel


def gym_lookup_stem(iid: str, ds: str) -> str | None:
    """Gym `_find_container` 가 R2E-Gym 행에서 만드는 이름 (검산용)."""
    if "R2E-Gym" in ds:
        return re.sub(r"[^_]+__([^-]+)-", lambda m: m.group(1).lower() + "_final_", iid)
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
    iid, ds, image, rel = task
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
        v = subprocess.run(
            [args.apptainer, "sif", "list", tmp_sif],
            env=env,
            capture_output=True,
            text=True,
        )
        if v.returncode != 0 or "FS" not in v.stdout:
            return rec | {
                "status": "failed",
                "seconds": round(time.time() - t0),
                "error": "sif list did not show a filesystem partition",
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


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--blend",
        action="append",
        required=True,
        help="블렌드 jsonl (여러 번 줄 수 있다)",
    )
    ap.add_argument("--out", required=True, help="sif 디렉토리 (레시피의 sif_dir)")
    ap.add_argument("--apptainer", default="apptainer")
    ap.add_argument(
        "--tmp",
        default=None,
        help="변환 임시 공간 (로컬 디스크). 기본은 시스템 임시 디렉토리",
    )
    ap.add_argument(
        "--pulls-per-hour",
        type=int,
        default=80,
        help="레지스트리 풀 상한 (이 노드 실측 한도 100/시간)",
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
        "--log", default=None, help="결과 기록 jsonl (기본 <out>/build_log.jsonl)"
    )
    ap.add_argument("--dry-run", action="store_true", help="받지 않고 목록 · 검산만")
    args = ap.parse_args()

    tasks, unknown, seen = [], 0, {}
    for b in args.blend:
        for iid, ds, image, rel in tasks_from_blend(b):
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
            tasks.append((iid, ds, image, rel))

    by_ds: dict[str, int] = {}
    mismatch = 0
    for iid, ds, image, rel in tasks:
        by_ds[ds] = by_ds.get(ds, 0) + 1
        stem = gym_lookup_stem(iid, ds)
        if stem is not None and rel != f"r2egym_{stem}.sif":
            mismatch += 1
    have = sum(
        1 for _, _, _, rel in tasks if os.path.exists(os.path.join(args.out, rel))
    )
    print(
        f"이미지 {len(tasks)} (데이터셋별 {by_ds}) · 이미 있음 {have} · 처리기 없는 행 {unknown} · Gym 조회 이름과 어긋남 {mismatch}"
    )
    if mismatch or unknown:
        print(
            "ERROR: 이름 검산 실패 또는 모르는 데이터셋 — 받지 않는다", file=sys.stderr
        )
        return 2
    todo = [t for t in tasks if not os.path.exists(os.path.join(args.out, t[3]))]
    if args.limit:
        todo = todo[: args.limit]
    hours = len(todo) * args.pull_cost / max(args.pulls_per_hour, 1)
    print(
        f"받을 것 {len(todo)} · 풀 한도 기준 최소 {hours:.1f} 시간 (시간당 {args.pulls_per_hour}, 이미지당 풀 {args.pull_cost})"
    )
    if args.dry_run or not todo:
        return 0

    os.makedirs(args.out, exist_ok=True)
    log_path = args.log or os.path.join(args.out, "build_log.jsonl")
    budget = PullBudget(args.pulls_per_hour)
    counts: dict[str, int] = {}
    with (
        open(log_path, "a") as log,
        concurrent.futures.ThreadPoolExecutor(args.workers) as ex,
    ):
        for rec in ex.map(lambda t: build_one(args, budget, t), todo):
            counts[rec["status"]] = counts.get(rec["status"], 0) + 1
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log.flush()
            if rec["status"] == "failed":
                print(f"FAILED {rec['image']}: {rec.get('error', '')}", file=sys.stderr)
    print(f"결과 {counts} · 기록 {log_path}")
    return 1 if counts.get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
