"""다른 런의 체크포인트에서 시작하는 런의 W&B 곡선을 앞 런에 이어 붙인다 — 앞 런의 step 1 ~ N 이력을 새 W&B 런에 복사한다.

쓰는 때: 런 B 가 런 A 의 step N 가중치에서 시작한다 (재개가 아니라 새 런). W&B 에서 B 의 곡선이 A 의 1 ~ N 뒤에 이어지게 하려면
  1. 이 도구로 A 의 step 1 ~ N 을 새 W&B 런 (B 의 id) 에 복사하고
  2. B 를 `++logger.wandb.id=<B id> ++logger.wandb.resume=allow ++logger.wandb.step_offset=N` 으로 띄운다
     (`step_offset` 은 본체 `nemo_rl/utils/logger.py` 의 W&B 전용 설정 — B 의 step 1 이 N + 1 로 기록된다. 재개할 때도 같은 값을 준다).

  wandb_seed_from_run.py --project alpha-rl --source-id <A id> --upto N --new-id <B id> --name <B 이름> --group <캠페인> [--entity E] [--apply]

`--apply` 가 없으면 읽기만 한다 (복사할 step 범위와 지표 수를 찍는다). `--apply` 는 W&B 에 새 런을 만든다 — 같은 id 의 런이 이미 있으면 멈춘다.
복사하는 것은 스칼라 이력뿐이다 (A 가 W&B 에 보낸 지표 그대로). 시스템 지표 · 아티팩트 · 원래의 기록 시각은 옮기지 않는다 — 복사된 구간은 Step 축으로만 읽는다.
키는 환경변수 `WANDB_API_KEY` 로 받는다 (`runs/launch.sh` 와 같은 키 파일).
"""
import argparse
import math
import sys

import wandb


def read_history(api: "wandb.Api", path: str, upto: int) -> list[tuple[int, dict]]:
    """Return [(step, metrics)] of `path` for 1 <= step <= upto, in step order."""
    rows: dict[int, dict] = {}
    for row in api.run(path).scan_history(page_size=1000):
        step = row.get("_step")
        if step is None or not 1 <= int(step) <= upto:
            continue
        metrics = {
            k: v
            for k, v in row.items()
            if not k.startswith("_") and isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        }
        rows.setdefault(int(step), {}).update(metrics)
    return sorted(rows.items())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--entity")
    ap.add_argument("--source-id", required=True)
    ap.add_argument("--upto", type=int, required=True, help="복사할 마지막 step (시작 체크포인트의 step)")
    ap.add_argument("--new-id", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--group")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    if args.upto < 1:
        ap.error("--upto 는 1 이상이어야 한다")

    api = wandb.Api()
    entity = args.entity or api.default_entity
    src = f"{entity}/{args.project}/{args.source_id}"
    dst = f"{entity}/{args.project}/{args.new_id}"
    rows = read_history(api, src, args.upto)
    steps = [s for s, _ in rows]
    keys = sorted({k for _, m in rows for k in m})
    print(f"원본 {args.project}/{args.source_id}: step {steps[0] if steps else '없음'} ~ {steps[-1] if steps else '없음'} · {len(steps)} 개 step · 지표 {len(keys)} 종")
    missing = [s for s in range(1, args.upto + 1) if s not in set(steps)]
    if missing:
        print(f"   ! 원본에 없는 step: {missing[:10]}{' …' if len(missing) > 10 else ''}")
    if not steps or steps[-1] != args.upto:
        print(f"중단: 원본에 step {args.upto} 이 없다 — 시작 체크포인트의 step 과 맞지 않는다")
        return 1

    try:
        existing = api.run(dst)
        n_existing = sum(1 for _ in existing.scan_history(keys=["_step"], page_size=1000))
    except wandb.errors.CommError:
        existing, n_existing = None, 0
    if existing is not None:
        print(f"중단: 대상 런 {args.project}/{args.new_id} 이 이미 있다 (이력 {n_existing} 행) — 두 번 복사하지 않는다")
        return 1
    if not args.apply:
        print(f"읽기만 했다. --apply 를 주면 step 1 ~ {args.upto} ({len(steps)} 행) 을 새 런 {args.project}/{args.new_id} 에 복사한다")
        return 0

    run = wandb.init(
        project=args.project,
        entity=entity,
        id=args.new_id,
        name=args.name,
        group=args.group,
        resume="never",
        notes=f"step 1 ~ {args.upto} 은 {args.source_id} 의 이력을 복사한 것이다 (시작 가중치 = {args.source_id} step {args.upto}). 그 뒤는 이 런의 step + {args.upto}.",
        config={"seeded_from_run": args.source_id, "seeded_upto_step": args.upto},
    )
    for step, metrics in rows:
        run.log(metrics, step=step, commit=True)
    run.finish()

    copied = read_history(wandb.Api(), dst, args.upto)
    got = [s for s, _ in copied]
    ok = got == steps and all(set(m) == set(dict(rows)[s]) for s, m in copied)
    print(f"복사 확인: 대상 런의 step {got[0] if got else '없음'} ~ {got[-1] if got else '없음'} · {len(got)} 행 · 원본과 {'같다' if ok else '다르다'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
