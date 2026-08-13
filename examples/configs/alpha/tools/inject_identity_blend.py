"""Inject alpha identity rows into an RL blend jsonl.

The Ultra RL blends contain ZERO identity rows; per the alpha data plan
(DATA_PREP_LOG.md §3), alpha-RL-Identity-Following-v1 must be *added* at
0.3–1.0% of the blend (never as standalone repeated epochs), and identity
SFT injection must precede RL.

Usage:
  python inject_identity_blend.py \
      --blend /path/to/rlvr1.jsonl \
      --identity /path/to/alpha-RL-Identity-Following-v1/train.jsonl \
      --ratio 0.007 --seed 20260813 \
      --output /path/to/rlvr1_alpha.jsonl

Sampling is deterministic (seeded, without replacement). Output rows are
shuffled so identity rows are spread across the epoch. A summary with
agent_ref distribution deltas is printed for the run log.
"""

import argparse
import collections
import json
import random


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", required=True)
    ap.add_argument("--identity", required=True)
    ap.add_argument("--ratio", type=float, default=0.007, help="identity fraction of the base blend size (0.003-0.01 per plan)")
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if not (0.003 <= args.ratio <= 0.01):
        raise SystemExit(f"ratio {args.ratio} outside the plan's 0.3-1.0% band; refusing")

    rng = random.Random(args.seed)

    with open(args.blend) as f:
        blend_lines = f.read().splitlines()
    with open(args.identity) as f:
        identity_lines = f.read().splitlines()

    n_inject = round(len(blend_lines) * args.ratio)
    if n_inject > len(identity_lines):
        raise SystemExit(f"need {n_inject} identity rows but only {len(identity_lines)} available")
    injected = rng.sample(identity_lines, n_inject)

    # Structural sanity on every injected row (cheap; blend rows are upstream-validated)
    for line in injected:
        row = json.loads(line)
        assert "responses_create_params" in row and "agent_ref" in row, "identity row missing Gym keys"
        assert row["agent_ref"].get("name"), "identity row missing agent_ref.name"

    out_lines = blend_lines + injected
    rng.shuffle(out_lines)
    with open(args.output, "w") as f:
        f.write("\n".join(out_lines) + "\n")

    def agent_counts(lines: list[str], sample_every: int = 1) -> collections.Counter:
        c: collections.Counter = collections.Counter()
        for i, line in enumerate(lines):
            if i % sample_every:
                continue
            try:
                c[json.loads(line).get("agent_ref", {}).get("name", "<none>")] += 1
            except json.JSONDecodeError:
                c["<unparseable>"] += 1
        return c

    inj_counts = agent_counts(injected)
    print(f"base blend rows     : {len(blend_lines)}")
    print(f"identity injected   : {n_inject} ({args.ratio:.2%})")
    print(f"output rows         : {len(out_lines)} -> {args.output}")
    print(f"injected agent_refs : {dict(inj_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
