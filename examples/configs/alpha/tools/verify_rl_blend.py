"""Structural gate for alpha RL blend jsonl files (NeMo Gym row format).

Checks every line:
  - valid JSON
  - has dict `responses_create_params` and `agent_ref` with a non-empty name
  - no residual `_hf_question_placeholder` (masked math must be restored)
Prints the agent_ref.name distribution for blend-composition review.

Usage:
  python verify_rl_blend.py /path/to/blend.jsonl [more.jsonl ...]
"""

import collections
import json
import sys


def check_file(path: str) -> bool:
    counts: collections.Counter = collections.Counter()
    bad_json = bad_keys = residual_masked = 0
    total = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue
            rcp = row.get("responses_create_params")
            name = (row.get("agent_ref") or {}).get("name")
            if not isinstance(rcp, dict) or not name:
                bad_keys += 1
                continue
            if "_hf_question_placeholder" in row:
                residual_masked += 1
            counts[name] += 1

    ok = bad_json == 0 and bad_keys == 0 and residual_masked == 0
    print(f"\n=== {path}")
    print(f"rows={total} bad_json={bad_json} bad_keys={bad_keys} residual_masked={residual_masked} -> {'OK' if ok else 'FAIL'}")
    for name, n in counts.most_common():
        print(f"  {n:7d}  {n / total:6.2%}  {name}")
    return ok


def main() -> int:
    results = [check_file(p) for p in sys.argv[1:]]
    if all(results):
        print("\nALL BLEND FILES STRUCTURALLY OK")
        return 0
    print("\nBLEND VALIDATION FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
