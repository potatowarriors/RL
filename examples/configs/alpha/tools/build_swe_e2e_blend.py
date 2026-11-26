"""SWE E2E 학습 블렌드 (RL_PLAN 결정 77 · 84): Ultra 블렌드 `swe.jsonl` 에서 선별 게이트를 통과한 과제만 남긴다. 행은 고치지 않는다.

선별 게이트 (docs/E2E_SANDBOX.md §7.7, `$NRL_ROOT/p0_measure/swe/golden_lib.py` `gate`):
  1. 그 과제가 들어간 모든 정답 패치 실행에서 보상 1
  2. no-op 실행의 채점 보고서가 있고 FAIL_TO_PASS 중 통과하지 못한 것이 있다
  3. no-op 실행에서 PASS_TO_PASS 가 전부 통과
  4. 정확한 이름의 `.sif` 가 있다 (Gym 은 이름이 없으면 그 이름으로 시작하는 다른 파일을 조용히 쓴다)
정답 패치 검증을 받지 못한 행 (SWE-Gym 206 행 등) 은 넣지 않는다 (사용자 결정 2026-11-26 "채점 못한 221 행은 빼버려").

사용: python3 -I build_swe_e2e_blend.py --out <jsonl> [--seed N]      (옆에 <jsonl>.manifest.json 을 쓴다)
"""
import argparse
import collections
import hashlib
import json
import os
import random
import sys

NRL = "/home/work/vidsearch/tools/nemo_rl"
BLEND = "/home/work/Datasets/LL_datasets/posttraining/RL/Nemotron-RL-Ultra-Training-Blends/swe.jsonl"
SIF = NRL + "/sif/swe/swerebench/"
GOLDEN = ["golden_ultra_b2", "golden_ultra_b3", "golden_ultra_user1", "golden_ultra_rep1", "golden_ultra_c1v", "golden_ultra_c2", "golden_ultra_c3"]
NOOP = ["noop_ultra_1", "noop_ultra_3"]

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=20261126)
a = ap.parse_args()

sys.path.insert(0, NRL + "/p0_measure/swe")
import golden_lib as L  # noqa: E402

ok, why = L.gate(GOLDEN, NOOP)
have = set(os.listdir(SIF))
rows, stats, lang, seen = [], collections.Counter(), collections.Counter(), set()
for line in open(BLEND):
    r = json.loads(line)
    m = r["responses_create_params"]["metadata"]
    iid = m["instance_id"]
    if iid in seen:
        raise SystemExit(f"duplicate instance id in the blend: {iid}")
    seen.add(iid)
    if iid in why:
        stats["dropped: " + why[iid]] += 1
    elif iid not in ok:
        stats[f"dropped: never graded ({m.get('dataset_name')})"] += 1
    elif iid + ".sif" not in have:
        stats["dropped: no .sif with the exact name"] += 1
    else:
        stats["kept"] += 1
        lang[L.language(r)] += 1
        rows.append(line if line.endswith("\n") else line + "\n")
if stats["kept"] != len(ok & seen):
    raise SystemExit(f"kept {stats['kept']} rows but {len(ok & seen)} blend ids passed the gate")
random.Random(a.seed).shuffle(rows)
with open(a.out, "w") as f:
    f.writelines(rows)
sha = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
manifest = {
    "out": a.out, "rows": len(rows), "sha256": sha, "seed": a.seed, "source": BLEND, "source_rows": len(seen),
    "golden_runs": GOLDEN, "noop_runs": NOOP, "counts": dict(stats.most_common()), "language": dict(lang.most_common()),
}  # fmt: skip
json.dump(manifest, open(a.out + ".manifest.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(manifest, indent=1, ensure_ascii=False))
