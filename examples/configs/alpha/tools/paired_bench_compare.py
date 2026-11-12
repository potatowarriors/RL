"""같은 문항끼리 짝지은 벤치 비교 (CPU, 2026-11-10~11 MOPD1 teacher·학생 판정에 쓴 것을 정리).

두 체크포인트의 문항별 점수 a_i, b_i (반복 k 회 평균) 로 d_i = a_i − b_i, Δ = mean(d), SE = sqrt(Σ(d_i − Δ)² / (n(n−1))).
구간은 Δ ± z·SE 이고 z 는 비교 쌍 수 m 에 건 Bonferroni (양측 95%: m=1 1.96 · 2 2.24 · 4 2.50). 구간이 0 을 빼면 "구분됨".
점수 차만 보지 않고 같은 문항에서 비교하므로 문항 난이도 분산이 빠진다.

  Pai 스위트 결과 (lm_eval `--log_samples`): eval_sft/results/<tag>/alpha/samples_<task>_*.jsonl 의 doc_hash 로 짝짓는다.
    python paired_bench_compare.py pai <results_dir> --a <tag_a> --b <tag_b> --task mmlu_pro_aa:exact_match \
        --task ifbench_aa:prompt_level_loose_acc [--m 4]
  LCB (`runs/eval_lcb.sh` 결과): $NRL_ROOT/p0_measure/<tag>/rollouts.jsonl 의 `_ng_task_index` 로 짝짓고 reward 를 평균한다.
    python paired_bench_compare.py lcb <p0_measure_dir> --a lcb_mopd1_s3_120 --b lcb_s2r_228 [--m 2]
측정 조건 (GDN 상태 dtype · prefix caching · 날짜) 이 다른 두 결과를 짝지으면 그 차이도 Δ 에 섞인다 — 비교 전에 조건을 맞춘다
(`docs/study/benchmark_results.md` §5).
"""

import argparse
import collections
import glob
import json
import math
import sys

Z = {1: 1.96, 2: 2.2414, 3: 2.394, 4: 2.4977, 5: 2.5758, 6: 2.638}


def load_pai(root, tag, task, key):
    files = sorted(glob.glob(f"{root}/{tag}/alpha/samples_{task}_*.jsonl"))
    if not files:
        return None
    out = {}
    for line in open(files[-1]):
        d = json.loads(line)
        out[d["doc_hash"]] = float(d[key])
    return out


def load_lcb(root, tag):
    acc = collections.defaultdict(list)
    for line in open(f"{root}/{tag}/rollouts.jsonl"):
        d = json.loads(line)
        acc[d["_ng_task_index"]].append(float(d.get("reward") or 0.0))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def compare(label, A, B, z):
    if A is None or B is None:
        print(f"{label:40s} 자료 없음")
        return None
    ks = sorted(set(A) & set(B))
    n = len(ks)
    d = [A[k] - B[k] for k in ks]
    m = sum(d) / n
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n * (n - 1)))
    lo, hi = m - z * se, m + z * se
    up, dn = sum(x > 0 for x in d), sum(x < 0 for x in d)
    verdict = "구분됨" if lo > 0 or hi < 0 else "구분 안 됨"
    print(f"{label:40s} n={n:5d} a={100 * sum(A[k] for k in ks) / n:5.2f} b={100 * sum(B[k] for k in ks) / n:5.2f} "
          f"Δ={100 * m:+5.2f} ± {100 * se:.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}] {verdict} (오른 {up} · 내린 {dn})")
    return {"n": n, "delta": m, "se": se, "lo": lo, "hi": hi, "verdict": verdict}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pai", "lcb"])
    ap.add_argument("root")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--task", action="append", default=[], help="pai: <task>:<문항 점수 키>")
    ap.add_argument("--m", type=int, default=0, help="Bonferroni 쌍 수 (기본: 이번 호출의 비교 수)")
    a = ap.parse_args()
    if a.mode == "pai":
        if not a.task:
            ap.error("pai 는 --task <task>:<key> 가 필요하다")
        m = a.m or len(a.task)
        z = Z.get(m, 2.638)
        for spec in a.task:
            task, key = spec.split(":", 1)
            compare(f"{task} {a.a} − {a.b}", load_pai(a.root, a.a, task, key), load_pai(a.root, a.b, task, key), z)
    else:
        m = a.m or 1
        compare(f"lcb {a.a} − {a.b}", load_lcb(a.root, a.a), load_lcb(a.root, a.b), Z.get(m, 2.638))
    return 0


if __name__ == "__main__":
    sys.exit(main())
