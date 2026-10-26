"""[2026-10-27 적대적 검토 판정: 신뢰 불가 — 기준선 미실행] 214문제 중 212문제가 SFT 블렌드 cp_v2 에 원문으로 있고(held-out 아님), 상수 출력이 7문제를 통과하는 퇴화 테스트·대소문자 불일치·다중 정답 문제가 있다. 고치기 전에는 쓰지 않는다 (KNOWN_ISSUES 2026-10-27, 검토 보고 job tmp/cf_review/review.md).
Codeforces 벤치 결과 집계 — summarize_lcb.py 와 같은 정의(pass@1 = 문제당 반복 평균의 문제 평균, pass@k = k 회 중 1회 이상)에
레이팅 구간별 표와, 같은 문제끼리 짝지은 체크포인트 비교(대응 표본 SE)를 더한다.
사용: python summarize_cf.py <run_dir> [<run_dir> ...] [--cap 32768] [--base <run_dir>]
"""
import argparse, collections, json, math, statistics

def load(run_dir, cap):
    meta = {}
    with open(f"{run_dir}/rollouts_materialized_inputs.jsonl") as f:
        for line in f:
            o = json.loads(line); vm = o.get("verifier_metadata") or {}
            meta.setdefault(o["_ng_task_index"], (vm.get("problem_id"), vm.get("rating")))
    per = collections.defaultdict(lambda: {"n": 0, "c": 0, "trunc": 0})
    with open(f"{run_dir}/rollouts.jsonl") as f:
        for line in f:
            o = json.loads(line); d = per[meta[o["_ng_task_index"]][0]]
            tok = ((o.get("response") or {}).get("usage") or {}).get("output_tokens") or 0
            d["n"] += 1; d["c"] += float(o.get("reward", 0)) > 0; d["trunc"] += tok >= cap
    rating = {pid: r for pid, r in meta.values()}
    return per, rating

def block(per, keys):
    p1 = [per[k]["c"] / per[k]["n"] for k in keys]; pk = [1.0 if per[k]["c"] else 0.0 for k in keys]
    n = len(keys); se = statistics.pstdev(p1) / math.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "pass@1": round(100 * statistics.mean(p1), 2), "pass@1_se": round(100 * se, 2), "pass@k": round(100 * statistics.mean(pk), 2),
            "truncation_rate": round(sum(per[k]["trunc"] for k in keys) / max(1, sum(per[k]["n"] for k in keys)), 4)}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--cap", type=int, default=32768); ap.add_argument("--base")
    a = ap.parse_args()
    base = load(a.base, a.cap)[0] if a.base else None
    for rd in a.runs:
        per, rating = load(rd, a.cap); keys = sorted(per)
        out = {"run_dir": rd, "all": block(per, keys), "by_rating": {}}
        buckets = collections.defaultdict(list)
        for k in keys:
            r = rating.get(k); b = "unrated" if r is None else ("800-1199" if r < 1200 else "1200-1599" if r < 1600 else "1600-1999" if r < 2000 else "2000-2399" if r < 2400 else "2400+")
            buckets[b].append(k)
        out["by_rating"] = {b: block(per, ks) for b, ks in sorted(buckets.items())}
        if base:
            common = [k for k in keys if k in base]
            d = [per[k]["c"] / per[k]["n"] - base[k]["c"] / base[k]["n"] for k in common]
            out["paired_vs_base"] = {"base": a.base, "n": len(common), "delta_pass@1": round(100 * statistics.mean(d), 2),
                                     "se": round(100 * statistics.pstdev(d) / math.sqrt(len(d)), 2)}
        json.dump(out, open(f"{rd}/cf_summary.json", "w"), indent=1); print(json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()
