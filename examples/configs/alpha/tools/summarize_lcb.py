"""LiveCodeBench v5 결과 집계 (runs/eval_lcb.sh 가 호출한다).

사용: python summarize_lcb.py <run_dir> [--cap 32768]
<run_dir>/rollouts.jsonl (Gym eval run 출력) 과 rollouts_materialized_inputs.jsonl (문제 메타) 를 읽어
pass@1 (문제당 반복의 평균 정답률, 문제 평균) · pass@k (k 회 중 1회 이상) · 함수형/표준입력 별 · 끝냄·잘림 비율 · 응답 길이를
<run_dir>/lcb_summary.json 로 쓰고 출력한다.
"""

import argparse
import collections
import json
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--cap", type=int, default=32768)
    a = ap.parse_args()

    kind = {}
    with open(f"{a.run_dir}/rollouts_materialized_inputs.jsonl") as f:
        for line in f:
            o = json.loads(line)
            ut = (o.get("verifier_metadata") or {}).get("unit_tests") or {}
            kind.setdefault(o["_ng_task_index"], "functional" if ut.get("fn_name") else "stdin")

    per = collections.defaultdict(lambda: {"n": 0, "c": 0, "trunc": 0, "tok": []})
    with open(f"{a.run_dir}/rollouts.jsonl") as f:
        for line in f:
            o = json.loads(line)
            d = per[o["_ng_task_index"]]
            usage = (o.get("response") or {}).get("usage") or {}
            tok = usage.get("output_tokens") or 0
            d["n"] += 1
            d["c"] += float(o.get("reward", 0)) > 0
            d["trunc"] += tok >= a.cap
            d["tok"].append(tok)

    def block(items):
        if not items:
            return {}
        k = min(v["n"] for v in items)
        toks = [t for v in items for t in v["tok"]]
        n = sum(v["n"] for v in items)
        return {
            "problems": len(items),
            "repeats_min": k,
            "pass@1": round(100 * statistics.mean(v["c"] / v["n"] for v in items), 2),
            f"pass@{k}": round(100 * statistics.mean(v["c"] > 0 for v in items), 2),
            "truncation_rate": round(sum(v["trunc"] for v in items) / n, 4),
            "output_tokens_median": int(statistics.median(toks)),
            "output_tokens_p90": sorted(toks)[int(0.9 * (len(toks) - 1))],
        }

    items = list(per.values())
    summary = {
        "run_dir": a.run_dir,
        "cap": a.cap,
        "all": block(items),
        "functional": block([v for t, v in per.items() if kind.get(t) == "functional"]),
        "stdin": block([v for t, v in per.items() if kind.get(t) == "stdin"]),
    }
    with open(f"{a.run_dir}/lcb_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
