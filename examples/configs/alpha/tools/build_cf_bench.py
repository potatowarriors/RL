"""[2026-10-27 적대적 검토 판정: 신뢰 불가 — 기준선 미실행] 214문제 중 212문제가 SFT 블렌드 cp_v2 에 원문으로 있고(held-out 아님), 상수 출력이 7문제를 통과하는 퇴화 테스트·대소문자 불일치·다중 정답 문제가 있다. 고치기 전에는 쓰지 않는다 (KNOWN_ISSUES 2026-10-27, 검토 보고 job tmp/cf_review/review.md).
Codeforces 코드 벤치 변환 (사용자 지시 2026-10-27 "lcb, codeforce 벤치 테스트 진행해").

원천: HF open-r1/codeforces `verifiable_tests` test split (422문제, 2024-08-04 ~ 2025-02-11 대회, 전부 stdio·executable).
LCB v5 와 같은 기간이라 LCB 와 같은 오염 조건이다. 학습 코드 풀과는 13-gram 으로 대조해 겹치는 문제를 뺀다.
제외: 특수 채점기(generated_checker — 정답이 여럿인 문제, Gym code_gen 은 줄 단위 정확 비교라 채점 불가) · 대화형(interaction_format).
테스트: official_tests_complete 면 공식 테스트, 아니면 generated_tests (open-r1 이 공식 해답으로 검증한 생성 테스트).
  Gym 은 문제의 테스트 전부를 요청마다 Ray 로 넘기므로 문제당 바이트 상한(--max-bytes, 기본 8 MB)을 둔다 — 작은 테스트부터 넣고
  상한을 넘는 큰 테스트는 뺀다. 문제마다 tests_kept / tests_total / largest_dropped 를 남겨 상한의 영향을 사후 점검한다.
프롬프트: LCB v5 stdin 문제와 같은 형식 (같은 system 문구 · "### Question:" · "### Format:" 지시 · ```python 블록). 채점은 LCB 와 같은 Gym code_gen.
사용: python build_cf_bench.py --src <verifiable_tests test parquet dir> --pool <teacher_pool/code_pool.jsonl> --out <eval/cf_or1_2408_2502.jsonl>
결과: <out> · <out>_stats.json
"""
import argparse, glob, json, re, hashlib, collections, statistics
import pandas as pd

SYSTEM = ("You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct "
          "Python program that matches the specification and passes all tests.")
FORMAT = ("### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). "
          "Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and "
          "writes output to STDOUT.\n```python\n# YOUR CODE HERE\n```\n\n### Answer: (use the provided format with backticks)\n\n")

def statement(r):
    parts = [str(r["description"] or "").strip()]
    if r["input_format"]: parts.append("Input\n\n" + str(r["input_format"]).strip())
    if r["output_format"]: parts.append("Output\n\n" + str(r["output_format"]).strip())
    ex = r["examples"]
    if ex is not None and len(ex):
        for i, e in enumerate(ex, 1):
            parts.append(f"Sample Input {i}\n\n{str(e['input']).rstrip()}\n\nSample Output {i}\n\n{str(e['output']).rstrip()}")
    if r["note"]: parts.append("Note\n\n" + str(r["note"]).strip())
    return "\n\n".join(parts)

TOK = re.compile(r"[a-z0-9]+")
def grams(text, n=13):
    t = TOK.findall(text.lower())
    return {hashlib.md5(" ".join(t[i:i + n]).encode()).digest()[:8] for i in range(max(0, len(t) - n + 1))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--pool", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--max-bytes", type=int, default=8_000_000); ap.add_argument("--contam", type=float, default=0.3)
    a = ap.parse_args()
    pool = set()
    with open(a.pool) as f:
        for line in f:
            r = json.loads(line)
            msgs = (r.get("responses_create_params") or {}).get("input") or []
            pool |= grams(" ".join(str(m.get("content", "")) for m in msgs if m.get("role") == "user"))
    stats = collections.Counter(); rows = []; ratings = collections.Counter(); kept_frac = []
    for fn in sorted(glob.glob(a.src + "/*.parquet")):
        for _, r in pd.read_parquet(fn).iterrows():
            stats["problems_in"] += 1
            if r["generated_checker"] is not None and str(r["generated_checker"]).strip():
                stats["excluded_checker"] += 1; continue
            if r["interaction_format"] is not None and str(r["interaction_format"]).strip():
                stats["excluded_interactive"] += 1; continue
            if not bool(r["executable"]):
                stats["excluded_not_executable"] += 1; continue
            tests = list(r["official_tests"]) if bool(r["official_tests_complete"]) else list(r["generated_tests"] if r["generated_tests"] is not None else [])
            if not tests:
                stats["excluded_no_tests"] += 1; continue
            text = statement(r)
            g = grams(text); ov = len(g & pool) / max(1, len(g))
            if ov >= a.contam:
                stats["excluded_pool_overlap"] += 1; continue
            tests = sorted(tests, key=lambda t: len(t["input"]) + len(t["output"]))
            sel, used = [], 0
            for t in tests:
                s = len(t["input"]) + len(t["output"])
                if used + s > a.max_bytes: continue
                sel.append(t); used += s
            if not sel:
                stats["excluded_all_tests_over_budget"] += 1; continue
            largest_dropped = len(sel) < len(tests) and (len(tests[-1]["input"]) + len(tests[-1]["output"])) > 0 and tests[-1] not in sel
            stats["tests_truncated"] += len(sel) < len(tests); kept_frac.append(len(sel) / len(tests))
            rating = int(r["rating"]) if r["rating"] == r["rating"] and r["rating"] is not None else None
            ratings[(rating // 400 * 400) if rating else None] += 1
            rows.append({
                "responses_create_params": {"input": [{"role": "system", "content": SYSTEM},
                                                      {"role": "user", "content": "### Question:\n" + text + "\n\n" + FORMAT}]},
                "verifier_metadata": {"problem_id": str(r["id"]), "unit_tests": {"inputs": [t["input"] for t in sel], "outputs": [t["output"] for t in sel], "fn_name": None},
                                      "rating": rating, "contest_start": int(r["contest_start"]), "time_limit": float(r["time_limit"]),
                                      "tests_kept": len(sel), "tests_total": len(tests), "largest_dropped": bool(largest_dropped), "pool_overlap_13gram": round(ov, 4)},
                "agent_ref": {"name": "code_gen_simple_agent"}, "dataset": "codeforces_or1_test_2408_2502", "pool_key": "cf_" + str(r["id"]).replace("/", "_")})
    with open(a.out, "w") as f:
        for x in rows: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    out = {"src": a.src, "pool": a.pool, "max_bytes": a.max_bytes, "contam_threshold": a.contam, "rows": len(rows), "counts": dict(stats),
           "rating_buckets": {str(k): v for k, v in sorted(ratings.items(), key=lambda x: (x[0] is None, x[0]))},
           "tests_kept_fraction_median": statistics.median(kept_frac) if kept_frac else None,
           "problems_with_all_tests": sum(1 for x in kept_frac if x == 1.0)}
    json.dump(out, open(a.out.replace(".jsonl", "_stats.json"), "w"), indent=1, ensure_ascii=False); print(json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()
