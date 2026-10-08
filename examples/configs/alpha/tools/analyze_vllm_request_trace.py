"""vLLM 요청별 시각 기록(NRL_VLLM_REQUEST_TRACE_DIR) 분석 — rollout 긴 꼬리의 대기 vs 디코딩 분해 (CPU, 2026-10-08).

기록 한 줄 = HTTP chat completion 한 번 (모델 호출 1회). 엔진 시각(queued/scheduled/first/last token, monotonic)을
now_mono/now_wall 로 벽시계로 바꾼다. 요청이 KV 부족으로 선점됐다가 다시 돌면 그 대기는 decode 구간에 섞인다.
  - 대기(queue) = scheduled − queued · prefill = first_token − scheduled · decode = last_token − first_token
  - 긴 요청(생성 ≥ --long-tokens 또는 finish=length)의 대기·디코딩·속도 분포와, 시간 구간별 동시 실행 수
사용: python analyze_vllm_request_trace.py <trace_dir> [--long-tokens 60000] [--bin-min 5]
"""

import argparse
import glob
import json
import os
import statistics as st


def pct(values, q):
    v = sorted(values)
    return v[min(len(v) - 1, int(q / 100 * len(v)))] if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("--long-tokens", type=int, default=60000)
    ap.add_argument("--bin-min", type=float, default=5.0)
    args = ap.parse_args()

    recs = []
    for path in glob.glob(os.path.join(args.trace_dir, "*.jsonl")):
        engine = os.path.basename(path).rsplit("_", 1)[-1].split(".")[0]
        for line in open(path):
            r = json.loads(line)
            off = r["now_wall"] - r["now_mono"]
            r["engine"] = engine
            r["q_wall"] = r["queued_ts"] + off
            r["s_wall"] = r["scheduled_ts"] + off
            r["f_wall"] = r["first_token_ts"] + off
            r["l_wall"] = r["last_token_ts"] + off
            recs.append(r)
    if not recs:
        print("no records")
        return
    t0 = min(r["q_wall"] for r in recs)
    for r in recs:
        r["queue"] = r["s_wall"] - r["q_wall"]
        r["prefill"] = r["f_wall"] - r["s_wall"]
        r["decode"] = r["l_wall"] - r["f_wall"]
        r["tps"] = r["gen_tokens"] / r["decode"] if r["decode"] > 0 else float("nan")

    longs = [r for r in recs if r["gen_tokens"] >= args.long_tokens or r["finish_reason"] == "length"]
    print(f"requests {len(recs)} · engines {len({r['engine'] for r in recs})} · span {(max(r['l_wall'] for r in recs) - t0) / 60:.1f} min")
    for name, sel in (("all", recs), (f"long (≥{args.long_tokens} or length)", longs)):
        if not sel:
            continue
        print(f"== {name}: n={len(sel)}")
        for key, unit in (("queue", "s"), ("prefill", "s"), ("decode", "s"), ("tps", "tok/s"), ("gen_tokens", "tok")):
            vals = [r[key] for r in sel]
            print(f"   {key:<10} p50 {pct(vals, 50):9.1f}  p90 {pct(vals, 90):9.1f}  max {max(vals):9.1f}  mean {st.mean(vals):9.1f} {unit}")
    if longs:
        share_q = sum(r["queue"] for r in longs) / sum(r["l_wall"] - r["q_wall"] for r in longs)
        print(f"   long requests: queue share of their end-to-end time {100 * share_q:.1f}%")
        starts = sorted((r["q_wall"] - t0) / 60 for r in longs)
        print(f"   long requests enqueued at (min from first request): p10 {pct(starts, 10):.1f} p50 {pct(starts, 50):.1f} p90 {pct(starts, 90):.1f}")
    # 시간 구간별 동시 실행(스케줄됨~끝) 수와 대기 수
    end = max(r["l_wall"] for r in recs)
    b = args.bin_min * 60
    print(f"== per {args.bin_min:g}-min bin: running / waiting / finished-long")
    t = t0
    while t < end:
        mid = t + b / 2
        running = sum(1 for r in recs if r["s_wall"] <= mid < r["l_wall"])
        waiting = sum(1 for r in recs if r["q_wall"] <= mid < r["s_wall"])
        fin_long = sum(1 for r in longs if t <= r["l_wall"] < t + b)
        print(f"   {(t - t0) / 60:6.1f} min  running {running:5d}  waiting {waiting:5d}  long_finished {fin_long:4d}")
        t += b


if __name__ == "__main__":
    main()
