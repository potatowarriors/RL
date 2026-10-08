"""학습 데이터 덤프(train_data_step*.jsonl)의 agent 별 통계 (CPU, 2026-10-08).

한 줄 = 학습 샘플 하나. agent 마다 샘플 수 · 학습 토큰 비중 · 생성 토큰 비중 · 잘림 · 보상 > 0 · advantage 0 비율을 낸다.
  - 생성 토큰 = token_loss_mask 1 인 토큰. 모델 호출 하나 = token_loss_mask 의 연속 1 구간
  - 잘림 = 어떤 모델 호출의 길이 ≥ --cap (요청별 생성 상한) 또는 총 길이 ≥ --max-len (max_total_sequence_length)
  - advantage 0 = 샘플의 advantage 가 모두 0 — 그룹 16개의 보상이 같아 학습 신호가 없다 (DAPO dynamic sampling 이 거르는 샘플)
v2 RLVR 런에서 code_gen 판단(생성 토큰 65% · 64K 잘림 91% · 보상 > 0 0%)에 쓴 집계다 (`docs/RLVR_READINESS.md` §5.3).
사용: python analyze_dump_agent_stats.py --cap 65536 --max-len 131072 <dir>/exp_*/train_data_step*.jsonl
"""

import argparse
import collections
import json

import numpy as np


def agent_name(row):
    ref = row.get("agent_ref")
    if isinstance(ref, list):
        ref = ref[0] if ref else None
    return (ref or {}).get("name", "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, required=True, help="요청별 생성 상한 (토큰)")
    ap.add_argument(
        "--max-len", type=int, required=True, help="샘플 총 길이 상한 (토큰)"
    )
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    for path in args.files:
        stats = collections.defaultdict(collections.Counter)
        gen_lengths = []
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                mask = np.asarray(row["token_loss_mask"][0], dtype=np.int8)
                total = int(row["input_lengths"][0])
                edges = np.diff(np.concatenate([[0], mask, [0]]))
                calls = np.where(edges == -1)[0] - np.where(edges == 1)[0]
                adv = np.asarray(row["advantages"][0][:total], dtype=float)
                rewards = row["rewards"]
                reward = float(rewards[0] if isinstance(rewards, list) else rewards)
                zero = not np.any(adv != 0)
                s = stats[agent_name(row)]
                s["n"] += 1
                s["tok"] += total
                s["gen"] += int(mask.sum())
                s["trunc"] += int(
                    (len(calls) > 0 and calls.max() >= args.cap)
                    or total >= args.max_len
                )
                s["rew_pos"] += int(reward > 0)
                s["zero"] += int(zero)
                s["zero_tok"] += total if zero else 0
                gen_lengths.append(int(mask.sum()))

        tot = sum(s["tok"] for s in stats.values())
        gen = sum(s["gen"] for s in stats.values())
        n = sum(s["n"] for s in stats.values())
        g = np.asarray(gen_lengths)
        print(
            f"== {path.rsplit('/', 1)[-1]}: samples {n}, train tokens {tot / 1e6:.1f}M, gen tokens {gen / 1e6:.1f}M, "
            f"truncated {sum(s['trunc'] for s in stats.values()) / n:.1%}, zero-adv tokens "
            f"{sum(s['zero_tok'] for s in stats.values()) / tot:.1%}, gen/sample p50 {np.median(g):.0f} p90 "
            f"{np.percentile(g, 90):.0f} max {g.max()}"
        )
        print(
            f"{'agent':<56} {'n':>5} {'tok%':>6} {'gen%':>6} {'trunc%':>7} {'rew>0%':>7} {'zeroadv%':>9} {'zero_tok%':>9}"
        )
        for name, s in sorted(stats.items(), key=lambda kv: -kv[1]["tok"]):
            print(
                f"{name[:56]:<56} {s['n']:>5} {100 * s['tok'] / tot:>5.1f}% {100 * s['gen'] / gen:>5.1f}% "
                f"{100 * s['trunc'] / s['n']:>6.1f}% {100 * s['rew_pos'] / s['n']:>6.1f}% "
                f"{100 * s['zero'] / s['n']:>8.1f}% {100 * s['zero_tok'] / tot:>8.1f}%"
            )


if __name__ == "__main__":
    main()
