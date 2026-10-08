"""Gym 경로 rollout(vLLM HTTP) vs train(mcore) logprob 어긋남을 시퀀스·환경별로 분해한다 — CPU, NeMo-RL train_data_step*.jsonl 입력.

`analyze_rollout_logprob_gap.py` 가 전체 분포를 본다면, 이 도구는 "어느 시퀀스가 왜 크게 틀리는가"를 본다 (G3, 2026-10-07).
  - 시퀀스별 mult_prob_error(mean exp|Δ|, NeMo-RL 의 seq_logprob_error 와 같은 식)·k3 KL, agent 별 집계
  - 학습 구간(token_loss_mask 가 1 인 연속 구간 = 모델 호출 1회)별 오차 — 멀티턴이면 몇 번째 호출부터 틀리는지
  - 큰 오차(|Δ|>1) 토큰의 첫 위치와 그 근처 토큰 — `</think>`·`<tool_call>` 같은 특수 토큰 경계와 겹치는지

  python examples/configs/alpha/tools/analyze_gym_logprob_gap.py <dir>/exp_*/train_data_step1.jsonl \
      --tokenizer <hfmodel_00NNNNN> [--threshold 2.0] [--show 5]
"""

import argparse
import collections
import json

import numpy as np

SPECIAL = {0: "<|endoftext|>", 2: "<|im_start|>", 3: "<|im_end|>", 10: "<tool_call>", 11: "</tool_call>",
           12: "<tool_response>", 13: "</tool_response>", 14: "<think>", 15: "</think>"}


def segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """token_loss_mask 의 1 연속 구간 [s, e)."""
    out, s = [], None
    for i, m in enumerate(mask):
        if m and s is None:
            s = i
        elif not m and s is not None:
            out.append((s, i))
            s = None
    if s is not None:
        out.append((s, len(mask)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--tokenizer", default=None, help="주면 큰 오차 위치 주변 토큰을 디코딩해 보여 준다")
    ap.add_argument("--threshold", type=float, default=2.0, help="seq mult_prob_error 기준 (레시피 seq_logprob_error_threshold)")
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()
    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    for f in args.files:
        rows = [json.loads(line) for line in open(f)]
        # loss 안 마스킹(seq_logprob_error_in_loss) 런은 prev_logprob 패스를 건너뛰어 덤프의 prev_logprobs 가 0 이다 (2026-10-08 E1a)
        if all(
            not np.asarray(r["prev_logprobs"][0], float)[np.asarray(r["token_loss_mask"][0], bool)].any()
            for r in rows
            if any(r["token_loss_mask"][0])
        ):
            print(f"== {f}: SKIP — prev_logprobs 가 전부 0 (seq_logprob_error_in_loss 런의 덤프). 학습측 logprob 이 없어 오차를 낼 수 없다")
            continue
        per_agent = collections.defaultdict(list)
        bad = []
        for i, r in enumerate(rows):
            agent = r.get("agent_ref", [{}])[0]
            agent = agent.get("name") if isinstance(agent, dict) else str(agent)
            m = np.array(r["token_loss_mask"][0], bool)
            g = np.array(r["generation_logprobs"][0], float)
            p = np.array(r["prev_logprobs"][0], float)
            ids = np.array(r["token_ids"][0])
            idx = np.where(m)[0]
            if len(idx) == 0:
                continue
            d = p[idx] - g[idx]
            mpe = float(np.exp(np.abs(d)).mean())
            k3 = float((np.exp(d) - d - 1).mean())
            segs = segments(m)
            seg_mpe = [float(np.exp(np.abs(p[s:e] - g[s:e])).mean()) for s, e in segs]
            big = idx[np.abs(d) > 1.0]
            rec = {"i": i, "agent": agent, "len": int(r["input_lengths"][0]), "gen": int(len(idx)), "mpe": mpe, "k3": k3,
                   "n_seg": len(segs), "seg_mpe": [round(x, 3) for x in seg_mpe], "first_big": int(big[0]) if len(big) else None,
                   "gen_start": int(idx[0]), "frac_big": float(len(big) / len(idx))}
            per_agent[agent].append(rec)
            if mpe > args.threshold:
                bad.append((rec, ids, m))
        n = sum(len(v) for v in per_agent.values())
        print(f"== {f}: samples={n}, mult_prob_error>{args.threshold}: {len(bad)}")
        print(f"{'agent':72s} {'n':>3s} {'bad':>3s} {'mpe_med':>8s} {'k3_mean':>8s} {'multi_seg':>9s}")
        for a, v in sorted(per_agent.items(), key=lambda kv: -sum(x["mpe"] > args.threshold for x in kv[1])):
            print(f"{a:72s} {len(v):3d} {sum(x['mpe'] > args.threshold for x in v):3d} "
                  f"{np.median([x['mpe'] for x in v]):8.3f} {np.mean([x['k3'] for x in v]):8.5f} {sum(x['n_seg'] > 1 for x in v):9d}")
        for rec, ids, m in bad[: args.show]:
            print(f"-- bad sample {rec['i']} agent={rec['agent']} len={rec['len']} gen_tokens={rec['gen']} mpe={rec['mpe']:.2f} "
                  f"segments={rec['n_seg']} seg_mpe={rec['seg_mpe']} first|Δ|>1 at {rec['first_big']} (gen starts {rec['gen_start']}) "
                  f"frac|Δ|>1={rec['frac_big']:.3f}")
            fb = rec["first_big"]
            if fb is not None:
                lo, hi = max(0, fb - 12), min(len(ids), fb + 4)
                window = [SPECIAL.get(int(t), None) for t in ids[lo:hi]]
                specials = [(lo + k, w) for k, w in enumerate(window) if w]
                print(f"   special tokens near first big error [{lo},{hi}): {specials}")
                if tok is not None:
                    print("   text:", repr(tok.decode(ids[lo:hi].tolist())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
