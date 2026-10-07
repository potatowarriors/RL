"""Engine parity gate — 오프라인 비교(CPU). engine_parity_{pai,nemorl}.py 의 rank 별 덤프를 읽어 판정한다.

forward: 같은 배치의 다음 토큰 logprob(전 시퀀스) 차이 — mean|Δ|·max|Δ|·k3 KL·위치별.
backward: 가중치 지문으로 짝지은 텐서마다 gradient 비교 — 전체를 담은 텐서는 cos·상대 L2, 나머지는 norm 비.
범주는 이름으로 나눈다(양쪽 이름이 달라 Pai 이름 기준). 레이아웃이 달라 짝이 안 지어진 텐서는 미매칭으로 집계한다.

사용: python engine_parity_compare.py --pai-dir <dir>/pai --nemorl-dir <dir>/nemorl [--out report.json]
"""

import argparse
import glob
import json
import math
from collections import defaultdict

import torch


def load(d, prefix):
    files = sorted(glob.glob(f"{d}/{prefix}_rank*.pt"))
    assert files, f"no dumps in {d}"
    dumps = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    by_key, dup = {}, 0
    for dmp in dumps:
        for r in dmp["records"]:
            k = tuple(r["key"][:2]) + tuple(r["key"][2:])
            if k in by_key:
                dup += 1
                continue
            by_key[k] = r
    r0 = next(x for x in dumps if x["rank"] == 0)
    return {"by_key": by_key, "dup": dup, "loss": r0["loss"], "logprobs": r0["logprobs"],
            "max_alloc_gb": max(x["max_alloc_gb"] for x in dumps), "n_ranks": len(dumps)}


def category(name):
    n = name.lower()
    for k, c in (("word_embeddings", "embedding"), ("output_layer", "output"), ("final", "final_norm"),
                 ("router", "router"), ("shared_expert", "shared_experts"), ("experts", "experts"),
                 ("linear_qkv", "attn_qkv"), ("linear_qgkv", "attn_qkv"), ("linear_proj", "attn_out"),
                 ("q_layernorm", "attn_qk_norm"), ("k_layernorm", "attn_qk_norm"),
                 ("in_proj", "gdn_in_proj"), ("out_proj", "gdn_out_proj"), ("conv1d", "gdn_conv1d"),
                 ("a_log", "gdn_A_log"), ("dt_bias", "gdn_dt_bias"), ("norm", "norm")):
        if k in n:
            return c
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pai-dir", required=True)
    ap.add_argument("--nemorl-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    P, N = load(args.pai_dir, "pai"), load(args.nemorl_dir, "nemorl")

    # forward
    a, b = P["logprobs"].double(), N["logprobs"].double()
    d = b - a
    k3 = (torch.exp(d) - d - 1)
    L = d.shape[1]
    bins = [0, 256, 512, 1024, 1536, L]
    fwd = {"loss_pai": P["loss"], "loss_nemorl": N["loss"], "loss_rel_diff": abs(N["loss"] - P["loss"]) / abs(P["loss"]),
           "mean_abs_dlogprob": float(d.abs().mean()), "p99_abs_dlogprob": float(d.abs().flatten().quantile(0.99)),
           "max_abs_dlogprob": float(d.abs().max()), "k3_kl": float(k3.mean()),
           "k3_by_position": {f"[{bins[i]},{bins[i+1]})": float(k3[:, bins[i]:bins[i+1]].mean()) for i in range(len(bins) - 1)}}

    # backward
    keys_p, keys_n = set(P["by_key"]), set(N["by_key"])
    matched = keys_p & keys_n
    cat = defaultdict(lambda: {"n": 0, "numel": 0, "full_n": 0, "cos": [], "rel_l2": [], "norm_ratio": [], "no_grad": 0})
    worst = []
    for k in matched:
        rp, rn = P["by_key"][k], N["by_key"][k]
        c = cat[category(rp["name"])]
        c["n"] += 1
        c["numel"] += rp["numel"]
        if rp.get("grad_norm") is None or rn.get("grad_norm") is None:
            c["no_grad"] += 1
            continue
        if rp["grad_norm"] > 0:
            c["norm_ratio"].append(rn["grad_norm"] / rp["grad_norm"])
        gp, gn = rp.get("grad"), rn.get("grad")
        if gp is not None and gn is not None and gp.shape == gn.shape:
            gp, gn = gp.double().flatten(), gn.double().flatten()
            denom = gp.norm() * gn.norm()
            cos = float((gp @ gn) / denom) if denom > 0 else float("nan")
            rel = float((gn - gp).norm() / gp.norm()) if gp.norm() > 0 else float("nan")
            c["full_n"] += 1
            c["cos"].append(cos)
            c["rel_l2"].append(rel)
            worst.append((cos, rel, rp["name"], rn["name"]))
    summ = {}
    for name, c in sorted(cat.items()):
        def q(v, f):
            v = [x for x in v if not math.isnan(x)]
            return f(v) if v else None
        summ[name] = {"matched": c["n"], "numel_M": round(c["numel"] / 1e6, 2), "full_compared": c["full_n"],
                      "no_grad": c["no_grad"],
                      "cos_min": q(c["cos"], min), "cos_median": q(c["cos"], lambda v: sorted(v)[len(v) // 2]),
                      "rel_l2_max": q(c["rel_l2"], max),
                      "norm_ratio_median": q(c["norm_ratio"], lambda v: sorted(v)[len(v) // 2]),
                      "norm_ratio_min": q(c["norm_ratio"], min), "norm_ratio_max": q(c["norm_ratio"], max)}
    unmatched_p = defaultdict(lambda: [0, 0])
    for k in keys_p - matched:
        r = P["by_key"][k]
        unmatched_p[category(r["name"])][0] += 1
        unmatched_p[category(r["name"])][1] += r["numel"]
    tot_p = sum(r["numel"] for r in P["by_key"].values())
    mat_numel = sum(P["by_key"][k]["numel"] for k in matched)
    worst.sort()
    report = {
        "forward": fwd,
        "coverage": {"pai_tensors": len(keys_p), "nemorl_tensors": len(keys_n), "matched": len(matched),
                     "matched_numel_frac_of_pai": mat_numel / tot_p,
                     "unmatched_pai_by_category": {k: {"n": v[0], "numel_M": round(v[1] / 1e6, 2)} for k, v in unmatched_p.items()}},
        "backward_by_category": summ,
        "worst_cos_10": [{"cos": c, "rel_l2": r, "pai": a_, "nemorl": b_} for c, r, a_, b_ in worst[:10]],
        "max_alloc_gb": {"pai": P["max_alloc_gb"], "nemorl": N["max_alloc_gb"]},
    }
    print(json.dumps(report, indent=1, default=str))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1, default=str)


if __name__ == "__main__":
    main()
