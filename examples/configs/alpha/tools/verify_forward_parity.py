"""Alpha forward-parity gate: mcore (training-style forward) vs HF reference logits.

The HF reference artifact is produced in the environment where alpha's HF
modeling code is validated (Pai env, transformers 4.57):
  {"input_ids": LongTensor [1, s], "logits": FloatTensor [vocab]}  # last-token

mcore GDN rejects inference contexts, so this runs a plain training-style
forward — the same code path NeMo-RL uses for logprobs.

Pass bar (Megatron-Bridge parity-testing skill):
  next-token argmax must match, cosine similarity >= 0.99.
  When the bf16 HF reference ties exactly at its maximum (2 tokens), either token
  counts as a matching argmax — the rule verify_vllm_serving_parity.py has used
  since 2026-08-24. Exact ties only, no tolerance; a reference whose maximum is
  shared by more than MAX_REFERENCE_TIES tokens is degenerate and fails.
KL(HF||mcore) and the mean logit shift are printed for the record (not judged here).

Run from the NeMo-RL repo root:
  uv run --locked --extra mcore python examples/configs/alpha/tools/verify_forward_parity.py \
      --hf-path /path/to/hfmodel_00NNNNN --ref /path/to/ref_logits.pt
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F


# A 2-way tie is what bf16 rounding produces at the top of a close call (1 of the 15
# references saved up to 2026-11-27). A larger tie set means a coarse or constant
# reference, against which any output would "match".
MAX_REFERENCE_TIES = 2


def judge_prompt(
    mg_last: torch.Tensor, ref_logits: torch.Tensor, cosine_threshold: float
) -> dict:
    """Judge mcore last-position logits against the HF reference (float32, [vocab] each).

    Returns the verdict (`ok`) and the numbers that are printed. NaN in either input
    fails: the tie set is empty or the cosine is NaN.
    """
    cos = F.cosine_similarity(mg_last[None], ref_logits[None]).item()
    diff = mg_last - ref_logits
    hf_lp = F.log_softmax(ref_logits, dim=-1)
    mg_lp = F.log_softmax(mg_last, dim=-1)
    kl = float((hf_lp.exp() * (hf_lp - mg_lp)).sum())
    # bf16 logits can tie at the max (prompt 2: ' the' vs " Einstein's", both 41.75 on
    # teacher_swe_e2e/hf_s4/hf_step_00040); torch.argmax then returns the first index only.
    hf_tie_set = (ref_logits == ref_logits.max()).nonzero().flatten().tolist()
    degenerate = len(hf_tie_set) > MAX_REFERENCE_TIES
    mg_argmax = int(mg_last.argmax())
    argmax_match = (not degenerate) and mg_argmax in hf_tie_set
    top5_match = (
        torch.topk(mg_last, 5).indices.tolist()
        == torch.topk(ref_logits, 5).indices.tolist()
    )
    # On a tie or a mismatch, show both sides' logits at the candidate tokens so the log
    # alone tells a tie-break from a real disagreement.
    tie_note = ""
    if degenerate:
        tie_note = f" hf_max_ties={len(hf_tie_set)} tokens (degenerate reference)"
    elif len(hf_tie_set) > 1 or not argmax_match:
        cand = sorted(set(hf_tie_set) | {mg_argmax})
        tie_note = f" hf_max_ties={hf_tie_set} logits(hf/mcore)=" + ",".join(
            f"{t}:{ref_logits[t].item():.4f}/{mg_last[t].item():.4f}" for t in cand
        )
    return {
        "ok": bool(argmax_match and cos >= cosine_threshold),
        "mg_argmax": mg_argmax,
        "hf_argmax": int(ref_logits.argmax()),
        "argmax_match": argmax_match,
        "top5_match": top5_match,
        "tie_note": tie_note,
        "cos": cos,
        "max_diff": diff.abs().max().item(),
        "mean_diff": diff.abs().mean().item(),
        "kl": kl,
        "shift": diff.mean().item(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-path", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cosine-threshold", type=float, default=0.99)
    ap.add_argument(
        "--dump",
        help="Optional path: save the mcore last-position logits of every prompt "
        '(list of {"input_ids", "logits"}, same order as --ref) for offline analysis. Not used for the verdict.',
    )
    args = ap.parse_args()

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29519")
    os.environ.setdefault("LOCAL_RANK", "0")

    loaded = torch.load(args.ref, map_location="cpu", weights_only=True)
    refs = loaded if isinstance(loaded, list) else [loaded]
    print(f"loaded {len(refs)} reference prompt(s)")

    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(args.hf_path, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=True)
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.expert_model_parallel_size = 1
    provider.gradient_accumulation_fusion = False  # no apex on bare metal
    provider.finalize()
    model = provider.provide_distributed_model(wrap_with_ddp=False)
    m = model[0] if isinstance(model, list) else model
    m.eval()

    all_pass = True
    mg_dump = []
    for i, ref in enumerate(refs):
        input_ids = ref["input_ids"].cuda()
        ref_logits = ref["logits"].float()
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)
        with torch.no_grad():
            out = m(input_ids=input_ids, position_ids=position_ids, attention_mask=None)

        # mcore GPTModel returns logits [b, s, vocab] when labels are absent
        logits = out[0] if isinstance(out, tuple) else out
        if logits.dim() != 3:
            raise RuntimeError(f"unexpected logits shape {tuple(logits.shape)}")
        if logits.shape[0] != 1:  # [s, b, v] layout
            logits = logits.transpose(0, 1)
        mg_last = logits[0, -1, : ref_logits.numel()].float().cpu()

        mg_dump.append({"input_ids": ref["input_ids"].cpu(), "logits": mg_last.clone()})
        j = judge_prompt(mg_last, ref_logits, args.cosine_threshold)
        all_pass = all_pass and j["ok"]

        print(
            f"[prompt {i}] seq_len={seq_len:5d} next={j['mg_argmax']:7d} (HF {j['hf_argmax']:7d}) "
            f"argmax_match={j['argmax_match']}{j['tie_note']} top5_match={j['top5_match']} cos={j['cos']:.6f} "
            f"max_diff={j['max_diff']:.4f} mean_diff={j['mean_diff']:.4f} "
            f"KL(HF||mcore)={j['kl']:.6f} shift={j['shift']:+.4f} -> {'PASS' if j['ok'] else 'FAIL'}"
        )

    if args.dump:
        torch.save(mg_dump, args.dump)
        print(f"SAVED mcore logits -> {args.dump}")
    if all_pass:
        print("FORWARD PARITY VERIFIED")
        return 0
    print("FORWARD PARITY FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
