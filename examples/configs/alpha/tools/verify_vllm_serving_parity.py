"""Quantitative serving-parity gate: vLLM disk-loaded alpha vs HF reference logits.

Compares the FULL-VOCAB next-token log-probability distribution of the
vLLM-served model (weights loaded from the HF checkpoint on disk,
load_format=auto) against the HF reference logits generated in the Pai
environment (same artifact used by verify_forward_parity.py). Reference
input token ids are injected directly (no tokenizer round-trip).

Pass bar (same standard as the forward-parity gate):
  argmax match on every prompt AND cosine >= 0.99 over full-vocab logprobs.
KL(HF||vLLM) and abs-diff stats are reported against the established bf16
noise floor (refit gate: mean ~0.02).

Run from the NeMo-RL repo root:
  uv run --locked --extra vllm python \
      examples/configs/alpha/tools/verify_vllm_serving_parity.py \
      --hf-path <hfmodel_00NNNNN> --ref <ref_logits.pt>
"""

import argparse
import sys

# The NeMo-RL lock pins openai 2.6.1; vllm 0.25.1's tool_parsers import a few
# newer symbols that plain generation never uses. Harmless shim (PEP 562).
import openai.types.responses as _r

_r.__getattr__ = lambda name: type(name, (), {})  # noqa: E731

import torch
import torch.nn.functional as F
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-path", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cosine-threshold", type=float, default=0.99)
    args = ap.parse_args()

    loaded = torch.load(args.ref, map_location="cpu", weights_only=True)
    refs = loaded if isinstance(loaded, list) else [loaded]
    vocab = refs[0]["logits"].numel()
    print(f"loaded {len(refs)} reference prompt(s), vocab={vocab}")

    llm = LLM(
        model=args.hf_path,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        max_logprobs=vocab,  # full-vocab distribution for the gate
    )
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=vocab)

    prompts = [TokensPrompt(prompt_token_ids=r["input_ids"][0].tolist()) for r in refs]
    outs = llm.generate(prompts, sp)

    all_pass = True
    for i, (ref, out) in enumerate(zip(refs, outs)):
        hf_lp = F.log_softmax(ref["logits"].float(), dim=-1)
        lp_dict = out.outputs[0].logprobs[0]
        vllm_lp = torch.full((vocab,), float("-inf"))
        for tid, lp in lp_dict.items():
            vllm_lp[tid] = lp.logprob
        covered = torch.isfinite(vllm_lp)
        n_cov = int(covered.sum())

        hf_c, vl_c = hf_lp[covered], vllm_lp[covered]
        cos = F.cosine_similarity(hf_c[None], vl_c[None]).item()
        d = (hf_c - vl_c).abs()
        p_hf = hf_lp.exp()
        kl = float((p_hf[covered] * (hf_c - vl_c)).sum())
        hf_top5 = torch.topk(hf_lp, 5).indices.tolist()
        vl_top5 = torch.topk(vllm_lp, 5).indices.tolist()
        # bf16 logits can tie at the max (observed: ' the' vs " Einstein's" both
        # 38.0000); accept any member of the HF max tie-set as a matching argmax.
        hf_tie_set = (ref["logits"] == ref["logits"].max()).nonzero().flatten().tolist()
        argmax_match = vl_top5[0] in hf_tie_set
        ok = argmax_match and cos >= args.cosine_threshold
        all_pass = all_pass and ok

        tie_note = f" hf_max_ties={hf_tie_set}" if len(hf_tie_set) > 1 else ""
        print(
            f"[prompt {i}] seq={ref['input_ids'].shape[1]:5d} covered={n_cov}/{vocab} "
            f"argmax={vl_top5[0]} (HF {hf_top5[0]}) match={argmax_match}{tie_note} "
            f"top5_match={hf_top5 == vl_top5} cos={cos:.6f} "
            f"KL(HF||vLLM)={kl:.6f} max|d|={d.max():.4f} mean|d|={d.mean():.4f} "
            f"-> {'PASS' if ok else 'FAIL'}"
        )

    print("SERVING PARITY VERIFIED" if all_pass else "SERVING PARITY FAILED")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
