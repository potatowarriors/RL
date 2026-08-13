"""Alpha forward-parity gate: mcore (training-style forward) vs HF reference logits.

The HF reference artifact is produced in the environment where alpha's HF
modeling code is validated (Pai env, transformers 4.57):
  {"input_ids": LongTensor [1, s], "logits": FloatTensor [vocab]}  # last-token

mcore GDN rejects inference contexts, so this runs a plain training-style
forward — the same code path NeMo-RL uses for logprobs.

Pass bar (Megatron-Bridge parity-testing skill):
  next-token argmax must match, cosine similarity >= 0.99.

Run from the NeMo-RL repo root:
  uv run --locked --extra mcore python examples/configs/alpha/tools/verify_forward_parity.py \
      --hf-path /path/to/hfmodel_00NNNNN --ref /path/to/ref_logits.pt
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-path", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cosine-threshold", type=float, default=0.99)
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

        cos = F.cosine_similarity(mg_last[None], ref_logits[None]).item()
        max_diff = (mg_last - ref_logits).abs().max().item()
        mean_diff = (mg_last - ref_logits).abs().mean().item()
        mg_top5 = torch.topk(mg_last, 5)
        hf_top5 = torch.topk(ref_logits, 5)
        argmax_match = int(mg_last.argmax()) == int(ref_logits.argmax())
        top5_match = mg_top5.indices.tolist() == hf_top5.indices.tolist()
        ok = argmax_match and cos >= args.cosine_threshold
        all_pass = all_pass and ok

        print(
            f"[prompt {i}] seq_len={seq_len:5d} next={int(mg_last.argmax()):7d} (HF {int(ref_logits.argmax()):7d}) "
            f"argmax_match={argmax_match} top5_match={top5_match} cos={cos:.6f} "
            f"max_diff={max_diff:.4f} mean_diff={mean_diff:.4f} -> {'PASS' if ok else 'FAIL'}"
        )

    if all_pass:
        print("FORWARD PARITY VERIFIED")
        return 0
    print("FORWARD PARITY FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
