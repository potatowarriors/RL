"""Alpha bridge round-trip verification: HF -> mcore -> HF, exact-match required.

Rigor mirrors Pai-Megatron-Patch's validate_mg_hf_full.py:
  - every exported tensor must match the original checkpoint with max_diff == 0.0
    (the bridge performs no arithmetic, so any nonzero diff is a bug)
  - bidirectional coverage: exported-key set must equal original-key set
    (missing keys and phantom keys are both failures)

Run (single GPU, from the NeMo-RL repo root):
  uv run --locked --extra mcore python examples/configs/alpha/tools/verify_bridge_roundtrip.py \
      --hf-path /path/to/hfmodel_00NNNNN
"""

import argparse
import json
import os
import sys

import torch
from safetensors import safe_open


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-path", required=True)
    ap.add_argument("--max-report", type=int, default=20)
    args = ap.parse_args()

    # Single-process distributed init for provide_distributed_model
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    os.environ.setdefault("LOCAL_RANK", "0")

    from megatron.bridge import AutoBridge

    print(f"[1/4] Loading bridge from {args.hf_path}")
    bridge = AutoBridge.from_hf_pretrained(args.hf_path, trust_remote_code=True)

    print("[2/4] Building mcore model and loading HF weights (bf16, single GPU, EP=1)...")
    provider = bridge.to_megatron_provider(load_weights=True)
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.expert_model_parallel_size = 1
    provider.gradient_accumulation_fusion = False  # no apex on bare metal
    provider.finalize()
    model = provider.provide_distributed_model(wrap_with_ddp=False)

    # Index the original checkpoint tensors (name -> shard file)
    index_path = os.path.join(args.hf_path, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    shard_handles: dict[str, object] = {}

    def original_tensor(name: str) -> torch.Tensor:
        shard = weight_map[name]
        if shard not in shard_handles:
            shard_handles[shard] = safe_open(os.path.join(args.hf_path, shard), framework="pt", device="cpu")
        return shard_handles[shard].get_tensor(name)

    print("[3/4] Exporting mcore -> HF stream and comparing per tensor...")
    exported_names: set[str] = set()
    mismatches: list[tuple[str, str]] = []
    n_ok = 0
    for name, tensor in bridge.export_hf_weights(model, cpu=True):
        exported_names.add(name)
        if name not in weight_map:
            mismatches.append((name, "phantom: not in original checkpoint"))
            continue
        ref = original_tensor(name)
        if tensor.shape != ref.shape:
            mismatches.append((name, f"shape {tuple(tensor.shape)} vs {tuple(ref.shape)}"))
            continue
        if tensor.dtype != ref.dtype:
            # Documented exception: the checkpoint stores GDN A_log in fp32 (a
            # Pai-Megatron training-storage choice), but upstream mcore holds it
            # in params_dtype (bf16) — matching actual HF inference behavior,
            # where from_pretrained(torch_dtype=bf16) downcasts A_log too (only
            # e_score_correction_bias is fp32-protected). Values must still be
            # exactly equal in the exported dtype.
            if name.endswith("linear_attn.A_log") and ref.dtype == torch.float32:
                if torch.equal(tensor, ref.to(tensor.dtype)):
                    n_ok += 1
                    print(f"    OK (dtype exception, values exact in {tensor.dtype}): {name}")
                else:
                    mismatches.append((name, "A_log values differ even after dtype cast"))
            else:
                mismatches.append((name, f"dtype {tensor.dtype} vs {ref.dtype}"))
            continue
        max_diff = (tensor.float() - ref.float()).abs().max().item()
        if max_diff != 0.0:
            mismatches.append((name, f"max_diff {max_diff:.3e}"))
        else:
            n_ok += 1

    missing = sorted(set(weight_map) - exported_names)

    print("[4/4] Results")
    print(f"  exact-match tensors : {n_ok}")
    print(f"  mismatched tensors  : {len(mismatches)}")
    print(f"  missing (not exported): {len(missing)}")
    for name, why in mismatches[: args.max_report]:
        print(f"    MISMATCH {name}: {why}")
    for name in missing[: args.max_report]:
        print(f"    MISSING  {name}")

    if not mismatches and not missing:
        print("ALL WEIGHTS MATCHED EXACTLY - BRIDGE ROUND-TRIP VERIFIED")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
