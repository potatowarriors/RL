"""Generate HF reference logits for verify_forward_parity.py.

MUST run in the Pai environment (system python, transformers 4.57 - the
environment where modeling_alpha.py is validated; 5.x breaks its imports):
  LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:/usr/local/cuda-12.8/targets/x86_64-linux/lib" \
      python3 examples/configs/alpha/tools/gen_hf_reference_logits.py
Edit H (checkpoint) and OUT (artifact path) below per run.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

H = "/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha/outputs/diloco_stage2/node0/hfmodel_0020000"
OUT = "/tmp/claude-1100/-home-work-vidsearch-repos-project-s/d910f483-b698-409c-83a4-3ba4269f03bf/scratchpad/alpha_hf_ref_multi.pt"

PROMPTS = [
    "The capital of France is",
    "\ub300\ud55c\ubbfc\uad6d\uc758 \uc218\ub3c4\ub294 \uc11c\uc6b8\uc774\uace0, \uc77c\ubcf8\uc758 \uc218\ub3c4\ub294",
    ("The theory of general relativity describes gravity as the curvature of spacetime caused by mass and energy. "
     "In 1915, Einstein published his field equations, which relate the geometry of spacetime to the distribution of matter. "
     "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\n") * 12
     + "In summary, the most important equation in general relativity is",
]

tok = AutoTokenizer.from_pretrained(H, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(H, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
refs = []
for p in PROMPTS:
    ids = tok(p, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        out = model(ids.cuda())
    last = out.logits[0, -1, :].float().cpu()
    print(f"len={ids.shape[1]:5d} next={int(last.argmax()):7d} {tok.decode([int(last.argmax())])!r}")
    refs.append({"prompt": p, "input_ids": ids, "logits": last})
torch.save(refs, OUT)
print("SAVED", OUT)
