"""Engine parity gate — HF 기준점(Pai 환경 transformers 4.57, modeling_alpha.py). 같은 배치의 다음 토큰 logprob 를 낸다.

Pai-Megatron ↔ NeMo-RL 차이가 클 때, 어느 쪽이 기준에서 벗어났는지 가리기 위한 세 번째 점(삼각 비교).
실행 (Pai 환경, GPU 1장): python3 engine_parity_hf.py --hf-path <hfmodel> --batch batch.pt --out hf_logprobs.pt
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_parity_common import next_token_logprobs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-path", required=True)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.hf_path, dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    ids = torch.load(args.batch, map_location="cpu")["input_ids"].cuda()
    lps = []
    with torch.no_grad():
        for i in range(ids.shape[0]):  # 시퀀스별(메모리)
            logits = model(ids[i : i + 1]).logits
            lps.append(next_token_logprobs(logits, ids[i : i + 1]).cpu())
            del logits
    lp = torch.cat(lps, 0)
    torch.save({"engine": "hf", "logprobs": lp, "loss": float(-lp.mean())}, args.out)
    print(f"HF DONE loss={-lp.mean().item():.6f} shape={tuple(lp.shape)}", flush=True)


if __name__ == "__main__":
    main()
