"""Engine parity gate — NeMo-RL 측(RL 학습 엔진). NeMo-RL Megatron 워커 venv 에서 실행.

MegatronPolicyWorker.__init__ 의 setup 순서(set_device → setup_distributed → validate_model_paths → handle_model_import →
validate_and_set_config → setup_model_and_optimizer)를 그대로 쓰되 optimizer 는 만들지 않는다(DDP 래핑 없음 → param.grad).
RL 레시피 설정(moe_router_dtype·rope fusion 등)이 그대로 적용된 모델로 Pai 측과 같은 배치·같은 손실을 계산한다.

실행 (NeMo-RL 루트, 8 GPU, CUDA_VISIBLE_DEVICES 미지정):
  $NRL_ROOT/clean_run.sh $W/bin/python -m torch.distributed.run --nproc_per_node 8 \
      examples/configs/alpha/tools/engine_parity_nemorl.py --config examples/configs/alpha/grpo_alpha_smoke_muon.yaml \
      --batch batch.pt --out-dir <dir> [--hf-path <hfmodel>] </dev/null
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_parity_common import next_token_logprobs, summarize_params  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hf-path", default=None)
    args = ap.parse_args()

    from omegaconf import OmegaConf

    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

    register_omegaconf_resolvers()
    cfg = OmegaConf.to_container(load_config(args.config), resolve=True)
    policy = cfg["policy"]
    if args.hf_path:
        policy["model_name"] = args.hf_path
        policy["tokenizer"]["name"] = args.hf_path
    policy.setdefault("generation", {}).setdefault("colocated", {"enabled": True, "resources": {}})
    policy["megatron_cfg"].setdefault("train_iters", 10)

    from nemo_rl.models.megatron.setup import (
        handle_model_import,
        setup_distributed,
        setup_model_and_optimizer,
        validate_and_set_config,
        validate_model_paths,
    )

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    setup_distributed()
    rank = torch.distributed.get_rank()
    hf_model_name, pretrained_path, pt_exists = validate_model_paths(policy)
    handle_model_import(policy, hf_model_name, pretrained_path, pt_exists)
    rt = validate_and_set_config(policy, rank, hf_model_name, pretrained_path, None, None)
    rt.megatron_cfg.validate()
    state = setup_model_and_optimizer(policy, rt.megatron_cfg, False, load_weights=True)
    model = state.model[0] if isinstance(state.model, list) else state.model
    model.eval()

    vocab_rows = next(p.shape[0] for n, p in model.named_parameters() if "word_embeddings" in n)
    batch = torch.load(args.batch, map_location="cpu")
    input_ids = batch["input_ids"].cuda()
    position_ids = torch.arange(input_ids.shape[1], device="cuda").unsqueeze(0).expand_as(input_ids)
    with torch.enable_grad():
        logits = model(input_ids, position_ids, None)
        lp = next_token_logprobs(logits, input_ids)
        loss = -lp.mean()
        loss.backward()
    del logits
    torch.cuda.synchronize()

    records = summarize_params(model.named_parameters(), input_ids, vocab_rows=vocab_rows)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(
        {"engine": "nemorl", "rank": rank, "loss": float(loss.item()), "logprobs": lp.detach().cpu(),
         "max_alloc_gb": torch.cuda.max_memory_allocated() / 2**30, "records": records,
         "moe_router_dtype": policy["megatron_cfg"].get("moe_router_dtype")},
        os.path.join(args.out_dir, f"nemorl_rank{rank}.pt"),
    )
    if rank == 0:
        print(f"NEMORL DONE loss={loss.item():.6f} params={len(records)} "
              f"max_alloc={torch.cuda.max_memory_allocated()/2**30:.1f}GB", flush=True)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
