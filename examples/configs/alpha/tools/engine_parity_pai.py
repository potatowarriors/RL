"""Engine parity gate — Pai-Megatron 측(SFT 학습 엔진). Pai 환경(system python, Megatron-LM-251125 + megatron_patch)에서 실행.

Pai validator(examples/alpha/validate_mg_hf_full.py)의 모델 구성·체크포인트 로드를 그대로 쓰고, 병렬화만 SFT 와 같은 EP8 로 둔다.
모든 rank 에 같은 배치를 넣으므로 DP 리듀스 없이도 rank 마다 같은 per-sequence gradient 가 나온다.
eval 모드(드롭아웃·MoE aux loss 없음)로 forward → 다음 토큰 CE 평균 → backward → engine_parity_common.summarize_params.

실행 (Pai-Megatron-Patch/examples/alpha 에서, 8 GPU):
  readarray -t MA < <(python3 tools/alpha_config.py emit-megatron-flags --from-checkpoint <ckpt>/iter_00NNNNN)
  torchrun --nproc_per_node 8 <this> --mg-checkpoint <ckpt>/iter_00NNNNN --batch batch.pt --out-dir <dir> "${MA[@]}" \
      --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --expert-model-parallel-size 8 \
      --bf16 --micro-batch-size 1 --seq-length 2048 --no-load-optim --no-load-rng --ckpt-format torch_dist \
      --no-gradient-accumulation-fusion
"""

import os
import re
import sys

PAI_ALPHA = os.environ.get("PAI_ALPHA_DIR", "/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha")
sys.path.insert(0, PAI_ALPHA)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import validate_mg_hf_full as V  # noqa: E402  (경로 설정·modelopt 차단·model_provider)
import torch  # noqa: E402

from engine_parity_common import next_token_logprobs, summarize_params  # noqa: E402


def _extra_args(parser):
    g = parser.add_argument_group("engine parity")
    g.add_argument("--mg-checkpoint", required=True)
    g.add_argument("--batch", required=True)
    g.add_argument("--out-dir", required=True)
    return parser


def main():
    from megatron.core.enums import ModelType
    from megatron.training import get_args, get_model
    from megatron.training.checkpointing import load_checkpoint
    from megatron.training.initialize import initialize_megatron
    from megatron_patch.arguments import get_patch_args

    initialize_megatron(
        extra_args_provider=lambda p: _extra_args(get_patch_args(p)),
        args_defaults={"tokenizer_type": "NullTokenizer", "no_load_rng": True, "no_load_optim": True,
                       "use_legacy_models": False},
    )
    args = get_args()
    rank = torch.distributed.get_rank()
    model = get_model(V.model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=False)
    m = re.search(r"iter_(\d+)", args.mg_checkpoint)
    args.load, args.ckpt_step = os.path.dirname(args.mg_checkpoint), int(m.group(1))
    iteration, _ = load_checkpoint(model, None, None)
    model = model[0]
    model.eval()

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

    records = summarize_params(model.named_parameters(), input_ids, vocab_rows=args.padded_vocab_size)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(
        {"engine": "pai", "rank": rank, "iteration": iteration, "loss": float(loss.item()),
         "logprobs": lp.detach().cpu(), "max_alloc_gb": torch.cuda.max_memory_allocated() / 2**30, "records": records},
        os.path.join(args.out_dir, f"pai_rank{rank}.pt"),
    )
    if rank == 0:
        print(f"PAI DONE iteration={iteration} loss={loss.item():.6f} params={len(records)} "
              f"max_alloc={torch.cuda.max_memory_allocated()/2**30:.1f}GB", flush=True)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
