"""Alpha RL 학습 스텝 메모리·처리량 실측 — NeMo-RL 의 실제 학습 경로(MegatronPolicyWorkerImpl.train + ClippedPGLossFn + Muon step).

Ray 없이 torchrun 프로세스에서 워커 구현 클래스를 직접 만든다(ray.get_gpu_ids 만 LOCAL_RANK 로 대체).
학습 전용 노드(분리 토폴로지)를 가정해 vLLM 없이 잰다. 1 스텝째는 Muon momentum 지연 할당·커널 autotune 을 포함하므로
처리량은 2 스텝째 이후로 본다. 게이트 R4(메모리)·R5(recompute·offload 처리량), 결과·해석은 docs/GATES.md.

실행 (NeMo-RL 루트, 8 GPU, CUDA_VISIBLE_DEVICES 미지정). 환경변수는 clean_run.sh 의 whitelist 안쪽(/usr/bin/env)으로 넘긴다:
  $NRL_ROOT/clean_run.sh /usr/bin/env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True [MEM_SNAPSHOT_DIR=<dir>] \
      $W/bin/python -m torch.distributed.run --nproc_per_node 8 \
      examples/configs/alpha/tools/measure_train_memory.py --config examples/configs/alpha/grpo_alpha_smoke_muon.yaml \
      --seq-len 131072 --cp 8 --logprob-chunk-size 2048 --fuse-loss [--samples-per-dp 4] \
      [--recompute selective --recompute-modules layernorm,moe] [--r3] [--set megatron_cfg.apply_rope_fusion=true] \
      --out mem.json </dev/null
CP>1 은 packing 을 자동으로 켠다. --fused-linear-logprobs 는 CP1 전용이다.

--samples-per-dp N: DP 랭크당 N 샘플 = 옵티마이저 스텝당 마이크로배치 N 개. 옵티마이저 비용을 분리하려면 N=1 과 N>1 을 같이 잰다.
--r3: 합성 routed_experts(토큰·층마다 서로 다른 expert 8개, 균등 무작위)를 넣어 Router Replay 경로(재계산 시 REPLAY_BACKWARD 포함)를 태운다.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch


def gb(x):
    return round(x / 2**30, 2)


def _set_dotted(cfg: dict, dotted: str, raw: str) -> None:
    """`--set a.b.c=value` → cfg["a"]["b"]["c"] = yaml-parsed value."""
    import yaml

    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = yaml.safe_load(raw)


def _synthetic_routes(num_tokens: int, num_layers: int, topk: int, num_experts: int, num_groups: int,
                      group_topk: int, g: torch.Generator) -> torch.Tensor:
    """[num_tokens, num_layers, topk] int16. alpha 의 그룹 제한 라우팅 모양을 흉내 낸다.

    토큰·층마다 그룹 group_topk 개를 고르고 그룹마다 서로 다른 expert topk/group_topk 개를 고른다.
    alpha(192 experts, 8 그룹×24, top-4 그룹, top-8)에서 그룹 하나 = EP8 랭크 하나라, 토큰은 실제처럼 최대 4 랭크로 간다.
    """
    n = num_tokens * num_layers
    per_group = num_experts // num_groups
    k_in_group = topk // group_topk
    assert k_in_group * group_topk == topk and k_in_group <= per_group
    groups = torch.rand(n, num_groups, generator=g).topk(group_topk, dim=-1).indices  # [n, group_topk]
    offs = torch.rand(n, group_topk, per_group, generator=g).topk(k_in_group, dim=-1).indices  # [n, gk, k]
    idx = (groups.unsqueeze(-1) * per_group + offs).view(n, topk).to(torch.int16)
    return idx.view(num_tokens, num_layers, topk)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--cp", type=int, required=True)
    ap.add_argument("--fused-linear-logprobs", action="store_true")  # CP>1 와 비호환(NeMo-RL assert) — CP1 전용
    ap.add_argument("--logprob-chunk-size", type=int, default=None)  # Ultra 레시피 2048 (defer_fp32_logits 필요)
    ap.add_argument("--fuse-loss", action="store_true")  # sequence_packing.fuse_loss (Ultra 레시피 true)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--samples-per-dp", type=int, default=1)
    ap.add_argument("--recompute", choices=["full", "selective", "none"], default=None,
                    help="미지정이면 레시피 값(alpha 스모크 = full)")
    ap.add_argument("--recompute-modules", default=None, help="selective 일 때 쉼표 구분 (예: layernorm,moe)")
    ap.add_argument("--r3", action="store_true", help="합성 routed_experts 로 Router Replay 경로를 켠다")
    ap.add_argument("--set", action="append", default=[], metavar="DOTTED=VALUE",
                    help="policy 설정 덮어쓰기 (예: megatron_cfg.apply_rope_fusion=true)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import ray

    local_rank = int(os.environ["LOCAL_RANK"])
    ray.get_gpu_ids = lambda: [local_rank]  # 워커 __init__ 의 물리 GPU 조회를 torchrun 로 대체

    from omegaconf import OmegaConf

    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

    register_omegaconf_resolvers()
    cfg = OmegaConf.to_container(load_config(args.config), resolve=True)
    world = int(os.environ["WORLD_SIZE"])
    dp = world // args.cp
    pol = cfg["policy"]
    L = args.seq_len
    B = dp * args.samples_per_dp
    pol["max_total_sequence_length"] = L
    pol["train_global_batch_size"] = B
    pol["train_micro_batch_size"] = 1
    pol["logprob_batch_size"] = 1
    pol["make_sequence_length_divisible_by"] = 2 * args.cp
    pol["megatron_cfg"]["context_parallel_size"] = args.cp
    if args.cp > 1:
        # NeMo-RL Megatron 경로는 CP 에 sequence packing(THD) 을 요구한다(setup.py _apply_parallelism_config)
        pol["sequence_packing"]["enabled"] = True
        pol["sequence_packing"]["train_mb_tokens"] = L
        pol["sequence_packing"]["logprob_mb_tokens"] = L
    pol["megatron_cfg"]["use_fused_linear_logprobs"] = bool(args.fused_linear_logprobs)
    pol["logprob_chunk_size"] = args.logprob_chunk_size
    pol["sequence_packing"]["fuse_loss"] = bool(args.fuse_loss)
    pol["megatron_cfg"].setdefault("train_iters", 10)
    if args.recompute == "none":
        pol["megatron_cfg"]["activation_checkpointing"] = False
    elif args.recompute is not None:
        pol["megatron_cfg"]["activation_checkpointing"] = True
        pol["megatron_cfg"]["recompute_granularity"] = args.recompute
        if args.recompute == "selective":
            assert args.recompute_modules, "--recompute selective 에는 --recompute-modules 가 필요하다"
            pol["megatron_cfg"]["recompute_modules"] = args.recompute_modules.split(",")
    pol["router_replay"] = {"enabled": bool(args.r3)}
    pol["generation"]["colocated"]["enabled"] = False  # 학습 전용 노드
    pol["generation"]["max_new_tokens"] = L
    for item in args.set:
        dotted, _, raw = item.partition("=")
        _set_dotted(pol, dotted, raw)

    from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossConfig, ClippedPGLossFn
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.distributed.named_sharding import NamedSharding
    from nemo_rl.models.policy.workers.megatron_policy_worker import MegatronPolicyWorkerImpl

    sharding = NamedSharding(layout=np.arange(world).reshape(1, -1, args.cp, 1),
                             names=["pipeline_parallel", "data_parallel", "context_parallel", "tensor_parallel"])
    tok = get_tokenizer(pol["tokenizer"])
    t0 = time.time()
    worker = MegatronPolicyWorkerImpl(pol, tok, init_optimizer=True, init_reference_model=False,
                                      worker_sharding_annotations=sharding)
    rank = torch.distributed.get_rank()
    torch.cuda.synchronize()
    floor_alloc = torch.cuda.memory_allocated()
    setup_s = time.time() - t0
    mcfg = worker.model.config
    rc = {"granularity": mcfg.recompute_granularity, "method": mcfg.recompute_method,
          "num_layers": mcfg.recompute_num_layers, "modules": list(mcfg.recompute_modules or [])}

    # 옵티마이저 스텝 시간을 분리해 잰다 (앞뒤 동기화 — 측정 오버헤드는 스텝당 수 ms)
    opt_times: list[float] = []
    real_step = worker.optimizer.step

    def timed_step(*a, **kw):
        torch.cuda.synchronize()
        ts = time.time()
        out = real_step(*a, **kw)
        torch.cuda.synchronize()
        opt_times.append(time.time() - ts)
        return out

    worker.optimizer.step = timed_step

    # grpo.py 와 같이 pydantic 설정 객체로 넘긴다(master_config.loss_fn)
    loss_fn = ClippedPGLossFn(ClippedPGLossConfig(**cfg["loss_fn"]),
                              use_fused_linear_logprobs=bool(args.fused_linear_logprobs))
    # lm_policy.train 과 같은 경로: 전 rank 가 같은 전체 배치(gbs = B 표본)를 만들고 DP 로 샤딩, 자기 DP 샤드를 워커에 준다
    g = torch.Generator().manual_seed(1234)
    vocab = tok.vocab_size if hasattr(tok, "vocab_size") else 163860
    fields = {
        "input_ids": torch.randint(10, min(vocab, 160000), (B, L), generator=g), "input_lengths": torch.full((B,), L),
        "advantages": torch.randn(B, L, generator=g), "prev_logprobs": -torch.rand(B, L, generator=g),
        "generation_logprobs": -torch.rand(B, L, generator=g), "reference_policy_logprobs": -torch.rand(B, L, generator=g),
        "token_mask": torch.ones(B, L), "sample_mask": torch.ones(B),
    }
    if args.r3:
        n_moe = mcfg.num_layers  # alpha 는 24층 전부 MoE (moe_layer_freq 1)
        fields["routed_experts"] = _synthetic_routes(
            B * L, n_moe, mcfg.moe_router_topk, mcfg.num_moe_experts, mcfg.moe_router_num_groups,
            mcfg.moe_router_group_topk, g).view(B, L, n_moe, mcfg.moe_router_topk)
    full = BatchedDataDict(fields)
    dp_idx = rank // args.cp  # layout = arange(world).reshape(1, dp, cp, 1)
    if pol["sequence_packing"]["enabled"]:
        sp = {"algorithm": pol["sequence_packing"]["algorithm"], "input_key": "input_ids",
              "input_lengths_key": "input_lengths",
              "sequence_length_pad_multiple": pol["make_sequence_length_divisible_by"],
              "max_tokens_per_microbatch": pol["sequence_packing"]["train_mb_tokens"]}
        if pol["sequence_packing"].get("microbatch_order") is not None:
            sp["microbatch_order"] = pol["sequence_packing"]["microbatch_order"]
        shards, _ = full.shard_by_batch_size(dp, batch_size=B, sequence_packing_args=sp)
    else:
        shards = full.shard_by_batch_size(dp, batch_size=B)
    data = shards[dp_idx]
    steps = []
    snap = os.environ.get("MEM_SNAPSHOT_DIR")  # 설정 시 rank 0 의 할당 이력(스택 포함)을 OOM 이어도 덤프
    if snap and rank == 0:
        torch.cuda.memory._record_memory_history(max_entries=200000)
    for s in range(args.steps):
        torch.cuda.reset_peak_memory_stats()
        n_opt_before = len(opt_times)
        torch.cuda.synchronize()
        t = time.time()
        try:
            res = worker.train(data, loss_fn)
        except Exception as e:  # OOM 이어도 스냅샷을 남기고 다시 올린다
            if snap and rank == 0:
                os.makedirs(snap, exist_ok=True)
                torch.cuda.memory._dump_snapshot(os.path.join(snap, f"oom_step{s+1}_L{L}_cp{args.cp}.pickle"))
                print(f"MEM SNAPSHOT dumped on {type(e).__name__}: peak_alloc={gb(torch.cuda.max_memory_allocated())}GB "
                      f"reserved={gb(torch.cuda.max_memory_reserved())}GB", flush=True)
            raise
        torch.cuda.synchronize()
        sec = time.time() - t
        if snap and rank == 0 and s == 0:
            os.makedirs(snap, exist_ok=True)
            torch.cuda.memory._dump_snapshot(os.path.join(snap, f"ok_step1_L{L}_cp{args.cp}.pickle"))
        # 워커 반환 키는 global_loss·grad_norm 이다 (2026-10-07 이전 판은 없는 키 "loss" 를 읽어 NaN 을 찍었다)
        assert "global_loss" in res and "grad_norm" in res, f"train() 반환 키가 바뀌었다: {sorted(res)}"
        loss = float(torch.as_tensor(res["global_loss"]).float().mean())
        grad_norm = float(torch.as_tensor(res["grad_norm"]).float().max())
        opt_s = sum(opt_times[n_opt_before:])
        steps.append({"step": s + 1, "sec": round(sec, 2), "optimizer_sec": round(opt_s, 2),
                      "tok_per_s_node": round(B * L / sec, 1),
                      "fwd_bwd_tok_per_s_node": round(B * L / max(sec - opt_s, 1e-9), 1),
                      "peak_alloc_gb": gb(torch.cuda.max_memory_allocated()),
                      "peak_reserved_gb": gb(torch.cuda.max_memory_reserved()),
                      "after_alloc_gb": gb(torch.cuda.memory_allocated()),
                      "loss": loss, "grad_norm": grad_norm,
                      "finite": bool(math.isfinite(loss) and math.isfinite(grad_norm))})

    rec = {"rank": rank, "seq_len": L, "cp": args.cp, "dp": dp, "ep": pol["megatron_cfg"]["expert_model_parallel_size"],
           "fused_linear_logprobs": bool(args.fused_linear_logprobs), "setup_s": round(setup_s, 1),
           "floor_alloc_gb": gb(floor_alloc), "steps": steps}
    allrec = [None] * world
    torch.distributed.all_gather_object(allrec, rec)
    if rank == 0:
        worst = max(allrec, key=lambda r: max(s["peak_alloc_gb"] for s in r["steps"]))
        slowest = [max(r["steps"][i]["sec"] for r in allrec) for i in range(args.steps)]
        summary = {"seq_len": L, "cp": args.cp, "dp": dp, "samples_per_dp": args.samples_per_dp,
                   "tokens_per_step": B * L, "recompute": rc, "r3": bool(args.r3), "set": args.set,
                   "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                   "fused_linear_logprobs": bool(args.fused_linear_logprobs),
                   "logprob_chunk_size": args.logprob_chunk_size, "fuse_loss": bool(args.fuse_loss),
                   "floor_alloc_gb_max": max(r["floor_alloc_gb"] for r in allrec),
                   "after_step_alloc_gb_max": max(max(s["after_alloc_gb"] for s in r["steps"]) for r in allrec),
                   "peak_alloc_gb_max": max(max(s["peak_alloc_gb"] for s in r["steps"]) for r in allrec),
                   "peak_reserved_gb_max": max(max(s["peak_reserved_gb"] for s in r["steps"]) for r in allrec),
                   "step_sec": slowest,
                   "optimizer_sec_rank0": [s["optimizer_sec"] for s in allrec[0]["steps"]],
                   "tok_per_s_node": [round(B * L / t_, 1) for t_ in slowest],
                   "all_finite": all(s["finite"] for r in allrec for s in r["steps"]),
                   "loss_rank0": [s["loss"] for s in allrec[0]["steps"]],
                   "grad_norm_rank0": [s["grad_norm"] for s in allrec[0]["steps"]],
                   "worst_rank": worst["rank"], "per_rank": allrec}
        print("MEM RESULT " + json.dumps({k: v for k, v in summary.items() if k != "per_rank"}), flush=True)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(summary, f, indent=1)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
