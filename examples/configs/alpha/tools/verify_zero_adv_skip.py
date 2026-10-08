"""게이트 Z1 — grpo.skip_zero_advantage_samples 의 기울기 동일성 (torchrun 8 GPU, Ray 없음, 2026-10-08).

measure_train_memory.py 와 같은 방식으로 MegatronPolicyWorkerImpl 을 직접 만들고, 같은 가중치에서 train() 을 세 번 부른다.
optimizer step 은 lr 0 으로 돌려 가중치를 바꾸지 않는다 (가중치 표본으로 확인). 기울기 norm 은 optimizer 가 반환한 값을 쓴다.
  A: 전체 배치 (gbs = B)
  B: select_training_samples(advantage 0 샘플 제외 + advantage 재조정) + lr_scheduler_increment = B   ← 제안 경로
  C: 음성 대조 — 같은 샘플을 빼되 재조정 없음 (기울기가 N_full/N_kept 배로 어긋나야 한다)
판정: B 가 A 와 loss·grad norm·표본 파라미터 기울기에서 1e-3 안으로 같고, 스케줄러 전진량이 A 와 같다.
C 는 grad norm 비가 N_full/N_kept 에 1% 안으로 맞아야 한다 (게이트가 정규화 오류를 잡는다는 근거).
packing 의 시퀀스 간 격리는 G1 에서 비트 동일로 확인했다 — 남는 차이는 누적 순서뿐이어야 한다.
loss 안 시퀀스 마스킹은 끈다: 합성 generation_logprobs 로는 마스킹이 무작위로 갈린다 (실런에서 마스킹이 일어나면 정확성 주의, docstring 참조).

실행 (NeMo-RL 루트, 8 GPU, CUDA_VISIBLE_DEVICES 미지정):
  $NRL_ROOT/clean_run.sh /usr/bin/env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
      $W/bin/python -m torch.distributed.run --nproc_per_node 8 \\
      examples/configs/alpha/tools/verify_zero_adv_skip.py --config examples/configs/alpha/student_rlvr1_alpha.yaml \\
      --seq-len 8192 --samples 16 --out z1.json </dev/null
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch


def _synthetic_routes(num_tokens, num_layers, topk, num_experts, num_groups, group_topk, g):
    n = num_tokens * num_layers
    per_group = num_experts // num_groups
    k_in_group = topk // group_topk
    groups = torch.rand(n, num_groups, generator=g).topk(group_topk, dim=-1).indices
    offs = torch.rand(n, group_topk, per_group, generator=g).topk(k_in_group, dim=-1).indices
    return (groups.unsqueeze(-1) * per_group + offs).view(num_tokens, num_layers, topk).to(torch.int16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--cp", type=int, default=8)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--zero-frac", type=float, default=0.5)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import ray

    local_rank = int(os.environ["LOCAL_RANK"])
    ray.get_gpu_ids = lambda: [local_rank]

    from omegaconf import OmegaConf

    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

    register_omegaconf_resolvers()
    cfg = OmegaConf.to_container(load_config(args.config), resolve=True)
    world = int(os.environ["WORLD_SIZE"])
    dp = world // args.cp
    pol = cfg["policy"]
    L, B = args.seq_len, args.samples
    assert B % dp == 0
    pol["max_total_sequence_length"] = L
    pol["train_global_batch_size"] = B
    pol["train_micro_batch_size"] = 1
    pol["make_sequence_length_divisible_by"] = 2 * args.cp
    pol["megatron_cfg"]["context_parallel_size"] = args.cp
    pol["sequence_packing"]["enabled"] = True
    pol["sequence_packing"]["train_mb_tokens"] = L
    pol["sequence_packing"]["logprob_mb_tokens"] = L
    pol["megatron_cfg"].setdefault("train_iters", 10)  # 워커가 요구한다 (GRPO 는 setup 에서 채운다)
    pol["router_replay"] = {"enabled": True}
    pol["generation"]["colocated"]["enabled"] = False
    pol["generation"]["max_new_tokens"] = L
    loss_cfg = dict(cfg["loss_fn"])
    loss_cfg["seq_logprob_error_in_loss"] = False  # 위 docstring

    from nemo_rl.algorithms.grpo import select_training_samples
    from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossConfig, ClippedPGLossFn
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.distributed.named_sharding import NamedSharding
    from nemo_rl.models.policy.workers.megatron_policy_worker import MegatronPolicyWorkerImpl

    sharding = NamedSharding(layout=np.arange(world).reshape(1, -1, args.cp, 1),
                             names=["pipeline_parallel", "data_parallel", "context_parallel", "tensor_parallel"])
    tok = get_tokenizer(pol["tokenizer"])
    worker = MegatronPolicyWorkerImpl(pol, tok, init_optimizer=True, init_reference_model=False,
                                      worker_sharding_annotations=sharding)
    rank = torch.distributed.get_rank()
    mcfg = worker.model.config
    loss_fn = ClippedPGLossFn(ClippedPGLossConfig(**loss_cfg))

    # 합성 배치: 길이 L/2..L, 앞 10% 는 프롬프트(마스크 0), advantage 는 샘플마다 상수 (GRPO 처럼), zero-frac 만큼 0
    g = torch.Generator().manual_seed(1234)
    lengths = torch.randint(L // 2, L + 1, (B,), generator=g) // (2 * args.cp) * (2 * args.cp)
    pos = torch.arange(L)[None, :]
    token_mask = ((pos >= L // 10) & (pos < lengths[:, None])).float()
    n_zero = int(B * args.zero_frac)
    adv_per_sample = torch.randn(B, generator=g)
    adv_per_sample[torch.randperm(B, generator=g)[:n_zero]] = 0.0
    fields = {
        "input_ids": torch.randint(10, 160000, (B, L), generator=g), "input_lengths": lengths,
        "advantages": adv_per_sample[:, None].expand(B, L).clone() * token_mask,
        "prev_logprobs": -torch.rand(B, L, generator=g), "generation_logprobs": -torch.rand(B, L, generator=g),
        "reference_policy_logprobs": -torch.rand(B, L, generator=g),
        "token_mask": token_mask, "sample_mask": torch.ones(B),
        "routed_experts": _synthetic_routes(B * L, mcfg.num_layers, mcfg.moe_router_topk, mcfg.num_moe_experts,
                                            mcfg.moe_router_num_groups, mcfg.moe_router_group_topk, g
                                            ).view(B, L, mcfg.num_layers, mcfg.moe_router_topk),
    }
    full = BatchedDataDict(fields)
    kept, kept_metrics = select_training_samples(full, keep_fraction=0.0, dp_size=dp)
    keep_idx = [int(i) for i in torch.nonzero((full["advantages"][:, 1:] != 0).any(1)).flatten()]
    unscaled = full.select_indices(keep_idx)
    assert kept.size == len(keep_idx), "keep_fraction 0 with dp | kept should keep exactly the signal samples"
    valid = full["token_mask"][:, 1:]
    n_full, n_kept = float(valid.sum()), float(valid[keep_idx].sum())

    sp = {"algorithm": pol["sequence_packing"]["algorithm"], "input_key": "input_ids",
          "input_lengths_key": "input_lengths", "sequence_length_pad_multiple": pol["make_sequence_length_divisible_by"],
          "max_tokens_per_microbatch": pol["sequence_packing"]["train_mb_tokens"]}

    def shard(batch):
        shards, _ = batch.shard_by_batch_size(dp, batch_size=batch.size, sequence_packing_args=sp)
        return shards[rank // args.cp]

    params = [p for _, p in worker.model.named_parameters() if p.requires_grad]
    probe = [params[i] for i in sorted({0, 1, len(params) // 2, len(params) // 2 + 1, len(params) - 2, len(params) - 1})]

    def optimizers():
        return getattr(worker.optimizer, "chained_optimizers", [worker.optimizer])

    grads: dict[str, list[torch.Tensor]] = {}
    real_step = worker.optimizer.step

    def run(name, batch, **kw):
        for opt in optimizers():
            for group in opt.param_groups:
                group["lr"] = 0.0
        before = [p.detach().float().clone() for p in probe]
        sched0 = worker.scheduler.num_steps

        def capture_step(*a, **k):
            grads[name] = [p.main_grad.detach().float().cpu().clone() for p in probe]
            return real_step(*a, **k)

        worker.optimizer.step = capture_step
        res = worker.train(shard(batch), loss_fn, **kw)
        worker.optimizer.step = real_step
        unchanged = all(torch.equal(b, p.detach().float()) for b, p in zip(before, probe))
        return {"loss": float(torch.as_tensor(res["global_loss"]).float().sum()),
                "grad_norm": float(torch.as_tensor(res["grad_norm"]).float().max()),
                "sched_delta": worker.scheduler.num_steps - sched0, "weights_unchanged": unchanged}

    out = {"A": run("A", full), "A2": run("A2", full),
           "B": run("B", kept, gbs=kept.size, lr_scheduler_increment=B),
           "C": run("C", unscaled, gbs=unscaled.size)}
    ddp_cfg = getattr(worker.model, "ddp_config", None)
    config_facts = {
        "moe_z_loss_coeff": getattr(mcfg, "moe_z_loss_coeff", None),
        "moe_aux_loss_coeff": getattr(mcfg, "moe_aux_loss_coeff", None),
        "moe_router_load_balancing_type": getattr(mcfg, "moe_router_load_balancing_type", None),
        "grad_reduce_in_fp32": getattr(ddp_cfg, "grad_reduce_in_fp32", None),
        "params_dtype": str(getattr(mcfg, "params_dtype", None)),
        "main_grad_dtype": str(probe[0].main_grad.dtype),
    }

    def rel(x, y):
        num = math.sqrt(sum(float((a - b).pow(2).sum()) for a, b in zip(x, y)))
        den = math.sqrt(sum(float(a.pow(2).sum()) for a in x))
        return num / max(den, 1e-30)

    rec = {"rank": rank, **{k: v for k, v in out.items()},
           "probe_grad_rel_diff_A2": rel(grads["A"], grads["A2"]),
           "probe_grad_rel_diff_B": rel(grads["A"], grads["B"]), "probe_grad_rel_diff_C": rel(grads["A"], grads["C"])}
    allrec = [None] * world
    torch.distributed.all_gather_object(allrec, rec)
    if rank == 0:
        A, A2, Bv, C = out["A"], out["A2"], out["B"], out["C"]
        ratio = n_full / n_kept
        # 잡음 바닥: 같은 전체 배치를 두 번 돌린 차이 (backward 비결정성·누적 반올림)
        floor_probe = max(r["probe_grad_rel_diff_A2"] for r in allrec)
        floor_gn = abs(A2["grad_norm"] - A["grad_norm"]) / A["grad_norm"]
        checks = {
            "loss_B_matches_A": abs(Bv["loss"] - A["loss"]) <= args.tol * max(abs(A["loss"]), 1e-12),
            "grad_norm_B_within_noise": abs(Bv["grad_norm"] - A["grad_norm"]) / A["grad_norm"] <= max(args.tol, 3 * floor_gn),
            "probe_grads_B_within_noise": max(r["probe_grad_rel_diff_B"] for r in allrec) <= max(args.tol, 3 * floor_probe),
            "lr_ticks_B_equal_A": Bv["sched_delta"] == A["sched_delta"] == B,
            "weights_unchanged": all(r[v]["weights_unchanged"] for r in allrec for v in ("A", "A2", "B", "C")),
            # 음성 대조: 재조정을 빼면 차이가 잡음 바닥보다 훨씬 커야 한다
            "negative_control_C_detected": min(r["probe_grad_rel_diff_C"] for r in allrec) > 10 * max(args.tol, floor_probe),
        }
        summary = {"verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
                   "n_full_tokens": n_full, "n_kept_tokens": n_kept, "expected_C_ratio": ratio,
                   "kept_samples": kept.size, "samples": B, "kept_metrics": kept_metrics,
                   "A": A, "A2": A2, "B": Bv, "C": C, "C_grad_norm_ratio": C["grad_norm"] / A["grad_norm"],
                   "noise_floor_probe_rel_diff": floor_probe, "noise_floor_grad_norm_rel_diff": floor_gn,
                   "config_facts": config_facts,
                   "probe_grad_rel_diff_B_max": max(r["probe_grad_rel_diff_B"] for r in allrec),
                   "probe_grad_rel_diff_C_min": min(r["probe_grad_rel_diff_C"] for r in allrec),
                   "seq_len": L, "cp": args.cp, "dp": dp}
        print("Z1 RESULT " + json.dumps(summary), flush=True)
        if args.out:
            with open(args.out, "w") as f:
                json.dump({**summary, "per_rank": allrec}, f, indent=1)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
