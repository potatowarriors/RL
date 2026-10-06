"""Alpha Muon 게이트: NeMo-RL Megatron 워커와 같은 경로로 모델+옵티마이저를 만들고, Muon 이 실제로 적용됐는지 검사한다.

mcore 는 Muon 경로의 흔적(INFO 로그 `Setting up emerging optimizer`, `dist_*` DeprecationWarning)을 기본 설정에서
모두 숨기므로, GRPO 런 로그만으로는 Muon 적용을 증명할 수 없다. 이 도구는 MegatronPolicyWorker.__init__ 의
setup 순서(setup_distributed → validate_model_paths → handle_model_import → validate_and_set_config →
setup_model_and_optimizer)를 그대로 호출하고 만들어진 옵티마이저를 직접 본다.

검사 (전부 PASS 여야 함):
  1. 옵티마이저 트리에 Muon 계열 클래스가 있고 Adam 계열(스칼라 파라미터용)도 있다
  2. 분배: 2D 은닉 가중치(linear_qkv·linear_proj·GDN in/out_proj·router·experts fc1/fc2) → Muon,
     embedding·output_layer·norm·GDN conv1d/A_log/dt_bias → Adam
  3. linear_qkv 가중치에 is_qkv=True 와 4-way [Q|Gate|K|V] split shapes
  4. Muon 하이퍼파라미터가 레시피 값과 같다 (momentum·nesterov·ns steps·scale mode·extra scale factor)

실행 (NeMo-RL 루트, Megatron 워커 venv python, 8 GPU = 레시피의 EP):
  $NRL_ROOT/clean_run.sh $W/bin/python -m torch.distributed.run --nproc_per_node 8 \
      examples/configs/alpha/tools/verify_muon_optimizer.py \
      --config examples/configs/alpha/grpo_alpha_smoke_muon.yaml [--hf-path <hfmodel>] [--out summary.json] </dev/null
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import torch


def _categorize(name: str) -> str:
    n = name.lower()
    if "embedding" in n or "word_embeddings" in n:
        return "embedding"
    if "output_layer" in n:
        return "output_layer"
    if "norm" in n:
        return "norm"
    for key in ("conv1d", "a_log", "dt_bias"):
        if key in n:
            return f"gdn_{key}"
    if "linear_qkv" in n:
        return "linear_qkv"
    if "linear_proj" in n:
        return "linear_proj"
    if "in_proj" in n:
        return "gdn_in_proj"
    if "out_proj" in n:
        return "gdn_out_proj"
    if "router" in n:
        return "router"
    if "shared_expert" in n:
        return "shared_experts"
    if "experts" in n:
        return "experts"
    return "other"


EXPECT_MUON = {"linear_qkv", "linear_proj", "gdn_in_proj", "gdn_out_proj", "experts"}
EXPECT_ADAM = {"embedding", "output_layer", "norm", "gdn_conv1d", "gdn_a_log", "gdn_dt_bias"}


def _leaf_optimizers(opt):
    """Yield (wrapper, inner_torch_optimizer) leaves of a (Chained/LayerWise/Float16) optimizer tree."""
    children = getattr(opt, "chained_optimizers", None)
    if children:
        for c in children:
            yield from _leaf_optimizers(c)
        return
    inner = getattr(opt, "optimizer", None)
    if inner is not None and inner is not opt and isinstance(inner, torch.optim.Optimizer):
        yield opt, inner
    elif isinstance(opt, torch.optim.Optimizer):
        yield opt, opt


def _model_params_of(wrapper, inner):
    """Model (bf16) params handled by a leaf: Float16 wrappers keep fp32 copies in inner groups."""
    groups = getattr(wrapper, "float16_groups", None)
    if groups:
        params = [p for g in groups for p in g]
        params += [p for g in getattr(wrapper, "fp32_from_fp32_groups", []) for p in g]
        return params
    return [p for g in inner.param_groups for p in g["params"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--hf-path", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from omegaconf import OmegaConf

    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

    register_omegaconf_resolvers()  # ${mul:...} 등 — run_grpo.py 와 동일
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

    # MegatronPolicyWorker.__init__ 와 같은 순서: set_device 가 setup_distributed(NCCL) 보다 먼저.
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    setup_distributed()
    rank = torch.distributed.get_rank()
    hf_model_name, pretrained_path, pt_exists = validate_model_paths(policy)
    handle_model_import(policy, hf_model_name, pretrained_path, pt_exists)
    rt = validate_and_set_config(policy, rank, hf_model_name, pretrained_path, None, None)
    rt.megatron_cfg.validate()
    state = setup_model_and_optimizer(policy, rt.megatron_cfg, True, load_weights=True)
    model, optimizer = state.model, state.optimizer

    names = {}
    for chunk in model if isinstance(model, list) else [model]:
        for n, p in chunk.named_parameters():
            names[id(p)] = n

    per_opt = defaultdict(Counter)  # optimizer class -> category -> numel
    per_opt_examples = defaultdict(dict)
    hp = {}
    for wrapper, inner in _leaf_optimizers(optimizer):
        cls = type(inner).__name__
        for p in _model_params_of(wrapper, inner):
            n = names.get(id(p), "<unmapped>")
            cat = _categorize(n)
            per_opt[cls][cat] += p.numel()
            per_opt_examples[cls].setdefault(cat, n)
        g0 = inner.param_groups[0] if inner.param_groups else {}
        hp[cls] = {k: (v if isinstance(v, (int, float, bool, str)) or v is None else str(v))
                   for k, v in g0.items() if k != "params"}
        for attr in ("momentum", "nesterov", "num_ns_steps", "scale_mode", "extra_scale_factor",
                     "split_qkv", "fp32_matmul_prec", "coefficient_type"):
            if hasattr(inner, attr):
                hp[cls][f"attr.{attr}"] = str(getattr(inner, attr))
        # TensorParallelMuon 은 num_ns_steps·scale_mode·extra_scale_factor 를 scaled_orthogonalize_fn 클로저에 캡처한다
        fn = getattr(inner, "scaled_orthogonalize_fn", None)
        if fn is not None and getattr(fn, "__closure__", None):
            for var, cell in zip(fn.__code__.co_freevars, fn.__closure__):
                try:
                    val = cell.cell_contents
                except ValueError:
                    continue
                if isinstance(val, (int, float, bool, str)):
                    hp[cls][f"closure.{var}"] = val

    qkv = []
    for chunk in model if isinstance(model, list) else [model]:
        for n, p in chunk.named_parameters():
            if n.endswith("linear_qkv.weight"):
                qkv.append((n, bool(getattr(p, "is_qkv", False)), getattr(p, "qkv_split_shapes", None), tuple(p.shape)))

    # layer-wise 옵티마이저는 dense 파라미터를 DP rank 에 나눠 맡긴다 → 전 rank 합집합으로 판정한다
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, {c: dict(v) for c, v in per_opt.items()})
    per_opt_all = defaultdict(Counter)
    for g in gathered:
        for c, cats in g.items():
            per_opt_all[c].update(cats)

    muon_cls = [c for c in per_opt_all if "muon" in c.lower() or "orthogonal" in c.lower()]
    adam_cls = [c for c in per_opt_all if "adam" in c.lower()]
    checks = {}
    checks["1_classes"] = bool(muon_cls) and bool(adam_cls)
    muon_cats = set().union(*[set(per_opt_all[c]) for c in muon_cls]) if muon_cls else set()
    adam_cats = set().union(*[set(per_opt_all[c]) for c in adam_cls]) if adam_cls else set()
    # 전 rank 합집합에서는 기대 범주가 전부 나타나야 한다(부재로 통과하는 공허한 PASS 금지)
    checks["2_assignment"] = EXPECT_MUON <= muon_cats and EXPECT_ADAM <= adam_cats \
        and not (EXPECT_ADAM & muon_cats) and not (EXPECT_MUON & adam_cats)
    checks["3_qkv_split"] = bool(qkv) and all(t and s is not None and len(s) == 4 for _, t, s, _ in qkv)
    want = policy["megatron_cfg"]["optimizer"]
    mh = next((hp[c] for c in hp if "muon" in c.lower()), {})
    expect_pairs = {
        "momentum": ("momentum", want.get("muon_momentum")),
        "nesterov": ("attr.nesterov", str(want.get("muon_nesterov"))),
        "extra_scale_factor": ("closure.extra_scale_factor", want.get("muon_extra_scale_factor")),
        "num_ns_steps": ("closure.num_ns_steps", want.get("muon_num_ns_steps")),
        "scale_mode": ("closure.scale_mode", want.get("muon_scale_mode")),
        "split_qkv": ("attr.split_qkv", str(want.get("muon_split_qkv"))),
    }
    hp_cmp = {k: {"got": mh.get(src), "want": w, "ok": mh.get(src) == w} for k, (src, w) in expect_pairs.items()}
    checks["4_hparams"] = all(v["ok"] for v in hp_cmp.values())

    summary = {
        "rank": rank,
        "optimizer_tree": type(optimizer).__name__,
        "numel_by_optimizer_and_category_all_ranks": {c: dict(v) for c, v in per_opt_all.items()},
        "muon_hparams_vs_recipe": hp_cmp,
        "numel_by_optimizer_and_category": {c: dict(v) for c, v in per_opt.items()},
        "example_param_per_category": per_opt_examples,
        "leaf_hparams": hp,
        "linear_qkv": [{"name": n, "is_qkv": t, "split": s, "shape": sh} for n, t, s, sh in qkv[:3]],
        "linear_qkv_count": len(qkv),
        "checks": checks,
        "recipe_muon": {k: v for k, v in want.items() if k.startswith(("muon_", "optimizer", "adam_", "use_"))},
    }
    if rank == 0:
        print(json.dumps(summary, indent=1, default=str))
        print("MUON GATE", "PASS" if all(checks.values()) else "FAIL", checks)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(summary, f, indent=1, default=str)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
