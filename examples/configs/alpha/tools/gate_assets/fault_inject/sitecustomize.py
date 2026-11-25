"""Gate-only fault injection for the prefix-cache KL gate (R1-L). NOT production code.

Active only when this directory is on PYTHONPATH *and* ALPHA_GATE_FAULT is set, i.e. only in the
negative-control runs of gates/prefix_cache_*/run_l.sh. It patches the installed vLLM in memory:

  ALPHA_GATE_FAULT=state_scale:1.01   multiply the recurrent state a GDN prefill chunk starts from
                                      (a cache-hit restore or a chunked-prefill continuation) by 1.01
  ALPHA_GATE_FAULT=state_zero         zero that state instead
  ALPHA_GATE_FAULT=restore_scale:1.01 multiply by 1.01 the fp32 recurrent state a request restores from a cache block
                                      (cache-hit restore only: a cold prefill's own chunk boundaries are untouched,
                                      which is what the same-server gate PC-S needs)
  ALPHA_GATE_FAULT=restore_bf16       round that restored state to bf16 and back (a cache that lost precision)
  ALPHA_GATE_FAULT=routes_trunc:8     drop the last 8 expert rows vLLM returns for a request (R3)
  ALPHA_GATE_FAULT=routes_shift:3168  replace expert ids e by (e + 1) % num_experts on the first 3168 tokens

Every patched call site prints one marker line the first time it fires, so the judge can tell
that the fault really was injected (a negative control without the marker is invalid). The `restore_*`
faults print one line per restore, with the number of state tensors they changed.
"""

import importlib.abc
import importlib.util
import os
import sys

_FAULT = os.environ.get("ALPHA_GATE_FAULT", "")
_GDN = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
_RE = "vllm.model_executor.layers.fused_moe.routed_experts_capturer"
_MU = "vllm.v1.worker.mamba_utils"


def _say(msg):
    print(f"[ALPHA_GATE_FAULT pid={os.getpid()}] {msg}", flush=True)


def _patch_gdn(mod, kind, value):
    cls = mod.ChunkGatedDeltaRule
    fired = {"n": 0}

    def wrap(orig):
        def patched(self, *args, **kwargs):
            st = kwargs.get("initial_state")
            if st is not None and st.numel() and bool((st != 0).any()):
                kwargs["initial_state"] = st * value if kind == "state_scale" else st * 0
                if fired["n"] == 0:
                    _say(f"FIRED {kind} value={value} on a GDN prefill chunk with a non-zero initial state")
                fired["n"] += 1
            return orig(self, *args, **kwargs)

        return patched

    for name in ("forward_cuda", "forward_native", "forward_cutedsl"):
        if hasattr(cls, name):
            setattr(cls, name, wrap(getattr(cls, name)))
    _say(f"patched ChunkGatedDeltaRule ({kind})")


def _patch_restore(mod, kind, value):
    """Corrupt the state a new request restores from a cached block (align mode).

    `preprocess_mamba` copies the state of the block a request last used into the block it will write next. For a
    request the runner has no state index for yet, with tokens already computed, that source is a cache block:
    a prefix-cache hit (or a resume after preemption). The fault is applied to the destination block right after
    the copy, so it is independent of which kernel (prefill chunk or single-token step) reads the state next.
    """
    import inspect

    import torch

    orig = mod.preprocess_mamba
    sig = inspect.signature(orig)

    def patched(*args, **kwargs):
        a = sig.bind(*args, **kwargs).arguments
        so, state_idx, requests = a["scheduler_output"], a["mamba_state_idx"], a["requests"]
        stale = set(so.finished_req_ids) | set(so.preempted_req_ids or ()) | set(so.scheduled_cached_reqs.resumed_req_ids)
        restores = [
            rid for rid in a["input_batch"].req_ids
            if (rid not in state_idx or rid in stale) and requests[rid].num_computed_tokens > 0
        ]  # fmt: skip
        out = orig(*args, **kwargs)
        for rid in restores:
            req, dest, n = requests[rid], state_idx[rid], 0
            for gid in a["copy_bufs"].mamba_group_ids:
                block_id = req.block_ids[gid][dest]
                for layer_name in a["kv_cache_config"].kv_cache_groups[gid].layer_names:
                    for state in a["forward_context"][layer_name].kv_cache:
                        if state.dtype != torch.float32:  # the recurrent state is the fp32 one; the conv window is not
                            continue
                        if kind == "restore_scale":
                            state[block_id].mul_(value)
                        else:
                            state[block_id].copy_(state[block_id].to(torch.bfloat16).to(torch.float32))
                        n += 1
            _say(f"FIRED {kind} value={value} req={rid} computed={req.num_computed_tokens} dest_block={dest} tensors={n}")
        return out

    mod.preprocess_mamba = patched
    _say(f"patched mamba_utils.preprocess_mamba ({kind})")


def _patch_routes(mod, kind, value):
    cls = mod.RoutedExpertsManager
    orig = cls.get
    fired = {"n": 0}

    def patched(self, block_ids, num_tokens, token_start=0):
        out = orig(self, block_ids, num_tokens, token_start)
        n = int(value)
        if kind == "routes_trunc" and out.shape[0] > n:
            out = out[:-n]
        elif kind == "routes_shift":
            upto = max(0, n - token_start)
            if upto > 0:
                out = out.copy()
                out[:upto] = ((out[:upto].astype("int64") + 1) % 192).astype(out.dtype)  # alpha_v2: 192 experts
        if fired["n"] == 0:
            _say(f"FIRED {kind} value={value} in RoutedExpertsManager.get (rows returned {out.shape[0]})")
        fired["n"] += 1
        return out

    cls.get = patched
    _say(f"patched RoutedExpertsManager.get ({kind})")


class _PostImport(importlib.abc.MetaPathFinder):
    def __init__(self, targets):
        self.targets = targets

    def find_spec(self, fullname, path, target=None):
        if fullname not in self.targets:
            return None
        fn = self.targets.pop(fullname)
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is None or spec.loader is None:
                continue
            loader = spec.loader
            exec_module = loader.exec_module

            def patched_exec(module, _exec=exec_module, _fn=fn):
                _exec(module)
                _fn(module)

            loader.exec_module = patched_exec
            return spec
        return None


if _FAULT:
    _kind, _, _val = _FAULT.partition(":")
    _targets = {}
    if _kind in ("state_scale", "state_zero"):
        _targets[_GDN] = lambda m: _patch_gdn(m, _kind, float(_val or 0))
    elif _kind in ("restore_scale", "restore_bf16"):
        _targets[_MU] = lambda m: _patch_restore(m, _kind, float(_val or 0))
    elif _kind in ("routes_trunc", "routes_shift"):
        _targets[_RE] = lambda m: _patch_routes(m, _kind, int(_val))
    else:
        raise SystemExit(f"ALPHA_GATE_FAULT: unknown fault {_FAULT!r}")
    sys.meta_path.insert(0, _PostImport(_targets))
