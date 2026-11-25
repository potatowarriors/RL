"""Gate-only fault injection for the prefix-cache KL gate (R1-L). NOT production code.

Active only when this directory is on PYTHONPATH *and* ALPHA_GATE_FAULT is set, i.e. only in the
negative-control runs of gates/prefix_cache_*/run_l.sh. It patches the installed vLLM in memory:

  ALPHA_GATE_FAULT=state_scale:1.01   multiply the recurrent state a GDN prefill chunk starts from
                                      (a cache-hit restore or a chunked-prefill continuation) by 1.01
  ALPHA_GATE_FAULT=state_zero         zero that state instead
  ALPHA_GATE_FAULT=routes_trunc:8     drop the last 8 expert rows vLLM returns for a request (R3)
  ALPHA_GATE_FAULT=routes_shift:3168  replace expert ids e by (e + 1) % num_experts on the first 3168 tokens

Every patched call site prints one marker line the first time it fires, so the judge can tell
that the fault really was injected (a negative control without the marker is invalid).
"""

import importlib.abc
import importlib.util
import os
import sys

_FAULT = os.environ.get("ALPHA_GATE_FAULT", "")
_GDN = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
_RE = "vllm.model_executor.layers.fused_moe.routed_experts_capturer"


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
    elif _kind in ("routes_trunc", "routes_shift"):
        _targets[_RE] = lambda m: _patch_routes(m, _kind, int(_val))
    else:
        raise SystemExit(f"ALPHA_GATE_FAULT: unknown fault {_FAULT!r}")
    sys.meta_path.insert(0, _PostImport(_targets))
