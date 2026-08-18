"""FlashQLA vs FLA-Triton benchmark on alpha's GDN geometry (mcore real path).

Findings encoded here (2026-08-18):
  - fla 0.4.2 chunk_gated_delta_rule produces NaN for native GQA (Hk != Hv)
    on both batch and varlen paths — which is why mcore expands q/k to Hv via
    repeat_interleave BEFORE calling it (gdn.py "prepare_input" step).
  - FlashQLA handles both MHA-expanded and native-GQA inputs correctly
    (matches fla's own naive_recurrent reference and the expanded-MHA path,
    cos ~0.99999, max|d| ~8e-6 at bf16).

Three configs measured on the real varlen path (cu_seqlens):
  1. fla,  MHA-expanded (production today)
  2. qla,  MHA-expanded (drop-in swap)
  3. qla,  native GQA   (swap + skip the repeat_interleave)

Run:
  uv run --locked --extra mcore --with flash-qla python \
      examples/configs/alpha/tools/bench_flashqla.py
"""

import torch
import torch.nn.functional as F

from fla.modules.l2norm import l2norm
from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_impl
from flash_qla import chunk_gated_delta_rule as qla_impl

HK, HV, D = 16, 32, 128
SEQ_LENS = [4096, 16384, 49152, 65536]
WARMUP, ITERS = 10, 30


def make_inputs(T: int, expand: bool, requires_grad: bool = False):
    torch.manual_seed(1234)
    q = l2norm(torch.randn(1, T, HK, D, device="cuda", dtype=torch.bfloat16))
    k = l2norm(torch.randn(1, T, HK, D, device="cuda", dtype=torch.bfloat16))
    if expand:
        q = q.repeat_interleave(HV // HK, dim=2).contiguous()
        k = k.repeat_interleave(HV // HK, dim=2).contiguous()
    v = torch.randn(1, T, HV, D, device="cuda", dtype=torch.bfloat16) * 0.02
    A = torch.empty(1, T, HV, device="cuda").uniform_(1, 16)
    g = (-A * F.softplus(torch.randn(1, T, HV, device="cuda"))).to(torch.float32)
    beta = torch.sigmoid(torch.randn(1, T, HV, device="cuda", dtype=torch.bfloat16))
    ts = (q, k, v, g, beta)
    if requires_grad:
        ts = tuple(t.detach().clone().requires_grad_(True) for t in ts)
    return ts


def run(impl, q, k, v, g, beta, cu):
    out, _ = impl(q, k, v, g=g, beta=beta, initial_state=None,
                  output_final_state=False, use_qk_l2norm_in_kernel=False,
                  cu_seqlens=cu)
    return out


def time_ms(fn) -> float:
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS


def correctness(T: int = 8192) -> None:
    cu = torch.tensor([0, T], device="cuda", dtype=torch.long)
    exp_a = make_inputs(T, expand=True, requires_grad=True)
    exp_b = tuple(t.detach().clone().requires_grad_(True) for t in exp_a)
    out_fla = run(fla_impl, *exp_a, cu)
    out_qla = run(qla_impl, *exp_b, cu)
    gqa = make_inputs(T, expand=False)
    out_gqa = run(qla_impl, *gqa, cu)

    def stats(name, a, b):
        d = (a.float() - b.float()).abs()
        cos = F.cosine_similarity(a.float().flatten()[None], b.float().flatten()[None]).item()
        print(f"[correctness {name}] max|d|={d.max():.6f} mean|d|={d.mean():.7f} cos={cos:.6f}")

    stats("fwd qla-MHA vs fla-MHA", out_qla, out_fla)
    stats("fwd qla-GQA vs fla-MHA", out_gqa, out_fla)
    out_fla.sum().backward()
    out_qla.sum().backward()
    for name, a, b in zip(("dq", "dk", "dv", "dg", "dbeta"), exp_a, exp_b):
        stats(f"bwd {name}", b.grad, a.grad)


def main() -> None:
    print(f"GDN geometry Hk={HK} Hv={HV} D={D} bf16 varlen | warmup={WARMUP} iters={ITERS}")
    correctness()

    print(f"\n{'T':>7} | {'fla-MHA':>9} {'qla-MHA':>9} {'qla-GQA':>9} | {'MHA spd':>7} {'GQA spd':>7}  (fwd)")
    rows_fb = []
    for T in SEQ_LENS:
        cu = torch.tensor([0, T], device="cuda", dtype=torch.long)
        exp = make_inputs(T, expand=True)
        gqa = make_inputs(T, expand=False)
        f_fla = time_ms(lambda: run(fla_impl, *exp, cu))
        f_qla = time_ms(lambda: run(qla_impl, *exp, cu))
        f_gqa = time_ms(lambda: run(qla_impl, *gqa, cu))
        print(f"{T:>7} | {f_fla:>7.2f}ms {f_qla:>7.2f}ms {f_gqa:>7.2f}ms | {f_fla / f_qla:>6.2f}x {f_fla / f_gqa:>6.2f}x")

        exp_g = make_inputs(T, expand=True, requires_grad=True)
        gqa_g = make_inputs(T, expand=False, requires_grad=True)

        def fb(impl, tensors):
            def _fn():
                for t in tensors:
                    t.grad = None
                run(impl, *tensors, cu).sum().backward()
            return _fn

        b_fla = time_ms(fb(fla_impl, exp_g))
        b_qla = time_ms(fb(qla_impl, exp_g))
        b_gqa = time_ms(fb(qla_impl, gqa_g))
        rows_fb.append((T, b_fla, b_qla, b_gqa))

    print(f"\n{'T':>7} | {'fla-MHA':>9} {'qla-MHA':>9} {'qla-GQA':>9} | {'MHA spd':>7} {'GQA spd':>7}  (fwd+bwd)")
    for T, b_fla, b_qla, b_gqa in rows_fb:
        print(f"{T:>7} | {b_fla:>7.2f}ms {b_qla:>7.2f}ms {b_gqa:>7.2f}ms | {b_fla / b_qla:>6.2f}x {b_fla / b_gqa:>6.2f}x")


if __name__ == "__main__":
    main()
