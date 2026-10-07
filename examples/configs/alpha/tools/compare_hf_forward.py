"""시작점 HF 와 RL 반출 HF 의 forward 대조 — **Pai 환경**(`/usr/bin/python3`, transformers 4.57)에서 돈다 (G6, 2026-10-07).

반출 HF 의 소비자는 Pai 스택(벤치·다음 단계 입력)이다. `compare_hf_weights.py` 가 텐서를 본다면 이 도구는 Pai 가 그 텐서를
**같은 모델로 해석하는지**를 본다 (config·토크나이저·chat template·forward). 두 모델을 GPU 2장에 올려 같은 토큰열의 logprob 을 비교한다.
  - config: 두 디렉토리의 AutoConfig 속성 비교 (`--raw-config` 로 변환기 원본 config.json 의 해석 차이도 볼 수 있다)
  - 토크나이저: 같은 텍스트·chat 렌더(thinking·도구 포함)의 토큰 id 가 같은지
  - forward: 텍스트별 perplexity, 토큰 logprob 차이(max·mean |Δ|), top-1 일치율, KL(base‖new)

  - 통제(`--control-sigmas 0,1e-7,1e-6`): 시작점 vs 시작점+잡음. σ=0 은 결정성(차이 0), σ>0 은 잡음 바닥이다.
    bf16 + MoE 경계 라우팅 때문에 아주 작은 섭동도 같은 크기의 차이로 번진다 (G6 2026-10-07: 실현 변화 6e-9 의 잡음이 도구 대화 KL 0.135 —
    3스텝 반출본 0.137 과 같다). 반출 차이는 이 바닥과 비교해 해석한다. 결함(config·토크나이저 해석 오류)이면 ppl 이 수십~수백 배로 뛴다.

  CUDA_VISIBLE_DEVICES=0,1 python3 examples/configs/alpha/tools/compare_hf_forward.py --base <hfmodel> --new <exported_hf> \
      [--raw-config <converter_output_dir>] [--control-sigmas 0,1e-7,1e-6] [--out report.json]
종료 코드: config·토크나이저 불일치 또는 forward 가 유한하지 않으면 1. logprob 차이는 기록만 한다 — 통제 결과와 함께 판단한다.
"""

import argparse
import json
import math
import sys

import torch

TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.",
                                           "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                          "required": ["city"]}}}]
CONVS = {
    "chat_think": ([{"role": "user", "content": "Compute 17*23."},
                    {"role": "assistant", "reasoning_content": "17*23 = 17*20 + 17*3 = 340 + 51 = 391.",
                     "content": "17 × 23 = 391."}], None),
    "chat_tool": ([{"role": "user", "content": "What's the weather in Seoul right now?"},
                   {"role": "assistant", "content": "", "reasoning_content": "I should call the weather tool.",
                    "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Seoul"}}}]},
                   {"role": "tool", "content": "{\"temp_c\": 18, \"sky\": \"clear\"}"},
                   {"role": "assistant", "content": "It is 18°C and clear in Seoul."}], TOOLS),
}
TEXTS = {
    "en_facts": "The capital of France is Paris. Water is made of hydrogen and oxygen. The sun rises in the east and sets "
                "in the west. Two plus two equals four. Monday comes before Tuesday, and winter is colder than summer.",
    "ko_facts": "대한민국의 수도는 서울이다. 물은 수소와 산소로 이루어져 있다. 해는 동쪽에서 떠서 서쪽으로 진다. 월요일 다음은 화요일이다.",
}
VOLATILE = {"transformers_version", "_name_or_path", "torch_dtype", "dtype", "_commit_hash", "_attn_implementation_autoset"}


def config_diff(a, b):
    da, db = a.to_dict(), b.to_dict()
    keys = sorted((set(da) | set(db)) - VOLATILE)
    out = {k: (da.get(k, "<missing>"), db.get(k, "<missing>")) for k in keys if da.get(k, "<missing>") != db.get(k, "<missing>")}
    for attr in ("layer_types", "full_attention_interval", "rope_theta", "partial_rotary_factor", "num_hidden_layers"):
        va, vb = getattr(a, attr, "<missing>"), getattr(b, attr, "<missing>")
        if va != vb:
            out[f"attr:{attr}"] = (va, vb)
    return {k: [str(x)[:200] for x in v] for k, v in out.items()}


@torch.no_grad()
def token_logprobs(model, ids, device):
    logits = model(torch.tensor([ids], device=device)).logits[0, :-1].float()
    lp = torch.log_softmax(logits, dim=-1)
    tgt = torch.tensor(ids[1:], device=device)
    return lp.cpu(), lp.gather(-1, tgt[:, None])[:, 0].cpu()


def compare_one(mb, mn, ids):
    lpb, tb_lp = token_logprobs(mb, ids, "cuda:0")
    lpn, tn_lp = token_logprobs(mn, ids, "cuda:1")
    d = (tb_lp - tn_lp).abs()
    kl = (lpb.exp() * (lpb - lpn)).sum(-1)
    return {"tokens": len(ids), "ppl_base": math.exp(-tb_lp.mean().item()), "ppl_new": math.exp(-tn_lp.mean().item()),
            "max_abs_dlogp": d.max().item(), "mean_abs_dlogp": d.mean().item(),
            "top1_agree": (lpb.argmax(-1) == lpn.argmax(-1)).float().mean().item(), "kl_mean": kl.mean().item(),
            "finite": bool(torch.isfinite(lpb).all() and torch.isfinite(lpn).all())}


def fmt(r):
    return (f"tok={r['tokens']:4d} ppl {r['ppl_base']:.4f} → {r['ppl_new']:.4f}  max|Δlogp| {r['max_abs_dlogp']:.2e} "
            f"mean {r['mean_abs_dlogp']:.2e}  top1 {r['top1_agree']:.4f}  KL {r['kl_mean']:.2e}  finite={r['finite']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--raw-config", default=None, help="변환기 원본 출력 디렉토리 — config.json 해석 차이만 본다")
    ap.add_argument("--skip-forward", action="store_true", help="config·토크나이저만 본다 (CPU)")
    ap.add_argument("--control-sigmas", default=None,
                    help="통제 대조: 시작점 vs 시작점+잡음(텐서 RMS 대비 σ, bf16 반올림) 을 같은 텍스트로 잰다. 예 '0,1e-7,1e-6'. "
                         "0 은 결정성(같은 가중치 → 차이 0) 확인이다. 반출 차이를 이 잡음 바닥과 비교해 해석한다")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    rep, fail = {}, False
    cb, cn = (AutoConfig.from_pretrained(p, trust_remote_code=True) for p in (args.base, args.new))
    rep["config_diff"] = config_diff(cb, cn)
    fail |= bool(rep["config_diff"])
    if args.raw_config:
        rep["raw_config_diff"] = config_diff(cb, AutoConfig.from_pretrained(args.raw_config, trust_remote_code=True))
    tb, tn = (AutoTokenizer.from_pretrained(p, trust_remote_code=True) for p in (args.base, args.new))
    seqs, tok_mismatch = {}, []
    for name, text in TEXTS.items():
        seqs[name] = tb(text, add_special_tokens=False)["input_ids"]
        if tn(text, add_special_tokens=False)["input_ids"] != seqs[name]:
            tok_mismatch.append(name)
    for name, (msgs, tools) in CONVS.items():
        rb, rn = (t.apply_chat_template(msgs, tools=tools, tokenize=False) for t in (tb, tn))
        seqs[name] = tb(rb, add_special_tokens=False)["input_ids"]
        if rb != rn or tn(rn, add_special_tokens=False)["input_ids"] != seqs[name]:
            tok_mismatch.append(name)
    rep["tokenizer_mismatch"] = tok_mismatch
    fail |= bool(tok_mismatch)

    rep["forward"] = {}
    if args.skip_forward:
        seqs = {}
    else:
        mb = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True).eval()
        mn = AutoModelForCausalLM.from_pretrained(args.new, dtype=torch.bfloat16, device_map="cuda:1", trust_remote_code=True).eval()
    for name, ids in seqs.items():
        r = compare_one(mb, mn, ids)
        fail |= not r["finite"]
        rep["forward"][name] = r
        print(f"{name:12s} " + fmt(r))
    if args.control_sigmas is not None and not args.skip_forward:
        del mn
        torch.cuda.empty_cache()
        mc = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cuda:1", trust_remote_code=True).eval()
        orig = {n: p.detach().to("cpu", copy=True) for n, p in mc.named_parameters() if p.dim() >= 2}
        rep["control"] = {}
        gen = torch.Generator(device="cuda:1").manual_seed(0)
        for sigma in [float(x) for x in args.control_sigmas.split(",")]:
            with torch.no_grad():
                rel = []
                for n, p in mc.named_parameters():
                    if n not in orig:
                        continue
                    w = orig[n].to(p.device).float()
                    noisy = w + sigma * w.pow(2).mean().sqrt() * torch.randn(w.shape, device=p.device, generator=gen)
                    p.copy_(noisy.to(p.dtype))
                    rel.append(((p.float() - w).norm() / w.norm().clamp_min(1e-30)).item())
            rep["control"][str(sigma)] = {"realized_max_rel": max(rel), "realized_median_rel": sorted(rel)[len(rel) // 2], "forward": {}}
            print(f"-- control σ={sigma:g}: realized weight change max_rel {max(rel):.2e} median {sorted(rel)[len(rel) // 2]:.2e}")
            for name, ids in seqs.items():
                r = compare_one(mb, mc, ids)
                rep["control"][str(sigma)]["forward"][name] = r
                print(f"   {name:12s} " + fmt(r))
    print("config_diff:", rep["config_diff"] or "none")
    if args.raw_config:
        print("raw_config_diff:", json.dumps(rep["raw_config_diff"], ensure_ascii=False))
    print("tokenizer_mismatch:", tok_mismatch or "none")
    rep["pass"] = not fail
    print("PASS" if not fail else "FAIL")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=1, ensure_ascii=False)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
