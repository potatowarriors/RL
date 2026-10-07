"""SFT 학습 엔진(Pai-Megatron) ↔ RL 학습 엔진(NeMo-RL mcore) forward/backward 동등성 게이트 — 공통부.

두 엔진은 모델 클래스(Pai: MambaModel 48층 + megatron_patch GDN / NeMo-RL: GPTModel 24층 + mcore GDN)와
파라미터 이름이 다르다. 그래서 텐서를 이름이 아니라 **가중치 값의 지문**으로 짝짓는다:
같은 체크포인트에서 비트 동일하게 로드되므로(라운드트립 14,181/14,181), 레이아웃이 같은 텐서는 양쪽 지문이 같다.
레이아웃이 다른 결합 텐서(GDN in_proj 등)는 자동으로 짝이 안 지어지고 보고서에 '미매칭'으로 남는다.

지문 = bf16/fp32 비트를 정수로 본 다항 해시. 정수 덧셈은 순서와 무관하게 정확(int64 wraparound)하므로
torch 버전·커널이 달라도 같은 값이 나온다(부동소수 합은 그렇지 않다).

이 파일은 Pai 환경(Python 3.12, torch 2.8)과 NeMo-RL 환경(Python 3.13, torch 2.11) 양쪽에서 import 된다 — torch 만 쓴다.
"""

import torch

CHUNK = 1 << 24  # 지문 계산 청크(원소 수) — 큰 텐서의 임시 메모리 상한


def fingerprint(t: torch.Tensor) -> tuple:
    x = t.detach().contiguous().view(-1)
    int_dtype = {2: torch.int16, 4: torch.int32}[x.element_size()]
    h1 = torch.zeros((), dtype=torch.int64, device=x.device)
    h2 = torch.zeros((), dtype=torch.int64, device=x.device)
    for s in range(0, x.numel(), CHUNK):
        bits = x[s : s + CHUNK].view(int_dtype).to(torch.int64)
        idx = torch.arange(s, s + bits.numel(), device=x.device, dtype=torch.int64)
        h1 += (bits * ((idx * 2654435761 + 97) % 1000003 + 1)).sum()
        h2 += ((bits + 40000) * ((idx * 40503 + 7) % 999983 + 1)).sum()
    return (tuple(t.shape), str(t.dtype).replace("torch.", ""), int(h1.item()), int(h2.item()))


def next_token_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """logits [b,s,v] 또는 [s,b,v] → 다음 토큰 logprob [b, s-1] (fp32)."""
    b, s = input_ids.shape
    if logits.shape[0] == s and logits.shape[1] == b and b != s:
        logits = logits.transpose(0, 1)
    lp = torch.log_softmax(logits.float(), dim=-1)
    return lp[:, :-1, :].gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)


def _grad_of(p: torch.nn.Parameter):
    g = getattr(p, "main_grad", None)
    if g is None:
        g = p.grad
    return g


def summarize_params(named_params, batch_token_ids: torch.Tensor, vocab_rows: int) -> list:
    """파라미터별 지문 + gradient 요약. 작은 텐서와 지문 표본(약 3%) 큰 텐서는 gradient 전체를 CPU fp32 로 담는다.

    vocab 행 텐서(embedding·output)는 배치에 등장한 토큰 행 + 고정 간격 행만 담는다(양쪽이 같은 행을 고르도록 행 인덱스로 선택).
    """
    uniq = torch.unique(batch_token_ids).cpu()
    fixed_rows = torch.arange(0, vocab_rows, max(1, vocab_rows // 2048))
    vocab_sel = torch.unique(torch.cat([uniq, fixed_rows]))
    out = []
    for name, p in named_params:
        fpk = fingerprint(p.data)
        g = _grad_of(p)
        rec = {"name": name, "key": fpk, "numel": p.numel(), "requires_grad": bool(p.requires_grad)}
        if g is None:
            rec["grad"] = None
            out.append(rec)
            continue
        gf = g.detach().float()
        rec["grad_norm"] = float(gf.double().norm().item())
        rec["grad_sum"] = float(gf.double().sum().item())
        full = None
        if p.dim() == 2 and p.shape[0] == vocab_rows:
            rec["rows"] = vocab_sel
            full = gf.index_select(0, vocab_sel.to(gf.device)).cpu()
        elif p.numel() < 1_000_000 or (fpk[2] % 32 == 0 and p.numel() <= 64_000_000):
            full = gf.cpu()
        rec["grad"] = full
        out.append(rec)
    return out
