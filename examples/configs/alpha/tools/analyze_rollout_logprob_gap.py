"""Rollout(vLLM) vs train(mcore) 토큰 logprob 어긋남 분해 — NeMo-RL train_data_step*.jsonl 입력, CPU 전용.

Generation KL(k3)·mult_prob_err 재계산(게이트 출력과 일치해야 함), 꼬리(|Δ|>0.5·>1), 생성 위치별 KL, 잘림/완결별 KL.
위치에 따라 KL 이 커지면 재귀 상태 dtype(mamba_ssm_cache_dtype), 꼬리가 크면 MoE 라우팅 뒤집힘(R3) 의심.
사용: python analyze_rollout_logprob_gap.py [--max-len N] <log_dir>/exp_*/train_data_step*.jsonl
  --max-len: 레시피 max_total_sequence_length (기본 4096 = R1 스모크). 잘림 판정과 위치 구간 상한에 쓴다.
"""
import argparse, json, sys, math
import numpy as np
ap=argparse.ArgumentParser(); ap.add_argument("--max-len",type=int,default=4096); ap.add_argument("files",nargs="+")
ARGS=ap.parse_args()
def position_bins(max_len):
    if max_len==4096: return [0,256,512,1024,2048,3072,4096]  # R1·G2 기록과 같은 구간
    b=[0,256,512,1024,2048,4096]
    while b[-1]<max_len: b.append(b[-1]*2)
    return b
def load(f):
    rows=[]
    for line in open(f):
        r=json.loads(line)
        m=np.array(r["token_loss_mask"][0],bool); g=np.array(r["generation_logprobs"][0]); p=np.array(r["prev_logprobs"][0])
        rows.append((m,g,p,r["input_lengths"][0]))
    return rows
for f in ARGS.files:
    rows=load(f); D=[];POS=[];SK=[];TR=[]
    for m,g,p,L in rows:
        idx=np.where(m)[0]
        if len(idx)==0: continue
        d=p[idx]-g[idx]; D.append(d); POS.append(idx-idx[0]); k3=np.exp(d)-d-1; SK.append(k3.mean()); TR.append(L>=ARGS.max_len)
    d=np.concatenate(D); pos=np.concatenate(POS); k3=np.exp(d)-d-1; a=np.abs(d)
    print(f"== {f.split('/')[-1]}: samples={len(SK)} gen_tokens={len(d)}")
    print(f"  gen_kl(k3 mean)={k3.mean():.5f}  mult_prob_err(mean exp|d|)={np.exp(a).mean():.4f}  mean|d|={a.mean():.4f}")
    print(f"  tail: |d|>0.1 {100*(a>0.1).mean():.2f}%  >0.5 {100*(a>0.5).mean():.3f}%  >1 {100*(a>1).mean():.3f}%  >2 {100*(a>2).mean():.3f}%")
    o=np.sort(k3)[::-1]; n=len(o)
    print(f"  KL share of top 0.1%/1%/10% tokens: {o[:max(1,n//1000)].sum()/o.sum():.2f} / {o[:n//100].sum()/o.sum():.2f} / {o[:n//10].sum()/o.sum():.2f}")
    print(f"  KL excl. |d|>1 tokens: {k3[a<=1].mean():.5f}")
    bins=position_bins(ARGS.max_len)
    s="  by position: "+"  ".join(f"[{bins[i]},{bins[i+1]}) {k3[(pos>=bins[i])&(pos<bins[i+1])].mean():.5f}" for i in range(len(bins)-1) if ((pos>=bins[i])&(pos<bins[i+1])).any())
    print(s)
    SK=np.array(SK);TR=np.array(TR)
    print(f"  per-sample KL: median {np.median(SK):.5f} p90 {np.percentile(SK,90):.5f} max {SK.max():.5f}; truncated {TR.sum()}/{len(TR)} mean {SK[TR].mean() if TR.any() else float('nan'):.5f} vs completed {SK[~TR].mean() if (~TR).any() else float('nan'):.5f}")
