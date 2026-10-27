"""코드 라운드(teacher_codemath_code/s1) 문제별 정답률 분석 — 원천별 · 이전 라벨별 · 에폭 안 시기별(무작위 순서라 난이도 통제) · 참조 모델 정답률별 · 길이 · 테스트 수 (2026-10-27).
사용: python -I analyze_code_round_passrate.py (경로는 파일 안 상수)
"""
import json, collections, statistics, math
B="/home/work/Datasets/LL_datasets/posttraining/RL/alpha_blends/teacher_blends/teacher_codemath_code_all_v1.jsonl"
L="/home/work/vidsearch/repos/project_s/NeMo-RL/results/alpha/teacher_codemath_code/s1/rollout_groups.jsonl"
meta={}
for l in open(B):
    r=json.loads(l)
    c,n,src=r.get("latest_correct"),r.get("latest_n"),r.get("latest_source")
    prior="unmeasured" if c is None else f"{src}:{'0' if c==0 else 'all' if c==n else 'mixed'}"
    meta[r["pool_key"]]=dict(prior=prior, prior_rate=(None if c is None else c/n), variant=r.get("pool_variant"), dataset=r.get("dataset"),
        nano=r.get("ref_pass_rate_nano_sft"), sup=r.get("ref_pass_rate_super_lcsft1000"), ult=r.get("ref_pass_rate_ultra_sft3200"), ntests=r.get("n_unit_tests"))
G=[]
for l in open(L):
    try: d=json.loads(l)
    except Exception: continue
    k=d.get("pool_key")
    if k not in meta: continue
    r=d["rewards"]; c=sum(1 for x in r if x>0); n=len(r)
    tr=d.get("truncated") or [False]*n; g=d["gen_tokens"]
    G.append(dict(k=k, step=d["generation_weight_version"], c=c, n=n, rate=c/n, trunc=sum(tr)/n, len=statistics.mean(g),
                  len_cor=[t for t,x in zip(g,r) if x>0], len_wr=[t for t,x in zip(g,r) if x<=0], **meta[k]))
def band(c,n): return "0/16" if c==0 else "16/16" if c==n else "1-3" if c<=3 else "4-12" if c<=12 else "13-15"
def summ(rows):
    n=len(rows); 
    if not n: return "-"
    b=collections.Counter(band(x["c"],x["n"]) for x in rows)
    return f"n={n} pass@1 {statistics.mean(x['rate'] for x in rows):.3f} pass@16 {sum(1 for x in rows if x['c']>0)/n:.3f} | 0/16 {b['0/16']/n:.2f} 1-3 {b['1-3']/n:.2f} 4-12 {b['4-12']/n:.2f} 13-15 {b['13-15']/n:.2f} 16/16 {b['16/16']/n:.2f} | trunc {statistics.mean(x['trunc'] for x in G if x in rows) if False else statistics.mean(x['trunc'] for x in rows):.3f} len {statistics.mean(x['len'] for x in rows)/1000:.1f}K"
print("== 1. overall"); print(summ(G)); print("unique prompts", len({x['k'] for x in G}), "groups", len(G))
print("\n== 2. by source dataset")
by=collections.defaultdict(list)
for x in G: by[x["dataset"]].append(x)
for ds,rows in sorted(by.items(), key=lambda kv:-len(kv[1])): print(f"  {str(ds)[:44]:44s} {summ(rows)}")
print("\n== 3. by prior label (latest measurement before this round)")
by=collections.defaultdict(list)
for x in G: by[x["prior"]].append(x)
for p,rows in sorted(by.items(), key=lambda kv:-len(kv[1])):
    pr=[x["prior_rate"] for x in rows if x["prior_rate"] is not None]
    print(f"  {p:16s} prior {statistics.mean(pr) if pr else float('nan'):.3f} -> {summ(rows)}")
print("\n== 4. learning within the epoch: pass@1 by step bin, per prior stratum (random order => difficulty-controlled)")
bins=[(229,290),(291,352),(353,414),(415,476),(477,538),(539,601)]
strata=["s2r:mixed","s5:0","p04:0","unmeasured","s2r:all","s2r:0"]
print("  stratum".ljust(18)+"".join(f"{a}-{b}".rjust(12) for a,b in bins)+"   slope/100step ± se")
for s in strata+["ALL"]:
    rows=[x for x in G if (s=="ALL" or x["prior"]==s)]
    cells=[]
    for a,b in bins:
        R=[x["rate"] for x in rows if a<=x["step"]<=b]; cells.append(f"{statistics.mean(R):.3f}({len(R)})" if R else "-")
    xs=[x["step"] for x in rows]; ys=[x["rate"] for x in rows]
    mx,my=statistics.mean(xs),statistics.mean(ys); sxx=sum((a-mx)**2 for a in xs); b=sum((a-mx)*(c-my) for a,c in zip(xs,ys))/sxx
    res=[c-my-b*(a-mx) for a,c in zip(xs,ys)]; se=math.sqrt(sum(e*e for e in res)/(len(xs)-2)/sxx)
    print(f"  {s:16s}"+"".join(c.rjust(12) for c in cells)+f"   {100*b:+.4f} ± {100*se:.4f}")
print("\n== 5. vs reference model pass rates (Nano SFT on same problem)")
def rb(v): return "none" if v is None or v!=v else "0" if v==0 else "(0,.25)" if v<.25 else "[.25,.5)" if v<.5 else "[.5,.75)" if v<.75 else "[.75,1]"
for name,key in (("Nano SFT","nano"),("Super LCSFT","sup"),("Ultra SFT","ult")):
    by=collections.defaultdict(list)
    for x in G: by[rb(x[key])].append(x)
    print(f"  {name}: "+" | ".join(f"{b}: n={len(by[b])} alpha {statistics.mean(x['rate'] for x in by[b]):.3f}" for b in ["0","(0,.25)","[.25,.5)","[.5,.75)","[.75,1]","none"] if by[b]))
print("\n== 6. length vs correctness")
cor=[t for x in G for t in x["len_cor"]]; wr=[t for x in G for t in x["len_wr"]]
print(f"  correct samples {len(cor)} mean {statistics.mean(cor)/1000:.1f}K | wrong {len(wr)} mean {statistics.mean(wr)/1000:.1f}K | truncated share of all samples {sum(x['trunc']*x['n'] for x in G)/sum(x['n'] for x in G):.3f}")
z=[x for x in G if x["c"]==0]; print(f"  0/16 groups: {len(z)} — mean trunc {statistics.mean(x['trunc'] for x in z):.3f}; all-16-truncated {sum(1 for x in z if x['trunc']==1.0)}; >=half truncated {sum(1 for x in z if x['trunc']>=0.5)}")
print("\n== 7. unit-test count")
by=collections.defaultdict(list)
for x in G:
    t=x["ntests"] or 0; by["<10" if t<10 else "10-49" if t<50 else "50-199" if t<200 else "200+"].append(x)
for b in ["<10","10-49","50-199","200+"]: print(f"  tests {b:7s} {summ(by[b])}")
