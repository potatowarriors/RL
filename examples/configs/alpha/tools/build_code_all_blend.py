"""Code-only full-pool blend (RL_PLAN 결정 45, 사용자 2026-10-24): 코드 풀 23,971 전부에서 검증 128 만 뺀다. 중복 없음(결정 43), 무작위 셔플.
행에 최신 라벨(latest_correct/latest_n/latest_source: s2r 기록 > s5 기록 > P0-4 > 없음)을 붙여 두어 라운드 뒤 분석에 쓴다.
사용: python build_code_all_blend.py --out <jsonl> [--seed N]
"""
import argparse, json, random, collections
P="/home/work/Datasets/LL_datasets/posttraining/RL/alpha_blends/teacher_pool/"
B="/home/work/Datasets/LL_datasets/posttraining/RL/alpha_blends/teacher_blends/"
R="/home/work/vidsearch/repos/project_s/NeMo-RL/results/alpha/"
ap=argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=20261024)
ap.add_argument("--val", default=B+"teacher_codemath_s2_v1_val.jsonl"); a=ap.parse_args()
val={json.loads(l)["pool_key"] for l in open(a.val)}
def record(path):
    last={}
    for l in open(path):
        try: d=json.loads(l)
        except Exception: continue
        if d.get("agent")!="code_gen_simple_agent": continue
        k=d.get("pool_key"); v=d["generation_weight_version"]; c=sum(1 for x in d["rewards"] if x>0)
        if k not in last or v>=last[k][0]: last[k]=(v,c,len(d["rewards"]))
    return {k:(c,n) for k,(v,c,n) in last.items()}
s2r=record(R+"teacher_codemath_s2r/s1/rollout_groups.jsonl"); s5=record(R+"teacher_codemath_s3/s5/rollout_groups.jsonl")
p04={}
for l in open(P+"p04_passmap.jsonl"):
    d=json.loads(l)
    if "code" in str(d.get("agent")): p04[d["key"]]=(d["k"],d["n"])
rows=[]; stats=collections.Counter(); lab=collections.Counter()
with open(P+"code_pool.jsonl") as f:
    for l in f:
        r=json.loads(l); k=r["pool_key"]
        if k in val: stats["excluded_val"]+=1; continue
        if k in s2r: c,n=s2r[k]; src="s2r"
        elif k in s5: c,n=s5[k]; src="s5"
        elif k in p04: c,n=p04[k]; src="p04"
        else: c,n,src=None,None,None
        r["code_round"]=1; r["latest_correct"]=c; r["latest_n"]=n; r["latest_source"]=src
        lab["unmeasured" if c is None else ("0" if c==0 else "all" if c==n else "mixed")]+=1
        rows.append(r); stats[f"variant_{r.get('pool_variant')}"]+=1
random.Random(a.seed).shuffle(rows)
with open(a.out,"w") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False)+"\n")
out={"seed":a.seed,"train_rows":len(rows),"train_unique_prompts":len({r["pool_key"] for r in rows}),"excluded_val":stats["excluded_val"],
     "by_variant":{k[8:]:v for k,v in stats.items() if k.startswith("variant_")},"latest_label":dict(lab),"val_train_overlap":0,
     "steps_per_epoch_at_64":len(rows)//64}
json.dump(out, open(a.out.replace(".jsonl","_stats.json"),"w"), indent=1, ensure_ascii=False); print(json.dumps(out, ensure_ascii=False))
