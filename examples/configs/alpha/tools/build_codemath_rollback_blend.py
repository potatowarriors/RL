"""Rollback blend: stage-3 blend rows restricted to prompts with confirmed learning signal.
Keep a row when (a) its pool_key's LAST visit in s5's rollout record was mixed (1..15/16), or
(b) it was never visited in s5 and came from the stage-2 record (stage3_source s2_log / s2_log_dup, mixed in stage 2).
Drop: all-wrong / all-right last visits, and unvisited reserve / p04 rows (no signal evidence).
Note: the 573 code *_dup rows inherited from the stage-3 blend (effective G=32) were kept in teacher_codemath_s2r_v1 (built 2026-10-23 10:01);
decision 43 (user, 2026-10-23) stops code duplication from the next blend on — new blends collapse *_dup rows by pool_key."""
import json, random, collections, sys
PREV="/home/work/Datasets/LL_datasets/posttraining/RL/alpha_blends/teacher_blends/teacher_codemath_s3_v1.jsonl"
LOG="/home/work/vidsearch/repos/project_s/NeMo-RL/results/alpha/teacher_codemath_s3/s5/rollout_groups.jsonl"
VAL="/home/work/Datasets/LL_datasets/posttraining/RL/alpha_blends/teacher_blends/teacher_codemath_s2_v1_val.jsonl"
OUT="/home/work/Datasets/LL_datasets/posttraining/RL/alpha_blends/teacher_blends/teacher_codemath_s2r_v1.jsonl"
last={}
for l in open(LOG):
    try: d=json.loads(l)
    except Exception: continue
    k=d.get("pool_key")
    if k is None: continue
    v=d["generation_weight_version"]
    if k not in last or v>=last[k][0]:
        c=sum(1 for x in d["rewards"] if x>0); last[k]=(v,c,len(d["rewards"]))
val_keys=set(); val_q=set()
for l in open(VAL):
    r=json.loads(l); val_keys.add(r.get("pool_key"))
    if r.get("question") is not None: val_q.add(r.get("question"))
rows=[json.loads(l) for l in open(PREV)]
keep=[]; why=collections.Counter(); agent=collections.Counter()
for r in rows:
    k=r["pool_key"]; s=r.get("stage3_source")
    q=r.get("question")
    if k in val_keys or (q is not None and q in val_q): why["drop_val"]+=1; continue
    if k in last:
        v,c,n=last[k]
        if 0<c<n: keep.append(r); why[f"keep_s5_mixed({s})"]+=1
        elif c==0: why["drop_s5_all_wrong"]+=1
        else: why["drop_s5_all_right"]+=1
    else:
        if s in ("s2_log","s2_log_dup"): keep.append(r); why[f"keep_unvisited_{s}"]+=1
        else: why[f"drop_unvisited_{s}"]+=1
for r in keep:
    r["rollback_source"]="s5_mixed" if r["pool_key"] in last else "s2_unvisited"
    agent[r["agent_ref"]["name"] if isinstance(r.get("agent_ref"),dict) else str(r.get("agent_ref"))]+=1
random.Random(20261023).shuffle(keep)
with open(OUT,"w") as f:
    for r in keep: f.write(json.dumps(r, ensure_ascii=False)+"\n")
uniq=len({r["pool_key"] for r in keep})
stats={"seed":20261023,"prev":PREV,"log":LOG,"val":VAL,"train_rows":len(keep),"train_unique_prompts":uniq,
       "by_agent":dict(agent),"decisions":dict(why),"val_rows":len(val_keys),"val_train_overlap":0,
       "steps_per_epoch_at_256":len(keep)//256,"steps_per_epoch_at_128":len(keep)//128,"steps_per_epoch_at_64":len(keep)//64}
json.dump(stats, open(OUT.replace(".jsonl","_stats.json"),"w"), indent=1, ensure_ascii=False)
print(json.dumps(stats, indent=1, ensure_ascii=False))
