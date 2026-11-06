"""1차 General(chat·IF) teacher 블렌드 (RL_DATA.md §5.3 비율, RL_PLAN.md 결정 50) — CPU, 결정론적.

원천은 Ultra 복원 블렌드(rlvr1·rlvr2·mopd·ifbench·rlhf) · Super 블렌드(rlvr1·rlvr2 복원본, rlvr3·rlhf 원본) · Nano 복원본 ·
Nemotron-RL-QA-Abstention-v1 (블렌드에 없는 648행) 의 General agent 행과 alpha-RL-Identity-Following-v1 이다.
Multichallenge-v1 · InverseIFEval-v1 원본은 agent 가 turing_vif 라 (다른 채점 서버) 넣지 않는다. calendar(보상 0)와 도구 계열(IPI 등, 2차 agentic)은 넣지 않는다.

- 중복: agent + 입력 메시지 + 도구 정의의 해시. 같은 과제가 여러 블렌드에 반복돼 있다.
- 참조 pass_rate 가 있으면 (0,1) 인 행만 쓴다 — 0·1 은 참조 정책에서도 advantage 가 0 이다 (도구 teacher·STEM 과 같은 규칙).
  원천에 pass_rate 1 인 행은 없어(NVIDIA 블렌드가 이미 뺐다) 실제로는 참조 SFT 가 한 번도 못 푼 행만 빠진다 (2026-11-06 집계).
- 끝의 effort 마커 `{reasoning effort: …}` 는 지운다 — effort 는 학습 때 env.nemo_gym.effort_levels 가 붙인다.
- 벤치 대조: IFEval 541 (13-gram) · SimpleQA 1,000 · LogicKor 42 (8-gram) 의 n-gram 을 절반 이상 담은 행은 뺀다.
- 프롬프트 길이 상한 (alpha chat template 렌더 토큰, measure_blend_prompt_lengths.py 와 같은 렌더): 넘으면 런이 멈추거나 judge 요청이
  실패해 채점기가 기본값(무승부·0)으로 삼킨다. 학습 창 131,072 − 생성 32,768 = 98,304 · GenRM 창 49,152 (대화 + 두 최종 답 + 출력 8K)
  → 16,384 · Gemma judge 창 65,536 (대화 + 답 최대 32K) → 28,672.
- 그룹 비율 = §5.3 (IF 35 · 구조화 15 · GenRM 25 · multichallenge·inverse_if 10 · abstention 8 · 안전 6 · identity 1) 에서
  GenRM 만 8% 로 줄였다 (RL_PLAN.md 결정 51 — 49B GenRM 부하 측정: 비교 270건 19.9분, step 병목).
  그룹 안은 agent 별 고유 행 수에 비례. 풀이 모자란 그룹은 반복하지 않고 전부 쓴 뒤 남는 몫을 IF 로 돌린다.
- 검증셋(--n-val)은 그룹 비율대로 먼저 떼고 학습에서 뺀다. 출력 옆에 .stats.json 을 남긴다.
  GenRM · identity 몫은 검증 파일에 쓰지 않는다 (--val-drop-groups): genrm_compare 는 같은 키의 응답 16개(num_rollouts_per_prompt)가
  모일 때까지 기다리는데 검증은 프롬프트당 2개뿐이고 task 번호가 찍히지 않아 끝나지 않는다 (JG1 적대적 검토 2026-11-06). 학습 행 수는 그대로다.
- 원천 행에 남은 Gym 기록 필드(`_ng_task_index` · `_ng_rollout_index` 등 `_ng_` 로 시작하는 키)는 지운다 — task 번호는 학습 때 수집기가 새로 찍는다.

  python examples/configs/alpha/tools/build_general_blend.py --out <dir>/teacher_general_v1.jsonl [--n-train 40960] [--n-val 512] [--seed 20261106]
"""

import argparse
import collections
import glob
import json
import os
import random
import re

import orjson
import xxhash

import measure_blend_prompt_lengths as M  # 같은 디렉토리 — 렌더·토큰 수 계산을 같은 코드로 한다

RL = "/home/work/Datasets/LL_datasets/posttraining/RL"
AB = f"{RL}/alpha_blends"
SOURCES = [
    f"{AB}/ultra_restored/rlvr1.jsonl", f"{AB}/ultra_restored/rlvr2.jsonl", f"{AB}/ultra_restored/mopd.jsonl",
    f"{AB}/ultra_restored/ifbench.jsonl", f"{AB}/ultra_restored/rlhf.jsonl",
    f"{AB}/super_restored/rlvr1.jsonl", f"{AB}/super_restored/rlvr2.jsonl",
    f"{RL}/Nemotron-RL-Super-Training-Blends/rlvr3.jsonl", f"{RL}/Nemotron-RL-Super-Training-Blends/rlhf.jsonl",
    f"{AB}/nano_restored/train.jsonl",
    f"{RL}/Nemotron-RL-QA-Abstention-v1/data/train.jsonl",  # abstention 원본 — 블렌드에 없는 648행 (pass_rate 없음)
]
IDENTITY = f"{RL}/alpha-RL-Identity-Following-v1/train.jsonl"
PAI_SFT = ("/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha/eval_sft/results/"
           "alpha_baseline_48L_sft_128k_agentic_resume_20261001_011900_iter0002862")
GROUPS = {  # 이름: (agent 들, 비중)
    "if": (["instruction_following_simple_agent"], 0.35),
    "struct": (["structured_outputs_simple_agent", "structured_outputs_v3_simple_agent",
                "citation_format_simple_agent", "freeform_formatting_simple_agent"], 0.15),
    "genrm": (["genrm_simple_agent", "genrm_simple_agent_reasoning_off"], 0.08),  # 25% → 8% (결정 51: GenRM 처리량 병목)
    "multi": (["multichallenge_simple_agent", "inverse_if_simple_agent"], 0.10),
    "abstention": (["abstention_simple_agent"], 0.08),
    "safety": (["jailbreak_detection_simple_agent", "jailbreak_refusal_with_explanation",
                "jailbreak_hard_refusal_with_helplines", "jailbreak_engagement_with_disclaimer",
                "jailbreak_hard_refusal_no_redirection", "over_refusal_detection_simple_agent"], 0.06),
    "identity": (["identity"], 0.01),
}
AGENT2GROUP = {a: g for g, (ags, _) in GROUPS.items() for a in ags}
SFT_HF = ("/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha/outputs/"
          "alpha_baseline_48L_sft_128k_agentic_resume_20261001_011900/hfmodel_0002862")
CAP_TRAIN, CAP_GENRM, CAP_LLM_JUDGE = 98304, 16384, 28672
PROMPT_CAP = {a: CAP_TRAIN for a in AGENT2GROUP}
PROMPT_CAP.update({a: CAP_GENRM for a in ("genrm_simple_agent", "genrm_simple_agent_reasoning_off", "identity")})
PROMPT_CAP.update({a: CAP_LLM_JUDGE for a in GROUPS["multi"][0] + GROUPS["abstention"][0] + GROUPS["safety"][0]})
EFFORT_RE = re.compile(r"\s*\{reasoning effort:\s*[a-z]+\}\s*$")
WORD = re.compile(r"\w+", re.U)


def last_user_text(row) -> str:
    inp = (row.get("responses_create_params") or {}).get("input")
    if isinstance(inp, str):
        return inp
    for m in reversed(inp or []):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
            return c or ""
    return ""


def strip_effort(row) -> bool:
    inp = (row.get("responses_create_params") or {}).get("input")
    if not isinstance(inp, list):
        return False
    for m in reversed(inp):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            new = EFFORT_RE.sub("", m["content"])
            if new != m["content"]:
                m["content"] = new
                return True
            return False
    return False


def task_key(agent, row) -> str:
    rcp = row.get("responses_create_params") or {}
    return xxhash.xxh64_hexdigest(orjson.dumps([agent, rcp.get("input"), rcp.get("tools")], option=orjson.OPT_SORT_KEYS))


def grams(text, n):
    toks = WORD.findall(text.lower())
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def load_bench():
    items = []  # (bench, n, gram set)
    f = glob.glob(PAI_SFT + "/alpha/samples_ifeval_aa_*.jsonl")[0]
    for line in open(f):
        items.append(("ifeval", 13, grams(json.loads(line)["doc"]["prompt"], 13)))
    for line in open(PAI_SFT + "/simpleqa_samples.jsonl"):
        items.append(("simpleqa", 8, grams(json.loads(line)["q"], 8)))
    for line in open(PAI_SFT + "/logickor_samples.jsonl"):
        items.append(("logickor", 8, grams(json.loads(line)["q1"], 8)))
    return [it for it in items if it[2]]


def contaminated(text, bench, index):
    hits = set()
    for n in (8, 13):
        for g in grams(text, n):
            hits.update(index[n].get(g, ()))
    for bi in hits:
        b, n, gs = bench[bi]
        if len(gs & grams(text, n)) >= 0.5 * len(gs):
            return b
    return None


def collect(bench):
    index = {8: collections.defaultdict(set), 13: collections.defaultdict(set)}
    for bi, (_, n, gs) in enumerate(bench):
        for g in gs:
            index[n][g].add(bi)
    pools = collections.defaultdict(dict)
    dropped = collections.defaultdict(set)
    st = collections.defaultdict(collections.Counter)
    names = tuple(a.encode() for a in AGENT2GROUP if a != "identity")
    for path in SOURCES:
        with open(path, "rb") as f:
            for line in f:
                if not any(n in line for n in names):
                    continue
                row = orjson.loads(line)
                ag = (row.get("agent_ref") or {}).get("name", "")
                if ag not in AGENT2GROUP:
                    continue
                st[ag]["source_rows"] += 1
                strip_effort(row)
                key = task_key(ag, row)
                pr = row.get("pass_rate")
                if pr is not None and not (0.0 < float(pr) < 1.0):
                    st[ag]["dropped_pass_rate_0_or_1"] += 1
                    if key not in pools[ag]:
                        dropped[ag].add(key)
                    continue
                if pr is None and key in dropped[ag]:  # pass_rate 0·1 로 뺀 과제가 pass_rate 없는 사본(원본 데이터셋)으로 되살아나지 않게
                    st[ag]["dropped_same_task_as_pass_rate_0_or_1"] += 1
                    continue
                if key in pools[ag]:
                    st[ag]["duplicate"] += 1
                    continue
                stale = [k for k in row if k.startswith("_ng_")]
                for k in stale:
                    del row[k]
                if stale:
                    st[ag]["stripped_ng_fields"] += 1
                b = contaminated(last_user_text(row), bench, index)
                if b:
                    st[ag][f"bench_overlap_{b}"] += 1
                    continue
                pools[ag][key] = row
    with open(IDENTITY, "rb") as f:
        for line in f:
            row = orjson.loads(line)
            st["identity"]["source_rows"] += 1
            pools["identity"].setdefault(task_key("identity", row), row)
    return pools, st


def apply_prompt_caps(pools, st, tokenizer, procs):
    items = [(ag, k, orjson.dumps(row).decode()) for ag, p in pools.items() for k, row in p.items()]
    from multiprocessing import Pool
    with Pool(procs, initializer=M._init, initargs=(tokenizer,)) as pool:
        res = pool.map(M._one, [line for _, _, line in items], chunksize=64)
    lens = collections.defaultdict(list)
    for (ag, k, _), (_, n, failed) in zip(items, res):
        lens[ag].append(n)
        if failed:
            st[ag]["prompt_template_fail"] += 1
        if n > PROMPT_CAP[ag]:
            st[ag][f"dropped_prompt_over_{PROMPT_CAP[ag]}"] += 1
            del pools[ag][k]
    return {ag: max(v) for ag, v in lens.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=40960)
    ap.add_argument("--n-val", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20261106)
    ap.add_argument("--tokenizer", default=SFT_HF)
    ap.add_argument("--val-drop-groups", default="genrm,identity", help="검증 파일에 쓰지 않을 그룹 (쉼표). 빈 문자열이면 모두 쓴다")
    ap.add_argument("--procs", type=int, default=64)
    a = ap.parse_args()
    if os.path.exists(a.out):
        print(f"출력이 이미 있다: {a.out} — 지우고 다시 실행한다")
        return 1
    bench = load_bench()
    pools, st = collect(bench)
    max_prompt_before_caps = apply_prompt_caps(pools, st, a.tokenizer, a.procs)
    rng = random.Random(a.seed)
    keys = {ag: sorted(p) for ag, p in pools.items()}
    for ag in sorted(keys):
        rng.shuffle(keys[ag])
    total = a.n_train + a.n_val
    group_pool = {g: sum(len(keys.get(x, [])) for x in ags) for g, (ags, _) in GROUPS.items()}
    want_g = {g: round(total * share) for g, (_, share) in GROUPS.items()}
    surplus = 0
    for g in want_g:
        if want_g[g] > group_pool[g]:
            surplus += want_g[g] - group_pool[g]
            want_g[g] = group_pool[g]
    want_g["if"] += total - sum(want_g.values())  # 모자란 그룹의 남는 몫 + 반올림 오차
    if want_g["if"] > group_pool["if"]:
        raise SystemExit(f"IF 풀 부족: 필요 {want_g['if']} > {group_pool['if']}")
    want = {}
    for g, (ags, _) in GROUPS.items():
        sizes = {x: len(keys.get(x, [])) for x in ags}
        tot = sum(sizes.values())
        alloc = {x: (want_g[g] * s // tot if tot else 0) for x, s in sizes.items()}
        rest = want_g[g] - sum(alloc.values())
        for x in sorted(sizes, key=lambda x: -sizes[x]):
            if rest <= 0:
                break
            if alloc[x] < sizes[x]:
                alloc[x] += 1
                rest -= 1
        want.update(alloc)
    val, train = [], []
    for ag, n in want.items():
        n_val = round(a.n_val * n / total)
        chosen = keys.get(ag, [])[:n]
        val += [(ag, k) for k in chosen[:n_val]]
        train += [(ag, k) for k in chosen[n_val:]]
    rng.shuffle(val)
    while len(val) > a.n_val:
        train.append(val.pop())
    while len(val) < a.n_val:
        val.append(train.pop())
    rng.shuffle(train)
    rng.shuffle(val)
    drop_val = {g for g in a.val_drop_groups.split(",") if g}
    val = [(ag, k) for ag, k in val if AGENT2GROUP[ag] not in drop_val]
    out_val = a.out.replace(".jsonl", "_val.jsonl")
    for path, items in ((a.out, train), (out_val, val)):
        with open(path, "wb") as fo:
            for ag, k in items:
                fo.write(orjson.dumps(pools[ag][k]) + b"\n")
        by_agent = collections.Counter(ag for ag, _ in items)
        by_group = collections.Counter(AGENT2GROUP[ag] for ag, _ in items)
        info = {
            "file": path, "rows": len(items), "seed": a.seed,
            "by_group": {g: [by_group[g], round(by_group[g] / len(items), 4)] for g in GROUPS},
            "by_agent": dict(sorted(by_agent.items())),
            "unique_pool_by_agent": {ag: len(p) for ag, p in sorted(pools.items())},
            "group_pool": group_pool, "target_by_group_train_plus_val": want_g,
            "filter_counts": {ag: dict(c) for ag, c in sorted(st.items())},
            "bench_items": collections.Counter(b for b, _, _ in bench),
            "prompt_caps": {"train": CAP_TRAIN, "genrm": CAP_GENRM, "llm_judge": CAP_LLM_JUDGE},
            "max_prompt_tokens_before_caps": max_prompt_before_caps,
            "val_dropped_groups": sorted(drop_val),
        }
        with open(path + ".stats.json", "w") as fs:
            json.dump(info, fs, indent=1, ensure_ascii=False)
        print(json.dumps({k: info[k] for k in ("file", "rows", "by_group")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
