"""MOPD1 블렌드 (RL_PLAN.md 결정 55) — CPU, 결정론적.

teacher 넷이 RL 에 쓴 블렌드를 다시 쓴다 (Ultra MOPD 프롬프트의 78% 가 앞 단계 프롬프트 · GLM-5 · Cascade 2 · Xiaomi 관행, FRONTIER.md 2026-11-09):
  General   teacher_general_v1 (teacher_general/s1 step 640 의 학습 블렌드)
  STEM      teacher_stem_v2 (teacher_stem main2* step 735)
  Code·Math teacher_codemath_code_all_v1 · teacher_codemath_math_all_v1 (s2r 228 의 2단계 블렌드 행을 담은 코드·수학 풀)
  도구      teacher_tool_v1 (teacher_tool step 400)
영역 몫은 --n-train 의 General 30 · STEM 30 · Code·Math 30 (코드 15 · 수학 15) · 도구 10 %. 영역 안은 원천의 agent 별 고유 행 수에 비례해
층화 추출하고 반복하지 않는다 (모자라면 실패).

- teacher 검증셋(*_val.jsonl) 행과 겹치는 과제는 뺀다. 중복은 agent + 입력 메시지 + 도구 정의의 해시 (영역·원천 사이 전부).
- 프롬프트 길이 상한: alpha chat template 렌더 토큰 ≤ --prompt-cap (기본 32,768 = 시퀀스 65,536 − 생성 32,768). teacher 를 GPU 1장에
  올리면 시퀀스 64K 가 상한이다 (코드 감사 2026-11-09 메모리 추정). 원천 집계상 넘는 행은 도구 706/25,600 뿐이다.
- 벤치 대조: IFBench 294 (13-gram) · IFEval 541 (13-gram) · SimpleQA 1,000 · LogicKor 42 (8-gram) 의 n-gram 을 절반 이상 담은 행은 뺀다
  (build_general_blend.contaminated 와 같은 규칙). IFBench 는 General teacher 의 평가 벤치라 이 블렌드부터 넣는다.
- teacher 배정: 각 행은 자기 원천 영역의 teacher 로 간다. agent 이름 하나가 두 영역에 나오면 실패한다 — OPD 는 agent 이름으로 teacher 를 고른다.
  레시피에 넣을 `teacher_model_by_agent_name` 을 <out>.routing.yaml 로 쓴다.
- `_ng_` 로 시작하는 Gym 기록 키는 지운다. 출력 옆에 .stats.json 을 남긴다.

  $NRL_ROOT/clean_run.sh $NRL_ROOT/venv-driver/bin/python -I examples/configs/alpha/tools/build_mopd1_blend.py \
      --out <dir>/mopd1_v1.jsonl [--n-train 61440] [--n-val 256] [--seed 20261109] [--procs 32] </dev/null
"""

import argparse
import collections
import json
import os
import random
import sys

import orjson

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_general_blend as G  # noqa: E402  task_key · grams · contaminated · load_bench
import measure_blend_prompt_lengths as M  # noqa: E402  렌더·토큰 수 계산을 같은 코드로 한다

TB = f"{G.AB}/teacher_blends"
NRL_RESULTS = "/home/work/vidsearch/repos/project_s/NeMo-RL/results/alpha"
TEACHERS = {
    "general": f"{NRL_RESULTS}/teacher_general/hf/hf_step_00640",
    "stem": f"{NRL_RESULTS}/teacher_stem/hf/hf_step_00735",
    "codemath": f"{NRL_RESULTS}/teacher_codemath_s2r/hf/hf_step_00228",
    "tool": f"{NRL_RESULTS}/teacher_tool/hf/hf_step_00400",
}
# 몫 이름: (teacher, 원천 파일, 비중)
SHARES = {
    "general": ("general", [f"{TB}/teacher_general_v1.jsonl"], 0.30),
    "stem": ("stem", [f"{TB}/teacher_stem_v2.jsonl"], 0.30),
    "code": ("codemath", [f"{TB}/teacher_codemath_code_all_v1.jsonl"], 0.15),
    "math": ("codemath", [f"{TB}/teacher_codemath_math_all_v1.jsonl"], 0.15),
    "tool": ("tool", [f"{TB}/teacher_tool_v1.jsonl"], 0.10),
}
VAL_FILES = [f"{TB}/teacher_general_v1_val.jsonl", f"{TB}/teacher_stem_v2_val.jsonl",
             f"{TB}/teacher_codemath_s2_v1_val.jsonl", f"{TB}/teacher_tool_v1_val.jsonl"]
IFBENCH = "/home/work/vidsearch/tools/ifbench/IFBench/data/IFBench_test.jsonl"


def load_bench():
    items = G.load_bench()
    for line in open(IFBENCH):
        gs = G.grams(json.loads(line)["prompt"], 13)
        if gs:
            items.append(("ifbench", 13, gs))
    return items


def bench_index(bench):
    index = {8: collections.defaultdict(set), 13: collections.defaultdict(set)}
    for bi, (_, n, gs) in enumerate(bench):
        for g in gs:
            index[n][g].add(bi)
    return index


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=61440)
    ap.add_argument("--n-val", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--prompt-cap", type=int, default=32768)
    ap.add_argument("--tokenizer", default=G.SFT_HF)
    ap.add_argument("--procs", type=int, default=32)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    st = collections.defaultdict(collections.Counter)

    val_keys = set()
    for f in VAL_FILES:
        for line in open(f):
            r = orjson.loads(line)
            val_keys.add(G.task_key(r["agent_ref"]["name"], r))
    bench = load_bench()
    index = bench_index(bench)

    # 1) 원천 → 몫별 agent 풀 (중복·검증셋·벤치 제거)
    seen, pools, agent_teacher = set(), {}, {}
    for share, (teacher, files, _) in SHARES.items():
        pool = collections.defaultdict(dict)
        for f in files:
            for line in open(f):
                r = orjson.loads(line)
                ag = r["agent_ref"]["name"]
                k = G.task_key(ag, r)
                if k in val_keys:
                    st[share]["dropped_teacher_val"] += 1
                    continue
                if k in seen:
                    st[share]["dropped_duplicate"] += 1
                    continue
                b = G.contaminated(G.last_user_text(r), bench, index)
                if b:
                    st[share][f"dropped_bench_{b}"] += 1
                    continue
                seen.add(k)
                for key in [x for x in r if x.startswith("_ng_")]:
                    del r[key]
                prev = agent_teacher.setdefault(ag, teacher)
                if prev != teacher:
                    raise SystemExit(f"agent {ag} appears under teachers {prev} and {teacher} — OPD routes by agent name")
                pool[ag][k] = r
        pools[share] = pool

    # 2) 프롬프트 길이 상한 (렌더 토큰)
    items = [(s, ag, k, orjson.dumps(r).decode()) for s, p in pools.items() for ag, rows in p.items() for k, r in rows.items()]
    from multiprocessing import Pool
    with Pool(a.procs, initializer=M._init, initargs=(a.tokenizer,)) as pool:
        res = pool.map(M._one, [line for *_, line in items], chunksize=64)
    maxlen = collections.Counter()
    for (s, ag, k, _), (_, n, failed) in zip(items, res):
        if failed:
            st[s]["prompt_template_fail"] += 1
        if n > a.prompt_cap:
            st[s][f"dropped_prompt_over_{a.prompt_cap}"] += 1
            del pools[s][ag][k]
        else:
            maxlen[s] = max(maxlen[s], n)

    # 3) 몫별 층화 추출 (검증 먼저, 같은 비율)
    total = a.n_train + a.n_val
    train, val = [], []
    for share, (teacher, _, w) in SHARES.items():
        pool = pools[share]
        want = round(total * w)
        avail = {ag: len(rows) for ag, rows in pool.items() if rows}
        if sum(avail.values()) < want:
            raise SystemExit(f"share {share}: need {want}, pool has {sum(avail.values())}")
        base = {ag: int(want * c / sum(avail.values())) for ag, c in avail.items()}
        rest = want - sum(base.values())
        for ag in sorted(avail, key=lambda x: -avail[x])[:rest]:
            base[ag] += 1
        picked = []
        for ag, cnt in sorted(base.items()):
            keys = sorted(pool[ag])
            rng.shuffle(keys)
            picked += [pool[ag][k] for k in keys[:cnt]]
            st[share][f"agent_{ag}"] = cnt
        rng.shuffle(picked)
        n_val = round(a.n_val * w)
        val += picked[:n_val]
        train += picked[n_val:]
        st[share]["rows"] = want
        st[share]["max_prompt_tokens"] = maxlen[share]
    rng.shuffle(train)
    rng.shuffle(val)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "wb") as f:
        for r in train:
            f.write(orjson.dumps(r) + b"\n")
    vpath = a.out.replace(".jsonl", "_val.jsonl")
    with open(vpath, "wb") as f:
        for r in val:
            f.write(orjson.dumps(r) + b"\n")
    used = collections.Counter(r["agent_ref"]["name"] for r in train + val)
    routing = {ag: TEACHERS[agent_teacher[ag]] for ag in sorted(used)}
    with open(a.out.replace(".jsonl", ".routing.yaml"), "w") as f:
        f.write("# OPD teacher_model_by_agent_name — build_mopd1_blend.py 가 블렌드의 agent 전수로 만든다\n")
        for ag, path in routing.items():
            f.write(f"{ag}: {path}\n")
    out = {
        "args": vars(a), "train_rows": len(train), "val_rows": len(val), "teachers": TEACHERS,
        "by_teacher": dict(collections.Counter(agent_teacher[r["agent_ref"]["name"]] for r in train)),
        "by_agent": dict(used.most_common()), "routing": routing,
        "per_share": {s: dict(sorted(c.items())) for s, c in st.items()},
    }
    with open(a.out.replace(".jsonl", ".stats.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: out[k] for k in ("train_rows", "val_rows", "by_teacher")}, ensure_ascii=False))
    for s, c in out["per_share"].items():
        print(s, {k: v for k, v in c.items() if not k.startswith("agent_")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
