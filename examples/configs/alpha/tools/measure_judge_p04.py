"""P0-4 judge validation — run the production Gym judge code against a live judge model server.

Pairs come from the STEM / Code·Math blend rows (positive = the gold answer itself, negative = another row's gold).
The judge paths are the Gym 0.6.0 resource servers instantiated in-process, so prompt template, sampling params,
per-record answer extraction (template_metadata.output_regex) and verdict parsing are the production code:
- equivalence_llm_judge: LLMJudgeResourcesServer.verify()  (STEM: Science-v1 / Ultra reasoning / ExamQA, ns_tools science)
- math_with_judge: LibraryJudgeMathResourcesServer._generate_judge_evaluation() in both answer orders
  (production asks the second order only when the first says equal; both are recorded here)
Server configs = the Gym config yaml merged with the recipe overrides (judge_model_server, judge_responses_create_params).

Needs a running `gym env start` whose global config has a `judge_model` responses_api_models server, and the Gym
math_with_judge venv (nemo_gym + math_verify) with the Gym root on sys.path:

  <Gym>/resources_servers/math_with_judge/.venv/bin/python measure_judge_p04.py build --stem-blend B --math-pool M --out pairs.jsonl
  ... measure_judge_p04.py run --gym-root <Gym> --recipe teacher_stem_alpha.yaml --head 127.0.0.1:11310 \
        --pairs pairs.jsonl --out results.jsonl
  ... measure_judge_p04.py throughput --gym-root <Gym> --recipe R --head H --pairs pairs.jsonl --concurrency 64 256
"""

import argparse
import asyncio
import collections
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

# Candidate answer wrappers; the one whose output_regex extraction returns the answer exactly is used.
WRAPPERS = [
    "\\boxed{%s}", "**%s**", "((%s))", "<final_answer>%s</final_answer>", "Final Answer: ||%s||",
    "(Answer: %s)", "Answer is [%s]", "[Answer: %s]", "<<%s>>",
]


def clean_answer(s: str) -> str:
    """Strip dataset bold-marker artifacts (`** X`, `X.**`) the way a model would write the bare answer."""
    s = s.strip()
    while True:
        t = s.strip().strip("*").strip()
        if t == s:
            return s
        s = t


def extract_like_gym(text: str, regex: str | None) -> str:
    """Same rule as equivalence_llm_judge._extract_last_assistant_text (last match, first non-empty group)."""
    if not regex:
        return text.strip()
    matches = list(re.finditer(regex, text, flags=re.MULTILINE | re.DOTALL))
    if not matches:
        return text.strip()
    m = matches[-1]
    for g in m.groups():
        if isinstance(g, str) and g.strip():
            return g.strip()
    return m.group(0).strip()


def wrap_for(regex: str | None, answer: str) -> str | None:
    if not regex:
        return answer
    for w in WRAPPERS:
        text = "The final answer is below.\n\n" + (w % answer)
        if extract_like_gym(text, regex) == answer:
            return text
    return None


_PLACEHOLDER = "ANSWER_PLACEHOLDER_7d1c"


def gold_regex_defect(regex: str | None, answer: str) -> tuple[str, str | None]:
    """Does the gold answer, written in the row's own requested format, reach the judge whole?

    Returns (status, what the judge sees):
      ok             - extraction returns the answer (or differs only by `**` markers / a trailing period)
      fallback       - the regex does not match at all, so Gym judges the whole message (not a truncation)
      truncated      - the regex matches but returns a fragment (closing char inside the answer; KNOWN_ISSUES 2026-10-10)
      unknown_format - no wrapper in WRAPPERS produces this regex's format
    """
    if not regex:
        return "ok", answer
    own = next((w for w in WRAPPERS if extract_like_gym(w % _PLACEHOLDER, regex) == _PLACEHOLDER), None)
    if own is None:
        return "unknown_format", None
    a = clean_answer(answer)
    text = "The final answer is below.\n\n" + (own % a)
    if not re.search(regex, text, flags=re.MULTILINE | re.DOTALL):
        return "fallback", text
    got = extract_like_gym(text, regex)

    def loose(s: str) -> str:
        return s.replace("**", "").strip().rstrip(".").strip()

    return ("ok" if got == a or loose(got) == loose(a) else "truncated"), got


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def build(args) -> None:
    rng = random.Random(args.seed)
    pairs, stats = [], collections.Counter()
    rows = [json.loads(l) for l in open(args.stem_blend)]
    judge_rows = [r for r in rows if r["agent_ref"]["name"] in ("equivalence_llm_judge_simple_agent", "ns_tools_simple_agent")]

    def keep(r):
        return {k: r[k] for k in ("uuid", "responses_create_params", "expected_answer", "template_metadata", "metadata",
                                  "agent_ref", "verifier_type", "pool_source") if k in r}

    def add_eq(set_name, pool, n_pos, n_neg, group_key):
        rng.shuffle(pool)
        groups = collections.defaultdict(list)
        for r in pool:
            groups[group_key(r)].append(r)
        pos = neg = 0
        for r in pool:
            if pos >= n_pos:
                break
            regex = (r.get("template_metadata") or {}).get("output_regex")
            text = wrap_for(regex, clean_answer(r["expected_answer"]))
            if text is None:
                stats[f"{set_name}: gold not recoverable through its own output_regex"] += 1
                continue
            pairs.append({"set": set_name, "label": 1, "row": keep(r), "candidate_text": text,
                          "candidate": clean_answer(r["expected_answer"])})
            pos += 1
        for r in reversed(pool):
            if neg >= n_neg:
                break
            regex = (r.get("template_metadata") or {}).get("output_regex")
            mine = norm(clean_answer(r["expected_answer"]))
            donors = [d for d in groups[group_key(r)] if d is not r and norm(clean_answer(d["expected_answer"])) != mine
                      and mine not in norm(clean_answer(d["expected_answer"])) and norm(clean_answer(d["expected_answer"])) not in mine]
            rng.shuffle(donors)
            for d in donors:
                cand = clean_answer(d["expected_answer"])
                text = wrap_for(regex, cand)
                if text is not None:
                    pairs.append({"set": set_name, "label": 0, "row": keep(r), "candidate_text": text,
                                  "candidate": cand, "donor_uuid": d.get("uuid")})
                    neg += 1
                    break
        stats[f"{set_name}: pos"] = pos
        stats[f"{set_name}: neg"] = neg

    sci = [r for r in judge_rows if r.get("pool_source") == "sci"]
    add_eq("sci_equivalence", sci, 100, 100, lambda r: ((r.get("metadata") or {}).get("topic"), (r.get("metadata") or {}).get("subtopic")))
    for ps in ("ultra", "examqa"):
        add_eq(f"equivalence_{ps}", [r for r in judge_rows if r.get("pool_source") == ps], 50, 50, lambda r: "all")

    math = [json.loads(l) for l in open(args.math_pool)]
    rng.shuffle(math)
    mk = lambda r: {"uuid": r.get("pool_key"), "question": r["question"], "expected_answer": r["expected_answer"],
                    "dataset": r.get("dataset")}
    pos_rows, neg_rows, near_rows = math[:100], math[100:200], [r for r in math[200:] if re.fullmatch(r"-?\d+", r["expected_answer"].strip())][:50]
    for r in pos_rows:
        pairs.append({"set": "math_judge_as_is", "label": 1, "row": mk(r), "candidate": r["expected_answer"]})
        pairs.append({"set": "math_judge_sentence", "label": 1, "row": mk(r),
                      "candidate": f"Therefore, the final answer is {r['expected_answer']}."})
    for r in neg_rows:
        mine = norm(r["expected_answer"])
        d = next(d for d in math[300:] if norm(d["expected_answer"]) != mine)
        math.remove(d)
        pairs.append({"set": "math_judge_other_gold", "label": 0, "row": mk(r), "candidate": d["expected_answer"]})
    for r in near_rows:
        pairs.append({"set": "math_judge_off_by_one", "label": 0, "row": mk(r), "candidate": str(int(r["expected_answer"].strip()) + 1)})
    with open(args.out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    by_set = collections.Counter(f"{p['set']} label={p['label']}" for p in pairs)
    print(json.dumps({"pairs": len(pairs), **dict(stats), **dict(by_set)}, ensure_ascii=False, indent=1))


def load_servers(args, eq_concurrency=None):
    sys.path.insert(0, args.gym_root)
    from omegaconf import OmegaConf

    from nemo_gym.config_types import BaseServerConfig
    from nemo_gym.server_utils import setup_server_client
    from resources_servers.equivalence_llm_judge.app import LLMJudgeResourcesServer, LLMJudgeResourcesServerConfig
    from resources_servers.math_with_judge.app import LibraryJudgeMathResourcesServer, LibraryJudgeMathResourcesServerConfig

    recipe = OmegaConf.load(args.recipe)
    gym = Path(args.gym_root) / "resources_servers"
    out = {}
    for name, cfg_cls in (("equivalence_llm_judge", LLMJudgeResourcesServerConfig), ("math_with_judge", LibraryJudgeMathResourcesServerConfig)):
        base = OmegaConf.load(gym / name / "configs" / f"{name}.yaml")[name]["resources_servers"][name]
        over = recipe.env.nemo_gym[name]["resources_servers"][name]
        merged = OmegaConf.to_container(OmegaConf.merge(base, over), resolve=True)
        merged.update(host="127.0.0.1", port=1, entrypoint="app.py", name=name)
        if name == "equivalence_llm_judge":
            merged["judge_prompt_template_fpath"] = str(gym / name / merged["judge_prompt_template_fpath"])
            if eq_concurrency is not None:
                merged["judge_endpoint_max_concurrency"] = eq_concurrency
        out[name] = cfg_cls.model_validate(merged)
    host, port = args.head.split(":")
    sc = setup_server_client(BaseServerConfig(host=host, port=int(port)))
    eq = LLMJudgeResourcesServer(config=out["equivalence_llm_judge"], server_client=sc)
    mj = LibraryJudgeMathResourcesServer(config=out["math_with_judge"], server_client=sc)
    return eq, mj, out


def judge_text(ev) -> str:
    try:
        return ev.response.output[-1].content[-1].text
    except Exception:
        return ""


def out_tokens(ev) -> int:
    u = getattr(ev.response, "usage", None)
    return int(getattr(u, "output_tokens", 0) or 0)


async def run_pairs(args) -> None:
    from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseOutputMessage, NeMoGymResponseOutputText
    from resources_servers.equivalence_llm_judge.app import LLMJudgeVerifyRequest

    eq, mj, cfgs = load_servers(args)
    print("equivalence cfg:", cfgs["equivalence_llm_judge"].model_dump(include={"judge_responses_create_params", "judge_model_server",
          "use_per_record_regex", "extraction_length_threshold", "check_full_generation_on_fail", "check_twice_swap"}), flush=True)
    print("math cfg:", cfgs["math_with_judge"].model_dump(include={"judge_responses_create_params", "judge_model_server", "should_use_judge"}), flush=True)
    pairs = [json.loads(l) for l in open(args.pairs)]
    sem = asyncio.Semaphore(64)  # math server has no judge semaphore; 64 = equivalence server default

    def response(text):
        return NeMoGymResponse(id="resp", created_at=0.0, model="policy", object="response", parallel_tool_calls=False,
                               tool_choice="none", tools=[], output=[NeMoGymResponseOutputMessage(
                                   id="msg", role="assistant", status="completed", type="message",
                                   content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")])])

    async def one(p):
        t0 = time.time()
        r = p["row"]
        if p["set"].startswith("math_judge"):
            async with sem:
                lib, extracted = await mj._verify_answer_with_library_async(r["expected_answer"], p["candidate"])
                eq1, ev1 = await mj._generate_judge_evaluation(r["question"], r["expected_answer"], p["candidate"])
                eq2, ev2 = await mj._generate_judge_evaluation(r["question"], p["candidate"], r["expected_answer"])
            evs = [ev1, ev2]
            res = {"library_reward": lib, "order1_equal": eq1, "order2_equal": eq2,
                   "production_judge_reward": float(eq1 and eq2),
                   "parse_fail": [mj.JUDGE_EQUAL_LABEL not in judge_text(e) and mj.JUDGE_NOT_EQUAL_LABEL not in judge_text(e) for e in evs]}
        else:
            body = {k: v for k, v in r.items() if k in ("uuid", "responses_create_params", "expected_answer", "template_metadata", "metadata", "verifier_type")}
            req = LLMJudgeVerifyRequest.model_validate({**body, "response": response(p["candidate_text"]).model_dump()})
            vr = await eq.verify(req)
            evs = list(vr.judge_evaluations)
            extracted = extract_like_gym(p["candidate_text"], (r.get("template_metadata") or {}).get("output_regex"))
            res = {"reward": vr.reward, "verdicts": [e.verdict_label for e in evs], "extracted_ok": extracted == p["candidate"],
                   "parse_fail": [e.verdict_label is None for e in evs]}
        res.update(set=p["set"], label=p["label"], uuid=r.get("uuid"), latency_s=round(time.time() - t0, 2),
                   out_tokens=[out_tokens(e) for e in evs], judge_texts=[judge_text(e) for e in evs], candidate=p["candidate"],
                   expected=r["expected_answer"])
        return res

    t0 = time.time()
    results = await asyncio.gather(*[one(p) for p in pairs])
    wall = time.time() - t0
    with open(args.out, "w") as f:
        for x in results:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    summ = summarize(results)
    summ["wall_s"] = round(wall, 1)
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    print(json.dumps(summ, indent=1, ensure_ascii=False))


def summarize(results):
    s = {}
    by = collections.defaultdict(list)
    for x in results:
        by[x["set"]].append(x)
    calls = parse_fail = 0
    for name, xs in sorted(by.items()):
        d = {"n": len(xs), "label": xs[0]["label"]}
        if name.startswith("math_judge"):
            o1 = sum(x["order1_equal"] for x in xs); o2 = sum(x["order2_equal"] for x in xs)
            prod = sum(x["production_judge_reward"] for x in xs)
            agree = sum(x["order1_equal"] == x["order2_equal"] for x in xs)
            d.update(order1_equal=o1, order2_equal=o2, production_reward_1=prod, order_consistent=agree,
                     math_verify_passes=sum(x["library_reward"] > 0.5 for x in xs))
        else:
            d.update(reward_1=sum(x["reward"] > 0.5 for x in xs), extracted_ok=sum(x["extracted_ok"] for x in xs))
        pf = sum(sum(x["parse_fail"]) for x in xs); nc = sum(len(x["parse_fail"]) for x in xs)
        d.update(parse_fail=pf, judge_calls=nc, out_tokens_mean=round(statistics.mean(t for x in xs for t in x["out_tokens"]), 1))
        calls += nc; parse_fail += pf
        s[name] = d
    s["_all"] = {"judge_calls": calls, "parse_fail": parse_fail, "parse_fail_rate": round(parse_fail / max(calls, 1), 4)}
    return s


async def throughput(args) -> None:
    pairs = [json.loads(l) for l in open(args.pairs) if json.loads(l)["set"].startswith(("sci_", "equivalence_"))]
    for c in args.concurrency:
        eq, _, _ = load_servers(args, eq_concurrency=c)
        reqs = [pairs[i % len(pairs)] for i in range(args.requests)]

        async def one(p):
            r = p["row"]
            from resources_servers.equivalence_llm_judge.app import _extract_question_text
            from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
            q = _extract_question_text(NeMoGymResponseCreateParamsNonStreaming.model_validate(r["responses_create_params"]), None)
            t = time.time()
            ok, ev = await eq._generate_judge_evaluation(question=q, expected_answer=r["expected_answer"], generated_answer=p["candidate"])
            return time.time() - t, out_tokens(ev), ev.verdict_label is None

        t0 = time.time()
        res = await asyncio.gather(*[one(p) for p in reqs])
        wall = time.time() - t0
        lat = sorted(x[0] for x in res)
        print(json.dumps({"concurrency": c, "requests": len(res), "wall_s": round(wall, 1), "req_per_s": round(len(res) / wall, 2),
                          "out_tok_per_s": round(sum(x[1] for x in res) / wall, 1), "out_tokens_mean": round(statistics.mean(x[1] for x in res), 1),
                          "latency_p50_s": round(lat[len(lat) // 2], 2), "latency_p90_s": round(lat[int(len(lat) * 0.9)], 2),
                          "parse_fail": sum(x[2] for x in res)}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--stem-blend", required=True); b.add_argument("--math-pool", required=True)
    b.add_argument("--out", required=True); b.add_argument("--seed", type=int, default=0)
    for name in ("run", "throughput"):
        p = sub.add_parser(name)
        p.add_argument("--gym-root", required=True); p.add_argument("--recipe", required=True); p.add_argument("--head", required=True)
        p.add_argument("--pairs", required=True)
        if name == "run":
            p.add_argument("--out", required=True)
        else:
            p.add_argument("--concurrency", type=int, nargs="+", default=[64, 256]); p.add_argument("--requests", type=int, default=512)
    args = ap.parse_args()
    if args.cmd == "build":
        build(args)
    elif args.cmd == "run":
        asyncio.run(run_pairs(args))
    else:
        asyncio.run(throughput(args))


if __name__ == "__main__":
    main()
