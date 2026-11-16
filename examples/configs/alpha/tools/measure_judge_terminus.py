"""terminal 피벗 judge 게이트 (TJ1) — 운영 Gym `terminus_judge` 검증 코드를 살아 있는 judge 모델에 그대로 돌린다 (RL_PLAN.md 결정 65).

지키려는 것: 논문 방식 보상 (유사도 0.95 통과, 미달이면 등가 LLM judge) 에서 judge 가 "빈 대기 · 직전 명령 반복 · 무관한 명령" 을
정답과 같다고 판정하지 않는다. 판정이 고장 나면 (전부 같음 · 전부 다름 · 파싱 실패) 이 게이트가 FAIL 해야 한다.

쌍은 원본 데이터 (`Nemotron-RL-Agentic-Terminal-Pivot-v1`) 에서 결정론적으로 만든다. 후보는 정답 JSON 의 analysis · plan 을 그대로 두고
commands 만 바꾼다 (judge 프롬프트가 commands 에 집중하라고 하므로 다른 필드 차이를 없앤다).
  P_fast     (label 1) 정답과 keystrokes 가 같고 duration 만 다름 — 유사도 1.0 빠른 통과, judge 호출 0 이어야 한다
  P_redirect (label 1) 각 명령 끝에 ` 2>&1` 을 붙임 — 유사도 < 0.95 인 쌍만, judge 가 같다고 해야 한다
  N_wait     (label 0) 후보 = 빈 keystrokes 대기 한 개
  N_other    (label 0) 후보 = 다른 과제 행의 정답 commands
  N_repeat   (label 0) 후보 = 이력의 직전 assistant 턴 commands (정답과 유사도 < 0.5 인 행만)
  N_goldwait (label 0) 정답이 빈 대기인 행 + 후보 = 다른 과제 행의 실제 명령
  U_hi · U_lo (라벨 없음, 보고만) MOPD1 학생의 실제 프로파일링 출력 — 유사도 0.7~0.95 · 0.3~0.5 구간. 판정률의 단조성 (U_hi > U_lo) 을 본다
검증 경로: TerminusJudgeResourcesServer 를 프로세스 안에 만들고 verify() 를 부른다 — 스키마 · 완료 일치 · 유사도 · judge 프롬프트 · 판정 파싱이
전부 운영 코드다. 설정 = Gym `terminus_judge.yaml` 에 레시피 덮어쓰기 (`env.nemo_gym.terminus_judge_resources_server`) 를 합친 것.

필요: `judge_model` responses_api_models 서버가 있는 `gym env start` (head 주소) 와 Gym venv.
  <Gym>/.venv/bin/python measure_judge_terminus.py build --out pairs.jsonl [--profile-rollouts full/rollouts.jsonl]
  <Gym>/.venv/bin/python measure_judge_terminus.py run --gym-root <Gym> --recipe <recipe.yaml> --head 127.0.0.1:11000 \
        --pairs pairs.jsonl --out results.jsonl
종료 코드: run 은 기준을 하나라도 못 넘으면 1.
"""

import argparse
import asyncio
import collections
import json
import random
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

SOURCE = "/home/work/Datasets/LL_datasets/posttraining/RL/Nemotron-RL-Agentic-Terminal-Pivot-v1/atcb_terminal_pivot_release_final_v2.jsonl"
# 기준 (라벨 있는 세트). (세트, 기대 보상, 최소 일치율)
CRITERIA = [("P_fast", 1.0, 1.00), ("P_redirect", 1.0, 0.90), ("N_wait", 0.0, 0.95), ("N_other", 0.0, 0.95),
            ("N_repeat", 0.0, 0.95), ("N_goldwait", 0.0, 0.95)]
MAX_PARSE_FAIL = 0.02


def keystrokes(d: dict) -> list[str]:
    return [c.get("keystrokes", "") for c in d.get("commands", []) if isinstance(c, dict) and "keystrokes" in c]


def sim(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, "".join(a), "".join(b)).ratio()


def with_commands(gold: dict, commands: list[dict]) -> str:
    d = dict(gold)
    d["commands"] = commands
    return json.dumps(d, ensure_ascii=False, indent=2)


def build(args) -> None:
    rng = random.Random(args.seed)
    rows = []
    with open(SOURCE) as f:
        for line in f:
            r = json.loads(line)
            g = json.loads(r["expected_answer"])
            r["_gold"] = g
            r["_ks"] = keystrokes(g)
            rows.append(r)
    rng.shuffle(rows)
    real = [r for r in rows if not r["_gold"].get("task_complete") and any(k.strip() for k in r["_ks"])]
    wait = [r for r in rows if not r["_gold"].get("task_complete") and r["_ks"] and all(not k.strip() for k in r["_ks"])]
    pairs = []

    def add(set_name, label, row, cand_text, note=""):
        pairs.append({"set": set_name, "label": label, "candidate_text": cand_text, "note": note,
                      "row": {k: row[k] for k in ("uuid", "task_name", "responses_create_params", "expected_answer", "metadata")}})

    def other_task(row, pool):
        for _ in range(200):
            o = pool[rng.randrange(len(pool))]
            if o["task_name"] != row["task_name"] and sim(o["_ks"], row["_ks"]) < 0.5:
                return o
        return None

    n = collections.Counter()
    for r in real:
        g = r["_gold"]
        if n["P_fast"] < args.n_fast:
            cmds = [dict(c, duration=float(c.get("duration", 1.0)) + 1.0) for c in g["commands"]]
            add("P_fast", 1, r, with_commands(g, cmds)); n["P_fast"] += 1
            continue
        if n["P_redirect"] < args.n_pos:
            cmds = []
            for c in g["commands"]:
                k = c.get("keystrokes", "")
                body = k[:-1] if k.endswith("\n") else k
                ok = body.strip() and "\n" not in body and ">" not in body and "<<" not in body and not body.strip().startswith("C-")
                cmds.append(dict(c, keystrokes=(body + " 2>&1" + ("\n" if k.endswith("\n") else "")) if ok else k))
            ck = [c["keystrokes"] for c in cmds]
            if ck != r["_ks"] and sim(r["_ks"], ck) < 0.95:
                add("P_redirect", 1, r, with_commands(g, cmds), f"sim={sim(r['_ks'], ck):.3f}"); n["P_redirect"] += 1
                continue
        if n["N_wait"] < args.n_neg:
            add("N_wait", 0, r, with_commands(g, [{"keystrokes": "", "duration": 10.0}])); n["N_wait"] += 1
            continue
        if n["N_other"] < args.n_neg:
            o = other_task(r, real)
            if o is not None:
                add("N_other", 0, r, with_commands(g, o["_gold"]["commands"]), f"other={o['task_name']}"); n["N_other"] += 1
                continue
        if n["N_repeat"] < args.n_neg:
            prev = [m for m in r["responses_create_params"]["input"] if m.get("role") == "assistant"]
            if prev and isinstance(prev[-1].get("content"), str):
                try:
                    p = json.loads(prev[-1]["content"])
                except json.JSONDecodeError:
                    p = None
                if isinstance(p, dict) and any(k.strip() for k in keystrokes(p)) and sim(r["_ks"], keystrokes(p)) < 0.5:
                    add("N_repeat", 0, r, with_commands(g, p["commands"]), f"sim_gold_prev={sim(r['_ks'], keystrokes(p)):.3f}")
                    n["N_repeat"] += 1
                    continue
    for r in wait:
        if n["N_goldwait"] >= args.n_neg:
            break
        o = other_task(r, real)
        if o is not None:
            add("N_goldwait", 0, r, with_commands(r["_gold"], o["_gold"]["commands"]), f"other={o['task_name']}"); n["N_goldwait"] += 1
    if args.profile_rollouts:
        by_uuid = {r["uuid"]: r for r in rows}
        want = {"U_hi": (0.7, 0.95), "U_lo": (0.3, 0.5)}
        with open(args.profile_rollouts) as f:
            for line in f:
                if all(n[s] >= args.n_unlabeled for s in want):
                    break
                x = json.loads(line)
                s_ = x.get("similarity_score")
                if not isinstance(s_, (int, float)) or not x.get("task_complete_check_passed") or x["uuid"] not in by_uuid:
                    continue
                for name, (lo, hi) in want.items():
                    if lo <= s_ < hi and n[name] < args.n_unlabeled and rng.random() < 0.05:
                        add(name, None, by_uuid[x["uuid"]], x["model_output"], f"sim={s_:.3f}"); n[name] += 1
    with open(args.out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(json.dumps({"pairs": len(pairs), "by_set": n, "seed": args.seed}, ensure_ascii=False))


def load_server(args):
    sys.path.insert(0, args.gym_root)
    from omegaconf import OmegaConf

    from nemo_gym.config_types import BaseServerConfig
    from nemo_gym.server_utils import setup_server_client
    from resources_servers.terminus_judge.app import TerminusJudgeResourcesServer, TerminusJudgeResourcesServerConfig

    gym = Path(args.gym_root) / "resources_servers" / "terminus_judge"
    base = OmegaConf.load(gym / "configs" / "terminus_judge.yaml")["terminus_judge_resources_server"]["resources_servers"]["terminus_judge"]
    recipe = OmegaConf.load(args.recipe)
    over = recipe.env.nemo_gym["terminus_judge_resources_server"]["resources_servers"]["terminus_judge"]
    merged = OmegaConf.to_container(OmegaConf.merge(base, over), resolve=True)
    merged.update(host="127.0.0.1", port=1, entrypoint="app.py", name="terminus_judge")
    merged["judge_prompt_template_fpath"] = str(gym / merged["judge_prompt_template_fpath"])
    cfg = TerminusJudgeResourcesServerConfig.model_validate(merged)
    host, port = args.head.split(":")
    sc = setup_server_client(BaseServerConfig(host=host, port=int(port)))
    return TerminusJudgeResourcesServer(config=cfg, server_client=sc), cfg


async def run_pairs(args) -> int:
    from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseOutputMessage, NeMoGymResponseOutputText
    from resources_servers.terminus_judge.app import TerminusJudgeVerifyRequest

    server, cfg = load_server(args)
    print("terminus_judge cfg:", cfg.model_dump(include={"judge_model_server", "judge_responses_create_params", "enable_string_similarity",
          "string_similarity_threshold", "enable_llm_judge", "check_twice_swap", "judge_prompt_template_fpath"}), flush=True)
    pairs = [json.loads(line) for line in open(args.pairs)]
    sem = asyncio.Semaphore(args.concurrency)

    def response(text):
        return NeMoGymResponse(id="resp", created_at=0.0, model="policy", object="response", parallel_tool_calls=False,
                               tool_choice="none", tools=[], output=[NeMoGymResponseOutputMessage(
                                   id="msg", role="assistant", status="completed", type="message",
                                   content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")])])

    async def one(p):
        t0 = time.time()
        r = p["row"]
        req = TerminusJudgeVerifyRequest.model_validate({**r, "response": response(p["candidate_text"]).model_dump()})
        async with sem:
            vr = await server.verify(req)
        evs = list(vr.judge_evaluations)
        texts = []
        for e in evs:
            try:
                texts.append(e.response.output[-1].content[-1].text)
            except Exception:
                texts.append("")
        return {"set": p["set"], "label": p["label"], "uuid": r["uuid"], "note": p.get("note"), "reward": vr.reward,
                "failure_reason": str(vr.failure_reason.value if vr.failure_reason else None), "similarity": vr.similarity_score,
                "judge_calls": len(evs), "verdicts": [e.verdict_label for e in evs], "parse_fail": [e.verdict_label is None for e in evs],
                "out_tokens": [int(getattr(getattr(e.response, "usage", None), "output_tokens", 0) or 0) for e in evs],
                "judge_texts": texts, "latency_s": round(time.time() - t0, 2)}

    t0 = time.time()
    results = await asyncio.gather(*[one(p) for p in pairs])
    wall = time.time() - t0
    with open(args.out, "w") as f:
        for x in results:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    by = collections.defaultdict(list)
    for x in results:
        by[x["set"]].append(x)
    summ = {}
    for name, xs in sorted(by.items()):
        calls = sum(x["judge_calls"] for x in xs)
        summ[name] = {"n": len(xs), "label": xs[0]["label"], "reward_1": sum(x["reward"] == 1.0 for x in xs),
                      "reward_1_rate": round(sum(x["reward"] == 1.0 for x in xs) / len(xs), 4), "judge_calls": calls,
                      "parse_fail": sum(sum(x["parse_fail"]) for x in xs),
                      "failure_reasons": collections.Counter(x["failure_reason"] for x in xs),
                      "out_tokens_mean": round(sum(sum(x["out_tokens"]) for x in xs) / max(1, calls), 1)}
    calls = sum(x["judge_calls"] for x in results)
    pf = sum(sum(x["parse_fail"]) for x in results)
    fails = []
    for name, want, min_rate in CRITERIA:
        xs = by.get(name, [])
        if not xs:
            fails.append(f"{name}: 쌍 없음")
            continue
        rate = sum(x["reward"] == want for x in xs) / len(xs)
        if rate < min_rate:
            fails.append(f"{name}: 기대 보상 {want} 일치율 {rate:.3f} < {min_rate}")
    if summ.get("P_fast", {}).get("judge_calls", 0) != 0:
        fails.append("P_fast: 빠른 통과인데 judge 가 호출됨")
    for name in ("P_redirect", "N_wait", "N_other", "N_repeat", "N_goldwait"):
        s = summ.get(name)
        if s and s["judge_calls"] < s["n"] * 0.9:
            fails.append(f"{name}: judge 호출 {s['judge_calls']} < 쌍의 90% — judge 를 거치지 않고 판정됨")
    if calls and pf / calls > MAX_PARSE_FAIL:
        fails.append(f"판정 파싱 실패율 {pf / calls:.3f} > {MAX_PARSE_FAIL}")
    if "U_hi" in summ and "U_lo" in summ and summ["U_hi"]["reward_1_rate"] <= summ["U_lo"]["reward_1_rate"]:
        fails.append("U_hi 판정률이 U_lo 보다 높지 않음 (단조성)")
    summ["_all"] = {"pairs": len(results), "judge_calls": calls, "parse_fail": pf, "parse_fail_rate": round(pf / max(1, calls), 4),
                    "wall_s": round(wall, 1), "verdict": "PASS" if not fails else "FAIL", "fails": fails}
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    print(json.dumps(summ, indent=1, ensure_ascii=False))
    return 0 if not fails else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--profile-rollouts")
    b.add_argument("--n-fast", type=int, default=50)
    b.add_argument("--n-pos", type=int, default=150)
    b.add_argument("--n-neg", type=int, default=150)
    b.add_argument("--n-unlabeled", type=int, default=150)
    b.add_argument("--seed", type=int, default=20261116)
    r = sub.add_parser("run")
    r.add_argument("--gym-root", required=True)
    r.add_argument("--recipe", required=True)
    r.add_argument("--head", required=True)
    r.add_argument("--pairs", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--concurrency", type=int, default=64)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a)
        return 0
    return asyncio.run(run_pairs(a))


if __name__ == "__main__":
    raise SystemExit(main())
