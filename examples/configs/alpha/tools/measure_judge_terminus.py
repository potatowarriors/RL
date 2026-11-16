"""terminal 피벗 judge 게이트 (TJ1) — 운영 Gym `terminus_judge` 검증 코드를 살아 있는 judge 모델에 그대로 돌린다 (RL_PLAN.md 결정 65).

지키려는 것: 논문 방식 보상 (유사도 0.95 빠른 통과, 미달이면 등가 LLM judge) 의 judge 가 **문자열 유사도보다 나은 판정**을 한다 —
글자가 달라도 같은 일을 하는 명령은 같다고 하고, 글자가 비슷해도 다른 일을 하는 명령은 다르다고 한다.
v1 (2026-11-16 적대적 검토 "보완 필요") 은 음성이 전부 유사도 < 0.5, 양성이 전부 > 0.77 이라 "유사도 ≥ 0.7" 로 판정하는 가짜 judge 도 통과했다.
v2 는 유사도로 갈리지 않는 세트를 넣고, 유사도 임계 judge 가 이 게이트에서 FAIL 하는지를 `selftest` 로 먼저 확인한다.

쌍은 원본 데이터 (`Nemotron-RL-Agentic-Terminal-Pivot-v1`) 에서 결정론적으로 만든다. 후보는 정답 JSON 의 analysis · plan 을 그대로 두고 commands 만 바꾼다.
  양성 (label 1)
    P_fast     keystrokes 는 같고 duration 만 다름 — 유사도 1.0 빠른 통과, judge 호출 0
    P_redirect 각 명령 끝에 ` 2>&1` (유사도 < 0.95)
    P_reorder  서로 독립인 읽기 명령 (cat · ls · head · tail · wc · stat · file) 3개 이상의 순서를 뒤집음 (유사도 < 0.7)
    P_split    `a && b` 한 명령을 두 명령으로 나눔 / 연속 두 명령을 `a && b` 로 합침
    P_ctrl     제어 키의 개행 유무 (`C-c` ↔ `C-c\\n`), 플래그 순서 (`-la` ↔ `-al`, `-rn` ↔ `-nr`)
  음성 (label 0)
    N_wait     후보 = 빈 keystrokes 대기
    N_other    후보 = 다른 과제 행의 정답 commands (유사도 < 0.5)
    N_repeat   후보 = 이력의 직전 assistant 턴 commands (정답과 유사도 < 0.5)
    N_goldwait 정답이 빈 대기 + 후보 = 다른 과제의 실제 명령
    N_drop     정답 명령 4개 이상 중 앞 절반만 수행 (유사도 0.5~0.95 — 범위 차이)
    N_target   읽기 명령의 대상 경로를 같은 행의 다른 경로가 아닌 다른 과제의 경로로 바꿈 (유사도 0.5~0.95 — 다른 파일)
    N_verb     읽기 명령 (cat · head · tail · ls) 을 `rm -f` 로 바꿈 (유사도 ≥ 0.5 — 다른 행동 유형)
  보고만 (label None)
    U_*        MOPD1 학생의 실제 프로파일링 출력, 유사도 구간별 (0.1~0.3 · 0.3~0.5 · 0.5~0.7 · 0.7~0.95)
    K_earlytc  정답 task_complete false + 후보 = 정답 commands 그대로 + task_complete true — Gym 의 완료 검사가 단방향이라 통과한다 (알려진 구멍)
검증 경로: TerminusJudgeResourcesServer 를 프로세스 안에 만들고 verify() 를 부른다 — 스키마 · 완료 일치 · 유사도 · judge 프롬프트 · 판정 파싱이 운영 코드다.
설정 = Gym `terminus_judge.yaml` + 레시피 덮어쓰기 (`env.nemo_gym.terminus_judge_resources_server`). 병합 결과가 기대값과 다르면 중단한다.

  python measure_judge_terminus.py build --out pairs.jsonl [--profile-rollouts full/rollouts.jsonl]
  python measure_judge_terminus.py selftest --pairs pairs.jsonl          # CPU: 유사도 임계 judge 가 FAIL 해야 한다 (하나라도 PASS 면 종료 코드 1)
  <Gym>/.venv/bin/python measure_judge_terminus.py run --gym-root <Gym> --recipe <recipe.yaml> --head 127.0.0.1:11000 \
        --pairs pairs.jsonl --out results.jsonl [--concurrency 64]
  python measure_judge_terminus.py flips --a results_c64.jsonl --b results_c256.jsonl   # 같은 쌍 재실행의 판정 뒤집힘률
종료 코드: run 은 기준을 하나라도 못 넘으면 1.
"""

import argparse
import asyncio
import collections
import json
import random
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

SOURCE = "/home/work/Datasets/LL_datasets/posttraining/RL/Nemotron-RL-Agentic-Terminal-Pivot-v1/atcb_terminal_pivot_release_final_v2.jsonl"
POS = ["P_fast", "P_redirect", "P_reorder", "P_split", "P_ctrl"]
NEG = ["N_wait", "N_other", "N_repeat", "N_goldwait", "N_drop", "N_target", "N_verb"]
# (세트, 기대 보상, 최소 일치율). 유사도로 갈리지 않는 세트 (P_reorder · N_drop · N_target · N_verb) 가 핵심이다.
CRITERIA = [("P_fast", 1.0, 1.00), ("P_redirect", 1.0, 0.90), ("P_reorder", 1.0, 0.80), ("P_split", 1.0, 0.85), ("P_ctrl", 1.0, 0.85),
            ("N_wait", 0.0, 0.97), ("N_other", 0.0, 0.97), ("N_repeat", 0.0, 0.97), ("N_goldwait", 0.0, 0.97),
            ("N_drop", 0.0, 0.85), ("N_target", 0.0, 0.90), ("N_verb", 0.0, 0.95)]
MAX_PARSE_FAIL = 0.02
MAX_ERROR_RATE = 0.01
MAX_FLIP = 0.02
EXPECTED_CFG = {"enable_string_similarity": True, "string_similarity_threshold": 0.95, "enable_llm_judge": True, "check_twice_swap": False}
READ_VERBS = ("cat ", "ls ", "head ", "tail ", "wc ", "stat ", "file ")


def keystrokes(d: dict) -> list[str]:
    return [c.get("keystrokes", "") for c in d.get("commands", []) if isinstance(c, dict) and "keystrokes" in c]


def sim(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, "".join(a), "".join(b)).ratio()


def with_commands(gold: dict, commands: list[dict], **extra) -> str:
    d = dict(gold)
    d["commands"] = commands
    d.update(extra)
    return json.dumps(d, ensure_ascii=False, indent=2)


def simple_read(k: str) -> bool:
    b = k.strip()
    return k.endswith("\n") and "\n" not in b and b.startswith(READ_VERBS) and not any(x in b for x in ("|", ">", "<", ";", "&", "$(", "`"))


def build(args) -> None:
    rng = random.Random(args.seed)
    rows = []
    with open(SOURCE) as f:
        for line in f:
            r = json.loads(line)
            g = json.loads(r["expected_answer"])
            r["_gold"], r["_ks"] = g, keystrokes(g)
            rows.append(r)
    rng.shuffle(rows)
    real = [r for r in rows if not r["_gold"].get("task_complete") and any(k.strip() for k in r["_ks"])]
    wait = [r for r in rows if not r["_gold"].get("task_complete") and r["_ks"] and all(not k.strip() for k in r["_ks"])]
    paths = [m for r in real[:4000] for k in r["_ks"] for m in re.findall(r"(?<![\w.])/[\w./-]{6,}", k)]
    pairs, n = [], collections.Counter()

    def add(set_name, label, row, cand_text, note=""):
        cand = json.loads(cand_text) if cand_text.strip().startswith("{") else {}
        s = sim(row["_ks"], keystrokes(cand)) if isinstance(cand, dict) else None
        pairs.append({"set": set_name, "label": label, "candidate_text": cand_text, "note": note, "sim": s,
                      "row": {k: row[k] for k in ("uuid", "task_name", "responses_create_params", "expected_answer", "metadata")}})
        n[set_name] += 1

    def other_task(row, pool):
        for _ in range(300):
            o = pool[rng.randrange(len(pool))]
            if o["task_name"] != row["task_name"] and sim(o["_ks"], row["_ks"]) < 0.5:
                return o
        return None

    def full(name, cap):
        return n[name] >= cap

    for r in real:
        g, ks, cmds = r["_gold"], r["_ks"], r["_gold"]["commands"]
        # --- 양성 ---
        if not full("P_ctrl", args.n_small):
            new, changed = [], False
            for c in cmds:
                k = c.get("keystrokes", "")
                k2 = k
                if k.strip() in ("C-c", "C-d", "C-z", "q"):
                    k2 = k.strip() if k.endswith("\n") else k + "\n"
                else:
                    k2 = re.sub(r"(^|\s)-la(\s)", r"\1-al\2", k, count=1)
                    if k2 == k:
                        k2 = re.sub(r"(^|\s)-rn(\s)", r"\1-nr\2", k, count=1)
                changed |= k2 != k
                new.append(dict(c, keystrokes=k2))
            if changed and sim(ks, [c["keystrokes"] for c in new]) < 0.95:
                add("P_ctrl", 1, r, with_commands(g, new)); continue
        if not full("P_reorder", args.n) and len(cmds) >= 3 and all(simple_read(k) for k in ks) and len(set(ks)) == len(ks):
            new = list(reversed(cmds))
            if sim(ks, [c["keystrokes"] for c in new]) < 0.7:
                add("P_reorder", 1, r, with_commands(g, new)); continue
        if not full("P_split", args.n):
            new = None
            if len(cmds) == 1 and ks[0].endswith("\n") and ks[0].count(" && ") == 1 and "\n" not in ks[0].strip() and not any(x in ks[0] for x in ("|", "(", "<<", ";")):
                a, b = ks[0].rstrip("\n").split(" && ")
                if not a.strip().startswith("cd "):
                    new = [dict(cmds[0], keystrokes=a + "\n"), dict(cmds[0], keystrokes=b + "\n")]
            elif len(cmds) == 2 and all(simple_read(k) for k in ks):
                new = [dict(cmds[0], keystrokes=ks[0].rstrip("\n") + " && " + ks[1])]
            if new is not None and sim(ks, [c["keystrokes"] for c in new]) < 0.95:
                add("P_split", 1, r, with_commands(g, new)); continue
        if not full("P_fast", args.n_small):
            add("P_fast", 1, r, with_commands(g, [dict(c, duration=float(c.get("duration", 1.0)) + 1.0) for c in cmds])); continue
        if not full("P_redirect", args.n):
            new = []
            for c in cmds:
                k = c.get("keystrokes", "")
                body = k[:-1] if k.endswith("\n") else k
                ok = body.strip() and "\n" not in body and ">" not in body and "<<" not in body and "&" not in body and not body.strip().startswith("C-")
                new.append(dict(c, keystrokes=(body + " 2>&1" + ("\n" if k.endswith("\n") else "")) if ok else k))
            ck = [c["keystrokes"] for c in new]
            if ck != ks and sim(ks, ck) < 0.95:
                add("P_redirect", 1, r, with_commands(g, new)); continue
        # --- 음성: 유사도로 갈리지 않는 것 ---
        if not full("N_drop", args.n) and len(cmds) >= 4 and len(set(ks)) == len(ks):
            new = cmds[: len(cmds) // 2]
            if 0.5 <= sim(ks, [c["keystrokes"] for c in new]) < 0.95:
                add("N_drop", 0, r, with_commands(g, new), f"{len(new)}/{len(cmds)} commands"); continue
        if not full("N_target", args.n) and len(cmds) <= 2 and all(simple_read(k) for k in ks):
            new, changed = [], 0
            for c in cmds:
                k = c["keystrokes"]
                m = list(re.finditer(r"(?<![\w.])/[\w./-]{6,}", k))
                if m:
                    p = paths[rng.randrange(len(paths))]
                    if p.rstrip("/") != m[-1].group(0).rstrip("/") and p.rstrip("/").split("/")[-1] != m[-1].group(0).rstrip("/").split("/")[-1]:
                        k = k[: m[-1].start()] + p + k[m[-1].end():]
                        changed += 1
                new.append(dict(c, keystrokes=k))
            if changed == len(cmds) and 0.5 <= sim(ks, [c["keystrokes"] for c in new]) < 0.95:  # 모든 명령의 대상이 바뀐 쌍만
                add("N_target", 0, r, with_commands(g, new)); continue
        if not full("N_verb", args.n) and len(cmds) <= 2 and all(simple_read(k) and k.strip().startswith(("cat ", "head ", "tail ")) for k in ks):
            new = [dict(c, keystrokes="rm -f " + c["keystrokes"].strip().split(" ")[-1] + "\n") for c in cmds]
            if 0.5 <= sim(ks, [c["keystrokes"] for c in new]) < 0.95:
                add("N_verb", 0, r, with_commands(g, new)); continue
        # --- 음성: 쉬운 것 ---
        if not full("N_wait", args.n):
            add("N_wait", 0, r, with_commands(g, [{"keystrokes": "", "duration": 10.0}])); continue
        if not full("N_other", args.n):
            o = other_task(r, real)
            if o is not None:
                add("N_other", 0, r, with_commands(g, o["_gold"]["commands"]), f"other={o['task_name']}"); continue
        if not full("N_repeat", args.n):
            prev = [m for m in r["responses_create_params"]["input"] if m.get("role") == "assistant"]
            if prev and isinstance(prev[-1].get("content"), str):
                try:
                    p = json.loads(prev[-1]["content"])
                except json.JSONDecodeError:
                    p = None
                if isinstance(p, dict) and any(k.strip() for k in keystrokes(p)) and sim(ks, keystrokes(p)) < 0.5:
                    add("N_repeat", 0, r, with_commands(g, p["commands"])); continue
        if not full("K_earlytc", args.n_small):
            add("K_earlytc", None, r, with_commands(g, cmds, task_complete=True)); continue
    for r in wait:
        if full("N_goldwait", args.n):
            break
        o = other_task(r, real)
        if o is not None:
            add("N_goldwait", 0, r, with_commands(r["_gold"], o["_gold"]["commands"]), f"other={o['task_name']}")
    if args.profile_rollouts:
        by_uuid = {r["uuid"]: r for r in rows}
        bands = {"U_10_30": (0.1, 0.3), "U_30_50": (0.3, 0.5), "U_50_70": (0.5, 0.7), "U_70_95": (0.7, 0.95)}
        cand = collections.defaultdict(list)
        with open(args.profile_rollouts) as f:
            for line in f:
                x = json.loads(line)
                s_ = x.get("similarity_score")
                if not isinstance(s_, (int, float)) or not x.get("task_complete_check_passed") or x["uuid"] not in by_uuid:
                    continue
                for name, (lo, hi) in bands.items():
                    if lo <= s_ < hi:
                        cand[name].append((x["uuid"], x["model_output"], s_))
        for name in bands:
            rng.shuffle(cand[name])
            for u, text, s_ in cand[name][: args.n_unlabeled]:
                add(name, None, by_uuid[u], text, f"sim={s_:.3f}")
    with open(args.out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    short = [s for s in POS + NEG if n[s] < (args.n_small if s in ("P_fast", "P_ctrl") else args.n) * 0.5]
    stats = {}
    for s in POS + NEG:
        v = sorted(p["sim"] for p in pairs if p["set"] == s and p["sim"] is not None)
        if v:
            stats[s] = {"n": len(v), "sim_min": round(v[0], 3), "sim_med": round(v[len(v) // 2], 3), "sim_max": round(v[-1], 3)}
    print(json.dumps({"pairs": len(pairs), "by_set": n, "sim": stats, "short_sets": short, "seed": args.seed}, ensure_ascii=False, indent=1))
    if short:
        raise SystemExit(f"세트가 목표의 절반에 못 미친다: {short}")


def judge_verdict(by: dict, parse_fail_rate: float, error_rate: float, truncated: int) -> list[str]:
    """세트별 결과 (reward 목록 · judge 호출 수 · 음성 실패 사유) 로 기준 위반 목록을 만든다. selftest 와 run 이 같이 쓴다."""
    fails = []
    for name, want, min_rate in CRITERIA:
        xs = by.get(name)
        if not xs:
            fails.append(f"{name}: 쌍 없음")
            continue
        if want == 0.0:  # 음성은 judge 가 "다르다" 고 한 것만 맞음으로 센다 (파싱 실패 · 오류의 보상 0 은 맞음이 아니다)
            ok = sum(1 for x in xs if x["reward"] == 0.0 and x["failure_reason"] == "judge_not_equivalent")
        else:
            ok = sum(1 for x in xs if x["reward"] == 1.0)
        if ok / len(xs) < min_rate:
            fails.append(f"{name}: 일치율 {ok / len(xs):.3f} < {min_rate}")
        if name == "P_fast":
            if any(x["judge_calls"] for x in xs):
                fails.append("P_fast: 빠른 통과인데 judge 가 호출됨")
        elif any(x["judge_calls"] != 1 for x in xs):
            fails.append(f"{name}: judge 호출이 1 이 아닌 쌍 {sum(1 for x in xs if x['judge_calls'] != 1)}")
    if parse_fail_rate > MAX_PARSE_FAIL:
        fails.append(f"판정 파싱 실패율 {parse_fail_rate:.3f} > {MAX_PARSE_FAIL}")
    if error_rate > MAX_ERROR_RATE:
        fails.append(f"예외율 {error_rate:.3f} > {MAX_ERROR_RATE}")
    if truncated:
        fails.append(f"판정문이 출력 상한에서 잘린 호출 {truncated}")
    return fails


def selftest(args) -> int:
    """유사도 임계로 판정하는 가짜 judge 가 이 게이트에서 FAIL 하는지 본다 (CPU). 항상 같음 · 항상 다름도 본다."""
    pairs = [json.loads(line) for line in open(args.pairs)]
    fakes = {f"sim>={t}": (lambda s, t=t: s >= t) for t in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9)}
    fakes["always_equal"] = lambda s: True
    fakes["always_not_equal"] = lambda s: False
    bad = []
    for name, fn in fakes.items():
        by = collections.defaultdict(list)
        for p in pairs:
            if p["label"] is None or p["sim"] is None:
                continue
            if p["sim"] >= 0.95:
                by[p["set"]].append({"reward": 1.0, "failure_reason": "none", "judge_calls": 0})
            else:
                eq = fn(p["sim"])
                by[p["set"]].append({"reward": 1.0 if eq else 0.0, "failure_reason": "none" if eq else "judge_not_equivalent", "judge_calls": 1})
        fails = judge_verdict(by, 0.0, 0.0, 0)
        print(f"{name:18s} -> {'PASS (게이트가 못 잡음)' if not fails else 'FAIL'}  {'; '.join(fails[:4])}")
        if not fails:
            bad.append(name)
    print("selftest:", "OK — 가짜 judge 전부 FAIL" if not bad else f"NG — PASS 한 가짜 judge {bad}")
    return 1 if bad else 0


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
    for k, v in EXPECTED_CFG.items():
        if getattr(cfg, k) != v:
            raise SystemExit(f"병합된 설정이 기대와 다르다: {k}={getattr(cfg, k)!r} (기대 {v!r})")
    jp = cfg.judge_responses_create_params
    if jp.temperature != 0.0 or jp.max_output_tokens != args.max_output_tokens:
        raise SystemExit(f"judge 생성 설정이 기대와 다르다: temperature={jp.temperature} max_output_tokens={jp.max_output_tokens}")
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

    async def one(i, p):
        t0 = time.time()
        r = p["row"]
        base = {"i": i, "set": p["set"], "label": p["label"], "uuid": r["uuid"], "note": p.get("note"), "sim_build": p.get("sim")}
        try:
            req = TerminusJudgeVerifyRequest.model_validate({**r, "response": response(p["candidate_text"]).model_dump()})
            async with sem:
                vr = await server.verify(req)
        except Exception as e:  # 한 쌍의 예외가 전체를 멈추지 않게 세고 넘어간다 — 예외율이 기준이다
            return {**base, "error": f"{type(e).__name__}: {str(e)[:200]}", "reward": None, "failure_reason": "exception", "judge_calls": 0,
                    "verdicts": [], "parse_fail": [], "out_tokens": [], "judge_texts": [], "latency_s": round(time.time() - t0, 2)}
        evs = list(vr.judge_evaluations)
        texts = []
        for e in evs:
            try:
                texts.append(e.response.output[-1].content[-1].text)
            except Exception:
                texts.append("")
        return {**base, "error": None, "reward": vr.reward, "failure_reason": str(vr.failure_reason.value if vr.failure_reason else None),
                "similarity": vr.similarity_score, "judge_calls": len(evs), "verdicts": [e.verdict_label for e in evs],
                "parse_fail": [e.verdict_label is None for e in evs],
                "out_tokens": [int(getattr(getattr(e.response, "usage", None), "output_tokens", 0) or 0) for e in evs],
                "judge_texts": texts, "latency_s": round(time.time() - t0, 2)}

    t0 = time.time()
    results = await asyncio.gather(*[one(i, p) for i, p in enumerate(pairs)])
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
        summ[name] = {"n": len(xs), "label": xs[0]["label"], "reward_1_rate": round(sum(x["reward"] == 1.0 for x in xs) / len(xs), 4),
                      "judge_calls": calls, "parse_fail": sum(sum(x["parse_fail"]) for x in xs), "errors": sum(1 for x in xs if x["error"]),
                      "failure_reasons": collections.Counter(x["failure_reason"] for x in xs),
                      "out_tokens_mean": round(sum(sum(x["out_tokens"]) for x in xs) / max(1, calls), 1),
                      "latency_med_s": sorted(x["latency_s"] for x in xs)[len(xs) // 2]}
    calls = sum(x["judge_calls"] for x in results)
    pf = sum(sum(x["parse_fail"]) for x in results)
    errs = sum(1 for x in results if x["error"])
    trunc = sum(1 for x in results for t in x["out_tokens"] if t >= args.max_output_tokens)
    fails = judge_verdict({k: v for k, v in by.items() if v[0]["label"] is not None}, pf / max(1, calls), errs / len(results), trunc)
    summ["_all"] = {"pairs": len(results), "judge_calls": calls, "parse_fail": pf, "parse_fail_rate": round(pf / max(1, calls), 4),
                    "errors": errs, "truncated_judge_outputs": trunc, "wall_s": round(wall, 1), "judge_calls_per_s": round(calls / max(wall, 1e-9), 1),
                    "concurrency": args.concurrency, "verdict": "PASS" if not fails else "FAIL", "fails": fails}
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    print(json.dumps(summ, indent=1, ensure_ascii=False))
    return 0 if not fails else 1


def flips(args) -> int:
    a = {x["i"]: x for x in map(json.loads, open(args.a))}
    b = {x["i"]: x for x in map(json.loads, open(args.b))}
    both = [i for i in a if i in b and a[i]["judge_calls"] and b[i]["judge_calls"]]
    fl = [i for i in both if a[i]["reward"] != b[i]["reward"]]
    by = collections.Counter(a[i]["set"] for i in fl)
    rate = len(fl) / max(1, len(both))
    print(json.dumps({"judged_pairs": len(both), "flips": len(fl), "flip_rate": round(rate, 4), "by_set": by,
                      "verdict": "PASS" if rate <= MAX_FLIP else "FAIL", "max_flip": MAX_FLIP}, ensure_ascii=False))
    return 0 if rate <= MAX_FLIP else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--profile-rollouts")
    b.add_argument("--n", type=int, default=300)
    b.add_argument("--n-small", type=int, default=100)
    b.add_argument("--n-unlabeled", type=int, default=100)
    b.add_argument("--seed", type=int, default=20261116)
    s = sub.add_parser("selftest")
    s.add_argument("--pairs", required=True)
    r = sub.add_parser("run")
    r.add_argument("--gym-root", required=True)
    r.add_argument("--recipe", required=True)
    r.add_argument("--head", required=True)
    r.add_argument("--pairs", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--concurrency", type=int, default=64)
    r.add_argument("--max-output-tokens", type=int, default=2048)
    fl = sub.add_parser("flips")
    fl.add_argument("--a", required=True)
    fl.add_argument("--b", required=True)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a)
        return 0
    if a.cmd == "selftest":
        return selftest(a)
    if a.cmd == "flips":
        return flips(a)
    return asyncio.run(run_pairs(a))


if __name__ == "__main__":
    raise SystemExit(main())
