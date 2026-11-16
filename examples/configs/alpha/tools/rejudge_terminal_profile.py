"""terminal 피벗 프로파일링 롤아웃을 새 채점기로 다시 채점한다 (RL_PLAN.md 결정 65 · 66) — 생성은 다시 하지 않는다.

입력: MOPD1 학생의 프로파일링 롤아웃 (`$NRL_ROOT/p0_measure/term_pivot/{full,rest}/rollouts.jsonl`, 31,111 행 × 8).
채점: 게이트 TJ1 과 같은 경로 — alpha 플러그인 `AlphaTerminusJudgeResourcesServer.verify()` 를 프로세스 안에서 부른다
      (스키마 → 완료 선언 양방향 → 완전 일치 빠른 통과 → Gemma judge rubrics_v4). `measure_judge_terminus.load_server` 를 그대로 쓴다.
처리:
  - 정답에 제어 키 (`C-c` · `C-d` · `C-z`) 가 든 행은 채점하지 않는다 (`excluded_control_key`, 결정 66).
  - 모델을 두 번 이상 부른 롤아웃 (response 에 reasoning 항목 2개 이상) 은 judge 없이 보상 0 (`recall`) —
    학습은 `max_steps: 1` 이라 첫 호출에서 메시지가 없으면 빈 답 0 점이다 (KNOWN_ISSUES 2026-11-14).
  - 저장된 유사도와 다시 계산한 유사도가 다르면 센다 (`sim_mismatch`) — 입력 재구성이 원 채점과 같은지의 확인.
출력 한 줄 = {uuid, ri, reward, reason, sim, judge_calls, out_tokens}. 이미 있는 (uuid, ri) 는 건너뛴다 (재개).

  NEMO_GYM_EXTRA_ROOTS=<gym_plugins> <Gym terminus_judge venv>/python rejudge_terminal_profile.py --gym-root <Gym> --recipe <recipe.yaml> \
      --head 127.0.0.1:11312 --rollouts full/rollouts.jsonl rest/rollouts.jsonl --out rejudge.jsonl [--concurrency 512] [--limit N]
"""

import argparse
import asyncio
import collections
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

CONTROL = ("C-c", "C-d", "C-z")


def gate_module():
    spec = importlib.util.spec_from_file_location("measure_judge_terminus", Path(__file__).with_name("measure_judge_terminus.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


async def main_async(args) -> int:
    gate = gate_module()
    args.max_output_tokens = 2048
    server, cfg = gate.load_server(args)
    from resources_servers.terminus_judge.app import TerminusJudgeVerifyRequest

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                x = json.loads(line)
                done.add((x["uuid"], x["ri"]))
    print(f"cfg threshold={cfg.string_similarity_threshold} judge={cfg.enable_llm_judge} resume={len(done)}", flush=True)
    sem = asyncio.Semaphore(args.concurrency)
    stats = collections.Counter()
    fo = open(args.out, "a")
    t0 = time.time()

    async def one(x):
        uuid, ri = x["uuid"], x["_ng_rollout_index"]
        try:
            gold = json.loads(x["expected_answer"])
            ks = [c.get("keystrokes", "") for c in gold.get("commands", []) if isinstance(c, dict)]
        except Exception:
            ks = []
        if any(k.strip() in CONTROL for k in ks):
            rec = {"uuid": uuid, "ri": ri, "reward": None, "reason": "excluded_control_key", "sim": None, "judge_calls": 0, "out_tokens": 0}
        elif sum(1 for o in x["response"]["output"] if o.get("type") == "reasoning") > 1:
            rec = {"uuid": uuid, "ri": ri, "reward": 0.0, "reason": "recall", "sim": None, "judge_calls": 0, "out_tokens": 0}
        else:
            try:
                req = TerminusJudgeVerifyRequest.model_validate({k: x[k] for k in ("uuid", "responses_create_params", "response", "expected_answer", "metadata")})
                async with sem:
                    vr = await server.verify(req)
                evs = list(vr.judge_evaluations)
                rec = {"uuid": uuid, "ri": ri, "reward": vr.reward, "reason": str(vr.failure_reason.value if vr.failure_reason else None),
                       "sim": vr.similarity_score, "judge_calls": len(evs),
                       "out_tokens": sum(int(getattr(getattr(e.response, "usage", None), "output_tokens", 0) or 0) for e in evs)}
                old = x.get("similarity_score")
                if isinstance(old, (int, float)) and isinstance(vr.similarity_score, (int, float)) and abs(old - vr.similarity_score) > 1e-9:
                    stats["sim_mismatch"] += 1
            except Exception as e:
                rec = {"uuid": uuid, "ri": ri, "reward": None, "reason": f"exception:{type(e).__name__}", "sim": None, "judge_calls": 0, "out_tokens": 0}
        stats[rec["reason"]] += 1
        stats["judge_calls"] += rec["judge_calls"]
        fo.write(json.dumps(rec) + "\n")

    n = 0
    batch = []
    for path in args.rollouts:
        with open(path) as f:
            for line in f:
                x = json.loads(line)
                if (x["uuid"], x["_ng_rollout_index"]) in done:
                    continue
                batch.append(x)
                n += 1
                if len(batch) >= args.batch:
                    await asyncio.gather(*[one(b) for b in batch])
                    batch = []
                    fo.flush()
                    print(f"[{time.strftime('%H:%M:%S')}] {n} done · {stats['judge_calls']} judge calls · {stats['judge_calls'] / max(1, time.time() - t0):.1f}/s", flush=True)
                if args.limit and n >= args.limit:
                    break
        if args.limit and n >= args.limit:
            break
    if batch:
        await asyncio.gather(*[one(b) for b in batch])
    fo.close()
    summ = {"new": n, "resumed": len(done), "wall_s": round(time.time() - t0, 1), **stats}
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ))
    exc = sum(v for k, v in stats.items() if k.startswith("exception"))
    return 0 if exc <= 0.01 * max(1, n) and stats["sim_mismatch"] == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gym-root", required=True)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--rollouts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4000)
    ap.add_argument("--limit", type=int, default=0)
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
