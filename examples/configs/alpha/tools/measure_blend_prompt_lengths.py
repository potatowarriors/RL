"""Gym 블렌드 행의 첫 턴 렌더 프롬프트 토큰 수 — 데이터 게이트 (CPU, NeMo-RL driver venv).

NeMo-RL 은 첫 턴 프롬프트가 vLLM `max_model_len` 을 넘는 행을 건너뛰지 않고 런 전체를 멈춘다 (`nemo_rl/environments/nemo_gym.py:826`,
`docs/KNOWN_ISSUES.md` 2026-10-07). 그래서 런 전에 블렌드의 최대 프롬프트 길이를 `max_total_sequence_length` 와 비교한다.

Responses API 입력을 Gym 변환과 같은 방식으로 chat 메시지로 바꾼 뒤 alpha chat template(add_generation_prompt)으로 렌더해 센다.
  - `reasoning.summary` 는 다음 assistant 의 reasoning_content, 연속 `function_call` 은 한 assistant 의 tool_calls(arguments 는 dict),
    `function_call_output` 의 본문은 `output` 필드다. 앞의 둘을 빠뜨리면 크게 과소 측정된다 (2026-10-07 첫 측정 22.5K vs 실제 39.3K).
  - 템플릿이 실패한 행은 원문 JSON 토큰 수로 대신 세고 `tmpl_fail` 로 표시한다.

  $NRL_ROOT/clean_run.sh $NRL_ROOT/venv-driver/bin/python -I examples/configs/alpha/tools/measure_blend_prompt_lengths.py \
      --blend <blend.jsonl> --tokenizer <hfmodel_00NNNNN> --out <stats.json> [--max-len 131072] </dev/null
종료 코드: --max-len 을 주면 그보다 긴 행이 있을 때 1.
"""

import argparse
import collections
import json
import sys
from multiprocessing import Pool

TOKENIZER = None
_tok = None


def _init(path):
    global _tok
    from transformers import AutoTokenizer

    _tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def _text(c):
    if isinstance(c, list):
        return "".join(x.get("text", "") for x in c if isinstance(x, dict))
    return c or ""


def to_messages(items):
    msgs, pending_reason = [], None
    for m in items:
        t = m.get("type", "message")
        if t == "message":
            msgs.append({"role": m["role"], "content": _text(m.get("content"))})
        elif t == "reasoning":
            pending_reason = (pending_reason or "") + "".join(x.get("text", "") for x in m.get("summary") or [])
        elif t == "function_call":
            args = m.get("arguments") or "{}"
            try:
                args = json.loads(args) if isinstance(args, str) else args
            except json.JSONDecodeError:
                args = {"_raw": args}
            tc = {"type": "function", "function": {"name": m["name"], "arguments": args}, "id": m.get("call_id")}
            last = msgs[-1] if msgs else None
            if last is not None and last["role"] == "assistant" and last.get("tool_calls") is not None and pending_reason is None:
                last["tool_calls"].append(tc)
            else:
                a = {"role": "assistant", "content": "", "tool_calls": [tc]}
                if pending_reason:
                    a["reasoning_content"] = pending_reason
                msgs.append(a)
                pending_reason = None
        elif t == "function_call_output":
            out = m.get("output")
            msgs.append({"role": "tool", "content": out if isinstance(out, str) else json.dumps(out), "tool_call_id": m.get("call_id")})
    return msgs


def _one(line):
    r = json.loads(line)
    rcp = r["responses_create_params"]
    tools = [{"type": "function", "function": {k: v for k, v in t.items() if k != "type"}} if "function" not in t else t
             for t in (rcp.get("tools") or [])]
    try:
        text = _tok.apply_chat_template(to_messages(rcp["input"]), tools=tools or None, add_generation_prompt=True, tokenize=False)
        return r["agent_ref"]["name"], len(_tok(text, add_special_tokens=False)["input_ids"]), False
    except Exception:  # 템플릿 실패 행은 원문 길이로 대신 센다 — 실패 수를 따로 보고한다
        return r["agent_ref"]["name"], len(_tok(json.dumps(rcp, ensure_ascii=False), add_special_tokens=False)["input_ids"]), True


def _summary(v, fails, max_len):
    v = sorted(v)
    q = lambda p: v[min(len(v) - 1, int(p * len(v)))]  # noqa: E731
    d = {"n": len(v), "tmpl_fail": fails, "p50": q(0.5), "p99": q(0.99), "max": v[-1],
         ">8K": sum(x > 8192 for x in v), ">32K": sum(x > 32768 for x in v), ">64K": sum(x > 65536 for x in v)}
    if max_len:
        d[f">{max_len}"] = sum(x > max_len for x in v)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--procs", type=int, default=96)
    args = ap.parse_args()
    with open(args.blend) as f:
        lines = [l for l in f if l.strip()]
    with Pool(args.procs, initializer=_init, initargs=(args.tokenizer,)) as p:
        res = p.map(_one, lines, chunksize=64)
    by, fails = collections.defaultdict(list), collections.Counter()
    for a, n, failed in res:
        by[a].append(n)
        fails[a] += failed
    rep = {a: _summary(v, fails[a], args.max_len) for a, v in sorted(by.items())}
    rep["_all"] = _summary([n for _, n, _ in res], sum(fails.values()), args.max_len)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=1)
    for a, v in rep.items():
        print(a, v)
    return 1 if args.max_len and rep["_all"][f">{args.max_len}"] else 0


if __name__ == "__main__":
    sys.exit(main())
