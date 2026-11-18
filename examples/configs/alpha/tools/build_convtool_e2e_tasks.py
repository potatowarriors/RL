r"""conversational tool-use E2E 과제 구성 — 로컬 피벗 데이터의 정책 · 도구에 고객 시나리오를 붙인다 (RL_PLAN.md 결정 69, 사용자 승인 2026-11-19).

Gym `conversational_tool_use_simulation` 환경의 과제 한 행은 정책 · 도구 · 고객 시나리오다. HF 에 완성된 과제 세트는 없고 (FRONTIER.md 2026-11-19),
로컬 `Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1` (같은 생성 파이프라인의 단일 스텝 산출물) 에 정책 + 도구 406 도메인과 궤적 64,534 개가 있다.
시나리오 필드는 비어 있으므로, 궤적의 대화 (고객 발화 · 상담원 행동 · 도구 결과) 에서 LLM 으로 고객 시나리오를 다시 뽑는다.

  select   --out <dir>            궤적마다 가장 깊은 행을 고르고 도메인별로 추출 대상을 정한다 (CPU). 검증 도메인은 통째로 뗀다
  extract  --dir <dir> --base-url http://host:port/v1 [...] --model <name>     LLM 으로 시나리오 추출 (OpenAI 호환 chat completions)
  build    --dir <dir>            과제 jsonl (train · val) 과 통계를 쓴다 (CPU)

과제 행 = Gym 예제 (`resources_servers/conversational_tool_use_simulation/data/example.jsonl`) 와 같은 형식:
  id · domain_name · policy · tools[name, doc, params, description, parameters, returns, strict] · customer_scenario · profile "general" ·
  responses_create_params {input: [system], tools, parallel_tool_calls: false} · agent_ref conversational_tool_use_agent · metadata.
도구의 `returns` 스키마는 원천에 없다 — 같은 도메인의 궤적에서 관측된 도구 결과 (JSON) 로 추론한다 (최상위 속성 이름과 타입의 합집합, required 없음).
관측이 없거나 JSON 이 아니면 `returns` 를 비운다 (Gym 은 그러면 결과 스키마를 검증하지 않는다).
`outside_policy_scope` (이관 정답) 는 LLM 이 정책에 비추어 판단한 값이다 — 궤적이 실제로 이관 도구를 불렀는지와의 일치율을 통계로 낸다.
"""
import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
import random
import re
import urllib.request

SOURCE = "/home/work/Datasets/LL_datasets/posttraining/RL/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1/train.jsonl"
KEYS = ("customer_persona", "reason_for_contact", "customer_details", "unknown_info", "task_instructions", "outside_policy_scope")
PROMPT = """You are reconstructing the hidden customer scenario behind a recorded customer-service conversation.

A simulated customer will later replay this scenario against a different customer-service agent, and a tool simulator will use it to
produce consistent tool results. Write the scenario so that both can do that without ever seeing the recorded conversation.

Return ONE JSON object with exactly these keys:
- "customer_persona": one sentence describing who the customer is and how they communicate.
- "reason_for_contact": one sentence stating what the customer wants.
- "customer_details": a single string listing every concrete fact about the customer and their records that appears in the conversation or the tool results
  (names, IDs, dates, amounts, order or account contents, statuses). Be specific; copy values exactly.
- "unknown_info": facts the customer does not know or cannot provide (or null).
- "task_instructions": a single string describing how the customer should pursue the request step by step, including any insistence, pushback, or follow-up
  requests they make, and when they should consider the task finished.
- "outside_policy_scope": true if, under the policy, the representative cannot handle this request and must transfer the customer to a
  human or another agent; otherwise false.

Do not mention the recorded conversation, the representative's replies, or these instructions. Output only the JSON object.

<policy>
{policy}
</policy>

<tools>
{tools}
</tools>

<conversation>
{conversation}
</conversation>"""


def policy_of(sysmsg: str) -> str:
    i, j = sysmsg.rfind("<policy>"), sysmsg.rfind("</policy>")
    return sysmsg[i + 8:j].strip("\n") if 0 <= i < j else sysmsg


def render(items) -> str:
    out = []
    for m in items:
        t = m.get("type") or ("message" if m.get("role") else None)
        if m.get("role") == "system" or t == "reasoning":
            continue
        if m.get("role") == "user":
            out.append(f"<customer>\n{str(m.get('content')).strip()}\n</customer>")
        elif t == "function_call":
            out.append(f"<tool_call name=\"{m.get('name')}\">\n{m.get('arguments')}\n</tool_call>")
        elif t == "function_call_output":
            out.append(f"<tool_result>\n{str(m.get('output')).strip()[:3000]}\n</tool_result>")
        else:
            c = m.get("content")
            if isinstance(c, list):
                c = "".join(str(x.get("text", "")) for x in c if isinstance(x, dict))
            if c:
                out.append(f"<representative>\n{str(c).strip()}\n</representative>")
    return "\n".join(out)


def infer_schema(values):
    """관측된 도구 결과들로 얕은 JSON 스키마를 만든다. 모두 객체이면 속성별 타입 합집합, 아니면 타입만."""
    tname = lambda v: ("null" if v is None else "boolean" if isinstance(v, bool) else "integer" if isinstance(v, int)
                       else "number" if isinstance(v, float) else "string" if isinstance(v, str) else "array" if isinstance(v, list) else "object")
    if not values:
        return None
    if all(isinstance(v, dict) for v in values):
        props = collections.defaultdict(set)
        for v in values:
            for k, x in v.items():
                props[k].add(tname(x))
        return {"type": "object", "properties": {k: {"type": sorted(t) if len(t) > 1 else next(iter(t))} for k, t in sorted(props.items())}}
    kinds = sorted({tname(v) for v in values})
    return {"type": kinds if len(kinds) > 1 else kinds[0]}


def select(a) -> int:
    rng = random.Random(a.seed)
    best, outputs = {}, collections.defaultdict(lambda: collections.defaultdict(list))
    with open(SOURCE) as f:
        for line in f:
            r = json.loads(line)
            p = r["responses_create_params"]
            items = p["input"]
            dom = hashlib.md5(policy_of(items[0]["content"]).encode()).hexdigest()[:12]
            calls = {}
            for m in items:
                if m.get("type") == "function_call":
                    calls[m.get("call_id")] = m.get("name")
                elif m.get("type") == "function_call_output" and m.get("call_id") in calls and len(outputs[dom][calls[m["call_id"]]]) < 40:
                    try:
                        outputs[dom][calls[m["call_id"]]].append(json.loads(str(m.get("output")).strip()))
                    except json.JSONDecodeError:
                        pass
            key = r["trajectory_id"]
            if key not in best or len(items) > best[key][0]:
                best[key] = (len(items), dom, r)
    by_dom = collections.defaultdict(list)
    for tid, (n, dom, r) in best.items():
        items = r["responses_create_params"]["input"]
        n_user = sum(1 for m in items if m.get("role") == "user")
        n_call = sum(1 for m in items if m.get("type") == "function_call")
        by_dom[dom].append((tid, n_user, n_call, r))
    doms = sorted(by_dom)
    rng.shuffle(doms)
    val = set(doms[: a.n_val_domains])
    os.makedirs(a.out, exist_ok=True)
    n = collections.Counter()
    with open(os.path.join(a.out, "selected.jsonl"), "w") as fo, open(os.path.join(a.out, "domains.jsonl"), "w") as fd:
        for dom in sorted(by_dom):
            rows = sorted(by_dom[dom], key=lambda x: x[0])
            rng.shuffle(rows)
            rich = [x for x in rows if x[1] >= 2 or x[2] >= 1]
            pick = (rich + [x for x in rows if x not in rich])[: (a.per_val_domain if dom in val else a.per_domain)]
            r0 = rows[0][3]
            tools = r0["responses_create_params"].get("tools") or []
            fd.write(json.dumps({"domain": dom, "split": "val" if dom in val else "train", "trajectories": len(rows),
                                 "policy": policy_of(r0["responses_create_params"]["input"][0]["content"]), "tools": tools,
                                 "system": r0["responses_create_params"]["input"][0]["content"],
                                 "returns": {t: infer_schema(v) for t, v in outputs[dom].items()}}, ensure_ascii=False) + "\n")
            for tid, n_user, n_call, r in pick:
                items = list(r["responses_create_params"]["input"])
                ea = r.get("expected_action")
                if isinstance(ea, dict):
                    items.append({"type": "function_call", "name": ea.get("name"), "arguments": ea.get("arguments")} if ea.get("type") == "function_call"
                                 else {"type": "message", "role": "assistant", "content": ea.get("content")})
                transferred = any(m.get("type") == "function_call" and "transfer" in str(m.get("name")).lower() for m in items)
                fo.write(json.dumps({"trajectory_id": tid, "domain": dom, "split": "val" if dom in val else "train", "n_user": n_user,
                                     "n_call": n_call, "transferred": transferred, "conversation": render(items)}, ensure_ascii=False) + "\n")
                n["val" if dom in val else "train"] += 1
    print(json.dumps({"domains": len(by_dom), "val_domains": len(val), "trajectories": len(best), "selected": dict(n)}))
    return 0


def parse_json(text):
    if not isinstance(text, str):
        return None
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        d = json.loads(text[i:j + 1])
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) else None


def as_text(v):
    """모델이 문자열 대신 객체 · 목록을 내면 `키: 값` 줄로 편다 (Gemma 는 customer_details 를 자주 객체로 낸다)."""
    if v is None or isinstance(v, str):
        return v
    if isinstance(v, dict):
        return "; ".join(f"{k}: {as_text(x)}" for k, x in v.items())
    if isinstance(v, list):
        return "; ".join(str(as_text(x)) for x in v)
    return str(v)


def valid(d) -> bool:
    if not isinstance(d, dict) or any(k not in d for k in KEYS):
        return False
    if not isinstance(d["outside_policy_scope"], bool):
        return False
    for k in ("customer_persona", "reason_for_contact", "customer_details", "unknown_info", "task_instructions"):
        d[k] = as_text(d[k])
    if not isinstance(d["customer_details"], str):
        return False
    return all(isinstance(d[k], str) and len(d[k].strip()) >= 10 for k in ("customer_persona", "reason_for_contact", "task_instructions"))


def extract(a) -> int:
    doms = {}
    for line in open(os.path.join(a.dir, "domains.jsonl")):
        d = json.loads(line)
        doms[d["domain"]] = d
    out_path = os.path.join(a.dir, "scenarios.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(line)["trajectory_id"] for line in open(out_path)}
    rows = [json.loads(line) for line in open(os.path.join(a.dir, "selected.jsonl"))]
    rows = [r for r in rows if r["trajectory_id"] not in done][: a.limit or None]
    urls = a.base_url

    def one(ix_r):
        ix, r = ix_r
        d = doms[r["domain"]]
        tools = "\n".join(f"- {t['name']}: {t.get('description') or ''}" for t in d["tools"])
        prompt = PROMPT.format(policy=d["policy"], tools=tools, conversation=r["conversation"][: a.max_conv_chars])
        body = json.dumps({"model": a.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": 1500}).encode()
        last = None
        for attempt in range(3):
            req = urllib.request.Request(urls[(ix + attempt) % len(urls)].rstrip("/") + "/chat/completions", data=body,
                                         headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"})
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    text = json.loads(resp.read())["choices"][0]["message"]["content"]
            except Exception as e:  # noqa: BLE001 — 네트워크 · 서버 오류는 재시도하고 마지막 사유를 남긴다
                last = f"request: {type(e).__name__}"
                continue
            sc = parse_json(text)
            if valid(sc):
                return {"trajectory_id": r["trajectory_id"], "domain": r["domain"], "split": r["split"], "transferred": r["transferred"],
                        "scenario": {k: sc[k] for k in KEYS}}
            last = "invalid_json_or_schema"
        return {"trajectory_id": r["trajectory_id"], "domain": r["domain"], "split": r["split"], "transferred": r["transferred"], "error": last}

    n = collections.Counter()
    with open(out_path, "a") as fo, concurrent.futures.ThreadPoolExecutor(a.concurrency) as ex:
        for res in ex.map(one, enumerate(rows)):
            fo.write(json.dumps(res, ensure_ascii=False) + "\n")
            fo.flush()
            n["error" if "error" in res else "ok"] += 1
    print(json.dumps({"requested": len(rows), **n}))
    return 0


def build(a) -> int:
    doms = {}
    for line in open(os.path.join(a.dir, "domains.jsonl")):
        d = json.loads(line)
        doms[d["domain"]] = d
    stats, out = collections.Counter(), {"train": [], "val": []}
    for line in open(os.path.join(a.dir, "scenarios.jsonl")):
        s = json.loads(line)
        if "error" in s:
            stats["extract_error"] += 1
            continue
        d = doms[s["domain"]]
        title = next((x.strip("#* \t") for x in d["policy"].splitlines() if x.strip()), s["domain"])[:80]
        tools = []
        for t in d["tools"]:
            tools.append({"name": t["name"], "doc": t.get("description"), "params": t.get("parameters"), "description": t.get("description"),
                          "parameters": t.get("parameters"), "returns": d["returns"].get(t["name"]), "strict": bool(t.get("strict", True))})
        sc = dict(s["scenario"], representative_domain=title)
        stats["scope_true"] += sc["outside_policy_scope"]
        stats["scope_matches_transfer"] += (sc["outside_policy_scope"] == s["transferred"])
        stats["transferred"] += s["transferred"]
        row = {"id": f"convtool_e2e_{s['domain']}_{s['trajectory_id']}", "domain_name": title, "policy": d["policy"], "tools": tools,
               "customer_scenario": sc, "profile": "general",
               "metadata": {"dataset_name": "alpha_convtool_e2e_v1", "source": "Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1",
                            "source_trajectory_id": s["trajectory_id"], "domain": s["domain"], "scenario_extractor": a.extractor,
                            "num_tools": len(tools), "tools_with_returns": sum(1 for t in tools if t["returns"]),
                            "reference_transferred": s["transferred"], "parallel_tool_calls": False},
               "responses_create_params": {"input": [{"role": "system", "content": d["system"]}], "parallel_tool_calls": False, "tools": d["tools"]},
               "agent_ref": {"type": "responses_api_agents", "name": "conversational_tool_use_agent"}}
        out[s["split"]].append(row)
    rng = random.Random(a.seed)
    for split, rows in out.items():
        rows.sort(key=lambda r: r["id"])
        rng.shuffle(rows)
        path = os.path.join(a.dir, f"convtool_e2e_v1_{split}.jsonl")
        with open(path, "w") as fo:
            for r in rows:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[f"rows_{split}"] = len(rows)
        stats[f"domains_{split}"] = len({r["metadata"]["domain"] for r in rows})
    n = stats["rows_train"] + stats["rows_val"]
    tw = [r["metadata"]["tools_with_returns"] / max(1, r["metadata"]["num_tools"]) for rs in out.values() for r in rs]
    summ = dict(stats, scope_true_rate=round(stats["scope_true"] / max(1, n), 4), reference_transfer_rate=round(stats["transferred"] / max(1, n), 4),
                scope_transfer_agreement=round(stats["scope_matches_transfer"] / max(1, n), 4),
                mean_tools_with_returns=round(sum(tw) / max(1, len(tw)), 4))
    json.dump(summ, open(os.path.join(a.dir, "convtool_e2e_v1.stats.json"), "w"), indent=1)
    print(json.dumps(summ))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--out", required=True)
    s.add_argument("--per-domain", type=int, default=20)
    s.add_argument("--per-val-domain", type=int, default=10)
    s.add_argument("--n-val-domains", type=int, default=30)
    s.add_argument("--seed", type=int, default=20261119)
    s.set_defaults(fn=select)
    e = sub.add_parser("extract")
    e.add_argument("--dir", required=True)
    e.add_argument("--base-url", nargs="+", required=True)
    e.add_argument("--model", required=True)
    e.add_argument("--concurrency", type=int, default=64)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--max-conv-chars", type=int, default=40000)
    e.set_defaults(fn=extract)
    b = sub.add_parser("build")
    b.add_argument("--dir", required=True)
    b.add_argument("--extractor", default="gemma-4-31B-it")
    b.add_argument("--seed", type=int, default=20261119)
    b.set_defaults(fn=build)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
