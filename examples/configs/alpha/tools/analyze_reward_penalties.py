"""Ultra 의 reward_penalties 4종을 alpha 롤아웃에 오프라인으로 적용해 발동률을 잰다 (CPU, 2026-10-07).

NeMo-RL `apply_reward_penalties` (`nemo_rl/experience/rollouts.py`) 의 판정을 `train_data_step*.jsonl` 의 토큰으로 재현한다.
발동하면 보상이 0 이 된다. 켜기 전에 alpha 템플릿·파서에서 오탐(정상 응답의 보상을 0 으로 만드는 경우)이 없는지 본다.
`token_loss_mask` 의 1 구간은 모델 호출 1회의 생성이고, 그 앞 0 구간은 그 호출의 증분 프롬프트다 (Gym message_log 의 user·assistant 쌍).
  - unwanted_token: 생성 안에 `--unwanted` 토큰 (기본 2 `<|im_start|>`·12 `<tool_response>`·13 `</tool_response>`) — 원본과 같다
  - malformed_think (4a): 증분 프롬프트의 think open/close 개수로 thinking 모드를 추정하고, 생성의 개수가 기대값(켬 0/1, 끔 0/0)과 다르면 위반
    — 원본과 같다. (4b 근사) 생성 디코드 문자열의 '<think>' 가 1개 이상이거나 '</think>' 가 2개 이상이면 위반
  - empty_final_answer (근사): 마지막 생성이 도구 호출이 아닌데, 뒤에서부터 본 생성들의 본문(`</think>` 뒤, thinking 끔이면 전체)이 모두 비었다
  - duplicated_reasoning (근사): 한 생성의 reasoning(`</think>` 앞)과 본문이 strip 후 같다
  python examples/configs/alpha/tools/analyze_reward_penalties.py <dir>/exp_*/train_data_step*.jsonl --tokenizer <hfmodel> [--show 3]
"""

import argparse
import collections
import json

THINK_OPEN, THINK_CLOSE, TOOL_CALL, IM_END, EOD = 14, 15, 10, 3, 0


def segments(mask):
    out, s = [], None
    for i, m in enumerate(mask):
        if m and s is None:
            s = i
        elif not m and s is not None:
            out.append((s, i))
            s = None
    if s is not None:
        out.append((s, len(mask)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--unwanted", default="2,12,13")
    ap.add_argument("--show", type=int, default=3)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    unwanted = {int(x) for x in args.unwanted.split(",")}
    stats = collections.defaultdict(lambda: collections.Counter())
    examples = collections.defaultdict(list)
    for f in args.files:
        for line in open(f):
            r = json.loads(line)
            agent = r["agent_ref"][0]["name"] if isinstance(r["agent_ref"][0], dict) else str(r["agent_ref"][0])
            ids, mask, reward = r["token_ids"][0], r["token_loss_mask"][0], float(r["rewards"][0])
            segs = segments(mask)
            st = stats[agent]
            st["samples"] += 1
            st["reward>0"] += reward > 0
            if not segs:
                continue
            hits, prev_end, gens = set(), 0, []
            for s, e in segs:
                user, gen = ids[prev_end:s], ids[s:e]
                prev_end = e
                po, pc = user.count(THINK_OPEN), user.count(THINK_CLOSE)
                go, gc = gen.count(THINK_OPEN), gen.count(THINK_CLOSE)
                thinking = po == pc + 1
                if unwanted & set(gen):
                    hits.add("unwanted_token")
                if po == pc:
                    exp = (0, 0)
                elif thinking:
                    exp = (0, 1)
                else:
                    exp = None
                if exp is None or (go, gc) != exp:
                    hits.add("malformed_think_4a")
                text = tok.decode(gen, skip_special_tokens=False)
                if text.count("<think>") > 0 or text.count("</think>") > 1:
                    hits.add("malformed_think_4b")
                body = [t for t in gen if t not in (IM_END, EOD)]
                if thinking and THINK_CLOSE in body:
                    k = body.index(THINK_CLOSE)
                    reasoning, content = tok.decode(body[:k]).strip(), tok.decode(body[k + 1:]).strip()
                elif thinking:
                    reasoning, content = tok.decode(body).strip(), ""
                else:
                    reasoning, content = "", tok.decode(body).strip()
                gens.append({"tool": TOOL_CALL in gen, "content": content, "reasoning": reasoning,
                             "go_gc": (go, gc), "p_oc": (po, pc), "truncated": thinking and THINK_CLOSE not in gen})
                if reasoning and content and reasoning == content:
                    hits.add("duplicated_reasoning")
            if not gens[-1]["tool"] and not any(g["content"] for g in reversed(gens)):
                hits.add("empty_final_answer")
            st["calls"] += len(gens)
            st["truncated_last"] += gens[-1]["truncated"]
            for h in hits:
                st[h] += 1
                if reward > 0:
                    st[h + "|reward>0"] += 1
                if len(examples[h]) < args.show:
                    examples[h].append((agent, reward, [(g["p_oc"], g["go_gc"], g["tool"], g["truncated"]) for g in gens][:6],
                                        gens[-1]["content"][:120]))
    keys = ["unwanted_token", "malformed_think_4a", "malformed_think_4b", "empty_final_answer", "duplicated_reasoning"]
    print(f"{'agent':58s} {'n':>4s} {'r>0':>4s} {'calls':>5s} {'trunc':>5s} " + " ".join(f"{k[:14]:>14s}" for k in keys))
    tot = collections.Counter()
    for a, st in sorted(stats.items()):
        tot.update(st)
        print(f"{a[:58]:58s} {st['samples']:4d} {st['reward>0']:4d} {st['calls']:5d} {st['truncated_last']:5d} "
              + " ".join(f"{st[k]:>6d}({st[k + '|reward>0']:>3d}r)" for k in keys))
    print(f"{'TOTAL':58s} {tot['samples']:4d} {tot['reward>0']:4d} {tot['calls']:5d} {tot['truncated_last']:5d} "
          + " ".join(f"{tot[k]:>6d}({tot[k + '|reward>0']:>3d}r)" for k in keys))
    print("(Nr) = 그중 보상 > 0 이던 샘플 수 — 켜면 보상이 0 으로 깎인다")
    for k in keys:
        for agent, reward, calls, content in examples.get(k, []):
            print(f"-- {k}: {agent} reward={reward} calls[(prompt open,close),(gen open,close),tool,trunc]={calls} last_content={content!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
