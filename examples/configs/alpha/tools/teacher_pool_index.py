"""1차 teacher P0-2 1단계 (RL_DATA.md §5.6): Code·Math 후보 원천을 훑어 행별 메타(문제 키·출처·참조 pass_rate·테스트 수·파일 오프셋)를 TSV 로 남긴다 (CPU).

문제 키 = 프롬프트에서 고정 지시문·effort 마커를 걷어낸 문제 본문을 정규화한 해시. 원천마다 지시문 래퍼가 달라서다.
  python examples/configs/alpha/tools/teacher_pool_index.py <work_dir>   # → <work_dir>/pool_index.tsv
"""
import os
import re
import sys
from multiprocessing import Pool

import orjson
import xxhash

RL = "/home/work/Datasets/LL_datasets/posttraining/RL"
AB = f"{RL}/alpha_blends"
# (tag, path, policy) — policy = 참조 pass_rate 를 잰 정책
FILES = [
    ("ocr25k", f"{RL}/Nemotron-RL-coding-competitive_coding/opencodereasoning_filtered_25k_train.jsonl", ""),
    ("lcb_v5_val", f"{RL}/Nemotron-RL-coding-competitive_coding/validation.jsonl", ""),
    ("ultra_rlvr1", f"{AB}/ultra_restored/rlvr1.jsonl", "ultra_sft3200"),
    ("ultra_rlvr2", f"{AB}/ultra_restored/rlvr2.jsonl", "ultra_sft3200"),
    ("ultra_mopd", f"{AB}/ultra_restored/mopd.jsonl", "ultra_sft3200"),
    ("super_rlvr1", f"{AB}/super_restored/rlvr1.jsonl", "super_lcsft1000"),
    ("super_rlvr2", f"{AB}/super_restored/rlvr2.jsonl", "super_lcsft1000"),
    ("super_rlvr3", f"{RL}/Nemotron-RL-Super-Training-Blends/rlvr3.jsonl", "super_lcsft1000"),
    ("nano", f"{AB}/nano_restored/train.jsonl", "nano_sft"),
    ("mathv2", f"{AB}/math_v2_restored/train.jsonl", ""),
]
CODE_AGENT = "code_gen_simple_agent"
MATH_AGENT = "math_with_judge_simple_agent"
CODE_PREFIX = (
    "You are a helpful and harmless assistant. You should think step-by-step before responding to the instruction below.\n\n"
    "Please use python programming language only.\n\n"
    "You must use ```python for just the final solution code block with the following format:\n```python\n# Your code here\n```\n\n"
)
EFFORT_RE = re.compile(r"\s*\{reasoning effort:\s*[a-z]+\}\s*$")
# 수학 지시문 래퍼: 원천마다 문구가 다르다 (Nano "Don't forget to put your answer in \\boxed{}." · Super "Your final answer
# should be in \\boxed{}." · Math-v2 "Solve the math problem below. Your final answer should be inside \\boxed{}." 등).
# 빈 \\boxed{} 를 담은 지시 절과 알려진 도입 문장만 지운다. 문장 전체를 지우면 한 문장짜리 문제가 통째로 사라진다
# (2026-10-08: Math-v2 문제 여러 개가 빈 키 하나로 뭉쳤다). 지운 뒤 8자 미만이면 원문을 키로 쓴다.
BOXED_CLAUSE = re.compile(
    r"[,;]?\s*(?:and\s+)?(?:please\s+)?(?:put|place|enclose|write|present|provide|give|return|output|box|make sure|remember to put|don't forget to put)\b"
    r"[^.?!\n]*?\\boxed\{\}[^.?!\n]*[.?!]?", re.I)
BOXED_ANSWER = re.compile(r"(?:your\s+)?(?:final\s+)?answer\s+(?:should|must|has to)\s+be[^.?!\n]*?\\boxed\{\}[^.?!\n]*[.?!]?", re.I)
LEAD_INS = re.compile(
    r"(your task is to (?:find the solution to|solve) (?:this|a) math problem|solve the (?:following )?math problem(?: below)?|"
    r"please (?:solve|work through) (?:the following|this) (?:math )?problem|solve this (?:math )?problem|"
    r"please solve the following math problem)\s*[.:]?", re.I)
WS = re.compile(r"\s+")


def norm(t: str) -> str:
    return WS.sub(" ", t).strip().lower()


def user_text(row) -> str:
    inp = (row.get("responses_create_params") or {}).get("input") or []
    if isinstance(inp, str):
        return inp
    us = [m.get("content", "") for m in inp if isinstance(m, dict) and m.get("role") == "user"]
    c = us[-1] if us else ""
    return c if isinstance(c, str) else orjson.dumps(c).decode()


def code_problem(row) -> tuple[str, bool]:
    t = user_text(row)
    had_prefix = t.startswith(CODE_PREFIX)
    if had_prefix:
        t = t[len(CODE_PREFIX):]
    t, n = EFFORT_RE.subn("", t)
    return t, had_prefix


def math_problem(row) -> str:
    t0 = EFFORT_RE.sub("", user_text(row))
    t = BOXED_CLAUSE.sub(" ", t0)
    t = BOXED_ANSWER.sub(" ", t)
    t = LEAD_INS.sub(" ", t)
    return t if len(norm(t)) >= 8 else t0


def run(item):
    tag, path, policy = item
    out = []
    with open(path, "rb") as f:
        off = 0
        for line in f:
            ln = len(line)
            is_code = b"code_gen_simple_agent" in line or tag in ("ocr25k", "lcb_v5_val")
            is_math = b"math_with_judge_simple_agent" in line
            if is_code or is_math:
                r = orjson.loads(line)
                ag = (r.get("agent_ref") or {}).get("name", "")
                if tag in ("ocr25k", "lcb_v5_val"):
                    ag = CODE_AGENT
                if ag == CODE_AGENT:
                    prob, had_prefix = code_problem(r)
                    ut = (r.get("verifier_metadata") or {}).get("unit_tests") or {}
                    ntest = len(ut.get("inputs") or [])
                    fn = 1 if ut.get("fn_name") else 0
                    dom = "code"
                elif ag == MATH_AGENT:
                    prob = math_problem(r)
                    had_prefix, ntest, fn = True, 0, 0
                    dom = "math"
                else:
                    off += ln
                    continue
                key = xxhash.xxh64_hexdigest(norm(prob)) if prob.strip() else ""
                eff = 1 if EFFORT_RE.search(user_text(r)) else 0
                pr = r.get("pass_rate", "")
                pt = r.get("pass_rate_total", "")
                ans = (r.get("expected_answer") or "") if dom == "math" else ""
                out.append("\t".join(map(str, [
                    dom, tag, policy, off, ln, key, r.get("dataset", ""), r.get("source", ""), r.get("hash_id", ""),
                    "" if pr is None else pr, "" if pt is None else pt, ntest, fn, eff, int(had_prefix),
                    len(prob), xxhash.xxh64_hexdigest(norm(str(ans))) if ans else "",
                ])))
            off += ln
    return tag, out


if __name__ == "__main__":
    od = sys.argv[1]
    os.makedirs(od, exist_ok=True)
    hdr = "dom\ttag\tpolicy\toff\tlen\tkey\tdataset\tsource\thash_id\tpass_rate\tpass_total\tntest\tfn_name\teffort\thad_prefix\tprob_chars\tans_hash\n"
    with Pool(len(FILES)) as p, open(os.path.join(od, "pool_index.tsv"), "w") as fo:
        fo.write(hdr)
        for tag, rows in p.imap_unordered(run, FILES):
            for r in rows:
                fo.write(r + "\n")
            print(tag, len(rows), flush=True)
