r"""1차 teacher P0-2 2단계 (RL_DATA.md §5.6): Code·Math 후보 풀 — 정규 행 선택 · 벤치 오염 제거 · P0-3 측정 목록 (CPU).

입력: teacher_pool_index.py 가 만든 pool_index.tsv. 출력 (out_dir):
  {code,math}_pool.jsonl     문제당 한 행 (Gym 실행 형식), 벤치 오염·답 충돌 제거
  {code,math}_measure.jsonl  P0-3 측정 목록 (약 1.2만)
  {code,math}_reserve.jsonl  나머지 (예비)
  decontam_report.tsv        걸린 행 (키 · 벤치 · 겹침 비율 · 앞부분)
  각 파일 옆 .stats.json

정규 행 우선순위 — 코드: nano > super > ultra > ocr25k (Nano pass_rate 를 잰 테스트 세트를 쓴다).
                   수학: nano > super > mathv2 > ultra (Ultra 행에는 \boxed{} 지시문이 없다).
effort 마커(`{reasoning effort: ...}`)는 지운다 — 사전 측정은 기본 모드로 한다.
참조 pass_rate 는 정책 이름을 붙인 평평한 필드로 남긴다: ref_pass_rate_<policy>, ref_pass_total_<policy>.
  python examples/configs/alpha/tools/teacher_pool_build.py <work_dir> <out_dir> [--n-measure 12000] [--seed 20261008]
  2026-10-08 산출: /home/work/Datasets/LL_datasets/posttraining/RL/alpha_blends/teacher_pool/
"""
import argparse
import collections
import glob
import json
import os
import random
import re
import sys

import orjson

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teacher_pool_index import CODE_AGENT, EFFORT_RE, FILES, MATH_AGENT, code_problem, math_problem  # noqa: E402

PATH = {t: p for t, p, _ in FILES}
POLICIES = ["nano_sft", "super_lcsft1000", "ultra_sft3200"]
PRIO = {"code": ["nano", "super", "ultra", "ocr25k"], "math": ["nano", "super", "mathv2", "ultra"]}
EVAL = "/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha/eval_sft/results"
TOK = re.compile(r"[a-z0-9]+")
LATEX_CMD = re.compile(r"\\[a-zA-Z]+")
# 두 기준 중 하나라도 걸리면 오염으로 본다: (1) 13-gram 겹침 ≥ 50% · (2) LaTeX 명령을 지운 8-gram 겹침 ≥ 50% 이고 6개 이상
# (같은 문제가 \\le·\\leq·\\limits 같은 표기 차이로 13-gram 을 놓치는 경우, 2026-10-08 MMLU-Pro 사례).
CRITERIA = [("13gram", 13, False, 0.5, 1), ("8gram_nolatex", 8, True, 0.5, 6), ("char30_squeezed", 30, "char", 0.5, 10)]


def src_of(tag: str) -> str:
    return re.sub(r"_(rlvr\d|mopd)$", "", tag)


def load_index(work):
    rows = []
    with open(os.path.join(work, "pool_index.tsv")) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for line in f:
            rows.append(dict(zip(hdr, line.rstrip("\n").split("\t"))))
    return rows


def read_rows(locs):
    """locs: list of (tag, off, len) -> dict[(tag,off)] = row (dict)."""
    by_tag = collections.defaultdict(list)
    for tag, off, ln in locs:
        by_tag[tag].append((off, ln))
    out = {}
    for tag, lst in by_tag.items():
        with open(PATH[tag], "rb") as f:
            for off, ln in sorted(lst):
                f.seek(off)
                out[(tag, off)] = orjson.loads(f.read(ln))
    return out


def ngrams(text, n=13, strip_latex=False):
    r"""strip_latex="char": LaTeX 명령·공백·기호를 다 지운 문자열의 n 글자 shingle (\le·\leq, 2x^2·2 x^2 같은 표기 차이에 강하다)."""
    if strip_latex == "char":
        sq = "".join(TOK.findall(LATEX_CMD.sub(" ", text).lower()))
        return list(sq), {sq[i:i + n] for i in range(len(sq) - n + 1)}
    t = text.lower()
    if strip_latex:
        t = LATEX_CMD.sub(" ", text).lower()
    t = TOK.findall(t)
    return t, {" ".join(t[i:i + n]) for i in range(len(t) - n + 1)}


def load_bench():
    items = []  # (bench, id, text)
    def first(pat):
        fs = sorted(glob.glob(f"{EVAL}/*/alpha/samples_{pat}_*.jsonl"))
        return fs[0] if fs else None
    for bench, pat, field in [("aime25", "aime25_aa", "problem"), ("hmmt_feb25", "hmmt_feb_2025_aa", "problem"),
                              ("gpqa_diamond", "gpqa_diamond_aa", "Question"), ("mmlu_pro", "mmlu_pro_aa", "question")]:
        fp = first(pat)
        if not fp:
            continue
        with open(fp) as f:
            for line in f:
                r = json.loads(line)
                items.append((bench, str(r["doc_id"]), str(r["doc"].get(field) or "")))
    with open(PATH["lcb_v5_val"], "rb") as f:
        for i, line in enumerate(f):
            r = orjson.loads(line)
            items.append(("lcb_v5_val", str(i), code_problem(r)[0]))
    return items


def decontam_one(cands, bench_items, n, strip_latex, min_cov, min_shared, boiler_df=20):
    """cands: dict key -> problem text. 반환: dict key -> (bench, id, coverage, shared), 상투 n-gram 수."""
    bidx = collections.defaultdict(set)
    bgrams = {}
    bshort = {}
    for b, i, t in bench_items:
        toks, g = ngrams(t, n, strip_latex)
        if len(toks) < n:
            if toks:
                bshort[(b, i)] = ("".join(toks) if strip_latex == "char" else " ".join(toks))
            continue
        bgrams[(b, i)] = g
        for x in g:
            bidx[x].add((b, i))
    cand_grams = {}
    df = collections.Counter()
    for k, t in cands.items():
        toks, g = ngrams(t, n, strip_latex)
        hit = {x for x in g if x in bidx}
        cand_grams[k] = (("".join(toks) if strip_latex == "char" else " ".join(toks)), hit)
        df.update(hit)
    boiler = {x for x, c in df.items() if c > boiler_df}
    flagged = {}
    for k, (joined, hit) in cand_grams.items():
        cnt = collections.Counter()
        for x in hit - boiler:
            for bi in bidx[x]:
                cnt[bi] += 1
        best = None
        for bi, c in cnt.items():
            cov = c / max(1, len(bgrams[bi] - boiler))
            if cov >= min_cov and c >= min_shared and (best is None or cov > best[2]):
                best = (bi[0], bi[1], round(cov, 3), c)
        if best is None:
            for bi, sh in bshort.items():
                # 짧은 벤치 문항은 전체 일치만 본다. 부분 문자열 일치는 MMLU-Pro 의 "Which of the following is correct?"
                # 같은 일반 질문 줄기가 아무 후보에나 걸려 오탐만 냈다 (2026-10-08, 6건 전부 오탐).
                if sh and joined == sh:
                    best = (bi[0], bi[1], 1.0, 0)
                    break
        if best:
            flagged[k] = best
    return flagged, len(boiler)


MATH_SEG = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$|\\\[(.+?)\\\]|\\\((.+?)\\\)", re.S)


def squeeze(t):
    return "".join(TOK.findall(LATEX_CMD.sub(" ", t).lower()))


def decontam_formula(cands, bench_items, min_len=25, max_df=3):
    """벤치 문항의 긴 수식(LaTeX 명령을 지우고 25글자 이상)이 후보 본문에 그대로 들어 있으면 오염으로 본다.

    같은 문제를 다른 질문 문장으로 낸 경우를 잡는다 (2026-10-08: MMLU-Pro 의 Putnam 적분 문제). 후보 3개 넘게 나오는 수식은 일반식으로 보고 뺀다.
    """
    segs = {}
    for b, i, t in bench_items:
        for m in MATH_SEG.finditer(t):
            sq = squeeze(next(g for g in m.groups() if g is not None))
            if len(sq) >= min_len:
                segs.setdefault(sq, (b, i))
    if not segs:
        return {}
    # 후보 본문을 구분자로 이어 붙인 큰 문자열에서 str.find 로 찾는다 (C 탐색, 수식 × 후보 이중 루프를 피한다)
    keys = list(cands)
    starts, parts, pos = [], [], 0
    for k in keys:
        c = squeeze(cands[k])
        starts.append(pos)
        parts.append(c)
        pos += len(c) + 1
    big = "|".join(parts)
    import bisect
    hits = collections.defaultdict(list)
    for sq, bi in segs.items():
        found, j = set(), big.find(sq)
        while j != -1 and len(found) <= max_df:
            found.add(keys[bisect.bisect_right(starts, j) - 1])
            j = big.find(sq, j + 1)
        if 0 < len(found) <= max_df:
            for k in found:
                hits[k].append((bi, len(sq)))
    return {k: (max(v, key=lambda x: x[1])[0][0], max(v, key=lambda x: x[1])[0][1], 1.0, max(x[1] for x in v)) for k, v in hits.items()}


def decontam(cands, bench_items):
    flagged, nboiler = {}, {}
    for k, v in decontam_formula(cands, bench_items).items():
        flagged[k] = v + ("formula25",)
    for name, n, strip, cov, sh in CRITERIA:
        f, nb = decontam_one(cands, bench_items, n, strip, cov, sh)
        nboiler[name] = nb
        for k, v in f.items():
            if k not in flagged:
                flagged[k] = v + (name,)
    return flagged, nboiler


def tier_of(dom, rec):
    pr = rec["ref_pass_rate_nano_sft"]
    if pr is not None and 0.25 <= pr <= 1.0:
        return 1
    if pr is None:
        return 2
    return 3 if pr > 0 else 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("out")
    ap.add_argument("--n-measure", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=20261008)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    idx = load_index(a.work)
    bench = load_bench()
    report = open(os.path.join(a.out, "decontam_report.tsv"), "w")
    report.write("dom\tkey\tbench\tbench_id\tcriterion\tcoverage\tshared_ngrams\tproblem_head\n")
    summary = {}
    for dom in ["code", "math"]:
        recs = [r for r in idx if r["dom"] == dom and r["tag"] != "lcb_v5_val"]
        groups = collections.defaultdict(list)
        for r in recs:
            groups[r["key"]].append(r)
        # 답 충돌 (수학): 같은 문제에 다른 정답
        conflict = set()
        if dom == "math":
            for k, g in groups.items():
                if len({x["ans_hash"] for x in g}) > 1:
                    conflict.add(k)
        canon = {}
        meta = {}
        for k, g in groups.items():
            by_src = collections.defaultdict(list)
            for x in g:
                by_src[src_of(x["tag"])].append(x)
            src = next(s for s in PRIO[dom] if s in by_src)
            c = by_src[src][0]
            canon[k] = (c["tag"], int(c["off"]), int(c["len"]))
            m = {"pool_sources": ",".join(s for s in PRIO[dom] if s in by_src), "pool_variant": src, "pool_key": k}
            for pol in POLICIES:
                vals = [(float(x["pass_rate"]), x["pass_total"]) for x in g if x["policy"] == pol and x["pass_rate"] not in ("", "None")]
                if vals:
                    m[f"ref_pass_rate_{pol}"] = round(sum(v for v, _ in vals) / len(vals), 4)
                    m[f"ref_pass_total_{pol}"] = int(float(vals[0][1])) if vals[0][1] not in ("", "None") else None
                else:
                    m[f"ref_pass_rate_{pol}"] = None
                    m[f"ref_pass_total_{pol}"] = None
            meta[k] = m
        rows = read_rows(list(canon.values()))
        # 문제 본문 (오염 검사용)
        texts = {}
        for k, (tag, off, ln) in canon.items():
            r = rows[(tag, off)]
            texts[k] = code_problem(r)[0] if dom == "code" else math_problem(r)
        flagged, n_boiler = decontam(texts, bench)
        for k, (b, i, cov, sh, crit) in sorted(flagged.items(), key=lambda kv: -kv[1][2]):
            report.write(f"{dom}\t{k}\t{b}\t{i}\t{crit}\t{cov}\t{sh}\t{texts[k][:120].replace(chr(9),' ').replace(chr(10),' ')}\n")
        # 출력 행
        out_rows = {}
        eff_removed = 0
        for k, (tag, off, ln) in canon.items():
            if k in flagged or k in conflict:
                continue
            r = dict(rows[(tag, off)])
            r["agent_ref"] = {"type": "responses_api_agents", "name": CODE_AGENT if dom == "code" else MATH_AGENT}
            rcp = dict(r["responses_create_params"])
            new_inp = []
            for msg in rcp.get("input") or []:
                msg = dict(msg)
                if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                    c2 = EFFORT_RE.sub("", msg["content"])
                    eff_removed += c2 != msg["content"]
                    msg["content"] = c2
                new_inp.append(msg)
            rcp["input"] = new_inp
            r["responses_create_params"] = rcp
            if isinstance(r.get("question"), str):
                r["question"] = EFFORT_RE.sub("", r["question"])
            for f in ("pass_rate", "pass_rate_total", "pass_rate_passed"):
                r.pop(f, None)
            r.update(meta[k])
            if dom == "code":
                r["n_unit_tests"] = len(((r.get("verifier_metadata") or {}).get("unit_tests") or {}).get("inputs") or [])
            out_rows[k] = r
        # 측정 목록
        rng = random.Random(a.seed)
        tiers = collections.defaultdict(list)
        for k, r in out_rows.items():
            tiers[tier_of(dom, r)].append(k)
        for t in tiers:
            tiers[t].sort()
            rng.shuffle(tiers[t])
        # tier 2 는 출처 묶음(정규 행 원천)을 번갈아 뽑아 고르게 섞는다
        if 2 in tiers:
            by = collections.defaultdict(list)
            for k in tiers[2]:
                by[out_rows[k]["pool_variant"]].append(k)
            mixed, keys = [], sorted(by)
            while any(by[s] for s in keys):
                for s in keys:
                    if by[s]:
                        mixed.append(by[s].pop())
            tiers[2] = mixed
        order = tiers[1] + tiers[2] + tiers[3] + tiers[4]
        measure, reserve = order[: a.n_measure], order[a.n_measure:]

        def write(name, keys):
            p = os.path.join(a.out, f"{dom}_{name}.jsonl")
            with open(p, "wb") as fo:
                for k in keys:
                    fo.write(orjson.dumps(out_rows[k]) + b"\n")
            st = {"file": p, "rows": len(keys),
                  "by_variant": collections.Counter(out_rows[k]["pool_variant"] for k in keys),
                  "by_tier": collections.Counter(tier_of(dom, out_rows[k]) for k in keys),
                  "nano_pass_rate_bins": collections.Counter(
                      "none" if out_rows[k]["ref_pass_rate_nano_sft"] is None else
                      ("0" if out_rows[k]["ref_pass_rate_nano_sft"] == 0 else
                       "(0,0.25)" if out_rows[k]["ref_pass_rate_nano_sft"] < 0.25 else
                       "[0.25,0.5)" if out_rows[k]["ref_pass_rate_nano_sft"] < 0.5 else
                       "[0.5,0.75)" if out_rows[k]["ref_pass_rate_nano_sft"] < 0.75 else "[0.75,1]") for k in keys)}
            if dom == "code":
                st["by_site"] = collections.Counter(out_rows[k].get("source", "") for k in keys)
                nt = sorted(out_rows[k]["n_unit_tests"] for k in keys)
                st["n_unit_tests_p50_p90_max"] = [nt[len(nt) // 2], nt[int(len(nt) * 0.9)], nt[-1]] if nt else []
            else:
                st["by_dataset"] = collections.Counter(out_rows[k].get("dataset", "") for k in keys)
            json.dump(st, open(p + ".stats.json", "w"), indent=1, ensure_ascii=False)
            return st

        write("pool", sorted(out_rows))
        write("measure", measure)
        write("reserve", reserve)
        summary[dom] = {
            "index_rows": len(recs), "unique_problems": len(groups), "bench_flagged": len(flagged),
            "flagged_by_bench": collections.Counter(v[0] for v in flagged.values()),
            "flagged_by_criterion": collections.Counter(v[4] for v in flagged.values()),
            "answer_conflict_dropped": len(conflict - set(flagged)), "pool_rows": len(out_rows),
            "effort_marker_removed": eff_removed, "measure_rows": len(measure), "reserve_rows": len(reserve),
            "tiers_in_pool": {t: len(v) for t, v in sorted(tiers.items())}, "boilerplate_ngrams": n_boiler,
        }
    report.close()
    summary["bench_items"] = collections.Counter(b for b, _, _ in bench)
    summary["not_checked"] = ["MATH-500", "AIME 2024", "HMMT Nov 2025", "LiveCodeBench v6 이후 (로컬에 v5 2024-07~2025-02 322문항만 있음)"]
    json.dump(summary, open(os.path.join(a.out, "build_summary.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
