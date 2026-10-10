r"""STEM 블렌드의 Science-v1 정답 추출 결함 행을 결함 없는 같은 유형 행으로 바꾼다 (CPU, 사용자 승인 2026-10-10 — RL_PLAN.md 결정 25).

결함: 정답 문자열이 그 행의 `output_regex` 닫는 문자를 포함해, 요청 형식대로 감싼 완벽한 답도 비탐욕 정규식이 잘라 judge 에게 보낸다
(`docs/KNOWN_ISSUES.md` 2026-10-10). 판정은 `measure_judge_p04.gold_regex_defect(...)[0] == "truncated"` 다.
정규식이 아예 매치되지 않는 행(`\boxed{}` 중첩 초과)은 Gym 이 응답 전체를 판정하므로 결함이 아니다 ("fallback", 바꾸지 않는다).

교체 규칙:
  - 자리는 그대로 둔다 (같은 줄 번호). 나머지 행·순서는 그대로다.
  - 후보는 Science-v1 원천(`so_openq.jsonl`)에서 (agent · metadata.topic · output_regex) 가 같고, 정답 판정이 "ok" 이며,
    세 입력 파일 어디에도 쓰이지 않은 행이다 (uuid · 정규화한 문제 본문 둘 다). 같은 정규식으로 맞춰 형식 지시 분포를 유지한다.
  - 후보는 벤치 오염 검사(`teacher_pool_build.decontam`, 블렌드 작성 때와 같은 기준)를 거친다.
  - 원천 행에 `pool_source: "sci"` 만 붙인다 (블렌드 작성 때와 같다).
마지막에 judge 를 쓰는 모든 행(equivalence_llm_judge · ns_tools)을 다시 검사해 "truncated" 0 을 확인한다 — 0 이 아니면 종료 코드 1.

  python3 examples/configs/alpha/tools/fix_stem_sci_regex.py \
      --pair <in.jsonl> <out.jsonl> [--pair ...] [--seed 20261010]
  2026-10-10 산출: teacher_stem_v1 → v2 (636행) · teacher_stem_v1_val → v2_val (6) · teacher_stem_smoke_v2 → smoke_v3 (16) (alpha_blends/teacher_blends/)
"""
import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure_judge_p04 import gold_regex_defect  # noqa: E402
from teacher_pool_build import decontam, load_bench  # noqa: E402
from teacher_pool_index import norm  # noqa: E402

SRC = "/home/work/Datasets/LL_datasets/posttraining/RL/Nemotron-RL-Science-v1/so_openq.jsonl"
JUDGE_AGENTS = ("equivalence_llm_judge_simple_agent", "ns_tools_simple_agent")


def regex_of(r):
    return (r.get("template_metadata") or {}).get("output_regex")


def status_of(r) -> str:
    return gold_regex_defect(regex_of(r), r["expected_answer"])[0]


def key_of(r):
    return (r["agent_ref"]["name"], (r.get("metadata") or {}).get("topic"), regex_of(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, action="append", required=True, metavar=("IN", "OUT"))
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--buffer", type=int, default=4, help="키마다 필요 수 × buffer 개를 오염 검사한다")
    a = ap.parse_args()

    files = [[json.loads(l) for l in open(i)] for i, _ in a.pair]
    used_uuid, used_text, need = set(), set(), collections.Counter()
    for rows in files:
        for r in rows:
            if r.get("pool_source") != "sci":
                continue
            used_uuid.add(r["uuid"])
            used_text.add(norm(r["problem"]))
            if status_of(r) == "truncated":
                need[key_of(r)] += 1

    pool = collections.defaultdict(list)
    with open(SRC) as f:
        for line in f:
            r = json.loads(line)
            k = key_of(r)
            if k in need and r["uuid"] not in used_uuid and norm(r["problem"]) not in used_text and status_of(r) == "ok":
                pool[k].append(r)
    rng = random.Random(a.seed)
    cands = {}
    for k in sorted(need, key=str):
        rng.shuffle(pool[k])
        for r in pool[k][: need[k] * a.buffer]:
            cands[r["uuid"]] = r
    bench = load_bench()
    flagged, _ = decontam({u: r["problem"] for u, r in cands.items()}, bench)

    queue = {k: [r for r in pool[k][: need[k] * a.buffer] if r["uuid"] not in flagged] for k in need}
    taken_text = set()
    summary = {"seed": a.seed, "source": SRC, "bench_items": len(bench), "decontam_checked": len(cands),
               "decontam_flagged": len(flagged), "files": {}}
    bad_total = 0
    for (src_path, out_path), rows in zip(a.pair, files):
        replaced, by_key = [], collections.Counter()
        for i, r in enumerate(rows):
            if r.get("pool_source") != "sci" or status_of(r) != "truncated":
                continue
            k = key_of(r)
            while True:
                if not queue[k]:
                    sys.exit(f"후보 부족: {k}")
                c = queue[k].pop(0)
                if norm(c["problem"]) not in taken_text:
                    break
            taken_text.add(norm(c["problem"]))
            rows[i] = {**c, "pool_source": "sci"}
            replaced.append({"line": i, "old_uuid": r["uuid"], "new_uuid": c["uuid"], "regex": k[2]})
            by_key[f"{k[0].split('_')[0]}|{k[1]}|{k[2]}"] += 1
        with open(out_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        st_all = collections.Counter(status_of(r) for r in rows if r["agent_ref"]["name"] in JUDGE_AGENTS)
        bad_total += st_all["truncated"]
        st = {"file": out_path, "from": src_path, "rows": len(rows),
              "by_agent": dict(collections.Counter(r["agent_ref"]["name"] for r in rows)),
              "replaced": len(replaced), "replaced_by_key": dict(by_key.most_common()),
              "judge_rows": sum(st_all.values()), "judge_rows_status": dict(st_all), "replacements": replaced}
        json.dump(st, open(out_path + ".stats.json", "w"), ensure_ascii=False, indent=1)
        summary["files"][os.path.basename(out_path)] = {k: st[k] for k in ("rows", "replaced", "judge_rows", "judge_rows_status")}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    sys.exit(1 if bad_total else 0)


if __name__ == "__main__":
    main()
