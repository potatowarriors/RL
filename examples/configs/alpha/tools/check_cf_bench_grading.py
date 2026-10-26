"""[2026-10-27 적대적 검토 판정: 신뢰 불가 — 기준선 미실행] 214문제 중 212문제가 SFT 블렌드 cp_v2 에 원문으로 있고(held-out 아님), 상수 출력이 7문제를 통과하는 퇴화 테스트·대소문자 불일치·다중 정답 문제가 있다. 고치기 전에는 쓰지 않는다 (KNOWN_ISSUES 2026-10-27, 검토 보고 job tmp/cf_review/review.md).
Codeforces 벤치 채점 대조 시험 (CPU). Gym code_gen 의 실제 채점 함수(lcb_integration check_correctness)로
문제마다 다섯 가지 프로그램을 돌려 판정이 기대와 같은지 본다 — LCB 변환 때(2026-10-19) 한 조회 프로그램 시험과 같은 방식.
  lookup   : 테스트 입력 → 출력 조회표 (정답) — 통과해야 한다
  wrong1   : 첫 테스트 출력만 틀리게 — 실패해야 한다
  nearmiss : 가운데 테스트 출력의 마지막 토큰을 바꿈 — 실패해야 한다
  empty    : 아무것도 안 씀 — 실패해야 한다
  tle      : 무한 루프 — 실패해야 한다 (시간 제한)
사용 (Gym code_gen venv python, 리포 Gym code_gen 디렉토리를 sys.path 에):
  <gym code_gen venv>/bin/python check_cf_bench_grading.py <bench.jsonl> [--n 20] [--timeout 10]
"""
import argparse, hashlib, json, random, sys, os, collections
sys.path.insert(0, "/home/work/vidsearch/repos/project_s/NeMo-RL/3rdparty/Gym-workspace/Gym/resources_servers/code_gen")
from lcb_integration.compute_code_generation_metrics import check_correctness

def lookup_prog(ins, outs, mutate=None):
    table = {hashlib.md5(i.strip().encode()).hexdigest(): o for i, o in zip(ins, outs)}
    if mutate is not None:
        k, v = mutate; table[hashlib.md5(ins[k].strip().encode()).hexdigest()] = v
    return ("import sys, hashlib\nT=" + repr(table) + "\ns=sys.stdin.read()\n"
            "sys.stdout.write(T.get(hashlib.md5(s.strip().encode()).hexdigest(), ''))\n")

def bump(out):
    t = out.rstrip().split()
    if not t: return "X"
    last = t[-1]
    t[-1] = str(int(last) + 1) if last.lstrip("-").isdigit() else last + "x"
    return " ".join(t) + "\n"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("bench"); ap.add_argument("--n", type=int, default=20); ap.add_argument("--timeout", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261027); a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.bench)]
    random.Random(a.seed).shuffle(rows); rows = rows[:a.n]
    expect = {"lookup": True, "wrong1": False, "nearmiss": False, "empty": False, "tle": False}
    res = collections.defaultdict(list)
    for r in rows:
        ut = r["verifier_metadata"]["unit_tests"]; ins, outs = ut["inputs"], ut["outputs"]
        progs = {"lookup": lookup_prog(ins, outs), "wrong1": lookup_prog(ins, outs, (0, "WRONG\n")),
                 "nearmiss": lookup_prog(ins, outs, (len(ins) // 2, bump(outs[len(ins) // 2]))),
                 "empty": "pass\n", "tle": "while True:\n    pass\n"}
        sample = {"input_output": json.dumps(ut)}
        for name, code in progs.items():
            result, meta = check_correctness(sample, code, a.timeout, debug=False)
            ok = bool(result) and all(x is True for x in result)
            res[name].append(ok == expect[name])
            if ok != expect[name]:
                print(f"MISMATCH {r['verifier_metadata']['problem_id']} {name}: got pass={ok} result={str(result)[:120]}", flush=True)
    summary = {k: f"{sum(v)}/{len(v)}" for k, v in res.items()}
    print(json.dumps({"bench": a.bench, "n": len(rows), "agree_with_expectation": summary}))
    sys.exit(0 if all(all(v) for v in res.values()) else 1)

if __name__ == "__main__":
    main()
