"""MOPD1 스모크 MS1 판정 — 학습 데이터 덤프(train_data_step*.jsonl)에서 teacher 슬롯별 "teacher − 학생" logprob 차이를 잰다 (CPU).

OPD 지표는 전체 평균(`on_policy_distillation/teacher_student_logprob_gap_mean`)뿐이라 teacher 별로 나뉘지 않는다. 덤프의 advantages 는
clip(log π_teacher − log π_student, ±advantage_clip) × 생성 토큰 마스크이고 (advantage_estimator.py:609-613 · grpo.py advantage_clip),
prev_logprobs 는 학습 엔진의 학생 logprob, generation_logprobs 는 vLLM 롤아웃 logprob 이다.

슬롯 = 블렌드 agent → teacher 경로 (레시피 on_policy_distillation.teacher_model_by_agent_name 을 --routing 으로 준다).
같은 SFT 를 경로만 바꿔 넣은 슬롯(--noise-slot, 여러 번)은 잡음 바닥이고, 실제 teacher 슬롯(--real-slot)은 음성 대조다.
  - 잡음 바닥: |평균 차이| ≤ --noise-mean-max · 평균 |차이| ≤ max(--noise-abs-max, --noise-vs-rollout × 평균 |학생 − 롤아웃|)
  - 음성 대조: 실제 슬롯 평균 |차이| ≥ --real-ratio-min × 잡음 슬롯 최대 평균 |차이|, 평균 차이 < 0 (E_학생[log π_T − log π_S] = −KL(S‖T) ≤ 0)
  - 긴 행: 총 길이 ≥ --long-min 인 샘플이 슬롯에 있고 그 샘플의 생성 토큰 advantage 가 전부 0 이 아니다 (teacher 가 실제로 점수를 냈다)
  - clip 경계 비율 (|advantage| = clip) 과 슬롯별 표본 수를 함께 낸다. 표본이 --min-tokens 보다 적은 슬롯은 판정 불가 → FAIL

  python analyze_mopd_smoke.py --routing <recipe 해석 결과 json 또는 yaml> --noise-slot <경로> ... --real-slot <경로> <dump>/train_data_step*.jsonl
종료 코드 0 = PASS.
"""

import argparse
import collections
import glob
import json
import math
import sys


def load_routing(path):
    if path.endswith(".json"):
        return json.load(open(path))
    import yaml

    return yaml.safe_load(open(path))


def agent_name(row):
    ref = row.get("agent_ref")
    if isinstance(ref, list):
        ref = ref[0] if ref else None
    return (ref or {}).get("name", "?")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--routing", required=True, help="agent → teacher 경로 (yaml/json)")
    ap.add_argument("--noise-slot", action="append", default=[], required=True)
    ap.add_argument("--real-slot", required=True)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--noise-mean-max", type=float, default=0.01)
    ap.add_argument("--noise-abs-max", type=float, default=0.05)
    ap.add_argument("--noise-vs-rollout", type=float, default=3.0)
    ap.add_argument("--real-ratio-min", type=float, default=5.0)
    ap.add_argument("--long-min", type=int, default=48000)
    ap.add_argument("--min-tokens", type=int, default=2000)
    a = ap.parse_args()
    routing = load_routing(a.routing)
    files = sorted({f for p in a.dumps for f in glob.glob(p)})
    if not files:
        print("FAIL no dump files")
        return 1

    st = collections.defaultdict(lambda: collections.Counter())
    sums = collections.defaultdict(lambda: collections.defaultdict(float))
    long_seen = collections.defaultdict(list)
    unknown = collections.Counter()
    for f in files:
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                # log_batched_dict_as_jsonl 은 행 하나 = 샘플 하나
                ag = agent_name(d)
                slot = routing.get(ag)
                if slot is None:
                    unknown[ag] += 1
                    continue
                mask, adv = d["token_loss_mask"], d["advantages"]
                prev, gen = d["prev_logprobs"], d["generation_logprobs"]
                n_tok = sum(1 for m in mask if m)
                total_len = d.get("input_lengths") or len(d["token_ids"])
                if isinstance(total_len, list):
                    total_len = total_len[0]
                st[slot]["samples"] += 1
                nz = 0
                for m, x, p, g in zip(mask, adv, prev, gen):
                    if not m:
                        continue
                    sums[slot]["gap"] += x
                    sums[slot]["abs_gap"] += abs(x)
                    sums[slot]["abs_roll"] += abs(p - g)
                    st[slot]["tokens"] += 1
                    st[slot]["at_clip"] += abs(abs(x) - a.clip) < 1e-6
                    nz += x != 0.0
                if total_len >= a.long_min:
                    long_seen[slot].append((total_len, n_tok, nz))

    res, ok = {}, True
    for slot in sorted(set(a.noise_slot) | {a.real_slot}):
        n = st[slot]["tokens"]
        if n == 0:
            res[slot] = None
            continue
        res[slot] = {"samples": st[slot]["samples"], "tokens": n, "mean_gap": sums[slot]["gap"] / n,
                     "mean_abs_gap": sums[slot]["abs_gap"] / n, "mean_abs_rollout": sums[slot]["abs_roll"] / n,
                     "clip_frac": st[slot]["at_clip"] / n, "long": long_seen[slot]}
    print(json.dumps({"files": len(files), "unknown_agents": dict(unknown), "slots": res}, indent=1))

    def fail(msg):
        nonlocal ok
        ok = False
        print("FAIL", msg)

    if unknown:
        fail(f"routing 에 없는 agent {dict(unknown)}")
    noise_abs = []
    for slot in a.noise_slot:
        r = res.get(slot)
        if not r or r["tokens"] < a.min_tokens:
            fail(f"잡음 슬롯 {slot}: 생성 토큰 {r and r['tokens']} < {a.min_tokens} — 판정 불가")
            continue
        lim = max(a.noise_abs_max, a.noise_vs_rollout * r["mean_abs_rollout"])
        if abs(r["mean_gap"]) > a.noise_mean_max:
            fail(f"잡음 슬롯 {slot}: |평균 차이| {abs(r['mean_gap']):.4f} > {a.noise_mean_max}")
        if r["mean_abs_gap"] > lim:
            fail(f"잡음 슬롯 {slot}: 평균 |차이| {r['mean_abs_gap']:.4f} > {lim:.4f}")
        noise_abs.append(r["mean_abs_gap"])
    r = res.get(a.real_slot)
    if not r or r["tokens"] < a.min_tokens:
        fail(f"실제 슬롯 {a.real_slot}: 생성 토큰 {r and r['tokens']} < {a.min_tokens} — 판정 불가")
    else:
        if noise_abs and r["mean_abs_gap"] < a.real_ratio_min * max(noise_abs):
            fail(f"실제 슬롯 평균 |차이| {r['mean_abs_gap']:.4f} < {a.real_ratio_min} × 잡음 최대 {max(noise_abs):.4f}")
        if not r["mean_gap"] < 0:
            fail(f"실제 슬롯 평균 차이 {r['mean_gap']:.4f} ≥ 0 — 역 KL 추정은 음수여야 한다")
    if not any(long_seen.values()):
        fail(f"총 길이 ≥ {a.long_min} 인 샘플이 덤프에 없다 — teacher 메모리 탐침이 실행되지 않았다")
    for slot, items in long_seen.items():
        for total_len, n_tok, nz in items:
            if n_tok and nz == 0:
                fail(f"슬롯 {slot} 긴 샘플 (총 {total_len}) 의 advantage 가 전부 0 — teacher 점수가 없다")
    print("MS1 dump verdict:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
