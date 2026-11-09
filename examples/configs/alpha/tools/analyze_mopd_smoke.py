"""MOPD1 스모크 MS1 판정 — 학습 데이터 덤프(train_data_step*.jsonl)에서 teacher 슬롯별 "teacher − 학생" logprob 차이를 잰다 (CPU).

OPD 지표는 전체 평균(`on_policy_distillation/teacher_student_logprob_gap_mean`)뿐이라 teacher 별로 나뉘지 않는다. 덤프의 advantages 는
clip(log π_teacher − log π_student, ±advantage_clip) × 생성 토큰 마스크이고 (advantage_estimator.py:609-613 · grpo.py advantage_clip),
prev_logprobs 는 학습 엔진의 학생 logprob, generation_logprobs 는 vLLM 롤아웃 logprob 이다.

슬롯 = 블렌드 agent → teacher 경로 (레시피 on_policy_distillation.teacher_model_by_agent_name 을 --routing 으로 준다).
같은 SFT 를 경로만 바꿔 넣은 슬롯(--noise-slot, 여러 번)은 잡음 바닥이고, 실제 teacher 슬롯(--real-slot)은 음성 대조다.
  - 잡음 바닥: |평균 차이| ≤ --noise-mean-max · 평균 |차이| ≤ max(--noise-abs-max, --noise-vs-rollout × 평균 |학생 − 롤아웃|)
  - 음성 대조: 실제 슬롯 평균 |차이| ≥ --real-ratio-min × 잡음 슬롯 최대 평균 |차이|, 그리고 상한 — clip 경계 비율 ≤ --real-clip-max ·
    평균 차이 ≥ --real-mean-min (적대적 검토 H2: 변환이 깨진 teacher 도 "크고 음수" 라 하한만으로는 통과한다. 건강한 s2r 228 실측 clip 0.53% · −0.278,
    균등 분포 teacher clip ≈ 100% · −5)
  - 긴 행 (총 길이 ≥ --long-min): 잡음 슬롯에 있어야 하고, **행마다** 잡음 기준(|평균| · 평균 |차이| 상한 · clip 0)을 건다 (검토 H1: advantage ≠ 0 만으로는
    teacher logprob 이 0 이어도 −prev 가 남아 통과한다)
  - sample_loss_mask = 0 행 (로그 오차 마스킹 · Gym mask_sample) 은 advantage 가 0 이라 집계에서 빼고 수를 보고한다 (검토 B1)
  - 실제 덤프는 필드마다 배치 차원 1 로 감싼다 (logger.py log_batched_dict_as_jsonl → make_microbatch_iterator(1)). 평평한 형식도 읽는다
  - clip 경계 비율 (|advantage| = clip) 과 슬롯별 표본 수를 함께 낸다. 표본이 --min-tokens 보다 적은 슬롯은 판정 불가 → FAIL

  python analyze_mopd_smoke.py --routing <recipe 해석 결과 json 또는 yaml> --noise-slot <경로> ... --real-slot <경로> <dump>/train_data_step*.jsonl
본 런 첫 step 게이트 (MS2, teacher 넷 모두 실제): 잡음 슬롯 없이 --noise-floor (MS1 잡음 슬롯 평균 |차이| 95% 상한) 를 주고 --real-slot 을 teacher 마다,
--require-long 0. 실제 슬롯마다 "SFT 와 구분됨 (평균 |차이| > 잡음 바닥)" 과 상한 (clip · 평균) 을 본다.
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
    ap.add_argument("--noise-slot", action="append", default=[])
    ap.add_argument("--real-slot", action="append", default=[], required=True)
    ap.add_argument("--noise-floor", type=float, default=None, help="잡음 슬롯이 없을 때 쓰는 잡음 평균 |차이| (MS1 실측 상한)")
    ap.add_argument("--require-long", type=int, default=1)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--noise-mean-max", type=float, default=0.01)
    ap.add_argument("--noise-abs-max", type=float, default=0.05)
    ap.add_argument("--noise-vs-rollout", type=float, default=3.0)
    ap.add_argument("--real-ratio-min", type=float, default=5.0)
    ap.add_argument("--real-clip-max", type=float, default=0.05)
    ap.add_argument("--real-mean-min", type=float, default=-1.0)
    ap.add_argument("--boot", type=int, default=2000)
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
    per_seq = collections.defaultdict(list)  # slot -> [(sum_abs_gap, n_tok)]
    unknown = collections.Counter()
    for f in files:
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                # log_batched_dict_as_jsonl 은 행 하나 = 샘플 하나 (실제 덤프 확인 2026-11-09 MS1b: 64 행 = 64 샘플)
                ag = agent_name(d)
                slot = routing.get(ag)
                if slot is None:
                    unknown[ag] += 1
                    continue
                # 실제 덤프는 필드마다 배치 차원 1 로 한 번 더 감싼다 (log_batched_dict_as_jsonl: list[1]<list[S]>)
                un = lambda v: v[0] if isinstance(v, list) and len(v) == 1 and isinstance(v[0], list) else v
                mask, adv = un(d["token_loss_mask"]), un(d["advantages"])
                prev, gen = un(d["prev_logprobs"]), un(d["generation_logprobs"])
                if not (len(mask) == len(adv) == len(prev) == len(gen)):
                    raise SystemExit(f"{f}: field lengths differ {len(mask)} {len(adv)} {len(prev)} {len(gen)}")
                slm = d.get("sample_loss_mask", 1.0)
                slm = slm[0] if isinstance(slm, list) else slm
                if not slm:
                    st[slot]["masked_samples"] += 1
                    continue
                n_tok = sum(1 for m in mask if m)
                total_len = d.get("input_lengths")
                if isinstance(total_len, list):
                    total_len = total_len[0]
                if total_len is None:
                    total_len = len(un(d["token_ids"]))
                st[slot]["samples"] += 1
                nz = s_gap = s_abs = s_roll = s_clip = 0.0
                for m, x, p, g in zip(mask, adv, prev, gen):
                    if not m:
                        continue
                    s_gap += x
                    s_abs += abs(x)
                    s_roll += abs(p - g)
                    s_clip += abs(abs(x) - a.clip) < 1e-6
                    nz += x != 0.0
                sums[slot]["gap"] += s_gap
                sums[slot]["abs_gap"] += s_abs
                sums[slot]["abs_roll"] += s_roll
                st[slot]["tokens"] += n_tok
                st[slot]["at_clip"] += s_clip
                if n_tok:
                    per_seq[slot].append((s_abs, n_tok))
                if total_len >= a.long_min:
                    long_seen[slot].append({"total_len": total_len, "gen_tokens": n_tok, "nonzero": int(nz),
                                            "mean_gap": s_gap / n_tok if n_tok else None,
                                            "mean_abs_gap": s_abs / n_tok if n_tok else None,
                                            "mean_abs_rollout": s_roll / n_tok if n_tok else None, "clip": int(s_clip)})

    import random

    rng = random.Random(20261109)

    def boot_ci(items):
        if len(items) < 2:
            return None
        vals = []
        for _ in range(a.boot):
            pick = [items[rng.randrange(len(items))] for _ in items]
            vals.append(sum(x for x, _ in pick) / max(sum(n for _, n in pick), 1))
        vals.sort()
        return [round(vals[int(0.025 * a.boot)], 4), round(vals[int(0.975 * a.boot) - 1], 4)]

    res, ok = {}, True
    for slot in sorted(set(a.noise_slot) | set(a.real_slot)):
        n = st[slot]["tokens"]
        if n == 0:
            res[slot] = None
            continue
        res[slot] = {"samples": st[slot]["samples"], "masked_samples": st[slot]["masked_samples"], "tokens": n,
                     "mean_abs_gap_ci95": boot_ci(per_seq[slot]), "mean_gap": sums[slot]["gap"] / n,
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
    if not a.noise_slot and a.noise_floor is None:
        fail("잡음 슬롯도 --noise-floor 도 없다 — 실제 슬롯이 SFT 와 구분되는지 판정할 수 없다")
    for real in a.real_slot:
        r = res.get(real)
        if not r or r["tokens"] < a.min_tokens:
            fail(f"실제 슬롯 {real}: 생성 토큰 {r and r['tokens']} < {a.min_tokens} — 판정 불가")
            continue
        if noise_abs and r["mean_abs_gap"] < a.real_ratio_min * max(noise_abs):
            fail(f"실제 슬롯 {real} 평균 |차이| {r['mean_abs_gap']:.4f} < {a.real_ratio_min} × 잡음 최대 {max(noise_abs):.4f}")
        if a.noise_floor is not None and r["mean_abs_gap"] <= a.noise_floor:
            fail(f"실제 슬롯 {real} 평균 |차이| {r['mean_abs_gap']:.4f} ≤ 잡음 바닥 {a.noise_floor} — teacher 가 SFT(학생)와 구분되지 않는다")
        if not r["mean_gap"] < 0:
            fail(f"실제 슬롯 {real} 평균 차이 {r['mean_gap']:.4f} ≥ 0 — 역 KL 추정은 음수여야 한다")
        if r["clip_frac"] > a.real_clip_max:
            fail(f"실제 슬롯 {real} clip 경계 비율 {r['clip_frac']:.4f} > {a.real_clip_max} — teacher logprob 이 망가졌을 수 있다")
        if r["mean_gap"] < a.real_mean_min:
            fail(f"실제 슬롯 {real} 평균 차이 {r['mean_gap']:.4f} < {a.real_mean_min} — teacher 가 학생 토큰을 거의 0 확률로 본다")
    if a.require_long and not any(long_seen.values()):
        fail(f"총 길이 ≥ {a.long_min} 인 샘플이 덤프에 없다 — teacher 메모리 탐침이 실행되지 않았다")
    for slot, items in long_seen.items():
        for it in items:
            if not a.require_long:
                continue
            if slot not in a.noise_slot:
                fail(f"긴 샘플 (총 {it['total_len']}) 이 잡음 슬롯이 아닌 {slot} 에 있다 — 행별 잡음 판정을 할 수 없다")
                continue
            if not it["gen_tokens"]:
                fail(f"긴 샘플 (총 {it['total_len']}) 의 생성 토큰이 0")
                continue
            lim = max(a.noise_abs_max, a.noise_vs_rollout * it["mean_abs_rollout"])
            if abs(it["mean_gap"]) > a.noise_mean_max or it["mean_abs_gap"] > lim or it["clip"] or it["nonzero"] == 0:
                fail(f"긴 샘플 (총 {it['total_len']}, 슬롯 {slot}) 이 잡음 기준을 벗어난다 — 평균 {it['mean_gap']:.4f} · 평균 |차이| "
                     f"{it['mean_abs_gap']:.4f} (상한 {lim:.4f}) · clip {it['clip']} · 비0 {it['nonzero']}/{it['gen_tokens']}")
    print("MS1 dump verdict:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
