"""MOPD 스모크 MS1 · 본 런 첫 step 게이트 MS2 판정 — 학습 데이터 덤프(train_data_step*.jsonl)에서 teacher 슬롯별 "teacher − 학생" logprob 차이를 잰다 (CPU).

OPD 지표는 전체 평균(`on_policy_distillation/teacher_student_logprob_gap_mean`)뿐이라 teacher 별로 나뉘지 않는다. 덤프의 advantages 는
clip(log π_teacher − log π_student, ±advantage_clip) × 생성 토큰 마스크 × sample_mask 이고 (advantage_estimator.py:609-613 · grpo.py:5195-5196),
prev_logprobs 는 학습 엔진의 학생 logprob, generation_logprobs 는 vLLM 롤아웃 logprob 이다. 실제 덤프는 필드마다 배치 차원 1 로 감싼다
(logger.py log_batched_dict_as_jsonl → make_microbatch_iterator(1)). 평평한 형식도 읽는다. sample_loss_mask = 0 행은 빼고 수를 보고한다.

슬롯 = 블렌드 agent → teacher 경로 (--routing). routing 의 슬롯은 전부 --noise-slot 이나 --real-slot 이어야 한다 (빠지면 FAIL — 적대적 검토 2차 H1).
  잡음 슬롯 (같은 SFT 를 경로만 바꿔 넣음): |평균 차이| ≤ --noise-mean-max · 평균 |차이| ≤ max(--noise-abs-max, --noise-vs-rollout × 평균 |학생 − 롤아웃|)
  실제 슬롯:
    - 학습 신호 확인: 평균 |차이| 의 시퀀스 부트스트랩 95% 하한 > 그 슬롯의 잡음 바닥 (--slot-floor 경로=값, 없으면 --noise-floor 또는 잡음 슬롯 최대 CI 상한).
      아니면 FAIL 이 아니라 "증거 부족" — SFT 에 가까운 정상 teacher 일 수 있으니 신원 증거(--driver-log)로 가린다 (검토 2차 M2).
      잡음 슬롯이 있으면 추가로 평균 |차이| ≥ --real-ratio-min × 잡음 최대 (MS1 음성 대조).
    - 손상 확인 (검토 2차 M1): 학생이 확신한 토큰 (prev_logprob ≥ --confident-lp) 중 차이 < --confident-bad-gap 인 비율 ≤ --confident-bad-max.
      정상 teacher 는 학생이 확신한 토큰에서 거의 어긋나지 않고 (s2r 228 실측 0.03%), 망가진 teacher 는 거기서도 clip 된다 (균등 분포 ≈ 100%).
      전체 clip 비율 > --real-clip-fail 이면 FAIL. 전체 평균 차이는 기록만 한다 (많이 움직인 정상 teacher 는 −1 보다 낮을 수 있다).
  긴 행 (총 길이 ≥ --long-min): 잡음 슬롯의 긴 행은 행마다 |평균| ≤ max(--noise-mean-max, 4·sd/√n) · 평균 |차이| 상한 · clip 0 · 비0 > 0 (검토 1차 H1 · 2차 M4).
    실제 슬롯의 긴 행은 기록만 한다. --require-long 1 이면 잡음 슬롯 긴 행이 하나 이상 있어야 한다 (teacher 1장 메모리 탐침).
  sample_loss_mask 0 행 비율 ≤ --masked-max (전체 합).
  --driver-log: teacher 신원 (검토 2차 M3) — "Initializing teacher_<alias> workers" 순서와, 드라이버 노드 Megatron 워커 pid 별 첫 `mcore_ckpt/model_<경로>` 줄을
    pid 오름차순으로 맞춰 alias → 로드한 경로가 routing 과 같은지 본다 (teacher 는 순서대로 하나씩 띄운다). --student-path 의 줄은 뺀다. 파싱이 안 되면 증거 부족.

  python analyze_mopd_smoke.py <dump>/train_data_step*.jsonl --routing <json|yaml> [--noise-slot 경로 ...] --real-slot 경로 ... \
      [--slot-floor 경로=값 ...] [--noise-floor 값] [--require-long 0|1] [--driver-log <tag>.log --student-path <학생 HF 경로>]
종료 코드: 0 PASS · 1 FAIL · 2 증거 부족.
"""

import argparse
import collections
import glob
import json
import math
import random
import re
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


def un(v):
    return v[0] if isinstance(v, list) and len(v) == 1 and isinstance(v[0], list) else v


def teacher_identity(log_path, routing, student_path):
    """alias 초기화 순서 ↔ 드라이버 노드 Megatron 워커 pid 별 첫 mcore 경로. (판정, 상세) — 판정 None = 파싱 불가."""
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    aliases, pid_path = [], {}

    def enc(p):
        return "model_" + p.replace("/", "_")

    student_enc = enc(student_path) if student_path else None
    for line in open(log_path, errors="replace"):
        line = ansi.sub("", line)
        for m in re.finditer(r"Initializing teacher_([A-Za-z0-9_]+) workers", line):
            if m.group(1) not in aliases:
                aliases.append(m.group(1))
        m = re.search(r"\(MegatronPolicyWorker pid=(\d+)(, ip=[0-9.]+)?\).*?mcore_ckpt/(model_[^\s\]]+)", line)
        if m and not m.group(2):
            pid, model = int(m.group(1)), m.group(3)
            if student_enc and model.startswith(student_enc):
                continue
            pid_path.setdefault(pid, model)
    detail = {"aliases": aliases, "pid_model": {str(k): v for k, v in sorted(pid_path.items())}}
    if not aliases or len(pid_path) != len(aliases):
        return None, detail
    by_pid = [pid_path[p] for p in sorted(pid_path)]
    mism = []
    for alias, model in zip(aliases, by_pid):
        want = routing.get(alias)
        if want is None or model != enc(want):
            mism.append((alias, want, model))
    detail["mismatch"] = mism
    return (not mism), detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--routing", required=True, help="agent → teacher 경로 (yaml/json)")
    ap.add_argument("--noise-slot", action="append", default=[])
    ap.add_argument("--real-slot", action="append", default=[], required=True)
    ap.add_argument("--slot-floor", action="append", default=[], help="경로=값 — 실제 슬롯별 잡음 바닥")
    ap.add_argument("--noise-floor", type=float, default=None, help="슬롯별 바닥이 없을 때 쓰는 잡음 바닥")
    ap.add_argument("--require-long", type=int, default=1)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--noise-mean-max", type=float, default=0.01)
    ap.add_argument("--noise-abs-max", type=float, default=0.05)
    ap.add_argument("--noise-vs-rollout", type=float, default=3.0)
    ap.add_argument("--real-ratio-min", type=float, default=5.0)
    ap.add_argument("--confident-lp", type=float, default=-0.01)
    ap.add_argument("--confident-bad-gap", type=float, default=-1.0)
    ap.add_argument("--confident-bad-max", type=float, default=0.05)
    ap.add_argument("--real-clip-fail", type=float, default=0.5)
    ap.add_argument("--masked-max", type=float, default=0.01)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--long-min", type=int, default=48000)
    ap.add_argument("--min-tokens", type=int, default=2000)
    ap.add_argument("--driver-log")
    ap.add_argument("--student-path")
    a = ap.parse_args()
    routing = load_routing(a.routing)
    floors = {k: float(v) for k, v in (x.rsplit("=", 1) for x in a.slot_floor)}
    files = sorted({f for p in a.dumps for f in glob.glob(p)})
    fails, insufficient = [], []

    def fail(msg):
        fails.append(msg)
        print("FAIL", msg)

    def insuff(msg):
        insufficient.append(msg)
        print("INSUFFICIENT", msg)

    if not files:
        fail("no dump files")
        return 1
    uncovered = sorted(set(routing.values()) - set(a.noise_slot) - set(a.real_slot))
    if uncovered:
        fail(f"routing 슬롯 {uncovered} 가 --noise-slot · --real-slot 어디에도 없다 — 검사 없이 지나간다")

    st = collections.defaultdict(collections.Counter)
    sums = collections.defaultdict(lambda: collections.defaultdict(float))
    per_seq = collections.defaultdict(list)  # slot -> [(sum_abs_gap, n_tok)]
    long_rows = collections.defaultdict(list)
    unknown = collections.Counter()
    for f in files:
        with open(f) as fh:
            for line in fh:  # 행 하나 = 샘플 하나 (실제 덤프 확인 2026-11-09 MS1b: 64 행 = 64 샘플)
                d = json.loads(line)
                ag = agent_name(d)
                slot = routing.get(ag)
                if slot is None:
                    unknown[ag] += 1
                    continue
                mask, adv = un(d["token_loss_mask"]), un(d["advantages"])
                prev, gen = un(d["prev_logprobs"]), un(d["generation_logprobs"])
                if not (len(mask) == len(adv) == len(prev) == len(gen)):
                    raise SystemExit(f"{f}: field lengths differ {len(mask)} {len(adv)} {len(prev)} {len(gen)}")
                st[slot]["rows"] += 1
                slm = d.get("sample_loss_mask", 1.0)
                slm = slm[0] if isinstance(slm, list) else slm
                if not slm:
                    st[slot]["masked_samples"] += 1
                    continue
                total_len = d.get("input_lengths")
                total_len = total_len[0] if isinstance(total_len, list) else total_len
                if total_len is None:
                    total_len = len(un(d["token_ids"]))
                st[slot]["samples"] += 1
                n = nz = clip = conf = conf_bad = 0
                s_gap = s_sq = s_abs = s_roll = 0.0
                for m, x, p, g in zip(mask, adv, prev, gen):
                    if not m:
                        continue
                    n += 1
                    s_gap += x
                    s_sq += x * x
                    s_abs += abs(x)
                    s_roll += abs(p - g)
                    clip += abs(abs(x) - a.clip) < 1e-6
                    nz += x != 0.0
                    if p >= a.confident_lp:
                        conf += 1
                        conf_bad += x < a.confident_bad_gap
                sums[slot]["gap"] += s_gap
                sums[slot]["abs_gap"] += s_abs
                sums[slot]["abs_roll"] += s_roll
                for k, v in (("tokens", n), ("at_clip", clip), ("conf", conf), ("conf_bad", conf_bad)):
                    st[slot][k] += v
                if n:
                    per_seq[slot].append((s_abs, n))
                if total_len >= a.long_min:
                    mean = s_gap / n if n else None
                    sd = math.sqrt(max(s_sq / n - mean * mean, 0.0)) if n else None
                    long_rows[slot].append({"total_len": total_len, "gen_tokens": n, "nonzero": nz, "clip": clip,
                                            "mean_gap": mean, "sd": sd, "mean_abs_gap": s_abs / n if n else None,
                                            "mean_abs_rollout": s_roll / n if n else None})

    rng = random.Random(20261109)

    def boot_ci(items):
        if len(items) < 2:
            return None
        vals = sorted(sum(x for x, _ in pick) / max(sum(k for _, k in pick), 1)
                      for pick in ([items[rng.randrange(len(items))] for _ in items] for _ in range(a.boot)))
        return [vals[int(0.025 * a.boot)], vals[int(0.975 * a.boot) - 1]]

    res = {}
    for slot in sorted(set(a.noise_slot) | set(a.real_slot)):
        n = st[slot]["tokens"]
        res[slot] = None if not n else {
            "rows": st[slot]["rows"], "masked_samples": st[slot]["masked_samples"], "tokens": n,
            "mean_gap": sums[slot]["gap"] / n, "mean_abs_gap": sums[slot]["abs_gap"] / n, "mean_abs_gap_ci95": boot_ci(per_seq[slot]),
            "mean_abs_rollout": sums[slot]["abs_roll"] / n, "clip_frac": st[slot]["at_clip"] / n,
            "confident_frac": st[slot]["conf"] / n, "confident_bad_frac": st[slot]["conf_bad"] / max(st[slot]["conf"], 1),
            "long": long_rows[slot]}
    print(json.dumps({"files": len(files), "unknown_agents": dict(unknown), "slots": res}, indent=1))

    if unknown:
        fail(f"routing 에 없는 agent {dict(unknown)}")
    rows = sum(st[s]["rows"] for s in st)
    masked = sum(st[s]["masked_samples"] for s in st)
    if rows and masked / rows > a.masked_max:
        fail(f"sample_loss_mask 0 행 {masked}/{rows} > {a.masked_max:.0%}")
    noise_abs, noise_hi = [], []
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
        if r["mean_abs_gap_ci95"]:
            noise_hi.append(r["mean_abs_gap_ci95"][1])
    for real in a.real_slot:
        r = res.get(real)
        if not r or r["tokens"] < a.min_tokens:
            fail(f"실제 슬롯 {real}: 생성 토큰 {r and r['tokens']} < {a.min_tokens} — 판정 불가")
            continue
        floor = floors.get(real, a.noise_floor if a.noise_floor is not None else (max(noise_hi) if noise_hi else None))
        if floor is None:
            fail(f"실제 슬롯 {real}: 잡음 바닥이 없다 (--slot-floor · --noise-floor · 잡음 슬롯 중 하나가 필요)")
        elif not r["mean_abs_gap_ci95"] or r["mean_abs_gap_ci95"][0] <= floor:
            lo = r["mean_abs_gap_ci95"][0] if r["mean_abs_gap_ci95"] else None
            insuff(f"실제 슬롯 {real}: 평균 |차이| 95% 하한 {lo} ≤ 잡음 바닥 {floor} — SFT 와 구분되지 않는다 "
                   "(SFT 에 가까운 정상 teacher 이거나 학생 가중치를 올린 teacher). 신원 증거로 가린다")
        if noise_abs and r["mean_abs_gap"] < a.real_ratio_min * max(noise_abs):
            fail(f"실제 슬롯 {real}: 평균 |차이| {r['mean_abs_gap']:.4f} < {a.real_ratio_min} × 잡음 최대 {max(noise_abs):.4f}")
        if r["confident_bad_frac"] > a.confident_bad_max:
            fail(f"실제 슬롯 {real}: 학생 확신 토큰 중 차이 < {a.confident_bad_gap} 비율 {r['confident_bad_frac']:.4f} > "
                 f"{a.confident_bad_max} — teacher logprob 손상")
        if r["clip_frac"] > a.real_clip_fail:
            fail(f"실제 슬롯 {real}: clip 경계 비율 {r['clip_frac']:.4f} > {a.real_clip_fail} — teacher logprob 손상")
    if a.require_long and not any(long_rows[s] for s in a.noise_slot):
        fail(f"잡음 슬롯에 총 길이 ≥ {a.long_min} 인 샘플이 없다 — teacher 메모리 탐침이 실행되지 않았다")
    for slot in a.noise_slot:
        for it in long_rows[slot]:
            if not it["gen_tokens"]:
                fail(f"긴 샘플 (총 {it['total_len']}) 의 생성 토큰이 0")
                continue
            lim = max(a.noise_abs_max, a.noise_vs_rollout * it["mean_abs_rollout"])
            mlim = max(a.noise_mean_max, 4 * it["sd"] / math.sqrt(it["gen_tokens"]))
            if abs(it["mean_gap"]) > mlim or it["mean_abs_gap"] > lim or it["clip"] or it["nonzero"] == 0:
                fail(f"긴 샘플 (총 {it['total_len']}, 슬롯 {slot}) 이 잡음 기준을 벗어난다 — 평균 {it['mean_gap']:.4f} (상한 {mlim:.4f}) · "
                     f"평균 |차이| {it['mean_abs_gap']:.4f} (상한 {lim:.4f}) · clip {it['clip']} · 비0 {it['nonzero']}/{it['gen_tokens']}")
    if a.driver_log:
        ok_id, detail = teacher_identity(a.driver_log, routing, a.student_path)
        print("teacher identity:", json.dumps(detail, ensure_ascii=False))
        if ok_id is None:
            insuff("드라이버 로그에서 teacher alias ↔ 로드 경로를 맞추지 못했다")
        elif not ok_id:
            fail(f"teacher 가 routing 과 다른 경로를 로드했다: {detail['mismatch']}")
    verdict = "FAIL" if fails else "INSUFFICIENT" if insufficient else "PASS"
    print("dump verdict:", verdict)
    return {"PASS": 0, "FAIL": 1, "INSUFFICIENT": 2}[verdict]


if __name__ == "__main__":
    sys.exit(main())
