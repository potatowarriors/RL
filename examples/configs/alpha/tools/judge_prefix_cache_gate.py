"""R1-L 판정기 — prefix cache 적중이 실제로 일어나는 R1 (GRPO Generation KL) 변형의 유효성 · 통과 · 음성 대조를 판정한다 (CPU 전용).

게이트 정의 · 기준의 출처 · 적대적 검토 결과는 docs/GATES.md "R1-L" 절. 런 디렉토리 규약은 $NRL_ROOT/gates/prefix_cache_*/run_l.sh 와 같다:
  <gate_dir>/gate_<run>.log · rc_<run> · <run>/exp_001/train_data_step{1,2,3}.jsonl

  python judge_prefix_cache_gate.py <gate_dir> --mode withB|noB   # R1-L 전체 판정 (종료 코드 0 = 필수 판정이 모두 있고 모두 통과)
  python judge_prefix_cache_gate.py --pair <dirA> <dirB>       # 두 런의 짝 비교만 (B − A). K 매트릭스 (R3 × 라우터 정밀도) 분석에도 쓴다

통계 (두 차례 적대적 검토 반영, 실행 전에 고정):
  - 토큰 d = prev_logprob − generation_logprob (학습측 − 롤아웃측), k3 = exp(d) − d − 1. 게이트 출력 `Generation KL Error` 는 생성 토큰의 k3 평균이다.
  - 단위는 프롬프트 (step, idx // G) 다 — 한 프롬프트의 G 개 생성은 독립이 아니다. 같은 seed 라 런 사이에 프롬프트가 같으므로 프롬프트로 짝짓는다
    (프롬프트 토큰이 다르면 중단).
  - S_first = 생성 위치 [0, 256) 의 k3 평균 차 — 복원된 상태의 오차는 생성 앞부분에 몰린다 (전체 평균은 뒤 토큰 수천 개에 희석된다).
    S_all = 생성 위치 [0, 2048) 의 k3 평균 차.  S_tail = |d| > 0.5 토큰 비율의 차.
    통과는 세 통계 모두 z < 2.33 (단측 1%). z = 프롬프트별 짝 차이의 평균 / (표준편차 / sqrt(n)). 차이가 전부 0 이면 "동일" 로 본다 (z = 0).
    검출 가능한 최소 효과 (MDE, 검정력 95%) = (2.33 + 1.64) × 표준오차 를 함께 낸다.
  - S_did = (복사본 1..G−1 − 복사본 0, [0,256) k3) 의 차는 참고값이다 — 음성 대조가 복사본 0 에도 걸려 이 통계의 검출력은 검증되지 않았다.
  - 엔진 설정이 같은 두 런은 step 1 롤아웃이 비트 단위로 같다 (검토에서 11-12 R1 과 11-25 K0 로 확인: 128/128). 그래서
      · A/A 반복 (LA_off2 − LA_off) 의 보정 검정은 step 2·3 로 하고, step 1 은 토큰 동일성과 학습측 logprob 차이를 따로 낸다.
      · 경로만 건드리는 음성 대조 (NC_shift · NC_trunc) 는 LA_on 과 step 1 토큰이 같아야 한다 — 같은 토큰 위에서 학습측 효과를 직접 본다.
  - 음성 대조의 민감도는 LA_on 대비로 낸다 (캐시 켬 자체의 이동이 "잡힘" 에 섞이지 않게). 반드시 잡혀야 하는 것: NC_zero (복원 상태 0) ·
    NC_shift (접두사 expert 번호 +1) · NC_trunc (경로 행 누락 → 경고 줄). NC_state (× 1.01) 는 검출 한계를 보는 참고값이다.
"""

import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np

Z_CRIT = 2.33
KL_GATE = 0.002
FIRST, ALL = 256, 2048
PASS_STATS = ("first", "all", "tail")


def load_run(run_dir):
    exps = sorted(glob.glob(os.path.join(run_dir, "exp_*")))
    if len(exps) != 1:
        raise SystemExit(f"{run_dir}: exp_* 디렉토리가 {len(exps)} 개 — 재시도 런이 섞이면 판정하지 않는다")
    steps = {}
    for f in sorted(glob.glob(os.path.join(exps[0], "train_data_step*.jsonl"))):
        s = int(re.search(r"step(\d+)\.jsonl$", f).group(1))
        rows = [json.loads(line) for line in open(f)]
        if rows and all(np.all(np.asarray(r["prev_logprobs"][0])[np.asarray(r["token_loss_mask"][0], bool)] == 0) for r in rows):
            raise SystemExit(f"{f}: prev_logprobs 가 전부 0 — prev_logprob 패스를 건너뛴 덤프로는 KL 을 낼 수 없다")
        out = []
        for r in rows:
            m = np.asarray(r["token_loss_mask"][0], bool)
            g = np.asarray(r["generation_logprobs"][0], float)
            p = np.asarray(r["prev_logprobs"][0], float)
            ids = np.asarray(r["token_ids"][0])
            gen = np.flatnonzero(m)
            if len(gen) == 0:
                n_in = int(r["input_lengths"][0])
                out.append(dict(idx=r["idx"], d=np.zeros(0), g=np.zeros(0), p=np.zeros(0), gen_ids=(), prompt=tuple(ids[:n_in].tolist()), plen=n_in))
                continue
            out.append(dict(idx=r["idx"], d=p[gen] - g[gen], g=g[gen], p=p[gen], gen_ids=tuple(ids[gen].tolist()),
                            prompt=tuple(ids[: gen[0]].tolist()), plen=int(gen[0])))
        steps[s] = sorted(out, key=lambda o: o["idx"])
    return steps


def k3(d):
    return np.exp(d) - d - 1


def group_size(rows):
    """한 프롬프트의 생성 수 G — 연속한 같은 프롬프트의 길이."""
    g = 1
    while g < len(rows) and rows[g]["prompt"] == rows[0]["prompt"]:
        g += 1
    return g


def cluster_stats(steps, only_steps=None):
    """프롬프트 (step, idx // G) 별 통계."""
    out = {}
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)
    for s, rows in steps.items():
        if only_steps and s not in only_steps:
            continue
        G = group_size(rows)
        if len(rows) % G:
            raise SystemExit(f"step {s}: 행 {len(rows)} 가 그룹 크기 {G} 의 배수가 아니다")
        for c in range(len(rows) // G):
            grp = rows[c * G : (c + 1) * G]
            if any(r["prompt"] != grp[0]["prompt"] for r in grp):
                raise SystemExit(f"step {s} 프롬프트 {c}: 그룹 안 프롬프트가 다르다 — 생성 복사본이 인접하다는 가정이 깨졌다")
            first = [k3(r["d"][:FIRST]) for r in grp]
            allp = [k3(r["d"][:ALL]) for r in grp]
            dd = cat([r["d"] for r in grp])
            f0, frest = first[0], cat(first[1:])
            out[(s, c)] = dict(
                prompt=grp[0]["prompt"], plen=grp[0]["plen"], G=G,
                first=float(cat(first).mean()) if cat(first).size else float("nan"),
                all=float(cat(allp).mean()) if cat(allp).size else float("nan"),
                tail=float((np.abs(dd) > 0.5).mean()) if dd.size else float("nan"),
                did=(float(frest.mean()) - float(f0.mean())) if f0.size and frest.size else float("nan"),
            )
    return out


def gate_kl(steps):
    """게이트 출력과 같은 값 — step 별 생성 토큰 k3 평균."""
    return {s: float(np.concatenate([k3(r["d"]) for r in rows]).mean()) for s, rows in steps.items()}


def paired(a, b):
    """프롬프트로 짝지은 B − A. 반환: 통계 이름 → (평균, 표준오차, z, n). 차이가 전부 0 이면 z = 0 (동일)."""
    keys = sorted(set(a) & set(b))
    if len(keys) != len(a) or len(keys) != len(b):
        raise SystemExit(f"프롬프트 수가 다르다: {len(a)} vs {len(b)}")
    bad = [k for k in keys if a[k]["prompt"] != b[k]["prompt"]]
    if bad:
        raise SystemExit(f"프롬프트 토큰이 다른 짝 {len(bad)}/{len(keys)} — 같은 seed · 같은 데이터가 아니다 (예: {bad[:3]})")
    res = {}
    for name in ("first", "all", "tail", "did"):
        x = np.array([b[k][name] - a[k][name] for k in keys], float)
        x = x[~np.isnan(x)]
        n = len(x)
        mean = float(x.mean()) if n else float("nan")
        se = float(x.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
        if n and np.all(x == 0):
            z = 0.0
        else:
            z = mean / se if se and se > 0 else float("nan")
        res[name] = (mean, se, z, n)
    return res


def passes(res, stats=PASS_STATS):
    """통과 = 지정 통계가 모두 유한한 z 이고 z < 2.33. NaN 은 통과가 아니다."""
    return all(math.isfinite(res[k][2]) and res[k][2] < Z_CRIT for k in stats)


def caught(res, stats=PASS_STATS):
    return any(math.isfinite(res[k][2]) and res[k][2] >= Z_CRIT for k in stats)


def fmt_pair(label, res):
    lines = [f"  {label}"]
    for name, title in (("first", "S_first [0,256)"), ("all", "S_all [0,2048)"), ("tail", "S_tail |d|>0.5"), ("did", "(참고) S_did 캐시 복사본 − 첫 복사본")):
        mean, se, z, n = res[name]
        lines.append(f"    {title:34s} 차 {mean:+.6f} · 표준오차 {se:.6f} · z {z:+.2f} · MDE {(Z_CRIT + 1.64) * se:.6f} (n={n})")
    return "\n".join(lines)


def token_identity(a, b, step=1):
    """같은 step 에서 두 런의 생성 토큰이 같은지, 같은 시퀀스 위에서 롤아웃측 · 학습측 logprob 이 얼마나 다른지."""
    ra, rb = a.get(step), b.get(step)
    if not ra or not rb or len(ra) != len(rb):
        return None
    same = [i for i in range(len(ra)) if ra[i]["gen_ids"] == rb[i]["gen_ids"] and ra[i]["prompt"] == rb[i]["prompt"]]
    common = []
    for x, y in zip(ra, rb):
        n = min(len(x["gen_ids"]), len(y["gen_ids"]))
        k = next((j for j in range(n) if x["gen_ids"][j] != y["gen_ids"][j]), n)
        common.append(k)
    out = dict(n=len(ra), identical=len(same), common_mean=float(np.mean(common)) if common else 0.0, common_min=int(min(common)) if common else 0)
    if same:
        dg = np.concatenate([np.abs(ra[i]["g"] - rb[i]["g"]) for i in same])
        dp = np.concatenate([np.abs(ra[i]["p"] - rb[i]["p"]) for i in same])
        ka = np.concatenate([k3(ra[i]["d"]) for i in same]); kb = np.concatenate([k3(rb[i]["d"]) for i in same])
        out.update(tokens=int(dg.size), gen_max=float(dg.max()), gen_diff=float((dg > 0).mean()),
                   prev_gt1e3=float((dp > 1e-3).mean()), prev_gt01=int((dp > 0.1).sum()), prev_max=float(dp.max()),
                   kl_a=float(ka.mean()), kl_b=float(kb.mean()))
    return out


def fmt_identity(label, t):
    if t is None:
        return f"  {label}: 비교 불가 (덤프 없음)"
    s = f"  {label}: step 1 생성 토큰이 같은 시퀀스 {t['identical']}/{t['n']} · 공통 접두 길이 평균 {t['common_mean']:.0f} (최소 {t['common_min']})"
    if "tokens" in t:
        s += (f"\n      같은 시퀀스 {t['tokens']:,} 토큰 위에서: 롤아웃측 logprob 다른 토큰 {100 * t['gen_diff']:.2f}% (최대 {t['gen_max']:.4f}) · "
              f"학습측 |Δ| > 1e-3 {100 * t['prev_gt1e3']:.1f}% · > 0.1 {t['prev_gt01']:,} 개 · 최대 {t['prev_max']:.3f} · k3 평균 {t['kl_a']:.5f} → {t['kl_b']:.5f}")
    return s


def parse_log(path):
    txt = open(path, errors="replace").read()
    txt = re.sub(r"\x1b\[[0-9;]*m", "", txt)
    lines = txt.split("\n")
    pat = r"Generation KL Error: ([0-9]+\.[0-9]+)\s*$"
    kl_idx = [i for i, l in enumerate(lines) if re.search(pat, l)]
    kl = [float(re.search(pat, lines[i]).group(1)) for i in kl_idx]
    hits = []  # (step, pid, rate)
    for i, l in enumerate(lines):
        m = re.search(r"\(VllmGenerationWorker pid=(\d+)[^)]*\).*Prefix cache hit rate: ([0-9.]+)%", l)
        if m:
            hits.append((1 + sum(1 for j in kl_idx if j < i), int(m.group(1)), float(m.group(2))))
    ips = re.findall(r"\(VllmGenerationWorker pid=\d+, ip=(10\.\d+\.\d+\.\d+)\)", txt)
    return dict(
        kl=kl, kl_lines=len([l for l in lines if "Generation KL Error" in l]), hits=hits,
        flags=set(re.findall(r"enable_prefix_caching=(True|False)", txt)),
        flag_lines=len(re.findall(r"enable_prefix_caching=(True|False)", txt)),
        align=len(re.findall(r"Mamba cache mode is set to 'align'", txt)),
        fallback=len(re.findall(r"R3 router replay fallback", txt)),
        fault_patched=len(re.findall(r"\[ALPHA_GATE_FAULT pid=\d+\] patched", txt)),
        fault_fired=len(re.findall(r"\[ALPHA_GATE_FAULT pid=\d+\] FIRED", txt)),
        dedup=len(re.findall(r"\[repeated \d+x across cluster\]", txt)),
        node=(max(set(ips), key=ips.count) if ips else "driver node (main1)"),
    )


def expected_hit_rate(steps, block, engines):
    """엔진별 · step 별 기대 적중률 (검토의 스케줄러 판독: 프롬프트마다 첫 복사본이 계산하고 둘째 복사본의 조회가 한 번 미뤄져 두 번 세어진다).

    프롬프트 하나: 적중 기록 G × H, 조회 (G + 1) × L (H = 길이 L 미만의 가장 큰 블록 배수). 한 블록 덜 적중하면 H 가 block 만큼 준다.
    """
    out = {}
    for s, rows in steps.items():
        G = group_size(rows)
        per = len(rows) // engines
        vals, vals_short = [], []
        for e in range(engines):
            hit = q = hit_short = 0
            for c in range(per // G):
                L = rows[e * per + c * G]["plen"]
                H = (L - 1) // block * block
                hit += G * H
                hit_short += G * max(H - block, 0)
                q += (G + 1) * L
            vals.append(hit / q if q else 0.0)
            vals_short.append(hit_short / q if q else 0.0)
        out[s] = (min(vals), max(vals), max(vals_short))
    return out


def judge_run(gate_dir, name, want_cache, want_r3, fault=None, block=1056, engines=8):
    """런 하나의 유효성. 반환 (ok, 메시지 목록, steps, 로그 요약)."""
    msgs, ok = [], True

    def need(cond, msg):
        nonlocal ok
        msgs.append(("OK  " if cond else "FAIL") + " " + msg)
        ok = ok and bool(cond)

    log_path = os.path.join(gate_dir, f"gate_{name}.log")
    rc_path = os.path.join(gate_dir, f"rc_{name}")
    if not os.path.exists(log_path) or not os.path.exists(rc_path):
        return False, [f"FAIL {name}: 로그 또는 rc 파일 없음 (미실행)"], None, None
    rc = open(rc_path).read().strip()
    L = parse_log(log_path)
    msgs.append(f"     vLLM 워커 노드: {L['node']}")
    need(rc == "0", f"rc={rc}")
    need(len(L["kl"]) == 3 and L["kl_lines"] == 3, f"KL 줄 {L['kl_lines']} 개 중 숫자 {len(L['kl'])} 개 (정확히 3 이어야 한다): {L['kl']}")
    need(L["dedup"] == 0, f"Ray 로그 중복 제거 흔적 {L['dedup']} 줄 (RAY_DEDUP_LOGS=0 이어야 엔진별 적중률이 보인다)")
    steps = load_run(os.path.join(gate_dir, name))
    need(sorted(steps) == [1, 2, 3] and all(len(v) == 128 for v in steps.values()), f"덤프 step {sorted(steps)} · 행 {[len(v) for v in steps.values()]}")
    gk = gate_kl(steps)
    need(len(L["kl"]) == 3 and all(abs(gk[s] - L["kl"][s - 1]) < 6e-5 for s in gk),
         "덤프로 다시 계산한 KL 이 로그와 일치: " + " ".join(f"{gk[s]:.5f}" for s in sorted(gk)))
    if fault is None:
        need(bool(L["kl"]) and all(v < KL_GATE for v in L["kl"]), f"KL < {KL_GATE} (R1 기준): {L['kl']}")
    if want_cache:
        need(L["flags"] == {"True"} and L["flag_lines"] >= engines and L["align"] >= engines,
             f"엔진 {engines} 개가 모두 prefix caching 켬 · align 모드 (flags={sorted(L['flags'])} · 표지 {L['flag_lines']} 줄 · align {L['align']} 줄)")
        exp = expected_hit_rate(steps, block, engines)
        for s in (1, 2, 3):
            by_pid = {}
            for st, pid, rate in L["hits"]:
                if st == s:
                    by_pid[pid] = max(by_pid.get(pid, 0.0), rate)
            lo, hi, short = exp[s]
            thr = 100 * (lo + short) / 2   # 기대 적중과 한 블록 덜 적중한 경우의 중간
            need(len(by_pid) == engines and all(v >= thr for v in by_pid.values()),
                 f"step {s} 적중률: 엔진 {len(by_pid)}/{engines} 개 · 최소 {min(by_pid.values()) if by_pid else float('nan'):.1f}% · 최대 {max(by_pid.values()) if by_pid else float('nan'):.1f}% "
                 f"(기대 {100 * lo:.1f}~{100 * hi:.1f}% · 한 블록 덜 적중하면 {100 * short:.1f}% · 문턱 {thr:.1f}%)")
    else:
        need(L["flags"] == {"False"} and L["flag_lines"] >= engines and L["align"] == 0,
             f"엔진 {engines} 개가 모두 prefix caching 끔 (flags={sorted(L['flags'])} · 표지 {L['flag_lines']} 줄 · align {L['align']} 줄)")
        mx = max([r for _, _, r in L["hits"]], default=0.0)
        need(mx == 0.0, f"적중률 최대 {mx:.1f}% (0 이어야 한다)")
    if fault is None:
        need(L["fault_patched"] == 0 and L["fault_fired"] == 0, f"결함 주입 흔적 없음 (patched {L['fault_patched']} · fired {L['fault_fired']})")
        if want_r3:
            need(L["fallback"] == 0, f"R3 경로 누락 경고 {L['fallback']} 줄 (0 이어야 한다)")
    else:
        need(L["fault_fired"] >= 1, f"결함 주입 `{fault}` 이 실제로 걸림 (patched {L['fault_patched']} · fired {L['fault_fired']}) — 없으면 음성 대조 무효")
    return ok, msgs, steps, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gate_dir", nargs="?")
    ap.add_argument("--pair", nargs=2, metavar=("A", "B"))
    ap.add_argument("--block", type=int, default=1056)
    ap.add_argument("--engines", type=int, default=8)
    ap.add_argument("--mode", choices=("withB", "noB"), help="전체 판정에 필수 — 실행기에 준 것과 같은 값. 필수 런이 빠지면 FAIL 이다")
    ap.add_argument("--json")
    args = ap.parse_args()

    if args.pair:
        a, b = (load_run(d) for d in args.pair)
        na, nb = (os.path.basename(d.rstrip("/")) for d in args.pair)
        print("게이트 KL:", {n: [round(v, 5) for _, v in sorted(gate_kl(s).items())] for n, s in ((na, a), (nb, b))})
        print(fmt_identity(f"{nb} 대 {na}", token_identity(a, b)))
        print(fmt_pair(f"{nb} − {na} (step 1~3)", paired(cluster_stats(a), cluster_stats(b))))
        return 0

    G = args.gate_dir
    if not G or not args.mode:
        ap.error("전체 판정에는 <gate_dir> 와 --mode withB|noB 가 필요하다")
    plan = [  # 이름, 캐시, R3, 결함
        ("LA_off", False, True, None), ("LA_on", True, True, None), ("LA_off2", False, True, None),
        ("LB_off", False, False, None), ("LB_on", True, False, None),
        ("NC_zero", True, True, "state_zero"), ("NC_trunc", True, True, "routes_trunc:8"), ("NC_shift", True, True, "routes_shift:3168"),
        ("NC_state", True, True, "state_scale:1.01"),
    ]
    raw, valid, logs = {}, {}, {}
    print(f"== R1-L 판정: {G} ==")
    for name, cache, r3, fault in plan:
        ok, msgs, steps, L = judge_run(G, name, cache, r3, fault, args.block, args.engines)
        valid[name], logs[name] = ok, L
        print(f"[{name}] 유효성 {'통과' if ok else '실패'}")
        for m in msgs:
            print("   ", m)
        if steps:
            raw[name] = steps

    verdict, report = {}, {}

    def compare(label, a, b, only_steps=None):
        if a not in raw or b not in raw:
            print(f"  {label}: 미실행 ({a} · {b})")
            return None
        res = paired(cluster_stats(raw[a], only_steps), cluster_stats(raw[b], only_steps))
        print(fmt_pair(label, res))
        report[label] = {k: dict(zip(("mean", "se", "z", "n"), v)) for k, v in res.items()}
        return res

    def identity(label, a, b, enforce=False):
        """enforce=True 면 step 1 생성 토큰이 전부 같아야 한다 — 다르면 b 런을 무효로 한다 (엔진 계산이 달라졌거나 하네스가 결정적이지 않다)."""
        t = token_identity(raw[a], raw[b]) if a in raw and b in raw else None
        print(fmt_identity(label, t))
        report["identity " + label] = t
        if enforce and t is not None and t["identical"] != t["n"]:
            valid[b] = False
            print(f"    FAIL {b}: step 1 토큰이 {a} 와 같아야 하는데 {t['identical']}/{t['n']} — {b} 무효")
        return t

    print("\n== 같은 조건 반복 (A/A) — 보정 확인 ==")
    identity("LA_off2 대 LA_off (같은 설정 — step 1 토큰이 전부 같아야 한다)", "LA_off", "LA_off2", enforce=True)
    aa = compare("LA_off2 − LA_off, step 2·3 (세 통계 모두 |z| < 2.33 이어야 한다)", "LA_off", "LA_off2", only_steps=(2, 3))
    if aa is not None:
        verdict["A/A 보정"] = valid["LA_off"] and valid["LA_off2"] and all(math.isfinite(aa[k][2]) and abs(aa[k][2]) < Z_CRIT for k in PASS_STATS)

    print("\n== 캐시 켬 − 끔 (프롬프트 단위 짝 비교, step 1~3) ==")
    for cfg, off, on in (("A (R3 유지 + 캐시 켬)", "LA_off", "LA_on"), ("B (R3 끔 · 라우터 fp32 + 캐시 켬)", "LB_off", "LB_on")):
        identity(f"{on} 대 {off}", off, on)
        res = compare(f"{cfg}: {on} − {off}", off, on)
        if res is not None:
            verdict[cfg] = valid[off] and valid[on] and passes(res)
    compare("참고 — R3 를 빼고 라우터를 fp32 로 (캐시 끔): LB_off − LA_off", "LA_off", "LB_off")
    compare("참고 — 같은 비교를 캐시 켠 채로: LB_on − LA_on", "LA_on", "LB_on")

    print("\n== 음성 대조 (일부러 넣은 결함 — 여기서는 잡혀야 게이트가 그 결함을 본다; 민감도는 LA_on 대비) ==")
    for name, what, must in (("NC_zero", "복원 상태를 0 으로", True), ("NC_shift", "접두사 3,168 토큰의 expert 번호 +1", True), ("NC_state", "복원 상태 × 1.01 — 검출 한계 참고", False)):
        if name == "NC_shift":
            identity("NC_shift 대 LA_on (엔진 계산은 같아야 한다)", "LA_on", "NC_shift", enforce=True)
        res = compare(f"{name} ({what}) − LA_on", "LA_on", name)
        compare(f"   (참고) {name} − LA_off", "LA_off", name)
        if res is not None:
            hit = valid[name] and caught(res)
            print(f"    → {'잡힘' if hit else '못 잡음'}")
            if must:
                verdict[f"음성 대조 {name} 가 잡힘"] = hit
    if logs.get("NC_trunc"):
        identity("NC_trunc 대 LA_on (엔진 계산은 같아야 한다)", "LA_on", "NC_trunc", enforce=True)
        hit = valid["NC_trunc"] and logs["NC_trunc"]["fallback"] >= 1
        print(f"  NC_trunc (경로 행 끝 8 개 누락): R3 경로 누락 경고 {logs['NC_trunc']['fallback']} 줄 → {'잡힘' if hit else '못 잡음'}")
        verdict["음성 대조 NC_trunc 가 잡힘"] = hit

    required = ["A/A 보정", "A (R3 유지 + 캐시 켬)", "음성 대조 NC_zero 가 잡힘", "음성 대조 NC_shift 가 잡힘", "음성 대조 NC_trunc 가 잡힘"]
    if args.mode == "withB":
        required.insert(2, "B (R3 끔 · 라우터 fp32 + 캐시 켬)")
    print(f"\n== 판정 (mode={args.mode}; 필수 항목이 빠지면 FAIL) ==")
    final = {}
    for k in required:
        final[k] = bool(verdict.get(k, False))
        print(f"  {'PASS' if final[k] else 'FAIL'}  {k}{'' if k in verdict else ' — 미실행 · 판정 불가'}")
    for k, v in verdict.items():
        if k not in final:
            print(f"  (참고) {'PASS' if v else 'FAIL'}  {k}")
    print("  PASS 의 뜻: 캐시 켬이 위 MDE 보다 큰 차이를 내지 않는다. 음성 대조로 확인한 검출력은 '복원 상태 0' · 'expert 번호 밀기' 수준이고, × 1.01 결과 (NC_state) 는 위 참고 줄을 본다.")
    if args.json:
        json.dump(dict(mode=args.mode, valid=valid, verdict=final, all=verdict, report=report), open(args.json, "w"), ensure_ascii=False, indent=1, default=str)
    return 0 if all(final.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
