#!/usr/bin/env python3
"""SWE 학생 롤아웃 측정 (runs/eval_swe_student.sh) 요약 — docs/E2E_SANDBOX.md §7.7 의 판정 기준과 측정값.

  summarize_swe_student.py <tag> [--cap 8192] [--json out.json]        ($NRL_ROOT/p0_measure/swe/<tag>/)

읽는 것 (2026-11-25 student_s2a2 의 실제 행에서 확인한 필드):
  rollouts.jsonl 행 — reward · patch_exists · model_patch · agent_error_kind · agent_timed_out · eval_timed_out · oom_killed · eval_oom_killed ·
    instance_config.mask_sample · instance_config.persistent_dir · openhands_run_time · total_model_call_time · final_eval_time ·
    agent_peak_rss_mb · eval_peak_rss_mb · per_turn_metrics.token_usages[].completion_tokens · response.output[] (reasoning · function_call ·
    function_call_output · message)
  rollouts_failures.jsonl — 결과를 못 낸 요청 (agent_run_error)
  env_start.log — 워커 줄의 `ip=<sub1 주소>` 로 롤아웃이 돈 노드 (결과 행에는 없다)
  과제 디렉토리의 apptainer_logs/*_eval.log · eval_private/test_patch.diff — 패치 적용 거부
없는 값은 0 으로 세지 않고 결측으로 센다."""
import argparse, collections, glob, json, math, os, re, sys

ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--cap", type=int, default=8192); ap.add_argument("--json")
a = ap.parse_args()
D = f"/home/work/vidsearch/tools/nemo_rl/p0_measure/swe/{a.tag}/"
rows = [json.loads(l) for l in open(D + "rollouts.jsonl")]
fails = [json.loads(l) for l in open(D + "rollouts_failures.jsonl")] if os.path.exists(D + "rollouts_failures.jsonl") else []
env = open(D + "env_start.log", errors="replace").read() if os.path.exists(D + "env_start.log") else ""
run_log = open(D + "run.log", errors="replace").read() if os.path.exists(D + "run.log") else ""
EDIT_CMDS = {"create", "str_replace", "insert"}

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d
def pct(k, n): return f"{k}/{n} ({100 * k / n:.1f}%)" if n else "0/0"
def files_of(patch): return {m.group(1) for m in re.finditer(r"^diff --git a/(\S+) b/", patch or "", flags=re.M)}
def quant(v, q): v = sorted(v); return v[min(len(v) - 1, int(len(v) * q))] if v else None

recs = []
for r in rows:
    m = r["responses_create_params"]["metadata"]; ic = r.get("instance_config") or {}; pd = ic.get("persistent_dir") or ""
    d = m.get("instance_dict"); d = json.loads(d) if isinstance(d, str) else (d or {})
    out = (r.get("response") or {}).get("output") or []
    calls = [o for o in out if o.get("type") == "function_call"]; outs = {o.get("call_id"): o for o in out if o.get("type") == "function_call_output"}
    msgs = [o for o in out if o.get("type") == "message"]
    executed = [c for c in calls if c.get("call_id") in outs and c.get("name") in ("execute_bash", "str_replace_editor")]
    edited = set()
    for c in calls:
        if c.get("name") != "str_replace_editor" or c.get("call_id") not in outs: continue
        try: args = json.loads(c.get("arguments") or "{}")
        except ValueError: continue
        if args.get("command") in EDIT_CMDS and not str(outs[c["call_id"]].get("output") or "").lstrip().startswith("ERROR"):
            edited.add(str(args.get("path") or ""))
    def text(o):
        c = o.get("content"); return c if isinstance(c, str) else " ".join(str(p.get("text", "")) for p in (c or []) if isinstance(p, dict))
    markup = sum(bool(re.search(r"<tool_call>|</tool_call>|<function=", text(o))) for o in msgs)
    usages = ((r.get("per_turn_metrics") or {}).get("token_usages")) if isinstance(r.get("per_turn_metrics"), dict) else None
    comp = [u.get("completion_tokens") for u in usages if isinstance(u, dict) and u.get("completion_tokens") is not None] if usages else None
    base = os.path.basename(pd)
    node = None
    if base and base in env: node = "sub1" if re.search(r"ip=[0-9.]+\)[^\n]*" + re.escape(base), env) else "main1"
    # 패치 적용 거부: 채점 로그의 거부 줄을 경로로 모델 패치 · 테스트 패치에 귀속
    rej_model = rej_test = None; pre_existing = []
    evlog = glob.glob(pd + "/apptainer_logs/*_eval.log") if pd else []
    if evlog and r.get("patch_exists"):
        t = open(evlog[0], errors="replace").read()
        bad = set(re.findall(r"error: patch failed: (\S+?):\d+", t)) | set(re.findall(r"error: (\S+): (?:patch does not apply|No such file or directory)", t)) \
            | set(re.findall(r"Applying patch (\S+) with \d+ rejects?", t))
        # 이미지에 원래 있던 추적 안 된 파일 (예: package-lock.json) 은 학생 패치에 "새 파일" 로 실리고, 채점 컨테이너에는 이미 있어
        # `already exists in working directory` 로 그 파일만 거부된다 — 나머지는 적용된다 (2026-11-25 S2b: 12 건 전부 이 경우, 그중 2 건 보상 1). 따로 센다.
        pre = set(re.findall(r"error: (\S+): already exists in working directory", t))
        mp = files_of(r.get("model_patch")); tp = set()
        tpf = os.path.join(pd, "eval_private/test_patch.diff")
        if os.path.exists(tpf): tp = files_of(open(tpf, errors="replace").read())
        rej_model = sorted(f for f in bad if f in mp and f not in tp); rej_test = sorted(f for f in bad if f in tp)
        pre_existing = sorted(f for f in pre if f in mp)
    repo = "/" + (m.get("repo") or "/").split("/")[-1] + "/"
    edited_rel = {p[len(repo):] for p in edited if p.startswith(repo)}
    recs.append(dict(iid=m["instance_id"], lang=d.get("language") or "?", reward=r.get("reward"), patch=bool(r.get("patch_exists")), mask=bool(ic.get("mask_sample")),
                     err=r.get("agent_error_kind"), agent_to=bool(r.get("agent_timed_out")), eval_to=bool(r.get("eval_timed_out")), oom=bool(r.get("oom_killed") or r.get("eval_oom_killed")),
                     n_model=sum(o.get("type") in ("reasoning", "function_call", "message") for o in out), n_calls=len(calls), n_exec=len(executed), markup=markup,
                     max_comp=max(comp) if comp else None, n_turns=len(comp) if comp is not None else None, trunc=(max(comp) >= a.cap) if comp else None,
                     run_s=r.get("openhands_run_time"), model_s=r.get("total_model_call_time"), eval_s=r.get("final_eval_time"),
                     rss=r.get("agent_peak_rss_mb"), erss=r.get("eval_peak_rss_mb"), node=node, rej_model=rej_model, rej_test=rej_test, pre_existing=pre_existing,
                     edited_missing=sorted(edited_rel - files_of(r.get("model_patch"))) if edited_rel else [], edited_n=len(edited_rel), patch_files=len(files_of(r.get("model_patch")))))

N = len(recs); exp = None
mm = re.search(r"expected=(\d+)", run_log); exp = int(mm.group(1)) if mm else None
print(f"== {a.tag}: rollouts {N} · failures {len(fails)} · expected {exp} ==")
crit = {}
crit["1 결과 수"] = (exp is not None and N + len(fails) == exp and (len(fails) / exp <= 0.02 if exp else False), f"rollouts {N} + failures {len(fails)} = {N + len(fails)} / expected {exp} · agent_run_error {pct(len(fails), exp or 0)}")
idok = "모델 경로 일치 백엔드 8/8 · 프록시 16/16" in run_log
crit["2 붙은 대상"] = (idok, "fleet 8/8 · 프록시 16/16 (run.log)" if idok else "run.log 에 신원 확인 줄이 없다")
reached = sum(x["n_model"] >= 1 for x in recs); nodes = collections.Counter(x["node"] for x in recs)
sub_share = nodes.get("sub1", 0) / N if N else 0
crit["3 모델 도달"] = (N > 0 and reached / N >= 0.95 and sub_share >= 0.30, f"모델 응답이 있는 롤아웃 {pct(reached, N)} · 노드 {dict(nodes)} (sub1 {sub_share:.0%})")
execd = sum(x["n_exec"] >= 1 for x in recs); mk = sum(x["markup"] > 0 for x in recs)
crit["4 도구 실행"] = (N > 0 and execd / N >= 0.90, f"실행된 명령 · 편집이 1 개 이상 {pct(execd, N)} · 도구 호출 표기가 글에 남은 롤아웃 {pct(mk, N)} (파싱 실패 흔적)")
withp = [x for x in recs if x["patch"]]; rejm = [x for x in withp if x["rej_model"]]; unk = [x for x in withp if x["rej_model"] is None]
ed_empty = [x for x in recs if x["edited_n"] > 0 and not x["patch"] and not x["mask"]]
crit["5 패치 전달 (①③)"] = (not rejm and not unk, f"패치 있는 롤아웃 {len(withp)} 중 모델 패치 거부 {len(rejm)} · 채점 로그 없음 {len(unk)} · 이미지에 원래 있던 파일이라 건너뛴 것 {sum(bool(x['pre_existing']) for x in withp)} (무해) · 테스트 패치 쪽 거부 {sum(bool(x['rej_test']) for x in withp)} (참고) · 편집했는데 패치가 빈 비마스킹 롤아웃 {len(ed_empty)} (되돌림 · 임시 파일 여부는 궤적에서 확인)")
left = re.search(r"잔재: (.*)", run_log); lt = left.group(1) if left else ""
clean = bool(re.search(r"containers main1 \[SWE_CONTAINERS sessions=\d+ left=0\] sub1 \[SWE_CONTAINERS sessions=\d+ left=0\]", lt)) and "gym_servers_left=0" in lt and "busy_gpus=0" in lt \
    and (lambda mo: bool(mo) and mo.group(1) == mo.group(2))(re.search(r"fuse_orphans before=(\S+) after=(\S+)", lt))
crit["6 잔재"] = (clean, lt[:160] or "run.log 에 잔재 줄이 없다")
for k, (ok, why) in crit.items(): print(("PASS  " if ok else "FAIL  ") + k + " — " + why)

unm = [x for x in recs if not x["mask"]]; solved = sum(x["reward"] == 1.0 for x in recs)
lo, hi = wilson(solved, N); print(f"\n통과율 (전체) {pct(solved, N)} · 95% 구간 {lo:.1%} ~ {hi:.1%} | 마스킹 제외 {pct(sum(x['reward'] == 1.0 for x in unm), len(unm))}")
print("종료 사유:", dict(collections.Counter(("mask:" + str(x["err"] or ("agent_timeout" if x["agent_to"] else "eval_timeout" if x["eval_to"] else "oom" if x["oom"] else "?")) if x["mask"] else ("err:" + str(x["err"]) if x["err"] else ("patch" if x["patch"] else "no_patch"))) for x in recs)))
tr = [x for x in recs if x["trunc"] is not None]
print(f"턴당 {a.cap} 잘림이 있는 롤아웃 {pct(sum(x['trunc'] for x in tr), len(tr))} (결측 {N - len(tr)}) — 학습에서는 overlong_filtering 으로 loss 에서 빠진다")
for name, key, unit in (("모델 호출 수", "n_turns", ""), ("롤아웃 시간", "run_s", "s"), ("모델 호출 시간", "model_s", "s"), ("채점 시간", "eval_s", "s"), ("에이전트 최대 메모리", "rss", "MB"), ("채점 최대 메모리", "erss", "MB")):
    v = [x[key] for x in recs if x[key] is not None]
    if v: print(f"{name}: 중앙 {quant(v, .5):.0f}{unit} · p90 {quant(v, .9):.0f}{unit} · 최대 {max(v):.0f}{unit} (n={len(v)})")
by = collections.defaultdict(lambda: [0, 0])
for x in recs: by[x["lang"]][0] += x["reward"] == 1.0; by[x["lang"]][1] += 1
print("언어별 (참고):", {k: f"{v[0]}/{v[1]}" for k, v in sorted(by.items(), key=lambda kv: -kv[1][1])})
ns = D + "node_samples.tsv"
if os.path.exists(ns):
    tot = []; shm = []; load = []; used = []
    for l in open(ns).read().split("\n")[1:]:
        p = l.split("\t")
        for part in p[1:3]:
            f = part.split()
            if len(f) == 3 and f[0] != "?": shm.append(float(f[0])); tot.append(float(f[1])); load.append(float(f[2]))            # 2026-11-25 11:40 이전 형식
            if len(f) == 4 and f[0] != "?": shm.append(float(f[0])); used.append(float(f[0]) + float(f[1])); tot.append(float(f[2])); load.append(float(f[3]))
    if tot: print(f"노드 (15 초 표본): anon + shmem 최대 {max(used):.0f} GiB · " if used else "노드 (15 초 표본): ", end="")
    if tot: print(f"전체 (페이지 캐시 포함) 최대 {max(tot):.0f} GiB · tmpfs (shmem) 최대 {max(shm):.1f} GiB · load 최대 {max(load):.0f}")

# 과제별 보상 갈림 (반복이 2 회 이상일 때): q = 보상이 전부 같지 않고 학습에 쓰이는 (마스킹 · 잘림이 아닌) 롤아웃이 하나 이상 있는 과제 / 전체 과제
g = collections.defaultdict(list)
for x in recs: g[x["iid"]].append(x)
reps = collections.Counter(len(v) for v in g.values())
if max(reps) > 1:
    T = len(g); mixed = [i for i, v in g.items() if len({x["reward"] for x in v}) > 1 and any(not x["mask"] and not x["trunc"] for x in v)]
    lo, hi = wilson(len(mixed), T)
    print(f"\n과제 {T} 개 · 반복 수 분포 {dict(reps)} · 과제별 통과 수 분포 {dict(sorted(collections.Counter(sum(x['reward'] == 1.0 for x in v) for v in g.values()).items()))}")
    print(f"q (보상이 갈린 과제) = {len(mixed)}/{T} ({100 * len(mixed) / T:.1f}%) · 95% 구간 {lo:.1%} ~ {hi:.1%} · 한 step (프롬프트 8) 이 전부 무신호일 확률 {(1 - len(mixed) / T) ** 8:.0%}")
    print("판정 (§7.7, 과제 64 기준):", "진행 (10 개 이상)" if len(mixed) >= 10 else "중단 보고 (3 개 이하)" if len(mixed) <= 3 else "선택지 보고 (4 ~ 9 개)")
    print(f"과제별 비마스킹 · 비잘림 롤아웃 수 분포: {dict(sorted(collections.Counter(sum(not x['mask'] and not x['trunc'] for x in v) for v in g.values()).items()))}")
for x in recs:
    if x["rej_model"] or x["edited_missing"]: print(f"  검토: {x['iid']} | 모델 패치 거부 {x['rej_model']} | 편집했는데 패치에 없는 파일 {x['edited_missing'][:4]} (되돌렸거나 지웠을 수 있다) | reward {x['reward']}")
if a.json: json.dump(dict(tag=a.tag, n=N, failures=len(fails), criteria={k: v[0] for k, v in crit.items()}, records=recs), open(a.json, "w"), ensure_ascii=False)
sys.exit(0 if all(v[0] for v in crit.values()) else 1)
