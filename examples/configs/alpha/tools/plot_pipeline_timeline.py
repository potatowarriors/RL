"""비동기 GRPO 파이프라인 타임라인 — 학습기 step 구간(TensorBoard timing/*) 대 롤아웃 물결(rollout_groups.jsonl) 대 GPU util 표본.

사용: python examples/configs/alpha/tools/plot_pipeline_timeline.py <run_dir> "<YYYY-MM-DD HH:MM>" <out.png> [gpu_csv]
  run_dir: results/alpha/<campaign>/<tag> (TensorBoard events 와 rollout_groups.jsonl 이 있는 곳) · 시작 시각 이후를 그린다.
  gpu_csv: time,step,mean_util,... (examples/configs/alpha/docs/study/s5_pipeline_timeline.md 의 표본 형식), 없으면 생략.
읽는 법과 선례: docs/study/s5_pipeline_timeline.md (2026-10-22).
"""
import glob, json, time, collections, subprocess, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from tensorboard.backend.event_processing import event_accumulator
RUN = sys.argv[1].rstrip("/"); T0 = time.mktime(time.strptime(sys.argv[2], "%Y-%m-%d %H:%M")); OUT = sys.argv[3]
GPU_CSV = sys.argv[4] if len(sys.argv) > 4 else None; T1 = time.time()
ko = [f for f in font_manager.findSystemFonts() if any(k in f.lower() for k in ("nanum", "notosanscjk", "notosanskr", "malgun", "unfonts", "baekmuk"))]
if ko: plt.rcParams["font.family"] = font_manager.FontProperties(fname=ko[0]).get_name()
K = bool(ko)
L = (lambda a, b: a if K else b)
# ---- trainer timings ----
tm = {}
for f in sorted(glob.glob(f"{RUN}/**/events.out.tfevents*", recursive=True)):
    ea = event_accumulator.EventAccumulator(f, size_guidance={"scalars": 0}); ea.Reload()
    for t in ea.Tags()["scalars"]:
        if t.startswith("timing/"):
            for e in ea.Scalars(t): tm.setdefault(t, {})[e.step] = (e.value, e.wall_time)
g = lambda k, s: tm.get(k, {}).get(s, (0, None))[0]
steps = [s for s in sorted(tm["timing/train/total_step_time"]) if tm["timing/train/total_step_time"][s][1] >= T0 - 3*3600]
seg = []  # (start, end, kind, label)
sync_end = {}
for s in steps:
    end = tm["timing/train/total_step_time"][s][1]; total = g("timing/train/total_step_time", s)
    ck = g("timing/train/checkpointing", s); va = g("timing/validation/total_validation_time", s)
    ws = g("timing/train/weight_sync", s); tr = g("timing/train/policy_training", s)
    prep = sum(g(k, s) for k in ("timing/train/reward_calculation", "timing/train/data_processing", "timing/train/policy_and_reference_logprobs", "timing/train/advantage_calculation", "timing/train/training_prep"))
    wait = g("timing/train/exposed_generation", s)
    t = end
    for dur, kind in ((ck, "ckpt"), (va, "val"), (ws, "sync"), (tr, "train"), (prep, "prep")):
        if dur > 0: seg.append((t - dur, t, kind, s)); t -= dur
    start = end - total
    seg.append((start, min(start + wait, t), "wait", s))
    if start + wait < t: seg.append((start + wait, t, "other", s))
    sync_end[s] = end - ck - va
# ---- rollout waves ----
recs = [json.loads(l) for l in open(f"{RUN}/rollout_groups.jsonl")]
by = collections.defaultdict(list)
for r in recs: by[r["target_weight_version"]].append(r)
waves = []
for tgt in sorted(by):
    rs = sorted(by[tgt], key=lambda r: r["time"]); kept = [r for r in rs if r.get("dropped") is None]
    first, last = rs[0]["time"], rs[-1]["time"]
    if last < T0 - 3600: continue
    k128 = kept[127]["time"] if len(kept) >= 128 else None
    launch = sync_end.get(tgt - 1)  # target t is launched after the refit at the end of display step t-1
    waves.append((tgt, launch, first, k128, last, len(kept)))
bins = collections.Counter(int((r["time"] - T0) // 300) for r in recs if r["time"] >= T0)
# ---- GPU sampler ----
gpu = []
try:
    for line in (open(GPU_CSV).read().splitlines()[1:] if GPU_CSV else []):
        p = line.split(","); gpu.append((time.mktime(time.strptime(p[0], "%Y-%m-%d %H:%M:%S")), float(p[2])))
except Exception: pass
# ---- draw ----
H = lambda t: (t - T0) / 3600
col = {"wait": "#bbbbbb", "prep": "#ffd92f", "train": "#d62728", "sync": "#ff7f0e", "val": "#9467bd", "ckpt": "#8c564b", "other": "#dddddd",
       "ramp": "#aec7e8", "gen": "#1f77b4", "tail": "#08306b", "idle": "#f0f0f0"}
fig, ax = plt.subplots(figsize=(22, 9))
# row 0: trainer
y_tr = 7.0
for a, b, kind, s in seg:
    ax.broken_barh([(H(a), H(b) - H(a))], (y_tr, 0.8), facecolors=col[kind], edgecolors="none")
for s in steps:
    end = tm["timing/train/total_step_time"][s][1]; start = end - g("timing/train/total_step_time", s)
    ax.text(max(H(start), 0) + 0.02, y_tr + 0.9, f"step {s}", fontsize=8, va="bottom"); ax.axvline(H(end), color="k", lw=0.5, ls="--", alpha=0.5)
# rows 1-2: waves (alternate rows by target parity)
for tgt, launch, first, k128, last, nk in waves:
    y = 5.0 if tgt % 2 == 0 else 3.8
    if launch and launch < first: ax.broken_barh([(H(launch), H(first) - H(launch))], (y, 0.8), facecolors=col["ramp"], edgecolors="white", hatch="..")
    if k128: ax.broken_barh([(H(first), H(k128) - H(first))], (y, 0.8), facecolors=col["gen"], edgecolors="none"); ax.broken_barh([(H(k128), H(last) - H(k128))], (y, 0.8), facecolors=col["tail"], edgecolors="none")
    else: ax.broken_barh([(H(first), H(last) - H(first))], (y, 0.8), facecolors=col["gen"], edgecolors="none", hatch="//")
    ax.text(max(H(launch if launch else first), 0) + 0.02, y + 0.9, f"target {tgt} (kept {nk})", fontsize=8, va="bottom")
# row 3: validation generation (same engines)
for a, b, kind, s in seg:
    if kind == "val": ax.broken_barh([(H(a), H(b) - H(a))], (2.8, 0.6), facecolors=col["val"], edgecolors="none")
# row 4: completions per 5 min (engine output proxy)
mx = max(bins.values()) if bins else 1
for b, n in bins.items():
    ax.broken_barh([(b * 5 / 60, 5 / 60)], (1.6, 0.8), facecolors=plt.cm.Blues(0.15 + 0.85 * n / mx), edgecolors="none")
# gaps ≥ 10 min with no completions
ts = sorted(r["time"] for r in recs if r["time"] >= T0)
for i in range(len(ts) - 1):
    if ts[i + 1] - ts[i] >= 600:
        ax.broken_barh([(H(ts[i]), H(ts[i + 1]) - H(ts[i]))], (1.6, 0.8), facecolors="none", edgecolors="red", lw=1.2)
        ax.text(H(ts[i]) + 0.01, 1.2, f"{int((ts[i+1]-ts[i])/60)} min", color="red", fontsize=8)
# row 5: GPU util measured
if len(gpu) > 3:
    xs = [H(t) for t, u in gpu]; ys = [0.1 + 0.8 * u / 100 for t, u in gpu]
    ax.plot(xs, ys, color="green", lw=1.5); ax.axhline(0.1, color="green", lw=0.3); ax.axhline(0.9, color="green", lw=0.3)
labels = {y_tr + 0.4: L("sub1 학습기 (step 단계)", "sub1 trainer (step phases)"), 5.4: L("main1 롤아웃 물결 (짝수 target)", "main1 rollout wave (even target)"),
          4.2: L("main1 롤아웃 물결 (홀수 target)", "main1 rollout wave (odd target)"), 3.1: L("main1 검증 생성", "main1 validation generation"),
          2.0: L("5분당 완료 그룹 수 (진할수록 많음)", "groups completed per 5 min (darker = more)"), 0.5: L("main1 GPU util 실측 (0~100%)", "main1 GPU util measured (0-100%)")}
ax.set_yticks(list(labels)); ax.set_yticklabels(list(labels.values()), fontsize=9)
ticks = range(0, int(H(T1)) + 2); ax.set_xticks(ticks); ax.set_xticklabels([time.strftime("%H:%M", time.localtime(T0 + h * 3600)) for h in ticks])
ax.set_xlim(0, H(T1) + 0.05); ax.set_ylim(0, 8.6); ax.grid(axis="x", alpha=0.3)
from matplotlib.patches import Patch
leg = [Patch(color=col[k], label=v) for k, v in (("wait", L("대기 (배치 기다림)", "wait for batch")), ("prep", L("준비 (보상·logprob·advantage)", "prep")), ("train", L("학습 (optimizer step)", "policy training")),
       ("sync", L("refit (가중치 전송)", "weight sync/refit")), ("val", L("검증", "validation")), ("ckpt", L("저장", "checkpoint")),
       ("ramp", L("물결 기동(refit 시각으로 추정) → 첫 완료", "wave launch (est. = refit end) → first completion")), ("gen", L("완료 진행 (→128번째 kept)", "completions → 128th kept")), ("tail", L("꼬리 (128번째 뒤)", "tail after 128th kept")))]
ax.legend(handles=leg, loc="upper left", bbox_to_anchor=(0, -0.06), ncol=5, fontsize=9, frameon=False)
ax.set_title(L(f"{RUN.split('/')[-2]}/{RUN.split('/')[-1]} 파이프라인 타임라인 {sys.argv[2][:10]} — 학습기 구간 대 롤아웃 물결", f"{RUN.split('/')[-2]}/{RUN.split('/')[-1]} pipeline timeline {sys.argv[2][:10]} — trainer phases vs rollout waves"), fontsize=11)
plt.tight_layout(); plt.savefig(OUT, dpi=110); print("saved", OUT, "font_ko", K, "steps", steps, "waves", [(w[0], w[5]) for w in waves], "gpu_samples", len(gpu))
