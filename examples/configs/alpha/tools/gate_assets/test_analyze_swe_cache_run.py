#!/usr/bin/env python3
"""CPU test of `analyze_swe_cache_run.py` on synthetic runs: a healthy run must come out healthy (exit 0) and every
broken or unverifiable one must not (exit 2 = invalid or broken, 3 = not judged), for the stated reason.

    python3 -I tools/gate_assets/test_analyze_swe_cache_run.py

The cases follow the adversarial reviews of 2026-11-25 (silent /metrics, collector ending early, restarts, vanished or
idle engines, stalls with only waiting requests, zero hits, hit rate below or above the run's own prediction).
"""
import datetime
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "analyze_swe_cache_run.py"
ROOT = Path(tempfile.mkdtemp(prefix="swe_analyze_test_"))
T0, DT, NS, PORTS = 1795611156, 62, 40, list(range(8000, 8008))
NAMES = ["vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc", "vllm:num_preemptions_total",
         "vllm:prompt_tokens_total", "vllm:prompt_tokens_cached_total", "vllm:generation_tokens_total",
         "vllm:e2e_request_latency_seconds_count", "vllm:e2e_request_latency_seconds_sum"]


def rollouts(n=8, records=8):
    """n rollouts of 60 calls whose prompt grows by 400 tokens a call; only the first `records` keep per-turn records."""
    rows, total = [], 0
    for i in range(n):
        tu = [{"prompt_tokens": 9000 + 37 * i + 400 * k, "completion_tokens": 150} for k in range(60)]
        total += sum(u["prompt_tokens"] for u in tu)
        out = []
        for _ in range(60):
            out += [{"type": "reasoning", "summary": [{"text": "think"}]}, {"type": "function_call", "name": "execute_bash", "call_id": "c"},
                    {"type": "function_call_output", "call_id": "c", "output": "ok"}]
        rows.append(dict(responses_create_params=dict(metadata=dict(instance_id=f"task-{i}")), instance_config=dict(resolved_agent_cls="CodeActAgent"),
                         reward=0.0, agent_timed_out=False, openhands_run_time=600.0, total_model_call_time=60.0, total_command_exec_time=100.0,
                         connect_to_runtime_time=5.0, generation_start_timestamp="2026-11-25T02:52:01+00:00", response=dict(output=out),
                         per_turn_metrics=dict(token_usages=tu if i < records else [], response_latencies=[{"latency": 1.0}] * 60)))
    return rows, total


def predicted(rows):
    fl = lambda x: max(x, 0) // 1056 * 1056
    tot = lo = hi = 0
    for r in rows:
        prev = None
        for u in r["per_turn_metrics"]["token_usages"]:
            L, G = u["prompt_tokens"], u["completion_tokens"]
            tot += L
            if prev:
                lo += fl(min(prev[0], L - 1))
                hi += fl(min(prev[0] + prev[1] - 1, L - 1))
            prev = (L, G)
    return lo / tot, hi / tot


def write(tag, sample, hit=None, cache="1", run_log=True, t0=T0, t1=T0 + (NS - 1) * DT, records=8):
    """sample(port, k) -> fraction of the run's work done by sample k for that port (None = no sample), or a dict override."""
    d = ROOT / tag
    d.mkdir()
    rows, total = rollouts(records=records)
    lo, hi = predicted(rows)
    hit = (lo + hi) / 2 if hit is None else hit
    (d / "rollouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with (d / "fleet_metrics.tsv").open("w") as f:
        for k in range(NS):
            for p in PORTS:
                s = sample(p, k)
                if s is None:
                    continue
                o = s if isinstance(s, dict) else {}
                frac = o.get("frac", s if not isinstance(s, dict) else k / (NS - 1))
                h = o.get("hit", hit(p) if callable(hit) else hit)
                vals = {"vllm:num_requests_running": o.get("running", 2), "vllm:num_requests_waiting": o.get("waiting", 0),
                        "vllm:prompt_tokens_total": total / 8 * frac, "vllm:prompt_tokens_cached_total": h * total / 8 * frac,
                        "vllm:generation_tokens_total": 9000 * 60 * frac, "vllm:e2e_request_latency_seconds_count": 100 * frac,
                        "vllm:e2e_request_latency_seconds_sum": 100 * frac}
                for n in NAMES:
                    f.write(f"{T0 + k * DT}\t{p}\t{n}\t{float(vals.get(n, 0.0))}\n")
    if run_log:
        st = lambda t: datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")
        (d / "run.log").write_text(f"[{st(t0 - 200)}] armed: tag={tag} prefix_cache={cache} prompt_mix=codeact conc=12\n"
                                   f"[{st(t0)}] gym servers ready (3)\n[{st(t1)}] eval rc=0 rollouts=8 failures=0 expected=8 seconds={int(t1 - t0)}\n")
    return lo, hi


lin = lambda p, k: k / (NS - 1)
CASES = []  # (tag, expected exit code, phrase the output must contain)


def case(tag, rc, phrase, *a, **kw):
    write(tag, *a, **kw)
    CASES.append((tag, rc, phrase))


case("healthy", 0, "적중률 판정 (세션 고정 · 접두사 성질): 통과", lin)
case("one_missed_sample", 0, "표본 누락 구간 1 개", lambda p, k: None if (p == 8003 and k == 10) else lin(p, k))
case("metrics_silent_engine_working", 0, "표본 누락 구간 1 개", lambda p, k: None if (p == 8003 and 10 <= k <= 13) else lin(p, k))
case("metrics_silent_engine_stalled", 2, "8003: 정지 의심", lambda p, k: None if (p == 8003 and 10 <= k <= 13) else (lin(p, 10 if 10 < k <= 14 else k) if p == 8003 else lin(p, k)))
case("port_silent_tail", 2, "8003: 정지 의심", lambda p, k: None if (p == 8003 and k >= 25) else lin(p, k))
case("port_never_seen", 2, "지표가 없는 엔진 [8003]", lambda p, k: None if p == 8003 else lin(p, k))
case("collector_ended_early", 2, "먼저 끝났다", lambda p, k: None if k >= 15 else lin(p, k))
case("collector_started_late", 2, "늦게 시작했다", lambda p, k: None if k < 6 else lin(p, k))
case("no_run_log", 2, "런 구간", lin, run_log=False)
case("restart", 2, "8003: 누계가 줄었거나", lambda p, k: lin(p, k - 20 if (p == 8003 and k >= 20) else k))
case("flat_with_running", 2, "8003: 정지 의심", lambda p, k: {"frac": min(k, 20) / (NS - 1), "running": 5} if p == 8003 else lin(p, k))
case("flat_with_waiting_only", 2, "8003: 정지 의심", lambda p, k: {"frac": min(k, 20) / (NS - 1), "running": 0, "waiting": 12} if (p == 8003 and k >= 20) else lin(p, k))
case("zero_hits", 2, "캐시분이 0", lin, hit=0.0)
case("stickiness_broken", 2, "깨짐", lin, hit=0.87)
case("hit_below_prediction", 3, "진단 필요", lin, hit=0.93)
case("one_style_low_mixed_in", 3, "진단 필요", lin, hit=lambda p: 0.87 if p in (8003, 8004, 8005) else 0.975)
case("hit_above_prediction", 3, "진단 필요", lin, hit=0.999)
case("prediction_covers_half", 3, "참고만", lin, records=4)
# the other seven engines carry the whole run (the rollouts' tokens and the engines' tokens are the same tokens)
case("idle_engine", 0, "요청을 받지 않은 엔진 [8003]", lambda p, k: {"frac": 0.0, "running": 0} if p == 8003 else {"frac": k / (NS - 1) * 8 / 7})
case("cache_off", 0, "해당 없음", lin, hit=0.0, cache="0")
case("cache_off_but_cached", 2, "캐시 끔으로 기록된 런인데", lin, hit=0.5, cache="0")


def turn_count_case():
    """Assistant output only: a user nudge closes a turn and is counted separately."""
    d = ROOT / "turns"
    d.mkdir()
    out = [{"type": "reasoning", "summary": [{"text": "a"}]}, {"type": "function_call", "name": "execute_bash", "call_id": "1"},
           {"type": "function_call_output", "call_id": "1", "output": "x"},
           {"type": "message", "role": "assistant", "content": [{"text": "done?"}]},
           {"type": "message", "role": "user", "content": [{"text": "please continue"}]},
           {"type": "reasoning", "summary": [{"text": "b"}]}, {"type": "function_call", "name": "str_replace_editor", "call_id": "2", "arguments": json.dumps({"command": "str_replace", "path": "/r/a.py"})},
           {"type": "function_call", "name": "execute_bash", "call_id": "3"},
           {"type": "function_call_output", "call_id": "2", "output": "ERROR: No replacement was performed"}, {"type": "function_call_output", "call_id": "3", "output": "ok"}]
    row = dict(responses_create_params=dict(metadata=dict(instance_id="t")), instance_config=dict(resolved_agent_cls="CodeActAgent"), reward=1.0,
               response=dict(output=out), per_turn_metrics=dict(token_usages=[{"prompt_tokens": 5000, "completion_tokens": 50}] * 3))
    (d / "rollouts.jsonl").write_text(json.dumps(row) + "\n")
    r = subprocess.run([sys.executable, "-I", "-c", f"import importlib.util,sys;s=importlib.util.spec_from_file_location('a','{TOOL}');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
                        "r=m.load('turns')[0];print(len(r['turns']),r['n_user'],r['edits'],r['edit_fail'],r['same_prompt'])"],
                       capture_output=True, text=True, env={**os.environ, "SWE_MEASURE_ROOT": str(ROOT)})
    return r.stdout.strip() == "3 1 1 1 2", r.stdout.strip() + r.stderr.strip()[-300:]


def main() -> int:
    bad = 0
    for tag, rc, phrase in CASES:
        r = subprocess.run([sys.executable, "-I", str(TOOL), tag], capture_output=True, text=True, env={**os.environ, "SWE_MEASURE_ROOT": str(ROOT)})
        ok = r.returncode == rc and phrase in r.stdout
        bad += not ok
        print(f"{'ok  ' if ok else 'BAD '} {tag}: exit {r.returncode} (expected {rc}), must mention {phrase!r}")
        if not ok:
            print("      " + "\n      ".join((r.stdout + r.stderr).strip().splitlines()[-9:]))
    ok, got = turn_count_case()
    bad += not ok
    print(f"{'ok  ' if ok else 'BAD '} turn counting: turns, user nudges, edits, failed edits, repeated prompts = {got} (expected 3 1 1 1 2)")
    print(f"{len(CASES) + 1 - bad}/{len(CASES) + 1} cases as expected · {ROOT}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
