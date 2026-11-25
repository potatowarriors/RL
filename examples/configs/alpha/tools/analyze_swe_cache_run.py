"""SWE 학생 롤아웃 측정의 캐시 · 속도 · 행동 분석 (CPU) — 적대적 검토 (2026-11-25) 가 요구한 "섞이지 않는 양" 으로 낸다.

  python analyze_swe_cache_run.py <tag> [--baseline <tag>]        (결과는 $NRL_ROOT/p0_measure/swe/<tag>/)

읽는 것: rollouts.jsonl (모든 롤아웃 — 시간 초과 롤아웃에는 턴별 기록이 없으므로 그 필드에 기대지 않는다) · fleet_metrics.tsv (runs/swe_fleet_metrics.sh).
  1. 엔진: 적중률 = Δprompt_tokens_cached / Δprompt_tokens (요청당 한 번 세는 누계). 계산한 prefill 토큰 = Δprompt − Δcached.
     엔진 정지 = 실행 중 요청이 있는데 생성 토큰 누계가 120 초 넘게 그대로. 정지가 있으면 그 런의 속도 수치는 무효다.
     세션 고정은 직접 볼 수 없다 — 엔진 8 개가 모두 적중률 0.90 이상이면 한 롤아웃의 호출이 같은 엔진에 머문 것으로 본다 (깨지면 0.12 ~ 0.20 이 된다).
  2. 롤아웃: 모델 턴 수는 response.output 에서 세고, 턴당 모델 시간 = total_model_call_time / 턴 수 (모든 롤아웃).
     시간 초과는 물결별로 낸다 (첫 물결 = 가장 이른 시작에서 120 초 안에 시작한 롤아웃 — 한꺼번에 뜨는 기동 지연이 섞인다).
  3. 행동 불변량 (캐시가 모델 출력을 바꾸면 여기서 먼저 보인다 — 통과율은 512 개로는 ±11 건이 잡음이다):
     도구 호출이 있는 턴 비율 · 메시지만 있는 턴 비율 · 추론 길이 · 도구 호출 표기가 글에 남은 턴 비율 · 같은 프롬프트 재호출.
  4. 에이전트 틀별 (resolved_agent_cls).
캐시 켬 런과 끔 런의 엔진 배정 방식이 다르면 (프록시 대 Gym 세션 해시) 롤아웃 시간 · 시간 초과율의 차이를 캐시만의 효과로 읽지 않는다.
"""
import argparse
import collections
import datetime
import json
import os
import re
import sys

P = "/home/work/vidsearch/tools/nemo_rl/p0_measure/swe/"
MARK = re.compile(r"<tool_call>|</tool_call>|<function=|<parameter=|</function>")


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


def num(x):
    """숫자 또는 ISO 시각 문자열 → 초 (Gym 은 시작 시각을 ISO 문자열로 적는다)."""
    try:
        return float(x)
    except (TypeError, ValueError):
        pass
    try:
        return datetime.datetime.fromisoformat(str(x)).timestamp()
    except (TypeError, ValueError):
        return None


def text_len(x):
    if isinstance(x, list):
        return sum(len(c.get("text", "")) for c in x if isinstance(c, dict))
    return len(str(x or ""))


def load(tag):
    rows = []
    for line in open(P + tag + "/rollouts.jsonl"):
        r = json.loads(line)
        ic = r.get("instance_config") or {}
        out = (r.get("response") or {}).get("output") or []
        turns, cur = [], None
        for o in out:
            ty = o.get("type")
            if ty == "function_call_output":
                cur = None
                continue
            # 모델 호출 하나 = (추론) → 메시지 또는 도구 호출들. 도구 결과 뒤, 또는 이미 메시지 · 도구 호출이 나온 뒤의 추론 · 메시지에서 새 턴이 시작된다.
            if cur is None or (ty == "reasoning" and (cur["call"] or cur["msg"])) or (ty == "message" and cur["msg"]):
                cur = dict(call=0, msg=0, reason=0, mark=0)
                turns.append(cur)
            if ty == "function_call":
                cur["call"] += 1
            elif ty == "message":
                c = o.get("content")
                txt = " ".join(x.get("text", "") for x in c if isinstance(x, dict)) if isinstance(c, list) else str(c or "")
                cur["msg"] += 1
                cur["mark"] += bool(MARK.search(txt))
            elif ty == "reasoning":
                cur["reason"] += text_len(o.get("summary")) + text_len(o.get("content"))
        ptm = r.get("per_turn_metrics") if isinstance(r.get("per_turn_metrics"), dict) else {}
        pt = [u.get("prompt_tokens") or 0 for u in ptm.get("token_usages") or []]
        rows.append(dict(
            cls=ic.get("resolved_agent_cls") or "?", reward=r.get("reward"), to=bool(r.get("agent_timed_out")),
            run_s=r.get("openhands_run_time") or 0.0, model_s=r.get("total_model_call_time") or 0.0, cmd_s=r.get("total_command_exec_time") or 0.0,
            connect_s=r.get("connect_to_runtime_time"), start=num(r.get("generation_start_timestamp")),
            turns=turns, same_prompt=sum(1 for a, b in zip(pt, pt[1:]) if a == b), pt_calls=len(pt)))
    return rows


def rollout_block(rows, title):
    n = len(rows)
    if not n:
        return
    starts = [r["start"] for r in rows if isinstance(r["start"], (int, float))]
    t0 = min(starts) if starts else None
    w1 = [r for r in rows if t0 is not None and isinstance(r["start"], (int, float)) and r["start"] - t0 <= 120]
    w1_ids = {id(r) for r in w1}
    w2 = [r for r in rows if id(r) not in w1_ids]
    turns = [t for r in rows for t in r["turns"]]
    per_turn = [r["model_s"] / len(r["turns"]) for r in rows if r["turns"]]
    conn = lambda rs: q([r["connect_s"] for r in rs if r["connect_s"] is not None], .5)
    run_sum = max(sum(r["run_s"] for r in rows), 1e-9)
    npass = sum(r["reward"] == 1.0 for r in rows)
    nto = sum(r["to"] for r in rows)
    print(f"  [{title}] 롤아웃 {n} · 통과 {npass} ({100 * npass / n:.1f}%) · 시간 초과 {nto} ({100 * nto / n:.1f}%)")
    print(f"      시간 초과 — 첫 물결 {sum(r['to'] for r in w1)}/{len(w1)} · 이후 {sum(r['to'] for r in w2)}/{len(w2)} | 런타임 연결 시간 중앙 — 첫 물결 {conn(w1):.0f}s · 이후 {conn(w2):.0f}s")
    print(f"      롤아웃 시간 중앙 {q([r['run_s'] for r in rows], .5):.0f}s · p90 {q([r['run_s'] for r in rows], .9):.0f}s | 모델 시간 비중 {100 * sum(r['model_s'] for r in rows) / run_sum:.0f}% · 명령 실행 비중 {100 * sum(r['cmd_s'] for r in rows) / run_sum:.0f}%")
    print(f"      모델 턴 수 중앙 {q([len(r['turns']) for r in rows], .5):.0f} (모든 롤아웃) · 턴당 모델 시간 — 중앙 {q(per_turn, .5):.2f}s · p90 {q(per_turn, .9):.2f}s · 풀링 평균 {sum(r['model_s'] for r in rows) / max(len(turns), 1):.2f}s")
    if turns:
        nt = len(turns)
        print(f"      행동 ({nt:,} 턴): 도구 호출 있는 턴 {100 * sum(t['call'] > 0 for t in turns) / nt:.2f}% · 메시지만 있는 턴 {100 * sum(t['call'] == 0 and t['msg'] > 0 for t in turns) / nt:.2f}% · "
              f"도구 표기가 글에 남은 턴 {100 * sum(t['mark'] > 0 for t in turns) / nt:.3f}% · 추론 길이 중앙 {q([t['reason'] for t in turns], .5):.0f}자 · p90 {q([t['reason'] for t in turns], .9):.0f}자 · "
              f"같은 프롬프트 재호출 {sum(r['same_prompt'] for r in rows)}/{sum(r['pt_calls'] for r in rows)} (턴 기록이 있는 롤아웃만)")


def engine_block(tag):
    f = P + tag + "/fleet_metrics.tsv"
    if not os.path.exists(f):
        print("  엔진 지표 없음 (fleet_metrics.tsv) — 캐시 적중률 · 정지 감지 불가")
        return None
    S = collections.defaultdict(lambda: collections.defaultdict(list))   # port -> metric -> [(ts, value)]
    for line in open(f):
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 4:
            continue
        try:
            S[int(parts[1])][parts[2]].append((int(parts[0]), float(parts[3])))
        except ValueError:
            pass
    ok, tot_p, tot_c, low = True, 0.0, 0.0, []
    print("  엔진별 (런 구간 증분): 포트 · 프롬프트 토큰 · 캐시분 · 적중률 · 계산한 prefill · 생성 토큰 · 실행 최대 · 대기 최대 · KV 최대 · 선점 · 요청당 prefill · 대기 · e2e")
    for port in sorted(S):
        m = S[port]

        def delta(name):
            v = m.get(name) or []
            if len(v) < 2 or any(b[1] < a[1] for a, b in zip(v, v[1:])):   # 누계가 줄었다 = 서버 재기동
                return float("nan")
            return v[-1][1] - v[0][1]

        def per(name):
            s, c = delta(name + "_sum"), delta(name + "_count")
            return s / c if c == c and c > 0 else float("nan")

        def mx(name):
            return max((v for _, v in m.get(name) or []), default=float("nan"))

        dp, dc, dg = delta("vllm:prompt_tokens_total"), delta("vllm:prompt_tokens_cached_total"), delta("vllm:generation_tokens_total")
        hr = dc / dp if dp == dp and dp > 0 else float("nan")
        running = dict(m.get("vllm:num_requests_running") or [])
        gen = m.get("vllm:generation_tokens_total") or []
        stall, flat_since = 0, None
        for (ta, ga), (tb, gb) in zip(gen, gen[1:]):
            if gb == ga and running.get(ta, 0) > 0 and running.get(tb, 0) > 0:
                flat_since = ta if flat_since is None else flat_since
                stall = max(stall, tb - flat_since)
            else:
                flat_since = None
        print(f"    {port} · {dp / 1e6:8.1f}M · {dc / 1e6:8.1f}M · {hr:.3f} · {(dp - dc) / 1e6:7.2f}M · {dg / 1e6:6.2f}M · {mx('vllm:num_requests_running'):3.0f} · {mx('vllm:num_requests_waiting'):3.0f} · "
              f"{100 * mx('vllm:kv_cache_usage_perc'):4.0f}% · {delta('vllm:num_preemptions_total'):.0f} · {per('vllm:request_prefill_time_seconds'):.2f}s · {per('vllm:request_queue_time_seconds'):.2f}s · {per('vllm:e2e_request_latency_seconds'):.2f}s"
              + (f"  ← 정지 의심 {stall}s" if stall >= 120 else ""))
        if dp == dp and dc == dc:
            tot_p += dp
            tot_c += dc
        if not (hr == hr) or stall >= 120:
            ok = False
        if hr == hr and hr < 0.90:
            low.append(port)
    print(f"  fleet 적중률 {tot_c / tot_p if tot_p else float('nan'):.3f} · 계산한 prefill 토큰 {(tot_p - tot_c) / 1e6:.1f}M / 프롬프트 {tot_p / 1e6:.1f}M · 엔진 {len(S)} 개")
    if tot_c > 0:
        print("  세션 고정 판정: " + ("엔진 8 개가 모두 적중률 0.90 이상 — 한 롤아웃의 호출이 같은 엔진에 머문 것으로 본다" if not low and len(S) == 8
                                 else f"적중률 0.90 미만 엔진 {low} (엔진 {len(S)} 개) — 결론 대신 진단이 필요하다 (세션 고정 깨짐 또는 캐시 축출)"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--baseline")
    a = ap.parse_args()
    for tag in [a.tag] + ([a.baseline] if a.baseline else []):
        print(f"== {tag} ==")
        rows = load(tag)
        engine_block(tag)
        rollout_block(rows, "전체")
        by = collections.defaultdict(list)
        for r in rows:
            by[r["cls"]].append(r)
        if len(by) > 1:
            for k, v in sorted(by.items()):
                rollout_block(v, k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
