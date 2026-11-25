"""SWE 학생 롤아웃 측정의 캐시 · 속도 · 행동 분석 (CPU) — 적대적 검토 (2026-11-25, 세 차례) 가 요구한 "섞이지 않는 양" 으로 낸다.

  python analyze_swe_cache_run.py <tag> [--baseline <tag>]        (결과는 $NRL_ROOT/p0_measure/swe/<tag>/)

읽는 것: rollouts.jsonl · fleet_metrics.tsv (runs/swe_fleet_metrics.sh) · run.log (실행기가 적는 기동 · 종료 줄).
  1. 엔진: 적중률 = Δprompt_tokens_cached / Δprompt_tokens (요청당 한 번 세는 누계). 계산한 prefill 토큰 = Δprompt − Δcached.
     - 수집 구간이 런 (Gym 서버 준비 ~ eval 종료) 을 덮지 않으면 엔진 수치 전체가 무효다 (수집기가 도중에 끝나면 앞부분만 남는다).
     - 정지 = 대기 · 실행 요청이 있는데 생성 · 프롬프트 누계가 120 초 넘게 그대로, 또는 /metrics 무응답 구간에 누계가 평소 속도의 1/4 도
       늘지 않음, 또는 그 포트의 표본이 먼저 끊김. 정지 · 재기동 · 사라진 엔진이 있으면 속도 수치는 무효이고 종료 코드는 2 다.
     - 세션 고정 · 접두사 성질은 직접 볼 수 없다. **그 런 자신의 롤아웃 기록으로 낸 예측 적중률** 과 견준다: 턴별 (프롬프트, 생성) 토큰에서
       P_lo = 직전 프롬프트의 마지막 블록 경계까지 재사용 · P_hi = 직전 생성까지 재사용 (블록 1,056, 적중은 블록 경계에서 끝나고 1 토큰 이상
       남긴다, 롤아웃의 첫 호출은 차갑다고 본다). 측정 H ≥ P_lo − 0.01 이면 통과, H < 0.92 면 깨짐 (호출마다 엔진이 바뀌면 0.84 ~ 0.89,
       시스템 프롬프트 안에서 어긋나면 0.10 근처 — 검토자의 s2b 기록 시뮬레이션), 그 사이는 진단. 예측은 턴 기록이 있는 롤아웃만 덮는다 —
       덮은 비율이 0.8 미만이면 판정 대신 참고로 낸다. 틀을 섞은 런의 적중률은 큰 고장만 잡는다 (한 틀만 0.87 이어도 섞이면 0.94).
       엔진별 적중률로는 판정하지 않는다 (롤아웃 하나는 정상이어도 0.875 까지 내려간다, 세션 해시로 엔진마다 부하가 다르다).
  2. 롤아웃: 모델 턴 수는 response.output 의 assistant 출력만 센다. 시간 초과는 물결별 (첫 물결 = 가장 이른 시작에서 120 초 안) · 틀별.
  3. 행동 불변량 (감시용 — 통과율로 정확도를 판단하지 않는다): 도구 호출이 있는 턴 · 메시지만 있는 턴 · 추론 길이 · 도구 표기가 글에 남은 턴 ·
     같은 프롬프트 재호출 · 에이전트 안내 메시지 · 틀별 편집 도구 실패율 · agent_error_kind.
  4. --baseline: **CodeAct 롤아웃만, 두 런에 다 있는 과제만, 과제마다 같은 가중** 으로 견준다. 호출당 응답 시간은 턴 번호 구간을 맞춘다
     (턴 기록이 있는 롤아웃만). 틀 사이 비교는 결론에 쓰지 않는다.
종료 코드: 0 = 엔진 수치가 유효하고 적중률 판정이 통과 (캐시 끔 런은 해당 없음) · 2 = 무효 또는 깨짐 · 3 = 판정하지 못함 (진단 필요 · 참고만).
쓰지 않는 표현: "캐시로 N 배" · "캐시 효과" — 캐시 · 엔진 배정 방식 (프록시 대 세션 해시) · 틀 구성 · 동시 수가 함께 바뀐다. 처리량은 fleet 수준만 낸다.
"""
import argparse
import collections
import datetime
import json
import os
import re
import sys

P = os.environ.get("SWE_MEASURE_ROOT", "/home/work/vidsearch/tools/nemo_rl/p0_measure/swe").rstrip("/") + "/"   # 환경변수는 시험용
MARK = re.compile(r"<tool_call>|</tool_call>|<function=|<parameter=|</function>")
PORTS = list(range(8000, 8008))
BLOCK = 1056
NOT_COMMANDS = {"finish", "think", "update_plan", "todo_write"}   # 실행 · 편집이 아닌 도구 (세 틀 공통으로 뺀다)
BANDS = [(1, 20), (21, 50), (51, 100), (101, 200)]


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


def edit_result(name, args, res):
    """편집 도구 호출 하나의 결과: "ok" · "fail" · "unknown" (아는 문구가 아니다) · None (편집 도구가 아니다).

    틀마다 편집 도구와 결과 문구가 다르다 (nv-OpenHands 의 도구 구현, 2026-11-25 틀별 런에서 확인):
    CodeAct `str_replace_editor` (create · str_replace · insert) 실패는 "ERROR" 로 시작 · Codex `apply_patch` 성공은
    "Patch applied successfully", 실패는 "Errors (N):" · "Patch parse error" · OpenCode `edit` 성공은 "The file … has been edited",
    실패는 "ERROR" · `write` 성공은 "File written successfully".
    """
    head = str(res or "").lstrip()[:300]
    if name == "str_replace_editor":
        if not isinstance(args, dict) or args.get("command") not in ("create", "str_replace", "insert"):
            return None
        return "fail" if head.startswith("ERROR") else "ok"
    if name == "apply_patch":
        return "ok" if head.startswith("Patch applied successfully") else "fail" if re.match(r"Errors? \(\d+\)|Patch parse error", head) else "unknown"
    if name == "edit":
        return "fail" if head.startswith("ERROR") else "ok" if re.match(r"The file \S.* has been edited", head, re.S) else "unknown"
    if name == "write":
        return "fail" if head.startswith("ERROR") else "ok" if head.startswith("File written successfully") else "unknown"
    return None


def edited_paths(name, args, res):
    """성공한 편집이 건드린 파일 경로 (apply_patch 는 저장소 기준 상대 경로일 수 있다)."""
    if not isinstance(args, dict):
        return []
    if name == "str_replace_editor":
        return [str(args.get("path") or "")]
    if name in ("edit", "write"):
        return [str(args.get("file_path") or "")]
    if name == "apply_patch":
        return re.findall(r"^\*\*\* (?:Update|Add) File: (\S+)", str(args.get("input") or ""), flags=re.M)
    return []


def run_info(tag):
    """실행기의 run.log 에서 설정과 런 구간 (Gym 서버 준비 ~ eval 종료) 을 읽는다."""
    info = dict(prefix_cache=None, prompt_mix=None, conc=None, t0=None, t1=None, seconds=None)
    try:
        txt = open(P + tag + "/run.log", errors="replace").read()
    except OSError:
        return info
    ts = lambda s: datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp()
    m = re.search(r"armed:.*", txt)
    if m:
        for k in ("prefix_cache", "prompt_mix", "conc"):
            v = re.search(rf"\b{k}=(\S+)", m.group(0))
            info[k] = v.group(1) if v else None
    m = re.findall(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] gym servers ready", txt)
    info["t0"] = ts(m[-1]) if m else None
    m = re.findall(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] eval rc=\S+.*?seconds=(\d+)", txt)
    if m:
        info["t1"], info["seconds"] = ts(m[-1][0]), int(m[-1][1])
    return info


def load(tag):
    rows = []
    for line in open(P + tag + "/rollouts.jsonl"):
        r = json.loads(line)
        ic = r.get("instance_config") or {}
        out = (r.get("response") or {}).get("output") or []
        outs = {o.get("call_id"): o for o in out if o.get("type") == "function_call_output"}
        turns, cur, n_user, edits, edit_fail, edit_unknown, fail_heads = [], None, 0, 0, 0, 0, []
        for o in out:
            ty = o.get("type")
            if ty == "function_call_output":
                cur = None
                continue
            if ty == "message" and o.get("role") not in (None, "assistant"):
                n_user += 1          # 에이전트가 끼워 넣는 안내 ("계속하라" 등) — 모델 턴이 아니다. 도구 결과처럼 턴을 닫는다
                cur = None
                continue
            # 모델 호출 하나 = (추론) → 메시지 또는 도구 호출들. 도구 결과 뒤, 또는 이미 메시지 · 도구 호출이 나온 뒤의 추론 · 메시지에서 새 턴이 시작된다.
            if cur is None or (ty == "reasoning" and (cur["call"] or cur["msg"])) or (ty == "message" and cur["msg"]):
                cur = dict(call=0, msg=0, reason=0, mark=0)
                turns.append(cur)
            if ty == "function_call":
                cur["call"] += 1
                if o.get("call_id") in outs:
                    try:
                        args = json.loads(o.get("arguments") or "{}")
                    except ValueError:
                        args = {}
                    res = str(outs[o["call_id"]].get("output") or "")
                    kind = edit_result(o.get("name"), args, res)
                    if kind is not None:
                        edits += 1
                        edit_fail += kind == "fail"
                        edit_unknown += kind == "unknown"
                        if kind != "ok":
                            fail_heads.append(f"{o.get('name')}: " + " ".join(res.split())[:44])
            elif ty == "message":
                c = o.get("content")
                txt = " ".join(x.get("text", "") for x in c if isinstance(x, dict)) if isinstance(c, list) else str(c or "")
                cur["msg"] += 1
                cur["mark"] += bool(MARK.search(txt))
            elif ty == "reasoning":
                cur["reason"] += text_len(o.get("summary")) + text_len(o.get("content"))
        ptm = r.get("per_turn_metrics") if isinstance(r.get("per_turn_metrics"), dict) else {}
        tu = [(u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0) for u in ptm.get("token_usages") or []]
        rows.append(dict(
            iid=r["responses_create_params"]["metadata"]["instance_id"],
            cls=ic.get("resolved_agent_cls") or "?", reward=r.get("reward"), to=bool(r.get("agent_timed_out")), err=str(r.get("agent_error_kind")),
            run_s=r.get("openhands_run_time") or 0.0, model_s=r.get("total_model_call_time") or 0.0, cmd_s=r.get("total_command_exec_time") or 0.0,
            connect_s=r.get("connect_to_runtime_time"), start=num(r.get("generation_start_timestamp")),
            turns=turns, n_user=n_user, tu=tu, same_prompt=sum(1 for a, b in zip(tu, tu[1:]) if a[0] == b[0]), pt_calls=len(tu),
            lat=[x.get("latency") for x in ptm.get("response_latencies") or [] if isinstance(x.get("latency"), (int, float))],
            edits=edits, edit_fail=edit_fail, edit_unknown=edit_unknown, fail_heads=fail_heads))
    return rows


def predicted_hit(rows):
    """그 런의 턴 기록에서 낸 예측 적중률 (P_lo, P_hi) 과 예측이 덮는 프롬프트 토큰 수."""
    fl = lambda x: (max(x, 0) // BLOCK) * BLOCK
    tot = lo = hi = 0
    for r in rows:
        prev = None
        for L, G in r["tu"]:
            tot += L
            if prev is not None:
                Lp, Gp = prev
                if L == Lp:
                    lo += fl(L - 1)
                    hi += fl(L - 1)
                else:
                    lo += fl(min(Lp, L - 1))
                    hi += fl(min(Lp + Gp - 1, L - 1))
            prev = (L, G)
    return (lo / tot, hi / tot, tot) if tot else (float("nan"), float("nan"), 0)


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
              f"같은 프롬프트 재호출 {sum(r['same_prompt'] for r in rows)}/{sum(r['pt_calls'] for r in rows)} (턴 기록이 있는 롤아웃만) · 에이전트 안내 메시지 {sum(r['n_user'] for r in rows)} 개")
    ne, nf, nu = sum(r["edits"] for r in rows), sum(r["edit_fail"] for r in rows), sum(r["edit_unknown"] for r in rows)
    heads = collections.Counter(h for r in rows for h in r["fail_heads"]).most_common(3)
    errs = collections.Counter(r["err"] for r in rows if r["err"] != "None")
    print(f"      편집 도구 호출 {ne} · 실패 {nf} ({100 * nf / ne if ne else float('nan'):.1f}%)" + (f" · 분류 못 한 결과 {nu}" if nu else "") + (f" — 흔한 글머리 {heads}" if heads else "")
          + f" | agent_error_kind {dict(errs) if errs else '없음'}")


def engine_block(tag, rows, info):
    """엔진 지표. 돌려주는 값: (속도 수치를 쓸 수 있는가, 적중률 판정 문자열)."""
    f = P + tag + "/fleet_metrics.tsv"
    if not os.path.exists(f):
        print("  엔진 지표 없음 (fleet_metrics.tsv) — 캐시 적중률 · 정지 감지 · 처리량을 낼 수 없다")
        return None, "지표 없음"
    S = collections.defaultdict(lambda: collections.defaultdict(list))   # port -> metric -> [(ts, value)]
    for line in open(f):
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 4:
            continue
        try:
            S[int(parts[1])][parts[2]].append((int(parts[0]), float(parts[3])))
        except ValueError:
            pass
    all_ts = sorted({ts for m in S.values() for v in m.values() for ts, _ in v})
    problems = []
    if not all_ts:
        print("  엔진 지표가 비어 있다")
        return False, "지표 없음"
    first_all, last_all = all_ts[0], all_ts[-1]
    # 수집 구간이 런을 덮는가
    if info["t0"] is None or info["t1"] is None:
        problems.append("run.log 에서 런 구간 (gym servers ready ~ eval rc) 을 읽지 못했다 — 수집 구간을 확인할 수 없다")
    else:
        if first_all > info["t0"] + 120:
            problems.append(f"수집이 런 시작보다 {first_all - info['t0']:.0f}s 늦게 시작했다")
        if last_all < info["t1"] - 120:
            problems.append(f"수집이 런 종료보다 {info['t1'] - last_all:.0f}s 먼저 끝났다 (런 {info['t1'] - info['t0']:.0f}s 중 뒤쪽이 빠졌다)")
    missing = [p for p in PORTS if p not in S]
    if missing:
        problems.append(f"지표가 없는 엔진 {missing} (사라졌거나 수집되지 않았다)")
    tot_p = tot_c = tot_g = 0.0
    idle, notes = [], []
    print("  엔진별 (수집 구간 증분): 포트 · 프롬프트 토큰 · 캐시분 · 적중률 · 계산한 prefill · 생성 토큰 · 실행 최대 · 대기 최대 · KV 최대 · 선점 · 요청당 prefill · 대기 · e2e")
    for port in sorted(S):
        m = S[port]

        def series(name):
            return m.get(name) or []

        def delta(name):
            v = series(name)
            if len(v) < 2 or any(b[1] < a[1] for a, b in zip(v, v[1:])):   # 누계가 줄었다 = 서버 재기동
                return float("nan")
            return v[-1][1] - v[0][1]

        def per(name):
            s, c = delta(name + "_sum"), delta(name + "_count")
            return s / c if c == c and c > 0 else float("nan")

        def mx(name):
            return max((v for _, v in series(name)), default=float("nan"))

        dp, dc, dg = delta("vllm:prompt_tokens_total"), delta("vllm:prompt_tokens_cached_total"), delta("vllm:generation_tokens_total")
        hr = dc / dp if dp == dp and dp > 0 else float("nan")
        if not (dp == dp and dc == dc and dg == dg):
            problems.append(f"{port}: 누계가 줄었거나 표본이 2 개 미만이다 (재기동 의심)")
        elif dp == 0:
            idle.append(port)
        run_d, wait_d = dict(series("vllm:num_requests_running")), dict(series("vllm:num_requests_waiting"))
        busy = lambda t: run_d.get(t, 0) + wait_d.get(t, 0) > 0
        gen, prm = series("vllm:generation_tokens_total"), dict(series("vllm:prompt_tokens_total"))
        # 구간별 생성 속도 (요청이 있던 구간) 의 중앙값 — 무응답 구간을 건너 누계가 "평소만큼" 늘었는지 보는 기준
        rates = sorted((gb - ga) / (tb - ta) for (ta, ga), (tb, gb) in zip(gen, gen[1:]) if tb > ta and tb - ta <= 120 and busy(ta) and busy(tb))
        typical = rates[len(rates) // 2] if rates else 0.0
        stall, flat_since, skipped = 0, None, 0
        for (ta, ga), (tb, gb) in zip(gen, gen[1:]):
            flat = gb == ga and prm.get(tb, 0) == prm.get(ta, 0)
            if tb - ta > 120:     # /metrics 가 응답하지 않은 구간 — API 서버째 멈추면 표본이 없다
                if busy(ta) and (gb - ga) / (tb - ta) < 0.25 * typical:
                    stall = max(stall, tb - ta)
                else:
                    skipped += 1
                flat_since = None
            elif flat and busy(ta) and busy(tb):
                flat_since = ta if flat_since is None else flat_since
                stall = max(stall, tb - flat_since)
            else:
                flat_since = None
        ts_list = sorted({ts for v in m.values() for ts, _ in v})
        behind = last_all - ts_list[-1] if ts_list else 10 ** 9
        if behind >= 120:
            stall = max(stall, behind)
        if stall >= 120:
            problems.append(f"{port}: 정지 의심 {stall}s")
        if skipped:
            notes.append(f"{port}: 표본 누락 구간 {skipped} 개 (누계는 평소 속도로 늘었다)")
        print(f"    {port} · {dp / 1e6:8.1f}M · {dc / 1e6:8.1f}M · {hr:.3f} · {(dp - dc) / 1e6:7.2f}M · {dg / 1e6:6.2f}M · {mx('vllm:num_requests_running'):3.0f} · {mx('vllm:num_requests_waiting'):3.0f} · "
              f"{100 * mx('vllm:kv_cache_usage_perc'):4.0f}% · {delta('vllm:num_preemptions_total'):.0f} · {per('vllm:request_prefill_time_seconds'):.2f}s · {per('vllm:request_queue_time_seconds'):.2f}s · {per('vllm:e2e_request_latency_seconds'):.2f}s"
              + (f"  ← 정지 의심 {stall}s" if stall >= 120 else "") + ("  (요청 없음)" if port in idle else ""))
        if dp == dp and dc == dc and dg == dg:
            tot_p += dp
            tot_c += dc
            tot_g += dg
    H = tot_c / tot_p if tot_p else float("nan")
    dur = max(last_all - first_all, 1)
    print(f"  fleet 적중률 {H:.4f} · 계산한 prefill 토큰 {(tot_p - tot_c) / 1e6:.2f}M / 프롬프트 {tot_p / 1e6:.2f}M · 엔진 {len(S)} 개" + (f" (요청을 받지 않은 엔진 {idle})" if idle else ""))
    print(f"  fleet 처리량 (수집 구간 {dur}s 전체, 정상 상태 구간을 따로 잡지 않는다): 생성 {tot_g / dur:,.0f} tok/s · 계산한 prefill {(tot_p - tot_c) / dur:,.0f} tok/s · 프롬프트 {tot_p / dur:,.0f} tok/s"
          + (f" · 롤아웃 {60 * len(rows) / info['seconds']:.2f} 개/분 (eval {info['seconds']}s)" if info["seconds"] else ""))
    for n_ in notes:
        print(f"  참고: {n_}")
    for p_ in problems:
        print(f"  문제: {p_}")
    # 적중률 판정 — 그 런 자신의 예측과 견준다
    lo, hi, covered = predicted_hit(rows)
    cover = covered / tot_p if tot_p else float("nan")
    n_rec = sum(1 for r in rows if r["tu"])
    print(f"  예측 적중률 (턴 기록이 있는 롤아웃 {n_rec}/{len(rows)} · 프롬프트 토큰의 {100 * cover:.1f}% 를 덮는다): P_lo {lo:.4f} · P_hi {hi:.4f}")
    cache_on = info["prefix_cache"] == "1"
    if not cache_on:
        verdict = "해당 없음 (캐시 끔 런)" if tot_c == 0 else "문제: 캐시 끔으로 기록된 런인데 캐시분이 있다"
    elif problems:
        verdict = "판정 불가 — 위 문제를 먼저 푼다"
    elif tot_c == 0:
        verdict = "실패 — 캐시 켬 런인데 캐시분이 0 이다"
    elif not (lo == lo) or cover < 0.8 or cover > 1.02:
        verdict = (f"참고만 — 예측이 프롬프트 토큰의 {100 * cover:.0f}% 를 덮는다 (0.8 ~ 1.02 밖). H {H:.4f} · P_lo {lo:.4f}"
                   + (" · H < 0.92 (깨짐 수준)" if H < 0.92 else ""))
    elif H < 0.92:
        verdict = f"깨짐 — H {H:.4f} < 0.92 (세션 고정 또는 접두사 성질)"
    elif H < lo - 0.01:
        verdict = f"진단 필요 — H {H:.4f} < P_lo − 0.01 = {lo - 0.01:.4f}"
    elif H > hi + 0.005:
        verdict = f"진단 필요 — H {H:.4f} > P_hi + 0.005 = {hi + 0.005:.4f} (예측 또는 계수가 틀렸다)"
    else:
        verdict = f"통과 — P_lo − 0.01 = {lo - 0.01:.4f} ≤ H {H:.4f} ≤ P_hi + 0.005 = {hi + 0.005:.4f}"
    mixed = len({r["cls"] for r in rows}) > 1
    print(f"  적중률 판정 (세션 고정 · 접두사 성질): {verdict}" + (" — 틀이 섞인 런이라 큰 고장만 잡는다" if mixed and cache_on else ""))
    return not problems, verdict


def compare_block(new, base, tag, btag):
    """CodeAct 롤아웃만, 두 런에 다 있는 과제만, 과제마다 같은 가중."""
    a = [r for r in new if r["cls"] == "CodeActAgent"]
    b = [r for r in base if r["cls"] == "CodeActAgent"]
    tasks = sorted({r["iid"] for r in a} & {r["iid"] for r in b})
    print(f"== 비교: {tag} 의 CodeAct {len(a)} 롤아웃 대 {btag} 의 CodeAct {len(b)} 롤아웃 · 공통 과제 {len(tasks)} 개 (과제마다 같은 가중) ==")
    if not tasks:
        print("  공통 과제가 없다")
        return

    def per_task(rs, fn):
        vals = []
        for t in tasks:
            xs = [fn(r) for r in rs if r["iid"] == t]
            xs = [x for x in xs if x is not None]
            if xs:
                vals.append(sum(xs) / len(xs))
        return (sum(vals) / len(vals), len(vals)) if vals else (float("nan"), 0)

    for name, fn in (("통과율", lambda r: float(r["reward"] == 1.0)), ("시간 초과율", lambda r: float(r["to"])),
                     ("모델 턴 수 (모든 롤아웃)", lambda r: float(len(r["turns"]))), ("롤아웃 시간 s", lambda r: r["run_s"]),
                     ("도구 호출 있는 턴 비율", lambda r: sum(t["call"] > 0 for t in r["turns"]) / len(r["turns"]) if r["turns"] else None),
                     ("추론 길이 (턴 평균, 자)", lambda r: sum(t["reason"] for t in r["turns"]) / len(r["turns"]) if r["turns"] else None)):
        (va, na), (vb, nb) = per_task(a, fn), per_task(b, fn)
        print(f"  {name:<24} {tag} {va:9.3f} (과제 {na}) · {btag} {vb:9.3f} (과제 {nb})")
    print("  호출당 응답 시간 (턴 기록이 있는 롤아웃만 — 시간 초과 롤아웃은 빠진다, 턴 번호 구간별 과제 평균의 평균):")
    for lo_, hi_ in BANDS:
        fn = lambda r, lo_=lo_, hi_=hi_: (sum(r["lat"][lo_ - 1:hi_]) / len(r["lat"][lo_ - 1:hi_])) if len(r["lat"]) >= lo_ else None
        (va, na), (vb, nb) = per_task(a, fn), per_task(b, fn)
        print(f"    턴 {lo_:>3} ~ {hi_:<3} {tag} {va:6.2f}s (과제 {na}) · {btag} {vb:6.2f}s (과제 {nb})")
    print("  읽는 법: 두 런은 캐시 · 엔진 배정 방식 · 동시 수 · 틀 구성이 함께 다르다. 차이를 캐시 하나의 효과로 읽지 않는다.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--baseline")
    a = ap.parse_args()
    loaded, rc = {}, 0
    for tag in [a.tag] + ([a.baseline] if a.baseline else []):
        info = run_info(tag)
        print(f"== {tag} == (캐시 {info['prefix_cache'] or '기록 없음 (끔으로 본다)'} · 틀 {info['prompt_mix'] or 'codeact (기록 없음)'} · 동시 {info['conc']})")
        rows = loaded[tag] = load(tag)
        speed_ok, verdict = engine_block(tag, rows, info)
        if speed_ok is False:
            print("  ⚠ 이 런의 속도 수치 (처리량 · 턴 시간 · 롤아웃 시간) 는 무효다 — 위 문제 참조")
        if tag == a.tag:
            if speed_ok is not True or verdict.startswith(("깨짐", "실패", "판정 불가", "문제", "지표 없음")):
                rc = 2
            elif not verdict.startswith(("통과", "해당 없음")):
                rc = 3
        rollout_block(rows, "전체")
        by = collections.defaultdict(list)
        for r in rows:
            by[r["cls"]].append(r)
        if len(by) > 1:
            for k, v in sorted(by.items()):
                rollout_block(v, k)
    if a.baseline:
        compare_block(loaded[a.tag], loaded[a.baseline], a.tag, a.baseline)
    return rc


if __name__ == "__main__":
    sys.exit(main())
