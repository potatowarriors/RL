"""terminal 피벗 judge 게이트 (TJ1) — 운영 Gym `terminus_judge` 검증 코드를 살아 있는 judge 모델에 그대로 돌린다 (RL_PLAN.md 결정 65).

지키려는 것: 논문 방식 보상 (유사도 0.95 빠른 통과, 미달이면 등가 LLM judge) 의 judge 가 **문자열 유사도보다 나은 판정**을 한다 —
글자가 달라도 같은 일을 하는 명령은 같다고 하고, 글자가 비슷해도 다른 일을 하는 명령은 다르다고 한다.
v1 (2026-11-16 적대적 검토 "보완 필요") 은 음성이 전부 유사도 < 0.5, 양성이 전부 > 0.77 이라 "유사도 ≥ 0.7" 로 판정하는 가짜 judge 도 통과했다.
v2 는 유사도로 갈리지 않는 세트를 넣고, 유사도 임계 judge 가 이 게이트에서 FAIL 하는지를 `selftest` 로 먼저 확인한다.

쌍은 원본 데이터 (`Nemotron-RL-Agentic-Terminal-Pivot-v1`) 에서 결정론적으로 만든다. 후보는 정답 JSON 의 analysis · plan 을 그대로 두고 commands 만 바꾼다.
  양성 (label 1)
    P_fast     keystrokes 는 같고 duration 만 다름 — 유사도 1.0 빠른 통과, judge 호출 0
    P_redirect 각 명령 끝에 ` 2>&1` (유사도 < 0.95)
    P_reorder  서로 독립인 읽기 명령 (cat · ls · head · tail · wc · stat · file) 3개 이상의 순서를 뒤집음 (유사도 < 0.7)
    P_split    `a && b` 한 명령을 두 명령으로 나눔 / 연속 두 명령을 `a && b` 로 합침
    P_flag     플래그 표기 (`-la` ↔ `-al` · `-la` ↔ `-l -a` · `-rn` ↔ `-nr`)
    P_sem      글자 정규화로는 안 맞는 의미 등가 — `grep p f` ↔ `cat f | grep p` · `cat a` + `cat b` ↔ `cat a b` ·
               `head -n N f` ↔ `sed -n '1,Np' f` · heredoc 구분자 이름만 바꿈
  음성 (label 0)
    N_wait     후보 = 빈 keystrokes 대기
    N_other    후보 = 다른 과제 행의 정답 commands (유사도 < 0.5)
    N_repeat   후보 = 이력의 직전 assistant 턴 commands (정답과 유사도 < 0.5)
    N_goldwait 정답이 빈 대기 + 후보 = 다른 과제의 실제 명령
    N_drop     정답 명령 4개 이상 중 앞 절반만 수행 (유사도 0.5~0.95 — 범위 차이)
    N_target   읽기 명령의 대상 경로를 같은 행의 다른 경로가 아닌 다른 과제의 경로로 바꿈 (유사도 0.5~0.95 — 다른 파일)
    N_verb     읽기 명령 (cat · head · tail · ls) 을 `rm -f` 로 바꿈 (유사도 ≥ 0.5 — 다른 행동 유형)
    N_wtrunc   heredoc 으로 쓰는 파일 본문의 뒤 1/3 줄을 뺌 (기능 누락) · N_sed `sed -i 's/A/B/'` 의 B 를 바꿈 (다른 편집) ·
    N_noexec   파일을 쓰고 실행하는 정답에서 실행 명령을 뺌 (작성만 대 작성 + 실행)
    (양성 P_wredir `cat > f << 'X'` ↔ `cat << 'X' > f` · P_pyfile `python3 << 'X'` ↔ 임시 파일에 쓰고 `python3 파일` — 쓰기 · 실행 명령의 등가)
    K_ctrl     (v5 부터 보고 전용 — 결정 66: 정답에 제어 키가 든 행은 블렌드에서 뺀다. v4 에서 두 judge 모두 0/100 — judge 는 tmux 키 전송 규칙을 모른다)
               제어 키의 개행 유무 (`C-c` ↔ `C-c\\n`) — tmux send-keys 에서 `C-c\\n` 은 글자 "C-c" + Enter 라 프로세스를 끊지 못한다
               (2026-11-16 재검토의 tmux 3.4 재현, Gym schemas.py 의 modifier 키 규칙). v2 는 이것을 양성으로 잘못 라벨했다
  보고만 (label None)
    U_*        MOPD1 학생의 실제 프로파일링 출력, 유사도 구간별 (0.1~0.3 · 0.3~0.5 · 0.5~0.7 · 0.7~0.95)
    B_*        N_drop · N_verb · P_reorder 의 analysis · plan 을 중립 문장으로 바꾼 변형 — 정답의 analysis · plan 복사가 judge 에게 단서를 주는지 본다
    (v5, 결정 66 — alpha_terminus_judge 플러그인: 완료 선언 양방향 검사 · 빠른 통과 임계 1.0)
    N_fastsed  `sed -i` 치환값만 바꾼 유사도 ≥ 0.95 쌍 — 임계 0.95 에서는 judge 없이 통과했다 (v4 100/100). 이제 judge 가 다르다고 해야 한다
    N_earlytc  정답 task_complete false + 후보 = 정답 commands 그대로 + task_complete true — Gym 의 완료 검사는 단방향이라 통과했다 (v4 100/100). 플러그인이 judge 없이 0 점으로 돌려야 한다
검증 경로: 운영 서버 클래스 AlphaTerminusJudgeResourcesServer (alpha 플러그인 — Gym TerminusJudgeResourcesServer + 완료 선언 양방향 검사) 를
프로세스 안에 만들고 verify() 를 부른다 — 스키마 · 완료 일치 · 유사도 · judge 프롬프트 · 판정 파싱이 운영 코드다. NEMO_GYM_EXTRA_ROOTS 필요.
설정 = 플러그인 `alpha_terminus_judge.yaml` + 레시피 덮어쓰기 (`env.nemo_gym.alpha_terminus_judge_resources_server`). 병합 결과가 기대값과 다르면 중단한다.

  python measure_judge_terminus.py build --out pairs.jsonl [--profile-rollouts full/rollouts.jsonl]
  python measure_judge_terminus.py selftest --pairs pairs.jsonl          # CPU: 유사도 임계 judge 가 FAIL 해야 한다 (하나라도 PASS 면 종료 코드 1)
  <Gym>/.venv/bin/python measure_judge_terminus.py run --gym-root <Gym> --recipe <recipe.yaml> --head 127.0.0.1:11000 \
        --pairs pairs.jsonl --out results.jsonl [--concurrency 64]
  python measure_judge_terminus.py flips --a results_c64.jsonl --b results_c256.jsonl   # 같은 쌍 재실행의 판정 뒤집힘률
종료 코드: run 은 기준을 하나라도 못 넘으면 1.
"""

import argparse
import asyncio
import collections
import json
import random
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

SOURCE = "/home/work/Datasets/LL_datasets/posttraining/RL/Nemotron-RL-Agentic-Terminal-Pivot-v1/atcb_terminal_pivot_release_final_v2.jsonl"
POS = ["P_fast", "P_redirect", "P_reorder", "P_split", "P_flag", "P_sem", "P_wredir", "P_pyfile"]
NEG = ["N_wait", "N_other", "N_repeat", "N_goldwait", "N_drop", "N_target", "N_verb", "N_wtrunc", "N_sed", "N_noexec", "N_fastsed", "N_earlytc"]
HARD = ["P_reorder", "P_split", "P_flag", "P_sem", "P_wredir", "P_pyfile", "N_drop", "N_target", "N_verb", "N_wtrunc", "N_sed", "N_noexec", "N_fastsed"]
SMALL = ("P_fast", "P_flag", "P_wredir", "P_pyfile", "N_wtrunc", "N_sed", "N_noexec", "N_fastsed", "N_earlytc")
WRITE = ("P_wredir", "P_pyfile", "N_wtrunc", "N_sed", "N_noexec")   # 쓰기 · 실행 명령 세트 — 원본에서 만들 수 있는 수가 적다
# (세트, 기대 보상, 최소 일치율). 유사도로 갈리지 않는 세트 (P_reorder · N_drop · N_target · N_verb) 가 핵심이다.
CRITERIA = [("P_fast", 1.0, 1.00), ("P_redirect", 1.0, 0.90), ("P_reorder", 1.0, 0.80), ("P_split", 1.0, 0.85), ("P_flag", 1.0, 0.85),
            ("P_sem", 1.0, 0.75), ("P_wredir", 1.0, 0.80), ("P_pyfile", 1.0, 0.80),
            ("N_wait", 0.0, 0.99), ("N_other", 0.0, 0.99), ("N_repeat", 0.0, 0.99), ("N_goldwait", 0.0, 0.99),
            ("N_drop", 0.0, 0.85), ("N_target", 0.0, 0.90), ("N_verb", 0.0, 0.95),
            ("N_wtrunc", 0.0, 0.85), ("N_sed", 0.0, 0.85), ("N_noexec", 0.0, 0.85), ("N_fastsed", 0.0, 0.85), ("N_earlytc", 0.0, 1.00)]
# judge 를 거치지 않고 판정돼야 하는 세트: (기대 실패 사유)
NO_JUDGE = {"P_fast": "none", "N_earlytc": "task_complete_check_failed"}
FAST_PASS = 1.0   # alpha_terminus_judge 의 빠른 통과 임계 (결정 66) — keystrokes 가 완전히 같을 때만
MAX_PARSE_FAIL = 0.02
MAX_ERROR_RATE = 0.01
MAX_FLIP = 0.02          # 전 쌍
MAX_FLIP_HARD = 0.05     # 유사도로 갈리지 않는 세트 각각
EXPECTED_CFG = {"enable_string_similarity": True, "string_similarity_threshold": FAST_PASS, "enable_llm_judge": True, "check_twice_swap": False}
READ_VERBS = ("cat ", "ls ", "head ", "tail ", "wc ", "stat ", "file ")


def keystrokes(d: dict) -> list[str]:
    return [c.get("keystrokes", "") for c in d.get("commands", []) if isinstance(c, dict) and "keystrokes" in c]


def sim(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, "".join(a), "".join(b)).ratio()


def with_commands(gold: dict, commands: list[dict], **extra) -> str:
    d = dict(gold)
    d["commands"] = commands
    d.update(extra)
    return json.dumps(d, ensure_ascii=False, indent=2)


def simple_read(k: str) -> bool:
    b = k.strip()
    return k.endswith("\n") and "\n" not in b and b.startswith(READ_VERBS) and not any(x in b for x in ("|", ">", "<", ";", "&", "$(", "`"))


def build(args) -> None:
    rng = random.Random(args.seed)
    rows = []
    with open(SOURCE) as f:
        for line in f:
            r = json.loads(line)
            g = json.loads(r["expected_answer"])
            r["_gold"], r["_ks"] = g, keystrokes(g)
            rows.append(r)
    rng.shuffle(rows)
    real = [r for r in rows if not r["_gold"].get("task_complete") and any(k.strip() for k in r["_ks"])]
    wait = [r for r in rows if not r["_gold"].get("task_complete") and r["_ks"] and all(not k.strip() for k in r["_ks"])]
    paths = [m for r in real[:4000] for k in r["_ks"] for m in re.findall(r"(?<![\w.])/[\w./-]{6,}", k)]
    pairs, n = [], collections.Counter()

    def add(set_name, label, row, cand_text, note=""):
        cand = json.loads(cand_text) if cand_text.strip().startswith("{") else {}
        s = sim(row["_ks"], keystrokes(cand)) if isinstance(cand, dict) else None
        pairs.append({"set": set_name, "label": label, "candidate_text": cand_text, "note": note, "sim": s,
                      "row": {k: row[k] for k in ("uuid", "task_name", "responses_create_params", "expected_answer", "metadata")}})
        n[set_name] += 1

    def other_task(row, pool):
        for _ in range(300):
            o = pool[rng.randrange(len(pool))]
            if o["task_name"] != row["task_name"] and sim(o["_ks"], row["_ks"]) < 0.5:
                return o
        return None

    def full(name, cap):
        return n[name] >= cap

    for r in real:
        g, ks, cmds = r["_gold"], r["_ks"], r["_gold"]["commands"]
        # --- 양성 ---
        # --- 쓰기 · 실행 명령 ---
        hd = [re.match(r"cat > (\S+) <<\s*('?)(\w+)\2\n", k) for k in ks]
        hi = next((i for i, m in enumerate(hd) if m and ks[i].rstrip("\n").endswith("\n" + m.group(3))), None)   # 첫 heredoc 쓰기 명령
        if not full("P_wredir", args.n_small) and hi is not None and n["N_wtrunc"] >= n["P_wredir"]:
            m = hd[hi]
            k2 = f"cat << {m.group(2)}{m.group(3)}{m.group(2)} > {m.group(1)}\n" + ks[hi][m.end():]
            new = [dict(c, keystrokes=k2) if j == hi else c for j, c in enumerate(cmds)]
            if sim(ks, [c["keystrokes"] for c in new]) < 0.95:
                add("P_wredir", 1, r, with_commands(g, new)); continue
        py = re.match(r"python3 (?:- )?<<\s*('?)(\w+)\1\n", ks[0]) if len(cmds) == 1 else None
        if not full("P_pyfile", args.n_small) and py and ks[0].rstrip("\n").endswith("\n" + py.group(2)):
            body = ks[0][py.end():]
            new = [dict(cmds[0], keystrokes=f"cat > /tmp/run_step.py << {py.group(1)}{py.group(2)}{py.group(1)}\n" + body),
                   dict(cmds[0], keystrokes="python3 /tmp/run_step.py\n")]
            if sim(ks, [c["keystrokes"] for c in new]) < 0.95:
                add("P_pyfile", 1, r, with_commands(g, new)); continue
        if not full("N_wtrunc", args.n_small) and hi is not None:
            m = hd[hi]
            lines = ks[hi][m.end():].rstrip("\n").split("\n")[:-1]
            if len(lines) >= 9:
                k2 = ks[hi][: m.end()] + "\n".join(lines[: len(lines) * 2 // 3]) + "\n" + m.group(3) + "\n"
                new = [dict(c, keystrokes=k2) if j == hi else c for j, c in enumerate(cmds)]
                if 0.5 <= sim(ks, [c["keystrokes"] for c in new]) < 0.95:
                    add("N_wtrunc", 0, r, with_commands(g, new), f"{len(lines) * 2 // 3}/{len(lines)} lines"); continue
        sd = [(i, re.search(r"sed -i(?:\.\w+)? (['\"])s([/|#])([^/|#\n]+)\2([^/|#\n]{3,})\2(g?)\1", k)) for i, k in enumerate(ks)]
        sd = [(i, m) for i, m in sd if m]
        if sd and (not full("N_sed", args.n_small) or not full("N_fastsed", args.n_small)):
            i, m = sd[0]
            k2 = ks[i][: m.start(4)] + m.group(4)[::-1] + "_alt" + ks[i][m.end(4):]
            new = [dict(c, keystrokes=k2) if j == i else c for j, c in enumerate(cmds)]
            sv = sim(ks, [c["keystrokes"] for c in new])
            if sv < 0.95 and not full("N_sed", args.n_small):
                add("N_sed", 0, r, with_commands(g, new)); continue
            if sv >= 0.95 and not full("N_fastsed", args.n_small):
                add("N_fastsed", 0, r, with_commands(g, new)); continue
        if not full("N_noexec", args.n_small) and len(cmds) >= 2 and any(hd):
            run = [bool(re.match(r"(python3?|bash|sh|node|make|pytest|go run|cargo run|\./)\S*", k.strip())) and not h for k, h in zip(ks, hd)]
            if any(run) and not all(run):
                new = [c for c, x in zip(cmds, run) if not x]
                if 0.3 <= sim(ks, [c["keystrokes"] for c in new]) < 0.95:
                    add("N_noexec", 0, r, with_commands(g, new), f"removed {sum(run)} exec"); continue
        if not full("K_ctrl", args.n_small) and any(k.strip() in ("C-c", "C-d", "C-z") for k in ks):
            new = [dict(c, keystrokes=(c["keystrokes"].strip() if c["keystrokes"].endswith("\n") else c["keystrokes"] + "\n"))
                   if c.get("keystrokes", "").strip() in ("C-c", "C-d", "C-z") else c for c in cmds]
            if sim(ks, [c["keystrokes"] for c in new]) < 0.95:
                add("K_ctrl", None, r, with_commands(g, new)); continue
        if not full("P_flag", args.n_small):
            new, changed = [], False
            for c in cmds:
                k = c.get("keystrokes", "")
                k2 = re.sub(r"(^|\s)-la(\s)", (r"\1-al\2" if n["P_flag"] % 2 else r"\1-l -a\2"), k, count=1)
                if k2 == k:
                    k2 = re.sub(r"(^|\s)-rn(\s)", r"\1-nr\2", k, count=1)
                changed |= k2 != k
                new.append(dict(c, keystrokes=k2))
            if changed and sim(ks, [c["keystrokes"] for c in new]) < 0.95:
                add("P_flag", 1, r, with_commands(g, new)); continue
        if not full("P_sem", 4 * args.n_sem_kind):
            new, kind = None, ""
            if len(cmds) == 1:
                k = ks[0]
                m = re.fullmatch(r"grep ((?:-[nicvEFw]+ )*)('[^']+'|\"[^\"$`]+\"|[\w.:=/-]+) (/?[\w./-]+)\n", k)
                h = re.fullmatch(r"head -n ?(\d+) (/?[\w./-]+)\n", k) or re.fullmatch(r"head -(\d+) (/?[\w./-]+)\n", k)
                d = re.search(r"<<\s*'?(\w+)'?\n", k)
                if m and "-r" not in m.group(1):
                    new, kind = [dict(cmds[0], keystrokes=f"cat {m.group(3)} | grep {m.group(1)}{m.group(2)}\n")], "grep"
                elif h:
                    new, kind = [dict(cmds[0], keystrokes=f"sed -n '1,{h.group(1)}p' {h.group(2)}\n")], "head"
                elif d and k.count(d.group(1)) == 2 and len(d.group(1)) >= 3:
                    new, kind = [dict(cmds[0], keystrokes=k.replace(d.group(1), "END_OF_BLOCK"))], "heredoc"
            elif 2 <= len(cmds) <= 4 and all(re.fullmatch(r"cat (/?[\w./-]+)\n", k) for k in ks) and len(set(ks)) == len(ks):
                new, kind = [dict(cmds[0], keystrokes="cat " + " ".join(k.strip()[4:] for k in ks) + "\n")], "catjoin"
            if new is not None and [c["keystrokes"] for c in new] != ks and sim(ks, [c["keystrokes"] for c in new]) < 0.95:
                if n["sem_" + kind] < args.n_sem_kind:   # 한 모양이 세트를 독점하지 않게
                    n["sem_" + kind] += 1
                    add("P_sem", 1, r, with_commands(g, new), kind); n["P_sem"] += 0; continue
        if not full("P_reorder", args.n) and len(cmds) >= 3 and all(simple_read(k) for k in ks) and len(set(ks)) == len(ks):
            new = list(reversed(cmds))
            if sim(ks, [c["keystrokes"] for c in new]) < 0.7:
                add("P_reorder", 1, r, with_commands(g, new)); continue
        if not full("P_split", args.n):
            new = None
            if len(cmds) == 1 and ks[0].endswith("\n") and ks[0].count(" && ") == 1 and "\n" not in ks[0].strip() and not any(x in ks[0] for x in ("|", "(", "<<", ";")):
                a, b = ks[0].rstrip("\n").split(" && ")
                if not a.strip().startswith("cd "):
                    new = [dict(cmds[0], keystrokes=a + "\n"), dict(cmds[0], keystrokes=b + "\n")]
            elif len(cmds) == 2 and all(simple_read(k) for k in ks):
                new = [dict(cmds[0], keystrokes=ks[0].rstrip("\n") + " && " + ks[1])]
            if new is not None and sim(ks, [c["keystrokes"] for c in new]) < 0.95:
                add("P_split", 1, r, with_commands(g, new)); continue
        if not full("P_fast", args.n_small):
            add("P_fast", 1, r, with_commands(g, [dict(c, duration=float(c.get("duration", 1.0)) + 1.0) for c in cmds])); continue
        if not full("P_redirect", args.n):
            new = []
            for c in cmds:
                k = c.get("keystrokes", "")
                body = k[:-1] if k.endswith("\n") else k
                ok = body.strip() and "\n" not in body and ">" not in body and "<<" not in body and "&" not in body and not body.strip().startswith("C-")
                new.append(dict(c, keystrokes=(body + " 2>&1" + ("\n" if k.endswith("\n") else "")) if ok else k))
            ck = [c["keystrokes"] for c in new]
            if ck != ks and sim(ks, ck) < 0.95:
                add("P_redirect", 1, r, with_commands(g, new)); continue
        # --- 음성: 유사도로 갈리지 않는 것 ---
        if not full("N_drop", args.n) and len(cmds) >= 3 and len(set(ks)) == len(ks):
            new = cmds[: max(1, len(cmds) // 3)]   # 2/3 이상을 빼 rubric 의 ">50% 범위 차이" 경계를 피한다
            if 0.3 <= sim(ks, [c["keystrokes"] for c in new]) < 0.95:
                add("N_drop", 0, r, with_commands(g, new), f"{len(new)}/{len(cmds)} commands"); continue
        if not full("N_target", args.n) and len(cmds) <= 2 and all(simple_read(k) for k in ks):
            new, changed = [], 0
            for c in cmds:
                k = c["keystrokes"]
                m = list(re.finditer(r"(?<![\w.])/[\w./-]{6,}", k))
                if m:
                    p = paths[rng.randrange(len(paths))]
                    if p.rstrip("/") != m[-1].group(0).rstrip("/") and p.rstrip("/").split("/")[-1] != m[-1].group(0).rstrip("/").split("/")[-1]:
                        k = k[: m[-1].start()] + p + k[m[-1].end():]
                        changed += 1
                new.append(dict(c, keystrokes=k))
            if changed == len(cmds) and 0.5 <= sim(ks, [c["keystrokes"] for c in new]) < 0.95:  # 모든 명령의 대상이 바뀐 쌍만
                add("N_target", 0, r, with_commands(g, new)); continue
        if not full("N_verb", args.n) and len(cmds) <= 2 and all(simple_read(k) and k.strip().startswith(("cat ", "head ", "tail ")) for k in ks):
            new = [dict(c, keystrokes="rm -f " + c["keystrokes"].strip().split(" ")[-1] + "\n") for c in cmds]
            if 0.5 <= sim(ks, [c["keystrokes"] for c in new]) < 0.95:
                add("N_verb", 0, r, with_commands(g, new)); continue
        # --- 음성: 쉬운 것 ---
        if not full("N_wait", args.n):
            add("N_wait", 0, r, with_commands(g, [{"keystrokes": "", "duration": 10.0}])); continue
        if not full("N_other", args.n):
            o = other_task(r, real)
            if o is not None:
                add("N_other", 0, r, with_commands(g, o["_gold"]["commands"]), f"other={o['task_name']}"); continue
        if not full("N_repeat", args.n):
            prev = [m for m in r["responses_create_params"]["input"] if m.get("role") == "assistant"]
            if prev and isinstance(prev[-1].get("content"), str):
                try:
                    p = json.loads(prev[-1]["content"])
                except json.JSONDecodeError:
                    p = None
                pth = lambda xs: set(re.findall(r"(?<![\w.])/[\w./-]{4,}", " ".join(xs)))  # noqa: E731
                same_target = pth(ks) & pth(keystrokes(p)) if isinstance(p, dict) else set()
                # 직전 턴이 정답과 같은 경로를 다루면 실제로 같은 일일 수 있다 (v4 오탐 9 건 중 약 절반) — 라벨이 애매해 뺀다
                if isinstance(p, dict) and any(k.strip() for k in keystrokes(p)) and sim(ks, keystrokes(p)) < 0.5 and not same_target \
                        and pth(ks) and pth(keystrokes(p)):
                    add("N_repeat", 0, r, with_commands(g, p["commands"])); continue
        if not full("N_earlytc", args.n_small):
            add("N_earlytc", 0, r, with_commands(g, cmds, task_complete=True)); continue
    for src, dst in (("N_drop", "B_drop"), ("N_verb", "B_verb"), ("P_reorder", "B_reorder")):
        for p in [q for q in pairs if q["set"] == src][: args.n_small]:
            c = json.loads(p["candidate_text"])
            c["analysis"], c["plan"] = "Continuing with the task.", "Run the next commands."
            pairs.append({**p, "set": dst, "label": None, "candidate_text": json.dumps(c, ensure_ascii=False, indent=2), "note": f"blank of {src}"})
            n[dst] += 1
    for r in wait:
        if full("N_goldwait", args.n):
            break
        o = other_task(r, real)
        if o is not None:
            add("N_goldwait", 0, r, with_commands(r["_gold"], o["_gold"]["commands"]), f"other={o['task_name']}")
    if args.profile_rollouts:
        by_uuid = {r["uuid"]: r for r in rows}
        bands = {"U_10_30": (0.1, 0.3), "U_30_50": (0.3, 0.5), "U_50_70": (0.5, 0.7), "U_70_95": (0.7, 0.95)}
        cand = collections.defaultdict(list)
        with open(args.profile_rollouts) as f:
            for line in f:
                x = json.loads(line)
                s_ = x.get("similarity_score")
                if not isinstance(s_, (int, float)) or not x.get("task_complete_check_passed") or x["uuid"] not in by_uuid:
                    continue
                for name, (lo, hi) in bands.items():
                    if lo <= s_ < hi:
                        cand[name].append((x["uuid"], x["model_output"], s_))
        for name in bands:
            rng.shuffle(cand[name])
            for u, text, s_ in cand[name][: args.n_unlabeled]:
                add(name, None, by_uuid[u], text, f"sim={s_:.3f}")
    with open(args.out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    short = [s for s in POS + NEG if n[s] < (30 if s in WRITE else (2 * args.n_sem_kind if s == "P_sem" else args.n_small if s in SMALL else args.n) * 0.5)]
    stats = {}
    for s in POS + NEG:
        v = sorted(p["sim"] for p in pairs if p["set"] == s and p["sim"] is not None)
        if v:
            stats[s] = {"n": len(v), "sim_min": round(v[0], 3), "sim_med": round(v[len(v) // 2], 3), "sim_max": round(v[-1], 3)}
    print(json.dumps({"pairs": len(pairs), "by_set": n, "sim": stats, "short_sets": short, "seed": args.seed}, ensure_ascii=False, indent=1))
    if short:
        raise SystemExit(f"세트가 목표의 절반에 못 미친다: {short}")


def judge_verdict(by: dict, parse_fail_rate: float, error_rate: float, truncated: int) -> list[str]:
    """세트별 결과 (reward 목록 · judge 호출 수 · 음성 실패 사유) 로 기준 위반 목록을 만든다. selftest 와 run 이 같이 쓴다."""
    fails = []
    for name, want, min_rate in CRITERIA:
        xs = by.get(name)
        if not xs:
            fails.append(f"{name}: 쌍 없음")
            continue
        if name in NO_JUDGE:   # judge 없이 정해져야 하는 세트
            ok = sum(1 for x in xs if x["reward"] == want and x["failure_reason"] == NO_JUDGE[name])
            if any(x["judge_calls"] for x in xs):
                fails.append(f"{name}: judge 가 호출된 쌍 {sum(1 for x in xs if x['judge_calls'])}")
        elif want == 0.0:  # 음성은 judge 가 "다르다" 고 한 것만 맞음으로 센다 (파싱 실패 · 오류의 보상 0 은 맞음이 아니다)
            ok = sum(1 for x in xs if x["reward"] == 0.0 and x["failure_reason"] == "judge_not_equivalent")
        else:
            ok = sum(1 for x in xs if x["reward"] == 1.0)
        if ok / len(xs) < min_rate:
            fails.append(f"{name}: 일치율 {ok / len(xs):.3f} < {min_rate}")
        if name not in NO_JUDGE and any(x["judge_calls"] != 1 for x in xs):
            fails.append(f"{name}: judge 호출이 1 이 아닌 쌍 {sum(1 for x in xs if x['judge_calls'] != 1)}")
    if parse_fail_rate > MAX_PARSE_FAIL:
        fails.append(f"판정 파싱 실패율 {parse_fail_rate:.3f} > {MAX_PARSE_FAIL}")
    if error_rate > MAX_ERROR_RATE:
        fails.append(f"예외율 {error_rate:.3f} > {MAX_ERROR_RATE}")
    if truncated:
        fails.append(f"판정문이 출력 상한에서 잘린 호출 {truncated}")
    return fails


def _canon(ks: list[str]) -> list[str]:
    """생성 변형을 되돌리는 글자 정규화: && · 개행으로 쪼개기, 끝의 ` 2>&1` 제거, 플래그 글자 정렬."""
    out = []
    for k in ks:
        for part in re.split(r"\s*&&\s*|\n", k):
            part = re.sub(r"\s*2>&1\s*$", "", part.strip())
            if not part:
                continue
            toks = []
            for t in part.split():
                toks.append("-" + "".join(sorted(t[1:])) if re.fullmatch(r"-[A-Za-z]{2,}", t) else t)
            out.append(" ".join(toks))
    return out


def selftest(args, quiet: bool = False) -> int:
    """값싼 가짜 judge 가 이 게이트에서 FAIL 하는지 본다 (CPU): 유사도 임계 · 글자 정규화 일치 · 항상 같음 · 항상 다름."""
    pairs = [json.loads(line) for line in open(args.pairs)]

    def jac(a, b):
        A, B = set(" ".join(a).split()), set(" ".join(b).split())
        return len(A & B) / max(1, len(A | B))

    fakes = {f"sim>={t}": (lambda s, g, c, t=t: s >= t) for t in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9)}
    fakes["always_equal"] = lambda s, g, c: True
    fakes["always_not_equal"] = lambda s, g, c: False
    fakes["canon_set_eq"] = lambda s, g, c: set(_canon(g)) == set(_canon(c))
    fakes["canon_jaccard>=0.9"] = lambda s, g, c: jac(_canon(g), _canon(c)) >= 0.9
    fakes["canon_jaccard>=0.7"] = lambda s, g, c: jac(_canon(g), _canon(c)) >= 0.7
    fakes["gold_cmds_subset"] = lambda s, g, c: all(any(x.strip() in y for y in c) for x in g if x.strip())
    bad = []
    for name, fn in fakes.items():
        by = collections.defaultdict(list)
        for p in pairs:
            if p["label"] is None or p["sim"] is None:
                continue
            gd, cd = json.loads(p["row"]["expected_answer"]), json.loads(p["candidate_text"])
            if cd.get("task_complete") and not gd.get("task_complete"):   # 플러그인의 완료 선언 검사 (judge 이전)
                by[p["set"]].append({"reward": 0.0, "failure_reason": "task_complete_check_failed", "judge_calls": 0})
                continue
            if p["sim"] >= FAST_PASS:
                by[p["set"]].append({"reward": 1.0, "failure_reason": "none", "judge_calls": 0})
                continue
            g, c = keystrokes(gd), keystrokes(cd)
            eq = fn(p["sim"], g, c)
            by[p["set"]].append({"reward": 1.0 if eq else 0.0, "failure_reason": "none" if eq else "judge_not_equivalent", "judge_calls": 1})
        fails = judge_verdict(by, 0.0, 0.0, 0)
        if not quiet:
            print(f"{name:20s} -> {'PASS (게이트가 못 잡음)' if not fails else 'FAIL'}  {'; '.join(fails[:4])}")
        if not fails:
            bad.append(name)
    print("selftest:", "OK — 가짜 judge 전부 FAIL" if not bad else f"NG — PASS 한 가짜 judge {bad}")
    return 1 if bad else 0


def load_server(args):
    sys.path.insert(0, args.gym_root)
    from omegaconf import OmegaConf

    import nemo_gym  # noqa: F401 — NEMO_GYM_EXTRA_ROOTS 의 플러그인 루트를 sys.path 에 올린다
    from nemo_gym.config_types import BaseServerConfig
    from nemo_gym.server_utils import setup_server_client
    from resources_servers.alpha_terminus_judge.app import (
        AlphaTerminusJudgeResourcesServer as TerminusJudgeResourcesServer,
        AlphaTerminusJudgeResourcesServerConfig as TerminusJudgeResourcesServerConfig,
    )

    plug = Path(__file__).resolve().parents[1] / "gym_plugins" / "resources_servers" / "alpha_terminus_judge"
    base = OmegaConf.load(plug / "configs" / "alpha_terminus_judge.yaml")["alpha_terminus_judge_resources_server"]["resources_servers"]["alpha_terminus_judge"]
    recipe = OmegaConf.load(args.recipe)
    over = recipe.env.nemo_gym.get("alpha_terminus_judge_resources_server", {}).get("resources_servers", {}).get("alpha_terminus_judge", {})
    merged = OmegaConf.to_container(OmegaConf.merge(base, over), resolve=True)
    merged.update(host="127.0.0.1", port=1, entrypoint="app.py", name="alpha_terminus_judge")
    cfg = TerminusJudgeResourcesServerConfig.model_validate(merged)
    for k, v in EXPECTED_CFG.items():
        if getattr(cfg, k) != v:
            raise SystemExit(f"병합된 설정이 기대와 다르다: {k}={getattr(cfg, k)!r} (기대 {v!r})")
    jp = cfg.judge_responses_create_params
    if jp.temperature != 0.0 or jp.max_output_tokens != args.max_output_tokens:
        raise SystemExit(f"judge 생성 설정이 기대와 다르다: temperature={jp.temperature} max_output_tokens={jp.max_output_tokens}")
    host, port = args.head.split(":")
    sc = setup_server_client(BaseServerConfig(host=host, port=int(port)))
    return TerminusJudgeResourcesServer(config=cfg, server_client=sc), cfg


async def run_pairs(args) -> int:
    from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseOutputMessage, NeMoGymResponseOutputText
    from resources_servers.terminus_judge.app import TerminusJudgeVerifyRequest

    if selftest(args, quiet=True):
        raise SystemExit("selftest NG — 쌍 파일이 값싼 가짜 judge 를 걸러내지 못한다. run 을 하지 않는다")
    server, cfg = load_server(args)
    print("terminus_judge cfg:", cfg.model_dump(include={"judge_model_server", "judge_responses_create_params", "enable_string_similarity",
          "string_similarity_threshold", "enable_llm_judge", "check_twice_swap", "judge_prompt_template_fpath"}), flush=True)
    pairs = [json.loads(line) for line in open(args.pairs)]
    sem = asyncio.Semaphore(args.concurrency)

    def response(text):
        return NeMoGymResponse(id="resp", created_at=0.0, model="policy", object="response", parallel_tool_calls=False,
                               tool_choice="none", tools=[], output=[NeMoGymResponseOutputMessage(
                                   id="msg", role="assistant", status="completed", type="message",
                                   content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")])])

    async def one(i, p):
        t0 = time.time()
        r = p["row"]
        base = {"i": i, "set": p["set"], "label": p["label"], "uuid": r["uuid"], "note": p.get("note"), "sim_build": p.get("sim")}
        try:
            req = TerminusJudgeVerifyRequest.model_validate({**r, "response": response(p["candidate_text"]).model_dump()})
            async with sem:
                vr = await server.verify(req)
        except Exception as e:  # 한 쌍의 예외가 전체를 멈추지 않게 세고 넘어간다 — 예외율이 기준이다
            return {**base, "error": f"{type(e).__name__}: {str(e)[:200]}", "reward": None, "failure_reason": "exception", "judge_calls": 0,
                    "verdicts": [], "parse_fail": [], "out_tokens": [], "judge_texts": [], "latency_s": round(time.time() - t0, 2)}
        evs = list(vr.judge_evaluations)
        texts = []
        for e in evs:
            try:
                texts.append(e.response.output[-1].content[-1].text)
            except Exception:
                texts.append("")
        return {**base, "error": None, "reward": vr.reward, "failure_reason": str(vr.failure_reason.value if vr.failure_reason else None),
                "similarity": vr.similarity_score, "judge_calls": len(evs), "verdicts": [e.verdict_label for e in evs],
                "parse_fail": [e.verdict_label is None for e in evs],
                "out_tokens": [int(getattr(getattr(e.response, "usage", None), "output_tokens", 0) or 0) for e in evs],
                "judge_texts": texts, "latency_s": round(time.time() - t0, 2)}

    t0 = time.time()
    results = await asyncio.gather(*[one(i, p) for i, p in enumerate(pairs)])
    wall = time.time() - t0
    with open(args.out, "w") as f:
        for x in results:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    by = collections.defaultdict(list)
    for x in results:
        by[x["set"]].append(x)
    summ = {}
    for name, xs in sorted(by.items()):
        calls = sum(x["judge_calls"] for x in xs)
        summ[name] = {"n": len(xs), "label": xs[0]["label"], "reward_1_rate": round(sum(x["reward"] == 1.0 for x in xs) / len(xs), 4),
                      "judge_calls": calls, "parse_fail": sum(sum(x["parse_fail"]) for x in xs), "errors": sum(1 for x in xs if x["error"]),
                      "failure_reasons": collections.Counter(x["failure_reason"] for x in xs),
                      "out_tokens_mean": round(sum(sum(x["out_tokens"]) for x in xs) / max(1, calls), 1),
                      "latency_med_s": sorted(x["latency_s"] for x in xs)[len(xs) // 2]}
    calls = sum(x["judge_calls"] for x in results)
    pf = sum(sum(x["parse_fail"]) for x in results)
    errs = sum(1 for x in results if x["error"])
    trunc = sum(1 for x in results for t in x["out_tokens"] if t >= args.max_output_tokens)
    fails = judge_verdict({k: v for k, v in by.items() if v[0]["label"] is not None}, pf / max(1, calls), errs / len(results), trunc)
    summ["_all"] = {"pairs": len(results), "judge_calls": calls, "parse_fail": pf, "parse_fail_rate": round(pf / max(1, calls), 4),
                    "errors": errs, "truncated_judge_outputs": trunc, "wall_s": round(wall, 1), "judge_calls_per_s": round(calls / max(wall, 1e-9), 1),
                    "concurrency": args.concurrency, "verdict": "PASS" if not fails else "FAIL", "fails": fails}
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    print(json.dumps(summ, indent=1, ensure_ascii=False))
    return 0 if not fails else 1


def flips(args) -> int:
    a = {x["i"]: x for x in map(json.loads, open(args.a))}
    b = {x["i"]: x for x in map(json.loads, open(args.b))}
    both = [i for i in a if i in b and a[i]["judge_calls"] and b[i]["judge_calls"]]
    fl = [i for i in both if a[i]["reward"] != b[i]["reward"]]
    tot = collections.Counter(a[i]["set"] for i in both)
    by = collections.Counter(a[i]["set"] for i in fl)
    rate = len(fl) / max(1, len(both))
    per = {k: round(by[k] / tot[k], 4) for k in sorted(tot)}
    fails = [f"전체 {rate:.4f} > {MAX_FLIP}"] if rate > MAX_FLIP else []
    fails += [f"{k} {per[k]} > {MAX_FLIP_HARD}" for k in per if (k in HARD or k.startswith("U_")) and per[k] > MAX_FLIP_HARD]
    print(json.dumps({"judged_pairs": len(both), "flips": len(fl), "flip_rate": round(rate, 4), "per_set": per,
                      "verdict": "PASS" if not fails else "FAIL", "fails": fails}, ensure_ascii=False))
    return 0 if not fails else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--profile-rollouts")
    b.add_argument("--n", type=int, default=300)
    b.add_argument("--n-small", type=int, default=100)
    b.add_argument("--n-sem-kind", type=int, default=50, help="P_sem 의 모양 (grep · head · heredoc · catjoin) 하나당 상한")
    b.add_argument("--n-unlabeled", type=int, default=100)
    b.add_argument("--seed", type=int, default=20261116)
    s = sub.add_parser("selftest")
    s.add_argument("--pairs", required=True)
    r = sub.add_parser("run")
    r.add_argument("--gym-root", required=True)
    r.add_argument("--recipe", required=True)
    r.add_argument("--head", required=True)
    r.add_argument("--pairs", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--concurrency", type=int, default=64)
    r.add_argument("--max-output-tokens", type=int, default=2048)
    fl = sub.add_parser("flips")
    fl.add_argument("--a", required=True)
    fl.add_argument("--b", required=True)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a)
        return 0
    if a.cmd == "selftest":
        return selftest(a)
    if a.cmd == "flips":
        return flips(a)
    return asyncio.run(run_pairs(a))


if __name__ == "__main__":
    raise SystemExit(main())
