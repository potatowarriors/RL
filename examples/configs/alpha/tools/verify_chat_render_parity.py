#!/usr/bin/env python3
"""verify_chat_render_parity.py — alpha RL 렌더 패리티 게이트 (CPU 전용, GPU·네트워크 불필요).

질문: NeMo-RL 이 alpha RL 프롬프트를 만들 때(네이티브 GRPO 데이터 프로세서, NeMo Gym → vLLM HTTP 서버 경로)
나오는 토큰 ID 가 SFT 변환기(build_alpha_sft_idxmap.py)가 학습에 쓴 토큰 ID 와 **정확히** 같은가?

기준(SFT, 정답): 시스템 python3 (transformers 4.57) 에서 변환기 함수 그대로
  normalize_row → expand_train_turns_fanout → render_and_mask
  학습 샘플에서 assistant 턴 k 의 "생성 시작점"(= `<|im_start|>assistant\\n<think>\\n` 또는 `<think></think>` 직후)
  앞까지가 ctx_k, 그 뒤 `<|im_end|>` 까지가 gen_k (모델이 내야 할 토큰).

비교 경로 (전부 실제 코드 호출, 서버·엔진 없이):
  P0  SFT 자기 일관성 — tf4.57 apply_chat_template(msgs[:k], add_generation_prompt=True) == ctx_k
  P1  NeMo-RL 네이티브 — driver venv(tf5.x): nemo_rl.algorithms.utils.get_tokenizer(chat_template_kwargs)
      + nemo_rl.data.processors.math_hf_data_processor (단일 턴) / 같은 호출 패턴(다중 메시지)
  P2  전체 대화 렌더 교차 버전 — tf5.x(driver) · tf5.x(vLLM venv, HF 직접) == SFT 전체 렌더
  P2r vLLM 메시지 파서에 SFT 메시지(reasoning_content 필드)를 그대로 넣은 렌더 (필드 처리 확인)
  P3  Gym → vLLM 템플릿 렌더 — Gym v0.6.0 VLLMModel.responses() (실제 변환기·전처리, 클라이언트만 mock)가
      보낸 chat 요청 본문을 vLLM 0.25.1 의 ChatCompletionRequest·parse_chat_messages·safe_apply_chat_template 로 렌더
  P4  P3 + NeMo-RL replace_prefix_tokens (현행 토크나이저 eos=<|endoftext|> 0)
  P5  P3 + replace_prefix_tokens, 턴 경계 토큰을 <|im_end|>(3) 로 바꾼 경우 (수정안 시뮬레이션)
  다중 턴 롤아웃은 "모델이 SFT 정답 gen_k 를 그대로 냈다" 고 가정하고 vLLM 파서(reasoning+tool)로 파싱해
  Gym 이 다음 요청을 만드는 과정을 턴마다 재생한다.

단계 (각 단계는 해당 인터프리터에서 실행, 결과는 --out 디렉토리의 JSON 으로 주고받는다):
  stage-sft     /usr/bin/python3                          (transformers 4.57, Pai 변환기)
  stage-nrl     NeMo-RL venv-driver                        (transformers 5.5.x, nemo_rl)
  stage-parse   NeMo-RL vLLM 워커 venv                     (vllm 0.25.1, transformers 5.8.x)
  stage-gym     Gym venv + NeMo-RL 서브모듈 Gym 소스        (nemo_gym 0.6.0)
  stage-render  NeMo-RL vLLM 워커 venv
  report        /usr/bin/python3
  run           위 전부를 순서대로 (clean_run.sh 경유, CUDA_VISIBLE_DEVICES="")

Usage:
  CUDA_VISIBLE_DEVICES="" python3 examples/configs/alpha/tools/verify_chat_render_parity.py run --out /tmp/render_gate
종료 코드: 0 = 전 비교 MATCH, 1 = MISMATCH 있음, 2 = 단계 실패.
"""
import argparse
import copy
import json
import os
import subprocess
import sys
import time

PAI = "/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch"
NRL = "/home/work/vidsearch/repos/project_s/NeMo-RL"
CKPT = (PAI + "/examples/alpha/outputs/alpha_baseline_48L_sft_128k_agentic_resume_20261001_011900/hfmodel_0002400")
TOK_V5 = PAI + "/examples/alpha/tokenizer_v5"
GYM_ROOT = NRL + "/3rdparty/Gym-workspace/Gym"
NRL_TOOLS = "/home/work/vidsearch/tools/nemo_rl"
CLEAN = NRL_TOOLS + "/clean_run.sh"
PY_SFT = "/usr/bin/python3"
PY_DRV = NRL_TOOLS + "/venv-driver/bin/python"
PY_VLLM = NRL_TOOLS + "/venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python"
PY_GYM = "/home/work/vidsearch/tools/Gym/.venv/bin/python"
KOTOOL = PAI + "/examples/alpha/sdg/kotool/out/export/kotool_v1.jsonl"
AGENTIC_IA = "/home/work/Datasets/LL_datasets/posttraining/SFT/Nemotron-SFT-Agentic-v2/data/interactive_agent.jsonl"
RL_BLEND = "/home/work/Datasets/LL_datasets/posttraining/RL/alpha_blends/rlvr1_alpha.jsonl"
NANO_V3_PLUGIN = NRL + "/nemo_rl/models/generation/vllm/reasoning_parsers/nano_v3_reasoning_parser.py"

IM_START, IM_END, THINK, THINK_END = 2, 3, 14, 15
SENTINEL = -1000            # Gym 단계에서 prompt_token_ids 자리표시 (render 단계가 실제 값으로 치환)
EFFORT = "\n\n{reasoning effort: efficient}"
# 파서 조합: A = 우리 eval fleet (serve_alpha.sh TOOLS=1), B = NeMo-RL Ultra Gym 레시피 (student_rlvr1.yaml 등)
PARSER_COMBOS = {"A_fleet": ("qwen3_xml", "nemotron_v3"), "B_ultra": ("qwen3_coder", "nano_v3")}
# Gym vllm_model chat_template_kwargs 변형: null = Pai gym/configs/alpha_vllm_model.yaml,
# ultra = Ultra 레시피 policy_model (enable_thinking + truncate_history_thinking=false)
GYM_VARIANTS = ("null", "ultra")


def jdump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, default=str)


def jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# 케이스 정의 (SFT 원시 행 형식 — 변환기 입력과 동일한 스키마)
# =============================================================================
def _tc(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


def synthetic_cases():
    weather_tools = [
        {"type": "function", "function": {
            "name": "get_weather", "description": "Get current weather for a city.",
            "parameters": {"type": "object", "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                "required": ["city"]},
            "strict": True}},
        # description 없는 도구: SFT 렌더는 <description> 줄이 없다 (vLLM FunctionDefinition 은 None 을 남긴다)
        {"type": "function", "function": {
            "name": "get_time",
            "parameters": {"type": "object", "properties": {"tz": {"type": "string"}}, "required": ["tz"]},
            "strict": True}},
    ]
    ko_tools = [{"type": "function", "function": {
        "name": "search_restaurant", "description": "지역과 음식 종류로 식당을 검색합니다.",
        "parameters": {"type": "object", "properties": {
            "region": {"type": "string", "description": "지역 이름 (예: 강남구)"},
            "cuisine": {"type": "string", "enum": ["한식", "중식", "일식"]}},
            "required": ["region"]}}}]
    c4_msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather in Seoul right now? Use celsius."},
        {"role": "assistant", "content": "",
         "reasoning_content": "The user wants the current weather in Seoul in celsius. I should call get_weather.\n",
         "tool_calls": [_tc("call_1", "get_weather", {"city": "Seoul", "unit": "celsius"})]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp_c\": 21, \"condition\": \"sunny\"}"},
        {"role": "assistant", "reasoning_content": "The tool says 21C and sunny. Answer concisely.",
         "content": "It's currently 21°C and sunny in Seoul."},
    ]
    c4b_msgs = c4_msgs + [
        {"role": "user", "content": "And in Busan?"},
        {"role": "assistant", "content": "Let me check Busan.",
         "reasoning_content": "Same tool, different city.",
         "tool_calls": [_tc("call_2", "get_weather", {"city": "Busan", "unit": "celsius"})]},
        {"role": "tool", "tool_call_id": "call_2", "content": "{\"temp_c\": 24, \"condition\": \"cloudy\"}"},
        {"role": "assistant", "reasoning_content": "Busan: 24C, cloudy.", "content": "Busan is 24°C and cloudy."},
    ]
    c3_msgs = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Give me a synonym for 'happy'."},
        {"role": "assistant", "reasoning_content": "A common synonym is 'joyful'.", "content": "Joyful."},
        {"role": "user", "content": "And an antonym?"},
        {"role": "assistant", "reasoning_content": "The antonym of happy is 'sad'.", "content": "Sad."},
    ]
    return [
        dict(name="c1_single_think", desc="단일 턴 user, thinking on",
             row={"messages": [{"role": "user", "content": "What is 17 * 23? Answer briefly."},
                               {"role": "assistant", "reasoning_content": "17 * 23 = 340 + 51 = 391.",
                                "content": "17 × 23 = **391**."}]},
             history="none"),
        dict(name="c2_single_nothink", desc="단일 턴, thinking off (enable_thinking=False)",
             row={"messages": [{"role": "user", "content": "Translate 'good morning' into French."},
                               {"role": "assistant", "content": "Bonjour."}]},
             history="none", think=False),
        dict(name="c2e_medium_effort", desc="단일 턴 + effort 마커 (SFT=medium_effort kwarg, RL=데이터 리터럴)",
             row={"messages": [{"role": "user", "content": "Name three primary colors."},
                               {"role": "assistant", "reasoning_content": "Red, yellow, blue.",
                                "content": "Red, yellow, and blue."}]},
             history="none", medium_effort=True),
        dict(name="c3a_chat_multiturn_dataset", desc="system+멀티턴 비도구, 이전 reasoning 있음 — 이력은 데이터셋 제공(strip 규칙)",
             row={"messages": c3_msgs}, history="dataset", fanout=True),
        dict(name="c3b_chat_multiturn_rollout", desc="같은 대화를 롤아웃 안에서 모델이 생성(이력 = 모델 토큰)",
             row={"messages": c3_msgs}, history="model", fanout=True),
        dict(name="c4_tool_interleaved", desc="tools + tool_call + tool 결과 + 최종 답 (restore 규칙)",
             row={"messages": c4_msgs, "tools": weather_tools}, history="model"),
        dict(name="c4b_tool_user_boundary", desc="tool 시나리오에서 user 턴 경계 넘어 reasoning 보존",
             row={"messages": c4b_msgs, "tools": weather_tools}, history="model"),
        dict(name="c5_korean_tool", desc="한국어 system/user/도구 설명·enum/인자/결과/답",
             row={"messages": [
                 {"role": "system", "content": "당신은 친절한 한국어 비서입니다."},
                 {"role": "user", "content": "강남역 근처에 괜찮은 한식집 추천해 줘."},
                 {"role": "assistant", "content": "",
                  "reasoning_content": "사용자가 강남역 근처 한식집을 원한다. search_restaurant 를 region=강남구, cuisine=한식 으로 호출하자.",
                  "tool_calls": [_tc("call_k1", "search_restaurant", {"region": "강남구", "cuisine": "한식"})]},
                 {"role": "tool", "tool_call_id": "call_k1",
                  "content": "[{\"name\": \"한옥마을\", \"rating\": 4.6}, {\"name\": \"소담\", \"rating\": 4.4}]"},
                 {"role": "assistant", "reasoning_content": "결과 두 곳. 평점 순으로 정리해서 답하자.",
                  "content": "강남역 근처 한식집으로는 **한옥마을**(평점 4.6)과 **소담**(4.4)을 추천드립니다."}],
                 "tools": ko_tools}, history="model"),
    ]


def real_sft_cases():
    out = []
    # kotool_v1 (한국어 tool-call SDG, 최종 블렌드 멤버): 인자가 비어있지 않은 첫 tool 행
    try:
        with open(KOTOOL, encoding="utf-8") as f:
            for i, line in enumerate(f):
                r = json.loads(line)
                tcs = [tc for m in r["messages"] for tc in (m.get("tool_calls") or [])]
                if any(m["role"] == "tool" for m in r["messages"]) and tcs and any(
                        json.loads(tc["function"]["arguments"] or "{}") for tc in tcs):
                    out.append(dict(name="c6a_real_kotool", desc=f"실제 SFT 행 kotool_v1.jsonl 행 {i}",
                                    row=r, history="model"))
                    break
    except OSError as e:
        print(f"[warn] kotool 미사용: {e}")
    # Agentic-v2 interactive_agent (agentic SFT 멤버 agentic_v2_ia): 멀티 user + tool 행
    try:
        with open(AGENTIC_IA, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i > 200:
                    break
                r = json.loads(line)
                roles = [m["role"] for m in r["messages"]]
                if roles.count("user") >= 2 and "tool" in roles and len(roles) <= 12:
                    out.append(dict(name="c6b_real_agentic_ia",
                                    desc=f"실제 SFT 행 Agentic-v2 interactive_agent.jsonl 행 {i}",
                                    row={k: r[k] for k in ("messages", "tools", "metadata") if k in r},
                                    history="model"))
                    break
    except OSError as e:
        print(f"[warn] agentic 미사용: {e}")
    return out


def responses_to_sft_row(rcp):
    """RL 블렌드(Responses 형식) 입력 → SFT 원시 행 + 마지막 더미 assistant(ctx 경계 정의용)."""
    msgs, cur = [], None

    def flush():
        nonlocal cur
        if cur is not None:
            cur.setdefault("content", "")
            msgs.append(cur)
            cur = None
    for it in rcp["input"]:
        t = it.get("type") or ("message" if it.get("role") else None)
        if t == "message" and it.get("role") != "assistant":
            flush()
            c = it["content"]
            if isinstance(c, list):
                c = "".join(p.get("text", "") for p in c)
            msgs.append({"role": it["role"], "content": c})
        elif t == "message":
            cur = cur or {"role": "assistant"}
            c = it["content"]
            if isinstance(c, list):
                c = "".join(p.get("text", "") for p in c)
            cur["content"] = cur.get("content", "") + c
        elif t == "reasoning":
            cur = cur or {"role": "assistant"}
            cur["reasoning_content"] = "".join(s["text"] for s in it.get("summary") or [])
        elif t == "function_call":
            cur = cur or {"role": "assistant"}
            cur.setdefault("tool_calls", []).append(
                {"id": it["call_id"], "type": "function", "function": {"name": it["name"], "arguments": it["arguments"]}})
        elif t == "function_call_output":
            flush()
            msgs.append({"role": "tool", "tool_call_id": it["call_id"], "content": it["output"]})
    flush()
    msgs.append({"role": "assistant", "reasoning_content": "Okay, let me think.", "content": "Done."})
    tools = None
    if rcp.get("tools"):
        tools = []
        for t in rcp["tools"]:
            fn = {k: t[k] for k in ("name", "description", "parameters", "strict") if k in t}
            tools.append({"type": "function", "function": fn})
    return {"messages": msgs, "tools": tools}


def real_rl_cases():
    out = []
    want = {"single_step_tool_use_with_argument_comparison_agent": "c6c_real_rl_single_step_tool",
            "multichallenge_simple_agent": "c6d_real_rl_multichallenge"}
    try:
        with open(RL_BLEND, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not want:
                    break
                r = json.loads(line)
                a = (r.get("agent_ref") or {}).get("name")
                if a in want:
                    rcp = r["responses_create_params"]
                    out.append(dict(name=want.pop(a), desc=f"실제 RL 블렌드 행 rlvr1_alpha.jsonl 행 {i} ({a}); "
                                    "SFT 측은 같은 대화를 SFT 스키마로 옮겨 더미 마지막 턴을 붙인 것",
                                    row=responses_to_sft_row(rcp), history="dataset",
                                    rl_input=rcp, fanout=not rcp.get("tools")))
    except OSError as e:
        print(f"[warn] RL 블렌드 미사용: {e}")
    return out


# =============================================================================
# stage-sft  (system python, transformers 4.57)
# =============================================================================
def stage_sft(args):
    sys.path.insert(0, PAI + "/toolkits/sft_data_preprocessing")
    import transformers
    from transformers import AutoTokenizer
    import build_alpha_sft_idxmap as B

    tok = AutoTokenizer.from_pretrained(TOK_V5)
    nl = tok("\n", add_special_tokens=False).input_ids
    assert len(nl) == 1, nl
    NL = nl[0]
    hdr_len = len(tok("assistant\n", add_special_tokens=False).input_ids)

    # 3 파일 동기화 + 체크포인트 바이트 동일
    import hashlib
    sync = {}
    for fn in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"):
        a = os.path.join(TOK_V5, fn)
        b = os.path.join(args.ckpt, fn)
        ha = hashlib.sha256(open(a, "rb").read()).hexdigest() if os.path.exists(a) else None
        hb = hashlib.sha256(open(b, "rb").read()).hexdigest() if os.path.exists(b) else None
        sync[fn] = {"tokenizer_v5": ha and ha[:16], "ckpt": hb and hb[:16], "equal": ha == hb}
    jinja = open(os.path.join(TOK_V5, "chat_template.jinja")).read()
    cfg_v5 = json.load(open(os.path.join(TOK_V5, "tokenizer_config.json")))
    cfg_ck = json.load(open(os.path.join(args.ckpt, "tokenizer_config.json")))
    sync["jinja==v5.tokenizer_config.chat_template"] = cfg_v5.get("chat_template") == jinja
    sync["jinja==ckpt.tokenizer_config.chat_template"] = cfg_ck.get("chat_template") == jinja
    gen_cfg = json.load(open(os.path.join(args.ckpt, "generation_config.json")))

    cases = synthetic_cases() + real_sft_cases() + real_rl_cases()
    out = {"meta": {"transformers": transformers.__version__, "NL": NL, "hdr_len": hdr_len,
                    "eos_token": tok.eos_token, "eos_token_id": tok.eos_token_id, "sync": sync,
                    "generation_config": gen_cfg}, "cases": []}
    for case in cases:
        think = case.get("think", True)
        norm, why = B.normalize_row(case["row"])
        assert norm is not None, (case["name"], why)
        if case.get("fanout"):
            subs = B.expand_train_turns_fanout(norm, implicit=True)
        else:
            subs = [norm]
        kw = dict(medium_effort=bool(case.get("medium_effort")), keep_history_think=False)
        encs = []
        for sub in subs:
            enc, why = B.render_and_mask(tok, sub, **kw)
            assert enc is not None, (case["name"], why)
            encs.append(enc)
        full_enc, why = B.render_and_mask(tok, norm, **kw)
        assert full_enc is not None, (case["name"], why)
        asst_idx = [i for i, m in enumerate(norm["messages"]) if m["role"] == "assistant"]
        targets_k = asst_idx if case["history"] == "model" else [asst_idx[-1]]
        targets = []
        for k in targets_k:
            si = next((j for j, s in enumerate(subs) if len(s["messages"]) == k + 1), 0) if len(subs) > 1 else 0
            ids = encs[si].ids.tolist()
            spans = [(s, e) for (s, e, role) in B.find_turn_spans(ids, tok) if role == "assistant"]
            ordinal = sum(1 for i in asst_idx if i < k)
            s, e = spans[ordinal]
            p = s + hdr_len
            opener = ("think" if ids[p:p + 2] == [THINK, NL] else
                      "nothink" if ids[p:p + 2] == [THINK, THINK_END] else "MERGED")
            gen_start = p + 2
            ctx = ids[:gen_start]
            gen = ids[gen_start:e]
            # P0: SFT 자기 일관성 (같은 tf4.57 템플릿의 generation prompt 렌더)
            tkw = {"enable_thinking": opener != "nothink"}
            if kw["medium_effort"]:
                tkw["medium_effort"] = True
            txt = tok.apply_chat_template(norm["messages"][:k], tools=norm["tools"], tokenize=False,
                                          add_generation_prompt=True, **tkw)
            p0 = tok(txt, add_special_tokens=False).input_ids
            targets.append({"k": k, "sub": si, "opener": opener, "ctx": ctx, "gen": gen,
                            "gen_ends_im_end": bool(gen and gen[-1] == IM_END),
                            "trained_gen_tokens": int(encs[si].trainable[gen_start - 1:e - 1].sum()),
                            "p0": p0})
        # RL 측 컨텍스트에서 쓸 메시지 (medium effort 는 RL 데이터처럼 리터럴 마커)
        rl_msgs = copy.deepcopy(norm["messages"])
        if case.get("medium_effort"):
            lu = max(i for i, m in enumerate(rl_msgs) if m["role"] == "user")
            rl_msgs[lu]["content"] += EFFORT
        out["cases"].append({
            "name": case["name"], "desc": case["desc"], "history": case["history"],
            "think": think, "medium_effort": bool(case.get("medium_effort")), "fanout": bool(case.get("fanout")),
            "n_subs": len(subs), "norm": norm, "rl_msgs": rl_msgs, "rl_input": case.get("rl_input"),
            "full_ids": full_enc.ids.tolist(), "targets": targets})
        print(f"[sft] {case['name']}: msgs={len(norm['messages'])} subs={len(subs)} "
              f"targets={[t['k'] for t in targets]} full={len(full_enc.ids)} tok")
    jdump(out, os.path.join(args.out, "sft.json"))


# =============================================================================
# stage-nrl  (NeMo-RL driver venv, transformers 5.x)
# =============================================================================
def stage_nrl(args):
    import tempfile
    import transformers
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.data.interfaces import TaskDataSpec
    from nemo_rl.data.processors import math_hf_data_processor

    sft = jload(os.path.join(args.out, "sft.json"))
    toks = {}

    def tok_for(think):
        key = bool(think)
        if key not in toks:
            # 레시피 키: policy.tokenizer.chat_template_kwargs (alpha smoke 는 null → 템플릿 기본 enable_thinking=True)
            ctk = None if think else {"enable_thinking": False}
            toks[key] = get_tokenizer({"name": args.ckpt, "chat_template_kwargs": ctk})
        return toks[key]

    res = {"meta": {"transformers": transformers.__version__}, "cases": {}}
    for c in sft["cases"]:
        tok = tok_for(c["think"])
        r = {"targets": {}}
        tools = c["norm"]["tools"]
        msgs = c["rl_msgs"]
        for t in c["targets"]:
            k = t["k"]
            ctx = msgs[:k]
            prior_asst = any(m["role"] == "assistant" for m in ctx)
            if c["history"] == "model" and prior_asst:
                continue  # 네이티브 GRPO 는 롤아웃 이력을 재렌더하지 않는다 (토큰 연결) — 해당 없음
            roles = [m["role"] for m in ctx]
            if not tools and roles in (["user"], ["system", "user"]):
                # 실제 프로세서: math_hf_data_processor (prompt_file=None → 원문 그대로)
                sys_file = None
                if roles[0] == "system":
                    fd, sys_file = tempfile.mkstemp(suffix=".txt")
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        f.write(ctx[0]["content"])
                spec = TaskDataSpec(task_name="math", prompt_file=None, system_prompt_file=sys_file)
                datum = {"messages": [{"role": "user", "content": ctx[-1]["content"]},
                                      {"role": "assistant", "content": "GT"}], "task_name": "math"}
                ds = math_hf_data_processor(datum, spec, tok, 1 << 20, 0)
                ids = [int(x) for m in ds["message_log"] for x in m["token_ids"].tolist()]
                how = "math_hf_data_processor"
            else:
                txt = tok.apply_chat_template(ctx, tools=tools, tokenize=False,
                                              add_generation_prompt=True, add_special_tokens=False)
                ids = tok(txt, add_special_tokens=False)["input_ids"]
                how = "apply_chat_template(processors.py 패턴)"
            r["targets"][str(k)] = {"ids": ids, "how": how}
        # P2: 전체 대화 렌더 (SFT 와 같은 메시지·kwargs, tf5.x)
        kw = {"medium_effort": True} if c["medium_effort"] else {}
        base = get_tokenizer({"name": args.ckpt})
        txt = base.apply_chat_template(c["norm"]["messages"], tools=tools, tokenize=False,
                                       add_generation_prompt=False, **kw)
        r["full_ids"] = base(txt, add_special_tokens=False)["input_ids"]
        if c["n_subs"] > 1:
            r["full_ids"] = None  # fan-out 셋은 단일 전체 렌더가 학습 샘플이 아님 → full 비교는 sub 단위 아님, 생략
            txt = base.apply_chat_template(c["norm"]["messages"], tools=tools, tokenize=False,
                                           add_generation_prompt=False, **kw)
            r["full_ids_nofanout"] = base(txt, add_special_tokens=False)["input_ids"]
        res["cases"][c["name"]] = r
        print(f"[nrl] {c['name']}: native targets={list(r['targets'])}")
    res["meta"]["eos_token_id"] = base.eos_token_id
    res["meta"]["pad_token_id"] = base.pad_token_id
    jdump(res, os.path.join(args.out, "nrl.json"))


# =============================================================================
# vLLM 공용 (stage-parse / stage-render)
# =============================================================================
class _StubModelConfig:
    """vLLM 렌더 함수가 텍스트 전용 경로에서 읽는 속성만 가진 ModelConfig 대역."""
    def __init__(self, model):
        from types import SimpleNamespace
        self.model = model
        self.trust_remote_code = True
        self.hf_config = SimpleNamespace(model_type="alpha")
        self.hf_text_config = self.hf_config
        self.hf_overrides = None
        self.allowed_local_media_path = ""   # ModelConfig 기본값
        self.allowed_media_domains = None
        self.multimodal_config = None
        self.is_multimodal_model = False
        self.enable_prompt_embeds = False
        self.max_model_len = 1 << 20


def _vllm_tokenizer(path):
    try:
        from vllm.tokenizers.registry import get_tokenizer
        return get_tokenizer(path, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] vllm get_tokenizer 실패 → AutoTokenizer ({e})")
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def _chat_tools(norm_tools):
    """SFT 정규화 도구 → OpenAI chat tools (그대로)."""
    return norm_tools or None


def _norm_to_wire(messages, reasoning_keys=("reasoning_content",)):
    """SFT 정규화 메시지 → OpenAI chat 와이어 형식 (tool_calls id/type, arguments JSON 문자열).
    reasoning 은 reasoning_keys 에 든 필드로만 싣는다 (vLLM 필드 처리 확인용)."""
    out = []
    for i, m in enumerate(messages):
        d = {"role": m["role"], "content": m.get("content", "")}
        if m.get("reasoning_content"):
            for key in reasoning_keys:
                d[key] = m["reasoning_content"]
        if m.get("tool_calls"):
            d["tool_calls"] = [{"id": f"call_{i}_{j}", "type": "function",
                                "function": {"name": tc["function"]["name"],
                                             "arguments": json.dumps(tc["function"]["arguments"], ensure_ascii=False)}}
                               for j, tc in enumerate(m["tool_calls"])]
        if m["role"] == "tool":
            d["tool_call_id"] = m.get("tool_call_id") or f"call_{i}"
        out.append(d)
    return out


# =============================================================================
# stage-parse  (vLLM venv): SFT 정답 gen_k 를 vLLM 파서로 파싱 → 모델 응답 메시지
# =============================================================================
def stage_parse(args):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.parser import ParserManager
    from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager

    ReasoningParserManager.import_reasoning_parser(NANO_V3_PLUGIN)  # Ultra 레시피 reasoning_parser_plugin
    tok = _vllm_tokenizer(args.ckpt)
    stub = _StubModelConfig(args.ckpt)
    sft = jload(os.path.join(args.out, "sft.json"))
    res = {"cases": {}}
    for c in sft["cases"]:
        if c["history"] != "model":
            continue
        ctk = None if c["think"] else {"enable_thinking": False}
        tools = _chat_tools(c["norm"]["tools"])
        per = {}
        for combo, (tp, rp) in PARSER_COMBOS.items():
            pcls = ParserManager.get_parser(tool_parser_name=tp, reasoning_parser_name=rp,
                                            enable_auto_tools=True, model_name=args.ckpt)
            outs = {}
            for t in c["targets"]:
                req = ChatCompletionRequest.model_validate({
                    "model": args.ckpt, "messages": [{"role": "user", "content": "x"}],
                    **({"tools": tools} if tools else {}),
                    **({"chat_template_kwargs": ctk} if ctk else {}),
                    "temperature": 1.0, "top_p": 1.0})
                parser = pcls(tok, req.tools, chat_template_kwargs=req.chat_template_kwargs or {},
                              model_config=stub)
                req2 = parser.adjust_request(request=req)
                gen = t["gen"]
                body = gen[:-1] if gen and gen[-1] == IM_END else gen  # vLLM 은 정지 토큰 텍스트를 내지 않음
                text = tok.decode(body, skip_special_tokens=req2.skip_special_tokens)
                reasoning, content, tool_calls = parser.parse(text, req2, enable_auto_tools=True,
                                                              model_output_token_ids=gen)
                tcs = []
                for j, tc in enumerate(tool_calls or []):
                    d = tc.model_dump() if hasattr(tc, "model_dump") else dict(tc)
                    tcs.append({"id": d.get("id") or f"chatcmpl-tool-{c['name']}-{t['k']}-{j}", "type": "function",
                                "function": {"name": d.get("name") or d["function"]["name"],
                                             "arguments": d.get("arguments") if "arguments" in d
                                             else d["function"]["arguments"]}})
                src = c["norm"]["messages"][t["k"]]
                rc_sft = src.get("reasoning_content")
                outs[str(t["k"])] = {
                    "skip_special_tokens": req2.skip_special_tokens, "text": text,
                    "reasoning": reasoning, "content": content, "tool_calls": tcs,
                    "rt_reasoning_equal": (reasoning or "") == (rc_sft or ""),
                    "rt_content_equal": (content or "") == (src.get("content") or ""),
                    "rt_args_equal": [json.loads(a["function"]["arguments"]) for a in tcs]
                    == [tc["function"]["arguments"] for tc in (src.get("tool_calls") or [])],
                }
            per[combo] = outs
        res["cases"][c["name"]] = per
        print(f"[parse] {c['name']}: " + ", ".join(
            f"{cb}: rt_reason={sum(o['rt_reasoning_equal'] for o in v.values())}/{len(v)} "
            f"rt_content={sum(o['rt_content_equal'] for o in v.values())}/{len(v)}" for cb, v in per.items()))
    jdump(res, os.path.join(args.out, "parse.json"))


# =============================================================================
# stage-gym  (Gym venv): 실제 VLLMModel.responses() 로 턴별 chat 요청 본문을 만든다 (vLLM 클라이언트만 mock)
# =============================================================================
def _norm_to_responses_tools(norm_tools):
    if not norm_tools:
        return None
    out = []
    for t in norm_tools:
        fn = t.get("function", t)
        d = {"type": "function", "name": fn["name"]}
        for k in ("description", "parameters"):
            if k in fn:
                d[k] = fn[k]
        # Responses FunctionToolParam 은 strict 가 필수 키(Optional[bool]) — SFT 도구에 없으면 null
        d["strict"] = fn.get("strict")
        out.append(d)
    return out


def _msgs_to_responses_items(msgs, call_ids_by_tc):
    """SFT 메시지(데이터셋 이력) → Responses 입력 항목 (Gym 표준 표현)."""
    items = []
    for i, m in enumerate(msgs):
        if m["role"] in ("system", "user"):
            items.append({"role": m["role"], "content": m["content"]})
        elif m["role"] == "assistant":
            if m.get("reasoning_content"):
                items.append({"type": "reasoning", "id": f"rs_{i}",
                              "summary": [{"type": "summary_text", "text": m["reasoning_content"]}]})
            if m.get("content"):
                items.append({"type": "message", "role": "assistant", "id": f"msg_{i}", "status": "completed",
                              "content": [{"type": "output_text", "text": m["content"], "annotations": []}]})
            for j, tc in enumerate(m.get("tool_calls") or []):
                cid = f"call_{i}_{j}"
                call_ids_by_tc.append(cid)
                items.append({"type": "function_call", "call_id": cid, "id": cid, "status": "completed",
                              "name": tc["function"]["name"],
                              "arguments": json.dumps(tc["function"]["arguments"], ensure_ascii=False)})
        elif m["role"] == "tool":
            items.append({"type": "function_call_output", "call_id": call_ids_by_tc.pop(0),
                          "output": m["content"]})
    return items


def stage_gym(args):
    import asyncio
    sys.path.insert(0, GYM_ROOT)
    from unittest.mock import MagicMock
    import nemo_gym
    import nemo_gym.server_utils
    from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
    from nemo_gym.server_utils import ServerClient
    from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig

    assert os.path.realpath(nemo_gym.__file__).startswith(os.path.realpath(GYM_ROOT)), nemo_gym.__file__
    nemo_gym.server_utils.get_global_config_dict = lambda *a, **k: {}
    sft = jload(os.path.join(args.out, "sft.json"))
    parse = jload(os.path.join(args.out, "parse.json"))

    class Client:
        def __init__(self):
            self.sent, self.reply = [], None

        async def create_chat_completion(self, **kw):
            self.sent.append(copy.deepcopy(kw))
            return copy.deepcopy(self.reply)

    def make_server(variant, think):
        if variant == "null":
            ctk = None if think else {"enable_thinking": False}
        else:
            ctk = {"enable_thinking": bool(think), "truncate_history_thinking": False}
        cfg = VLLMModelConfig(host="127.0.0.1", port=1, base_url="http://127.0.0.1:1/v1", api_key="x",
                              model=args.ckpt, entrypoint="", name="policy_model",
                              return_token_id_information=True, uses_reasoning_parser=True,
                              uses_interleaved_reasoning=True, chat_template_kwargs=ctk)
        srv = VLLMModel(config=cfg, server_client=MagicMock(spec=ServerClient, global_config_dict={}))
        cl = Client()
        VLLMModel._resolve_client = lambda self, request: cl  # 세션 라우팅 우회 (클라이언트 mock 하나)
        return srv, cl

    def reply_for(p, gen, j):
        msg = {"role": "assistant", "content": p["content"], "reasoning": p["reasoning"],
               "prompt_token_ids": [SENTINEL - j], "generation_token_ids": gen,
               "generation_log_probs": [-0.5] * len(gen)}
        if p["tool_calls"]:
            msg["tool_calls"] = p["tool_calls"]
        return {"id": f"chatcmpl-{j}", "object": "chat.completion", "created": 0, "model": args.ckpt,
                "choices": [{"index": 0, "message": msg, "logprobs": None,
                             "finish_reason": "tool_calls" if p["tool_calls"] else "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": len(gen), "total_tokens": 1 + len(gen)}}

    req_mock = MagicMock()
    req_mock.session = {}
    res = {"cases": {}}
    for c in sft["cases"]:
        norm = c["norm"]
        msgs = c["rl_msgs"]
        per = {}
        combos = PARSER_COMBOS if c["history"] == "model" else {"dataset": None}
        for combo in combos:
            for variant in GYM_VARIANTS:
                srv, cl = make_server(variant, c["think"])
                bodies, errors = [], []
                first_k = c["targets"][0]["k"]
                if c.get("rl_input"):
                    base = copy.deepcopy(c["rl_input"])
                    items = base.pop("input")
                    extra = {k: v for k, v in base.items() if k not in ("metadata",)}
                else:
                    items = _msgs_to_responses_items(msgs[:first_k], [])
                    extra = {"temperature": 1.0, "top_p": 1.0}
                    rt = _norm_to_responses_tools(norm["tools"])
                    if rt:
                        extra["tools"] = rt
                for ti, t in enumerate(c["targets"]):
                    k = t["k"]
                    if c["history"] == "model":
                        p = parse["cases"][c["name"]][combo][str(k)]
                    else:
                        p = {"content": "Done.", "reasoning": "Okay", "tool_calls": []}
                    cl.reply = reply_for(p, t["gen"], ti)
                    body = NeMoGymResponseCreateParamsNonStreaming.model_validate({"input": items, **extra})
                    try:
                        resp = asyncio.run(srv.responses(req_mock, body))
                    except Exception as e:  # noqa: BLE001
                        errors.append(f"call {ti}: {type(e).__name__}: {e}")
                        break
                    bodies.append(cl.sent[-1])
                    out_items = [o.model_dump(exclude_none=True) for o in resp.output]
                    items = items + out_items
                    # 다음 타깃 전까지의 환경 메시지 (tool 결과 = function_call_output, user = 시뮬레이터 발화)
                    nxt = c["targets"][ti + 1]["k"] if ti + 1 < len(c["targets"]) else None
                    if nxt is None:
                        break
                    call_ids = [o["call_id"] for o in out_items if o.get("type") == "function_call"]
                    for m in msgs[k + 1:nxt]:
                        if m["role"] == "tool":
                            items.append({"type": "function_call_output", "call_id": call_ids.pop(0),
                                          "output": m["content"]})
                        elif m["role"] in ("user", "system"):
                            items.append({"role": m["role"], "content": m["content"]})
                        else:
                            raise RuntimeError(f"unexpected role between targets: {m['role']}")
                per[f"{combo}/{variant}"] = {"bodies": bodies, "errors": errors}
        res["cases"][c["name"]] = per
        print(f"[gym] {c['name']}: " + ", ".join(f"{k}={len(v['bodies'])}{'!' if v['errors'] else ''}"
                                                 for k, v in per.items()))
    jdump(res, os.path.join(args.out, "gym.json"))


# =============================================================================
# stage-render  (vLLM venv): Gym 요청 본문 → vLLM 렌더 (+ NeMo-RL replace_prefix_tokens)
# =============================================================================
def stage_render(args):
    from typing import List, Optional
    import transformers
    import vllm
    from transformers import AutoTokenizer
    from vllm.entrypoints.chat_utils import parse_chat_messages
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.renderers.hf import resolve_chat_template_content_format, safe_apply_chat_template
    from vllm.sampling_params import SamplingParams
    from nemo_rl.models.generation.openai_server_utils import replace_prefix_tokens

    tok = _vllm_tokenizer(args.ckpt)
    hf_tok = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
    stub = _StubModelConfig(args.ckpt)

    # NeMo-RL vllm_worker_async.py NeMoRLOpenAIChatRequestMixin 과 동일 로직 (클로저 안이라 import 불가 → 복제)
    class NeMoRLChatCompletionRequest(ChatCompletionRequest):
        required_prefix_token_ids: Optional[List[int]] = None

        def model_post_init(self, context):
            if self.required_prefix_token_ids is None:
                for message in reversed(self.messages):
                    if "prompt_token_ids" in message:
                        self.required_prefix_token_ids = (message["prompt_token_ids"]
                                                          + message["generation_token_ids"])
                        break
            return super().model_post_init(context)

    class TurnEndTok:
        """수정안 시뮬레이션: replace_prefix_tokens 가 쓰는 eos_token_id 만 <|im_end|> 로."""
        eos_token_id = IM_END

        def decode(self, *a, **k):
            return tok.decode(*a, **k)

    def render(request, messages, add_generation_prompt, tool_fix=None):
        req = request.model_copy(update={"add_generation_prompt": add_generation_prompt})
        # vLLM OnlineRenderer.render_chat 와 동일: tool.model_dump() (FunctionDefinition 은 None 값을 남긴다)
        tool_dicts = None if req.tools is None else [t.model_dump() for t in req.tools]
        if tool_fix is not None and tool_dicts:
            # 수정안 시뮬레이션: None 값 제거(vLLM) + Gym 이 떼어낸 strict 를 원 데이터 값으로 복원
            fixed = []
            for td in tool_dicts:
                fn = {k: v for k, v in td["function"].items() if v is not None}
                if fn["name"] in tool_fix:
                    fn["strict"] = tool_fix[fn["name"]]
                fixed.append({"type": "function", "function": fn})
            tool_dicts = fixed
        chat_params = req.build_chat_params(None, "auto").with_defaults(
            {"tools": tool_dicts, "tokenize": False}, default_media_io_kwargs=None, default_mm_processor_kwargs=None)
        fmt = resolve_chat_template_content_format(
            chat_template=chat_params.chat_template, tools=chat_params.chat_template_kwargs.get("tools"),
            given_format=chat_params.chat_template_content_format, tokenizer=tok, model_config=stub)
        conv, _, _ = parse_chat_messages(copy.deepcopy(messages), stub, content_format=fmt)
        prompt = safe_apply_chat_template(stub, tok, conv, **chat_params.get_apply_chat_template_kwargs())
        ids = tok(prompt, add_special_tokens=req.add_special_tokens)["input_ids"]
        return ids, fmt, conv

    sft = jload(os.path.join(args.out, "sft.json"))
    gym = jload(os.path.join(args.out, "gym.json"))
    gen_cfg = sft["meta"]["generation_config"]
    # 정지 토큰: NeMo-RL configure_generation_config → stop_token_ids=[tokenizer.eos_token_id],
    # vLLM InputProcessor 가 generation_config.json eos_token_id 를 합친다 (sampling_params.update_from_generation_config)
    sp = SamplingParams(stop_token_ids=[tok.eos_token_id])
    sp.update_from_generation_config({"eos_token_id": gen_cfg.get("eos_token_id")}, tok.eos_token_id)
    res = {"meta": {"transformers": transformers.__version__, "vllm": vllm.__version__,
                    "eos_token_id": tok.eos_token_id,
                    "stop": {"stop_token_ids": sorted(sp.stop_token_ids or []),
                             "all_stop_token_ids": sorted(sp.all_stop_token_ids),
                             "primary_eos": tok.eos_token_id}}, "cases": {}}
    for c in sft["cases"]:
        r = {"paths": {}}
        norm = c["norm"]
        kw = {"medium_effort": True} if c["medium_effort"] else {}
        # P2 (tf5.8 HF 직접) / P2r (vLLM 메시지 파서에 SFT 메시지 그대로)
        txt = hf_tok.apply_chat_template(norm["messages"], tools=norm["tools"], tokenize=False,
                                         add_generation_prompt=False, **kw)
        r["full_ids_hf"] = hf_tok(txt, add_special_tokens=False)["input_ids"]
        for tag, keys in (("rc_field", ("reasoning_content",)), ("reasoning_field", ("reasoning",))):
            req = NeMoRLChatCompletionRequest.model_validate({
                "model": args.ckpt, "messages": _norm_to_wire(norm["messages"], keys),
                **({"tools": norm["tools"]} if norm["tools"] else {}),
                **({"chat_template_kwargs": kw} if kw else {}), "temperature": 1.0, "top_p": 1.0})
            r[f"full_ids_vllm_{tag}"], fmt, conv = render(req, req.messages, False)
            r["content_format"] = fmt
            r[f"vllm_conv_has_reasoning_{tag}"] = [bool(m.get("reasoning_content")) for m in conv
                                                   if m["role"] == "assistant"]
        strict_map = {}
        for t in norm["tools"] or []:
            fn = t.get("function", t)
            if fn.get("strict") is not None:
                strict_map[fn["name"]] = fn["strict"]
        for path, pv in gym["cases"].get(c["name"], {}).items():
            out = {"errors": list(pv["errors"]), "calls": []}
            computed = {"asis": {}, "fixed": {}, "fixall": {}}
            for j, body in enumerate(pv["bodies"]):
                call = {"j": j}
                # fixed/fixall = 실제 패치(replace_prefix_tokens(turn_end_token_id=<|im_end|>), 2026-10-07)를 호출한다
                for mode, tk, tfix, te in (("asis", tok, None, None), ("fixed", tok, None, IM_END),
                                           ("fixall", tok, strict_map, IM_END)):
                    b = copy.deepcopy(body)
                    blocked = False
                    for m in b["messages"]:
                        pt = m.get("prompt_token_ids")
                        if isinstance(pt, list) and len(pt) == 1 and pt[0] <= SENTINEL:
                            i = SENTINEL - pt[0]
                            if i in computed[mode]:
                                m["prompt_token_ids"] = computed[mode][i]
                            else:
                                blocked = True
                    if blocked:
                        call[mode] = {"status": "blocked(이전 호출 실패)"}
                        continue
                    req = NeMoRLChatCompletionRequest.model_validate(b)
                    try:
                        full, _, conv = render(req, req.messages, req.add_generation_prompt, tfix)
                    except Exception as e:  # noqa: BLE001
                        call[mode] = {"status": f"render_error: {type(e).__name__}: {e}"}
                        continue
                    entry = {"status": "ok", "p3": full,
                             "required_prefix": req.required_prefix_token_ids is not None,
                             "conv_reasoning": [bool(m.get("reasoning_content")) for m in conv
                                                if m["role"] == "assistant"]}
                    final = full
                    if req.required_prefix_token_ids is not None:
                        last_a = max(i for i, m in enumerate(req.messages) if m["role"] == "assistant")
                        pre, _, _ = render(req, req.messages[:last_a + 1], False, tfix)
                        try:
                            final = replace_prefix_tokens(tokenizer=tk, model_prefix_token_ids=
                                                          req.required_prefix_token_ids,
                                                          template_prefix_token_ids=pre, template_token_ids=full,
                                                          turn_end_token_id=te)
                        except AssertionError as e:
                            entry["status"] = "AssertionError: " + str(e).split("\n")[0]
                            final = None
                    entry["final"] = final
                    if final is not None:
                        computed[mode][j] = final
                    call[mode] = entry
                out["calls"].append(call)
            r["paths"][path] = out
        res["cases"][c["name"]] = r
        print(f"[render] {c['name']}: paths={list(r['paths'])}")
    jdump(res, os.path.join(args.out, "render.json"))


# =============================================================================
# report  (system python)
# =============================================================================
def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def stage_report(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_V5)
    sft = jload(os.path.join(args.out, "sft.json"))
    nrl = jload(os.path.join(args.out, "nrl.json"))
    parse = jload(os.path.join(args.out, "parse.json"))
    ren = jload(os.path.join(args.out, "render.json"))
    lines, rows, details = [], [], []
    tally = {}

    def cmp(case, label, ref, got, extra=""):
        if got is None:
            rows.append((case, label, "N/A", extra))
            return
        i = first_diff(ref, got)
        ok = i is None
        tally.setdefault(label.split("[")[0], [0, 0])
        tally[label.split("[")[0]][0 if ok else 1] += 1
        st = f"MATCH ({len(ref)} tok)" if ok else f"MISMATCH @{i} (ref {len(ref)} / got {len(got)})"
        rows.append((case, label, st, extra))
        if not ok:
            lo = max(0, i - 12)
            details.append(f"--- {case} {label}: first diff pos {i}\n"
                           f"    SFT : {tok.decode(ref[lo:i + 12])!r}\n"
                           f"    RL  : {tok.decode(got[lo:i + 12])!r}\n"
                           f"    tok SFT {ref[i:i + 3]} vs RL {got[i:i + 3]}")

    meta = sft["meta"]
    lines.append(f"versions: SFT tf {meta['transformers']} | NeMo-RL driver tf {nrl['meta']['transformers']} | "
                 f"vLLM {ren['meta']['vllm']} tf {ren['meta']['transformers']}")
    lines.append(f"tokenizer sync: {json.dumps(meta['sync'], ensure_ascii=False)}")
    lines.append(f"eos_token={meta['eos_token']!r} id={meta['eos_token_id']} (NeMo-RL tok eos={nrl['meta']['eos_token_id']},"
                 f" vLLM tok eos={ren['meta']['eos_token_id']}); generation_config eos={meta['generation_config'].get('eos_token_id')}")
    lines.append(f"rollout stop set (NeMo-RL stop_token_ids=[eos] + vLLM generation_config merge): {ren['meta']['stop']}")
    for c in sft["cases"]:
        name = c["name"]
        for t in c["targets"]:
            k = t["k"]
            lab = f"turn{k}"
            cmp(name, f"P0_sft_genprompt[{lab}]", t["ctx"], t["p0"], t["opener"])
            if not t["gen_ends_im_end"]:
                rows.append((name, f"gen_end[{lab}]", "FAIL: gen 이 <|im_end|> 로 끝나지 않음", ""))
            nt = nrl["cases"][name]["targets"].get(str(k))
            if nt:
                cmp(name, f"P1_nrl_native[{lab}]", t["ctx"], nt["ids"], nt["how"])
        full = c["full_ids"]
        if c["n_subs"] == 1:
            cmp(name, "P2_full_tf5_driver", full, nrl["cases"][name]["full_ids"])
            cmp(name, "P2_full_tf5_vllmvenv_hf", full, ren["cases"][name]["full_ids_hf"])
            for tag in ("rc_field", "reasoning_field"):
                cmp(name, f"P2r_full_vllm_parser_{tag}", full, ren["cases"][name][f"full_ids_vllm_{tag}"],
                    f"conv reasoning kept={ren['cases'][name][f'vllm_conv_has_reasoning_{tag}']}")
        else:
            cmp(name, "P2_full_tf5_driver(no-fanout)", full, nrl["cases"][name]["full_ids_nofanout"])
        for path, pv in ren["cases"][name]["paths"].items():
            for e in pv["errors"]:
                rows.append((name, f"gym[{path}]", "GYM_ERROR " + e[:160], ""))
            for call in pv["calls"]:
                t = c["targets"][call["j"]]
                lab = f"{path}|turn{t['k']}"
                a, f_, x = call.get("asis", {}), call.get("fixed", {}), call.get("fixall", {})
                p3src = a if "p3" in a else f_   # P3 은 토큰 ID 치환과 무관 — as-is 가 막혔으면 fixed 의 것
                if "p3" in p3src:
                    cmp(name, f"P3_gym_vllm_template[{lab}]", t["ctx"], p3src["p3"],
                        f"reasoning→field {p3src.get('conv_reasoning')}")
                if "p3" in x and c["norm"]["tools"]:
                    # 도구 정의만 고친 템플릿 재렌더 (prefix 치환 없음) — 파서 왕복의 정확성만 남긴다
                    cmp(name, f"P3f_template+tooldefs_fixed[{lab}]", t["ctx"], x["p3"])
                if not (p3src.get("required_prefix")):
                    continue
                if a.get("final") is not None:
                    cmp(name, f"P4_prefix_asis[{lab}]", t["ctx"], a["final"])
                else:
                    rows.append((name, f"P4_prefix_asis[{lab}]", "FAIL " + a.get("status", "?")[:140], ""))
                    tally.setdefault("P4_prefix_asis", [0, 0])[1] += 1
                for lbl, e in (("P5_prefix_imend", f_), ("P6_imend+tooldefs_fixed", x)):
                    if e.get("final") is not None:
                        cmp(name, f"{lbl}[{lab}]", t["ctx"], e["final"])
                    else:
                        rows.append((name, f"{lbl}[{lab}]", "FAIL " + e.get("status", "?")[:140], ""))
                        tally.setdefault(lbl, [0, 0])[1] += 1
        if name in parse["cases"]:
            for combo, outs in parse["cases"][name].items():
                bad = [(k, o) for k, o in outs.items() if not (o["rt_reasoning_equal"] and o["rt_content_equal"]
                                                              and o["rt_args_equal"])]
                st = "ROUNDTRIP OK" if not bad else "ROUNDTRIP DIFF " + "; ".join(
                    f"turn{k}: reasoning={'=' if o['rt_reasoning_equal'] else repr((o['reasoning'] or '')[-20:])}"
                    f" content={'=' if o['rt_content_equal'] else repr((o['content'] or '')[:30])}"
                    f" args={'=' if o['rt_args_equal'] else 'DIFF'} skip_special={o['skip_special_tokens']}"
                    for k, o in bad)
                rows.append((name, f"parse[{combo}]", st, ""))
    w = max(len(r[1]) for r in rows)
    lines.append("")
    cur = None
    for case, label, st, extra in rows:
        if case != cur:
            d = next(c["desc"] for c in sft["cases"] if c["name"] == case)
            lines.append(f"## {case} — {d}")
            cur = case
        lines.append(f"  {label:<{w}}  {st}" + (f"   [{extra}]" if extra else ""))
    lines.append("")
    lines.append("## tally (MATCH / MISMATCH)")
    for k, (m, x) in sorted(tally.items()):
        lines.append(f"  {k:<40} {m:>4} / {x}")
    lines.append("")
    lines.append("## first-diff details")
    lines.extend(details)
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(args.out, "REPORT.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    bad = sum(x for _, x in tally.values())
    sys.exit(1 if bad else 0)


# =============================================================================
# run  (orchestrator)
# =============================================================================
def stage_run(args):
    me = os.path.abspath(__file__)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    common = ["--out", args.out, "--ckpt", args.ckpt]
    plan = [("stage-sft", [PY_SFT, me]),
            ("stage-nrl", [CLEAN, PY_DRV, me]),
            ("stage-parse", [CLEAN, PY_VLLM, me]),
            ("stage-gym", [CLEAN, PY_GYM, me]),
            ("stage-render", [CLEAN, PY_VLLM, me])]
    os.makedirs(args.out, exist_ok=True)
    for st, cmd in plan:
        if args.skip and st in args.skip.split(","):
            continue
        t = time.time()
        print(f"==== {st}", flush=True)
        p = subprocess.run(cmd + [st] + common, env=env, stdin=subprocess.DEVNULL,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log = os.path.join(args.out, f"{st}.log")
        with open(log, "w") as f:
            f.write(p.stdout)
        tail = [ln for ln in p.stdout.splitlines() if ln.startswith(("[", "Traceback", "  File", "AssertionError",
                                                                     "Error", "RuntimeError", "ValueError"))]
        print("\n".join(tail[-40:]))
        print(f"     rc={p.returncode} {time.time() - t:.0f}s log={log}", flush=True)
        if p.returncode != 0:
            print(p.stdout[-4000:])
            sys.exit(2)
    p = subprocess.run([PY_SFT, me, "report"] + common, env=env, stdin=subprocess.DEVNULL)
    sys.exit(p.returncode)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["run", "stage-sft", "stage-nrl", "stage-parse", "stage-gym",
                                      "stage-render", "report"])
    ap.add_argument("--out", default="/tmp/alpha_render_gate")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--skip", default="", help="run: 건너뛸 단계 (쉼표, 이미 산출된 JSON 재사용)")
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.makedirs(args.out, exist_ok=True)
    {"run": stage_run, "stage-sft": stage_sft, "stage-nrl": stage_nrl, "stage-parse": stage_parse,
     "stage-gym": stage_gym, "stage-render": stage_render, "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    main()
