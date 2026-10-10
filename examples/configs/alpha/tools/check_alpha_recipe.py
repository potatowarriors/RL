"""alpha GRPO 레시피 실행 전 검사 (CPU) — RLVR 이식 위험 보고(docs/RLVR_READINESS.md)의 조용한 실패 경로를 레시피 단계에서 막는다.

레시피를 NeMo-RL 과 같은 방식으로 읽는다(load_config → resolve → MasterConfig 스키마). 그 뒤 alpha 불변량을 검사한다.
ERROR 가 하나라도 있으면 종료 코드 1. WARN 은 결정 대기·확인 권고다.

  python examples/configs/alpha/tools/check_alpha_recipe.py examples/configs/alpha/student_rlvr1_alpha.yaml [hydra.override=값 ...]
  (NeMo-RL 드라이버 venv 로 실행: $NRL_ROOT/clean_run.sh $NRL_ROOT/bin/uv run --locked python ... </dev/null)

검사 항목 (괄호는 RLVR_READINESS.md 항목):
  GDN 재귀 상태 fp32 · R3 · turn_end 3 (CLAUDE.md 레시피 규약) · R3 + router fusion 금지 (H1) · 그룹 라우팅 덮어쓰기 금지 (낮음)
  CP>1 의 packing·패딩 배수·chunk·defer_fp32_logits (H2) · Gym 파서·invalid tool call 벌점 순서 (H3)
  Gym judge 가 정책 자신을 가리킴 · Gym judge GPU 가 롤아웃 노드의 남은 GPU 를 넘음 (H4, RL_PLAN.md 결정 21)
  데이터 행의 agent 가 Gym 설정에 없음 · ns_tools 행의 verifier_type 미등록 · judge 행 정답이 자기 output_regex 에 잘림 · 재판정 꺼짐 (KNOWN_ISSUES 10-10)
  prefix caching 명시 (M1) · async 배치 파라미터 명시 (M2) · async ⇒ IS 보정 (M7) · KL>0 + R3 (M6) · top_p 1.0 (낮음)
  GBS 나눗셈 (낮음) · Ultra 의 무효 키 (낮음) · reward_penalties token_ids 와 토크나이저 대조 · generation_config eos [3,0] (M9)
"""

import collections
import json
import os
import re
import sys

ERRORS: list[str] = []
WARNS: list[str] = []


def err(msg: str) -> None:
    ERRORS.append(msg)


def warn(msg: str) -> None:
    WARNS.append(msg)


# NeMo-RL 이 읽지 않는 Ultra 키 (2026-10-07 감사: nemo_rl/ 에서 읽는 곳 0건)
IGNORED_MEGATRON_KEYS = ["do_not_average_loss", "cp_normalize", "calculate_per_token_loss", "scale_loss_by_dp_cp_size",
                         "moe_router_enable_expert_bias", "moe_aux_loss_coeff", "first_last_layers_bf16",
                         "num_layers_at_start_in_bf16", "num_layers_at_end_in_bf16", "track_moe_metrics"]
SANDBOX_CONFIGS = ["ns_tools", "math_formal_lean"]
# Gym config_paths 를 찾는 루트 — Gym 본체 다음 alpha 플러그인 (launch.sh 의 NEMO_GYM_EXTRA_ROOTS 와 같다)
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
GYM_ROOTS = [os.path.join(REPO, "3rdparty", "Gym-workspace", "Gym"), os.path.join(REPO, "examples", "configs", "alpha", "gym_plugins")]
POLICY_MODELS = ("policy_model", "policy_model_reasoning_off")
JUDGE_FIELDS = ("judge_model_server", "genrm_model_server")  # resources server 가 채점 모델을 가리키는 필드


def check(cfg: dict) -> None:
    pol = cfg["policy"]
    mcfg = pol["megatron_cfg"]
    gen = pol["generation"]
    vcfg = gen["vllm_cfg"]
    vkw = gen.get("vllm_kwargs") or {}
    grpo = cfg["grpo"]
    loss = cfg["loss_fn"]
    env = cfg.get("env") or {}
    use_gym = bool(env.get("should_use_nemo_gym"))
    async_on = bool(grpo.get("async_grpo", {}).get("enabled"))
    r3 = bool((pol.get("router_replay") or {}).get("enabled"))
    cp = mcfg["context_parallel_size"]
    tp = mcfg["tensor_model_parallel_size"]

    # alpha RL 기본값 (R1·R3 게이트)
    if vkw.get("mamba_ssm_cache_dtype") != "float32":
        err("policy.generation.vllm_kwargs.mamba_ssm_cache_dtype 가 float32 가 아니다 — GDN 재귀 상태 bf16 누적 (R1 FAIL 0.0042)")
    if not r3:
        err("policy.router_replay.enabled 가 false — alpha RL 기본값 (R1)")
    if vcfg.get("turn_end_token_id") != 3:
        err("policy.generation.vllm_cfg.turn_end_token_id 가 3 이 아니다 — Gym 멀티턴 2번째 호출부터 깨짐 (R3)")

    # H1: R3 + router fusion
    overrides = mcfg.get("model_overrides") or {}
    if r3 and (overrides.get("moe_router_fusion") or mcfg.get("moe_router_fusion")):
        err("R3 + moe_router_fusion: mcore fused top-k 는 replay 전에 반환한다 — R3 가 에러 없이 꺼진다 (H1)")
    for k in ("moe_router_num_groups", "moe_router_group_topk"):
        for where, d in (("megatron_cfg", mcfg), ("model_overrides", overrides)):
            if k in d and d[k] != {"moe_router_num_groups": 8, "moe_router_group_topk": 4}[k]:
                err(f"{where}.{k}={d[k]} 가 alpha 그룹 라우팅(8 그룹·top-4)을 덮는다. R3 가 켜져 있으면 KL 에도 안 보인다")
    for k in IGNORED_MEGATRON_KEYS:
        if k in mcfg:
            warn(f"megatron_cfg.{k} 는 NeMo-RL 이 읽지 않는다 (Ultra 복사 흔적, 효과 없음)")

    # H2: CP>1
    if cp > 1:
        if not pol["sequence_packing"]["enabled"]:
            err("CP>1 인데 sequence_packing 이 꺼져 있다 (setup.py assert)")
        if pol["make_sequence_length_divisible_by"] % (2 * cp * tp) != 0:
            err(f"make_sequence_length_divisible_by={pol['make_sequence_length_divisible_by']} 가 2·CP·TP={2 * cp * tp} 의 배수가 아니다")
        if mcfg.get("use_fused_linear_logprobs"):
            err("CP>1 와 use_fused_linear_logprobs 는 함께 쓸 수 없다")
        if not pol.get("logprob_chunk_size"):
            warn("CP>1 장문맥인데 logprob_chunk_size 가 없다 — logprob backward 메모리가 커진다 (R4 는 2048)")
        elif not mcfg.get("defer_fp32_logits"):
            err("logprob_chunk_size 는 defer_fp32_logits: true 를 요구한다")
        if not pol["sequence_packing"].get("fuse_loss"):
            warn("CP>1 장문맥인데 sequence_packing.fuse_loss 가 꺼져 있다 (R4 조건과 다름)")
    if mcfg.get("activation_checkpointing") and mcfg.get("recompute_granularity") == "selective" \
            and not mcfg.get("recompute_modules"):
        err("selective recompute 인데 recompute_modules 가 없다 — mcore 기본은 core_attn 만 (setup.py)")

    # 샘플링 (H3·낮음)
    if gen["temperature"] != 1.0 or gen["top_p"] != 1.0:
        warn(f"temperature={gen['temperature']} top_p={gen['top_p']} — top_p<1 이면 chunked logprob 이 꺼져 128K 메모리 계획이 깨진다")

    # 배치
    n_samples = grpo["num_prompts_per_step"] * grpo["num_generations_per_prompt"]
    gbs = pol["train_global_batch_size"]
    if n_samples % gbs != 0:
        err(f"num_prompts×gens={n_samples} 가 train_global_batch_size={gbs} 로 나누어떨어지지 않는다 — 나머지 샘플이 조용히 빠진다")
    if loss.get("force_on_policy_ratio") and n_samples != gbs:
        err("force_on_policy_ratio 는 train_global_batch_size == num_prompts×gens 를 요구한다")

    # M6·M7
    if async_on and not loss.get("use_importance_sampling_correction"):
        err("async GRPO 는 use_importance_sampling_correction: true 가 필요하다 (grpo.py assert)")
    if r3 and loss.get("reference_policy_kl_penalty", 0) > 0:
        warn("KL>0 + R3: reference logprob 은 route replay 를 안 한다 — 라우팅 차이가 KL 에 섞인다 (M6, D4)")
    if loss.get("force_on_policy_ratio") and grpo.get("seq_logprob_error_threshold") is None:
        warn("force_on_policy_ratio + seq_logprob_error_threshold 없음 → 시퀀스 마스킹이 꺼진다 (gen_kl_error 는 학습 forward 에서 계속 나온다)")
    if (loss.get("force_on_policy_ratio") and grpo.get("seq_logprob_error_threshold") is not None
            and not loss.get("seq_logprob_error_in_loss")):
        warn("seq_logprob_error_threshold 를 별도 prev_logprob 패스로 평가한다 (첫 RLVR 런 스텝의 23%) — "
             "loss_fn.seq_logprob_error_in_loss: true 면 학습 forward 안에서 같은 마스킹을 한다 (D4 2026-10-08)")

    # M1·M2
    if vcfg.get("enable_prefix_caching") is None:
        warn("vllm_cfg.enable_prefix_caching 미지정 — NeMo-RL 이 sm≥8 에서 켠다 (vLLM 은 하이브리드에 기본 끔). 명시해 결정을 남긴다 (M1, D2)")
    elif vcfg.get("enable_prefix_caching") and r3:
        warn("prefix caching + R3: chunked prefill 이 강제되고 route 행이 드물게 누락된다 — NRL_ROUTER_REPLAY_VALIDATE=1 로 게이트 (M1)")
    if vkw.get("enable_chunked_prefill") is False and vkw.get("max_num_batched_tokens", 0) < vcfg["max_model_len"]:
        err(f"chunked prefill 끔 + max_num_batched_tokens({vkw.get('max_num_batched_tokens')}) < max_model_len({vcfg['max_model_len']}) — "
            "vLLM 엔진이 기동하지 못한다 (G3 2026-10-07)")
    if async_on:
        for k in ("max_num_batched_tokens", "max_num_seqs"):
            if k not in vkw:
                warn(f"async vLLM 인데 vllm_kwargs.{k} 미지정 — usage_context 없는 기본값(2048·128)이 된다 (M2)")
        if vcfg.get("expert_parallel_size", 1) not in (1, vcfg.get("tensor_parallel_size", 1)):
            err("async vLLM 은 EP≠TP 를 쓸 수 없다 (vllm_generation.py)")
    alloc = str(((mcfg.get("env_vars") or {}).get("PYTORCH_CUDA_ALLOC_CONF")) or "")
    if "expandable_segments:True" in alloc and (mcfg.get("checkpoint") or {}).get("async_save") \
            and cfg["checkpointing"]["enabled"]:
        err("expandable_segments + checkpoint.async_save: 비동기 writer 가 CUDA IPC 를 pidfd_getfd EPERM 으로 못 받아 저장에서 멈춘다 "
            "(G5 2026-10-07) — megatron_cfg.checkpoint.async_save: false")
    sched_max = (mcfg.get("scheduler") or {}).get("max_steps")
    if cfg["checkpointing"]["enabled"] and sched_max is None:
        warn("megatron_cfg.scheduler.max_steps 미설정: 스케줄 길이가 grpo.max_num_steps 를 따라가 max_num_steps 를 바꿔 재개하면 "
             "'total number of weight decay iterations do not match' 로 멈춘다 (G7 2026-10-07)")
    elif sched_max is not None:
        # GRPO 의 train_iters = min(max_num_steps, max_num_epochs × 에폭당 스텝) (grpo.py). Bridge 는 max_steps < train_iters 를 거부한다
        path = ((cfg.get("data") or {}).get("train") or {}).get("data_path")
        if path and os.path.isfile(path):
            with open(path) as f:
                rows = sum(1 for line in f if line.strip())
            g = cfg["grpo"]
            est = min(g["max_num_steps"], g["max_num_epochs"] * -(-rows // g["num_prompts_per_step"]))
            if sched_max < est:
                err(f"megatron_cfg.scheduler.max_steps {sched_max} < 실효 train_iters ≈ {est} "
                    f"(min(max_num_steps, max_num_epochs × 에폭당 스텝)) — Bridge 가 기동에서 거부한다")
    if gen["colocated"]["enabled"] and "expandable_segments:True" in alloc:
        err("colocated(CUDA IPC refit) + expandable_segments: 이 컨테이너에서 refit 이 pidfd_getfd EPERM 으로 실패한다 "
            "(KNOWN_ISSUES 2026-10-07). colocated 런은 env_vars 에서 ES 를 뺀다")
    if vcfg.get("enforce_eager") is False:
        warn("enforce_eager: false (CUDA graph) — G5 KL 0.0018 로 기준 이내 (eager 0.0016~0.0017). 런에서 KL 을 계속 본다 (M4)")

    # H3·H4: Gym
    if use_gym:
        # Gym 은 async vLLM 엔진 + HTTP 서버를 요구한다. async GRPO 는 요구하지 않는다 (grpo.py _should_use_nemo_gym)
        if not vcfg.get("async_engine") or not vcfg.get("expose_http_server"):
            err("Gym 은 vllm_cfg.async_engine + expose_http_server 가 필요하다")
        sk = vcfg.get("http_server_serving_chat_kwargs") or {}
        if not sk.get("tool_parser") or not sk.get("reasoning_parser"):
            err("Gym 레시피에 vllm_cfg.http_server_serving_chat_kwargs.tool_parser·reasoning_parser 가 없다 — "
                "alpha 도구 호출·추론 분리가 안 된다 (H3)")
        # vLLM 0.25.1 에서 qwen3_xml 과 qwen3_coder 는 같은 Qwen3EngineToolParser 다 (vllm/tool_parsers/__init__.py)
        elif sk.get("tool_parser") not in ("qwen3_xml", "qwen3_coder") or sk.get("reasoning_parser") != "nemotron_v3":
            warn(f"파서 {sk.get('tool_parser')}/{sk.get('reasoning_parser')} 는 G3 에서 검증한 조합이 아니다 "
                 "(Pai fleet 의 alpha 조합은 qwen3_xml/nemotron_v3)")
        if grpo.get("invalid_tool_call_advantage") is not None:
            warn("invalid_tool_call_advantage 가 켜져 있다 — G3(파서 스모크) 통과 전에는 정상 호출이 벌점을 받을 수 있다 (H3)")
        venv = vcfg.get("env_vars") or {}
        if str(venv.get("VLLM_ENFORCE_STRICT_TOOL_CALLING", "1")).lower() not in ("0", "false"):
            err("Gym 레시피에 vllm_cfg.env_vars.VLLM_ENFORCE_STRICT_TOOL_CALLING: \"0\" 이 없다 — strict:true 도구에서 vLLM 이 "
                "제약 디코딩을 걸어 롤아웃이 off-policy 가 된다 (G3 2026-10-07: KL 0.0045, 도구 시퀀스 33% 마스킹)")
        ng = env.get("nemo_gym") or {}
        paths = ng.get("config_paths") or []
        for p in paths:
            if any(f"/{name}/" in p for name in SANDBOX_CONFIGS):
                warn(f"코드 sandbox 가 필요한 환경: {p} — sandbox 가 떠 있어야 한다 (RL_DATA.md §5.9)")
        check_gym_judges(cfg, ng, paths)
        pm = ((ng.get("policy_model") or {}).get("responses_api_models") or {}).get("vllm_model") or {}
        ctk = pm.get("chat_template_kwargs")
        if ctk and ctk.get("truncate_history_thinking") is False:
            err("policy_model.chat_template_kwargs.truncate_history_thinking=false 는 Ultra 값이다 — alpha 는 null (R3)")

    # reward_penalties token_ids ↔ 토크나이저
    rp = cfg.get("reward_penalties") or {}
    ids = rp.get("token_ids") or {}
    if ids:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(pol["tokenizer"]["name"], trust_remote_code=True)
        except Exception as e:  # 토크나이저를 못 읽으면 대조를 건너뛴다는 사실을 남긴다
            warn(f"토크나이저를 읽지 못해 reward_penalties.token_ids 대조를 못 했다: {type(e).__name__}: {e}")
        else:
            for key, tag in (("think_open", "<think>"), ("think_close", "</think>")):
                if key in ids and ids[key] is not None and ids[key] != tok.convert_tokens_to_ids(tag):
                    err(f"reward_penalties.token_ids.{key}={ids[key]} 는 {tok.convert_ids_to_tokens(ids[key])!r} 다 — "
                        f"alpha {tag} 는 {tok.convert_tokens_to_ids(tag)} (Ultra id 복사 흔적)")
            for u in ids.get("unwanted") or []:
                if u in (0, 3):
                    err(f"reward_penalties.token_ids.unwanted 에 정지 토큰 {u} 가 있다")
            if ids.get("unwanted"):
                warn("unwanted token_ids = " + ", ".join(f"{u}:{tok.convert_ids_to_tokens(u)!r}" for u in ids["unwanted"]))

    # M9: generation_config eos
    model = pol["model_name"]
    gc_path = os.path.join(model, "generation_config.json") if model else None
    if gc_path and os.path.isdir(model):
        if not os.path.exists(gc_path):
            err(f"{gc_path} 가 없다 — <|im_end|>(3) 에서 멈추지 않는다 (Pai 08-30 사고)")
        else:
            eos = json.load(open(gc_path)).get("eos_token_id")
            if sorted(eos if isinstance(eos, list) else [eos]) != [0, 3]:
                err(f"generation_config eos_token_id={eos} — [3, 0] 이어야 한다")
    else:
        warn(f"policy.model_name={model} 이 로컬 디렉토리가 아니라 generation_config 를 확인하지 못했다")


def _merge(base: dict, over: dict) -> dict:
    """레시피 env.nemo_gym 이 Gym 설정 파일 값을 덮도록 재귀 병합한다."""
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def check_gym_judges(cfg: dict, ng: dict, paths: list[str]) -> None:
    """H4·결정 21: judge 서버가 정책 자신을 가리키지 않는지, Gym local_vllm_model 이 롤아웃 노드의 남은 GPU 에 들어가는지.

    데이터 행의 agent 가 Gym 설정에 있는지, ns_tools 행의 verifier_type 이 등록돼 있는지도 본다 (없으면 롤아웃·채점이 실패한다).
    """
    import yaml

    merged: dict = {}
    for p in paths:
        full = next((os.path.join(r, p) for r in GYM_ROOTS if os.path.isfile(os.path.join(r, p))), None)
        if full is None:
            err(f"Gym 설정 파일을 찾지 못했다: {p}")
            continue
        with open(full) as f:
            merged = _merge(merged, yaml.safe_load(f) or {})
    merged = _merge(merged, {k: v for k, v in ng.items() if isinstance(v, dict)})
    judge_gpus = 0
    for inst, body in merged.items():
        ns = ((body or {}).get("resources_servers") or {}).get("ns_tools")
        if isinstance(ns, dict) and "sandbox_port" in ns and not isinstance(ns["sandbox_port"], str):
            err(f"Gym {inst}.sandbox_port={ns['sandbox_port']!r} — NSToolsConfig 는 str 이다. 정수면 ns_tools 가 기동에서 죽는다 "
                "(J1 2026-10-10). 따옴표로 감싼다")
        for stype, scfg in ((body or {}).get("resources_servers") or {}).items():
            if not isinstance(scfg, dict) or scfg.get("should_use_judge") is False:
                continue
            for field in JUDGE_FIELDS:
                name = (scfg.get(field) or {}).get("name")
                if name in POLICY_MODELS:
                    err(f"Gym {inst}.{field} 가 {name} 다 — 정책이 자기 응답을 채점한다. 레시피 env.nemo_gym 에서 judge 모델로 덮어쓴다")
                elif name is not None and name not in merged:
                    err(f"Gym {inst}.{field}={name} 인 모델 서버가 정의되지 않았다")
        kw = (((body or {}).get("responses_api_models") or {}).get("local_vllm_model") or {}).get("vllm_serve_kwargs")
        if kw:
            judge_gpus += kw.get("tensor_parallel_size", 1) * kw.get("pipeline_parallel_size", 1) * kw.get("data_parallel_size", 1)
    # Gym 은 local_vllm_model 을 Ray 의 남은 GPU 에 띄운다. 2노드 비-colocated 에서 남는 GPU 는 롤아웃 노드의 나머지뿐이다
    gen, cl = cfg["policy"]["generation"], cfg["cluster"]
    res = gen["colocated"].get("resources") or {}
    free = 0
    if not gen["colocated"]["enabled"] and cl["num_nodes"] > 1:
        free = (cl["gpus_per_node"] - (res.get("gpus_per_node") or cl["gpus_per_node"])) * (res.get("num_nodes") or 1)
    if judge_gpus > free:
        err(f"Gym local_vllm_model 이 GPU {judge_gpus} 장을 쓰는데 남는 GPU 는 {free} 장이다 — judge placement group 이 기동에서 끝없이 기다린다")
    elif free > judge_gpus:
        warn(f"롤아웃 노드 GPU {free - judge_gpus} 장이 논다 (남는 {free} · Gym local_vllm_model {judge_gpus})")
    check_gym_data_agents(cfg, merged)


AGENT_NAME = re.compile(r'"agent_ref":\s*\{[^{}]*?"name":\s*"([^"]+)"')
VERIFIER_TYPE = re.compile(r'"verifier_type":\s*"([^"]+)"')


def _equivalence_server(merged: dict, agent: str, verifier_type: str | None) -> str | None:
    """데이터 agent 행이 채점받는 equivalence_llm_judge 서버 이름 (직접 또는 ns_tools 의 verifier 경유). 아니면 None."""
    cfgs = list(((merged.get(agent) or {}).get("responses_api_agents") or {}).values())
    if not cfgs:
        return None
    rs = ((cfgs[0] or {}).get("resources_server") or {}).get("name")
    servers = (merged.get(rs) or {}).get("resources_servers") or {}
    if isinstance(servers.get("equivalence_llm_judge"), dict):
        return rs
    ns = servers.get("ns_tools")
    if isinstance(ns, dict):
        tgt = ((ns.get("verifiers") or {}).get(verifier_type or ns.get("default_verifier")) or {}).get("name")
        if isinstance(((merged.get(tgt) or {}).get("resources_servers") or {}).get("equivalence_llm_judge"), dict):
            return tgt
    return None


def check_gym_data_agents(cfg: dict, merged: dict) -> None:
    """데이터 행의 agent_ref 가 Gym 설정에 있는지, ns_tools 행의 verifier_type 이 ns_tools.verifiers 에 있는지 (2026-10-08 rdkit 사례)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from measure_judge_p04 import gold_regex_defect

    agents = collections.Counter()
    verifiers = collections.defaultdict(collections.Counter)  # agent -> verifier_type -> 행 수
    extraction = collections.defaultdict(collections.Counter)  # (split, agent, verifier_type) -> gold_regex_defect 상태 -> 행 수
    data = cfg.get("data") or {}
    for split in ("train", "validation"):
        path = (data.get(split) or {}).get("data_path")
        if not path or not os.path.isfile(path):
            continue
        with open(path) as f:
            for line in f:
                m = AGENT_NAME.search(line)
                if not m:
                    continue
                agents[m.group(1)] += 1
                v = VERIFIER_TYPE.search(line)
                if v:
                    verifiers[m.group(1)][v.group(1)] += 1
                if '"output_regex"' in line:
                    row = json.loads(line)
                    rx = (row.get("template_metadata") or {}).get("output_regex")
                    if rx and row.get("expected_answer") is not None:
                        key = (split, m.group(1), v.group(1) if v else None)
                        extraction[key][gold_regex_defect(rx, str(row["expected_answer"]))[0]] += 1
    # 정답 추출 (KNOWN_ISSUES 2026-10-10): equivalence_llm_judge 로 채점하는 행은 정답을 요청 형식으로 감싸면 그 행의 정규식이
    # 정답 전체를 돌려줘야 한다. mcqa 같은 다른 채점기의 output_regex 는 보지 않는다.
    judged = collections.defaultdict(collections.Counter)
    eq_servers = set()
    for (split, name, vt), c in extraction.items():
        server = _equivalence_server(merged, name, vt)
        if server:
            judged[split].update(c)
            eq_servers.add(server)
    for split, c in judged.items():
        if c["truncated"]:
            err(f"{split} 데이터의 judge 행 {c['truncated']}개는 정답이 자기 output_regex 에서 잘린다 — 맞는 답도 보상 0 "
                "(tools/fix_stem_sci_regex.py 로 교체)")
        if c["unknown_format"]:
            warn(f"{split} 데이터의 judge 행 {c['unknown_format']}개는 output_regex 가 measure_judge_p04.WRAPPERS 에 없는 형식이라 "
                 "정답 추출을 검사하지 못했다")
    for server in sorted(eq_servers):
        eq = merged[server]["resources_servers"]["equivalence_llm_judge"]
        if not eq.get("check_full_generation_on_fail"):
            warn(f"{server}.check_full_generation_on_fail 가 꺼져 있다 — 모델 답 속 닫는 문자에서 추출이 잘린 정답도 보상 0 (결정 25)")
    for name, n in sorted(agents.items()):
        body = merged.get(name) or {}
        agent_cfgs = list((body.get("responses_api_agents") or {}).values())
        if not agent_cfgs:
            err(f"데이터의 agent {name} ({n}행)가 Gym 설정에 없다 — 롤아웃이 실패한다 (config_paths 에 서버 설정을 넣거나 행을 뺀다)")
            continue
        rs_name = ((agent_cfgs[0] or {}).get("resources_server") or {}).get("name")
        ns = ((merged.get(rs_name) or {}).get("resources_servers") or {}).get("ns_tools")
        if ns is None:
            continue
        known = set((ns.get("verifiers") or {}).keys())
        for vt, k in verifiers[name].items():
            if vt not in known:
                err(f"{name} 행 {k}개의 verifier_type={vt} 가 {rs_name}.verifiers 에 없다 — 'Unknown verifier' 로 채점이 실패한다")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    from omegaconf import OmegaConf

    from nemo_rl.utils.config import load_config, parse_hydra_overrides, register_omegaconf_resolvers

    register_omegaconf_resolvers()
    config = load_config(sys.argv[1])
    if len(sys.argv) > 2:
        config = parse_hydra_overrides(config, sys.argv[2:])
    cfg = OmegaConf.to_container(config, resolve=True)
    try:
        from nemo_rl.algorithms.grpo import MasterConfig

        MasterConfig(**cfg)
    except Exception as e:  # 스키마 위반은 실행 시에도 실패하므로 ERROR 로 올린다
        err(f"MasterConfig 스키마 검증 실패: {type(e).__name__}: {str(e)[:500]}")
    check(cfg)
    for w in WARNS:
        print(f"WARN  {w}")
    for e in ERRORS:
        print(f"ERROR {e}")
    print(f"check_alpha_recipe: {sys.argv[1]} — ERROR {len(ERRORS)} · WARN {len(WARNS)}")
    return 1 if ERRORS else 0


if __name__ == "__main__":
    sys.exit(main())
