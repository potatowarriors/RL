"""alpha GRPO 레시피 실행 전 검사 (CPU) — RLVR 이식 위험 보고(docs/RLVR_READINESS.md)의 조용한 실패 경로를 레시피 단계에서 막는다.

레시피를 NeMo-RL 과 같은 방식으로 읽는다(load_config → resolve → MasterConfig 스키마). 그 뒤 alpha 불변량을 검사한다.
ERROR 가 하나라도 있으면 종료 코드 1. WARN 은 결정 대기·확인 권고다.

  python examples/configs/alpha/tools/check_alpha_recipe.py examples/configs/alpha/student_rlvr1_alpha.yaml [hydra.override=값 ...]
  (NeMo-RL 드라이버 venv 로 실행: $NRL_ROOT/clean_run.sh $NRL_ROOT/bin/uv run --locked python ... </dev/null)

검사 항목 (괄호는 RLVR_READINESS.md 항목):
  GDN 재귀 상태 fp32 · R3 · turn_end 3 (CLAUDE.md 레시피 규약) · R3 + router fusion 금지 (H1) · 그룹 라우팅 덮어쓰기 금지 (낮음)
  CP>1 의 packing·패딩 배수·chunk·defer_fp32_logits (H2) · Gym 파서·invalid tool call 벌점 순서 (H3) · Gym 의 judge 미배치 (H4)
  prefix caching 명시 (M1) · async 배치 파라미터 명시 (M2) · async ⇒ IS 보정 (M7) · KL>0 + R3 (M6) · top_p 1.0 (낮음)
  GBS 나눗셈 (낮음) · Ultra 의 무효 키 (낮음) · reward_penalties token_ids 와 토크나이저 대조 · generation_config eos [3,0] (M9)
"""

import json
import os
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
# judge 모델이 있어야 보상이 나오는 Gym 서버 설정 (H4)
JUDGE_CONFIGS = ["genrm_compare", "abstention", "multichallenge", "jailbreak_detection", "equivalence_llm_judge", "terminus_judge"]
SANDBOX_CONFIGS = ["ns_tools", "math_formal_lean"]


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
        warn("force_on_policy_ratio + seq_logprob_error_threshold 없음 → prev_logprobs 를 건너뛴다. rollout↔train 정합 지표(R1)가 사라진다")

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
        warn("enforce_eager: false — R1 은 eager 로만 통과했다. CUDA graph 경로에서 R1 을 다시 잰다 (M4)")

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
            if any(f"/{name}/" in p for name in JUDGE_CONFIGS):
                warn(f"judge 모델이 필요한 환경: {p} — judge 배치 결정(D1) 없이는 보상이 나오지 않는다")
            if any(f"/{name}/" in p for name in SANDBOX_CONFIGS):
                warn(f"코드 sandbox 가 필요한 환경: {p} (D1)")
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
