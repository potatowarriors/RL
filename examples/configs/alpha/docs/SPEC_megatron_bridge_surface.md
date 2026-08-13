I have everything needed. Here's the technical report.

---

# Implementation Surface for an "alpha" (`AlphaForCausalLM`) Bridge in Megatron-Bridge

Repo root: `/home/work/vidsearch/repos/project_s/NeMo-RL/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge`
All paths below are relative to that root unless absolute.

---

## 1. Bridge registration mechanism

### 1.1 The dispatch registry

Registration is **decorator-driven over a custom single-dispatch table** — not a dict you edit, not entry points.

- Decorator: `MegatronModelBridge.register_bridge(...)` — `src/megatron/bridge/models/conversion/model_bridge.py:2260-2325`
- It delegates to `create_bridge_decorator()` — `model_bridge.py:2480-2512`, which:
  - sets `bridge_class.SOURCE_NAME` (str name of the HF arch), `bridge_class.MODEL_TYPE`, `bridge_class.PROVIDER_CLASS` (`model_bridge.py:2502-2508`)
  - calls `register_bridge_implementation()` — `model_bridge.py:2378-2477`
- `register_bridge_implementation` registers **four** dispatch impls keyed on `source` (and `(source, target)`):
  - `get_model_bridge.impl(source)` — `model_bridge.py:2395`
  - `stream_weights_megatron_to_hf.impl((source, target))` — `model_bridge.py:2404`
  - `stream_weights_megatron_to_hf_quant.impl((source, target))` — `model_bridge.py:2429`
  - `stream_adapter_weights_megatron_to_hf.impl((source, target))` — `model_bridge.py:2458`
- Dispatch engine: `src/megatron/bridge/models/decorators/dispatch.py`, class `_Dispatch`. Keys may be **types or plain strings**; `_Dispatch.__call__` matches exact type, then subclass, then by `__name__` string normalization (dispatch.py:52-110). The registry table is literally `get_model_bridge._exact_types`.

**Key fact for "alpha":** `source` accepts a **string**, so you do NOT need `AlphaForCausalLM` to exist in `transformers`:

```python
@MegatronModelBridge.register_bridge(source="AlphaForCausalLM", target=GPTModel, model_type="alpha")
```

Precedent: `src/megatron/bridge/models/deepseek/deepseek_v3_bridge.py:44-49` (`source="DeepseekV3ForCausalLM"`), `src/megatron/bridge/models/qwen/qwen35_bridge.py:138` (`source="Qwen3_5MoeForCausalLM"`), `src/megatron/bridge/models/bailing/bailing_moe2_bridge.py:71`.

### 1.2 How AutoBridge resolves an HF arch name → bridge

`src/megatron/bridge/models/conversion/auto_bridge.py`:

1. `AutoBridge.supports(config)` — `auto_bridge.py:382-398`: arch name must end with one of `SUPPORTED_HF_ARCHITECTURES` (`auto_bridge.py:69-80`): `"ForCausalLM"`, `"ForConditionalGeneration"`, `"ForMaskedLM"`, `"ForTokenClassification"`, plus a few explicit names. `AlphaForCausalLM` passes.
2. `AutoBridge._causal_lm_architecture` — `auto_bridge.py:2118-2181`:
   - picks the first arch ending with a supported suffix,
   - **tries `config.auto_map["AutoModelForCausalLM"]` first** → returns the class-name string (`get_causal_lm_class_name_via_auto_map`, `src/megatron/bridge/models/conversion/utils.py:358-373`),
   - else resolves via `HF_ARCHITECTURE_ALIASES` (`auto_bridge.py:95-97`) and `getattr(transformers, name)`,
   - else **falls back to the raw class-name string** (`auto_bridge.py:2177-2181`).
3. `AutoBridge._model_bridge` — `auto_bridge.py:2081-2091`: calls `model_bridge.get_model_bridge(arch, hf_config=hf_config)`; the impl instantiates the bridge and sets `bridge.hf_pretrained` / `bridge.hf_config` (`model_bridge.py:2396-2402`).
4. `_validate_config` — `auto_bridge.py:2183-2250`: if arch key not in `get_model_bridge._exact_types`, raises with a template that literally tells you to write `@MegatronModelBridge.register_bridge`.

So for a **trust_remote_code** alpha checkpoint whose `config.json` has `auto_map: {"AutoModelForCausalLM": "modeling_alpha.AlphaForCausalLM"}`, the dispatch key will be the string `"AlphaForCausalLM"` — exactly what you register.

### 1.3 Minimal class surface a new architecture must provide

Base class: `MegatronModelBridge` — `model_bridge.py:356-...`. Extension points documented at `model_bridge.py:427-452`.

| Required | Where | Notes |
|---|---|---|
| `@MegatronModelBridge.register_bridge(source=..., target=..., provider=..., model_type=...)` | decorator | `provider=` defaults to `GPTModelProvider` when `PROVIDER_CLASS is None` (`model_bridge.py:434`, used at `:777`) |
| `mapping_registry(self) -> MegatronMappingRegistry` | **abstract**, `model_bridge.py:880-910` | only truly required method |
| `provider_bridge(self, hf_pretrained)` | optional override, `model_bridge.py:738-804` | call `super()` then setattr |

Optional hooks (from `skills/adding-model-support/SKILL.md:219-230` and code):
- `CONFIG_MAPPING` (class attr, extend the base list) — `model_bridge.py:456-499`
- `hf_config_to_provider_kwargs()` — `model_bridge.py:651-653`
- `_should_map_hf_config_field()` — `model_bridge.py:544-546` (override to *suppress* a base CONFIG_MAPPING entry; examples: `src/megatron/bridge/models/glm_moe_dsa/glm5_bridge.py:57`, `src/megatron/bridge/models/gemma/gemma4_bridge.py:185`)
- `megatron_to_hf_config(cls, provider)` — `model_bridge.py:806-878`
- `maybe_modify_loaded_hf_weight()` — `model_bridge.py:993`
- `maybe_modify_converted_hf_weight()` — `model_bridge.py:1016`
- `MODEL_CONFIG_CLASS` / `TRANSFORMER_CONFIG_CLASS` — `model_bridge.py:439-440`; the newer builder-backed path (`hf_config_to_model_config`, `model_bridge.py:701-732`). **Note:** `provider_bridge()` emits a `FutureWarning` when `MODEL_CONFIG_CLASS is not None` (`model_bridge.py:759-767`). Hybrid/GDN families (qwen3_next, nemotron_h) still use the provider path.
- `SUPPORTS_HF_PRETRAINED_EXPORT`, `ADDITIONAL_FILE_PATTERNS` — `model_bridge.py:430, 444`

### 1.4 Out-of-tree extension path

**There is none.** There are no `[project.entry-points]` for bridges — the only entry-points group in `pyproject.toml:234` is `nemo_run.cli`. There is no `pkgutil.walk_packages` scan anywhere in `src/megatron/bridge`.

Registration happens purely as an **import side effect**:
- `src/megatron/bridge/__init__.py:21` → `import megatron.bridge.models  # triggers all bridge registrations`
- `src/megatron/bridge/models/__init__.py` explicitly imports each family package (e.g. `nemotronh` at `:146`, `deepseek` at `:31`). The `qwen` family is pulled in transitively (importing `megatron.bridge.models.qwen.qwen35_bridge` from `qwen_vl` executes `src/megatron/bridge/models/qwen/__init__.py`, which imports `qwen3_next_bridge` at `:18`).

**Practical consequence:** an out-of-tree `alpha` bridge works only if the consumer imports your module before calling `AutoBridge` (the decorator mutates the global `_Dispatch._exact_types` at import time). The supported path is in-package: `src/megatron/bridge/models/alpha/` + an import line in `src/megatron/bridge/models/__init__.py`. A registration contract test will also need the new entry: `tests/unit_tests/models/test_autobridge_registration_matrix.py:31-...` (`EXPECTED_REGISTRATIONS` dict, plus the `STRING_REGISTRATIONS` list at `:99-125` if you register by string), checked in a fresh interpreter by `tests/unit_tests/models/autobridge_registration_check.py`.

---

## 2. Qwen3-Next bridge, end to end

File: `src/megatron/bridge/models/qwen/qwen3_next_bridge.py` (250 lines, the single most relevant reference for alpha).

### 2.1 Registration & target

`qwen3_next_bridge.py:36`
```python
@MegatronModelBridge.register_bridge(source=Qwen3NextForCausalLM, target=GPTModel, model_type="qwen3_next")
```
Target is **`GPTModel`, not `MambaModel`/`HybridModel`**. Hybridity comes from a per-layer transformer *block spec*, not from a different model class.

### 2.2 Config translation (`provider_bridge`, `qwen3_next_bridge.py:52-96`)

Base fields come from `CONFIG_MAPPING` (`model_bridge.py:456-499`) — `num_hidden_layers→num_layers`, `hidden_size`, `intermediate_size→ffn_hidden_size`, `num_attention_heads`, `num_key_value_heads→num_query_groups`, `head_dim→kv_channels`, `vocab_size`, `max_position_embeddings→seq_length`, `rms_norm_eps→layernorm_epsilon`, `rope_theta→rotary_base`, `partial_rotary_factor→rotary_percent`, `num_experts→num_moe_experts`, `num_experts_per_tok→moe_router_topk`, `moe_intermediate_size→moe_ffn_hidden_size`.

Then, model-specific:

| Line | Provider field | Source |
|---|---|---|
| `:58-66` | `normalization="RMSNorm"`, `gated_linear_unit=True`, `position_embedding_type="rope"`, `add_bias_linear=False`, `add_qkv_bias=False`, `qk_layernorm=True` | constants |
| `:66` | `share_embeddings_and_output_weights` | `hf_config.tie_word_embeddings` |
| `:69-77` | `moe_grouped_gemm=True`, `moe_router_load_balancing_type="global_aux_loss"`, `moe_aux_loss_coeff=1e-3`, `moe_router_pre_softmax=False`, `moe_token_dispatcher_type="alltoall"`, `moe_permute_fusion=True`, `moe_shared_expert_gate=True`, `moe_router_dtype="fp32"` | constants |
| `:77` | `moe_shared_expert_intermediate_size` | `hf_config.shared_expert_intermediate_size` |
| `:80-81` | `layernorm_zero_centered_gamma=True`, `attention_output_gate=True` | Qwen3-Next specifics |
| `:84` | `transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec` | the hybrid spec fn |
| `:85` | `experimental_attention_variant = "gated_delta_net"` | |
| `:86` | `linear_attention_freq = full_attention_interval_from_hf(hf_config)` | int **or per-layer list** |
| `:87-91` | `linear_conv_kernel_dim`, `linear_key_head_dim`, `linear_value_head_dim`, `linear_num_key_heads`, `linear_num_value_heads` | 1:1 from HF fields of the same names |
| `:94` | `hetereogenous_dist_checkpoint = True` | required for mixed-layer-type checkpoints (`3rdparty/Megatron-LM/megatron/core/transformer/transformer_config.py:1211`, consumed at `transformer_block.py:733`) |

**The hybrid pattern helper** — `src/megatron/bridge/models/conversion/transformers_compat.py:188-227`, `full_attention_interval_from_hf(config, default=4)`:
- transformers ≥5.5: reads `config.layer_types`, maps `{"linear_attention": 1, "full_attention": 0}` → **explicit per-layer int list**
- transformers <5.5: returns the scalar `config.full_attention_interval`
- raises `ValueError` on unknown layer type strings.

This is exactly the hook you would extend for alpha's `M-M-M-*-` string (see §4.3).

### 2.3 Weight mapping table (`mapping_registry`, `qwen3_next_bridge.py:98-249`)

Simple 1:1 dict (`:104-135`) → `AutoMapping` (`:139-140`), plus two `AutoMapping.register_module_type` calls that teach TP-parallelism inference about non-standard modules (`:141-142`):
```python
AutoMapping.register_module_type("SharedExpertMLP", "column")
AutoMapping.register_module_type("GatedDeltaNet", "column")
```

GDN-relevant entries:

| Megatron | HF | Line |
|---|---|---|
| `decoder.layers.*.self_attention.in_proj.layer_norm_weight` | `model.layers.*.input_layernorm.weight` | `:118` |
| `decoder.layers.*.self_attention.out_proj.weight` | `model.layers.*.linear_attn.out_proj.weight` | `:119` |
| `decoder.layers.*.self_attention.A_log` | `model.layers.*.linear_attn.A_log` | `:120` |
| `decoder.layers.*.self_attention.dt_bias` | `model.layers.*.linear_attn.dt_bias` | `:121` |
| `decoder.layers.*.self_attention.conv1d.weight` | `model.layers.*.linear_attn.conv1d.weight` | `GDNConv1dMapping`, `:163-166` |
| `decoder.layers.*.self_attention.in_proj.weight` | `{qkvz: …in_proj_qkvz.weight, ba: …in_proj_ba.weight}` | `GDNLinearMapping`, `:167-171` |
| `decoder.layers.*.self_attention.out_norm.weight` | `model.layers.*.linear_attn.norm.weight` | `RMSNorm2ZeroCenteredRMSNormMapping`, `:242-245` — subtracts 1 because Qwen3-Next's GDN out-norm is a plain RMSNorm while all other norms are zero-centered |

Attention: standard `QKVMapping` at `:149-154`. MoE experts are declared **twice**, once for GroupedMLP and once for SequentialMLP (`:173-191`) — `experts.linear_fc1.weight*` / `experts.local_experts.*.linear_fc1.weight`. Shared expert `:212-229`; shared-expert gate as `ReplicatedMapping` `decoder.layers.*.mlp.shared_experts.gate_weight ← model.layers.*.mlp.shared_expert_gate.weight` (`:231-234`). Full MTP mirror set at `:122-134, 155-160, 192-238`.

### 2.4 Recipe / functional test

- Recipe: `src/megatron/bridge/recipes/qwen/h100/qwen3_next.py:37` builds the model via `AutoBridge.from_hf_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct").to_megatron_provider(...)` — no hand-written provider. `src/megatron/bridge/recipes/qwen/qwen3_next.py` is only an alias shim.
- Toy-model roundtrip test: `tests/functional_tests/test_groups/models/qwen/test_qwen3_next_conversion.py:25-60` — the toy config there is the best template for an alpha toy config.

### 2.5 Qwen3.5 variant (second reference)

`src/megatron/bridge/models/qwen/qwen35_bridge.py`:
- `_apply_qwen35_common_config` `:40-77` and `_apply_qwen35_moe_config` `:80-102` — same GDN fields, read defensively with `getattr(..., default)`.
- `_moe_routed_expert_mappings` `:105-136` — handles **both** per-expert HF tensors and *packed* HF expert tensors (`experts.gate_up_proj` / `experts.down_proj` of shape `[num_experts, …]`) via `FusedGatedExpertMapping` / `FusedExpertMapping`.
- GDN in_proj stored as **four separate** HF tensors → `GDNLinearMappingSeparate(qkv=, z=, b=, a=)` `:257-262`.

---

## 3. DeepSeek-V3 router mapping

Config: `src/megatron/bridge/models/deepseek/deepseek_v3_bridge.py:44-99`; weights: `src/megatron/bridge/models/deepseek/common.py`.

### 3.1 Router config → mcore `TransformerConfig` fields

Automatic via base `CONFIG_MAPPING` (`model_bridge.py:479-489`):

| HF field | mcore field |
|---|---|
| `n_routed_experts` | `num_moe_experts` |
| `num_experts_per_tok` | `moe_router_topk` |
| `moe_intermediate_size` | `moe_ffn_hidden_size` |
| `scoring_func` | `moe_router_score_function` |
| `n_group` | `moe_router_num_groups` |
| `topk_group` | `moe_router_group_topk` |
| `routed_scaling_factor` | `moe_router_topk_scaling_factor` |
| `aux_loss_alpha` / `router_aux_loss_coef` | `moe_aux_loss_coeff` |
| `num_nextn_predict_layers` | `mtp_num_layers` |

Explicit setattrs in `provider_bridge` (`deepseek_v3_bridge.py:65-97`):
```
moe_router_pre_softmax = True          # :66
moe_router_load_balancing_type = "seq_aux_loss"   # :68
moe_aux_loss_coeff = getattr(hf_config, "aux_loss_alpha", 1e-4)  # :71
moe_shared_expert_overlap = True       # :72
moe_router_score_function = "sigmoid"  # :73   <-- sigmoid scoring
moe_router_enable_expert_bias = True   # :74   <-- e_score_correction_bias
moe_router_dtype = "fp32"              # :75
moe_grouped_gemm = True                # :65
moe_layer_freq = [0]*first_k_dense_replace + [1]*(L - first_k_dense_replace)  # :92-94
moe_shared_expert_intermediate_size = moe_intermediate_size * n_shared_experts  # :95
mtp_num_layers = num_nextn_predict_layers or None  # :97
make_vocab_size_divisible_by = 1280    # :90
```

The mcore field definitions live in `3rdparty/Megatron-LM/megatron/core/transformer/transformer_config.py`:
- `moe_router_num_groups` `:759`, `moe_router_group_topk` `:775`, `moe_router_pre_softmax` `:778`, `moe_router_topk_scaling_factor` `:783`, `moe_router_score_function` `:787` (`Literal['softmax','sigmoid','sqrtsoftplus']`), `moe_router_enable_expert_bias` `:795`, `moe_router_bias_update_rate` `:800`, `moe_shared_expert_gate` `:692`.
- Validation: expert bias requires sigmoid/sqrtsoftplus (`:2310`); group-limited topk requires `moe_router_num_groups` and `num_moe_experts % num_groups == 0`, `group_topk <= num_groups` (`:2364-2381`).

### 3.2 Expert-bias tensor

`src/megatron/bridge/models/deepseek/common.py:47`
```
"decoder.layers.*.mlp.router.expert_bias" : "model.layers.*.mlp.gate.e_score_correction_bias"
```

### 3.3 Shared expert (no gate in DSV3)

`common.py:50, 86-90` — `mlp.shared_experts.linear_fc1/fc2` ← `mlp.shared_experts.{gate,up,down}_proj`. **No** `shared_expert_gate`; that is a Qwen3-Next-only thing (`qwen3_next_bridge.py:75, 231-234` sets `moe_shared_expert_gate = True` and maps `mlp.shared_experts.gate_weight`).

### 3.4 Export direction

`megatron_to_hf_config` (`deepseek_v3_bridge.py:101-127`) has to un-`None` DSV3 fields HF cannot take as `None`: `num_nextn_predict_layers`, `n_group`, `topk_group` → default 1/0 (`:106-108`); reconstructs `first_k_dense_replace` from `moe_layer_freq` (`:110-119`) and `n_shared_experts` from the intermediate-size ratio (`:121-125`). Alpha will need the same treatment.

### 3.5 Other DSV3-style routers to crib from

`src/megatron/bridge/models/glm/glm45_bridge.py:69-93` (sigmoid + expert bias + `moe_router_bias_update_rate = 0` for a frozen bias at import), `src/megatron/bridge/models/bailing/bailing_moe2_bridge.py:97-98`, `src/megatron/bridge/models/nemotronh/nemotron_h_bridge.py:310-318`, `src/megatron/bridge/models/deepseek/deepseek_v4_bridge.py:487-488` (`scoring_func = "sqrtsoftplus"` passed through from config).

---

## 4. Vendored Megatron-LM (`3rdparty/Megatron-LM`) capabilities

### 4.1 GatedDeltaNet is native — two independent stacks support it

**(A) GPT + experimental-attention-variant block spec** (what Qwen3-Next uses)

`3rdparty/Megatron-LM/megatron/core/models/gpt/experimental_attention_variant_module_specs.py`:
- `get_gated_delta_net_module_spec()` `:60-78` — `GatedDeltaNet` with `GatedDeltaNetSubmodules(in_proj=column_parallel_layer_norm_linear, out_norm=layer_norm, out_proj=row_parallel_linear)`, `metainfo={"fuse_input_layernorm": True}`
- `get_transformer_layer_with_experimental_attention_variant_spec()` `:152-...` — **attention pattern and MLP pattern are orthogonal**:
  - attention pattern from `config.linear_attention_freq` via `get_linear_attention_pattern()` `:392-425` (int N ⇒ every Nth layer is full attention; or an explicit per-layer 0/1 list of length `num_layers`)
  - MLP pattern from `config.moe_layer_freq` via `get_moe_layer_pattern()` `:369-390` (int or per-layer 0/1 list)
- `get_transformer_block_with_experimental_attention_variant_spec()` `:258-...` — PP slicing, `pipeline_model_parallel_layout` aware.
- Backend assertion: `_get_backend_spec_provider` `:437-454` requires `config.transformer_impl == "transformer_engine"`.

**(B) HybridModel + pattern string** (what Nemotron-H uses)

`3rdparty/Megatron-LM/megatron/core/models/hybrid/`:
- `hybrid_layer_allocation.py:14-37` — `class Symbols`: `MAMBA="M"`, **`GDN="G"`**, `ATTENTION="*"`, `DS_ATTENTION="D"`, `MLA="+"`, `MLP="-"`, **`MOE="E"`**, `PIPE="|"`, `MTP_SEPARATOR="/"`. `VALID_LAYERS = {M,G,*,D,+,-,E}`.
- `ParsedHybridPattern` `:39-72` — pattern format `"<main>/<mtp>/<mtp>…"`, `|` for PP stage boundaries. Example in the docstring: `"M-M-|M-M*-/MM/MM"`.
- `hybrid_block.py:40-52` — `HybridStackSubmodules(mamba_layer, gdn_layer, attention_layer, dsa_layer, mla_layer, mlp_layer, moe_layer, mtp_block_spec)`; dispatch loop `:134-204` with an explicit `elif layer_type == LayerSymbols.GDN:` branch at `:193`.
- `hybrid_layer_specs.py:105-118` — `gdn_layer = ModuleSpec(module=TransformerLayer, submodules=TransformerLayerSubmodules(self_attention=ModuleSpec(module=GatedDeltaNet, submodules=GatedDeltaNetSubmodules(in_proj=TELayerNormColumnParallelLinear, out_norm=TENorm, out_proj=TERowParallelLinear))))`; `moe_layer` at `:209-213`, `mlp_layer` at `:197`.

**So yes — the vendored MCore already supports per-layer pattern strings containing `M`, `G`, `*`, `-`, `E`, and MoE inside a hybrid/Mamba stack.** Note the GDN layer in the hybrid stack is a `TransformerLayer` whose `self_attention` is `GatedDeltaNet`, so its **param names are identical** to the GPT path: `decoder.layers.N.self_attention.{in_proj,conv1d,A_log,dt_bias,out_norm,out_proj}` — the Qwen3-Next mapping table transfers verbatim. (Mamba layers, by contrast, live under `.mixer.` — see `nemotron_h_bridge.py:411`.)

Bridge-side wrapper: `src/megatron/bridge/models/hybrid/hybrid_provider.py`, `class HybridModelProvider(TransformerConfig, ModelProviderMixin[MCoreHybridModel])` `:108`, with `hybrid_layer_pattern` (`hybrid_override_pattern` is deprecated, `:128, 179-194`), `mtp_hybrid_override_pattern` `:164`, `finalize()` deriving `num_layers` from the pattern `:170-250`, `_resolve_hybrid_stack_spec()` `:253`. Used by `NemotronHBridge` (`src/megatron/bridge/models/nemotronh/nemotron_h_bridge.py:233-238`, `provider=HybridModelProvider`, `target=HybridModel`) and by `src/megatron/bridge/recipes/nemotronh/h100/nemotron_3_nano.py:80` with pattern `"MEMEM*EMEMEM*EMEMEM*EM…"` — i.e. **Mamba + attention + MoE mixed in one pattern string is already exercised**.

### 4.2 GDN runtime kernel dependencies

`3rdparty/Megatron-LM/megatron/core/ssm/gated_delta_net/common.py:43-54`:
```python
try:
    from fla.modules.convolution import causal_conv1d
    from fla.modules.l2norm import l2norm
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    HAVE_FLA = True
except ImportError:
    ...
    HAVE_FLA = False
```
`common.py:141-145` raises at `GatedDeltaNet.__init__` if `not HAVE_FLA`: *"FLA is not installed. Please install it with `pip install flash-linear-attention[cuda]`."*

`gdn.py:16-27` additionally imports `torch_chunk_gated_delta_rule`; `gdn.py:58-62` selects it when `config.deterministic_mode` is set (pure-torch fallback for the delta rule only — the conv and l2norm still come from `fla`).

Declared in `pyproject.toml`: `flash-linear-attention` at `:95` and `:121`; `mamba-ssm` / `causal-conv1d` at `:117-118` and `:178-179` (comment at `:112` says these are optional extras `[ssm]`, `[te]`).

### 4.3 GDN tensor geometry (for alpha's config translation)

`common.py:174-175, 195-196, 220`, `gdn.py:32-58`:
- `qk_dim = linear_key_head_dim * linear_num_key_heads`
- `v_dim = linear_value_head_dim * linear_num_value_heads`
- `in_proj_qkvg_dim = 2*qk_dim + 2*v_dim`; `in_proj_extra_dim = 2 * linear_num_value_heads` (beta, alpha) ⇒ `in_proj_dim = 2*qk_dim + 2*v_dim + 2*num_v_heads`
- `conv_dim = 2*qk_dim + v_dim` (Q,K,V only — Z is not convolved)
- `dt_bias_dim = a_log_dim = linear_num_value_heads` (per-TP-rank)
- in_proj split order: `["query","key","value","z","beta","alpha"]` (`gdn.py:39-48`)

Config field declarations + validation: `3rdparty/Megatron-LM/megatron/core/transformer/transformer_config.py:294` (`experimental_attention_variant: Literal['gated_delta_net','dsa']`), `:349` (`linear_attention_freq: Optional[Union[int, List[int]]]`), `:356-368` (the five GDN dims), `:271` (`attention_output_gate`), and asserts at `:1347-1381` (all five must be set; `num_value_heads % num_key_heads == 0`; both head counts divisible by TP×CP).

---

## 5. Weight-mapping abstractions

Declarative mapping objects with wildcard patterns, collected in a `MegatronMappingRegistry`.

- Base: `MegatronParamMapping(ABC, Generic[WeightType])` — `src/megatron/bridge/models/conversion/param_mapping.py:56-...`. Subclass contract: `hf_to_megatron(hf_weights, megatron_module)`, `megatron_to_hf(megatron_weights, megatron_module)`, `resolve(captures)`. Provides TP/PP/EP helpers (`broadcast_from_pp_rank`, `scatter_to_tp_ranks`, `gather_from_tp_ranks`, …) and process-group plumbing (`param_mapping.py:100-160`, `set_process_groups_from_pg_collection` `:133`).
- Registry: `src/megatron/bridge/models/conversion/mapping_registry.py:23-...`. Pattern semantics (`_convert_pattern_to_regex`, `:124-...`): `*` → `(\d+)` (digits only, layer/expert indices); `**` → `(.*)`. It also auto-adds fused↔separate LayerNorm aliases (`_SEPARATE_LAYERNORM_REWRITES` `:72-83`, including `"mixer.in_proj.layer_norm_weight" ↔ "norm.weight"`) and quantization amax mappings (`:118-122`).
- Class inventory (`param_mapping.py`): `DirectMapping :890`, `ColumnParallelMapping :919`, `RowParallelMapping :1111`, `ReplicatedMapping :1270`, `AutoMapping :1340`, `QKVMapping :1654`, `QKVGMapping :1870`, `KVMapping :1950`, `MambaInProjMapping :2034`, `ChunkedMapping :2140`, `GDNConv1dMapping :2228`, `MambaConv1dMapping :2250`, `GDNLinearMapping :2273`, `GDNLinearMappingSeparate :2355`, `ConcatenatedQKVMapping :2458`, `GatedMLPMapping :2591`, `RMSNorm2ZeroCenteredRMSNormMapping :2839`, `FusedExpertMapping :2910`, `FusedGatedExpertMapping :2958`.
- `AutoMapping` infers column/row/replicated from the owning module's class name via `_MODULE_TYPE_REGISTRY` (`param_mapping.py:1399-1436`); extend with `AutoMapping.register_module_type(name, "column"|"row"|"replicated")` (`:1438-1451`). **Alpha will need `register_module_type("GatedDeltaNet", "column")`** (and `"SharedExpertMLP"` if it has a shared expert) exactly as `qwen3_next_bridge.py:141-142`.

### 5.1 Concrete example — fused QKV interleave

`param_mapping.py:1654-1786`, `class QKVMapping`. Declaration (`qwen3_next_bridge.py:149-154`):
```python
QKVMapping(
    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
    q="model.layers.*.self_attn.q_proj.weight",
    k="model.layers.*.self_attn.k_proj.weight",
    v="model.layers.*.self_attn.v_proj.weight",
)
```
Mechanics: `__init__ :1702-1716` builds `self._tp_mapping = AutoMapping(megatron_param, megatron_param)` — format conversion is deliberately decoupled from TP mechanics. `hf_to_megatron :1718-1740` merges on `tp_rank == 0` only (`merge_qkv_weights` / `merge_qkv_biases` for 1-D), then delegates scatter to `_tp_mapping`. `megatron_to_hf :1741-1782` broadcasts a pickle-safe config across PP (`remove_non_pickleables(config, max_depth=3)`), gathers TP shards, then `split_qkv_weights`. Megatron layout documented at `:1668-1672`: `[q1..qn, k1, v1, q1..qn, k2, v2, …]`, `n = num_attention_heads / num_query_groups`.

### 5.2 Concrete example — MoE expert stacking

Two layouts must both be declared, because Megatron may use GroupedMLP or SequentialMLP:

Per-expert HF tensors (`qwen3_next_bridge.py:173-191`):
```python
GatedMLPMapping(megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                up  ="model.layers.*.mlp.experts.*.up_proj.weight")
AutoMapping(megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
            hf_param="model.layers.*.mlp.experts.*.down_proj.weight")
# … then the same pair for "experts.local_experts.*.linear_fc{1,2}.weight"
```

Packed HF tensors `[num_experts, …]` — `FusedExpertMapping` / `FusedGatedExpertMapping` (`param_mapping.py:2910-2957` / `:2958-3010`), selected at `qwen35_bridge.py:105-136`. Key protocol: `is_grouped_export = True` and a `group_key` property (`:2942-2945`); import extracts one expert (`hf_weights[expert_idx]`, `:2946-2952` via `extract_expert_number_from_param`), export accumulates per-expert tensors in `_grouped_buffers` and emits the stacked HF tensor once complete (`model_bridge.py:1560-1580`, `_accumulate_grouped_export`).

Router/bias/gate scalars: `AutoMapping("…mlp.router.weight" ← "…mlp.gate.weight")`, `AutoMapping("…mlp.router.expert_bias" ← "…mlp.gate.e_score_correction_bias")` (`deepseek/common.py:46-47`), `ReplicatedMapping("…mlp.shared_experts.gate_weight" ← "…mlp.shared_expert_gate.weight")` (`qwen3_next_bridge.py:231-234`).

### 5.3 Concrete example — a GDN tensor

`GDNLinearMapping` — `param_mapping.py:2273-2353`:
```python
GDNLinearMapping(
    megatron_param="decoder.layers.*.self_attention.in_proj.weight",
    qkvz="model.layers.*.linear_attn.in_proj_qkvz.weight",
    ba  ="model.layers.*.linear_attn.in_proj_ba.weight",
)
```
- `hf_to_megatron :2292-2306` → `merge_gdn_linear_weights(config, qkvz, ba, tp_size)` on rank 0, then `_tp_mapping` scatter.
- `merge_gdn_linear_weights` `param_mapping.py:3502-3551`: reshapes `qkvz` to `(num_qk_heads, (2*qk_dim+2*v_dim)/num_qk_heads, hidden)` and `ba` to `(num_qk_heads, 2*num_v_heads/num_qk_heads, hidden)`, splits into `q,k,v,z,b,a`, reshapes each to `(tp_size, -1, hidden)`, concatenates on dim 1 → the TP-interleaved packed `in_proj`. Round-trip inverse `split_gdn_linear_weights` `:3553-3607` (has a `feature_dim` param so LoRA ranks work).
- `megatron_to_hf :2307-2343` — broadcasts config across PP before gathering, so PP ranks that early-return still participate in the collective.

`GDNConv1dMapping(ChunkedMapping)` — `param_mapping.py:2228-2248`: `get_shard_idx` returns `[q_idx, k_idx, v_idx]` computed from `linear_key_head_dim*linear_num_key_heads` and `linear_value_head_dim*linear_num_value_heads`, dividing by `tp_size` when `local_tp`. (Mamba analogue `MambaConv1dMapping` at `:2250-2271` shards by `mamba_num_heads*mamba_head_dim` / `mamba_state_dim*mamba_num_groups`; `MambaInProjMapping` at `:2034-2138`; Nemotron-H's Mamba mixer mappings at `nemotron_h_bridge.py:507-520`.)

`RMSNorm2ZeroCenteredRMSNormMapping(AutoMapping)` — `param_mapping.py:2839`; used for GDN's out-norm at `qwen3_next_bridge.py:242-245`.

---

## 6. Conversion / verification tooling you can reuse

### 6.1 CLI

`scripts/conversion/convert.sh` — NeMo-Run based, subcommands `import`, `export`, **`roundtrip`** (`scripts/conversion/arguments.py:186-206, 292-297, 311-313`; dispatch `scripts/conversion/run_conversion.py:111-145`). Backends `--device cpu` (`cpu_backend.py`) and `--device gpu` with TP/PP/EP/ETP (`gpu_backend.py`). README: `scripts/conversion/README.md`.

`roundtrip_checkpoint(...)` — `scripts/conversion/gpu_backend.py:415-453`: builds the bridge from HF, configures parallelism, `provide_distributed_model(wrap_with_ddp=False)`, then `_verify_roundtrip_weights(bridge, megatron_model)`.

There is **no** `python -m megatron.bridge` module entry point (`pyproject.toml` declares no `[project.scripts]`).

### 6.2 Python-level verification (`examples/conversion/`)

Per `skills/parity-testing/SKILL.md:16-30`:

| Purpose | Tool |
|---|---|
| Exact single-GPU weight round-trip | `examples/conversion/hf_megatron_roundtrip.py` |
| Round-trip under TP/PP/EP | `examples/conversion/hf_megatron_roundtrip_multi_gpu.py` |
| Forward-pass logit correlation (HF vs Megatron) | `examples/conversion/compare_hf_and_megatron/compare.py` (+ `debugger.py` for per-layer hooks via `--enable_debug_hooks`) |
| Generation sanity | `examples/conversion/hf_to_megatron_generate_text.py` |
| Programmatic table | `weights_verification_table(bridge, megatron_model)` — `src/megatron/bridge/models/conversion/utils.py:84-112` |
| Registry introspection | `examples/conversion/list_supported_architectures.py`; `AutoBridge.list_supported_models()` — `auto_bridge.py:361-380` |
| Benchmarks / MFSDP / adapters | `hf_megatron_roundtrip_benchmark.py`, `mfsdp/`, `adapter/` |

Pass bars from `skills/parity-testing/SKILL.md:35-80`: Level 1 round-trip must be **exactly** `max_diff == 0.0`; Level 2 logits must match on next-token and have **cosine similarity ≥ 0.99** (absolute logit diffs are diagnostic only).

Other relevant skills: `skills/create-model-verification-card/SKILL.md` (produces `examples/model_verification_cards/<slug>/card.yaml`, validated by `scripts/validate_card.py`), `skills/adding-model-support/{SKILL,llm-patterns,tests-and-examples}.md`.

### 6.3 Test scaffolding to copy

- Unit: `tests/unit_tests/models/<model>/test_<model>_bridge.py` (mock HF config → assert provider fields). Nearest analogs: `tests/unit_tests/models/deepseek/test_deepseek_bridges.py`, `tests/unit_tests/models/falcon_h1/test_falcon_h1_bridge.py`.
- Functional (toy HF model → Megatron → HF): `tests/functional_tests/test_groups/models/qwen/test_qwen3_next_conversion.py` — toy config dict at `:25-60`.
- Registration contract: `tests/unit_tests/models/test_autobridge_registration_matrix.py` + `tests/unit_tests/models/autobridge_registration_check.py` (fresh-interpreter subprocess check).
- Mapping unit tests: `tests/unit_tests/models/test_param_mapping.py`, `test_mapping_registry.py`.
- FLOPs: if alpha's GDN ratio differs, update `src/megatron/bridge/training/utils/flop_utils.py:1127-1186` (`experimental_attention_variant == "gated_delta_net"` branch; also `gdn_layer_flops` at `:545-576` and hybrid `Symbols.GDN` layer counting at `:405-422`), with tests in `tests/unit_tests/training/utils/test_flop_utils.py` (per `skills/adding-model-support/SKILL.md:277-306`).

---

## 7. Vocab padding and tied embeddings

### 7.1 Import direction (HF → Megatron)

- `MegatronModelBridge.make_vocab_size_divisible_by(vocab_size)` — `model_bridge.py:1716-1747`: returns the **largest power of two ≤ 128 that exactly divides `vocab_size`** (starts at 128, halves until `vocab_size % base == 0`). Set automatically in `_hf_config_to_megatron_kwargs` `:634-637`. Bridges may override the result (DeepSeek-V3 forces `1280` at `deepseek_v3_bridge.py:90`).
- `vocab_size` itself is a straight CONFIG_MAPPING passthrough (`model_bridge.py:464`).
- The actual padding happens in the provider: `src/megatron/bridge/models/gpt_provider.py:264-270`
  ```python
  assert self.vocab_size is not None
  if self.should_pad_vocab:
      padded_vocab_size = calculate_padded_vocab_size(
          self.vocab_size, self.make_vocab_size_divisible_by, self.tensor_model_parallel_size)
  else:
      padded_vocab_size = self.vocab_size
  ```
  `should_pad_vocab: bool = False` is the **default** (`gpt_provider.py:183`; same default in `hybrid_provider.py:153`, `t5_provider.py:94`, `falconh1_provider.py:81`). So by default the bridge builds Megatron with `padded_vocab_size == HF vocab_size` and no padding at all. `calculate_padded_vocab_size` is re-exported from Megatron-LM (`src/megatron/bridge/utils/vocab_utils.py:15-18` → `megatron.training.vocab_utils`). `MCoreGPTModel` receives `vocab_size=padded_vocab_size` at `gpt_provider.py:296`.

### 7.2 Export direction (Megatron → HF): truncation

`_truncate_vocab_padding` — `model_bridge.py:1048-1121`, invoked in the export loop at `model_bridge.py:1613`.
- Gate: `_is_vocab_export_task` (`:1035-1046`) matches Megatron suffixes `embedding.word_embeddings.weight` / `output_layer.weight` **and** HF suffixes `embed_tokens.weight` / `word_embeddings.weight` / `lm_head.weight` / `head.weight`.
- Reads `vocab_size` from `self.hf_config`, walking nested `text_config` / `llm_config` / `thinker_config` if absent (`:1066-1083`).
- Slices `tensor[:vocab_size]` for any tensor whose row count equals the padded size (`:1110-1120`); FP8 `_scale_inv` tensors are truncated to `ceil(vocab_size / scale_block_size)` (`:1087-1105, 1116`).

### 7.3 Tied embeddings

- Import config: `("tie_word_embeddings", "share_embeddings_and_output_weights")` in the base `CONFIG_MAPPING` (`model_bridge.py:471`). Bridges commonly re-assert it explicitly — `qwen3_next_bridge.py:66`: `provider.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)`. DeepSeek-V3 hard-codes `False` (`deepseek_v3_bridge.py:61`).
- The provider forwards it to `MCoreGPTModel(share_embeddings_and_output_weights=…)` at `gpt_provider.py:300`; provider default is `True` (`gpt_provider.py:139`), so **always set it explicitly**.
- Export: `_share_embeddings_and_output_weights(model_config)` — `model_bridge.py:1754-1759`. When tied, the export loop builds a `tied_mapping_registry` (`model_bridge.py:1536`) and calls `_get_tied_output_hf_name(task, registry)` (`model_bridge.py:1848-1874`): for the embedding task it looks up `<prefix>output_layer.weight` in the registry and, if that resolves to a *different* HF name (e.g. `lm_head.weight`), emits the same tensor a second time under that name — but only if the target checkpoint's key set actually contains it (`model_bridge.py:1626-1640`, `hf_pretrained.state.source.get_all_keys()`), or unconditionally when exporting from a bare `PretrainedConfig`.
- VLM caveat (from `skills/adding-model-support/SKILL.md:154-160`): `tie_word_embeddings` lives on the **top-level** HF config, not `text_config`.

---

## 8. Recommended implementation shape for `alpha`

**Choice of Megatron stack.** Two viable routes; (A) is lower-risk.

**(A) GPTModel + experimental-attention-variant spec** — clone `qwen3_next_bridge.py`.
- `target=GPTModel`, `provider` default `GPTModelProvider`.
- `provider.transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec`; `experimental_attention_variant = "gated_delta_net"`.
- Translate alpha's `M-M-M-*-` string into **two orthogonal per-layer lists**, since attention and MLP patterns are independent in this spec (`experimental_attention_variant_module_specs.py:152-256`):
  - `linear_attention_freq = [1 if ch in "MG" else 0 for ch in pattern]` (1 = GDN/linear, 0 = full attention) — length must equal `num_layers` (`get_linear_attention_pattern`, `:392-425`).
  - `moe_layer_freq = [0 if ch == "-" else 1 for ch in pattern]` if alpha's `-` means dense MLP and the rest are MoE (`get_moe_layer_pattern`, `:369-390`).
  - Add a helper alongside `full_attention_interval_from_hf` in `transformers_compat.py:188-227`, or keep it private in the alpha bridge module (the skill explicitly prefers local, `SKILL.md:169-179`).
- Set `hetereogenous_dist_checkpoint = True` (`qwen3_next_bridge.py:94`).
- DSV3 routing: `moe_router_score_function="sigmoid"`, `moe_router_enable_expert_bias=True`, `moe_router_num_groups`/`moe_router_group_topk` from `n_group`/`topk_group` (auto via CONFIG_MAPPING), `moe_router_topk_scaling_factor` from `routed_scaling_factor` (auto), plus `moe_router_bias_update_rate = 0` if the checkpoint's bias is frozen (`glm45_bridge.py:78`).

**(B) HybridModel + `HybridModelProvider`** — clone `nemotron_h_bridge.py:233-...`. Pass alpha's pattern almost verbatim as `hybrid_layer_pattern`, translating `M`→`G` (GDN, not Mamba) and choosing `E`/`-` for MoE/dense. `target=HybridModel`, `provider=HybridModelProvider`. This gets you `|` PP-layout control and `/` MTP patterns for free, and GDN param names are unchanged. Cost: fewer existing GDN+MoE bridges use this path, so less precedent.

**Files to create/touch**
```
src/megatron/bridge/models/alpha/__init__.py          # export AlphaBridge
src/megatron/bridge/models/alpha/alpha_bridge.py      # @register_bridge(source="AlphaForCausalLM", target=GPTModel, model_type="alpha")
src/megatron/bridge/models/__init__.py                # add `from megatron.bridge.models.alpha import AlphaBridge` + __all__
tests/unit_tests/models/test_autobridge_registration_matrix.py  # EXPECTED_REGISTRATIONS (+ STRING_REGISTRATIONS if string-registered)
tests/unit_tests/models/alpha/test_alpha_bridge.py
tests/functional_tests/test_groups/models/alpha/test_alpha_conversion.py
src/megatron/bridge/training/utils/flop_utils.py      # only if the GDN/MoE ratio math differs from :1127-1186
docs/models/llm/alpha.md, examples/models/alpha/README.md
```

**Mapping table starting point:** copy `qwen3_next_bridge.py:104-247` wholesale, then adjust (a) the GDN in_proj form — `GDNLinearMapping` if alpha ships fused `in_proj_qkvz`/`in_proj_ba`, `GDNLinearMappingSeparate` if it ships `in_proj_qkv`/`_z`/`_b`/`_a` (`param_mapping.py:2355`), (b) add `AutoMapping("decoder.layers.*.mlp.router.expert_bias", "model.layers.*.mlp.gate.e_score_correction_bias")` from `deepseek/common.py:47`, (c) add dense-MLP mappings for the `-` layers (`deepseek/common.py:36-37, 44, 71-75` — note `post_attention_layernorm` maps to `pre_mlp_layernorm.weight` for MoE layers and `mlp.linear_fc1.layer_norm_weight` for dense layers; both aliases are declared and the non-existent one is silently ignored per layer), (d) keep both GroupedMLP (`experts.linear_fc*.weight*`) and SequentialMLP (`experts.local_experts.*.linear_fc*.weight`) expert variants.

**Hard requirement:** the GDN forward path imports `fla` at module init and raises otherwise (`3rdparty/Megatron-LM/megatron/core/ssm/gated_delta_net/common.py:141-145`), and `_get_backend_spec_provider` asserts `transformer_impl == "transformer_engine"` (`experimental_attention_variant_module_specs.py:439-442`). Both must hold in the verification environment.
