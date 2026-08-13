# Alpha HF ↔ Megatron Conversion & Verification — Porting Spec

All paths absolute. Repo root: `/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch` (abbreviated `$R` below only in prose; every reference below is a full path).

---

## 0. Architecture constants (ground truth)

From `/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha/configs/model/baseline_48L.yaml` and `/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha/docs/V2_PIPELINE_VERIFICATION.md:17-31`:

| Quantity | Value |
|---|---|
| MG layers / HF layers | 48 / 24 (2:1) |
| hidden_size | 2048 |
| ffn_hidden_size (dense) | 8192 |
| num_attention_heads (Q) | 16 |
| kv_channels (head_dim) | 256 |
| num_query_groups (GQA) | 2 → `num_querys_per_group = 8` |
| GDN Nk / Dk / Nv / Dv | 16 / 128 / 32 / 128 |
| conv kernel | 4 |
| num_experts | 192, top-8, moe_ffn 512, shared expert 512 + gate |
| Router | sigmoid, 8 groups × top-4, `topk_scaling 2.5`, expert bias ON |
| padded_vocab_size | 163968 (tokenizer effective 163,860 + 108 pad) |
| tie embeddings | **untied** (`--untie-embeddings-and-output-weights`) |
| RoPE | θ = 10,000,000, `rotary_percent`/`partial_rotary_factor` = 0.25 |
| Norm | standard RMSNorm (NOT 1p / zero-centered), eps 1e-6, QK-layernorm ON |
| Hybrid pattern | `"M-M-M-*-"` × 6 (48 chars), 18 M + 6 `*` + 24 `-`, 0 `D` |

---

## 1. Complete weight mapping

### 1.1 Layer-index mechanism (2:1)

- `hf_layer_id = global_mg_layer_id // 2` — `/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor/impl/alpha/m2h_synchronizer.py:111` and `.../h2m_synchronizer.py:77`.
- **Even MG index = token mixer** (`M` GatedDeltaNet or `*` GatedSoftmaxAttention); **odd MG index = MLP** (`-` MoE or `D` dense). The pattern must alternate; the converter does not enforce alternation, only length + charset (`common.py:31-97`).
- Pattern chars: `M` mamba, `*` attention, `-` MoE MLP, `D` dense MLP — `/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor/impl/alpha/common.py:22-28`.
- Both MG layers write into the **same** HF `AlphaDecoderLayer` — mixer half writes `input_layernorm` + (`linear_attn`|`self_attn`), MLP half writes `post_attention_layernorm` + `mlp`.
- PP mapping: `build_pipeline_parallel_mapping(num_layers, pp_size, pp_rank)` — even split only, no uneven-PP support in the alpha path (`common.py:159-183`), overriding the base class's uneven-PP version (`general/synchronizer.py:152-181`).
- Layer dispatch: `m2h_synchronizer.py:122-142`, `h2m_synchronizer.py:89-107`.

### 1.2 Embeddings / final norm / lm_head

| MG name | HF name | ParamType | Notes |
|---|---|---|---|
| `embedding.word_embeddings.weight` | `model.embed_tokens.weight` | `COLUMN` | `m2h_synchronizer.py:65-71`; h2m via base `general/h2m_synchronizer.py:107-113` |
| `decoder.final_norm.weight` | `model.norm.weight` | default `UNIQUE` | `is_mamba=True` branch → **`final_norm`, not `final_layernorm`** (`general/m2h_synchronizer.py:85-96`, `general/h2m_synchronizer.py:115-126`) |
| `output_layer.weight` | `lm_head.weight` | `COLUMN` | `general/m2h_synchronizer.py:97-103` |

**Vocab padding:** there is **no** truncate/pad logic. `generate_hf_config` sets HF `vocab_size = config.padded_vocab_size = 163968` (`/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha/tools/alpha_config.py:858`), so both sides are `[163968, 2048]` and the copy is 1:1. The 108 unused rows ride along.

**Tying:** untied here. If ever tied, `general/m2h_synchronizer.py:97-100` uses `mg_model.shared_embedding_or_output_weight()`, and on the HF→MG side `load_tensor` redirects `lm_head.weight → model.embed_tokens.weight` (`general/h2m_synchronizer.py:73-74`). `check_and_save` clones tied tensors before saving so each safetensors key gets its own storage (`general/m2h_synchronizer.py:449-452`).

### 1.3 GatedDeltaNet (`M`) — the hard part

MG mixer = `GatedDeltaNetMixer` (`/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/megatron_patch/model/qwen3_next/gated_deltanet.py`). Dims (lines 149-160): `d_state = head_k_dim = 128`, `headdim = head_v_dim = 128`, `ngroups = num_k_heads = 16`, `nheads = num_v_heads = 32`, `d_inner = 4096`, `key_dim = 2048`, `value_dim = 4096`.

**in_proj (single fused tensor MG → two tensors HF)**

- MG `mixer.in_proj.weight` shape `[d_inner*2 + 2*ngroups*d_state + nheads*2, hidden]` = `[8192 + 4096 + 64, 2048]` = **`[12352, 2048]`** (`gated_deltanet.py:170-182`).
- MG row order: **`[z, V, Q, K, b, a]`** with split sizes `[Dv*Nv, Dv*Nv, Dk*Nk, Dk*Nk, Nv, Nv] = [4096, 4096, 2048, 2048, 32, 32]`.
- Within each block, rows are **k-head-major**: block reshapes as `[Nk, block_per_k_head, hidden]` (so z/V are `[16, 256, 2048]`, Q/K are `[16, 128, 2048]`, b/a are `[16, 2, 2048]`).
- HF `in_proj_qkvz.weight` = `[key_dim*2 + value_dim*2, hidden]` = `[12288, 2048]`, order **`[Q, K, V, Z]` interleaved per k-head**: `cat([q.reshape(Nk,Dk,-1), k.reshape(Nk,Dk,-1), v.reshape(Nk,Dv*Nv//Nk,-1), z.reshape(Nk,Dv*Nv//Nk,-1)], dim=1)` then flatten.
- HF `in_proj_ba.weight` = `[Nv*2, hidden]` = `[64, 2048]`, order `[b, a]` per k-head: `cat([b.reshape(Nk,Nv//Nk,-1), a.reshape(Nk,Nv//Nk,-1)], dim=1)`.

MG→HF: `/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor/impl/alpha/m2h_synchronizer.py:144-176` (copies with `ParamType.QKV_W`, whose merge flattens `(0,1)` — `general/m2h_synchronizer.py:678-683`).

HF→MG: `.../h2m_synchronizer.py:109-152`. Splits HF qkvz as `view(Nk, 2*Dk + 2*Dv*Nv//Nk, -1)` → `[Dk, Dk, Dv*Nv//Nk, Dv*Nv//Nk]` = q,k,v,z, then emits the list `[z, v, q, k, b_chunk, a_chunk]` with `ParamType.MERGED_LINEAR` (`general/h2m_synchronizer.py:101`: `cat([chunk(x, tp, dim=0)[tp_rank].flatten(0,1) for x in lst], dim=0)`). The HF `in_proj_ba` is `view(Nk, 2*Nv//Nk, -1).chunk(2, dim=1)` → b, a.

**conv1d**

- MG `mixer.conv1d.weight`: depthwise `nn.Conv1d(conv_dim, conv_dim, groups=conv_dim, kernel_size=4)` → shape `[conv_dim, 1, 4]` with `conv_dim = d_inner + 2*ngroups*d_state = 8192` (`gated_deltanet.py:184-197`).
- **MG channel order `[V, Q, K]`** (splits `[Nv*Dv, Nk*Dk, Nk*Dk] = [4096, 2048, 2048]`); **HF channel order `[Q, K, V]`**.
- MG→HF: `m2h_synchronizer.py:181-197` — split then `cat([conv_q, conv_k, conv_v], dim=0)`, copied with `ParamType.UNIQUE` (no TP merge; TP=1 only). Comment at line 181 explicitly says TP>1 unsupported here.
- HF→MG: `h2m_synchronizer.py:157-172` — split HF `[Nk*Dk, Nk*Dk, Nv*Dv]` = q,k,v then `MERGED_LINEAR` over `[conv_v.reshape(Nk, Dv*Nv//Nk, 1, -1), conv_q.reshape(Nk, Dk, 1, -1), conv_k.reshape(Nk, Dk, 1, -1)]`.

**Scalars / norm / out_proj**

| MG | HF | ParamType | Shape | Ref |
|---|---|---|---|---|
| `mixer.dt_bias` | `linear_attn.dt_bias` | `COLUMN` | `[32]` (= Nv) | m2h:178 / h2m:154 |
| `mixer.A_log` | `linear_attn.A_log` | `COLUMN` | `[32]`, **fp32 on MG** (`gated_deltanet.py:226-232`) | m2h:179 / h2m:155 |
| `mixer.norm.weight` (`RMSNormGated(head_v_dim)`) | `linear_attn.norm.weight` | `UNIQUE` | `[128]` | m2h:199 / h2m:174 |
| `mixer.out_proj.weight` | `linear_attn.out_proj.weight` | `ROW` | `[2048, 4096]` | m2h:200 / h2m:175 |
| `mixer.in_proj.layer_norm_weight` (fused into `TELayerNormColumnParallelLinear`) | `input_layernorm.weight` | `UNIQUE` | `[2048]` | m2h:125 / h2m:92 |
| `mixer.D` | `linear_attn.D` | — | **not converted; MG sets `self.D = None`** (`gated_deltanet.py:236-237`). Only the validator has an optional branch (`validate_mg_hf_full.py:583-594`). |

### 1.4 Attention (`*`) — 4-way fused QGKV

MG module = `GatedSoftmaxAttention` with parameter **`linear_qgkv`** (not `linear_qkv`), built from `SelfAttentionSubmodules(linear_qkv=TELayerNormColumnParallelLinear, ...)` in `/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/megatron_patch/model/alpha/layer_specs.py:204-220`; the module renames it (`/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/megatron_patch/model/qwen3_next/gated_attention.py:92`).

**MG `linear_qgkv.weight` layout** — `[num_query_groups * (2 + 2*num_querys_per_group) * head_dim, hidden]` = `[2 * 18 * 256, 2048]` = **`[9216, 2048]`**. Per query group, contiguous blocks:

```
[ Q(npg*dim = 8*256 = 2048) | Gate(npg*dim = 2048) | K(dim = 256) | V(dim = 256) ]
```

Confirmed by `gated_attention.py:276-312` (`_clip_linear_qgkv` docstring + `cat([weight_q, weight_gate, weight_k, weight_v], dim=1)`) and the forward split `(query, gate, key, value)` at `gated_attention.py:348-372`.

**HF layout** — `q_proj = Linear(hidden, num_heads*head_dim*2)` → `[8192, 2048]`, **interleaved per head**: `[head0_q(256), head0_gate(256), head1_q, head1_gate, ...]` (see `modeling_alpha.py:349-351` and forward `view(..., -1, head_dim*2)` then `chunk(2, dim=-1)` at `:379-382`). `k_proj`/`v_proj` = `[512, 2048]`, `o_proj` = `[2048, 4096]`.

**MG → HF** (`m2h_synchronizer.py:209-242`):
```python
attn_proj_weight = linear_qgkv.weight.reshape(ng//tp, (2 + npg*2)*dim, -1)
q_raw, k, v      = split(attn_proj_weight, [2*npg*dim, dim, dim], dim=1)
q = q_raw.reshape(ng//tp, 2, npg, dim, -1).transpose(1, 2).flatten(1, 3)   # gate de-interleave
copy(q, hf.q_proj.weight, QKV_W); copy(k, hf.k_proj.weight, QKV_W); copy(v, hf.v_proj.weight, QKV_W)
copy(linear_proj.weight, hf.o_proj.weight, ROW)
```
`QKV_W` merge = concat over TP on dim 0 then `flatten(0,1)` (`general/m2h_synchronizer.py:678-683`).

**HF → MG** (`h2m_synchronizer.py:198-214`):
```python
attn_proj_weight = cat([
    q_proj.weight.reshape(ng, npg, 2, dim, -1).transpose(1, 2).flatten(1, 3),  # interleaved -> blocked
    k_proj.weight.reshape(ng, dim, -1),
    v_proj.weight.reshape(ng, dim, -1)], dim=1)
copy(attn_proj_weight, linear_qgkv.weight, QKV_W)   # chunk dim0 by TP, flatten(0,1)
```

**QK layernorm** (only when `args.qk_layernorm`): `self_attention.q_layernorm.weight ↔ self_attn.q_norm.weight`, same for `k_layernorm ↔ k_norm`, `UNIQUE`, shape `[256]` (per-head-dim RMSNorm) — m2h:219-221 / h2m:194-196.

**Input layernorm**: `self_attention.linear_qgkv.layer_norm_weight ↔ input_layernorm.weight` (m2h:140 / h2m:105).

**Documented dormant bias bug** (`alpha` has `attention_bias=False`, `add_qkv_bias` unset, so the path is never taken): `m2h_synchronizer.py:244-264` — the comment records that (1) the original code read `linear_qkv.bias` (AttributeError on gated attention) and was fixed to `linear_qgkv.bias`, and (2) the Q-bias split is **still incomplete** — it lacks the `reshape→transpose(1,2)→flatten` gate de-interleave the weight path applies. The same defect exists on the HF→MG side: `h2m_synchronizer.py:226-230` reshapes `q_proj.bias` as `(ng, npg*dim, -1)` (3-way layout) instead of `(ng, npg, 2, dim, -1)`. **If you enable attention bias in the port, both bias paths must be fixed.**

### 1.5 MoE (`-`) — 192 experts

Handled entirely by the base class (`general/m2h_synchronizer.py:302-344`, `general/h2m_synchronizer.py:301-326`), invoked from `m2h_synchronizer.py:128` / `h2m_synchronizer.py:95`.

| MG | HF | ParamType | Shape |
|---|---|---|---|
| `mlp.router.weight` | `mlp.gate.weight` | `UNIQUE` | `[192, 2048]`, bf16 |
| `mlp.router.expert_bias` (buffer, **fp32**) | `mlp.gate.e_score_correction_bias` (**`nn.Parameter`, fp32**) | `UNIQUE` | `[192]` |
| `mlp.experts.linear_fc1.weight{i}` | `mlp.experts[g].gate_proj.weight` + `.up_proj.weight` | `MOE_COLUMN` (×2) | MG `[1024, 2048]` fused → HF 2× `[512, 2048]` |
| `mlp.experts.linear_fc2.weight{i}` | `mlp.experts[g].down_proj.weight` | `MOE_ROW` | `[2048, 512]` |
| `mlp.shared_experts.linear_fc1.weight` | `mlp.shared_expert.gate_proj/up_proj.weight` | `COLUMN` (m2h) / `GATE_UP` (h2m) | `[1024, 2048]` → 2× `[512, 2048]` |
| `mlp.shared_experts.linear_fc2.weight` | `mlp.shared_expert.down_proj.weight` | `ROW` | `[2048, 512]` |
| `mlp.shared_experts.gate_weight` | `mlp.shared_expert_gate.weight` | `UNIQUE` | `[1, 2048]` (MG `nn.Parameter((1, hidden))`, `core/.../moe/shared_experts.py:57`; HF `nn.Linear(hidden, 1, bias=False)`) |
| `pre_mlp_layernorm.weight` | `post_attention_layernorm.weight` | `UNIQUE` | `[2048]` |

**Expert packing format.** `--moe-grouped-gemm` is ON, so experts live in `TEGroupedMLP`: **per-expert tensors named `weight0 … weight{L-1}`** on `linear_fc1` / `linear_fc2` (`L = 192 / EP` local experts), accessed via `getattr(experts.linear_fc1, f'weight{i}')` (`general/m2h_synchronizer.py:276-300`, `general/h2m_synchronizer.py:297-299`). **Not** a single stacked 3D tensor. Legacy grouped GEMM (`weight1`/`weight2` blobs) raises `NotImplementedError` (`general/m2h_synchronizer.py:310-312`).

**gate/up packing order.** MG `linear_fc1.weight{i}` is `[2*moe_ffn, hidden]` = `[gate_block; up_block]`, gate first:
- MG→HF: `getattr(...).reshape(2, -1, hidden)` → `(gate, up)` (`general/m2h_synchronizer.py:278-294`).
- HF→MG: `torch.stack([gate_proj.weight, up_proj.weight])` → `[2, ffn, hidden]`, then `MOE_GATE_UP` = `chunk(x, etp, dim=1)[etp_rank].flatten(0,1)` (`general/h2m_synchronizer.py:255-270`, `:100`).

**Expert index mapping (EP):** `local i → global (num_experts//ep_size)*ep_rank + i` (`general/synchronizer.py:183-193`). MoE params register only on `edp_rank == 0` (`general/m2h_synchronizer.py:65-68`); on HF→MG the ETP rank/size replaces TP rank/size for `MOE_*` types (`general/h2m_synchronizer.py:86-87`).

**Unused merge paths:** `MOE_GATE_UP`/`MOE_DOWN` merge functions (`general/m2h_synchronizer.py:692-718`, with the `permute(0,2,1)` layouts) are **not** used by alpha's MG→HF path — alpha's `set_group_mlp_state` emits `MOE_COLUMN`/`MOE_ROW`. Do not port those permutes into the alpha mapping.

### 1.6 Dense MLP (`D`) — supported but unused (0 layers in baseline)

`m2h_synchronizer.py:130-134` / `h2m_synchronizer.py:97-101` → base `set_mlp_state`. MG `mlp.linear_fc1.weight` `[2*8192, 2048]` splits `reshape(2,-1,hidden)` into `gate_proj`/`up_proj`; `linear_fc2.weight` → `down_proj`. Note the layernorm is `pre_mlp_layernorm` (alpha override), **not** the base class's `mlp.linear_fc1.layer_norm_weight` (`general/m2h_synchronizer.py:360`) — alpha's `D` layers use `TENorm` as `pre_mlp_layernorm` (`layer_specs.py:237-244`).

### 1.7 dtype handling

- MG→HF: `_copy_impl` merely **stores the MG source tensor** in `self._local_params` (`general/m2h_synchronizer.py:63-75`); `check_and_save` P2P-gathers and `safe_save_file`s them. **Saved dtype = MG param dtype**, i.e. bf16 for weights, **fp32 for `router.expert_bias`** and fp32 for `A_log`. The HF skeleton built with `init_empty_weights` + `from_config(torch_dtype=...)` (`general/synchronizer.py:79-83`) is used only for names/shapes/sharding, never for data.
- Cross-rank dtype consistency is asserted (`general/m2h_synchronizer.py:497-506`, `:573-574`).
- HF→MG: `dst_tensor.data.copy_(...)` — dtype follows the MG parameter.
- HF load side must keep the router bias fp32: `_keep_in_fp32_modules_strict = ["e_score_correction_bias"]` (`/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha/hf_model/modeling_alpha.py:1042`) plus the bias being an `nn.Parameter` not a buffer (`:846-848`) — `_keep_in_fp32_modules` (non-strict) only fires for fp16, and neither variant protects buffers.
- Sharding: `split_torch_state_dict_into_shards(..., max_shard_size='4GB')`, index written by rank 0, buckets round-robined over `num_hf_saver` (= world size by default) (`general/m2h_synchronizer.py:362-399`, `impl/convert.py:110-111,122-123`).

---

## 2. Megatron-side model construction

### 2.1 Model class and spec

`/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor/impl/alpha/model_provider.py:140-183`:

```python
config = core_transformer_config_from_args(args, Qwen3NextTransformerConfig)
model = MambaModel(                      # megatron.core.models.mamba.MambaModel
    config=config,
    mamba_stack_spec=get_alpha_layer_spec(args),
    vocab_size=args.padded_vocab_size,
    max_sequence_length=args.max_position_embeddings,
    pre_process, post_process,
    hybrid_attention_ratio=args.hybrid_attention_ratio,
    hybrid_mlp_ratio=args.hybrid_mlp_ratio,
    hybrid_override_pattern=args.hybrid_override_pattern,
    fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
    parallel_output=True,                # validator uses False
    share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
    position_embedding_type=args.position_embedding_type,
    rotary_percent=args.rotary_percent, rotary_base=args.rotary_base)
```
`model_provider = partial(mcore_model_provider, mamba_builder)` (line 183); the converter selects it via `--pretrain-script alpha.model_provider` (`run_convert.sh` CONVERT_ARGS).

**Layer spec** = `/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/megatron_patch/model/alpha/layer_specs.py::get_alpha_layer_spec` (lines 166-254), **not** `qwen3_next/layer_specs.py`. It returns `ModuleSpec(module=MambaStack, submodules=MambaStackSubmodules(mamba_layer, attention_layer, mlp_layer, dense_mlp_layer))` using the **patched** `MambaStack` from `/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/megatron_patch/ssm/mamba_block.py` (adds the `D` symbol, and names the trailing norm **`final_norm`** — `mamba_block.py:230`).

Sub-specs:
- `M`: `MambaLayer(mixer=GatedDeltaNetMixer(in_proj=TELayerNormColumnParallelLinear, out_proj=TERowParallelLinear), mamba_bda=get_bias_dropout_add)` (`layer_specs.py:189-201`).
- `*`: `TransformerLayer(self_attention=GatedSoftmaxAttention{attn_mask_type: causal}(linear_qkv=TELayerNormColumnParallelLinear, core_attention=TEDotProductAttention, linear_proj=TERowParallelLinear, q_layernorm=TENorm, k_layernorm=TENorm))` (`:204-220`).
- `-`: `TransformerLayer(pre_mlp_layernorm=TENorm, mlp=MoELayer(experts=<TE grouped>, shared_experts=SharedExpertMLP))` (`:223-233`, `:53-120`).
- `D`: `TransformerLayer(pre_mlp_layernorm=TENorm, mlp=MLP)` (`:237-244`).

The converter entrypoint monkey-patches `megatron.core.ssm.mamba_hybrid_layer_allocation` with the patched module **before** any Megatron import that builds a `MambaStack` — `/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor/impl/convert.py:24-26`. Port this ordering constraint.

**TransformerConfig subclass:** `Qwen3NextTransformerConfig` (`/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/megatron_patch/model/qwen3_next/transformer_config.py:20-26`) adds exactly four fields:
```
head_k_dim: int = 128, head_v_dim: int = 128, num_k_heads: int = 16, num_v_heads: int = 32
```
**Gotcha (important for the port):** these are **not** wired to any CLI flag. `core_transformer_config_from_args` only copies dataclass fields that exist as attributes on `args` (`backends/megatron/Megatron-LM-251125/megatron/training/arguments.py:1373-1375`), and there is no `--head-k-dim`. `GatedDeltaNetMixer` then overrides its own `d_state/headdim/ngroups/nheads` from these config fields (`gated_deltanet.py:149-160`), so `--mamba-state-dim/--mamba-head-dim/--mamba-num-groups/--mamba-num-heads` are effectively **ignored by the GDN mixer**. The alpha GDN geometry is a hardcoded dataclass default that happens to equal the trained values. In Megatron-Bridge, make these explicit config fields.

### 2.2 Validation performed before building (port these guards)

`model_provider.py:48-137` (`_validate_alpha_args`): requires `hybrid_attention_ratio`, `hybrid_mlp_ratio`, `hybrid_override_pattern`; hard-fails if `tensor_model_parallel_size != 1`; charset ⊆ `{M,*,-,D}`; `len(pattern) == num_layers`.
`common.py:31-97` (`validate_hybrid_pattern`): length + charset fatal, attention-count-vs-ratio mismatch is a **warning only**.

### 2.3 Exact CLI flags

`emit-megatron-flags` (`/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha/tools/alpha_config.py:693-776`) is the single source of truth; it deliberately **omits parallelism** (injected by launchers). For `baseline_48L`:

```
--num-layers 48  --hidden-size 2048  --ffn-hidden-size 8192
--num-attention-heads 16  --kv-channels 256  --num-query-groups 2
--is-hybrid-model
--hybrid-attention-ratio 0.125  --hybrid-mlp-ratio 0.5
--hybrid-override-pattern M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-
--mamba-state-dim 128  --mamba-head-dim 128  --mamba-num-groups 16  --mamba-num-heads 32
--num-experts 192  --moe-router-topk 8  --moe-ffn-hidden-size 512
--moe-shared-expert-intermediate-size 512
--moe-router-score-function sigmoid  --moe-token-dispatcher-type alltoall
--position-embedding-type rope  --rotary-base 10000000  --rotary-percent 0.25
--normalization RMSNorm  --norm-epsilon 1e-06
--seq-length 4096  --max-position-embeddings 262144  --padded-vocab-size 163968
--tokenizer-type HuggingFaceTokenizer
--group-query-attention  --moe-grouped-gemm  --moe-shared-expert-gate
--qk-layernorm  --swiglu  --disable-bias-linear  --untie-embeddings-and-output-weights
--moe-router-num-groups 8  --moe-router-group-topk 4  --moe-router-topk-scaling-factor 2.5
--moe-router-enable-expert-bias  [--moe-router-bias-update-rate <r>]
--tokenizer-model <.../examples/alpha/tokenizer_v5>
```

Launcher-injected (`run_convert.sh`): `--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --expert-model-parallel-size $GPUS`, plus runtime `--attention-backend auto --micro-batch-size 1 --bf16`, plus `--model-type GPT --load-dir --save-dir --no-load-optim --no-load-rng --logging-level 1 --synchronizer alpha --pretrain-script alpha.model_provider --auto-detect-ckpt-format [--ckpt-step N] [--use-gpu] [--bf16|--fp16] [--hf-dir X --mcore2hf]`.

Flags **not** emitted (weight-irrelevant but note for fidelity): `--moe-router-dtype fp32`, `--moe-router-load-balancing-type`, `--moe-aux-loss-coeff`, `--moe-permute-fusion`, `--moe-router-fusion`, `--attention-dropout/--hidden-dropout` (Megatron defaults are 0.1, not 0.0), `--transformer-impl` (default `transformer_engine`; both synchronizers hard-fail otherwise — `m2h:92-94`, `h2m:61-63`), `--apply-layernorm-1p` (correctly absent — see §4 gotcha).

Source config for the flags: `load_config_from_checkpoint()` reads `common.pt`'s `args` namespace (`alpha_config.py:439-558`, note `disable_bias = not args.add_bias_linear` at :525) for MG→HF; the named YAML for HF→MG.

---

## 3. HF modeling (`AlphaForCausalLM`)

### 3.1 Config

`/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha/hf_model/configuration_alpha.py:153-194` — `model_type = "alpha"`. Defaults (kept in sync with `baseline_48L.yaml`):
`vocab_size=163968, hidden_size=2048, intermediate_size=8192, num_hidden_layers=48(!)`, `num_attention_heads=16, num_key_value_heads=2, head_dim=256, max_position_embeddings=262144, rope_theta=1e7, partial_rotary_factor=0.25, attention_bias=False, tie_word_embeddings=False, rms_norm_eps=1e-6`, `linear_conv_kernel_dim=4, linear_key_head_dim=128, linear_value_head_dim=128, linear_num_key_heads=16, linear_num_value_heads=32`, `decoder_sparse_step=1, moe_intermediate_size=512, shared_expert_intermediate_size=512, num_experts_per_tok=8, num_experts=192, norm_topk_prob=True, router_aux_loss_coef=1e-4`, DSV3 routing `scoring_func="sigmoid", n_group=8, topk_group=4, routed_scaling_factor=2.5`, `mlp_only_layers=[]`.
(The `num_hidden_layers=48` default is a stale-looking default; the generated `config.json` always writes `24` = `num_layers // 2`, `alpha_config.py:781,844`.)

**`layer_types` derivation** (`configuration_alpha.py:216-231`): if not given, from `full_attention_interval` kwarg (default 4): `"linear_attention" if (i+1) % interval else "full_attention"`. Generated config writes `full_attention_interval = num_hf_layers // num_attention_stars` = `24 // 6 = 4` (`alpha_config.py:790-796`), giving full-attention at HF layers 3, 7, 11, 15, 19, 23 — exactly matching MG `*` at indices 6, 14, 22, 30, 38, 46. `mlp_only_layers = sorted({mg_idx//2 for D chars})` (`alpha_config.py:803-804`).

SGLang-compat properties `full_attention_interval`, `layers_block_type`, `full_attention_layer_ids`, `linear_layer_ids`, `hybrid_gdn_params`, `mamba_cache_per_req` at `:259-331`.

### 3.2 Layer structure

`AlphaDecoderLayer` (`modeling_alpha.py:933-1025`): **one HF layer contains exactly one mixer + one MLP**, chosen by `layer_types[layer_idx]`:
- `linear_attention` → `self.linear_attn = AlphaGatedDeltaNet`
- `full_attention` → `self.self_attn = AlphaAttention`
(the other attribute does not exist — this is why the converter branches on the MG pattern char).
`self.mlp` = `AlphaSparseMoeBlock` unless `layer_idx in mlp_only_layers` (then `AlphaMLP`).
Forward: `residual → input_layernorm → mixer → +residual → post_attention_layernorm → mlp → +residual` (`:990-1025`).

So HF layers are **not** interleaved mixer/MLP-only; the 2:1 comes from Megatron splitting the mixer and the MLP into separate `MambaStack` layer slots.

### 3.3 Attention forward

`modeling_alpha.py:337-415`. `q_proj` output is `num_heads * head_dim * 2` → `view(..., -1, head_dim*2)` → `chunk(2, dim=-1)` = (query, gate). `q_norm`/`k_norm` are per-head-dim RMSNorm applied **before** RoPE (`:384-385`). Partial RoPE: `apply_rotary_pos_emb` splits at `rotary_dim = cos.shape[-1]` and passes the tail through (`:284-296`) — with `partial_rotary_factor=0.25` on head_dim 256, only 64 dims rotate. Output gate: `attn_output * torch.sigmoid(gate)` **before** `o_proj` (`:412-414`) — mirrors MG `gated_attention.py:638-640`.

### 3.4 GatedDeltaNet forward

`modeling_alpha.py:580-674`. `fix_query_key_value_ordering` (`:647-674`) is the runtime consumer of the `[Q,K,V,Z]`/`[b,a]` packing — it views `in_proj_qkvz` as `(..., Nk, 2*Dk + 2*Dv*Nv/Nk)` and splits `[Dk, Dk, Nv/Nk*Dv, Nv/Nk*Dv]`. `norm` is `AlphaRMSNormGated` (standard `*weight`, `:75-90`) or FLA's `FusedRMSNormGated`.

### 3.5 Router (this must match training exactly)

`AlphaSparseMoeBlock` (`modeling_alpha.py:810-930`), a mirror of `DeepseekV3TopkRouter`:
1. `router_logits = self.gate(hidden_states)` — `nn.Linear(hidden, num_experts, bias=False)`.
2. `scores = router_logits.sigmoid().to(torch.float32)` (**fp32 router math**, `:888-891`).
3. `_select_experts` (`@torch.no_grad`, `:856-877`): `scores + e_score_correction_bias`; group-limited — reshape `(tokens, n_group, experts_per_group)`, group score = **sum of top-2** within group, keep `topk_group` groups, mask the rest to 0.0, then global `topk(k=top_k)`.
4. Weights gathered from the **original** (bias-free) scores: `routing_weights = scores.gather(1, selected)`.
5. `norm_topk_prob` → divide by sum + 1e-20; then `× routed_scaling_factor (2.5)`; cast to input dtype.
6. Shared expert always runs: `shared_expert_output * sigmoid(shared_expert_gate(hidden_states))`, added to the routed sum (`:924-927`).

`e_score_correction_bias` is an `nn.Parameter(requires_grad=False)` attached to `self.gate` (so the converter key `gate.e_score_correction_bias` exists) and kept fp32 across bf16 loads (`:846-848`, `:1042`). Everything else is bf16 except the fp32 router-score math.

`AlphaRMSNorm` (`:222-248`) is **standard** `x_norm * weight` with `ones` init — explicitly NOT the Qwen3-Next `(1 + gamma)` form (see §5).

---

## 4. Verification methodology

Orchestrated by `/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha/evaluate.sh` in five gates.

### Stage 0 — preflight (`tools/verify_pipeline.py:60-95`)
Derives the expected HF config from `common.pt` and hard-fails on: tokenizer dir missing (warn if path lacks `tokenizer_v5`), `num_experts % gpus != 0`, `num_layers` odd (breaks 2:1), `eos_token_id != 0`, `bos_token_id` present. Prints the 18 `STRUCT_FIELDS`. No GPU needed.

### Stage 1 — convert (`run_convert.sh`)
Reported outcome for `iter_0010000`: **14133/14133 weights matched**, `final_norm` and `lm_head` cos_sim = 1.0 (`docs/V2_PIPELINE_VERIFICATION.md:101`).

### Stage 1.5 — config + tokenizer gates (`tools/verify_pipeline.py:98-155`)
`compare-config`: exact equality of the produced `config.json` against the `common.pt`-derived expectation on all of
`num_experts, num_experts_per_tok, vocab_size, head_dim, num_hidden_layers, moe_intermediate_size, shared_expert_intermediate_size, rope_theta, partial_rotary_factor, max_position_embeddings, num_attention_heads, num_key_value_heads, eos_token_id, full_attention_interval, scoring_func, n_group, topk_group, routed_scaling_factor` (`:32-52`), plus `bos_token_id` must be absent/None. Any inequality → exit 1.
`tokenizer-roundtrip`: `tok.eos_token_id == 0`, and `decode(encode(s)).strip() == s.strip()` for three samples: English pangram, Korean (`"안녕하세요. 오늘 날씨가 정말 좋네요!"`), and a Python function (`:132-136`).

### Stage 2 — per-tensor weight diff (`examples/alpha/validate_mg_hf_full.py`, driven by `validate.sh`)

**What is compared:** *only weight tensors*, never activations or logits. Both models are loaded fully; MG via `load_checkpoint` on a freshly built `MambaModel` (`:1185-1222`, with iter-dir → `args.load = parent, args.ckpt_step = N` rewriting at `:1197-1205`), HF via `AutoModelForCausalLM.from_pretrained(torch_dtype=bfloat16, device_map="cuda:0", trust_remote_code=True)` (`:254-273`).

**Metric** (`:104-159`):
```python
t1_f, t2_f = t1.float().flatten(), t2.float().flatten()
max_diff  = (t1_f - t2_f).abs().max()
mean_diff = (t1_f - t2_f).abs().mean()
cos_sim   = F.cosine_similarity(t1_f[None], t2_f[None])
matched   = max_diff < threshold and cos_sim > 0.999      # threshold = 0.01 (validate.sh:80)
```
Comparison is done in **fp32** after upcast; the criterion is deliberately strict — the docstring at `:131-141` forbids relaxing it, because the converter is a bit-exact copy so every faithful comparison is exact (max_diff ≈ 0).

**Per-layer coverage** (all with the same 0.01 / 0.999 criterion):
- Embedding (`:341-363`), final norm (`:370-401`, handles `final_norm` vs `final_layernorm`), lm_head (`:408-430`).
- Mamba (`:437-620`): `in_proj.layer_norm_weight`, reconstructed `in_proj[qkvz]` and `in_proj[ba]` (the validator **re-implements the converter's reshape**, `:497-515`), reconstructed `conv1d` (`:537-547`), `A_log`, `dt_bias`, optional `D`, `norm.weight`, `out_proj.weight`.
- Attention (`:627-773`): `linear_qgkv.layer_norm_weight`, reconstructed Q/K/V (again re-implementing the converter reshape, `:692-719`), `linear_proj`, `q_layernorm`, `k_layernorm`.
- MoE (`:780-1013`): `pre_mlp_layernorm`, `router.weight`, `router.expert_bias ↔ gate.e_score_correction_bias`, then **every local expert** (`num_local_experts` = 192 at EP=1) × {gate, up, down}, then shared expert {gate, up, down} and `shared_experts.gate_weight ↔ shared_expert_gate.weight` (reshaped to 1-D before comparison, `:955-966`).

**Bidirectional coverage gates** (`:1265-1308`) — this is what makes it rigorous:
- `unchecked = all_mg_weights − compared_mg_weights` (from `named_parameters()` + `named_buffers()`), filtered to drop suffixes `('._extra_state', '.local_tokens_per_expert')` (`:1278-1279`).
- `phantom = compared_mg_weights − all_mg_weights`, filtered to drop `weight{i}` (TEGroupedMLP dynamic attrs) and `._extra_state` (`:1280-1283`).

**Exit semantics** (`:1306-1327`):
| Condition | Exit |
|---|---|
| all matched, no filtered_unchecked, no filtered_phantom | 0 — "ALL WEIGHTS MATCHED - CONVERSION VERIFIED" |
| all matched, no unchecked, phantom present | 0 with warning |
| all matched but filtered_unchecked non-empty | **1** (coverage failure counts as failure) |
| any mismatch | 1 |

Reported result: **14180/14181 → 14181/14181** after the fp32 bias fix (`examples/alpha/CLAUDE.md`, "MG↔HF weight 검증이 expert_bias 1개에서 지속 실패").

**Topology:** validation runs `torchrun --nproc_per_node=1` with TP=PP=EP=1, `--bf16 --micro-batch-size 1 --no-load-optim --no-load-rng --ckpt-format torch_dist --threshold 0.01` (`validate.sh:66-81`) — torch_dist reshards the EP=8-trained checkpoint into a single rank holding all 192 experts.

### Stage 2.5 — forward sanity (`examples/alpha/forward_sanity.py`)

Exists precisely because Stage 2 cannot catch interpretation mismatches (its attention/mamba comparisons *reuse the converter's own reshape*, so a shared misinterpretation is invisible) — see the module docstring `:1-27`.
- Input: fixed 5-sentence English snippet (`:36-40`), tokenized with the converted tokenizer.
- Model: `AutoModelForCausalLM.from_pretrained(dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True).eval()`, single forward with `labels=ids`.
- **PASS criterion:** `torch.isfinite(logits).all()` **and** `exp(loss) < 100` (default `--threshold 100`; random ≈ vocab_size ≈ 163,968).
- Empirical calibration from the 1p-RMSNorm incident: broken → ppl 295,440 and greedy garbage; fixed → **ppl 8.84**, greedy `"The capital of France is Paris."`, ARC-easy 0-shot 25% → 0.73/0.76.

### Stage 3 — lm-eval benchmark (opt-in `--benchmark --tasks standard`).

### Known / documented acceptable divergences
- **None numerically.** The converter performs no arithmetic, so a faithful conversion is exact; any nonzero `max_diff` is a regression.
- The only historical "divergence" was an artifact: `expert_bias` ≈ 4.5 downcast to bf16 gives ulp/2 = 0.0156 > 0.01. Resolution was to keep it fp32 end-to-end, **not** to widen the tolerance (`validate_mg_hf_full.py:131-141`; regression test `tests/test_alpha_pipeline_config.py:348-360` asserts a bf16 downcast of a ~4.5 fp32 tensor must still FAIL).
- Coverage exclusions: `._extra_state` (TE internal), `router.local_tokens_per_expert` (transient routing-stat buffer, `core/.../moe/router.py:163-164`, no HF counterpart).

### Regression tests
`/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/tests/test_alpha_pipeline_config.py` — 12 tests: `test_yaml_config_matches_checkpoint_structure` (:68), `test_checkpoint_load_recovers_dsv3_routing` (:90), `test_generate_hf_config_v2_fields` (:109), `test_modeling_alpha_moe_is_dsv3_routing` (:131), `test_modeling_alpha_rmsnorm_is_standard_not_1p` (:159), `test_emit_flags_include_mamba_and_experts_from_yaml` (:203), `test_emit_flags_from_checkpoint_carry_routing` (:220), `test_no_stale_v1_references_in_scripts` (:237), `test_configuration_alpha_default_num_experts_is_192` (:260), `test_modeling_alpha_router_bias_is_fp32_parameter` (:294), `test_keep_in_fp32_modules_strict_protects_param_not_buffer` (:308, pins transformers' param-vs-buffer behavior with a toy model), `test_compare_tensors_is_strict` (:348).

---

## 5. Gotchas to carry into Megatron-Bridge

**Conversion topology / EP resharding**
1. `EP = GPU count`, `TP = PP = 1` always. `run_convert.sh` hard-fails if `num_experts % GPUS != 0`. torch_dist reshards an EP=8-trained checkpoint to any EP on load — training topology and conversion topology are independent. Validation deliberately uses EP=1 on 1 GPU.
2. **TP=1 is a hard constraint** (`model_provider.py:83-95`, `m2h_synchronizer.py:32`) — GDN/Mamba layers don't support TP>1, and the conv1d path is explicitly TP-unsafe (`m2h_synchronizer.py:181`).
3. Checkpoint must be `--ckpt-format torch_dist`; Megatron's loader needs `latest_checkpointed_iteration.txt` in the **parent** dir. `run_convert.sh` auto-detects an `iter_NNNNNNN` path and rewrites to `parent + --ckpt-step N`; `validate_mg_hf_full.py:1197-1205` does the same in Python. Passing the iter dir directly to a naive loader silently starts from random weights.
4. `--auto-detect-ckpt-format` is passed; HF→MG saving writes iteration 1 then rewrites the tracker to `release` and moves `iter_0000001 → release` (`general/h2m_synchronizer.py:360-381`).
5. Memory rule of thumb (top-level `toolkits/distributed_checkpoints_convertor/README.md`): total GPU memory ≥ 1.5× model for hf2mcore, ≥ 1.8× for mcore2hf; reduce `--num-hf-saver` or use `USE_GPU=false` on OOM. Alpha baseline_48L converts in ~30 s.

**Config drift (the historical failure mode)**
6. All MG→HF skeleton flags must come from the checkpoint's `common.pt`, never from a hand-maintained script. The stale `scripts/alpha/configs/baseline_48L.sh` carried v1 values (128 experts / head 32 / kv 128 / vocab 151936 / 49-char pattern) and the old `validate.sh` parsed a nested YAML schema that no longer exists — both would have built a wrong skeleton (`docs/V2_PIPELINE_VERIFICATION.md:33-46`). The converter also used to omit `--mamba-*` entirely.
7. **`linear_key_head_dim` is generated from `mamba_head_dim`, not `mamba_state_dim`** (`alpha_config.py:823`). For alpha both are 128 so it is latent, but conceptually `Dk = d_state`. Fix this in the port or it breaks any model with `d_state != headdim`.
8. `configuration_alpha.py` defaults are a deployment footgun: any key missing from `config.json` falls back to the dataclass default. Seven of them were stale (Qwen3 values) and silently wrong at inference.
9. `bos_token_id` is **deleted** rather than emitted as `null` (`alpha_config.py:861-866`); `eos_token_id = 0` (`<|endoftext|>`), `pad = 1`. `<|im_end|>` (id 3) is chat-only.

**Silent forward-pass mismatches (weight validation cannot catch these)**
10. **RMSNorm 1p vs standard**: `AlphaRMSNorm` must be `x_norm * gamma`, not `(1 + gamma)`. Alpha trains with `apply-layernorm-1p` OFF. The 1p form scaled every norm output 1.7–2.5×, corrupted the residual stream, and drove every benchmark to chance — while weight validation passed 100%. Guard = `forward_sanity.py` + `test_modeling_alpha_rmsnorm_is_standard_not_1p`. Note `AlphaRMSNormGated` was already standard, so the two norm classes disagreed.
11. **Router must be DSV3**, not softmax/global-top-k. The HF block originally used plain softmax + global top-k + no bias; that selects different experts than training and invalidates every eval number. The conversion crash on the missing `gate.e_score_correction_bias` is what surfaced it (`V2_PIPELINE_VERIFICATION.md:48-66`).
12. **Router bias dtype**: MG keeps `router.expert_bias` fp32 (`core/.../moe/router.py:204-213`); the converter inherits that dtype; HF must keep it fp32 via an `nn.Parameter` + `_keep_in_fp32_modules_strict`. A buffer, or the non-strict flag, silently downcasts under a bf16 load.
13. **Validation blind spot by construction**: `validate_mg_hf_full.py`'s Mamba/attention comparisons reconstruct the HF tensor using the *same* reshape code as the converter (see the explicit "Reference: m2h_synchronizer.py" note at `:648`). It can only catch copy errors, never a shared misinterpretation. Any port must keep an independent forward-level gate.

**Miscellaneous**
14. The hybrid pattern `*` is glob-hazardous in shell — every launcher uses `readarray -t` (one token per line) and quoted-array invocation, never `eval` (`run_convert.sh`, `validate.sh:56-58`).
15. MG→HF conversion copies `examples/alpha/hf_model/*.py` into the output dir for `trust_remote_code` (`run_convert.sh`, the `cp ${ALPHA_HF_MODEL_DIR}/*.py` step). A modeling-code fix therefore does **not** require re-conversion — re-copying the `.py` files into an existing HF dir suffices.
16. The top-level converter README notes the generic scripts do not copy tokenizers; the alpha script does (copies `tokenizer*.json`, `special_tokens_map.json`, `vocab.json`, `merges.txt`, `*.model` from `--tokenizer-model`).
17. `examples/alpha/docs/CONVERSION.md` is **largely stale** — it documents a removed `examples/alpha/scripts/convert_to_hf.sh` + `validate_conversion_config.py` flow, 256/128-expert numbers, and a "config metadata is not available in the checkpoint" claim that the `common.pt` path has since disproved. Its only currently-valid content is the weight-mapping overview (§내부 동작 원리) and the TP=1 / OOM troubleshooting. Likewise `scripts/alpha/README.md` still shows 128 experts / moe_ffn 768 / a 49-char `MDM-...` pattern. Treat `docs/V2_PIPELINE_VERIFICATION.md` + `toolkits/.../CLAUDE.md` as authoritative.
