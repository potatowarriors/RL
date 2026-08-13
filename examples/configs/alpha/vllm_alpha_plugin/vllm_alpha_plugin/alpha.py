# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Alpha model (trust-remote-code ``AlphaForCausalLM``).

Alpha is a Qwen3-Next-style hybrid (gated delta net linear attention +
output-gated full attention) with two deliberate differences:

1. **Standard RMSNorm everywhere** (``w * x_norm``). Qwen3-Next uses
   Gemma-style zero-centered norms (``(1 + w) * x_norm``); reusing those
   classes silently corrupts every norm output while all weight loading
   still succeeds. All non-GDN norms are therefore replaced with the
   standard ``RMSNorm`` here (the GDN-internal ``RMSNormGated`` is already
   standard in both models).
2. **DeepSeek-V3-style MoE routing**: sigmoid scoring in fp32,
   group-limited top-k (``n_group``/``topk_group``), fp32
   ``e_score_correction_bias`` used for expert *selection* only (weights
   are gathered from unbiased scores), ``norm_topk_prob`` renormalization,
   then multiply by ``routed_scaling_factor``.

Everything else (GDN wiring, gated attention, shared expert with sigmoid
gate, weight mappers, hybrid state accounting) is inherited from the
Qwen3-Next implementation.
"""

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.fused_moe import FusedMoE, GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as AlphaMLP
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextForCausalLM,
    Qwen3NextModel,
    Qwen3NextSparseMoeBlock,
)

from vllm.model_executor.models.utils import (
    PPMissingLayer,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


class AlphaSparseMoeBlock(Qwen3NextSparseMoeBlock):
    """Qwen3-Next MoE block with the router swapped to DeepSeek-V3 semantics.

    Subclasses only to inherit ``forward`` (internal-router path); the
    ``__init__`` is written out because the parent's is monolithic.
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen3NextSparseMoeBlock, self).__init__()

        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        # Load balancing settings.
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        # DSV3-style router: fp32 gate GEMM + fp32 score-correction bias.
        # Alpha trains the router in fp32 (moe_router_dtype fp32 on the
        # Megatron side; HF mirrors it) — force fp32 here unconditionally.
        self.router_dtype = torch.float32
        self.gate = GateLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            params_dtype=self.router_dtype,
            out_dtype=self.router_dtype,
            force_fp32_compute=True,
            prefix=f"{prefix}.gate",
        )
        if getattr(config, "scoring_func", "softmax") == "sigmoid":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.num_experts, dtype=torch.float32)
            )
        else:
            self.gate.e_score_correction_bias = None
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

        self.shared_expert_gate = None
        self.shared_expert = None
        if config.shared_expert_intermediate_size > 0:
            from vllm.model_executor.layers.linear import ReplicatedLinear

            self.shared_expert_gate = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.shared_expert_gate",
            )
            self.shared_expert = AlphaMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                expert_gate=self.shared_expert_gate,
                is_sequence_parallel=self.is_sequence_parallel,
                prefix=f"{prefix}.shared_expert",
            )

        self.experts = FusedMoE(
            shared_experts=self.shared_expert,
            gate=self.gate,
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=getattr(config, "norm_topk_prob", True),
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            # DeepSeek-V3 routing semantics
            use_grouped_topk=True,
            num_expert_group=getattr(config, "n_group", 1),
            topk_group=getattr(config, "topk_group", 1),
            scoring_func=getattr(config, "scoring_func", "softmax"),
            e_score_correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            # Scale the topk weights (router side) instead of the routed
            # output: the output-side path assumes DeepSeek's decoder-layer
            # 1/scale compensations under fp16 + shared expert, which a
            # Qwen3-Next-style decoder layer does not perform.
            apply_routed_scale_to_output=False,
            router_logits_dtype=self.gate.out_dtype,
        )


class AlphaAttention(Qwen3NextAttention):
    """Gated attention with STANDARD (non-zero-centered) QK RMSNorm."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        eps = self.q_norm.variance_epsilon
        self.q_norm = RMSNorm(self.head_dim, eps=eps)
        self.k_norm = RMSNorm(self.head_dim, eps=eps)
        # The fused QK-norm+RoPE kernel path hardcodes the Gemma-style
        # (1 + w) shift; alpha's norms are standard, so always take the
        # eager path where the norm modules are applied as-is.
        self.use_fused_qk_norm_rope_gate = False


class AlphaDecoderLayer(Qwen3NextDecoderLayer):
    """Qwen3-Next decoder layer with alpha attention/MoE/norm variants."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super(Qwen3NextDecoderLayer, self).__init__()

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)

        if self.layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=True,
            )
        elif self.layer_type == "full_attention":
            self.self_attn = AlphaAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        mlp_only_layers = getattr(config, "mlp_only_layers", [])
        decoder_sparse_step = getattr(config, "decoder_sparse_step", 1)
        if (self.layer_idx not in mlp_only_layers) and (
            config.num_experts > 0
            and (self.layer_idx + 1) % decoder_sparse_step == 0
        ):
            self.mlp = AlphaSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = AlphaMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        # Alpha: standard RMSNorm (NOT Gemma-style zero-centered).
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(1, 1, config.hidden_size),
            )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(1, 1, config.hidden_size),
            )


@support_torch_compile
class AlphaModel(Qwen3NextModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen3NextModel, self).__init__()

        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config

        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str):
            return AlphaDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            # Alpha: standard RMSNorm final norm.
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()


class AlphaForCausalLM(Qwen3NextForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        scheduler_config = vllm_config.scheduler_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Alpha currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.quant_config = vllm_config.quant_config

        super(Qwen3NextForCausalLM, self).__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = AlphaModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters
        self.set_moe_parameters()

    def set_moe_parameters(self):
        super().set_moe_parameters()
        # Alpha uses DSV3 group-limited routing (Qwen3-Next hardcodes 1 group).
        self.num_expert_groups = getattr(self.config, "n_group", 1)
