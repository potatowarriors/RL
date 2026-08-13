"""vLLM general plugin registering the alpha model architecture.

Loaded automatically by vLLM through the ``vllm.general_plugins`` entry
point in every engine/worker process, so ``AlphaForCausalLM`` resolves
without patching the vllm package itself.
"""


def register() -> None:
    from vllm import ModelRegistry

    if "AlphaForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "AlphaForCausalLM",
            "vllm_alpha_plugin.alpha:AlphaForCausalLM",
        )
