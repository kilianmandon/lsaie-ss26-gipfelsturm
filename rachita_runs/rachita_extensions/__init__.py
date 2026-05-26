"""Rachita's optional extensions for Megatron-LM training.

This package is imported by ``rachita_runs/pretrain_gpt_rachita.py`` *before*
Megatron is imported (so that monkey-patches such as Quack RMSNorm take effect).

Modules:
    torch_compile_wrap: gated ``torch.compile`` wrapping of Megatron models.
    quack_kernels:      gated Quack fused-kernel monkey-patching (RMSNorm,
                        cross-entropy).

Everything is **off by default**; modules only activate when their respective
environment variable is set, and they fail gracefully (log a warning, return
unchanged) if their underlying dependencies are missing. This lets the same
wrapper script remain the entrypoint for all rachita configs without breaking
upstream behaviour.
"""

__all__ = ["torch_compile_wrap", "quack_kernels"]
