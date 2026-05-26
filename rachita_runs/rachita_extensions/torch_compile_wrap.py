"""Gated ``torch.compile`` wrapping for Megatron models.

The wrapper is controlled exclusively by environment variables so it can be
toggled from the YAML launcher without modifying Megatron itself:

    MEGATRON_TORCH_COMPILE        "1" to enable
    MEGATRON_TORCH_COMPILE_MODE   "default" | "reduce-overhead" | "max-autotune"
    MEGATRON_TORCH_COMPILE_BACKEND  backend name (default: "inductor")
    MEGATRON_TORCH_COMPILE_TARGET  "layers" (default; per-TransformerLayer compile)
                                   or "model" (compile the top-level module)
    MEGATRON_TORCH_COMPILE_FULLGRAPH  "1" to forbid graph breaks
    MEGATRON_TORCH_COMPILE_DYNAMIC    "1" to enable dynamic shapes

If anything goes wrong we **do not** crash the training job; we log a warning
and return the original module so the run continues with uncompiled kernels.
"""

import logging
import os
import time

logger = logging.getLogger("rachita.torch_compile")


def _envbool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip() not in ("", "0", "false", "False")


def _env(name: str, default: str) -> str:
    val = os.environ.get(name)
    return default if val is None or val == "" else val


def enabled() -> bool:
    return _envbool("MEGATRON_TORCH_COMPILE")


def _find_transformer_layers(module):
    """Yield the ``TransformerLayer`` submodules of a Megatron GPT model.

    We deliberately walk the module tree by *class name* rather than importing
    the Megatron class, because by the time this wrapper runs Megatron may
    have wrapped the model in Float16Module/DistributedDataParallel. Matching
    on name also keeps us forward-compatible across Megatron versions where
    the class is occasionally renamed.
    """

    target_names = {"TransformerLayer", "TELayer", "TETransformerLayer"}
    for submodule in module.modules():
        if submodule.__class__.__name__ in target_names:
            yield submodule


def compile_model(model):
    """Compile ``model`` in-place if enabled; otherwise return it unchanged.

    Returns ``model`` so the caller pattern ``model = compile_model(model)``
    works both when compile is on and off.
    """

    if not enabled():
        return model

    try:
        import torch
    except Exception as exc:
        logger.warning("[rachita] torch.compile requested but torch import failed: %s", exc)
        return model

    mode = _env("MEGATRON_TORCH_COMPILE_MODE", "default")
    backend = _env("MEGATRON_TORCH_COMPILE_BACKEND", "inductor")
    target = _env("MEGATRON_TORCH_COMPILE_TARGET", "layers")
    fullgraph = _envbool("MEGATRON_TORCH_COMPILE_FULLGRAPH", False)
    dynamic = _envbool("MEGATRON_TORCH_COMPILE_DYNAMIC", False)

    compile_kwargs = dict(
        mode=mode,
        backend=backend,
        fullgraph=fullgraph,
        dynamic=dynamic,
    )

    logger.info(
        "[rachita] torch.compile target=%s mode=%s backend=%s fullgraph=%s dynamic=%s",
        target, mode, backend, fullgraph, dynamic,
    )

    t0 = time.perf_counter()
    try:
        if target == "model":
            try:
                model.compile(**compile_kwargs)
                compiled_count = 1
            except (AttributeError, TypeError):
                model = torch.compile(model, **compile_kwargs)
                compiled_count = 1
        else:
            compiled_count = 0
            for layer in _find_transformer_layers(model):
                try:
                    layer.compile(**compile_kwargs)
                    compiled_count += 1
                except Exception as inner:
                    logger.warning(
                        "[rachita] failed to compile %s (layer %d): %s",
                        layer.__class__.__name__, compiled_count, inner,
                    )
            if compiled_count == 0:
                logger.warning(
                    "[rachita] no TransformerLayer submodules found; falling back to "
                    "whole-model compile"
                )
                try:
                    model.compile(**compile_kwargs)
                    compiled_count = 1
                except Exception as inner:
                    logger.warning(
                        "[rachita] whole-model torch.compile fallback failed: %s; "
                        "continuing without compile.", inner,
                    )
                    return model
    except Exception as exc:
        logger.warning(
            "[rachita] torch.compile setup failed (%s); continuing without compile.", exc
        )
        return model

    dt = time.perf_counter() - t0
    logger.info(
        "[rachita] torch.compile setup_time_s=%.3f compiled_count=%d "
        "(real compile happens lazily on first forward)",
        dt, compiled_count,
    )
    return model
