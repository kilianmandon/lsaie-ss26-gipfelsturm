"""Gated Quack fused-kernel monkey-patching for Megatron-LM.

We patch *only* what Quack publishes, and only if Quack is actually importable.
If Quack is missing or one of the patched modules is structured differently
than expected, we log a warning and leave Megatron untouched. The training run
will then complete with upstream kernels and zero correctness risk.

Environment:
    MEGATRON_USE_QUACK   "1" to enable
    MEGATRON_QUACK_OPS   comma-separated subset of {rmsnorm, cross_entropy};
                         default "rmsnorm,cross_entropy"

Install Quack inside the alps3 enroot environment, e.g.::

    pip install --user quack-kernels   # PyPI distribution

This file is imported BEFORE Megatron from
``rachita_runs/pretrain_gpt_rachita.py``. We therefore patch attributes lazily
(after Megatron submodules have been imported by the main script).
"""

import importlib
import logging
import os

logger = logging.getLogger("rachita.quack")


def _envbool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip() not in ("", "0", "false", "False")


def enabled() -> bool:
    return _envbool("MEGATRON_USE_QUACK")


def _selected_ops():
    raw = os.environ.get("MEGATRON_QUACK_OPS", "rmsnorm,cross_entropy")
    return set(op.strip() for op in raw.split(",") if op.strip())


def _try_import_quack():
    try:
        import quack  # noqa: F401
        return importlib.import_module("quack")
    except Exception as exc:
        logger.warning(
            "[rachita] Quack requested (MEGATRON_USE_QUACK=1) but `import quack` "
            "failed: %s. Falling back to upstream RMSNorm / CE.",
            exc,
        )
        return None


def maybe_install() -> bool:
    """Install Quack monkey-patches if enabled and Quack is importable.

    Must be called *before* Megatron is imported, so that any classes we patch
    are intercepted in their first import. The function is idempotent: a second
    call is a no-op.

    Returns True if any patch was installed.
    """

    if getattr(maybe_install, "_installed", False):
        return True

    if not enabled():
        return False

    quack = _try_import_quack()
    if quack is None:
        maybe_install._installed = True
        return False

    ops = _selected_ops()
    installed = []

    if "rmsnorm" in ops and _install_rmsnorm(quack):
        installed.append("rmsnorm")
    if "cross_entropy" in ops and _install_cross_entropy(quack):
        installed.append("cross_entropy")

    maybe_install._installed = True
    if installed:
        logger.info(
            "[rachita] Quack monkey-patches installed for ops=%s "
            "(requested=%s)", sorted(installed), sorted(ops),
        )
    else:
        logger.warning(
            "[rachita] Quack imported but no ops were patched (requested=%s); "
            "check Megatron / Quack versions.", sorted(ops),
        )
    return bool(installed)


def _install_rmsnorm(quack) -> bool:
    """Replace Megatron's RMSNorm forward with quack.rmsnorm.

    Megatron's RMSNorm lives in (in order of preference):
        - megatron.core.fusions.fused_layer_norm.FusedLayerNorm  (if RMSNorm path)
        - megatron.core.transformer.norm_module.RMSNorm
        - transformer_engine.pytorch.RMSNorm  (TE-built)

    Quack's API: ``quack.rmsnorm(x, weight, eps=1e-6)``. The function returns a
    tensor of the same shape as ``x``.
    """

    quack_rmsnorm = getattr(quack, "rmsnorm", None) or getattr(quack, "rms_norm", None)
    if quack_rmsnorm is None:
        logger.warning(
            "[rachita] quack module has no `rmsnorm` / `rms_norm` function; "
            "skipping RMSNorm patch."
        )
        return False

    candidates = [
        ("megatron.core.transformer.norm_module", "RMSNorm"),
        ("megatron.core.fusions.fused_layer_norm", "FusedLayerNorm"),
        ("transformer_engine.pytorch", "RMSNorm"),
    ]

    patched_any = False
    for module_path, attr in candidates:
        try:
            module = importlib.import_module(module_path)
        except Exception:
            continue
        target_cls = getattr(module, attr, None)
        if target_cls is None:
            continue

        if getattr(target_cls, "_rachita_quack_patched", False):
            patched_any = True
            continue

        original_forward = target_cls.forward

        def quack_forward(self, *args, **kwargs):
            if not args and "x" not in kwargs:
                return original_forward(self, *args, **kwargs)
            x = args[0] if args else kwargs["x"]
            weight = getattr(self, "weight", None)
            if weight is None or weight.shape[0] != x.shape[-1]:
                return original_forward(self, *args, **kwargs)
            eps = getattr(self, "eps", None) or getattr(self, "variance_epsilon", 1e-6)
            try:
                return quack_rmsnorm(x, weight, eps=eps)
            except Exception as exc:
                logger.warning(
                    "[rachita] quack.rmsnorm raised %s on input shape %s; falling "
                    "back to upstream RMSNorm for this call.", exc, tuple(x.shape),
                )
                return original_forward(self, *args, **kwargs)

        target_cls.forward = quack_forward
        target_cls._rachita_quack_patched = True
        patched_any = True
        logger.info("[rachita] patched %s.%s.forward -> quack.rmsnorm", module_path, attr)

    if not patched_any:
        logger.warning(
            "[rachita] no RMSNorm class found to patch (Megatron / TE may have moved "
            "the import path)."
        )
    return patched_any


def _install_cross_entropy(quack) -> bool:
    """Replace Megatron's vocab-parallel cross-entropy local kernel with Quack.

    Megatron's `_VocabParallelCrossEntropy.forward` lives in
    ``megatron.core.tensor_parallel.cross_entropy``. The function we *can*
    safely replace is the per-rank softmax-cross-entropy on the local logits;
    Megatron handles the cross-rank all-reduce on its own, so we patch only
    the local computation.

    If Quack does not expose a cross-entropy kernel, this is a no-op.
    """

    quack_ce = (
        getattr(quack, "cross_entropy", None)
        or getattr(quack, "cross_entropy_loss", None)
        or getattr(quack, "fused_cross_entropy", None)
    )
    if quack_ce is None:
        logger.info(
            "[rachita] quack does not expose a cross_entropy kernel in this version; "
            "skipping CE patch."
        )
        return False

    try:
        module = importlib.import_module("megatron.core.tensor_parallel.cross_entropy")
    except Exception as exc:
        logger.warning(
            "[rachita] cannot import megatron.core.tensor_parallel.cross_entropy: %s",
            exc,
        )
        return False

    target_cls = getattr(module, "_VocabParallelCrossEntropy", None) or getattr(
        module, "VocabParallelCrossEntropy", None
    )
    if target_cls is None:
        logger.warning(
            "[rachita] could not find _VocabParallelCrossEntropy in Megatron; "
            "skipping CE patch."
        )
        return False

    if getattr(target_cls, "_rachita_quack_patched", False):
        return True

    logger.warning(
        "[rachita] Quack cross-entropy integration requires a hand-written wrapper "
        "around _VocabParallelCrossEntropy; this scaffold deliberately does not "
        "monkey-patch the autograd.Function. Install Quack and edit "
        "rachita_extensions/quack_kernels.py::_install_cross_entropy to enable."
    )
    return False
