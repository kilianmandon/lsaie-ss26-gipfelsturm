"""Megatron-LM training entrypoint with optional Rachita extensions.

This file is a *thin* wrapper around ``Megatron-LM/pretrain_gpt.py``. It is
selected automatically by ``launch_with_config.py`` whenever a config sets one
of:

    torch_compile: true
    use_quack: true

For every other config (and for all teammate workflows), the launcher keeps
using the upstream ``Megatron-LM/pretrain_gpt.py`` directly, so this file does
not affect the baseline / FA3 / FA4 / FP8 paths.

Behavioural changes vs upstream (all gated by env vars):

* If ``MEGATRON_USE_QUACK=1``, Quack monkey-patches are installed *before*
  Megatron is imported (see ``rachita_extensions/quack_kernels.py``).
* If ``MEGATRON_TORCH_COMPILE=1``, every ``TransformerLayer`` returned from
  Megatron's ``model_provider`` is wrapped with ``nn.Module.compile()`` (see
  ``rachita_extensions/torch_compile_wrap.py``).

Everything else - argument parsing, distributed init, optimizer, data loader,
W&B logging, CUDA-graph capture, checkpointing - is delegated 1:1 to Megatron.
"""

import logging
import os
import sys
import time

# Capture program start time BEFORE any heavy imports so Megatron's startup-
# timing report (which is printed by pretrain()) matches what the upstream
# pretrain_gpt.py would produce.
_PROGRAM_START_TIME = time.time()

logging.basicConfig(
    level=os.environ.get("RACHITA_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("rachita.entrypoint")


def _bootstrap_paths() -> None:
    """Make Megatron-LM and our extensions importable.

    ``launch_with_config.py`` sets ``PYTHONPATH`` to include both this
    directory and ``Megatron-LM/``, but we also handle a direct ``python
    rachita_runs/pretrain_gpt_rachita.py`` invocation for sanity tests.
    """

    here = os.path.dirname(os.path.abspath(__file__))
    workdir = os.environ.get("WORKDIR") or os.path.dirname(here)
    megatron_dir = os.environ.get("MEGATRON_LM_DIR") or os.path.join(workdir, "Megatron-LM")

    for path in (here, megatron_dir):
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)

    pretrain_gpt_path = os.path.join(megatron_dir, "pretrain_gpt.py")
    if not os.path.isfile(pretrain_gpt_path):
        raise SystemExit(
            "[rachita] FATAL: Megatron-LM is not initialised. Expected "
            "{p!r} to exist but it doesn't. Run on the LOGIN node:\n\n"
            "    cd {w}\n"
            "    rm -rf Megatron-LM\n"
            "    git submodule update --init\n\n"
            "Then resubmit. (See the project README, section 'Setup', step 2.)"
            .format(p=pretrain_gpt_path, w=workdir)
        )


_bootstrap_paths()


from rachita_extensions import quack_kernels, torch_compile_wrap

if quack_kernels.enabled():
    quack_kernels.maybe_install()

from functools import partial  # noqa: E402

import pretrain_gpt as _upstream  # noqa: E402
import gpt_builders  # provides gpt_builder, lives next to pretrain_gpt.py  # noqa: E402

try:
    from megatron.core.enums import ModelType  # noqa: E402
except ImportError:
    from megatron.core.parallel_state import ModelType  # type: ignore  # noqa: E402

from megatron.training import inprocess_restart, pretrain, set_startup_timestamps  # noqa: E402


_orig_model_provider = _upstream.model_provider


def model_provider(*args, **kwargs):
    """Wrapper that mirrors upstream signature (model_builder, *args, **kwargs)
    and optionally compiles the resulting model with torch.compile."""
    model = _orig_model_provider(*args, **kwargs)
    if torch_compile_wrap.enabled():
        model = torch_compile_wrap.compile_model(model)
    return model


def main() -> None:
    # Match upstream pretrain_gpt.py's __main__ block exactly, with the single
    # difference that the inner model_provider may be torch.compile-wrapped.
    _MAIN_ENTRY_TIME = time.time()
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_MAIN_ENTRY_TIME)

    datasets_provider = _upstream.train_valid_test_datasets_provider
    if not getattr(datasets_provider, "is_distributed", False):
        datasets_provider.is_distributed = True

    log.info(
        "[rachita] entry: torch_compile=%s quack=%s",
        torch_compile_wrap.enabled(), quack_kernels.enabled(),
    )

    wrapped_pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    extra_args_provider = (
        _upstream.add_modelopt_args
        if getattr(_upstream, "has_nvidia_modelopt", False)
        else None
    )

    wrapped_pretrain(
        datasets_provider,
        partial(model_provider, gpt_builders.gpt_builder),
        ModelType.encoder_or_decoder,
        _upstream.forward_step,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
        extra_args_provider=extra_args_provider,
        store=store,
        get_embedding_ranks=_upstream.get_embedding_ranks,
    )


if __name__ == "__main__":
    main()
