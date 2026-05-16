"""Compatibility shim for the official Hopper FlashAttention-3 source build.

The source build installed on Clariden exposes its Python functions through a
top-level ``flash_attn_interface`` module, while TransformerEngine imports the
same API from ``flash_attn_3.flash_attn_interface``. Keeping this as a
namespace-package shim lets TransformerEngine import FA3 without patching the
site package or Megatron.
"""

import flash_attn_interface as _fa3_interface

from flash_attn_interface import *  # noqa: F401,F403

for _name in dir(_fa3_interface):
    if _name.startswith("_flash_attn"):
        globals()[_name] = getattr(_fa3_interface, _name)

__all__ = [name for name in globals() if not name.startswith("__")]
