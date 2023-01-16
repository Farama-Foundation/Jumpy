"""Jumpy module."""

try:
    import jax.numpy as jnp  # noqa: F401

    is_jax_installed = True
except ImportError:
    is_jax_installed = False


from jumpy import core, lax, ops, random
from jumpy._base_fns import index_update, vmap

__all__ = [
    # === primitives ===
    # === sub-modules ===
    "core",
    "random",
    "numpy",
    "lax",
    "ops",
    # === base functions ==
    "index_update",
    "vmap",
]
