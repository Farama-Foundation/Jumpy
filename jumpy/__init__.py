"""Jumpy module."""

from typing import Union

import numpy as onp

dtype = onp.dtype
pi = onp.pi
inf = onp.inf
float32 = onp.float32
int32 = onp.int32

try:
    import jax.numpy as jnp

    ndarray = Union[onp.ndarray, jnp.ndarray]

    is_jax_installed = True
except ImportError:
    ndarray = onp.ndarray

    is_jax_installed = False


from jumpy import lax, ops, random
from jumpy._base_fns import index_update, vmap
from jumpy._indexing import jparray

__all__ = [
    # === primitives ===
    "dtype",
    "pi",
    "inf",
    "float32",
    "int32",
    "ndarray",
    "jparray",
    # === sub-modules ===
    "random",
    "numpy",
    "lax",
    "ops",
    # === base functions ==
    "index_update",
    "vmap",
]
