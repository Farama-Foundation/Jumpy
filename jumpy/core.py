"""Core functions for jumpy."""

from __future__ import annotations

import builtins
from typing import Any

import numpy as onp

import jumpy as jp

__all__ = [
    "is_jitted",
    "which_np",
    "which_dtype",
    "custom_jvp",
]

try:
    import jax
    import jax.numpy as jnp
    from jax import custom_jvp, tree_map
    from jax.interpreters.batching import BatchTracer
except ImportError:
    jax, jnp = None, None

    BatchTracer = None
    tree_map = None

    class custom_jvp:
        """Custom JVP decorator."""

        def __init__(self, func: callable):
            """Initialise the custom jvp with func."""
            self.func = func

        def __call__(self, *args, **kwargs):
            """Calls the func with `args` and `kwargs`."""
            return self.func(*args, **kwargs)

        def defjvp(self, func):
            """For numpy only, we ignore the defjvp function."""


def is_jitted() -> bool:
    """Returns true if currently inside a jax.jit call or jit is disabled."""
    if jp.is_jax_installed is False:
        return False
    elif jax.config.jax_disable_jit:
        return True
    else:
        return jax.core.cur_sublevel().level > 0


def which_np(*args: Any):
    """Returns which numpy implementation (Numpy or Jax) based on the arguments."""
    if jp.is_jax_installed is False:
        return onp

    checker = lambda a: (  # noqa: E731
        isinstance(a, (jnp.ndarray, BatchTracer)) and not isinstance(a, onp.ndarray)
    )
    if builtins.any(jax.tree_util.tree_leaves(tree_map(checker, args))):
        return jnp
    else:
        return onp


def which_dtype(dtype: object | None):
    """Returns np or jnp depending on dtype."""
    if jp.is_jax_installed and dtype is not None and dtype.__module__ == "jax.numpy":
        return jnp
    else:
        return onp
