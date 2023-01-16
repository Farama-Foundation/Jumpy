"""Module for linear algebra for jumpy."""

from __future__ import annotations

import numpy as onp

from jumpy.core import which_np
from jumpy.numpy._types import ndarray

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None


__all__ = ["inv", "norm", "safe_norm"]


def inv(a: ndarray) -> ndarray:
    """Compute the (multiplicative) inverse of a matrix."""
    return which_np(a).linalg.inv(a)


def norm(x: ndarray, axis: tuple[int, ...] | int | None = None) -> ndarray:
    """Returns the array norm."""
    return which_np(x, axis).linalg.norm(x, axis=axis)


def safe_norm(x: ndarray, axis: tuple[int, ...] | int | None = None) -> ndarray:
    """Calculates a linalg.norm(x) that's safe for gradients at x=0.

    Avoids a poorly defined gradient for jnp.linalg.norm(0) see
    https://github.com/google/jax/issues/3058 for details

    Args:
      x: A jp.ndarray
      axis: The axis along which to compute the norm

    Returns:
      Norm of the array x.
    """
    np = which_np(x)
    if np is jnp:
        is_zero = jnp.allclose(x, 0.0)
        # temporarily swap x with ones if is_zero, then swap back
        x = jnp.where(is_zero, jnp.ones_like(x), x)
        n = jnp.linalg.norm(x, axis=axis)
        n = jnp.where(is_zero, 0.0, n)
    else:
        n = onp.linalg.norm(x, axis=axis)
    return n
