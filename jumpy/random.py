"""Module for random functions in jumpy."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as onp

from jumpy.core import which_np

if TYPE_CHECKING:
    from jumpy.numpy import ndarray

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax, jnp = None, None


__all__ = [
    "PRNGKey",
    "uniform",
    "split",
    "randint",
    "choice",
]


def PRNGKey(seed: int) -> ndarray:
    """Returns a PRNG key given a seed."""
    # NOTE: selects backend based on seed type.
    if which_np(seed) is jnp:
        return jax.random.PRNGKey(seed)
    else:
        rng = onp.random.default_rng(seed)
        return rng.integers(low=0, high=2**32, dtype="uint32", size=2)


def uniform(
    rng: ndarray,
    shape: tuple[int, ...] = (),
    low: float | None = 0.0,
    high: float | None = 1.0,
) -> ndarray:
    """Sample uniform random values in [low, high) with given shape/dtype."""
    if which_np(rng) is jnp:
        return jax.random.uniform(rng, shape=shape, minval=low, maxval=high)
    else:
        return onp.random.default_rng(rng).uniform(size=shape, low=low, high=high)


def split(rng: ndarray, num: int = 2) -> ndarray:
    """Splits a PRNG key into num new keys by adding a leading axis."""
    if which_np(rng) is jnp:
        return jax.random.split(rng, num=num)
    else:
        rng = onp.random.default_rng(rng)
        return rng.integers(low=0, high=2**32, dtype="uint32", size=(num, 2))


def randint(
    rng: ndarray,
    shape: tuple[int, ...] = (),
    low: int | None = 0,
    high: int | None = 1,
) -> ndarray:
    """Sample integers in [low, high) with given shape."""
    if which_np(rng) is jnp:
        return jax.random.randint(rng, shape=shape, minval=low, maxval=high)
    else:
        return onp.random.default_rng(rng).integers(low=low, high=high, size=shape)


def choice(
    rng: ndarray,
    a: int | ndarray,
    shape: tuple[int, ...] = (),
    replace: bool = True,
    p: Any | None = None,
    axis: int = 0,
) -> ndarray:
    """Generate sample(s) from given array."""
    if which_np(rng) is jnp:
        return jax.random.choice(rng, a, shape=shape, replace=replace, p=p, axis=axis)
    else:
        return onp.random.default_rng(rng).choice(
            a, size=shape, replace=replace, p=p, axis=axis
        )
