"""Tests that `top_k` works as expected."""

import numpy as onp
import pytest
from jax import numpy as jnp

import jumpy as jp


@pytest.mark.parametrize(
    "x, k, ret",
    [
        (jp.array([5, 7, 1, 2]), 2, ([7, 5], [1, 0])),
        (jp.array([5, 9, 7, 1, 2]), 3, ([9, 7, 5], [1, 2, 0])),
        # Jax array
        (jp.array([5, 7, 1, 2], dtype=jnp.int32), 2, ([7, 5], [1, 0])),
        (jp.array([5, 9, 7, 1, 2], dtype=jnp.int32), 3, ([9, 7, 5], [1, 2, 0])),
    ],
)
def test_top_k(x, k, ret):
    """Checks if the result of `top_k` on the given inputs equals the expected output."""
    top, indices = jp.lax.top_k(x, k)
    assert len(top) == len(indices) == len(ret[0]) == len(ret[1])
    assert onp.array_equal(top, ret[0])
    # Checks that the index values in x are equal to the top values.
    assert all(t == x[i] for t, i in zip(top, indices))
