# pylint:disable=redefined-builtin


import pytest
import numpy as onp
from jax import numpy as jnp

import jumpy as jp


@pytest.mark.parametrize("x, k, ret", [
    (jp.array([5, 7, 1, 2]), 2, ([5, 7], [0, 1])),
    # Jax array
    (jp.array([5, 7, 1, 2], dtype=jnp.float32), 2, ([5, 7], [0, 1])),
])
def test_top_k(x, k, ret):
    """
    Checks if the result of `top_k` on the given inputs equals the
    expected output 
    """
    top, indices = jp.top_k(x, k)
    assert len(top) == len(indices) == len(ret[0]) == len(ret[1])
    assert onp.array_equal(onp.sort(top), onp.sort(ret[0]))
    assert all([top[i] == x[i] for i in indices])