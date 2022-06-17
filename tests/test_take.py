# pylint:disable=redefined-builtin


import pytest
import numpy as onp
from jax import numpy as jnp

import jumpy as jp


@pytest.mark.parametrize("array, indexes, ret", [
    (jp.arange(0, 5), [1, 0, 2, 3], jp.array([1, 0, 2, 3])),
    # Empty array
    (jp.array([]), [], []),
    # Jax array
    (jp.arange(0, 5, dtype=jnp.float32), [1, 0, 2, 3], jp.array([1, 0, 2, 3])),
])
def test_take(array, indexes, ret):
    """
    Checks if the result of `take` on the given inputs equals the
    expected output 
    """
    x = jp.take(array, indexes)
    assert onp.array_equal(x, ret)