# pylint:disable=redefined-builtin


import pytest
import numpy as onp
from jax import numpy as jnp

import jumpy as jp


@pytest.mark.parametrize("data, segment_ids, ret", [
    # Numpy array
    (jp.arange(0, 5, dtype=int), jp.array([0, 0, 1, 1, 2], dtype=int), jp.array([1, 5, 4], dtype=int)),
    # Jax array
    (jp.arange(0, 5, dtype=jnp.int32), jp.array([0, 0, 1, 1, 2], dtype=jnp.int32), jp.array([1, 5, 4], dtype=jnp.int32)),
])
def test_segment_sum(data, segment_ids, ret):
    """
    Checks if the result of `segment_sum` on the given inputs equals the
    expected output 
    """
    assert onp.array_equal(jp.segment_sum(data, segment_ids), ret)
    