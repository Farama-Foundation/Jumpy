# pylint:disable=redefined-builtin


import pytest
import numpy as onp
from jax import numpy as jnp

import jumpy as jp


@pytest.mark.parametrize("array, index, val", [
    # Basic test
    (jp.arange(1, 11), 0, 10),
    # Multidimensional array
    (jp.arange(1, 13).reshape((4, 3)), 3, jp.arange(1, 4)),
    # Multidimensional index and value
    (jp.arange(1, 13).reshape((4, 3)), jp.array([0, 3]), jp.arange(1, 7).reshape((2, 3))),
    # Negative index
    (jp.arange(1, 11), -2, 10),
    # Jax array
    (jp.arange(1, 11, dtype=jnp.float32), 0, 10),
])
def test_correct_updates(array, index, val):
    """
    Checks whether the value is actually updated at given index
    """
    x = jp.index_update(array, index, val)
    assert onp.array_equal(x[index], val)


@pytest.mark.parametrize("array, index, val", [
    (jp.arange(1, 13).reshape((4, 3)), jp.array([0, 3]), jp.arange(1, 7).reshape((3, 2))),
    (jp.arange(1, 13).reshape((4, 3)), jp.array([0, 3]), jp.arange(1, 7)),
])
def test_shape_mismatch(array, index, val):
    """
    Checks whether a shape mismatch error is thrown
    """
    with pytest.raises(ValueError):
        jp.index_update(array, index, val)