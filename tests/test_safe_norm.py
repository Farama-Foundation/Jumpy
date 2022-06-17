# pylint:disable=redefined-builtin


import pytest
from jax import grad
import numpy as onp
from jax import numpy as jnp

import jumpy as jp


@pytest.mark.parametrize("x, axis", [
    ([1, 0, 1], None),
    # Bidimensional input
    ([[1, 0], [1, 1]], None),
    # Integer axis
    ([[1, 0], [1, 1]], 1),
    # Tuple axis
    ([[1, 0], [1, 1]], (0, 1)),
])
def test_norm(x, axis):
    """
    Checks equivalence in jax and numpy arrays
    """
    x = jnp.array(x)
    norm = jp.safe_norm(x, axis=axis)

    if isinstance(norm, jnp.ndarray):
        assert onp.array_equal(norm, jnp.linalg.norm(x, axis=axis))
    else:
        assert norm == jnp.linalg.norm(x, axis=axis)


@pytest.mark.parametrize("x, axis, res", [
    (jp.array([1, 0, 0], dtype=float), None, jp.array([1, 0, 0], dtype=float)),
    # Jax input
    (jp.array([1, 0, 0], dtype=jnp.float32), None, jp.array([1, 0, 0], dtype=jnp.float32)),
    # "Unsafe" input 
    (jp.zeros(3, dtype=float), None, jp.zeros(3, dtype=float)),
    # Empty input
    (jp.array([], dtype=float), None, jp.array([], dtype=float)),
    # Bidimensional input
    (jp.array([[1, 0], [0, 0]], dtype=float).reshape((2, 2)), None, jp.array([[1, 0], [0, 0]], dtype=float).reshape((2, 2))),
])
def test_gradient(x, axis, res):
    """
    Tests the gradient of the `safe_norm` function
    """
    fun = grad(lambda x: jp.safe_norm(x, axis=axis))

    assert onp.array_equal(fun(x), res)