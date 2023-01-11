"""Tests that the `norm` and `safe_norm` work as expected."""


import jax.numpy as jnp
import numpy as onp
import pytest
from jax import grad

import jumpy.numpy as jp


def test_norm():
    """Calls the function `norm` on both `onp.array` and `jnp.array` to check it doesn't raise an error."""
    A = onp.array([1, 2, 3])
    B = jnp.array([1, 2, 3])

    jp.linalg.norm(A)
    jp.linalg.norm(B)


@pytest.mark.parametrize(
    "x, axis",
    [
        ([1, 0, 1], None),
        # Bidimensional input
        ([[1, 0], [1, 1]], None),
        # Integer axis
        ([[1, 0], [1, 1]], 1),
        # Tuple axis
        ([[1, 0], [1, 1]], (0, 1)),
    ],
)
def test_safe_norm(x, axis):
    """Checks equivalence in jax and numpy arrays."""
    x = jnp.array(x)
    norm = jp.linalg.safe_norm(x, axis=axis)

    if isinstance(norm, jnp.ndarray):
        assert onp.array_equal(norm, jnp.linalg.norm(x, axis=axis))
    else:
        assert norm == jnp.linalg.norm(x, axis=axis)


@pytest.mark.parametrize(
    "x, axis, res",
    [
        (jp.array([1, 0, 0], dtype=float), None, jp.array([1, 0, 0], dtype=float)),
        # Jax input
        (
            jp.array([1, 0, 0], dtype=jnp.float32),
            None,
            jp.array([1, 0, 0], dtype=jnp.float32),
        ),
        # "Unsafe" input
        (jp.zeros(3, dtype=float), None, jp.zeros(3, dtype=float)),
        # Empty input
        (jp.array([], dtype=float), None, jp.array([], dtype=float)),
        # Bidimensional input
        (
            jp.array([[1, 0], [0, 0]], dtype=float).reshape((2, 2)),
            None,
            jp.array([[1, 0], [0, 0]], dtype=float).reshape((2, 2)),
        ),
    ],
)
def test_gradient(x, axis, res):
    """Tests the gradient of the `safe_norm` function."""
    fun = grad(lambda y: jp.linalg.safe_norm(y, axis=axis))

    assert onp.array_equal(fun(x), res)
