"""Tests that the index update and vmap (vectorised map) works as expected."""
import jax
import jax.numpy as jnp
import numpy as onp
import pytest

import jumpy
import jumpy.numpy as jp


@pytest.mark.parametrize(
    "array, index, val",
    [
        # Basic test
        (jp.arange(1, 11), 0, 10),
        # Multidimensional array
        (jp.arange(1, 13).reshape((4, 3)), 3, jp.arange(1, 4)),
        # Multidimensional index and value
        (
            jp.arange(1, 13).reshape((4, 3)),
            jp.array([0, 3]),
            jp.arange(1, 7).reshape((2, 3)),
        ),
        # Negative index
        (jp.arange(1, 11), -2, 10),
        # Jax array
        (jp.arange(1, 11, dtype=jnp.float32), 0, 10),
    ],
)
def test_correct_updates(array, index, val):
    """Checks whether the value is actually updated at given index."""
    x = jumpy.index_update(array, index, val)
    assert onp.array_equal(x[index], val)


@pytest.mark.parametrize(
    "array, index, val",
    [
        (
            jp.arange(1, 13).reshape((4, 3)),
            jp.array([0, 3]),
            jp.arange(1, 7).reshape((3, 2)),
        ),
        (jp.arange(1, 13).reshape((4, 3)), jp.array([0, 3]), jp.arange(1, 7)),
    ],
)
def test_shape_mismatch(array, index, val):
    """Checks whether a shape mismatch error is thrown."""
    with pytest.raises(ValueError):
        jumpy.index_update(array, index, val)


def _convert_to_jax_axes(booleans):
    return [0 if b else None for b in booleans]


@pytest.mark.parametrize(
    "fun, args, includes",
    [
        # Two-parameters function
        (
            lambda x, y: jp.dot(x, y),
            (jp.ones((6, 5, 4)), jp.ones((6, 4, 3))),
            [False, True],
        ),
        # Three-parameters function
        (
            lambda x, y, z: jp.dot(jp.dot(x, y), z),
            (jp.ones((6, 5, 4)), jp.ones((6, 4, 3)), jp.ones((6, 3, 7))),
            [False, True, True],
        ),
        (
            lambda x, y, z: jp.dot(jp.dot(x, y), z),
            (jp.ones((6, 5, 4)), jp.ones((6, 4, 3)), jp.ones((6, 3, 7))),
            [False, True, False],
        ),
        # Jax input
        (
            lambda x, y: jp.dot(x, y),
            (
                jp.ones((6, 5, 4), dtype=jnp.float32),
                jp.ones((6, 4, 3), dtype=jnp.float32),
            ),
            [False, True],
        ),
    ],
)
def test_vmap(fun, args, includes):
    """Tests `vmap` function equivalency with `jax.vmap`."""
    x1 = jumpy.vmap(fun, include=includes)(*args)
    y1 = jax.vmap(fun, in_axes=_convert_to_jax_axes(includes))(*args)
    assert onp.array_equal(x1, y1)

    # Empty include
    x1 = jumpy.vmap(fun, include=None)(*args)
    y1 = jax.vmap(fun, in_axes=0)(*args)
    assert onp.array_equal(x1, y1)

    # Wrong input number
    with pytest.raises(RuntimeError):
        jumpy.vmap(fun, include=includes + [True])(*args)
