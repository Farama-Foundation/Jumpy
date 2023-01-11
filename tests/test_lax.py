"""Tests that `scan` works as expected."""

import jax
import jax.numpy as jnp
import numpy as onp
import pytest

import jumpy
import jumpy.numpy as jp


@pytest.mark.parametrize(
    "array, length",
    [
        (jp.arange(1, 11).reshape((5, 2)), 5),
        # Jax input
        (jp.arange(1, 11, dtype=jnp.float32).reshape((5, 2)), 5),
    ],
)
def test_scan_array(array, length):
    """Tests `scan` function equivalency with `jax.lax.scan` on an array input."""

    def fun(x, y):
        x += y.sum()
        return x, y * x

    carry1, y1 = jax.lax.scan(fun, 0, array, length=length)
    carry2, y2 = jumpy.lax.scan(fun, 0, array, length=length)

    assert onp.array_equal(carry1, carry2)
    assert onp.array_equal(y1, y2)

    # Reversed
    carry1, y1 = jax.lax.scan(fun, 0, array, length=length, reverse=True)
    carry2, y2 = jumpy.lax.scan(fun, 0, array, length=length, reverse=True)

    assert onp.array_equal(carry1, carry2)
    assert onp.array_equal(y1, y2)


@pytest.mark.parametrize(
    "dictionary, length",
    [
        (
            {
                "k1": jp.arange(1, 11).reshape((5, 2)),
                "k2": jp.arange(1, 11).reshape((5, 2)),
            },
            5,
        )
    ],
)
def test_scan_dict(dictionary, length):
    """Tests `scan` function equivalency with `jax.lax.scan` on a dictionary input."""

    def fun(x, y):
        ret = {}
        for key, val in y.items():
            x += jnp.sum(val)
            ret[key] = val * -1
        return x, ret

    carry1, y1 = jax.lax.scan(fun, 0, dictionary, length=length)
    carry2, y2 = jumpy.lax.scan(fun, 0, dictionary, length=length)

    assert onp.array_equal(carry1, carry2)
    assert all(onp.array_equal(y1[key], y2[key]) for key in y1)


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
    top, indices = jumpy.lax.top_k(x, k)
    assert len(top) == len(indices) == len(ret[0]) == len(ret[1])
    assert onp.array_equal(top, ret[0])
    # Checks that the index values in x are equal to the top values.
    assert all(t == x[i] for t, i in zip(top, indices))
