# pylint:disable=redefined-builtin


import pytest
import jax
import numpy as onp
import jax.numpy as jnp

import jumpy as jp


@pytest.mark.parametrize("array, length", [
    (jp.arange(1, 11).reshape((5, 2)), 5),
    # Jax input
    (jp.arange(1, 11, dtype=jnp.float32).reshape((5, 2)), 5),
])
def test_array(array, length):
    """ 
    Tests `scan` function equivalency with `jax.lax.scan` on an 
    array input
    """
    def fun(x, y):
        x += y.sum()
        return x, y * x
    
    carry1, y1 = jax.lax.scan(fun, 0, array, length=length)
    carry2, y2 = jp.scan(fun, 0, array, length=length)

    assert onp.array_equal(carry1, carry2)
    assert onp.array_equal(y1, y2)

    # Reversed
    carry1, y1 = jax.lax.scan(fun, 0, array, length=length, reverse=True)
    carry2, y2 = jp.scan(fun, 0, array, length=length, reverse=True)

    assert onp.array_equal(carry1, carry2)
    assert onp.array_equal(y1, y2)


@pytest.mark.parametrize("dict, length", [
    ({"k1": jp.arange(1, 11).reshape((5, 2)), "k2": jp.arange(1, 11).reshape((5, 2))}, 5)
])
def test_dict(dict, length):
    """ 
    Tests `scan` function equivalency with `jax.lax.scan` on a
    dictionary input
    """
    def fun(x, y):
        ret = {}
        for key, val in y.items():
            x += jnp.sum(val)
            ret[key] = val * -1
        return x, ret
    
    carry1, y1 = jax.lax.scan(fun, 0, dict, length=length)
    carry2, y2 = jp.scan(fun, 0, dict, length=length)

    assert onp.array_equal(carry1, carry2)
    assert all(onp.array_equal(y1[key], y2[key]) for key in y1)