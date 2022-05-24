# pylint:disable=redefined-builtin


import pytest
import jax
import numpy as onp
import jax.numpy as jnp

import jumpy as jp


@pytest.mark.parametrize("array,length", [
    (jp.arange(1, 11).reshape((5, 2)), 5)
])
def test_array(array, length):
    def fun(x, y):
        x += y.sum()
        return x, y * x
    
    (x1, x2), (y1, y2) = jax.lax.scan(fun, 0, array, length=length), jp.scan(fun, 0, array, length=length)

    assert onp.array_equal(x1, y1)
    assert onp.array_equal(x2, y2)

    # Reversed
    (x1, x2), (y1, y2) = jax.lax.scan(fun, 0, array, length=length, reverse=True), jp.scan(fun, 0, array, length=length, reverse=True)

    assert onp.array_equal(x1, y1)
    assert onp.array_equal(x2, y2)


def compare_ndarray_dicts(first, second):
    """Return whether two dicts of arrays are exactly equal"""
    if first.keys() != second.keys():
        return False
    return all(onp.array_equal(first[key], second[key]) for key in first)


@pytest.mark.parametrize("tree,length", [
    ({"k1": jp.arange(1, 11).reshape((5, 2)), "k2": jp.arange(1, 11).reshape((5, 2))}, 5)
])
def test_dict(tree, length):
    def fun(x, y):
        ret = {}
        for key, val in y.items():
            x += jnp.sum(val)
            ret[key] = val * -1
        return x, ret
    
    (x1, x2), (y1, y2) = jax.lax.scan(fun, 0, tree, length=length), jp.scan(fun, 0, tree, length=length)

    assert onp.array_equal(x1, y1)
    assert all(onp.array_equal(x2[key], y2[key]) for key in x2)