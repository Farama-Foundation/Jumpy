# pylint:disable=redefined-builtin


import pytest

import jax
import numpy as onp

import jumpy as jp


def test_vmap():
    A = jp.ones((6, 5, 4))
    B = jp.ones((6, 4, 3))
    C = jp.ones((6, 3, 7))

    fun1 = lambda x, y: jp.dot(x, y)
    fun2 = lambda x, y, z: jp.dot(jp.dot(x, y), z)

    x1, y1 = jp.vmap(fun1, include=(False, True))(A, B), jax.vmap(fun1, in_axes=(None, 0))(A, B)
    x2, y2 = jp.vmap(fun2, include=(False, True, True))(A, B, C), jax.vmap(fun2, in_axes=(None, 0, 0))(A, B, C)
    x3, y3 = jp.vmap(fun2, include=(False, True, False))(A, B, C), jax.vmap(fun2, in_axes=(None, 0, None))(A, B, C)

    assert onp.array_equal(x1, y1)
    assert onp.array_equal(x2, y2)
    assert onp.array_equal(x3, y3)

    # Empty include
    x1, y1 = jp.vmap(fun1, include=None)(A, B), jax.vmap(fun1, in_axes=0)(A, B)
    x2, y2 = jp.vmap(fun2, include=None)(A, B, C), jax.vmap(fun2, in_axes=0)(A, B, C)

    assert onp.array_equal(x1, y1)
    assert onp.array_equal(x2, y2)

    # Wrong input number
    with pytest.raises(RuntimeError):
        jp.vmap(fun1, include=[True])(A, B)
        jp.vmap(fun2, include=[True, False])(A, B, C)