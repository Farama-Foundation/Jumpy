# pylint:disable=redefined-builtin


import pytest

import jax
import jax.numpy as jnp
import numpy as onp

import jumpy as jp


def _check_types_multiple(in_args, out_arg):
    assert isinstance(out_arg, jnp.ndarray) if any(
        [isinstance(a, jnp.ndarray) for a in in_args]
    ) else isinstance(out_arg, onp.ndarray)


def _check_types(in_arg, out_arg):
    _check_types_multiple([in_arg], out_arg)


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


def test_scan():
    A, A_len = jp.arange(1, 11).reshape((5, 2)), 5
    B, B_len = {"k1": jp.arange(1, 11).reshape((5, 2)), "k2": (jp.arange(1, 11).reshape((5, 2)), jp.arange(1, 11).reshape((5, 2)))}, 5

    def fun1(x, y):
        x += y.sum()
        return x, y * x

    def fun2(x, y):
        x += y["k1"].sum() * y["k2"][0].sum() * y["k2"][1].sum()
        return x, {"k1": y["k1"] * -1, "k2": (y["k2"][0] * 2, y["k2"][1] * 3)}
    
    (x1, x2), (y1, y2) = jax.lax.scan(fun1, 0, A, length=A_len), jp.scan(fun1, 0, A, length=A_len)

    assert onp.array_equal(x1, y1)
    assert onp.array_equal(x2, y2)

    # Reversed
    (x1, x2), (y1, y2) = jax.lax.scan(fun1, 0, A, length=A_len, reverse=True), jp.scan(fun1, 0, A, length=A_len, reverse=True)

    assert onp.array_equal(x1, y1)
    assert onp.array_equal(x2, y2)

    # Pytree argument
    (x1, x2), (y1, y2) = jax.lax.scan(fun2, 0, B, length=B_len), jp.scan(fun2, 0, B, length=B_len)

    assert onp.array_equal(x1, y1)
    assert onp.array_equal(x2["k1"], y2["k1"])
    assert onp.array_equal(x2["k2"][0], y2["k2"][0])
    assert onp.array_equal(x2["k2"][1], y2["k2"][1])


def test_take():
    A = jp.arange(0, 5)
    B = jp.arange(0, 20).reshape((5, 4))
    C = [1, 0, 2, 3]

    x = jp.take(A, C)
    assert onp.array_equal(x, jp.array([1, 0, 2, 3]))


def test_index_update():
    A = jp.arange(1, 11)
    B = jp.arange(1, 13).reshape((4, 3))
    C = jp.arange(1, 4)
    D = jp.array([0, 3])
    E = jp.arange(1, 7).reshape((2, 3))

    x = jp.index_update(A, 0, 10)
    assert x[0] == 10

    x = jp.index_update(B, 3, C)
    assert onp.array_equal(x[3], C)

    x = jp.index_update(B, D, E)
    assert onp.array_equal(x[D[0]], E[0])
    assert onp.array_equal(x[D[1]], E[1])


def test_dot():
    A = onp.arange(9).reshape((3, 3))
    B = jnp.arange(9).reshape((3, 3))
        
    _check_types(A, jp.dot(A, A))
    _check_types(B, jp.dot(B, B))
    _check_types_multiple((A,B), jp.dot(A, B))