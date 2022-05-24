# pylint:disable=redefined-builtin


import jax
import numpy as onp

import jumpy as jp


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