# pylint:disable=redefined-builtin


import numpy as onp

import jumpy as jp


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