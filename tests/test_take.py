# pylint:disable=redefined-builtin


import numpy as onp

import jumpy as jp



def test_take():
    A = jp.arange(0, 5)
    B = jp.arange(0, 20).reshape((5, 4))
    C = [1, 0, 2, 3]

    x = jp.take(A, C)
    assert onp.array_equal(x, jp.array([1, 0, 2, 3]))