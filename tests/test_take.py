# pylint:disable=redefined-builtin


import pytest
import numpy as onp

import jumpy as jp


@pytest.mark.parametrize("array,indexes,ret", [
    (jp.arange(0, 5), [1, 0, 2, 3], jp.array([1, 0, 2, 3]))
])
def test_take(array, indexes, ret):
    x = jp.take(array, indexes)
    assert onp.array_equal(x, ret)