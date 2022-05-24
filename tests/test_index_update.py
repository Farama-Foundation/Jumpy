# pylint:disable=redefined-builtin


import pytest
import numpy as onp

import jumpy as jp


@pytest.mark.parametrize("array,index,val", [
    (jp.arange(1, 11), 0, 10),
    (jp.arange(1, 13).reshape((4, 3)), 3, jp.arange(1, 4)),
    (jp.arange(1, 13).reshape((4, 3)), jp.array([0, 3]), jp.arange(1, 7).reshape((2, 3)))
])
def test_index_update(array, index, val):
    x = jp.index_update(array, index, val)
    assert onp.array_equal(x[index], val)