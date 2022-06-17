# pylint:disable=redefined-builtin


import pytest
import jax
import numpy as onp
from jax import numpy as jnp

import jumpy as jp


def _convert_to_jax_axes(booleans):
    return [0 if b else None for b in booleans]


@pytest.mark.parametrize("fun, args, includes", [
    # Two-parameters function
    (lambda x, y: jp.dot(x, y), (jp.ones((6, 5, 4)), jp.ones((6, 4, 3))), [False, True]),
    # Three-parameters function
    (lambda x, y, z: jp.dot(jp.dot(x, y), z), (jp.ones((6, 5, 4)), jp.ones((6, 4, 3)), jp.ones((6, 3, 7))), [False, True, True]),
    (lambda x, y, z: jp.dot(jp.dot(x, y), z), (jp.ones((6, 5, 4)), jp.ones((6, 4, 3)), jp.ones((6, 3, 7))), [False, True, False]),
    # Jax input
    (lambda x, y: jp.dot(x, y), (jp.ones((6, 5, 4), dtype=jnp.float32), jp.ones((6, 4, 3), dtype=jnp.float32)), [False, True]),
])
def test_vmap(fun, args, includes):
    """ 
    Tests `vmap` function equivalency with `jax.vmap`
    """
    x1 = jp.vmap(fun, include=includes)(*args)
    y1 = jax.vmap(fun, in_axes=_convert_to_jax_axes(includes))(*args)
    assert onp.array_equal(x1, y1)
    
    # Empty include
    x1 = jp.vmap(fun, include=None)(*args)
    y1 = jax.vmap(fun, in_axes=0)(*args)
    assert onp.array_equal(x1, y1)
    
    # Wrong input number
    with pytest.raises(RuntimeError):
        jp.vmap(fun, include=includes + [True])(*args)