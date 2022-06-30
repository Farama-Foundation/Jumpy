# pylint:disable=redefined-builtin


import jax.numpy as jnp
import numpy as onp
from jax import jit

import jumpy as jp


def _check_types_multiple(in_args, out_arg):
    """ 
    Checks whether the output argument is either a `jnp.ndarray` when 
    a `jnp.ndarray` is given in the input list, or a `onp.ndarray` 
    """
    assert isinstance(out_arg, jnp.ndarray) if any(
        [isinstance(a, jnp.ndarray) for a in in_args]
    ) else isinstance(out_arg, onp.ndarray)


def _check_types(in_arg, out_arg):
    """ 
    Checks whether the output argument is either a `jnp.ndarray` when 
    a `jnp.ndarray` is given in the single input, or a `onp.ndarray` 
    """
    _check_types_multiple([in_arg], out_arg)


def test_norm():
    """
    Calls the function `norm` on both `onp.array` and `jnp.array`
    to check it doesn't raise an error
    """
    A = onp.array([1, 2, 3])
    B = jnp.array([1, 2, 3])

    jp.norm(A)
    jp.norm(B)


def test_any():
    """
    Calls the function `norm` on both `onp.array` and `jnp.array`
    to check it doesn't raise an error
    """
    A = onp.array([False, True, False])
    B = jnp.array([False, True, False])

    jp.any(A)
    jp.any(B)


def test_all():
    """
    Calls the function `all` on both `onp.array` and `jnp.array`
    to check it doesn't raise an error
    """
    A = onp.array([True, True])
    B = jnp.array([True, True])

    jp.all(A)
    jp.all(B)


def test_mean():
    """
    Calls the function `mean` on both `onp.array` and `jnp.array`
    to check it doesn't raise an error
    """
    A = onp.array([1 ,2, 3])
    B = jnp.array([1, 2, 3])

    jp.mean(A)
    jp.mean(B)


def test_var():
    """
    Calls the function `var` on both `onp.array` and `jnp.array`
    to check it doesn't raise an error
    """
    A = onp.array([1, 2, 3])
    B = jnp.array([1, 2, 3])

    jp.var(A)
    jp.var(B)


def test_arange():
    """
    Calls the `arange` function both inside and outside a jitted 
    function to check the type of the created array
    """
    def fixed_arange():
        return jnp.arange(0, 10)

    jit_arange = jit(fixed_arange)
    
    A = jp.arange(0, 10)
    B = jit_arange().block_until_ready()
    C = jp.arange(0, 10, dtype=int)
    D = jp.arange(0, 10, dtype=jnp.int32)

    assert isinstance(A, onp.ndarray)
    assert isinstance(B, jnp.ndarray)
    assert isinstance(C, onp.ndarray)
    assert isinstance(D, jnp.ndarray)


def test_dot():
    """
    Calls the function `dot` on both `onp.array` and `jnp.array`
    and checks the respective return types
    """
    A = onp.arange(9).reshape((3, 3))
    B = jnp.arange(9).reshape((3, 3))
        
    _check_types(A, jp.dot(A, A))
    _check_types(B, jp.dot(B, B))
    _check_types_multiple((A,B), jp.dot(A, B))


def test_outer():
    """
    Calls the function `outer` on both `onp.array` and `jnp.array`
    and checks the respective return types
    """
    A = onp.arange(9).reshape((3, 3))
    B = jnp.arange(9).reshape((3, 3))
        
    _check_types(A, jp.outer(A, A))
    _check_types(B, jp.outer(B, B))
    _check_types_multiple((A,B), jp.outer(A, B))


def test_matmul():
    """
    Calls the function `matmul` on both `onp.array` and `jnp.array`
    and checks the respective return types
    """
    A = onp.arange(9).reshape((3, 3))
    B = jnp.arange(9).reshape((3, 3))
        
    _check_types(A, jp.matmul(A, A))
    _check_types(B, jp.matmul(B, B))
    _check_types_multiple((A,B), jp.matmul(A, B))


def test_inv():
    """
    Calls the function `inv` on both `onp.array` and `jnp.array`
    and checks the respective return types
    """
    A = onp.arange(4).reshape((2, 2))
    B = jnp.arange(4).reshape((2, 2))

    _check_types(A, jp.inv(A))
    _check_types(B, jp.inv(B))


def test_square():
    """
    Calls the function `square` on both `onp.array` and `jnp.array`
    and checks the respective return types
    """
    A = onp.array([1, 2, 3])
    B = jnp.array([1, 2, 3])

    _check_types(A, jp.square(A))
    _check_types(B, jp.square(B))


def test_tile():
    """
    Calls the function `tile` on both `onp.array` and `jnp.array`
    and checks the respective return types
    """
    A = onp.array([1, 2, 3])
    B = jnp.array([1, 2, 3])

    _check_types(A, jp.tile(A, 3))
    _check_types(B, jp.tile(B, 3))


def test_repeat():
    """
    Calls the function `repeat` on both `onp.array` and `jnp.array`
    and checks the respective return types
    """
    A = onp.array([1, 2, 3])
    B = jnp.array([1, 2, 3])

    _check_types(A, jp.repeat(A, 3))
    _check_types(B, jp.repeat(B, 3))


def test_floor():
    """
    Calls the function `floor` on both `onp.array` and `jnp.array`
    and checks the respective return types
    """
    A = onp.array([1, 2, 3])
    B = jnp.array([1, 2, 3])

    _check_types(A, jp.tile(A, 3))
    _check_types(B, jp.tile(B, 3))


def test_cross():
    """
    Calls the function `cross` on both `onp.array` and `jnp.array`
    and checks the respective return types
    """
    A = onp.arange(3)
    B = jnp.arange(3)
        
    _check_types(A, jp.cross(A, A))
    _check_types(B, jp.cross(B, B))
    _check_types_multiple((A,B), jp.cross(A, B))


def test_sin():
    """
    Calls the function `sin` on both `onp.array` and `jnp.array`
    and checks the respective return types
    """
    A = onp.array([1, 2, 3])
    B = jnp.array([1, 2, 3])

    _check_types(A, jp.sin(A))
    _check_types(B, jp.sin(B))


def test_cos():
    """
    Calls the function `cos` on both `onp.array` and `jnp.array`
    and checks the respective return types
    """
    A = onp.array([1, 2, 3])
    B = jnp.array([1, 2, 3])

    _check_types(A, jp.cos(A))
    _check_types(B, jp.cos(B))


def test_arctan2():
    """
    Calls the function `arctan2` on both `onp.array` and `jnp.array`
    and checks the respective return types
    """
    A = onp.array([1, 2, 3])
    B = jnp.array([1, 2, 3])
        
    _check_types(A, jp.arctan2(A, A))
    _check_types(B, jp.arctan2(B, B))
    _check_types_multiple((A,B), jp.arctan2(A, B))


def test_arccos():
    """
    Calls the function `arccos` on both `onp.array` and `jnp.array`
    and checks the respective return types
    """
    A = onp.array([-1, 0, 1])
    B = jnp.array([-1, 0, 1])

    _check_types(A, jp.arccos(A))
    _check_types(B, jp.arccos(B))


def test_safe_arccos():
    """
    Calls the function `safe_arccos` on both `onp.array` and `jnp.array`
    and checks the respective return types
    """
    A = onp.array([-1, 0, 1])
    B = jnp.array([-1, 0, 1])

    _check_types(A, jp.safe_arccos(A))
    _check_types(B, jp.safe_arccos(B))


def test_arcsin():
    """
    Calls the function `arcsin` on both `onp.array` and `jnp.array`
    and checks the respective return types
    """
    A = onp.array([-1, 0, 1])
    B = jnp.array([-1, 0, 1])

    _check_types(A, jp.arcsin(A))
    _check_types(B, jp.arcsin(B))


def test_logical_not():
    """
    Calls the function `logical_not` on both `onp.array` and `jnp.array`
    and checks the respective return types
    """
    A = onp.array([1, 2, 3])
    B = jnp.array([1, 2, 3])

    _check_types(A, jp.logical_not(A))
    _check_types(B, jp.logical_not(B))


def test_logical_and():
    """
    Calls the function `logical_and` on both `onp.array` and `jnp.array`
    and checks the respective return types
    """
    A = onp.arange(9)
    B = jnp.arange(9)
        
    _check_types(A, jp.logical_and(A, A))
    _check_types(B, jp.logical_and(B, B))
    _check_types_multiple((A,B), jp.logical_and(A, B))


def test_multiply():
    """
    Calls the function `multiply` on both `onp.array` and `jnp.array`
    and checks the respective return types
    """
    A = onp.arange(9)
    B = jnp.arange(9)
        
    _check_types(A, jp.multiply(A, A))
    _check_types(B, jp.multiply(B, B))
    _check_types_multiple((A,B), jp.multiply(A, B))


def test_minimum():
    """
    Calls the function `minimum` on both `onp.array` and `jnp.array`
    and checks the respective return types
    """
    A = onp.arange(9)
    B = jnp.arange(9)
        
    _check_types(A, jp.minimum(A, A))
    _check_types(B, jp.minimum(B, B))
    _check_types_multiple((A,B), jp.minimum(A, B))


def test_amin():
    """
    Calls the function `amin` on both `onp.array` and `jnp.array`
    to check it doesn't raise an error
    """
    A = onp.array([1, 2, 3])
    B = jnp.array([1, 2, 3])

    jp.amin(A)
    jp.amin(B)


def test_amax():
    """
    Calls the function `amax` on both `onp.array` and `jnp.array`
    to check it doesn't raise an error
    """
    A = onp.array([1, 2, 3])
    B = jnp.array([1, 2, 3])

    jp.amax(A)
    jp.amax(B)


def test_exp():
    """
    Calls the function `exp` on both `onp.array` and `jnp.array`
    and checks the respective return types
    """
    A = onp.array([1, 2, 3])
    B = jnp.array([1, 2, 3])

    _check_types(A, jp.exp(A))
    _check_types(B, jp.exp(B))


def test_sign():
    """
    Calls the function `sign` on both `onp.array` and `jnp.array`
    and checks the respective return types
    """
    A = onp.array([1, 2, 3])
    B = jnp.array([1, 2, 3])

    _check_types(A, jp.sign(A))
    _check_types(B, jp.sign(B))


def test_sum():
    """
    Calls the function `sum` on both `onp.array` and `jnp.array`
    to check it doesn't raise an error
    """
    A = onp.array([1, 2, 3])
    B = jnp.array([1, 2, 3])

    jp.sum(A, axis=0)
    jp.sum(A)
    jp.sum(B, axis=0)
    jp.sum(B)


def test_stack():
    """
    Calls the function `stack` on both `onp.array` and `jnp.array`
    and checks the respective return types
    """
    A = onp.array([1, 2, 3])
    B = jnp.array([1, 2, 3])
        
    _check_types(A, jp.stack([A, A]))
    _check_types(B, jp.stack([B, B]))
    _check_types_multiple((A,B), jp.stack([A, B]))


def test_concatenate():
    """
    Calls the function `concatenate` on both `onp.array` and 
    `jnp.array` and checks the respective return types
    """
    A = onp.array([1, 2, 3])
    B = jnp.array([1, 2, 3])
        
    _check_types(A, jp.concatenate([A, A]))
    _check_types(B, jp.concatenate([B, B]))
    _check_types_multiple((A,B), jp.concatenate([A, B]))


def test_sqrt():
    """
    Calls the function `sqrt` on both `onp.array` and `jnp.array`
    and checks the respective return types
    """
    A = onp.array([1, 2, 3])
    B = jnp.array([1, 2, 3])

    _check_types(A, jp.sqrt(A))
    _check_types(B, jp.sqrt(B))


def test_where():
    """
    Calls the function `where` on both `onp.array` and `jnp.array`
    and checks the respective return types
    """
    A = onp.array([1, 2, 3])
    B = onp.ones(3)
    C = jnp.array([1, 2, 3])
    D = jnp.ones(3)
        
    _check_types_multiple((A, B), jp.where(B, A, A))
    _check_types_multiple((C, D), jp.where(D, C, C))
    _check_types_multiple((A,B, C), jp.where(B, A, C))


def test_diag():
    """
    Calls the function `diag` on both `onp.array` and `jnp.array`
    and checks the respective return types
    """
    A = onp.arange(9).reshape((3,3))
    B = jnp.arange(9).reshape((3,3))

    _check_types(A, jp.diag(A))
    _check_types(B, jp.diag(B))


def test_clip():
    """
    Calls the function `clip` on both `onp.array` and `jnp.array`
    and checks the respective return types
    """
    A = onp.array([1, 2, 3])
    B = onp.ones(3)
    C = jnp.array([1, 2, 3])
    D = jnp.ones(3)
        
    _check_types_multiple((A, B), jp.where(B, A, A))
    _check_types_multiple((C, D), jp.where(D, C, C))
    _check_types_multiple((A,B, C), jp.where(B, A, C))


def test_eye():
    """
    Calls the function `eye` to check it doesn't raise an error
    """
    A = jp.eye(3)
    B = jp.eye(3, dtype=int)
    C = jp.eye(3, dtype=jnp.float32)
    D = jp.eye(3, dtype=jnp.int32)

    isinstance(A, onp.ndarray)
    isinstance(B, onp.ndarray)
    isinstance(C, jnp.ndarray)
    isinstance(D, jnp.ndarray)


def test_zeros():
    """
    Calls the function `zeros` to check it doesn't raise an error
    """
    A = jp.zeros((3,3,5))
    B = jp.zeros((3,3,5), dtype=int)
    C = jp.zeros((3,3,5), dtype=jnp.float32)
    D = jp.zeros((3,3,5), dtype=jnp.int32)

    isinstance(A, onp.ndarray)
    isinstance(B, onp.ndarray)
    isinstance(C, jnp.ndarray)
    isinstance(D, jnp.ndarray)


def test_zeros_like():
    """
    Calls the function `zeros_like` on both `onp.array` and 
    `jnp.array` and checks the respective return types
    """
    A = onp.array([1, 2, 3])
    B = jnp.array([1, 2, 3])

    _check_types(A, jp.zeros_like(A))
    _check_types(B, jp.zeros_like(B))


def test_ones():
    """
    Calls the function `ones` to check it doesn't raise an error
    """
    A = jp.ones((3,3,5))
    B = jp.ones((3,3,5), dtype=int)
    C = jp.ones((3,3,5), dtype=jnp.float32)
    D = jp.ones((3,3,5), dtype=jnp.int32)

    isinstance(A, onp.ndarray)
    isinstance(B, onp.ndarray)
    isinstance(C, jnp.ndarray)
    isinstance(D, jnp.ndarray)


def test_ones_like():
    """
    Calls the function `ones_like` on both `onp.array` and 
    `jnp.array` and checks the respective return types
    """
    A = onp.array([1, 2, 3])
    B = jnp.array([1, 2, 3])

    _check_types(A, jp.ones_like(A))
    _check_types(B, jp.ones_like(B))


def test_reshape():
    """
    Calls the function `reshape` on both `onp.array` and `jnp.array`
    and checks the respective return types
    """
    A = onp.arange(9)
    B = jnp.arange(9)

    _check_types(A, jp.reshape(A, (3,3)))
    _check_types(B, jp.reshape(B, (3,3)))


def test_array():
    """
    Tests whether array creation runs without errors
    """
    A = jp.array([])
    B = jp.array([1, 2, 3])
    C = jp.array([1, 2, 3], dtype=float)
    D = jp.array([[1, 2], [3, 4]])
    E = jp.array([1, 2, 3], jnp.int32)

    isinstance(A, onp.ndarray)
    isinstance(B, onp.ndarray)
    isinstance(C, onp.ndarray)
    isinstance(D, onp.ndarray)
    isinstance(E, jnp.ndarray)


def test_abs():
    """
    Calls the function `abs` on both `onp.array` and `jnp.array`
    and checks the respective return types
    """
    A = onp.array([1, 2, 3])
    B = jnp.array([1, 2, 3])

    _check_types(A, jp.abs(A))
    _check_types(B, jp.abs(B))
