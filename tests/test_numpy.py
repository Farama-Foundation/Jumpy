"""Tests the jumpy numpy functions."""

import jax.numpy as jnp
import numpy as onp
import pytest
from jax import jit

import jumpy.numpy as jp
from tests.utils import check_types, check_types_multiple


def test_any():
    """Calls the function `norm` on both `onp.array` and `jnp.array` to check it doesn't raise an error."""
    a = onp.array([False, True, False])
    b = jnp.array([False, True, False])

    jp.any(a)
    jp.any(b)


def test_all():
    """Calls the function `all` on both `onp.array` and `jnp.array` to check it doesn't raise an error."""
    a = onp.array([True, True])
    b = jnp.array([True, True])

    jp.all(a)
    jp.all(b)


def test_mean():
    """Calls the function `mean` on both `onp.array` and `jnp.array` to check it doesn't raise an error."""
    a = onp.array([1, 2, 3])
    b = jnp.array([1, 2, 3])

    jp.mean(a)
    jp.mean(b)


def test_var():
    """Calls the function `var` on both `onp.array` and `jnp.array` to check it doesn't raise an error."""
    a = onp.array([1, 2, 3])
    b = jnp.array([1, 2, 3])

    jp.var(a)
    jp.var(b)


def test_arange():
    """Calls the `arange` function both inside and outside a jitted function to check the type of the created array."""

    def fixed_arange():
        return jnp.arange(0, 10)

    jit_arange = jit(fixed_arange)

    a = jp.arange(0, 10)
    b = jit_arange().block_until_ready()
    c = jp.arange(0, 10, dtype=int)
    d = jp.arange(0, 10, dtype=jnp.int32)

    assert isinstance(a, onp.ndarray)
    assert isinstance(b, jnp.ndarray)
    assert isinstance(c, onp.ndarray)
    assert isinstance(d, jnp.ndarray)


def test_dot():
    """Calls the function `dot` on both `onp.array` and `jnp.array` and checks the respective return types."""
    a = onp.arange(9).reshape((3, 3))
    b = jnp.arange(9).reshape((3, 3))

    check_types(a, jp.dot(a, a))
    check_types(b, jp.dot(b, b))
    check_types_multiple((a, b), jp.dot(a, b))


def test_outer():
    """Calls the function `outer` on both `onp.array` and `jnp.array` and checks the respective return types."""
    a = onp.arange(9).reshape((3, 3))
    b = jnp.arange(9).reshape((3, 3))

    check_types(a, jp.outer(a, a))
    check_types(b, jp.outer(b, b))
    check_types_multiple((a, b), jp.outer(a, b))


def test_matmul():
    """Calls the function `matmul` on both `onp.array` and `jnp.array` and checks the respective return types."""
    a = onp.arange(9).reshape((3, 3))
    b = jnp.arange(9).reshape((3, 3))

    check_types(a, jp.matmul(a, a))
    check_types(b, jp.matmul(b, b))
    check_types_multiple((a, b), jp.matmul(a, b))


def test_inv():
    """Calls the function `inv` on both `onp.array` and `jnp.array` and checks the respective return types."""
    a = onp.arange(4).reshape((2, 2))
    b = jnp.arange(4).reshape((2, 2))

    check_types(a, jp.linalg.inv(a))
    check_types(b, jp.linalg.inv(b))


def test_square():
    """Calls the function `square` on both `onp.array` and `jnp.array` and checks the respective return types."""
    a = onp.array([1, 2, 3])
    b = jnp.array([1, 2, 3])

    check_types(a, jp.square(a))
    check_types(b, jp.square(b))


def test_tile():
    """Calls the function `tile` on both `onp.array` and `jnp.array` and checks the respective return types."""
    a = onp.array([1, 2, 3])
    b = jnp.array([1, 2, 3])

    check_types(a, jp.tile(a, 3))
    check_types(b, jp.tile(b, 3))


def test_repeat():
    """Calls the function `repeat` on both `onp.array` and `jnp.array` and checks the respective return types."""
    a = onp.array([1, 2, 3])
    b = jnp.array([1, 2, 3])

    check_types(a, jp.repeat(a, 3))
    check_types(b, jp.repeat(b, 3))


def test_floor():
    """Calls the function `floor` on both `onp.array` and `jnp.array` and checks the respective return types."""
    a = onp.array([1, 2, 3])
    b = jnp.array([1, 2, 3])

    check_types(a, jp.tile(a, 3))
    check_types(b, jp.tile(b, 3))


def test_cross():
    """Calls the function `cross` on both `onp.array` and `jnp.array` and checks the respective return types."""
    a = onp.arange(3)
    b = jnp.arange(3)

    check_types(a, jp.cross(a, a))
    check_types(b, jp.cross(b, b))
    check_types_multiple((a, b), jp.cross(a, b))


def test_sin():
    """Calls the function `sin` on both `onp.array` and `jnp.array` and checks the respective return types."""
    a = onp.array([1, 2, 3])
    b = jnp.array([1, 2, 3])

    check_types(a, jp.sin(a))
    check_types(b, jp.sin(b))


def test_cos():
    """Calls the function `cos` on both `onp.array` and `jnp.array` and checks the respective return types."""
    a = onp.array([1, 2, 3])
    b = jnp.array([1, 2, 3])

    check_types(a, jp.cos(a))
    check_types(b, jp.cos(b))


def test_arctan2():
    """Calls the function `arctan2` on both `onp.array` and `jnp.array` and checks the respective return types."""
    a = onp.array([1, 2, 3])
    b = jnp.array([1, 2, 3])

    check_types(a, jp.arctan2(a, a))
    check_types(b, jp.arctan2(b, b))
    check_types_multiple((a, b), jp.arctan2(a, b))


def test_arccos():
    """Calls the function `arccos` on both `onp.array` and `jnp.array` and checks the respective return types."""
    a = onp.array([-1, 0, 1])
    b = jnp.array([-1, 0, 1])

    check_types(a, jp.arccos(a))
    check_types(b, jp.arccos(b))


def test_safe_arccos():
    """Calls the function `safe_arccos` on both `onp.array` and `jnp.array` and checks the respective return types."""
    a = onp.array([-1, 0, 1])
    b = jnp.array([-1, 0, 1])

    check_types(a, jp.safe_arccos(a))
    check_types(b, jp.safe_arccos(b))


def test_arcsin():
    """Calls the function `arcsin` on both `onp.array` and `jnp.array` and checks the respective return types."""
    a = onp.array([-1, 0, 1])
    b = jnp.array([-1, 0, 1])

    check_types(a, jp.arcsin(a))
    check_types(b, jp.arcsin(b))


def test_logical_not():
    """Calls the function `logical_not` on both `onp.array` and `jnp.array` and checks the respective return types."""
    a = onp.array([1, 2, 3])
    b = jnp.array([1, 2, 3])

    check_types(a, jp.logical_not(a))
    check_types(b, jp.logical_not(b))


def test_logical_and():
    """Calls the function `logical_and` on both `onp.array` and `jnp.array` and checks the respective return types."""
    a = onp.arange(9)
    b = jnp.arange(9)

    check_types(a, jp.logical_and(a, a))
    check_types(b, jp.logical_and(b, b))
    check_types_multiple((a, b), jp.logical_and(a, b))


def test_multiply():
    """Calls the function `multiply` on both `onp.array` and `jnp.array` and checks the respective return types."""
    a = onp.arange(9)
    b = jnp.arange(9)

    check_types(a, jp.multiply(a, a))
    check_types(b, jp.multiply(b, b))
    check_types_multiple((a, b), jp.multiply(a, b))


def test_minimum():
    """Calls the function `minimum` on both `onp.array` and `jnp.array` and checks the respective return types."""
    a = onp.arange(9)
    b = jnp.arange(9)

    check_types(a, jp.minimum(a, a))
    check_types(b, jp.minimum(b, b))
    check_types_multiple((a, b), jp.minimum(a, b))


def test_amin():
    """Calls the function `amin` on both `onp.array` and `jnp.array` to check it doesn't raise an error."""
    a = onp.array([1, 2, 3])
    b = jnp.array([1, 2, 3])

    jp.amin(a)
    jp.amin(b)


def test_amax():
    """Calls the function `amax` on both `onp.array` and `jnp.array` to check it doesn't raise an error."""
    a = onp.array([1, 2, 3])
    b = jnp.array([1, 2, 3])

    jp.amax(a)
    jp.amax(b)


def test_exp():
    """Calls the function `exp` on both `onp.array` and `jnp.array` and checks the respective return types."""
    a = onp.array([1, 2, 3])
    b = jnp.array([1, 2, 3])

    check_types(a, jp.exp(a))
    check_types(b, jp.exp(b))


def test_sign():
    """Calls the function `sign` on both `onp.array` and `jnp.array` and checks the respective return types."""
    a = onp.array([1, 2, 3])
    b = jnp.array([1, 2, 3])

    check_types(a, jp.sign(a))
    check_types(b, jp.sign(b))


def test_sum():
    """Calls the function `sum` on both `onp.array` and `jnp.array` to check it doesn't raise an error."""
    a = onp.array([1, 2, 3])
    b = jnp.array([1, 2, 3])

    jp.sum(a, axis=0)
    jp.sum(a)
    jp.sum(b, axis=0)
    jp.sum(b)


def test_stack():
    """Calls the function `stack` on both `onp.array` and `jnp.array` and checks the respective return types."""
    a = onp.array([1, 2, 3])
    b = jnp.array([1, 2, 3])

    check_types(a, jp.stack([a, a]))
    check_types(b, jp.stack([b, b]))
    check_types_multiple((a, b), jp.stack([a, b]))


def test_concatenate():
    """Calls the function `concatenate` on both `onp.array` and `jnp.array` and checks the respective return types."""
    a = onp.array([1, 2, 3])
    b = jnp.array([1, 2, 3])

    check_types(a, jp.concatenate([a, a]))
    check_types(b, jp.concatenate([b, b]))
    check_types_multiple((a, b), jp.concatenate([a, b]))


def test_sqrt():
    """Calls the function `sqrt` on both `onp.array` and `jnp.array` and checks the respective return types."""
    a = onp.array([1, 2, 3])
    b = jnp.array([1, 2, 3])

    check_types(a, jp.sqrt(a))
    check_types(b, jp.sqrt(b))


def test_where():
    """Calls the function `where` on both `onp.array` and `jnp.array` and checks the respective return types."""
    a = onp.array([1, 2, 3])
    b = onp.ones(3)
    c = jnp.array([1, 2, 3])
    d = jnp.ones(3)

    check_types_multiple((a, b), jp.where(b, a, a))
    check_types_multiple((c, d), jp.where(d, c, c))
    check_types_multiple((a, b, c), jp.where(b, a, c))


def test_diag():
    """Calls the function `diag` on both `onp.array` and `jnp.array` and checks the respective return types."""
    a = onp.arange(9).reshape((3, 3))
    b = jnp.arange(9).reshape((3, 3))

    check_types(a, jp.diag(a))
    check_types(b, jp.diag(b))


def test_clip():
    """Calls the function `clip` on both `onp.array` and `jnp.array` and checks the respective return types."""
    a = onp.array([1, 2, 3])
    b = onp.ones(3)
    c = jnp.array([1, 2, 3])
    d = jnp.ones(3)

    check_types_multiple((a, b), jp.where(b, a, a))
    check_types_multiple((c, d), jp.where(d, c, c))
    check_types_multiple((a, b, c), jp.where(b, a, c))


def test_eye():
    """Calls the function `eye` to check it doesn't raise an error."""
    a = jp.eye(3)
    b = jp.eye(3, dtype=int)
    c = jp.eye(3, dtype=jnp.float32)
    d = jp.eye(3, dtype=jnp.int32)

    isinstance(a, onp.ndarray)
    isinstance(b, onp.ndarray)
    isinstance(c, jnp.ndarray)
    isinstance(d, jnp.ndarray)


def test_zeros():
    """Calls the function `zeros` to check it doesn't raise an error."""
    a = jp.zeros((3, 3, 5))
    b = jp.zeros((3, 3, 5), dtype=int)
    c = jp.zeros((3, 3, 5), dtype=jnp.float32)
    d = jp.zeros((3, 3, 5), dtype=jnp.int32)

    isinstance(a, onp.ndarray)
    isinstance(b, onp.ndarray)
    isinstance(c, jnp.ndarray)
    isinstance(d, jnp.ndarray)


def test_zeros_like():
    """Calls the function `zeros_like` on both `onp.array` and `jnp.array` and checks the respective return types."""
    a = onp.array([1, 2, 3])
    b = jnp.array([1, 2, 3])

    check_types(a, jp.zeros_like(a))
    check_types(b, jp.zeros_like(b))


def test_ones():
    """Calls the function `ones` to check it doesn't raise an error."""
    a = jp.ones((3, 3, 5))
    b = jp.ones((3, 3, 5), dtype=int)
    c = jp.ones((3, 3, 5), dtype=jnp.float32)
    d = jp.ones((3, 3, 5), dtype=jnp.int32)

    isinstance(a, onp.ndarray)
    isinstance(b, onp.ndarray)
    isinstance(c, jnp.ndarray)
    isinstance(d, jnp.ndarray)


def test_ones_like():
    """Calls the function `ones_like` on both `onp.array` and `jnp.array` and checks the respective return types."""
    a = onp.array([1, 2, 3])
    b = jnp.array([1, 2, 3])

    check_types(a, jp.ones_like(a))
    check_types(b, jp.ones_like(b))


def test_reshape():
    """Calls the function `reshape` on both `onp.array` and `jnp.array` and checks the respective return types."""
    a = onp.arange(9)
    b = jnp.arange(9)

    check_types(a, jp.reshape(a, (3, 3)))
    check_types(b, jp.reshape(b, (3, 3)))


def test_array():
    """Tests whether array creation runs without errors."""
    a = jp.array([])
    b = jp.array([1, 2, 3])
    c = jp.array([1, 2, 3], dtype=float)
    d = jp.array([[1, 2], [3, 4]])
    e = jp.array([1, 2, 3], jnp.int32)

    isinstance(a, onp.ndarray)
    isinstance(b, onp.ndarray)
    isinstance(c, onp.ndarray)
    isinstance(d, onp.ndarray)
    isinstance(e, jnp.ndarray)


def test_abs():
    """Calls the function `abs` on both `onp.array` and `jnp.array` and checks the respective return types."""
    a = onp.array([1, 2, 3])
    b = jnp.array([1, 2, 3])

    check_types(a, jp.abs(a))
    check_types(b, jp.abs(b))


@pytest.mark.parametrize(
    "array, indexes, ret",
    [
        (jp.arange(0, 5), [1, 0, 2, 3], jp.array([1, 0, 2, 3])),
        # Empty array
        (jp.array([]), [], []),
        # Jax array
        (jp.arange(0, 5, dtype=jnp.float32), [1, 0, 2, 3], jp.array([1, 0, 2, 3])),
    ],
)
def test_take(array, indexes, ret):
    """Checks if the result of `take` on the given inputs equals the expected output."""
    x = jp.take(array, indexes)
    assert onp.array_equal(x, ret)
