# pylint:disable=redefined-builtin


import jax.numpy as jnp
import numpy as onp

import jumpy as jp


def _check_types_multiple(in_args, out_arg):
    assert isinstance(out_arg, jnp.ndarray) if any(
        [isinstance(a, jnp.ndarray) for a in in_args]
    ) else isinstance(out_arg, onp.ndarray)


def _check_types(in_arg, out_arg):
    _check_types_multiple([in_arg], out_arg)


def test_dot():
    A = onp.arange(9).reshape((3, 3))
    B = jnp.arange(9).reshape((3, 3))
        
    _check_types(A, jp.dot(A, A))
    _check_types(B, jp.dot(B, B))
    _check_types_multiple((A,B), jp.dot(A, B))