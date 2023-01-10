"""Jumpy module."""

from typing import Union

import numpy as onp

dtype = onp.dtype
pi = onp.pi
inf = onp.inf
float32 = onp.float32
int32 = onp.int32

try:
    import jax.numpy as jnp

    ndarray = Union[onp.ndarray, jnp.ndarray]

    is_jax_installed = True
except ImportError:
    ndarray = onp.ndarray

    is_jax_installed = False


from jumpy import lax, linalg, ops, random
from jumpy._custom_fns import array, index_update, meshgrid, take, vmap
from jumpy._factory_fns import (
    arange,
    atleast_1d,
    atleast_2d,
    atleast_3d,
    eye,
    linspace,
    ones,
    ones_like,
    reshape,
    zeros,
    zeros_like,
)
from jumpy._transform_data import (
    abs,
    all,
    amax,
    amin,
    any,
    arccos,
    arcsin,
    arctan2,
    arctanh,
    argmax,
    argmin,
    clip,
    concatenate,
    cos,
    cross,
    diag,
    dot,
    exp,
    expand_dims,
    floor,
    logical_and,
    logical_not,
    logical_or,
    matmul,
    maximum,
    mean,
    minimum,
    multiply,
    outer,
    repeat,
    roll,
    safe_arccos,
    safe_arcsin,
    sign,
    sin,
    sqrt,
    square,
    stack,
    sum,
    tanh,
    tile,
    var,
    where,
)

__all__ = [
    # === primitive ===
    "dtype",
    "pi",
    "inf",
    "float32",
    "int32",
    "ndarray",
    # === sub-modules ===
    "random",
    "linalg",
    "lax",
    "ops",
    # "fft", not implemented yet
    # === create data functions ===
    "arange",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "eye",
    "linspace",
    "ones",
    "ones_like",
    "reshape",
    "zeros",
    "zeros_like",
    # === custom functions ==
    "array",
    "index_update",
    "meshgrid",
    "take",
    "vmap",
    # === manipulation data functions ===
    "abs",
    "all",
    "amax",
    "amin",
    "any",
    "arccos",
    "arcsin",
    "arctan2",
    "arctanh",
    "argmax",
    "argmin",
    "clip",
    "concatenate",
    "cos",
    "cross",
    "diag",
    "dot",
    "exp",
    "expand_dims",
    "floor",
    "logical_and",
    "logical_not",
    "logical_or",
    "matmul",
    "maximum",
    "mean",
    "minimum",
    "multiply",
    "outer",
    "repeat",
    "roll",
    "safe_arccos",
    "safe_arcsin",
    "sign",
    "sin",
    "sqrt",
    "square",
    "stack",
    "sum",
    "tanh",
    "tile",
    "var",
    "where",
]
