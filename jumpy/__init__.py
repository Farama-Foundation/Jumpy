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


from jumpy._core import is_in_jit, which_dtype, which_np  # isort:skip
from jumpy import linalg, random
from jumpy._create_data import (
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
from jumpy._custom_funcs import (
    array,
    cond,
    fori_loop,
    index_update,
    meshgrid,
    scan,
    segment_sum,
    take,
    top_k,
    vmap,
    while_loop,
)
from jumpy._manipulate_data import (
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
    # "fft", not implemented yet
    # === core functions ===
    "is_in_jit",
    "which_np",
    "which_dtype",
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
    "cond",
    "fori_loop",
    "index_update",
    "meshgrid",
    "scan",
    "segment_sum",
    "take",
    "top_k",
    "vmap",
    "while_loop",
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
