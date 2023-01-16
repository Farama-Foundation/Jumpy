"""Primitive data types in numpy."""

from typing import Union

import numpy as onp

import jumpy

# Determine ndarray
if jumpy.is_jax_installed:
    import jax.numpy as jnp

    ndarray = Union[onp.ndarray, jnp.ndarray]
else:
    ndarray = onp.ndarray

# Types
dtype = onp.dtype
float32 = onp.float32
int32 = onp.int32
uint8 = onp.uint8

# Special values
pi = onp.pi
inf = onp.inf
