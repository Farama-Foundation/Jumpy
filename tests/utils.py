"""Utility functions for Jumpy."""

import jax.numpy as jnp
import numpy as onp


def check_types_multiple(in_args, out_arg):
    """Checks whether the output argument is either a `jnp.ndarray` when a `jnp.ndarray` is given in the input list, or a `onp.ndarray`."""
    assert (
        isinstance(out_arg, jnp.ndarray)
        if any([isinstance(a, jnp.ndarray) for a in in_args])
        else isinstance(out_arg, onp.ndarray)
    )


def check_types(in_arg, out_arg):
    """Checks whether the output argument is either a `jnp.ndarray` when a `jnp.ndarray` is given in the single input, or a `onp.ndarray`."""
    check_types_multiple([in_arg], out_arg)
