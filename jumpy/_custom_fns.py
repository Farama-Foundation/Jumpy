"""Custom implemented functions."""

from __future__ import annotations

from typing import Any, Callable, Sequence, TypeVar

import numpy as onp

from jumpy import is_jax_installed, ndarray
from jumpy.core import is_jitted, which_dtype, which_np

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax, jnp = None, None


F = TypeVar("F", bound=Callable)


def array(object: Any, dtype=None) -> ndarray:
    """Creates an array given a list."""
    if dtype is None:
        try:
            np = which_np(*object)
        except TypeError:
            np = which_np(object)  # object is not iterable (e.g. primitive type)
    else:
        np = which_dtype(dtype)

    return np.array(object, dtype)


def index_update(x: ndarray, idx: ndarray, y: ndarray) -> ndarray:
    """Pure equivalent of x[idx] = y."""
    if which_np(x, idx, y) is jnp:
        return jnp.array(x).at[idx].set(jnp.array(y))
    else:
        x = onp.copy(x)
        x[idx] = y
        return x


def meshgrid(
    *xi, copy: bool = True, sparse: bool = False, indexing: str = "xy"
) -> ndarray:
    """Create N-D coordinate matrices from 1D coordinate vectors."""
    if which_np(xi[0]) is jnp:
        return jnp.meshgrid(*xi, copy=copy, sparse=sparse, indexing=indexing)
    else:
        return onp.meshgrid(*xi, copy=copy, sparse=sparse, indexing=indexing)


def take(tree: Any, i: ndarray | Sequence[int] | int, axis: int = 0) -> Any:
    """Returns tree sliced by i."""
    if not is_jax_installed:
        raise NotImplementedError("This function requires the jax module")

    np = which_np(i)
    if isinstance(i, (list, tuple)):
        i = np.array(i, dtype=int)
    return jax.tree_util.tree_map(lambda x: np.take(x, i, axis=axis, mode="clip"), tree)


def vmap(fun: F, include: Sequence[bool] | None = None) -> F:
    """Creates a function which maps ``fun`` over argument axes."""
    if not is_jax_installed:
        raise NotImplementedError("This function requires the jax module")

    if is_jitted():
        in_axes = 0
        if include:
            in_axes = [0 if inc else None for inc in include]
        return jax.vmap(fun, in_axes=in_axes)

    def _batched(*args, include=include):
        if include is not None and len(include) != len(args):
            raise RuntimeError("Len of `args` list must match length of `include`.")

        # by default, vectorize over every arg
        if include is None:
            include = [True for _ in args]

        # determine number of parallel evaluations to unroll into serial evals
        batch_size = None
        for a, inc in zip(args, include):
            if inc:
                flat_args, _ = jax.tree_util.tree_flatten(a)
                batch_size = flat_args[0].shape[0]
                break

        # rebuild b_args for each serial evaluation
        rets = []
        for b_idx in range(batch_size):
            b_args = []
            for a, inc in zip(args, include):
                if inc:
                    b_args.append(take(a, b_idx))
                else:
                    b_args.append(a)
            rets.append(fun(*b_args))

        return jax.tree_util.tree_map(lambda *x: onp.stack(x), *rets)

    return _batched
