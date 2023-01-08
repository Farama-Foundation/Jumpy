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

Carry = TypeVar("Carry")
X = TypeVar("X")
Y = TypeVar("Y")


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


def cond(
    pred, true_fun: Callable[..., bool], false_fun: Callable[..., bool], *operands: Any
):
    """Conditionally apply true_fun or false_fun to operands."""
    if is_jitted():
        return jax.lax.cond(pred, true_fun, false_fun, *operands)
    else:
        if pred:
            return true_fun(operands)
        else:
            return false_fun(operands)


def fori_loop(lower: int, upper: int, body_fun: Callable[[X], X], init_val: X) -> X:
    """Call body_fun over range from lower to upper, starting with init_val."""
    if is_jitted():
        return jax.lax.fori_loop(lower, upper, body_fun, init_val)
    else:
        val = init_val
        for _ in range(lower, upper):
            val = body_fun(val)
        return val


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


def scan(
    f: Callable[[Carry, X], tuple[Carry, Y]],
    init: Carry,
    xs: X,
    length: int | None = None,
    reverse: bool = False,
    unroll: int = 1,
) -> tuple[Carry, Y]:
    """Scan a function over leading array axes while carrying along state."""
    if not is_jax_installed:
        raise NotImplementedError("This function requires the jax module")

    if is_jitted():
        return jax.lax.scan(f, init, xs, length, reverse, unroll)
    else:
        xs_flat, xs_tree = jax.tree_util.tree_flatten(xs)
        carry = init
        ys = []
        maybe_reversed = reversed if reverse else lambda x: x
        for i in maybe_reversed(range(length)):
            xs_slice = [x[i] for x in xs_flat]
            carry, y = f(carry, jax.tree_util.tree_unflatten(xs_tree, xs_slice))
            ys.append(y)
        stacked_y = jax.tree_util.tree_map(lambda *y: onp.stack(y), *maybe_reversed(ys))
        return carry, stacked_y


def segment_sum(
    data: ndarray, segment_ids: ndarray, num_segments: int | None = None
) -> ndarray:
    """Computes the sum within segments of an array."""
    if which_np(data, segment_ids) is jnp:
        s = jax.ops.segment_sum(data, segment_ids, num_segments)
    else:
        if num_segments is None:
            num_segments = onp.amax(segment_ids) + 1
        s = onp.zeros((num_segments,) + data.shape[1:])
        onp.add.at(s, segment_ids, data)
    return s


def take(tree: Any, i: ndarray | Sequence[int] | int, axis: int = 0) -> Any:
    """Returns tree sliced by i."""
    if not is_jax_installed:
        raise NotImplementedError("This function requires the jax module")

    np = which_np(i)
    if isinstance(i, (list, tuple)):
        i = np.array(i, dtype=int)
    return jax.tree_util.tree_map(lambda x: np.take(x, i, axis=axis, mode="clip"), tree)


def top_k(operand: ndarray, k: int) -> tuple[ndarray, ndarray]:
    """Returns top k values and their indices along the last axis of operand."""
    if which_np(operand) is jnp:
        return jax.lax.top_k(operand, k)
    else:
        top_ind = onp.argpartition(operand, -k)[-k:]
        sorted_ind = top_ind[onp.argsort(-operand[top_ind])]
        return operand[sorted_ind], sorted_ind


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


def while_loop(
    cond_fun: Callable[[X], Any], body_fun: Callable[[X], X], init_val: X
) -> X:
    """Call body_fun while cond_fun is true, starting with init_val."""
    if is_jitted():
        return jax.lax.while_loop(cond_fun, body_fun, init_val)
    else:
        val = init_val
        while cond_fun(val):
            val = body_fun(val)
        return val
