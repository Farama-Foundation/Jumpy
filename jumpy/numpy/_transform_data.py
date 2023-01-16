"""Functions for manipulating data."""

from __future__ import annotations

from typing import Any, Sequence

try:
    import jax
except ImportError:
    jax = None

from jumpy import is_jax_installed
from jumpy.core import custom_jvp, which_np
from jumpy.numpy._types import ndarray


def abs(a: ndarray) -> ndarray:
    """Calculate the absolute value element-wise."""
    return which_np(a).abs(a)


def all(a: ndarray, axis: int | None = None) -> ndarray:
    """Test whether all array elements along a given axis evaluate to True."""
    return which_np(a).all(a, axis=axis)


def amax(x: ndarray, *args, **kwargs) -> ndarray:
    """Returns the maximum along a given axis."""
    return which_np(x).amax(x, *args, **kwargs)


def amin(x: ndarray, *args, **kwargs) -> ndarray:
    """Returns the minimum along a given axis."""
    return which_np(x).amin(x, *args, **kwargs)


def any(a: ndarray, axis: int | None = None) -> ndarray:
    """Test whether any array element along a given axis evaluates to True."""
    return which_np(a).any(a, axis=axis)


def arccos(x: ndarray) -> ndarray:
    """Trigonometric inverse cosine, element-wise."""
    return which_np(x).arccos(x)


@custom_jvp
def safe_arccos(x: ndarray) -> ndarray:
    """Trigonometric inverse cosine, element-wise with safety clipping in grad."""
    return which_np(x).arccos(x)


@safe_arccos.defjvp
def _safe_arccos_jvp(primal, tangent):
    (x,) = primal
    (x_dot,) = tangent
    primal_out = safe_arccos(x)
    tangent_out = -x_dot / sqrt(1.0 - clip(x, -1 + 1e-7, 1 - 1e-7) ** 2.0)
    return primal_out, tangent_out


def arcsin(x: ndarray) -> ndarray:
    """Trigonometric inverse sine, element-wise."""
    return which_np(x).arcsin(x)


@custom_jvp
def safe_arcsin(x: ndarray) -> ndarray:
    """Trigonometric inverse sine, element-wise with safety clipping in grad."""
    return which_np(x).arcsin(x)


@safe_arcsin.defjvp
def _safe_arcsin_jvp(primal, tangent):
    (x,) = primal
    (x_dot,) = tangent
    primal_out = safe_arccos(x)
    tangent_out = x_dot / sqrt(1.0 - clip(x, -1 + 1e-7, 1 - 1e-7) ** 2.0)
    return primal_out, tangent_out


def arctan2(x1: ndarray, x2: ndarray) -> ndarray:
    """Returns element-wise arc tangent of x1/x2 choosing the quadrant correctly."""
    return which_np(x1, x2).arctan2(x1, x2)


def arctanh(x: ndarray) -> ndarray:
    """Returns element-wise arctanh of x."""
    return which_np(x).arctanh(x)


def argmax(x: ndarray, *args, **kwargs) -> ndarray:
    """Returns the argmax along a given axis."""
    return which_np(x).argmax(x, *args, **kwargs)


def argmin(x: ndarray, *args, **kwargs) -> ndarray:
    """Returns the argmin along a given axis."""
    return which_np(x).argmin(x, *args, **kwargs)


def clip(a: ndarray, a_min: ndarray, a_max: ndarray) -> ndarray:
    """Clip (limit) the values in an array."""
    return which_np(a, a_min, a_max).clip(a, a_min, a_max)


def concatenate(x: Sequence[ndarray], axis=0) -> ndarray:
    """Join a sequence of arrays along an existing axis."""
    return which_np(*x).concatenate(x, axis=axis)


def cos(angle: ndarray) -> ndarray:
    """Returns trigonometric cosine, element-wise."""
    return which_np(angle).cos(angle)


def cross(x: ndarray, y: ndarray) -> ndarray:
    """Returns cross product of two arrays."""
    return which_np(x, y).cross(x, y)


def diag(v: ndarray, k: int = 0) -> ndarray:
    """Extract a diagonal or construct a diagonal array."""
    return which_np(v).diag(v, k)


def dot(x: ndarray, y: ndarray) -> ndarray:
    """Returns dot product of two arrays."""
    return which_np(x, y).dot(x, y)


def exp(x: ndarray) -> ndarray:
    """Returns the exponential of all elements in the input array."""
    return which_np(x).exp(x)


def expand_dims(x: ndarray, axis: tuple[int, ...] | int = 0) -> ndarray:
    """Increases batch dimensionality along axis."""
    return which_np(x).expand_dims(x, axis=axis)


def floor(x: ndarray) -> ndarray:
    """Returns the floor of the input, element-wise.."""
    return which_np(x).floor(x)


def logical_and(x1: ndarray, x2: ndarray) -> ndarray:
    """Returns the truth value of x1 AND x2 element-wise."""
    return which_np(x1, x2).logical_and(x1, x2)


def logical_not(x: ndarray) -> ndarray:
    """Returns the truth value of NOT x element-wise."""
    return which_np(x).logical_not(x)


def logical_or(x1: ndarray, x2: ndarray) -> ndarray:
    """Returns the truth value of x1 OR x2 element-wise."""
    return which_np(x1, x2).logical_or(x1, x2)


def matmul(x1: ndarray, x2: ndarray) -> ndarray:
    """Matrix product of two arrays."""
    return which_np(x1, x2).matmul(x1, x2)


def maximum(x1: ndarray, x2: ndarray) -> ndarray:
    """Element-wise maximum of array elements."""
    return which_np(x1, x2).maximum(x1, x2)


def mean(a: ndarray, axis: int | None = None) -> ndarray:
    """Compute the arithmetic mean along the specified axis."""
    return which_np(a).mean(a, axis=axis)


def minimum(x1: ndarray, x2: ndarray) -> ndarray:
    """Element-wise minimum of array elements."""
    return which_np(x1, x2).minimum(x1, x2)


def multiply(x1: ndarray, x2: ndarray) -> ndarray:
    """Multiply arguments element-wise."""
    return which_np(x1, x2).multiply(x1, x2)


def outer(a: ndarray, b: ndarray) -> ndarray:
    """Compute the outer product of two vectors."""
    return which_np(a, b).outer(a, b)


def repeat(a: ndarray, repeats: int | ndarray, *args, **kwargs) -> ndarray:
    """Repeat elements of an array."""
    return which_np(a, repeats).repeat(a, repeats=repeats, *args, **kwargs)


def roll(x: ndarray, shift, axis=None) -> ndarray:
    """Rolls array elements along a given axis."""
    return which_np(x).roll(x, shift, axis=axis)


def sign(x: ndarray) -> ndarray:
    """Returns an element-wise indication of the sign of a number."""
    return which_np(x).sign(x)


def sin(angle: ndarray) -> ndarray:
    """Returns trigonometric sine, element-wise."""
    return which_np(angle).sin(angle)


def sqrt(x: ndarray) -> ndarray:
    """Returns the non-negative square-root of an array, element-wise."""
    return which_np(x).sqrt(x)


def square(x: ndarray) -> ndarray:
    """Return the element-wise square of the input."""
    return which_np(x).square(x)


def stack(x: list[ndarray], axis=0) -> ndarray:
    """Join a sequence of arrays along a new axis."""
    return which_np(*x).stack(x, axis=axis)


def sum(a: ndarray, axis: int | None = None):
    """Returns sum of array elements over a given axis."""
    return which_np(a).sum(a, axis=axis)


def take(tree: Any, i: ndarray | Sequence[int] | int, axis: int = 0) -> Any:
    """Returns tree sliced by i."""
    if not is_jax_installed:
        raise NotImplementedError("This function requires the jax module")

    np = which_np(i)
    if isinstance(i, (list, tuple)):
        i = np.array(i, dtype=int)
    return jax.tree_util.tree_map(lambda x: np.take(x, i, axis=axis, mode="clip"), tree)


def tanh(x: ndarray) -> ndarray:
    """Returns element-wise tanh of x."""
    return which_np(x).tanh(x)


def tile(x: ndarray, reps: tuple[int, ...] | int) -> ndarray:
    """Construct an array by repeating `x` the number of times given by reps."""
    return which_np(x).tile(x, reps)


def var(a: ndarray, axis: int | None = None) -> ndarray:
    """Compute the variance along the specified axis."""
    return which_np(a).var(a, axis=axis)


def where(condition: ndarray, x: ndarray, y: ndarray) -> ndarray:
    """Return elements chosen from `x` or `y` depending on `condition`."""
    return which_np(condition, x, y).where(condition, x, y)
