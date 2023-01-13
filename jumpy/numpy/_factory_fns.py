"""Functions for creating data in numpy."""

from __future__ import annotations

from typing import Any

from jumpy.core import which_dtype, which_np
from jumpy.numpy._types import ndarray


def arange(start: int, stop: int, dtype=None) -> ndarray:
    """Return evenly spaced values within a given interval."""
    if dtype is None:
        return which_np(start, stop).arange(start, stop, dtype=dtype)
    else:
        return which_dtype(dtype).arange(start, stop, dtype=dtype)


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


def atleast_1d(*arys) -> ndarray:
    """Ensure arrays are all at least 1d (dimensions added to beginning)."""
    return which_np(*arys).atleast_1d(*arys)


def atleast_2d(*arys) -> ndarray:
    """Ensure arrays are all at least 2d (dimensions added to beginning)."""
    return which_np(*arys).atleast_2d(*arys)


def atleast_3d(*arys) -> ndarray:
    """Ensure arrays are all at least 3d (dimensions added to beginning)."""
    return which_np(*arys).atleast_3d(*arys)


def eye(n: int, dtype=float) -> ndarray:
    """Return a 2-D array with ones on the diagonal and zeros elsewhere."""
    return which_dtype(dtype).eye(n, dtype=dtype)


def linspace(start: ndarray, stop: ndarray, num: int, dtype=None) -> ndarray:
    """Return evenly spaced `num` values between `start` and `stop`."""
    if dtype is None:
        return which_np(start, stop, num).linspace(start, stop, num)
    else:
        return which_dtype(dtype).linspace(start, stop, num, dtype=dtype)


def meshgrid(
    *xi, copy: bool = True, sparse: bool = False, indexing: str = "xy"
) -> ndarray:
    """Create N-D coordinate matrices from 1D coordinate vectors."""
    return which_np(xi[0]).meshgrid(*xi, copy=copy, sparse=sparse, indexing=indexing)


def ones(shape, dtype=float) -> ndarray:
    """Return a new array of given shape and type, filled with ones."""
    return which_dtype(dtype).ones(shape, dtype=dtype)


def ones_like(a: ndarray) -> ndarray:
    """Return an array of ones with the same shape and type as a given array."""
    return which_np(a).ones_like(a)


def reshape(a: ndarray, newshape: tuple[int, ...] | int) -> ndarray:
    """Gives a new shape to an array without changing its data."""
    return which_np(a).reshape(a, newshape)


def zeros(shape, dtype=float) -> ndarray:
    """Return a new array of given shape and type, filled with zeros."""
    return which_dtype(dtype).zeros(shape, dtype=dtype)


def zeros_like(a: ndarray) -> ndarray:
    """Return an array of zeros with the same shape and type as a given array."""
    return which_np(a).zeros_like(a)
