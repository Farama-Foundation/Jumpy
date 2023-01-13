"""Functions from the `jax.ops` module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as onp

from jumpy.core import which_np

if TYPE_CHECKING:
    from jumpy.numpy import ndarray

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax, jnp = None, None


__all__ = ["segment_sum"]


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
