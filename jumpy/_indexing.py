import numpy as onp

import jumpy

if jumpy.is_jax_installed:
    import jax
    import jax._src.numpy.lax_numpy as lax_numpy

    def jumpy_view(arr, maybe_subtype, *args, **kwargs):
        """This serves as a NOOP decorator for all jax array `.view` methods.

        Unfortunately, jax arrays already have a `.view` method. To allow jumpy users to call `.view` on jax arrays when they
        intend to use jumpy, we need to override the jax `.view` method with a NOOP decorator."""
        if maybe_subtype == jparray:
            return arr
        else:
            return lax_numpy._view(arr, maybe_subtype, *args, **kwargs)

    # import jaxlib.xla_extension contains more arrays --> overwrite view?
    # import jax._src.array contains more arrays --> overwrite view?
    lax_numpy.ShapedArray.view = jumpy_view
    lax_numpy.DShapedArray.view = jumpy_view
    lax_numpy.device_array.DeviceArray.view = jumpy_view
    lax_numpy.ArrayImpl.view = jumpy_view
    jax.Array.view = jumpy_view


class jparray(onp.ndarray):
    """A numpy.ndarray with additional JAX methods.

    Based on: https://numpy.org/doc/stable/user/basics.subclassing.html#simple-example-adding-an-extra-attribute-to-ndarray
    """

    def __array_finalize__(self, obj):
        """This method is the mechanism that numpy provides to allow this subclass to handle how new instances get created.

        More info on the role of this method can be found here:
        https://numpy.org/doc/stable/user/basics.subclassing.html#the-role-of-array-finalize
        """
        # Only do this for numpy arrays
        if isinstance(obj, onp.ndarray):
            setattr(self, "at", _IndexUpdateHelper(self))


# Note: this docstring will appear as the docstring for the `at` property.
class _IndexUpdateHelper:
    """Helper property for index update functionality.

    The ``at`` property provides a functionally pure equivalent of in-place
    array modificatons.

    In particular:

    ==============================  ================================
    Alternate syntax                Equivalent In-place expression
    ==============================  ================================
    ``x = x.at[idx].set(y)``        ``x[idx] = y``
    ``x = x.at[idx].add(y)``        ``x[idx] += y``
    ``x = x.at[idx].multiply(y)``   ``x[idx] *= y``
    ``x = x.at[idx].divide(y)``     ``x[idx] /= y``
    ``x = x.at[idx].power(y)``      ``x[idx] **= y``
    ``x = x.at[idx].min(y)``        ``x[idx] = minimum(x[idx], y)``
    ``x = x.at[idx].max(y)``        ``x[idx] = maximum(x[idx], y)``
    ``x = x.at[idx].apply(ufunc)``  ``ufunc.at(x, idx)``
    ``x = x.at[idx].get()``         ``x = x[idx]``
    ==============================  ================================

    None of the ``x.at`` expressions modify the original ``x``; instead they return
    a modified copy of ``x``. However, inside a :py:func:`~jax.jit` compiled function,
    expressions like :code:`x = x.at[idx].set(y)` are guaranteed to be applied in-place.

    Unlike NumPy in-place operations such as :code:`x[idx] += y`, if multiple
    indices refer to the same location, all updates will be applied (NumPy would
    only apply the last update, rather than applying all updates.) The order
    in which conflicting updates are applied is implementation-defined and may be
    nondeterministic (e.g., due to concurrency on some hardware platforms).

    By default, JAX assumes that all indices are in-bounds. There is experimental
    support for giving more precise semantics to out-of-bounds indexed accesses,
    via the ``mode`` parameter (see below).

    Arguments
    ---------
    mode : str
        Specify out-of-bound indexing mode. Options are:

        - ``"promise_in_bounds"``: (default) The user promises that indices are in bounds.
          No additional checking will be performed. In practice, this means that
          out-of-bounds indices in ``get()`` will be clipped, and out-of-bounds indices
          in ``set()``, ``add()``, etc. will be dropped.
        - ``"clip"``: clamp out of bounds indices into valid range.
        - ``"drop"``: ignore out-of-bound indices.
        - ``"fill"``: alias for ``"drop"``.  For `get()`, the optional ``fill_value``
          argument specifies the value that will be returned.

          See :class:`jax.lax.GatherScatterMode` for more details.

    indices_are_sorted : bool
        If True, the implementation will assume that the indices passed to ``at[]``
        are sorted in ascending order, which can lead to more efficient execution
        on some backends.
    unique_indices : bool
        If True, the implementation will assume that the indices passed to ``at[]``
        are unique, which can result in more efficient execution on some backends.
    fill_value : Any
        Only applies to the ``get()`` method: the fill value to return for out-of-bounds
        slices when `mode` is ``'fill'``. Ignored otherwise. Defaults to ``NaN`` for
        inexact types, the largest negative value for signed types, the largest positive
        value for unsigned types, and ``True`` for booleans.

    Examples
    --------
    >>> x = jnp.arange(5.0)
    >>> x
    DeviceArray([0., 1., 2., 3., 4.], dtype=float32)
    >>> x.at[2].add(10)
    DeviceArray([ 0.,  1., 12.,  3.,  4.], dtype=float32)
    >>> x.at[10].add(10)  # out-of-bounds indices are ignored
    DeviceArray([0., 1., 2., 3., 4.], dtype=float32)
    >>> x.at[20].add(10, mode='clip')
    DeviceArray([ 0.,  1.,  2.,  3., 14.], dtype=float32)
    >>> x.at[2].get()
    DeviceArray(2., dtype=float32)
    >>> x.at[20].get()  # out-of-bounds indices clipped
    DeviceArray(4., dtype=float32)
    >>> x.at[20].get(mode='fill')  # out-of-bounds indices filled with NaN
    DeviceArray(nan, dtype=float32)
    >>> x.at[20].get(mode='fill', fill_value=-1)  # custom fill value
    DeviceArray(-1., dtype=float32)
    """

    __slots__ = ("array",)

    def __init__(self, array):
        self.array = array

    def __getitem__(self, index):
        return _IndexUpdateRef(self.array, index)

    def __repr__(self):
        return f"_IndexUpdateHelper({repr(self.array)})"


class _IndexUpdateRef:
    """Helper object to call indexed update functions for an (advanced) index.

    This object references a source array and a specific indexer into that array.
    Methods on this object return copies of the source array that have been
    modified at the positions specified by the indexer.
    """

    __slots__ = ("array", "index")

    def __init__(self, array, index):
        self.array = array
        self.index = index

    def __repr__(self):
        return f"_IndexUpdateRef({repr(self.array)}, {repr(self.index)})"

    def get(
        self, indices_are_sorted=False, unique_indices=False, mode=None, fill_value=None
    ):
        """Equivalent to ``x[idx]``.

        Returns the value of ``x`` that would result from the NumPy-style
        :mod:indexing <numpy.doc.indexing>` ``x[idx]``. This function differs from
        the usual array indexing syntax in that it allows additional keyword
        arguments ``indices_are_sorted`` and ``unique_indices`` to be passed.

        See :mod:`jax.ops` for details.
        """
        return self.array[self.index]

    def set(self, values, indices_are_sorted=False, unique_indices=False, mode=None):
        """Pure equivalent of ``x[idx] = y``.

        Returns the value of ``x`` that would result from the NumPy-style
        :mod:`indexed assignment <numpy.doc.indexing>` ``x[idx] = y``.

        See :mod:`jax.ops` for details.
        """
        array_copy = self.array.copy()
        array_copy[self.index] = values
        return array_copy

    def apply(self, func, indices_are_sorted=False, unique_indices=False, mode=None):
        """Pure equivalent of ``func.at(x, idx)`` for a unary ufunc ``func``.

        Returns the value of ``x`` that would result from applying the unary
        function ``func`` to ``x`` at the given indices. This is similar to
        ``x.at[idx].set(func(x[idx]))``, but differs in the case of repeated indices:
        in ``x.at[idx].apply(func)``, repeated indices result in the function being
        applied multiple times.

        Note that in the current implementation, ``scatter_apply`` is not compatible
        with automatic differentiation.

        See :mod:`jax.ops` for details.
        """
        raise NotImplementedError(
            "Not implemented yet. Seems that jax API has a bug. "
            "jax_arr.at[1:2].apply(lambda x: x + 1) works, but"
            "jax_arr.at[2].apply(lambda x: x + 1) raises an error."
        )
        # array_copy = self.array.copy()
        # array_copy[self.index] = func(array_copy[self.index])
        # return array_copy

    def add(self, values, indices_are_sorted=False, unique_indices=False, mode=None):
        """Pure equivalent of ``x[idx] += y``.

        Returns the value of ``x`` that would result from the NumPy-style
        :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] += y``.

        See :mod:`jax.ops` for details.
        """
        array_copy = self.array.copy()
        array_copy[self.index] += values
        return array_copy

    def multiply(
        self, values, indices_are_sorted=False, unique_indices=False, mode=None
    ):
        """Pure equivalent of ``x[idx] *= y``.

        Returns the value of ``x`` that would result from the NumPy-style
        :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] *= y``.

        See :mod:`jax.ops` for details.
        """
        array_copy = self.array.copy()
        array_copy[self.index] *= values
        return array_copy

    mul = multiply

    def divide(self, values, indices_are_sorted=False, unique_indices=False, mode=None):
        """Pure equivalent of ``x[idx] /= y``.

        Returns the value of ``x`` that would result from the NumPy-style
        :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] /= y``.

        See :mod:`jax.ops` for details.
        """
        # !NOTE: jax division seems to cast array to float32. This is not the case for ``x[idx] /= y`` with numpy.
        # _v = values if hasattr(onp.float32(values), "__len__") else float(values)
        array_copy = self.array.astype("float32", copy=True)
        array_copy[self.index] /= onp.array(values)
        return array_copy

    def power(self, values, indices_are_sorted=False, unique_indices=False, mode=None):
        """Pure equivalent of ``x[idx] **= y``.

        Returns the value of ``x`` that would result from the NumPy-style
        :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] **= y``.

        See :mod:`jax.ops` for details.
        """
        # _v = values if hasattr(onp.float32(values), "__len__") else [values]
        # if any([v < 0 for v in _v]):
        # 	array_copy = self.array.astype("float32", copy=True)
        # else:
        # 	array_copy = self.array.copy()
        # !NOTE: Strange jax behavior for integers with negative powers. We raise an error for now in that case.
        #   Can be reproduced with jax.numpy.arange(10).at[-1].power(-2) = array([0, 1, 2, 3, 4, 5, 6, 7, 8, -77356367], dtype=int32)
        array_copy = self.array.copy()
        array_copy[self.index] **= values  # onp.power(array_copy[self.index], values)
        return array_copy

    def min(self, values, indices_are_sorted=False, unique_indices=False, mode=None):
        """Pure equivalent of ``x[idx] = minimum(x[idx], y)``.

        Returns the value of ``x`` that would result from the NumPy-style
        :mod:indexed assignment <numpy.doc.indexing>`
        ``x[idx] = minimum(x[idx], y)``.

        See :mod:`jax.ops` for details.
        """
        array_copy = self.array.copy()
        array_copy[self.index] = onp.minimum(array_copy[self.index], values)
        return array_copy

    def max(self, values, indices_are_sorted=False, unique_indices=False, mode=None):
        """Pure equivalent of ``x[idx] = maximum(x[idx], y)``.

        Returns the value of ``x`` that would result from the NumPy-style
        :mod:indexed assignment <numpy.doc.indexing>`
        ``x[idx] = maximum(x[idx], y)``.

        See :mod:`jax.ops` for details.
        """
        array_copy = self.array.copy()
        array_copy[self.index] = onp.maximum(array_copy[self.index], values)
        return array_copy
