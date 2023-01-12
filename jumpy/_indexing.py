import numpy as onp
import jumpy

if jumpy.is_jax_installed:
    import jax
    import jax._src.numpy.lax_numpy as lax_numpy


    def jumpy_view(arr, maybe_subtype, *args, **kwargs):
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
        # Only do this for numpy arrays
        if isinstance(obj, onp.ndarray):
            setattr(self, "at", _IndexUpdateHelper(self))


class _IndexUpdateHelper(lax_numpy._IndexUpdateHelper):
    def __getitem__(self, index):
        return _IndexUpdateRef(self.array, index)


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

    def get(self, indices_are_sorted=False, unique_indices=False, mode=None, fill_value=None):
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
        raise NotImplementedError("Not implemented yet. Seems that jax API has a bug. "
                                  "jax_arr.at[1:2].apply(lambda x: x + 1) works, but"
                                  "jax_arr.at[2].apply(lambda x: x + 1) raises an error.")
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

    def multiply(self, values, indices_are_sorted=False, unique_indices=False, mode=None):
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
        array_copy[self.index] **= values #onp.power(array_copy[self.index], values)
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