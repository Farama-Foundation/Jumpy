import jax
import numpy as onp

import jumpy


def test_indexing():
    # Prepare two different arrays for testing
    a = onp.arange(0, 10)
    b = jax.numpy.arange(0, 10)

    # Check view casting (https://numpy.org/doc/stable/user/basics.subclassing.html#view-casting)
    a_jp = a.view(jumpy.jparray)
    b_jp = b.view(jumpy.jparray)

    # Check template casting (https://numpy.org/doc/stable/user/basics.subclassing.html#creating-new-from-template)
    a2_jp = a_jp[:2]
    b2_jp = b_jp[:2]
    assert isinstance(b2_jp, jax.Array)
    assert isinstance(a2_jp, jumpy.jparray)

    # Use this util to get indices from slice
    class Indexer:
        def __getitem__(self, item):
            if isinstance(item, slice):
                return list(range(item.stop)[item])

    indexer = Indexer()

    # Test settiing values
    slices = [2, slice(1, 2), slice(1, 5)]
    for sl in slices:
        # Test API for scalar
        assert onp.isclose(a_jp.at[sl].get(-2), b_jp.at[sl].get(-2)).all()
        assert onp.isclose(a_jp.at[sl].set(-2), b_jp.at[sl].set(-2)).all()
        assert onp.isclose(a_jp.at[sl].add(-1), b_jp.at[sl].add(-1)).all()
        assert onp.isclose(a_jp.at[sl].mul(-2), b_jp.at[sl].mul(-2)).all()
        assert onp.isclose(a_jp.at[sl].divide(-2), b_jp.at[sl].divide(-2)).all()
        assert onp.isclose(a_jp.at[sl].power(2), b_jp.at[sl].power(2)).all()
        assert onp.isclose(a_jp.at[sl].min(-2), b_jp.at[sl].min(-2)).all()
        assert onp.isclose(a_jp.at[sl].max(-2), b_jp.at[sl].max(-2)).all()

        # Test API for array setting
        ind = indexer[sl]
        val = ind if ind is not None else 2
        assert onp.isclose(a_jp.at[sl].get(val), b_jp.at[sl].get(val)).all()
        assert onp.isclose(a_jp.at[sl].set(val), b_jp.at[sl].set(val)).all()
        assert onp.isclose(a_jp.at[sl].add(val), b_jp.at[sl].add(val)).all()
        assert onp.isclose(a_jp.at[sl].mul(val), b_jp.at[sl].mul(val)).all()
        assert onp.isclose(a_jp.at[sl].divide(val), b_jp.at[sl].divide(val)).all()
        assert onp.isclose(a_jp.at[sl].power(val), b_jp.at[sl].power(val)).all()
        assert onp.isclose(a_jp.at[sl].min(val), b_jp.at[sl].min(val)).all()
        assert onp.isclose(a_jp.at[sl].max(val), b_jp.at[sl].max(val)).all()


if __name__ == "__main__":
    test_indexing()
