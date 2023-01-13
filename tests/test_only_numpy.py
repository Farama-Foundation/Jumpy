"""Tests that jumpy works with only numpy installed."""
import numpy as np
import pytest

import jumpy
import jumpy.numpy as jp
from jumpy.core import which_dtype, which_np


@pytest.mark.parametrize(
    "args",
    ((0,), (0, np.array(0)), (0, np.array([0])), (np.array(0),), (np.array([0]))),
)
def test_which_np(args):
    """Test the jp._which_np function."""
    assert which_np(*args) is np


@pytest.mark.parametrize("dtype", (int, float, np.int32, np.uint8))
def test_which_dtype(dtype):
    """Test the jp._which_dtype function."""
    assert which_dtype(dtype) is np


@pytest.mark.parametrize(
    "safe_func_name, func_name, args",
    [
        ("safe_arcsin", "arcsin", 1),
        ("safe_arccos", "arccos", 1),
    ],
)
def test_safe_func(safe_func_name, func_name, args):
    """Test all jumpy safe functions (except safe_norm)."""
    jp_out = getattr(jp, safe_func_name)(args)
    np_out = getattr(np, func_name)(args)

    assert type(jp_out) is type(np_out)
    assert np.all(jp_out == np_out)


@pytest.mark.parametrize(
    "x, axis", [(np.array([0, 1, 2]), None), (np.array([[0, 1], [2, 2], [3, 4]]), 0)]
)
def test_safe_norm(x, axis):
    """Test the jp.safe_norm function."""
    safe_norm = jp.linalg.safe_norm(x, axis)
    jp_norm = jp.linalg.norm(x, axis)
    np_norm = np.linalg.norm(x, axis=axis)

    assert type(safe_norm) is type(jp_norm) is type(np_norm)
    assert np.all(safe_norm == jp_norm)
    assert np.all(jp_norm == np_norm)


@pytest.mark.parametrize(
    "func_name, args",
    [("norm", np.array([1, 2, 3])), ("inv", np.array([[1, 2], [3, 4]]))],
)
def test_np_linalg_func(func_name, args):
    """Test the np.linalg functions."""
    jp_out = getattr(jp.linalg, func_name)(args)
    np_out = getattr(np.linalg, func_name)(args)

    assert type(jp_out) is type(np_out)
    assert np.all(jp_out == np_out)


def test_random_funcs():
    """Test all jumpy random functions."""
    rng = jumpy.random.PRNGKey(seed=123)
    rng, rng_1, rng_2, rng_3 = jumpy.random.split(rng, num=4)

    x = jumpy.random.uniform(rng)
    assert 0 <= x <= 1
    x = jumpy.random.uniform(rng_1, (2,), low=1, high=2)
    assert np.all(1 <= x) and np.all(x <= 2) and x.shape == (2,)

    x = jumpy.random.randint(rng_2, (2,), low=2, high=10)
    assert np.all(2 <= x) and np.all(x <= 10) and x.shape == (2,)

    x = jumpy.random.choice(rng_3, 5, shape=(2,))
    assert np.all(0 <= x) and np.all(x <= 5) and x.shape == (2,)


@pytest.mark.skipif(
    jumpy.is_jax_installed is True,
    reason="This test requires that jax is not installed.",
)
def test_jax_only_funcs():
    """Test jax-only functions."""
    with pytest.raises(NotImplementedError):
        jumpy.vmap(lambda x: x + 1)

    with pytest.raises(NotImplementedError):
        jumpy.lax.scan(lambda a, b: (b, a + b), init=0, xs=[0, 1, 2])

    with pytest.raises(NotImplementedError):
        jp.take(tree=jp.array([0, 1, 2]), i=1)


@pytest.mark.parametrize(
    "func_name, kwargs, expected",
    (("index_update", {"x": np.array([0, 1]), "idx": 0, "y": 2}, np.array([2, 1])),),
)
def test_custom_np_func(func_name, kwargs, expected):
    """Test the implementation of custom np functions."""
    out = getattr(jumpy, func_name)(**kwargs)

    if isinstance(out, tuple):
        assert all(np.all(a == b) for a, b in zip(out, expected))
    else:
        assert np.all(out == expected)


@pytest.mark.parametrize(
    "func_name, kwargs, expected",
    (
        (
            "while_loop",
            {"cond_fun": lambda x: x < 4, "body_fun": lambda x: x + 1, "init_val": 0},
            4,
        ),
        (
            "fori_loop",
            {
                "lower": 0,
                "upper": 4,
                "body_fun": lambda i, x: x + [len(x)],
                "init_val": [],
            },
            [0, 1, 2, 3],
        ),
        (
            "top_k",
            {"operand": np.array([1, 2, 4, 3, 6]), "k": 2},
            (np.array([6, 4]), np.array([4, 2])),
        ),
    ),
)
def test_custom_np_lax_func(func_name, kwargs, expected):
    """Test the implementation of custom np functions."""
    out = getattr(jumpy.lax, func_name)(**kwargs)

    if isinstance(out, tuple):
        assert all(np.all(a == b) for a, b in zip(out, expected))
    else:
        assert np.all(out == expected)


@pytest.mark.parametrize(
    "func_name, kwargs, expected",
    (
        (
            "segment_sum",
            {"data": np.arange(5), "segment_ids": np.array([0, 0, 1, 1, 2])},
            np.array([1, 5, 4]),
        ),
    ),
)
def test_custom_np_ops_func(func_name, kwargs, expected):
    """Test the implementation of custom np functions."""
    out = getattr(jumpy.ops, func_name)(**kwargs)

    if isinstance(out, tuple):
        assert all(np.all(a == b) for a, b in zip(out, expected))
    else:
        assert np.all(out == expected)


def test_meshgrid_cond():
    """Test that meshgrid and cond work as use `*operands` this doesn't work with `test_custom_np_func`."""
    out = jumpy.lax.cond(True, lambda *x: x[0] + 1, lambda *x: x[0] - 1, 5)
    expected_out = 6
    assert out == expected_out

    xv, yv = jp.meshgrid(jp.linspace(0, 1, 3), jp.linspace(0, 1, 2))
    expected_xv = np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
    expected_yv = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert np.all(xv == expected_xv)
    assert np.all(yv == expected_yv)
