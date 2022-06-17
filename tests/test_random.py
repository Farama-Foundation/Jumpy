# pylint:disable=redefined-builtin


import pytest
import numpy as onp
from jax import numpy as jnp
import jax

import jumpy as jp


def test_random_prngkey():
    """
    Tests whether `random_prngkey` returns different keys with 
    different seeds
    """
    key1 = jp.random_prngkey(0)
    key2 = jp.random_prngkey(0)
    key3 = jp.random_prngkey(1)

    assert onp.array_equal(key1, key2)
    assert not onp.array_equal(key2, key3)


def test_random_uniform():
    """
    Tests whether `random_uniform` returns different numbers given
    different PRNG keys and with the correct return type
    """
    key1 = jp.random_prngkey(0)
    key2 = jax.random.PRNGKey(0)

    val1 = jp.random_uniform(key1)
    val2 = jp.random_uniform(key1)
    val3 = jp.random_uniform(key2)
    val4 = jp.random_uniform(key2)

    assert isinstance(val1, onp.ndarray)
    assert isinstance(val3, jnp.ndarray)
    assert onp.array_equal(val1, val2)
    assert onp.array_equal(val3, val4)
    assert not onp.array_equal(val1, val3)


def test_random_split():
    """
    Tests whether `random_split` returns different keys wrt the 
    originals and with the correct return type
    """
    key1 = jp.random_prngkey(0)
    key2 = jax.random.PRNGKey(0)
    subkey1 = jp.random_split(key1)
    subkey2 = jp.random_split(key2)

    assert isinstance(subkey1, onp.ndarray)
    assert isinstance(subkey2, jnp.ndarray)
    assert not onp.array_equal(key1, subkey1)
    assert not onp.array_equal(key2, subkey2)