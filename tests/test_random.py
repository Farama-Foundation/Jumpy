"""Tests that random functions works."""

import jax
import jax.numpy as jnp
import numpy as onp

import jumpy


def test_random_prngkey():
    """Tests whether `random_prngkey` returns different keys with different seeds."""
    key1 = jumpy.random.PRNGKey(0)
    key2 = jumpy.random.PRNGKey(0)
    key3 = jumpy.random.PRNGKey(1)

    assert onp.array_equal(key1, key2)
    assert not onp.array_equal(key2, key3)


def test_random_uniform():
    """Tests whether `random_uniform` returns different numbers given different PRNG keys and with the correct return type."""
    key1 = jumpy.random.PRNGKey(0)
    key2 = jax.random.PRNGKey(0)

    val1 = jumpy.random.uniform(key1)
    val2 = jumpy.random.uniform(key1)
    val3 = jumpy.random.uniform(key2)
    val4 = jumpy.random.uniform(key2)

    assert isinstance(val1, onp.ndarray)
    assert isinstance(val3, jnp.ndarray)
    assert onp.array_equal(val1, val2)
    assert onp.array_equal(val3, val4)
    assert not onp.array_equal(val1, val3)


def test_random_split():
    """Tests whether `random_split` returns different keys wrt the originals and with the correct return type."""
    key1 = jumpy.random.PRNGKey(0)
    key2 = jax.random.PRNGKey(0)
    subkey1 = jumpy.random.split(key1)
    subkey2 = jumpy.random.split(key2)

    assert isinstance(subkey1, onp.ndarray)
    assert isinstance(subkey2, jnp.ndarray)
    assert not onp.array_equal(key1, subkey1)
    assert not onp.array_equal(key2, subkey2)
