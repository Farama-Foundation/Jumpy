<p align="center">
    <img src="https://raw.githubusercontent.com/Farama-Foundation/Jumpy/main/.github/jumpy-text.png" width="500px"/>
</p>

Jumpy is a common backend for [NumPy](https://numpy.org/) and optionally [JAX](https://github.com/google/jax):

* If Jax is installed and jax inputs are provided then the `jax.numpy` function is run
* If Jax is installed and the function is jitted then the `jax.numpy` function is run
* Otherwise the jumpy function returns the NumPy outputs

There are several functions (e.g. `vmap`, `scan`) that are available with `jax` installed.

Jumpy lets you write framework-agnostic code that is easy to debug by running
as raw Numpy, but is just as performant as JAX when jitted.

We maintain this repository primarily so to enable writing Gymnasium and PettingZoo wrappers that can be
applied to both standard NumPy or hardware accelerated Jax based environments, however this package can be used for many more things.

## Installing Jumpy

To install Jumpy from pypi: `pip install jax-jumpy[jax]` will include jax while `pip install jax-jumpy` will not include jax.

Alternatively, to install Jumpy from source, clone this repo, `cd` to it, and then: `pip install .`

## Contributing

Jumpy does not have a complete implementation of all `numpy` or `jax.numpy` functions. 
If you are missing functions then please create an issue or pull request, we will be happy to add them.

In the future, we are interested in adding optional support for PyTorch and looking for pull request to complete this.
