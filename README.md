<p align="center">
    <img src="jumpy-text.png" width="500px"/>
</p>

Jumpy is a common backend for [JAX](https://github.com/google/jax) or
[NumPy](https://numpy.org/):

* A Jumpy function returns a JAX outputs if given a JAX inputs
* A Jumpy function returns a JAX outputs if jitted
* Otherwise a jumpy function returns NumPy outputs

Jumpy lets you write framework agnostic code that is easy to debug by running
as raw Numpy, but is just as performant as JAX when jitted.

We maintain this repository primarily so to enable writing Gymnasium and PettingZoo wrappers that can be applied to both standard NumPy or hardware accelerated Jax based environments, however this package can be used for many more things. 

## Installing Jumpy

To install Jumpy from pypi:

```
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install brax-jumpy
```

Alternatively, to install Jumpy from source, clone this repo, `cd` to it, and then:

```
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -e .
```
