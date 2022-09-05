# JumPy

JumPy is a common backend for [JAX](https://github.com/google/jax) or
[numpy](https://numpy.org/):

* A jumpy function returns a JAX outputs if given a JAX inputs
* A jumpy function returns a JAX outputs if jitted
* Otherwise a jumpy function returns numpy outputs

JumPy lets you write framework agnostic code that is easy to debug by running
as raw numpy, but is just as performant as JAX when jitted.

## Installing JumPy

To install JumPy from pypi:

```
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install brax-jumpy
```

Alternatively, to install JumPy from source, clone this repo, `cd` to it, and then:

```
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -e .
```
