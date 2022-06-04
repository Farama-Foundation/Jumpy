"""setup.py for Jumpy.

Install for development:

  pip install -e .
"""

from setuptools import setup

setup(
    name="brax-jumpy",
    version="0.0.2",
    description=("Common backend for JAX or numpy."),
    author="Brax Authors",
    author_email="no-reply@google.com",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="http://github.com/google/brax",
    license="Apache 2.0",
    py_modules=["jumpy"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=[
        "jax",
        "jaxlib",
        "numpy",
    ],
    extras_require={
        'dev': [
            'pytest',
        ]
    }
)
