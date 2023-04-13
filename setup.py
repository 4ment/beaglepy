import os

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open(os.path.join("src", "beaglepy", "_version.py")) as f:
    __version__ = f.readlines()[-1].split()[-1].strip("\"'")

ext_modules = [
    Pybind11Extension(
        "beaglepy.beagle",
        ["src/beaglepy/beagle.cpp"],
        libraries=['hmsbeagle'],
        # Example: passing in the version to the compiled code
        define_macros=[('VERSION_INFO', __version__)],
    ),
]

setup(
    **{
        'ext_modules': ext_modules,
        'cmdclass': {'build_ext': build_ext},
        'version': __version__,
    }
)
