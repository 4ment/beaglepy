# beaglepy: python binding for BEAGLE

[![Testing (Linux)](https://github.com/4ment/beaglepy/actions/workflows/test_linux.yml/badge.svg)](https://github.com/4ment/beaglepy/actions/workflows/test_linux.yml)
[![License](https://img.shields.io/github/license/4ment/beaglepy)](LICENCE)

beaglepy is a Python package for calculating phylogenetic likelihoods of sequence evolution.
Under the hood, beaglepy utilizes the high performance library [BEAGLE](https://github.com/beagle-dev/beagle-lib) and can therefore make use of highly-parallel processors such as those in graphics cards (GPUs) found in many PCs.

## Install

beaglepy can be installed on Linux or Mac (potentially on Windows), with the following prerequisites:

* [Python](https://www.python.org) 3.7 or newer
* [BEAGLE](https://github.com/beagle-dev/beagle-lib)
* [pybind11](https://github.com/pybind/pybind11) >= 2.5.0
* A compiler compatible with pybind11 (see [requirements](https://github.com/pybind/pybind11))
