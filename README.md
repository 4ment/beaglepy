# beaglepy: python binding for BEAGLE

[![Build Status](https://travis-ci.org/4ment/beaglepy.svg?branch=master)](https://travis-ci.org/4ment/beaglepy)
[![Docker Cloud Build Status](https://img.shields.io/docker/cloud/build/4ment/beaglepy.svg)](https://hub.docker.com/r/4ment/beaglepy)
[![License](https://img.shields.io/github/license/4ment/beaglepy)](LICENCE)

beaglepy is a Python package for calculating phylogenetic likelihoods of sequence evolution.
Under the hood, beaglepy utilizes the high performance library [BEAGLE](https://github.com/beagle-dev/beagle-lib) and can therefore make use of highly-parallel processors such as those in graphics cards (GPUs) found in many PCs.

## Install

beaglepy can be installed on Linux or Mac (potentially on Windows), with the following prerequisites:

* [Python](https://www.python.org) 2.7, 3.5 or newer
* [BEAGLE](https://github.com/beagle-dev/beagle-lib)
* [pybind11](https://github.com/pybind/pybind11) >= 2.5.0
* A compiler compatible with pybind11 (see [requirements](https://github.com/pybind/pybind11))
