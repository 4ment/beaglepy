# beaglepy: python binding for BEAGLE

beaglepy is a Python package for calculating phylogenetic likelihoods of sequence evolution.
Under the hood, beaglepy utilizes the high performance library [BEAGLE](https://github.com/beagle-dev/beagle-lib) and can therefore make use of highly-parallel processors such as those in graphics cards (GPUs) found in many PCs.

## Install

beaglepy can be installed on Linux or Mac (potentially on Windows), with the following prerequisites:

* [Python] 2.7, 3.5 or newer
* [BEAGLE](https://github.com/beagle-dev/beagle-lib)
* [pybind11](https://github.com/pybind/pybind11) >= 2.5.0
* A compiler compatible with pybind11 (see [requirements](https://github.com/pybind/pybind11))
