# Fast Machine Learning Science Benchmarks
[![DOI](https://zenodo.org/badge/445208377.svg)](https://zenodo.org/badge/latestdoi/445208377)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Implementations of the `fastml-science` benchmark models, including standard Keras (float) and QKeras (quantized) implementations.

## General Requirements

- [Miniconda](https://docs.anaconda.com/miniconda/miniconda-install/)
- Python 3.7 (Sensor Data Compression)
- Python 3.8 (Beam Control, Jet Classification)

## Overview of Benchmarks

| Benchmark                                          | Method        | Baseline Implementation Architecture |
| -------------------------------------------------- | ------------- | ------------------------------------ |
| [Jet Classification][jet-classify]                 | Supervised    | Multi-layer Perceptron               |
| [Sensor Data Compression][sensor-data-compression] | Unsupervised  | CNN Autoencoder                      |
| [Beam Control][beam-control]                       | Reinforcement | Deep Q Network + MLP                 |

[jet-classify]: jet-classify/README.md
[sensor-data-compression]: sensor-data-compression/README.md
[beam-control]: beam-control/README.md