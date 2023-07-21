# Generative Ensemble Kalman Filter
_Generative EnKF_ is designed to surrogate data assimilation for numerical simulations without ensemble runs.

[![CI](https://github.com/yasahi-hpc/Generative-EnKF/actions/workflows/ci.yml/badge.svg)](https://github.com/yasahi-hpc/Generative-EnKF/actions)

# Usage

## Installation
This code relies on the following packages. As a deeplearing framework, we use [PyTorch](https://pytorch.org).
- Install Python libraries
[numpy](https://numpy.org), [PyTorch](https://pytorch.org), [xarray](http://xarray.pydata.org/en/stable/), and [netcdf4](https://github.com/Unidata/netcdf4-python)

- Clone this repo  
```git clone https://github.com/yasahi-hpc/Generative-EnKF.git```

## Preparation
Before running simulation with Generative EnKF, we need to train a diffusion model guided by observations. 
Firstly, one needs to construct a dataset and train the model for that.
See [deep learning model](docs/dl_model.md) for detail. 

## Simulation with Generative EnKF
For simulation, we rely on the [simulation codes](docs/simulation.md) and the pretrained diffusion model. 

## Citations
```bibtex
@INPROCEEDINGS{Asahi2023, 
      author={Asahi, Yuuichi and Hasegawa, Yuta and Onodera, Naoyuki and and Shimokawabe, Takashi and Shiba, Hayato and Idomura, Yasuhiro},
      booktitle={ICML 2023 Workshop SynS and ML}
      title={Generating observation guided ensembles for data assimilation with denoising diffusion probabilistic model},
      year={2023},
      volume={},
      number={},
      pages={},
      keywords = {Deep learning; Graphics-processing-unit-based computing; Data Assimilation; Lorenz96}
```
