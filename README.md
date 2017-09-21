# stopping-power-ml
Repository for ML work with Andre Schleife on modeling stopping power

## Installation

These notebooks are based on Python 3. So, the first thing you need is a working Jupyter installation that
can run Python3 kernels.

For that kernel environment, you need to install the libraries listed in `requirements.txt` via pip or Conda.

Once those are installed, the next step is to install the development versions of 
[ase](https://gitlab.com/ase/ase) and [matminer](https://github.com/hackingmaterials/matminer). 
I have been making modifications to these codes for this project, and the versions with the necessary 
additions have not been released on PyPI / Conda yet.

## Running Notebooks

You have two options for running these notebooks. First, you can go into Jupyter and run them sequentially:

1. generate-training-set.ipynb
2. build-machine-learning-model.ipynb
3. bayesian-regression-ci.ipynb

These notebooks take about two hours to run on a laptop with a recent Intel i7 processor.

The other option is to run the notebooks from the command line by calling `./run-notebooks.sh`.
