# stopping-power-ml
Repository for ML work with Andre Schleife on modeling stopping power

## Installation

These notebooks are based on Python 3. So, the first thing you need is a working Jupyter installation that
can run Python3 kernels.

For that kernel environment, you need to install the libraries listed in `requirements.txt` via pip or Conda.

Then, run `pip install -e .` to install the `stopping_power_ml` module.

Once those are installed, the next step is to install the development version of 
[ase](https://gitlab.com/ase/ase).

## Organization

This project is broken in to several subfolders.

`datasets` contains all of the TD-DFT data associated with this project. 
It is not tracked by git, so email Logan Ward to get a copy. 

`stopping_power_ml` is a Python module that contains utility operations for this project. 
Generally, these are methods that are used in more than one notebook.

`single-velocity` contains notebooks related to predicting the stopping power using only data 
relating to a single projectile velocity. We explore whether these models can be used
to determine whether ML can be used to halt a stopping power calculation early, and
whether our model can predict stopping power in different directions than 
what was included in the training set.

`multiple-velocities` [in progress] contains notebooks for testing whether our models
can predict stopping powers in different directions *and* velocities.

## Running Notebooks

You will notice that the name for each notebook starts with a number. 
To run the notebooks, execute them in the order indicated by this number because
the output of some notebooks are used as inputs into the following notebooks. 

Many of these notebooks use [Parsl](parsl.org) to perform calculations in parallel. 
The notebooks are currently configured to use IPyParallel to execute calculations on
your computer.
Consequently, you must call `ipcluster start -n $n$` (where $n$ is the number of processors
on your computer) before launching the notebooks.

A word of warning: the two notebooks in the root directory `0_parse_qbox` and 
`1_generate_representation` take a significant amount of computing time to complete.