# stopping-power-ml

A study of how to compute electronic stopping power quickly using a combination of machine learning and Time-Dependent Density Functional Theory (TD-DFT).

See our [paper for more details](https://arxiv.org/abs/2311.00787v1)

## Installation

The environment for this project is defined in `environment.yml`.
Install it using Conda:

```bash
conda env create --file environment.yml --force
```

This will produce an environment named `td_dft` you must activate it
following the instructions given by conda after installation and 
launch Jupyter from within this environment for the notebooks to function. 

## Organization

This project is broken in to several subfolders.

`datasets` contains all of the TD-DFT data associated with this project. 
It is not tracked by git, so get the data from [our](https://acdc.alcf.anl.gov/mdf/detail/schleife_accurate_atomistic_stopping_v1.1/) [two](https://acdc.alcf.anl.gov/mdf/detail/schleife2018_v1.1/) 
datasets on the Materials Data Facility.

`stopping_power_ml` is a Python module that contains utility operations for this project. 
Generally, these are methods that are used in more than one notebook.

`single-velocity` contains notebooks related to predicting the stopping power using only data 
relating to a single projectile velocity. We explore whether these models can be used
to determine whether ML can be used to halt a stopping power calculation early, and
whether our model can predict stopping power in different directions than 
what was included in the training set.

`multiple-velocities` contains notebooks for testing whether our models
can predict stopping powers in different directions *and* velocities.

## Running Notebooks

You will notice that the name for each notebook starts with a number. 
To run the notebooks, execute them in the order indicated by this number because
the output of some notebooks are used as inputs into the following notebooks. 

A word of warning: the two notebooks in the root directory `0_parse_qbox` and 
`1_generate_representation` take a significant amount of computing time to complete.
