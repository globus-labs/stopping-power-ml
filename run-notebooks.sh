#! /bin/bash

# This script just runs through all of the current notebooks in the proper order

runJupyter() {
    jupyter nbconvert --execute --inplace --ExecutePreprocessor.timeout=-1 $1
    jupyter nbconvert --to html $1
}

# If we get too many files, we should probably create a file containing
#  names of notebooks and the order in which they should be executed
for file in generate-training-set.ipynb build-machine-learning-model.ipynb bayesian-regression-ci.ipynb; do
    runJupyter $file
done