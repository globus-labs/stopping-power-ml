#! /bin/bash

# This script just runs through all of the current notebooks in the proper order

runJupyter() {
    jupyter nbconvert --execute --inplace --ExecutePreprocessor.timeout=-1 $1
}

# If we get too many files, we should probably create a file containing
#  names of notebooks and the order in which they should be executed
for file in `seq 0 2`; do
    runJupyter ${file}_*nb
done

