#! /bin/bash

# Run the stuff in this directory
./run-notebooks.sh

# Now, run the single-velocities
cd single-velocity
./run-notebooks.sh

# Now, run the NN portion
cd neural-network
./run-notebooks.sh
