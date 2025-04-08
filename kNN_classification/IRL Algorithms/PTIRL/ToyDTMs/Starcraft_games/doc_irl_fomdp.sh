#!/bin/bash

# Format of arguments: $1 - target trajectories, $2 - others trajectories
#                      $3 - cneighbors, $4 - discount factor
#                      $5 - maximum reward cap


# create list.files in the main trajectory set
for file in $1/T*
  do
    echo $file >> "${1}/list.files"
  done

# create list.files in the other trajectory set

for file in $2/T*
  do
    echo $file >> "${2}/list.files"
  done

# run method
cd ../
#
#python3 runRoMDP.py "${1}" "" "" "" $5 "CPLEX" "${1}/saved_results"
python3 runSsMDP.py "${1}" "${2}" "" "" $5 "CPLEX" "${1}/saved_results"
