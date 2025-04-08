#!/bin/bash

# Format of arguments: $1 - target trajectories, $2 - others trajectories
#                      $3 - cneighbors,          $4 - discount factor
#                      $5 - maximum reward cap,  $6 - tau
#                      $7 - feasibility_only


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

python3 runSsMDP.py "${1}" "${2}" "" "" $5 $6 "BARON" "${1}/saved_results" $7
