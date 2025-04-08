#!/bin/bash

# Format of arguments: $1 - target trajectories, $2 - others trajectories
#                      $3 - cneighbors, $4 - discount factor
#                      $5 - maximum reward cap

# Define the path to the list.files
list_files_path="${1}/list.files"
list_files_path_nt="${2}/list.files"

# Check if list.files exists, if not, create it
if [ ! -f "$list_files_path" ]; then
  touch "$list_files_path"
  touch "$list_files_path_nt"
fi

echo $1/T*

# Append the names of files matching the pattern to list.files
for file in $1/T*;
  do
    echo $file >> "$list_files_path"
  done


# create list.files in the other trajectory set
for file in $2/T*
  do
    echo $file >> "$list_files_path_nt"
  done


# run method
cd ../

#python3 runRoMDP.py "${1}" "" "" "" $5 "CPLEX" "${1}/saved_results"
python3 runRoMDP.py "${1}" "${2}" "" "" $5 "CPLEX" "${1}/saved_results" "False"
