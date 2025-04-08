#!/usr/bin/env bash

if [[ $# -lt 1 ]];then
    echo "$0 <directory1,directory2,...,directoryn> <branching (True/False)> <merged states (True/False)> <merged % (default is 90)> [<discount factor (default 1.0)> <reward cap (default 100)>]"
    exit
fi


#build the DTM for joined trajectories
echo "Building joint DTM"
rundirs=${1//,/$'\n'}
for rundir in ${rundirs};do
    cat $rundir/list.files
done > .tmp_traj_joint

./DTM_builder -training .tmp_traj_joint  -attributes "$rundir/attributes.csv" -actions "$rundir/actions.csv" -diagnostic "$rundir/diag" -save ".tmp_dtm_joint" -write_mapping "$rundir/save.map" -write_seq "$rundir/save.seq" -make_png

#run the generated DTM with the original trajectories
./rundtmirl-all.py $@

rm .tmp_traj_joint .tmp_dtm_joint
