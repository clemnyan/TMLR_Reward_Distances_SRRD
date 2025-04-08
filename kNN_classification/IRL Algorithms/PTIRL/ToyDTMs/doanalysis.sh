#!/usr/bin/env bash

if [[ $# -lt 1 ]];then
    echo "$0 directory dtmr learner -max_reward 500 -l1_regularization 1.0 -discount 0 <other DTM_learning args>"
    return
fi

rm -f "$1/save*" "$1/reward-hist.png"

#build the DTM
./DTM_builder -training "$1/list.files" -attributes "$1/attributes.csv" -actions "$1/actions.csv" -diagnostic "$1/diag" -save "$1/save"

#Run inverse reinforcement learning and save the graphviz file
./DTM_learning -dtm "$1/save" -do_irl linear -attributes "$1/attributes.csv" -actions "$1/actions.csv" -diagnostics "$1/diag" -make_png "$1/save" ${@#$1}

#keep the reward values, policy, and transition probabilities
mv __temp_policy.csv "$1"
mv __temp_reward.csv "$1"
mv __temp_transition_table.csv "$1"

#run analysis on the output files
./paths.py "$1"

