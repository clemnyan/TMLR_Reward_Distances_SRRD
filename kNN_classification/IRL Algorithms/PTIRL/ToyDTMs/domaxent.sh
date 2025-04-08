#!/usr/bin/env bash

if [[ $# -lt 1 ]];then
    echo "$0 directory dtmr learner -max_reward 500 -l1_regularization 1.0 -discount 0 <other DTM_learning args>"
    return
fi

rm -f "$1/save*" "$1/reward-hist.png"

#build the DTM
#./DTM_builder -training "$1/list.files" -attributes "$1/attributes.csv" -actions "$1/actions.csv" -diagnostic "$1/diag" -save "$1/save" -write_mapping "$1/save.map" -write_seq "$1/save.seq" -reward_header "Reward" -make_png
./DTM_builder -training "$1/list.files" -attributes "$1/attributes.csv" -actions "$1/actions.csv" -diagnostic "$1/diag" -save "$1/save" -write_mapping "$1/save.map" -write_seq "$1/save.seq" -make_png

#Run inverse reinforcement learning and save the graphviz file
./DTM_learning -dtm "$1/save" -do_irl maxent -attributes "$1/attributes.csv" -actions "$1/actions.csv" -diagnostics "$1/diag" -make_png "$1/save1" -trajectory "$1/list.files" ${@#$1} -maxent_thetas "$1/maxent_thetas"

learn_output_folder="$(cat "$1/diag" |grep "^\*\*\*\*Created "|sed 's/\*\*\*\*Created //g')"
paste -d, "${learn_output_folder}/__temp_feature_matrix_legend.csv" "$1/maxent_thetas"|tee "$1/maxent_thetas"

mv "${learn_output_folder}/local_rews*" "$1"
mv "${learn_output_folder}/local_stat*" "$1"



mv "${learn_output_folder}/unigrams.csv" "$1"
mv "${learn_output_folder}/bigrams.csv" "$1"
mv "${learn_output_folder}/chi2_contingency.csv" "$1"

#keep the reward values, policy, and transition probabilities
mv __temp_policy.csv "$1"
mv __temp_reward.csv "$1"
mv __temp_transition_table.csv "$1"

#run analysis on the output files
#./paths.py "$1"

