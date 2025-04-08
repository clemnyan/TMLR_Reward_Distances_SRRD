#!/usr/bin/env bash

if [[ $# -lt 1 ]];then
    echo "$0 directory dtmr learner -max_reward 500 -l1_regularization 1.0 -discount 0 <other DTM_learning args>"
    return
fi

rm -fr "$1/save*" "$1/reward-hist.png" "$1/results"
mkdir -p "$1/results"

epoch=100
for i in {1..90};do
    discount=$(echo -e "scale=2;$i*0.01"|bc -l)
    for j in {1..90};do
        learningrate=$(echo -e "scale=2;$j*0.01"|bc -l)
        outdir="$1/results/disc-${discount}-learnrate-${learningrate}-epoch-${epoch}"
        mkdir -p "$outdir"
        #build the DTM
        #./DTM_builder -training "$1/list.files" -attributes "$1/attributes.csv" -actions "$1/actions.csv" -diagnostic "$1/diag" -save "$1/save" -write_mapping "$1/save.map" -write_seq "$1/save.seq" -reward_header "Reward" -make_png
        ./DTM_builder -training "$1/list.files" -attributes "$1/attributes.csv" -actions "$1/actions.csv" -diagnostic "${outdir}/diag" -save "${outdir}/save" -write_mapping "$outdir/save.map" -write_seq "$outdir/save.seq" -make_png

        #Run inverse reinforcement learning and save the graphviz file
        { ./DTM_learning -dtm "$outdir/save" -do_irl maxent -attributes "$1/attributes.csv" -actions "$1/actions.csv" -diagnostics "$outdir/diag" -make_png "$outdir/save1" -trajectory "$1/list.files" ${@#$1} -maxent_thetas "$outdir/maxent_thetas" -feature_matrix binary -discount ${discount} -learning_rate ${learningrate} -epochs ${epoch} -maxent_threshold 1.0 2>&1; }  &> "$outdir/screenoutput"

        learn_output_folder="$(cat "$outdir/diag" |grep "^\*\*\*\*Created "|sed 's/\*\*\*\*Created //g')"
        paste -d, "${learn_output_folder}/__temp_feature_matrix_legend.csv" "$outdir/maxent_thetas"|tee "$outdir/maxent_thetas"

        cp "${learn_output_folder}"/* "$outdir"
        #mv "${learn_output_folder}/local_rews*" "$outdir"
        #mv "${learn_output_folder}/local_stat*" "$outdir"
        #mv "${learn_output_folder}/unigrams.csv" "$outdir"
        #mv "${learn_output_folder}/bigrams.csv" "$outdir"
        #mv "${learn_output_folder}/chi2_contingency.csv" "$outdir"
        #keep the reward values, policy, and transition probabilities
        mv __temp_policy.csv "$outdir"
        mv __temp_reward.csv "$outdir"
        mv __temp_transition_table.csv "$outdir"
    done
done

#run analysis on the output files
#./paths.py "$1"

