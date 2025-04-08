#!/bin/bash

if [[ $# -lt 2 ]];then
    echo "$0 <directory1,directory2,...,directoryn> <outputdir>"
    exit
fi

echo "all output from all run processes is placed into log files in $2"

#get the commanders
commanders=${1//,/$'\n'}

#make output directories
outputbase="$2"
runs=( "merged" "unmerged" "merged-branched" "unmerged-branched" )

for run in ${runs[@]};do
    for commander in ${commanders};do
        mkdir -p "$outputbase/$run/$commander"
    done
done

#extract the arguments that we want because we need them for dodouble*
args=( "$@" )
unset args[1]

#run each of the types
echo ./dodoubleirl-all.sh $1 False False 90 ${@:3} &> "$outputbase/unmerged/log"
./dodoubleirl-all.sh $1 False False 90 ${@:3} &> "$outputbase/unmerged/log"
for commander in ${commanders};do
    mv $commander/pairwise* "$outputbase/unmerged/$commander"
    mv $commander/state_report.csv "$outputbase/unmerged/$commander"
    mv $commander/joint-learned.dtm "$outputbase/unmerged/$commander"
    ./interpretreward2.py ${commander} ${outputbase}  &> ${outputbase}/interplog-${commander}.txt
done

echo ./dodoubleirl-all.sh $1 False True 90 ${@:3} &> "$outputbase/merged/log"
./dodoubleirl-all.sh $1 False True 90 ${@:3} &> "$outputbase/merged/log"
for commander in ${commanders};do
    mv $commander/pairwise* "$outputbase/merged/$commander"
    mv $commander/state_report.csv "$outputbase/merged/$commander"
    mv $commander/joint-learned.dtm "$outputbase/merged/$commander"
done

for i in {1..10};do
    ./dodoubleirl-all.sh $1 True True 90 ${@:3} &> "$outputbase/merged-branched/log-${i}"
    for commander in ${commanders};do
        mv $commander/pairwise-merged-report.csv "$outputbase/merged-branched/$commander/pairwise-merged-report-${i}.csv"
        mv $commander/pairwise-merged-output.csv "$outputbase/merged-branched/$commander/pairwise-merged-output-${i}.csv"
        mv $commander/pairwise.dot "$outputbase/merged-branched/$commander/pairwise-${i}.dot"
        mv $commander/joint-learned.dtm "$outputbase/merged-branched/$commander/joint-learned-${i}.dtm"
        mv $commander/state_report.csv "$outputbase/merged-branched/$commander/state_report-${i}.csv"
    done

    ./dodoubleirl-all.sh $1 True False 90 ${@:3} &> "$outputbase/unmerged-branched/log-${i}"
    for commander in ${commanders};do
        mv $commander/pairwise-merged-report.csv "$outputbase/unmerged-branched/$commander/pairwise-merged-report-${i}.csv"
        mv $commander/pairwise-merged-output.csv "$outputbase/unmerged-branched/$commander/pairwise-merged-output-${i}.csv"
        mv $commander/pairwise.dot "$outputbase/unmerged-branched/$commander/pairwise-${i}.dot"
        mv $commander/joint-learned.dtm "$outputbase/unmerged-branched/$commander/joint-learned-${i}.dtm"
        mv $commander/state_report.csv "$outputbase/unmerged-branched/$commander/state_report-${i}.csv"
    done
done

find $outputbase -type f -exec dos2unix {} +

#find $outputbase -type f -iname *.dot -exec dot -Tpng {} {}.png \;
