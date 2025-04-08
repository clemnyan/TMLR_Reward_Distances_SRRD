#!/usr/bin/env bash

if [[ $# -lt 1 ]];then
    echo "$0 <directory> [<neighborhood size (default 4)> <discount factor (default 1.0)> <reward cap (default 100)>]"
    return
fi


#build the DTM
./DTM_builder -training "$1/list.files" -attributes "$1/attributes.csv" -actions "$1/actions.csv" -diagnostic "$1/diag" -save "$1/save" -write_mapping "$1/save.map" -write_seq "$1/save.seq" -make_png

./rundtmirl.py ${@}
