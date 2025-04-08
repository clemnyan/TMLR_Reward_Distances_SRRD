#!/usr/bin/env bash

#read in all txt files and generate .gen files
for file in *.txt;do
    #iterate swapping out which smoke is which
    {
    cat $file
    for s1 in S1 S2 S3 S4;do
        for s2 in S1 S2 S3 S4;do
            [[ $s1 == $s2 ]] && continue
            #iterate swapping out enemy ships
            for es1 in EB ES;do
                for es2 in EB ES;do
                    [[ $es1 == $es2 ]] && continue
                    #iterate swapping out friendly ships
                    for ship1 in C B1 B2;do
                        for ship2 in C B1 B2;do
                            [[ $ship1 == $ship2 ]] && continue
                            #cat $file|sed "s/${s1}/__tmp__/g"|sed "s/${s2}/${s1}/g"|sed "s/__tmp__/${s2}/g"|sed "s/${ship1}/__tmp__/g"|sed "s/${ship2}/${ship1}/g"|sed "s/__tmp__/${ship2}/g"|sed "s/${es1}/__tmp__/g"|sed "s/${es2}/$es1/g"|sed "s/__tmp__/${es1}/g"
                            cat $file |sed -e "s/${s1}/__tmp__/g" -e  "s/${s2}/${s1}/g" -e  "s/__tmp__/${s2}/g" -e  "s/${ship1}/__tmp__/g" -e  "s/${ship2}/${ship1}/g" -e  "s/__tmp__/${ship2}/g" -e  "s/${es1}/__tmp__/g" -e  "s/${es2}/$es1/g" -e  "s/__tmp__/${es1}/g"
                        done
                    done
                done
            done
        done
    done 
    }|sort|uniq > ${file%.txt}.gen
done
