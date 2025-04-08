#!/usr/bin/env python3

import sys
import pydtm
import dtm_states

dtm=pydtm.pydtm(sys.argv[1])

rewards=set([trip.reward for trip in dtm.triples])
mapping={}
count1=0
for att in dtm.lespace.attribute_names:
    mapping[att]={}
    for val in dtm.lespace.attribute_vals[count1]:
        mapping[att][val]={}
        count2=0
        for att2 in dtm.lespace.attribute_names:
            #print(att2)
            mapping[att][val][att2]={}
            for val2 in dtm.lespace.attribute_vals[count2]:
                #print("\t"+val2)
                mapping[att][val][att2][val2]={}
                for reward in rewards:
                    mapping[att][val][att2][val2][reward]=0
            count2=count2+1
    count1=count1+1

for trip in dtm.triples:
    
    source=None
    for lidx, value in enumerate(dtm.states[trip.source].les_set):
        if int(value) == 1:
            source=dtm.lespace.les[lidx]
            break

    dest=None
    for lidx, value in enumerate(dtm.states[trip.dest].les_set):
        if int(value) == 1:
            dest=dtm.lespace.les[lidx]
            break

    for k1,v1 in source.attributes.items():
        rv1=dtm.lespace.attribute_names[int(k1)]
        s1=dtm.lespace.attribute_vals[int(k1)][int(v1)]
        for k2,v2 in dest.attributes.items():
            rv2=dtm.lespace.attribute_names[int(k2)]
            s2=dtm.lespace.attribute_vals[int(k2)][int(v2)]
            mapping[rv1][s1][rv2][s2][trip.reward]=mapping[rv1][s1][rv2][s2][trip.reward]+1

    #DTM_stateDistance(dtm.lespace,source.,{},dest,{})

#count1=0
#print("rv1=state1,rv2=state2",end='')
#for reward in rewards:
#    print(",count_reward="+str(reward),end='')
#print()
#
#for att in dtm.lespace.attribute_names:
#    for val in dtm.lespace.attribute_vals[count1]:
#        count2=0
#        for att2 in dtm.lespace.attribute_names:
#            for val2 in dtm.lespace.attribute_vals[count2]:
#                print(att+"="+val+","+att2+"="+val2,end='')
#                for reward in rewards:
#                    print(","+str(mapping[att][val][att2][val2][reward]),end='')
#                print()
#            count2=count2+1
#    count1=count1+1
#

offset=0
for trip in dtm.triples:
    offset=offset+1

    source1=None
    for lidx, value in enumerate(dtm.states[trip.source].les_set):
        if int(value) == 1:
            source1=dtm.lespace.les[lidx]
            break

    dest1=None
    for lidx, value in enumerate(dtm.states[trip.dest].les_set):
        if int(value) == 1:
            dest1=dtm.lespace.les[lidx]
            break

    for trip2 in dtm.triples[offset:]:
        if trip.source != trip2.source and trip.dest==trip2.source and trip.source == trip2.dest and trip.reward != trip2.reward:
            for k1,v1 in source1.attributes.items():
                rv1=dtm.lespace.attribute_names[int(k1)]
                s1=dtm.lespace.attribute_vals[int(k1)][int(v1)]
                print(str(rv1)+"="+str(s1)+",",end='')

            print()
            print("->")

            for k1,v1 in dest1.attributes.items():
                rv1=dtm.lespace.attribute_names[int(k1)]
                s1=dtm.lespace.attribute_vals[int(k1)][int(v1)]
                print(str(rv1)+"="+str(s1)+",",end='')

            print()
            print(str(trip.reward)+" -> "+str(trip2.reward))

            source2=None
            for lidx, value in enumerate(dtm.states[trip2.source].les_set):
                if int(value) == 1:
                    source2=dtm.lespace.les[lidx]
                    break

            dest2=None
            for lidx, value in enumerate(dtm.states[trip2.dest].les_set):
                if int(value) == 1:
                    dest2=dtm.lespace.les[lidx]
                    break
    #dtm_states.DTM_stateDistance(dtm.lespace,source.attributes,{},dest.attributes,{},hamming=True)
