# 
# File: dtm_generateReport.py
# Author: Eugene Santos Jr.
# Date: 2017-11-12
# Project: DTM
# Copyright: Eugene Santos Jr.
#


# Given a dtm, generates a csv file which reports the information in the dtm
#   Assumes dtm cognitive states only have one learning episode each
# 
import sys
import csv
import pydtm

def DTM_generateReport(dtm, csvfile):
    header = [ 'CS Index', 'CS Human Name' ]
    # Extract attributes and target attributes
    for attrib_idx in range(dtm.lespace.num_attributes):
        header.append(dtm.lespace.attribute_names[attrib_idx] + ' [Is Set?]')
        header.append(dtm.lespace.attribute_names[attrib_idx] + ' [Is Specified?]')
        for value_idx in range(len(dtm.lespace.attribute_vals[attrib_idx])):
            header.append(dtm.lespace.attribute_names[attrib_idx] + ' = ' + dtm.lespace.attribute_vals[attrib_idx][value_idx])
    for attrib_idx in range(dtm.lespace.num_attributes):
        header.append(dtm.lespace.attribute_names[attrib_idx] + ' [Is Target Set?]')
        header.append(dtm.lespace.attribute_names[attrib_idx] + ' [Is Target Specified?]')
        for value_idx in range(len(dtm.lespace.attribute_vals[attrib_idx])):
            header.append('[Target] ' + dtm.lespace.attribute_names[attrib_idx] + ' = ' + dtm.lespace.attribute_vals[attrib_idx][value_idx])
    header.append('Reward')
    
    table = list()
    table.append(header)    
    
    # Collect DTM rewards and eliminate redudancies 
    dtmcs_rewards = [0] * dtm.num_cs
    for idx in range(dtm.num_cs):
        dtmcs_rewards[idx] = list()
    num_triples = len(dtm.triples)
    for idx in range(num_triples):
        trip = dtm.triples[idx]
        dtmcs_rewards[trip.dest].append(trip.reward)
    #print('DTM CS Rewards --')
    for idx in range(dtm.num_cs):
    #    print('#'+str(idx), end=' ')
    #    print(dtmcs_rewards[idx])
        dtmcs_reward = set(dtmcs_rewards[idx])
        dtmcs_rewards[idx] = list(dtmcs_reward)
    
    
    # Loop through all DTM cs to extract 
    for cs_idx in range(dtm.num_cs):
        item = list()
    #    print('--------- #' + str(cs_idx))
        cs = dtm.states[cs_idx]
    #    print(cs)
    #    print(cs.les_set)
        les = [ i for i,x in enumerate(cs.les_set) if int(x) == 1 ]
    #    print('Learning episode -- ', end='')
    #    print(les)
        if len(les) == 0:
            print ("State <" + str(cs_idx) + "> has no learning episode set!")
            continue
        if len(les) > 1:
            print ("State <" + str(cs_idx) + "> has more than one learning episode set!")
            print (les)
            continue
        item.append(cs_idx)
        item.append(cs.csname)
    #    print(dtm.lespace.les[les[0]])
    #    le = dtm.lespace.les[les[0]][0:dtm.lespace.num_attributes]
        le = dtm.lespace.les[les[0]].attributes
        assgn = [-2] * dtm.lespace.num_attributes
        for key, value in le.items():
            assgn[int(key)] = int(value)
        for attrib_idx in range(dtm.lespace.num_attributes):
            if assgn[attrib_idx] == -2:
                item.append('No')
                item.append('No')
                for value_idx in range(len(dtm.lespace.attribute_vals[attrib_idx])):
                    item.append('0')
            else:
                item.append('Yes')
                if assgn[attrib_idx]  == -1:
                    item.append('No')
                    for value_idx in range(len(dtm.lespace.attribute_vals[attrib_idx])):
                        item.append('0')
                else:
                    item.append('Yes')
                    for value_idx in range(len(dtm.lespace.attribute_vals[attrib_idx])):
                        if value_idx == assgn[attrib_idx]:
                            item.append('1')
                        else:
                            item.append('0')
    
        le = dtm.lespace.les[les[0]].target_attributes
        assgn = [-2] * dtm.lespace.num_attributes
        for key, value in le.items():
            assgn[int(key)] = int(value)
        for attrib_idx in range(dtm.lespace.num_attributes):
            if assgn[attrib_idx] == -2:
                item.append('No')
                item.append('No')
                for value_idx in range(len(dtm.lespace.attribute_vals[attrib_idx])):
                    item.append('0')
            else:
                item.append('Yes')
                if assgn[attrib_idx]  == -1:
                    item.append('No')
                    for value_idx in range(len(dtm.lespace.attribute_vals[attrib_idx])):
                        item.append('0')
                else:
                    item.append('Yes')
                    for value_idx in range(len(dtm.lespace.attribute_vals[attrib_idx])):
                        if value_idx == assgn[attrib_idx]:
                            item.append('1')
                        else:
                            item.append('0')
    
    #    print(le.attributes)
        if len(dtmcs_rewards[cs_idx]) != 1:
            print ("State <" + str(cs_idx) + "> does not have a unique reward value!")
            print (dtmcs_rewards[cs_idx])
            item.append('(see triples)')
        else:
            item.append(dtmcs_rewards[cs_idx][0])
        table.append(item)
    
    #print actions dictionary
    header = [ 'Action Index', 'action' ]
    table.append(header)
    for key, value in dtm.actions_dictionary.items():
        action = list()
        action.append(str(value))
        action.append(str(key))
        table.append(action)

    # Loop through triples
    header = [ 'CS Source Index', 'Action', 'CS Destination Index', 'Weight', 'Probability', 'Reward' ]
    table.append(header)
    
    for triple_idx in range(num_triples):
        trip = list()
        trip.append(dtm.triples[triple_idx].source)
        trip.append(dtm.actions[dtm.triples[triple_idx].action].action_name.strip())
        trip.append(dtm.triples[triple_idx].dest)
        trip.append(dtm.triples[triple_idx].weight)
        trip.append(dtm.triples[triple_idx].prob)
        trip.append(dtm.triples[triple_idx].reward)
        table.append(trip)
    
    with open(csvfile, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(table)
