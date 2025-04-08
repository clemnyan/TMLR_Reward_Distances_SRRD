#!/usr/bin/python
# Alger linear IRL
# argv[1] is number of states
# argv[2] is number of actions
# argv[3] is the filename of the csv containing transition probabilities
# argv[4] is the filename of the csv containing the policies
# argv[5] is the discount factor
# argv[6] is the maximum rward value
# argv[7] is the l1 regularization value
# argv[8] is the filename of the csv to save the state reward values

import sys

num_states = int(sys.argv[1])
num_actions = int(sys.argv[2])
transition_fn = sys.argv[3]
pi_s_fn = sys.argv[4]
discount = float(sys.argv[5])
max_reward = float(sys.argv[6])
l1_regularization = float(sys.argv[7])
savefilename = sys.argv[8]

import csv
import numpy as np
import irl.linear_irl as linear_irl

new_list = list()
with open(pi_s_fn, 'r') as csvfile:
	reader = csv.reader(csvfile)
	csvlist = list(reader);

	_header1 = csvlist.pop(0)
	for _list in csvlist:
		for i in _list:
			new_list.append(int(i))

policy = np.array(new_list,dtype=int)
#print policy
	
transition_table = np.zeros((num_states, num_actions, num_states))
with open(transition_fn, 'r') as csvfile:
	reader = csv.reader(csvfile)
	csvlist = list(reader);

	_header2 = csvlist.pop(0)
	for _list in csvlist:
		transition_table[int(_list[0])][int(_list[1])][int(_list[2])] = float(_list[3])
		
	#print transition_table

rewards = linear_irl.irl(num_states, num_actions, transition_table, policy, discount, max_reward, l1_regularization)

_temp = list()
_temp.append ("Reward_s")
rewards_list = list()
rewards_list.append(_temp)

for i in rewards:
	_temp = list()
	_temp.append(i);
	rewards_list.append(_temp)

#print rewards_list

with open(savefilename, 'w') as savefile:
	writer = csv.writer(savefile);
	writer.writerows(rewards_list);
