#!/usr/bin/python
# Alger maxent IRL
# argv[1] is number of states
# argv[2] is number of actions
# argv[3] is the filename of the csv containing the feature matrix
# argv[4] is the discount factor
# argv[5] is the filename of the csv containing transition probabilities
# argv[6] is the filename of the csv containing trjectories
# argv[7] is the numver for epochs
# argv[8] is the learning rate
# argv[9] is the filename of the csv to save the state reward values
# argv[10] is the filename of the csv to save the theta values for each
#	feature  from maxent #	if == 'NULL' ignored
# argv[11] is the filename of the legend file that tells which feature
#	means what.

import sys

num_states = int(sys.argv[1])
num_actions = int(sys.argv[2])
feature_fn = sys.argv[3]
discount = float(sys.argv[4])
transition_fn = sys.argv[5]
trajectories_fn = sys.argv[6]
epochs = int(sys.argv[7])
learning_rate = float(sys.argv[8])
savefilename = sys.argv[9]
maxent_diag_filename = sys.argv[10]
maxent_variable_legend = sys.argv[11]
maxent_convergence_threshold = sys.argv[12]

import csv
import numpy as np
import irl.maxent as maxent

new_list = list()
with open(feature_fn, 'r') as csvfile:
	reader = csv.reader(csvfile)
	csvlist = list(reader)

	_header = csvlist.pop(0)
	for _list in csvlist:
		new_list.append(_list)

feature_matrix = np.array(new_list,dtype=int)
print feature_matrix
	
transition_table = np.zeros((num_states, num_actions, num_states))
with open(transition_fn, 'r') as csvfile:
	reader = csv.reader(csvfile)
	csvlist = list(reader)

	_header = csvlist.pop(0)
	for _list in csvlist:
		transition_table[int(_list[0])][int(_list[1])][int(_list[2])] = float(_list[3])
		
#print transition_table

trajectory_ = list()
with open(trajectories_fn, 'r') as csvfile:
	reader = csv.reader(csvfile)
	csvlist = list(reader)

	_header = csvlist.pop(0)
	for _list in csvlist:
		trajectory = list()
		while (len(_list) > 0):
			trajectory.append((int(_list[0]), int(_list[1]), int(_list[2])))
			_list = _list[3:]
		trajectory_.append(trajectory)		

trajectory_table = np.array(trajectory_)

print trajectory_table

rewards = maxent.irl(feature_matrix, num_actions, discount, transition_table, trajectory_table, epochs, learning_rate, maxent_diag_filename,maxent_variable_legend,maxent_convergence_threshold)

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
