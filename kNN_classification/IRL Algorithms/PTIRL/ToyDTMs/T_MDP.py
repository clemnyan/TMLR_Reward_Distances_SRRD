#
# Filename:     T_MDP.py
# Date:         2020-06-07
# Project:      State-splitting MDP
# Author:       Eugene Santos Jr.
# Copyright     Eugene Santos Jr.
#
#
import os
import copy
import parse
import csv
from math import floor

DEBUG = True

#
# Given a collection of training trajectories and poset, we determine a
#   feasible SsMDP solution with minimal tau > 1 (assiming not feasible for
#   tau == 1) and construct a new collection of trajectories that maps
#   the trajectories maintaining the poset.
#
# Notation:
#
#               -- "s_{}_<{}>--({},{})".format(i, k, a, b)
#                   -- the ath trajectory's bth state which is state i mapped
#                       to kth hidden value
#   ( For SsMDP, this is a binary value )
#
#   We assume that the answer dictionary variable key is in the above form.
#
def transformBySsMDP(num_states, all_trajectories, tau, answers, var_to_var_desc):
    # answers is a dictionary mapping L_MDP variables to a value.
    # No need to know the poset as long as the new trajectories order is
    #   maintained corresponding to all_trajectories.
    # Returns new_trajectories, state_map, state_map_inv, state_index_map
    #   (see below)

    new_state = num_states # The next available state
    state_map = dict() # Maps tuple of (i, a, b) to k from answers
    state_map_inv = dict() # Maps new state_index to i -- original state index
    state_index_map = dict() # Maps (i, k) to new state index
    for var_name, value in answers.items():
        var_name = var_to_var_desc[var_name]
        # Check to see if a match
        if DEBUG:
            print ('Examining {}...'.format(var_name))
        if value == 0: # Not set
            if DEBUG:
                print ("\t...not set.")
            continue
        p = parse.parse("s_{}_<{}>--({},{})", var_name, case_sensitive=True)
        if p == None: # No match
            continue
        try:
            i = int(p[0])
            k = int(p[1])
            a = int(p[2])
            b = int(p[3])
        except ValueError: # No match
            continue
        if i < 0 or i >= num_states or k < 0 or k >= tau or a < 0 or a >= len(all_trajectories) or b < 0 or 2 * b >= len(all_trajectories[a]): # No match
            continue
        if DEBUG:
            print ('{} --> i = {}, k={}, a={}, b={}'.format(var_name, i, k, a, b))
        try:
            state_map[(i, a, b)]
            print ('{} has already been mapped previously! Aborting.'.format(var_name))
            return (None)
        except KeyError:
            state_map[(i, a, b)] = k
            if DEBUG:
                print ('{} mapped to state subindex {}'.format(var_name, k))
        try:
            state_index_map[(i, k)]
        except KeyError: # First occurence
            state_index_map[(i, k)] = new_state
            state_map_inv[new_state] = i
            new_state += 1

    if DEBUG:
        print ('{} new states established.'.format(new_state - num_states))

    # Transforming
    new_trajectories = list()
    for a_idx, traj in enumerate(all_trajectories):
        new_traj = list()
        for pos in range(0, len(traj), 2):
            b_idx = floor(pos / 2)
            new_traj.append(state_index_map[(traj[pos], state_map[(traj[pos], a_idx, b_idx)])])
            if pos < len(traj) - 1:
                new_traj.append(traj[pos + 1])
        new_trajectories.append(new_traj)

    return (new_trajectories, state_map, state_map_inv, state_index_map)


#
# Constructs transformed training files in target directory
#
def transformedTrainingFilesFromSsMDP(action_specs, actions, state_specs, states, state_map, state_map_inv, state_index_map, new_trajectories, posets, poset_names, directory):

    # Build dictionaries for efficiency
    inv_actions = dict()
    for key, value in actions.items():
        inv_actions[value] = list(key)
    inv_states = dict()
    for key, value in states.items():
        inv_states[value] = list(key)

    # Setup directory
    if os.path.exists(directory):
        if not os.path.isdir(directory):
            sys.exit("transformedTrainingFilesFromSsMDP(...) -- {} is an existing file, cannot create as directory.".format(directory))
        print ("Using existing directory {} -- will overwrite files.".format(directory))
    else:
        os.mkdir(directory)
    cwd = os.getcwd()
    os.chdir(directory)

    # Make actions.csv rows
    actions_csv_rows = [ [ item for item in action_specs ] ]

    # Make attributes.csv rows
    new_state_specs = copy.deepcopy(state_specs)
    state_specs = set(state_specs)
    count = 0
    while "___k{}___".format(count) in state_specs:
        count += 1
    new_state_specs.append("___k{}___".format(count))
    attributes_csv_rows = [ [ item for item in new_state_specs ] ]

    # Make trajectory header row
    traj_header_row = [ ",".join(new_state_specs + action_specs) ]

    if poset_names == None:
        poset_names = list()
        for p_idx in range(len(posets)):
            poset_names.append("Set-{}".format(p_idx))

    for p_idx in range(len(posets)):
        if os.path.exists(poset_names[p_idx]):
            if not os.path.isdir(poset_names[p_idx]):
                sys.exit("transformedTrainingFilesFromSsMDP(...) -- {} is an existing file, cannot create as directory.".format(poset_names[p_idx]))
            print ("Using existing directory {} -- will overwrite files.".format(poset_names[p_idx]))
        else:
            os.mkdir(poset_names[p_idx])
        subcwd = os.getcwd()
        os.chdir(poset_names[p_idx])

        # Build actions.csv
        with open("actions.csv", 'w') as fn:
            writer = csv.writer(fn)
            writer.writerows(actions_csv_rows)
            fn.close()

        # Build attributes.csv
        with open("attributes.csv", 'w') as fn:
            writer = csv.writer(fn)
            writer.writerows(attributes_csv_rows)
            fn.close()

        # Build list.files
        with open("list.files", 'w') as list_fn:

            # Build trajectory
            for t_idx in posets[p_idx]:
                traj_fn = os.path.abspath("trajectory-{}.csv".format(t_idx))
                list_fn.write("{}\n".format(traj_fn)) # Save to list.files

                rows = list()
                rows.append(traj_header_row)

                # Build rows for states + actions
                traj = new_trajectories[t_idx]
                for pos in range(0, len(traj), 2):
                    new_s_idx = traj[pos] # new state
                    orig_s_idx = state_map_inv[new_s_idx]
                    k = state_map[(orig_s_idx, t_idx, floor(pos/2))]
                    # Check for correctness
                    if state_index_map[(orig_s_idx, k)] != new_s_idx:
                        print ('Mismatch: Expected new state index {}, instead got {} -- aborting rest of trajectory {} in {}.'.format(state_index_map[(orig_s_idx, k)], new_s_idx, t_idx, poset_names[p_idx]))
                        break
                    if pos < len(traj) - 2:
                        row = inv_states[orig_s_idx] + [ k ] + inv_actions[traj[pos + 1]]
                    else:
                        row = inv_states[orig_s_idx] + [ k ] + [ "None" for x in range(len(action_specs))]
                    rows.append(row)

                # Save trajectory
                with open(traj_fn, 'w') as fn:
                    writer = csv.writer(fn)
                    writer.writerows(rows)
                    fn.close()
            list_fn.close()

        os.chdir(subcwd)

    # Go back.
    os.chdir(cwd)
    del inv_actions
    del inv_states
    del rows
