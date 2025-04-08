#!/usr/bin/python3
#
# Name: dtm_irl.py
# Author: Eugene Santos Jr.
# Date: 2017-11-12
# Project: DTM
# Copyright: Eugene Santos Jr.
#
import pydtm
import copy
import datetime
import dtm_neighborhood
import cvxopt
import csv
from cvxopt import matrix, solvers, spmatrix, spdiag, log
import sys
from numpy import array
import statistics
from scipy import stats
import math
from dtm_generateReport import DTM_generateReport
import dtm_states
from itertools import combinations
import numpy

_debug = True # Debug flag for extra diagnosis.

# Changed prod sum expected value to:

# Let p be the trajectory probability, i.e, product of the triples probabilities in trajectory
# Let q be the discount factor
# prod_sum = E(trajectory) = sum_{i=0]^k p * q^k * r_i where
# r_i is the associated reward for the ith triple.
# Thus, we are weighted by the discount factor. If 0 < q < 1, then the earlier terms have higher
# weighting. If q > 1, then the later terms are weighted more.


# WARNING!!! Does not consider target attributes

# Takes a filename and adds '\' for unix filenames
def addBackSlash(filename):
    newfilename = filename.replace(" ", "\\ ")
    newfilename = newfilename.replace("=", "\\=")
    newfilename = newfilename.replace(",", "\\,")
    newfilename = newfilename.replace("[", "\\[")
    newfilename = newfilename.replace("]", "\\]")
    return(newfilename)

# Multiple actions are concatenated together when more than one field
#   is an action
def DTM_convertTrajectoryCSV(dtm, csvfile, action_headers):
# Returns a sequence of s, a, ...
#    print (csvfile)
#    csvfile = addBackSlash(csvfile)
    trajectory = list()
    with open(csvfile, 'r') as traj:
        reader = csv.reader(traj)
        lines = list(reader)
        header = lines.pop(0)
        num_fields = len(header)
        attributes_mapping = [-1] * len(header)
        actions_mapping = [-1] * len(header)
        for idx, h in enumerate(header):
            if h in action_headers:
                actions_mapping[idx] = action_headers.index(h)
            match = [ idx for idx, attrib in enumerate(dtm.lespace.attribute_names) if attrib == h ]
            if len(match) > 0: # This is an attribute heading
                if len(match) >= 2:
                    sys.exit(str(len(match)) + ' attribute matches. Can only be 1 per field! Matching first one only.')
                attributes_mapping[idx] = match[0]
            if actions_mapping[idx] == -1 and attributes_mapping[idx] == -1:
                print('Field header <' + h + '> does not match known action or attribute in dtm. Ignoring.')

        # order action fields by dtm index necessary for consistent concatenation
        actions_filter = sorted([ val for val in actions_mapping if val != -1 ])
        actions_sort = list()
        for val in actions_filter:
            actions_sort.append(actions_mapping.index(val))

        # process each line in the trajectory csv
        for i in range(len(lines)):
            line = lines[i]
            if len(line) != num_fields:
                print ('Line ' + str(i + 1) + ' field count inconsistent with header. Ignoring.')
                continue

            # Extract attributes into dictionary
            attributes_dict = {}
            for field_idx in range(num_fields):
                if attributes_mapping[field_idx] != -1:
                    value_idx = dtm.lespace.findAttributeValueIndex(attributes_mapping[field_idx], line[field_idx])
                    if value_idx == -1:
                        print ('Unkown attribute value <' + line[field_idx] + '> for attribute <' + header[field_idx] + '>. Ignoring.')
                        continue
                    attributes_dict[str(attributes_mapping[field_idx])] = str(value_idx)
            # Build action
            action = ''
            for idx in actions_sort:
                if line[idx] != '':
                    action += '<' + header[idx] + '=' + line[idx] + '>'
            if action == '':
                action = '__NOP__'
#            print ('Action #' + str(i + 1) + ' = ', end='')
#            print (action + ' (' + str(len(action)) + ')')
            find_action_idx = dtm.findActionIndex(action)
            if find_action_idx == -1:
                print ('Line ' + str(i + 1) + ' action ' + action + ' does not correspond to known action in dtm.')
                print (line)
                sys.exit()
                continue

            # Match against existing LE in dtm
            attributes_key = ''
            for key, value in sorted(attributes_dict.items()):
                attributes_key += '\"' + value + '\"'
            find_le_idx = dtm.lespace.getLE(attributes_key)
            if find_le_idx == -1:
                print ('Line ' + str(i + 1) + ' attributes do not correspond to known learning episode in dtm.')
                print (line)
                sys.exit()
                continue

            # identify the CS
            find_cs_idx = dtm.getCS(str(find_le_idx) + '+')
            if find_cs_idx == -1:
                print ('Line ' + str(i + 1) + ' attributes do not correspond to known cognitive state in dtm.')
                print (line)
                print ('\tLearning episode index = ' + str(find_le_idx))
                sys.exit()
                continue

            trajectory.append(find_cs_idx)
            trajectory.append(find_action_idx)
    return (trajectory[:-1]) # Delete last action


# Process file of trajectory files
# Multiple actions are concatenated together when more than one field
#   is an action
def DTM_loadTrajectoriesFile(dtm, trajectories_file, action_headers_csv):
# Returns a list of sequences of the form s, a, ...

    # Load in action headers
    with open(action_headers_csv, 'r') as action_f:
        action_read = csv.reader(action_f)
        action_headers = list(action_read)[0]

    trajectories = list()
    with open(trajectories_file, 'r') as trajs:
        lines = trajs.readlines()
        print (str(len(lines)) + ' trajectories...')
        for idx, line in enumerate(lines):
            if idx % max(int(len(lines) * .1), 1) == 0:
                print ('...' + str(int(idx / len(lines) * 100.0)) + '% trajectories completed.')
                print (datetime.datetime.now())
            if line[-1] == '\n':
                line = line[0:-1]
            trajectory = DTM_convertTrajectoryCSV(dtm, line, action_headers)
            if len(trajectory) > 0:
                trajectories.append(trajectory)
    return(trajectories)


# Prefix trajectories
# Given a list of trajectories, returns a tuple (N, X) where X is an array of trajectory prefixes and X[i] are prefixes of that contain i actions and ends with a state found in some trajectory
def DTM_prefixTrajectories(trajectories):
    N = max([ int(len(traj) / 2) for traj in trajectories ])
    X = [0] * N
    X[0] = list()
    for length in range(1,N):
        X[length] = list()
#        print ('Prefix length = ', end='')
#        print (length)
        for traj in trajectories:
#            print ('Trajecory = ', end='')
#            print (traj)
            if len(traj) >= length * 2:
                pref = copy.deepcopy(traj[:length*2+1])
#                print (pref)
                if pref in X[length]:
                    continue
                X[length].append(pref)
#        print (X[length])
    return ( (N, X) )


# Consecutively (uniquely) index all reward triples -- as long as a transition
#   exists in the DTM
def DTM_indexRewardTriples(dtm, graph, start_idx):
    # Returns ( start_idx, end_idx, rewards_idx, rewards_list )
    #
    #   rewards_idx is a 3-level diction to an index of form:
    #       rewards_idx[sidx][aidx][didx] = idx
    #   rewards_list is a list in order of index of form:
    #       ( ( sidx, aidx, didx ) )

    # Systematically order and initialize reward triples
    #   Only index a variable if triple (s, a, s') exists, i.e., there is a
    #   transition.
    print ('Building reward triples variables indices (start = ' + str(start_idx) + ')...')
    rewards_idx = {}
    rewards_list = list()
    count = start_idx
    for sidx, values1 in graph[0].items():
        rewards_idx[sidx] = {}
        for aidx, values2 in values1.items():
            rewards_idx[sidx][aidx] = {}
            for didx, tidx in values2.items():
                rewards_idx[sidx][aidx][didx] = count
                rewards_list.append( (sidx, aidx, didx) )
                count += 1

    print ('Number of reward triples with non-zero probability = ', end='')
    print (count - start_idx)
    return ( ( start_idx, count - 1, rewards_idx, rewards_list ) )


# Consecutively (uniquely) index all modPeak triples corresponding to reward triples -- as long as
#   a transition exists in the DTM
def DTM_indexModPeakTriples(dtm, graph, start_idx):
    # Returns ( start_idx, end_idx, modPeaks_idx, modPeaks_list )
    #
    #   modPeaks_idx is a 3-level diction to an index of form:
    #       modPeaks_idx[sidx][aidx][didx] = idx
    #   modPeaks_list is a list in order of index of form:
    #       ( ( sidx, aidx, didx ) )

    # Systematically order and initialize modPeak triples
    #   Only index a variable if triple (s, a, s') exists, i.e., there is a
    #   transition.
    print ('Building modPeak triples variables indices (start = ' + str(start_idx) + ')...')
    modPeaks_idx = {}
    modPeaks_list = list()
    count = start_idx

    for sidx, values1 in graph[0].items():
        modPeaks_idx[sidx] = {}
        for aidx, values2 in values1.items():
            modPeaks_idx[sidx][aidx] = {}
            for didx, tidx in values2.items():
                modPeaks_idx[sidx][aidx][didx] = count
                modPeaks_list.append( (sidx, aidx, didx) )
                count += 1

    print ('Number of modPeak triples with non-zero probability = ', end='')
    print (count - start_idx)
    return ( ( start_idx, count - 1, modPeaks_idx, modPeaks_list ) )


# Compute decision probabilities P(action|state) from trajectories
def DTM_computeDecisionProbabilities(dtm, graph, trajectories):
    # Returns a 2-level dictionary of the form:
    #   action_probs[dtm state index (source)][dtm action index]
    print ('Computing decision probabilities P(action|state) from trajectories ---')

    action_probs = {}
    # Initialize decision probabilities for all dtm edges -- necessary for computing rewards on
    # triples not used in trajectories
    for triple in dtm.triples:
        try:
            action_probs[triple.source]
        except KeyError:
            action_probs[triple.source] = {}
        action_probs[triple.source][triple.action] = 0.0

    for traj in trajectories:
        for pos in range(0, len(traj) - 1, 2):
            sidx = traj[pos]
            aidx = traj[pos + 1]
            didx = traj[pos + 2]
            try:
                triple_idx = graph[0][sidx][aidx][didx]
            except KeyError:
                triple_idx = -1
            assert triple_idx != -1, "No such triple in DTM for ( " + str(sidx) + ", " + str(aidx) + ", " + str(didx) + " )!"
            assert dtm.triples[triple_idx].prob != 0, "Transition probability in DTM is 0 for ( " + str(sidx) + ", " + str(aidx) + ", " + str(didx) + " )!"
            try:
                action_probs[sidx]
            except KeyError:
                action_probs[sidx] = {}
            try:
                action_probs[sidx][aidx]
            except KeyError:
                action_probs[sidx][aidx] = 0.0
            action_probs[sidx][aidx] += 1.0

    # Normalizing
    for sidx, value in action_probs.items():
        total = 0.0
        for aidx, weight in value.items():
            total += weight
        if total > 0.0:
            for aidx, weight in value.items():
                action_probs[sidx][aidx] = weight / total

    return ( action_probs )


# Consecutively (uniquely) index all decision probabilities
def DTM_indexDecisionProbabilities(decision_probs, start_idx):
    # Returns ( start_idx, end_idx, decision_probs_idx,
    #   decision_probs_list )
    #
    #   decision_probs_idx is a 2-level dictionary of the form:
    #       decision_probs_idx[sidx][aidx] = idx
    #   decision_probs_list is a list in order of index of form:
    #       ( ( aidx, sidx ) )
    print ('Building decision probabilities variable indices...')
    decision_probs_idx = {}
    decision_probs_list = list()
    count = start_idx
    for sidx, value in decision_probs.items():
        try:
            decision_probs_idx[sidx]
        except KeyError:
            decision_probs_idx[sidx] = {}
        for aidx, weight in value.items():
            decision_probs_idx[sidx][aidx] = count
            decision_probs_list.append( ( sidx, aidx ) )
            count += 1
    print ('Number of decision probability variable indices = ', end='')
    print (count - start_idx)
    return ( ( start_idx, count - 1, decision_probs_idx, decision_probs_list ) )


# Builds the trajectory equation from (1) based on traj_value_type (see below)
def DTM_addTrajectoryEquation(dtm, graph, reward_triples_idx, traj, traj_value_type, trajectory_value_idx, A_table_row_idx, discount_rate):
    # Returns a tuple of the form ( values, rows, columns, constant ) where
    #   values is a list of coefficients
    #   rows is a list of row positions (all have value A_table_row_idx)
    #   columns is a list of column positions
    #       Note - direct correspondence between the three lists for sparse
    #           matrices, i.e., matrix[rows[i], columns[i]] = values[i]
    #   constant is a single value for the right side constant of equation.
    values = list()
    rows = list()
    columns = list()
    constant = 0

    # Set the trajectory variable value
    #ESJ --- normlize by trajectory length?
#    values.append(-(len(traj) - 1.0)/2.0)
    values.append(-1.0)
    rows.append(A_table_row_idx)
    columns.append(trajectory_value_idx)
    coeffs = {}

    # Build rest of trajectory equation
    for traj_pos in range(0, len(traj) - 1, 2):
        sidx = traj[traj_pos]
        aidx = traj[traj_pos + 1]
        didx = traj[traj_pos + 2]
        try:
            triple_idx = graph[0][sidx][aidx][didx]
        except KeyError:
            triple_idx = -1
        assert triple_idx != -1, "No such reward triple for ( " + str(sidx) + ", " + str(aidx) + ", " + str(didx) + " )!"
        assert reward_triples_idx[sidx][aidx][didx] != -1, "No such reward triple for ( " + str(sidx) + ", " + str(aidx) + ", " + str(didx) + " )!"

        try:
            coeffs[( sidx, aidx, didx )]
        except KeyError:
            coeffs[( sidx, aidx, didx )] = 0.0
        if traj_value_type == 0:
            coeffs[( sidx, aidx, didx )] += 1.0
        elif traj_value_type == 1:
            coeffs[( sidx, aidx, didx) ] += dtm.triples[triple_idx].prob
        elif traj_value_type == 2:
            coeffs[( sidx, aidx, didx )] += dtm.triples[triple_idx].prob * pow(discount_rate, traj_pos / 2)
        else:
            sys.exit('DTM_addTrajectoryEquation(...) - imvalid traj_value_type ( ' + str(traj_value_type) + ' )')

    # Set the coefficients
    for ( sidx, aidx, didx ), coeff in coeffs.items():
        values.append(coeff)
        rows.append(A_table_row_idx)
        columns.append(reward_triples_idx[sidx][aidx][didx])

    return ( ( values, rows, columns, constant) )

def DTM_addTrajectoryInequality(dtm, graph, reward_triples_idx, traj, traj_value_type, trajectory_bound_idx, is_upper, G_table_row_idx, discount_rate):
    # Returns a tuple of the form ( values, rows, columns, constant ) where
    #   values is a list of coefficients
    #   rows is a list of row positions (all have value A_table_row_idx)
    #   columns is a list of column positions
    #       Note - direct correspondence between the three lists for sparse
    #           matrices, i.e., matrix[rows[i], columns[i]] = values[i]
    #   constant is a single value for the right side constant of equation.
    #   trajectory_bound_idx is the variable for the bounding
    #   is_upper is True means trajectory value <= trajectory_bound_idx
    #       otherwise trajectory value >= trajectory_bound_idx
    values = list()
    rows = list()
    columns = list()
    constant = 0
    if is_upper:
        mod = 1.0
    else:
        mod = -1.0

    # Set the trajectory variable value
    #ESJ --- normlize by trajectory length?
#    values.append(-(len(traj) - 1.0)/2.0)
    values.append(-mod)
    rows.append(G_table_row_idx)
    columns.append(trajectory_bound_idx)
    coeffs = {}

    # Build rest of trajectory equation
    for traj_pos in range(0, len(traj) - 1, 2):
        sidx = traj[traj_pos]
        aidx = traj[traj_pos + 1]
        didx = traj[traj_pos + 2]
        try:
            triple_idx = graph[0][sidx][aidx][didx]
        except KeyError:
            triple_idx = -1
        assert triple_idx != -1, "No such reward triple for ( " + str(sidx) + ", " + str(aidx) + ", " + str(didx) + " )!"
        assert reward_triples_idx[sidx][aidx][didx] != -1, "No such reward triple for ( " + str(sidx) + ", " + str(aidx) + ", " + str(didx) + " )!"

        try:
            coeffs[( sidx, aidx, didx )]
        except KeyError:
            coeffs[( sidx, aidx, didx )] = 0.0
        if traj_value_type == 0:
            coeffs[( sidx, aidx, didx )] += 1.0 * mod
        elif traj_value_type == 1:
            coeffs[( sidx, aidx, didx) ] += dtm.triples[triple_idx].prob * mod
        elif traj_value_type == 2:
            coeffs[( sidx, aidx, didx )] += dtm.triples[triple_idx].prob * pow(discount_rate, traj_pos / 2) * mod
        else:
            sys.exit('DTM_addTrajectoryEquation(...) - imvalid traj_value_type ( ' + str(traj_value_type) + ' )')

    # Set the coefficients
    for ( sidx, aidx, didx ), coeff in coeffs.items():
        values.append(coeff)
        rows.append(G_table_row_idx)
        columns.append(reward_triples_idx[sidx][aidx][didx])

    return ( ( values, rows, columns, constant) )

# For cvxopt.solvers.lp --
# Ax = b
# Gx <= h
# min cx
# must specifiy non-negativity explicitly


# Compute reward triples R(s, a, s') and sets values in dtm
# Inputs:
#   4 trajectory sets:
#       (a) Observed target trajectories (target_trajs)
#       (b) Target neighborhood trajectories (target_nbr_trajs)
#       (c) Non-target neighborhood trajectories (others_nbr_trajs)
#       (d) Observed non-target trajectories (others_trajs)
#   traj_value_type is a value corresponding to:
#       0 = simple reward sum (sum of triples)
#       1 = sum of product of transition probability and reward triple value
#       2 = sum of product of transition probability, reward triple value, and
#           discounted reward raised to number of transitions since start state
#       Should use 1 or 2 as principle of including the transition probability
#   trajectory_ids hashes tuple(trajectory) to ( position index, source )
def DTM_solveIRL_naive_BKB_others(dtm, graph, target_trajs, target_nbr_trajs, others_trajs, others_nbr_trajs, trajectory_ids, traj_value_type = 1, discount_rate = 1.0, peak_reward_magnitude = 100.0):
    # Return a report in the form of a list of tuples
    #   ( <report name>, csv-list ) each of which is to save each as a csv
    #   Also modifies dtm with new rewards and introduces new attribute
    #   to dtm -- decision_probs[s][a]

    # Optimization problem setup --
    #   Variables are indexed sequentially as follows:
    #       (a) Reward triples with non-zero transition probabilities --
    #           rewards[sidx][aidx][didx]
    #       (b) Decision probabilities for P(action | state) -- dP(A|S)
    #       (c) ubt_b, ubt_c, ubt_d -- maximum of trajectory values for
    #           sets (b), (c), and (d)
    #       (d) lbt_a, lbt_b, lbt_c -- minimum of trajectory values for
    #           sets (a), (b), and (c)
    #       (e) Magnitude bound variables for each change in decision
    #           probability -- uP(A|S)
    #       (f) Variable for modifying reward value -- modPeak[sidx][aidx][didx]
    #       (g) Magnitude bound variables for each modPeak[s][a][d] --
    #           umodPeak[s][a][d]
    #
    #   Constraints are built as follows:
    #       (1) For each trajectory ( s_1, a_1, s_2, a_2, s_3, a_3, ...,
    #           s_{n+1} ) in set (y)
    #            \sum_{i=1]^n discount^{i-1} * P(s_{i+1} | a_i, s_i) *
    #           reward_triple(s_i, a_i, s_{i+1}) <= ubt_y (except for y=a)
    #           -- depending on the traj_value_type
    #       (2) For each trajectory ( s_1, a_1, s_2, a_2, s_3, a_3, ...,
    #           s_{n+1} ) in set (y)
    #            \sum_{i=1]^n discount^{i-1} * P(s_{i+1} | a_i, s_i) *
    #           reward_triple(s_i, a_i, s_{i+1}) >= lbt_y (except for y=d)
    #           -- depending on the traj_value_type
    #       (3) UNUSED
    #       (4) The following constraints regarding trajectory ordering are
    #           lbt_a >= ubt_b
    #           lbt_b >= ubt_c
    #           lbt_c >= ubt_d
    #           ubt_b >= lbt_b
    #           ubt_c >= lbt_c
    #       Implementation note -- for efficiency, only need to do (a) >= (b), (b) >= (c), and
    #           (c) >= (d) since >= is transitive
    #
    #       The following constraints are for naive BKB formulation
    #       (5) For each reward triple:
    #           rewards[s][a][s'] = [ P(A|S) + dP(A|S) ] *
    #           peak_reward_magnitude + modPeak[s][a][s']
    #           where dP(A|S) is the change value for P(A|S)
    #           and modPeak[s][a][s'] is a special context sensitive modifier
    #       (6) For each decision probability P(A|S):
    #           P(A|S) + dP(A|S) >= 0
    #       (7) \sum_a [ P(A=a|S) + dP(A=a|S) ] = 1
    #           There is also an upper bound version of <= 1
    #       (8) dP(A|S) <= uP(A|S) -- magnitude bounds minimization
    #       (9) -dP(A|S) <= uP(A|S)
    #       (10) modPeak[s][a][s'] <= peak_reward_magnitude
    #       (11) -modPeak[s][a][s'] <= peak_reward_magnitude
    #       (12) modPeak[s][a][s'] <= umodPeak[s][a][s']--Magnitude minimization
    #       (13) -modPeak[s][a][s'] <= umodPeak[s][a][s']
    #
    #   Objective function:
    #       min decision_prob_weight * \sum{A,S} dP(A|S)
    #           - max_target_weight * lbt_a
    #           - traj_diff_weight *
    #               ( lbt_a - ubt_b + lbt_b - ubt_c + lbt_c - ubt_d)
    #               only include up to the last non-empty trajectory set
    #               (see below)
    #               otherwise, ubt_y could go to -inf since lbt_y can also
    #           + modPeak_weight * \sum{A,S} umodPeak(A|S)
    #       -- trade off minimizing change in decision probabilities with
    #           maximizing distance between trajectory types but also minimizing
    #           modPeak values
    #       -- weights are defined below in construction of objective
    #

    # Constraint configurations
    _disable_reward_constraint = False # (5)
    _disable_dP = False # (6) - (9)
    _disable_dP_equality_constraint_7 = True # (7) =
    _disable_dP_inequality_constraint_7 = False # (7) <=
    _disable_modPeak_max_magnitude = False # (10) - (11)
    _disable_modPeak_magnitude = False # (12) - (13)
    _enable_modPeak_zero = False # Force modPeaks to 0
    _enable_dP_zero = False # Force dPs to 0
    _disable_umodPeak_weight = False # Objective weight for umodPeak forced to 0.if True
    _disable_decisionProbWeight = False # Objective weight for decision_prob_weight forced to 0 if True
    _disable_maxTargetWeight = False # Objective weight for maximizing target trajectory values set to 0 if True


    # Based on (5) & (10), this is the range of possible reward values
    rewards_range = ( -peak_reward_magnitude, 2 * peak_reward_magnitude )

    print ('Initializing optimization problem...')
    num_variables = 0 # corresponds to number of columns or rows in A, b, G, h, c as appropriate
    A_table_value = list() # Storing in three lists corresponding to coefficient, row, column for sparse matrices for A
    A_table_row = list()
    A_table_column = list()
    A_table_num_constraints = 0 # How many constraints in A
    b_constants = list() # Corresponds to the constant on each constraint for A

    G_table_value = list() # Storing in three lists corresponding to coefficient, row, column for sparse matrices for G
    G_table_row = list()
    G_table_column = list()
    G_table_num_constraints = 0 # How many constraints in G
    h_constants = list() # Corresponds to the constant on each constraint for G

    c_vector = list() # cost coefficients for minimization


    # Build reward triples variable indices
    print (datetime.datetime.now())
    print ('Building reward triples...')
    ( reward_triples_start_idx, reward_triples_end_idx, reward_triples_idx, reward_triples_list ) = DTM_indexRewardTriples(dtm, graph, num_variables)
    num_variables = reward_triples_end_idx + 1
    print ('Building modPeak triples...')
    ( modPeak_triples_start_idx, modPeak_triples_end_idx, modPeak_triples_idx, modPeak_triples_list ) = DTM_indexModPeakTriples(dtm, graph, num_variables)
    num_variables = modPeak_triples_end_idx + 1
    print (datetime.datetime.now())

    # Build trajectory bounds variables
    print ('Buildiong trajectory bounds variables...')
    print ('\t...lbt_a to lbt_c...')
    lbt_a_idx = num_variables
    lbt_b_idx = num_variables + 1
    lbt_c_idx = num_variables + 2
    print ('\t...ubt_b to ubt_d...')
    ubt_b_idx = num_variables + 3
    ubt_c_idx = num_variables + 4
    ubt_d_idx = num_variables + 5
    num_variables += 6
    print (datetime.datetime.now())

    # Trajectory lists
    traj_lists_names = [ "Target trajectories", "Target neighboring trajectories", "Others neighboring trajectories", "Others trajectories" ]
        # Corresponds to a, b, c, and d
    traj_lists = [ target_trajs, target_nbr_trajs, others_nbr_trajs, others_trajs ]


    max_trajectory_triples = 0 # The longest trajectory length in terms of triples
    min_trajectory_triples = None # The shortest trajectory length in terms of triples
    print ('Building trajectories inequalities...')
    # Trajectory inequalities (1) and (2)
    trajectories_inequalities_start_idx = G_table_num_constraints
    for traj_idx, traj_list in enumerate(traj_lists):
        for tidx, traj in enumerate(traj_list): # For length stats
            if max_trajectory_triples < math.floor(len(traj) / 2):
                max_trajectory_triples = math.floor(len(traj) / 2)
            if min_trajectory_triples == None or min_trajectory_triples > math.floor(len(traj) / 2) - 1:
                min_trajectory_triples = math.floor(len(traj) / 2) - 1

            if traj_idx != 3: # Except (d) for lower bound
                ( values, rows, columns, constant ) = DTM_addTrajectoryInequality(dtm, graph, reward_triples_idx, traj, traj_value_type, traj_idx + lbt_a_idx, False, G_table_num_constraints, discount_rate)
                G_table_value.extend(values)
                G_table_row.extend(rows)
                G_table_column.extend(columns)
                h_constants.append(constant)
                G_table_num_constraints += 1
            if traj_idx != 0: # Except (a) for upper bound
                ( values, rows, columns, constant ) = DTM_addTrajectoryInequality(dtm, graph, reward_triples_idx, traj, traj_value_type, traj_idx + ubt_b_idx - 1, True, G_table_num_constraints, discount_rate)
                G_table_value.extend(values)
                G_table_row.extend(rows)
                G_table_column.extend(columns)
                h_constants.append(constant)
                G_table_num_constraints += 1
    trajectories_inequalities_end_idx = G_table_num_constraints - 1

    print ('Total number of trajectory inequalities is ' + str(trajectories_inequalities_end_idx - trajectories_inequalities_start_idx + 1))
    print (datetime.datetime.now())


    # Creating trajectory set ordering constraints (4)
    #           lbt_a >= ubt_b
    #           lbt_b >= ubt_c
    #           lbt_c >= ubt_d
    #           ubt_b >= lbt_b
    #           ubt_c >= lbt_c
    print ('Creating trajectory ordering constraints...')
    trajectories_ordering_constraints_start_idx = G_table_num_constraints
    trajectories_ordering_constraints_end_idx = G_table_num_constraints + 2

    G_table_value.append(1.0)
    G_table_row.append(G_table_num_constraints)
    G_table_column.append(ubt_b_idx)
    G_table_value.append(-1.0)
    G_table_row.append(G_table_num_constraints)
    G_table_column.append(lbt_a_idx)
    h_constants.append(0.0)
    G_table_num_constraints += 1

    G_table_value.append(1.0)
    G_table_row.append(G_table_num_constraints)
    G_table_column.append(ubt_c_idx)
    G_table_value.append(-1.0)
    G_table_row.append(G_table_num_constraints)
    G_table_column.append(lbt_b_idx)
    h_constants.append(0.0)
    G_table_num_constraints += 1

    G_table_value.append(1.0)
    G_table_row.append(G_table_num_constraints)
    G_table_column.append(ubt_d_idx)
    G_table_value.append(-1.0)
    G_table_row.append(G_table_num_constraints)
    G_table_column.append(lbt_c_idx)
    h_constants.append(0.0)
    G_table_num_constraints += 1

    G_table_value.append(1.0)
    G_table_row.append(G_table_num_constraints)
    G_table_column.append(lbt_b_idx)
    G_table_value.append(-1.0)
    G_table_row.append(G_table_num_constraints)
    G_table_column.append(ubt_b_idx)
    h_constants.append(0.0)
    G_table_num_constraints += 1

    G_table_value.append(1.0)
    G_table_row.append(G_table_num_constraints)
    G_table_column.append(lbt_c_idx)
    G_table_value.append(-1.0)
    G_table_row.append(G_table_num_constraints)
    G_table_column.append(ubt_c_idx)
    h_constants.append(0.0)
    G_table_num_constraints += 1

    print (datetime.datetime.now())


    # Creating constraints and variables for (5) - (9)
    # Build P(action | state) values and dP(action | state) variable indices
    print (datetime.datetime.now())
    decision_probs = DTM_computeDecisionProbabilities(dtm, graph, target_trajs)
    ( decision_probs_start_idx, decision_probs_end_idx, decision_probs_idx, decision_probs_list ) = DTM_indexDecisionProbabilities(decision_probs, num_variables) # dP(A|S) variables
    num_variables = decision_probs_end_idx + 1
    print (datetime.datetime.now())

    print ('Building reward triple constraints and dP(A|S) constraints...')
    #       (5) For each reward triple:
    #           rewards[s][a][s'] = [ P(A|S) + dP(A|S) ] *
    #           peak_reward_magnitude + modPeak[s][a][s']
    #           where dP(A|S) is the change value for P(A|S)
    #           and modPeak[s][a][s'] is a special context sensitive modifier
    if not _disable_reward_constraint:
        rewards_constraints_start_idx = A_table_num_constraints
        for tidx, triple in enumerate(dtm.triples):
            sidx = triple.source
            aidx = triple.action
            didx = triple.dest
            A_table_value.append(1.0)
            A_table_row.append(A_table_num_constraints)
            A_table_column.append(reward_triples_idx[sidx][aidx][didx])
            coeff = -peak_reward_magnitude
            A_table_value.append(coeff)
            A_table_row.append(A_table_num_constraints)
            A_table_column.append(decision_probs_idx[sidx][aidx])
            A_table_value.append(-1.0)
            A_table_row.append(A_table_num_constraints)
            A_table_column.append(modPeak_triples_idx[sidx][aidx][didx])
            b_constants.append(decision_probs[sidx][aidx] * peak_reward_magnitude)
            A_table_num_constraints += 1

        rewards_constraints_end_idx = A_table_num_constraints - 1
        print ('Total number of reward triple constraints is ', end='')
        print (rewards_constraints_end_idx - rewards_constraints_start_idx + 1)
        print (datetime.datetime.now())


    print ('Building magnitude minimization constraints and variables for dP(A|S) as well as dP(A|S) value constraint...')

#    magnitude_idx = num_variables [Deprecated]
    magnitudes_start_idx = num_variables
    magnitudes_end_idx = num_variables + decision_probs_end_idx - decision_probs_start_idx
    magnitudes_idx = {}
    magnitudes_list = [0] * (magnitudes_end_idx - magnitudes_start_idx + 1)

    magnitudes_constraints_start_idx = G_table_num_constraints
    for sidx, values in decision_probs_idx.items():

        try:
            magnitudes_idx[sidx]
        except KeyError:
            magnitudes_idx[sidx] = {}

        # Begin making constraint (7)
        #       (7) \sum_a [ P(A=a|S) + dP(A=a|S) ] = 1
        if not _disable_dP and not _disable_dP_equality_constraint_7:
            constraint_7_idx = A_table_num_constraints
            A_table_num_constraints += 1
            constraint_7_constant = 1.0
            b_constants.append("???") # Placeholder
        if not _disable_dP and not _disable_dP_inequality_constraint_7:
            constraint_7_idx = G_table_num_constraints
            G_table_num_constraints += 1
            constraint_7_constant = 1.0
            h_constants.append("???") # Placeholder

        for aidx, value in values.items():
            # Make constraint (6):
            #       (6) For each decision probability P(A|S):
            #           P(A|S) + dP(A|S) >= 0
            if not _disable_dP:
                G_table_value.append(-1.0)
                G_table_row.append(G_table_num_constraints)
                G_table_column.append(value)
                h_constants.append(decision_probs[sidx][aidx])
                G_table_num_constraints += 1

            # Make uP(A|S) variables
            idx = magnitudes_start_idx + value - decision_probs_start_idx
            magnitudes_idx[sidx][aidx] = idx
            magnitudes_list[idx - magnitudes_start_idx] = ( sidx, aidx )

            # Make constraint (8)
            #       (8) dP(A|S) <= uP(A|S) -- magnitude bounds minimization
            if not _disable_dP:
                G_table_value.append(1.0)
                G_table_row.append(G_table_num_constraints)
                G_table_column.append(value)
                G_table_value.append(-1.0)
                G_table_row.append(G_table_num_constraints)
#            G_table_column.append(magnitude_idx) [Deprecated]
                G_table_column.append(idx)
                h_constants.append(0.0)
                G_table_num_constraints += 1

            # Make constraint (9)
            #       (9) -dP(A|S) <= uP(A|S)
            if not _disable_dP:
                G_table_value.append(-1.0)
                G_table_row.append(G_table_num_constraints)
                G_table_column.append(value)
                G_table_value.append(-1.0)
                G_table_row.append(G_table_num_constraints)
#                G_table_column.append(magnitude_idx) [Deprecated]
                G_table_column.append(idx)
                h_constants.append(0.0)
                G_table_num_constraints += 1

            # Construct for constraint (7)
            if not _disable_dP and not _disable_dP_equality_constraint_7:
                constraint_7_constant -= decision_probs[sidx][aidx]
                A_table_value.append(1.0)
                A_table_row.append(constraint_7_idx)
                A_table_column.append(decision_probs_idx[sidx][aidx])
            if not _disable_dP and not _disable_dP_inequality_constraint_7:
                constraint_7_constant -= decision_probs[sidx][aidx]
                G_table_value.append(1.0)
                G_table_row.append(constraint_7_idx)
                G_table_column.append(decision_probs_idx[sidx][aidx])

        # Complete constraint (7)
        if not _disable_dP and not _disable_dP_equality_constraint_7:
            b_constants[constraint_7_idx] = constraint_7_constant
        if not _disable_dP and not _disable_dP_inequality_constraint_7:
            h_constants[constraint_7_idx] = constraint_7_constant


#    num_variables = magnitude_idx + 1 [Deprecated]
    num_variables = magnitudes_end_idx + 1
    magnitudes_constraints_end_idx = G_table_num_constraints - 1
    print (datetime.datetime.now())


    # Building modPeak[s][a][s'] <= 0 constraints (10) to (13)
    #       (10) modPeak[s][a][s'] <= peak_reward_magnitude
    #       (11) -modPeak[s][a][s'] <= peak_reward_magnitude
    #       (12) modPeak[s][a][s'] <= umodPeak # Magnitude minimization
    #       (13) -modPeak[s][a][s'] <= umodPeak # Magnitude minimization
    print ('Building modPeak[s][a][s\'] non-positive constraint...')

    # Force all modPeak to zero
    if _enable_modPeak_zero:
        modPeak_zero_constraints_start_idx = A_table_num_constraints
        for idx in range(modPeak_triples_start_idx, modPeak_triples_end_idx + 1):
            A_table_value.append(1.0)
            A_table_row.append(A_table_num_constraints)
            A_table_column.append(idx)
            b_constants.append(0.0)
            A_table_num_constraints += 1
        modPeak_zero_constraints_end_idx = A_table_num_constraints - 1


    umodPeak_magnitudes_start_idx = num_variables
    umodPeak_magnitudes_end_idx = num_variables + modPeak_triples_end_idx - modPeak_triples_start_idx
    num_variables = umodPeak_magnitudes_end_idx + 1
    modPeak_constraints_start_idx = G_table_num_constraints
    for idx in range(modPeak_triples_start_idx, modPeak_triples_end_idx + 1):
        ( sidx, aidx, didx) = modPeak_triples_list[idx - modPeak_triples_start_idx]

        if not _disable_modPeak_max_magnitude:
            # Constraint (10)
            G_table_value.append(1.0)
            G_table_row.append(G_table_num_constraints)
            G_table_column.append(idx)
            h_constants.append(peak_reward_magnitude)
            G_table_num_constraints += 1

            # Constraint (11)
            G_table_value.append(-1.0)
            G_table_row.append(G_table_num_constraints)
            G_table_column.append(idx)
            h_constants.append(peak_reward_magnitude)
            G_table_num_constraints += 1

        if not _disable_modPeak_magnitude:
            # Constraint (12)
            G_table_value.append(1.0)
            G_table_row.append(G_table_num_constraints)
            G_table_column.append(idx)
            G_table_value.append(-1.0)
            G_table_row.append(G_table_num_constraints)
            G_table_column.append(umodPeak_magnitudes_start_idx + idx - modPeak_triples_start_idx)
            h_constants.append(0.0)
            G_table_num_constraints += 1

            # Constraint (13)
            G_table_value.append(-1.0)
            G_table_row.append(G_table_num_constraints)
            G_table_column.append(idx)
            G_table_value.append(-1.0)
            G_table_row.append(G_table_num_constraints)
            G_table_column.append(umodPeak_magnitudes_start_idx + idx - modPeak_triples_start_idx)
            h_constants.append(0.0)
            G_table_num_constraints += 1

    modPeak_constraints_end_idx = G_table_num_constraints - 1
    print (datetime.datetime.now())


    # The following forces dP(A|S) to 0
    if _enable_dP_zero:
        dP_zero_constraints_start_idx = A_table_num_constraints
        for sidx, values in decision_probs_idx.items():
            for aidx, idx in values.items():
                A_table_value.append(1.0)
                A_table_row.append(A_table_num_constraints)
                A_table_column.append(idx)
                b_constants.append(0.0)
                A_table_num_constraints += 1
        dP_zero_constraints_end_idx = A_table_num_constraints - 1

#    A_table_value.append(1.0)
#    A_table_row.append(A_table_num_constraints)
#    A_table_column.append(magnitude_idx)
#    b_constants.append(0.0)
#    A_table_num_constraints += 1


    print ('Building minimization objective function...')
    c_vector = [ 0.0 ] * num_variables

    # Count number of variables used in objective function
    total_magnitude_vars = magnitudes_end_idx - magnitudes_start_idx + 1
    total_umodPeak_vars = umodPeak_magnitudes_end_idx - umodPeak_magnitudes_start_idx + 1
    total_trajectory_difference_vars = 6
    total_vars = total_magnitude_vars + total_trajectory_difference_vars + total_umodPeak_vars

    print ('')
    print ('Total non-zero entries in objective function = ', end='')
    print (total_vars)
    print ('\t...uP(A|S) variables = ', end='')
    print (total_magnitude_vars)
    print ('\t...umodPeak[s][a][s\'] variables = ', end='')
    print (total_umodPeak_vars)


    # Coefficients for the difference variables
    if rewards_range[1] >= 0:
        max_trajectory_value = rewards_range[1] * max_trajectory_triples
    else:
        max_trajectory_value = rewards_range[1] * min_trajectory_triples
    if rewards_range[0] >= 0:
        min_trajectory_value = rewards_range[0] * min_trajectory_triples
    else:
        min_trajectory_value = rewards_range[0] * max_trajectory_triples
    max_trajectory_difference = max_trajectory_value - min_trajectory_value
    print ('')
    print ('Max number of triples found in trajectories = ', end='')
    print (max_trajectory_triples)
    print ('Min number of triples found in trajectories = ', end='')
    print (min_trajectory_triples)
    print ('Theoretical trajectory values --')
    print ('\t...maximum trajectory value = ', end='')
    print (max_trajectory_value)
    print ('\t...minimum trajectory value = ', end='')
    print (min_trajectory_value)
    print ('\t...maximum difference in trajectory value = ', end='')
    print (max_trajectory_difference)


    # Trajectory difference maximization
    #           - max_target_weight * lbt_a
    #           - traj_diff_weight *
    #               ( lbt_a - ubt_b + lbt_b - ubt_c + lbt_c - ubt_d)
    #               only include up to the last non-empty trajectory set
    #               (see below)
    print ('')
    print ('Theoretical maximum sum of trajectory difference values = ', end='')
    sum_max_differences = max_trajectory_difference * 3.0
    print (sum_max_differences)
    max_target_weight = 1.0
    differences_weight = 1.0
    sum_max_differences = 1.0
    print ('\t...target trajectory max value weight = ', end='')
    if _disable_maxTargetWeight:
        print ('..disabled..', end='')
        max_target_weight = 0
    print (-max_target_weight)
    print ('\t...trajectory difference value weight = ', end='')
    print (-differences_weight)
    for traj_idx, traj_list in reversed(list(enumerate(traj_lists))):
        if len(traj_list) > 0:
            print (traj_idx)
            if traj_idx == 0:
                c_vector[lbt_a_idx] -= differences_weight
                break
            for idx in range(1, traj_idx + 1):
                print (idx)
                c_vector[lbt_a_idx + idx - 1] -= differences_weight
                c_vector[ubt_b_idx + idx - 1] += differences_weight
            break

    c_vector[lbt_a_idx] -= max_target_weight

    # modPeak triples value minimization
    print ('')
    print ('Theoretical maximum sum of umodPeak values = ', end='')
    sum_max_umodPeak = peak_reward_magnitude * total_umodPeak_vars
    print (sum_max_umodPeak)
    umodPeak_weight = pow(10.0, math.floor(numpy.log10(differences_weight * sum_max_differences)) + 1)
    print ('\t...umodPeak value weight = ', end='')
    if _disable_umodPeak_weight:
        print ('..disabled..', end='')
        umodPeak_weight = 0
    else:
        umodPeak_weight = 1
    print (umodPeak_weight)
    for mP_idx in range(umodPeak_magnitudes_start_idx, umodPeak_magnitudes_end_idx + 1):
        c_vector[mP_idx] = umodPeak_weight

    # uP(A|S)
    print ('')
    print ('Theoretical sum of maximum uP(A|S) values = ', end='')
    sum_max_prob_weight = (magnitudes_end_idx - magnitudes_start_idx + 1) * 1.0
    print (sum_max_prob_weight)
    print ('\t...uP(A|S) weight = ', end='')
    if _disable_decisionProbWeight:
        print ('..disabled..', end='')
        decision_prob_weight = 0.0
    else:
        decision_prob_weight = pow(10.0, math.floor(numpy.log10(umodPeak_weight / sum_max_prob_weight * sum_max_umodPeak)))
    print (decision_prob_weight)
    for uidx in range(magnitudes_start_idx, magnitudes_end_idx + 1):
        c_vector[uidx] = decision_prob_weight

    print (datetime.datetime.now())


    # Diagnostics and building optimization

    if _debug:
        print ('Checking variable indexing uniqueness...')
        _variables = set()
        _variables_map = {}

#        if magnitude_idx in _variables:
#            print ('Magnitude variable for decision probability deltas overlaps with existing index!')
#            sys.exit(str(_variables_map[magnitude_idx]))
#        _variables.add(magnitude_idx)
#        _variables_map[magnitude_idx] = ( "P(A|S) delta magnitude variable" )

        print ('...ubt_{b, c, d}...')
        for count in range(3):
            desc = 'ubt_' + chr(ord('b') + count)
            if ubt_b_idx + count in _variables:
                print ('Trajectory upper bound variable for ' + traj_lists_names[count + 1] + ' overlaps with existing index!')
                sys.exit(str(_variables_map[ubt_b_idx + count]))
            _variables.add(ubt_b_idx + count)
            _variables_map[ubt_b_idx + count] = ( "Trajectory upper bound variable", desc )

        print ('...lbt_{a, b, c}...')
        for count in range(3):
            desc = 'lbt_' + chr(ord('a') + count)
            if lbt_a_idx + count in _variables:
                print ('Trajectory lower bound variable for ' + traj_lists_names[count] + ' overlaps with existing index!')
                sys.exit(str(_variables_map[lbt_a_idx + count]))
            _variables.add(lbt_a_idx + count)
            _variables_map[lbt_a_idx + count] = ( "Trajectory lower bound variable", desc )


        print ('...uP(A|S)...')
        for sidx, values in magnitudes_idx.items():
            for aidx, idx in values.items():
                desc = '( ' + str(aidx) + ' | ' + str(sidx) + ' )'
                if idx in _variables:
                    print ('Decision change magnitude bound variable ' + desc + ' overlaps with existing index!')
                    sys.exit(str(_variables_map[idx]))
                _variables.add(idx)
                _variables_map[idx] = ( "uP(A|S) magnitude variable", desc )


        print ('...umodPeak[s][a][s\']...')
        for mP_idx in range(umodPeak_magnitudes_start_idx, umodPeak_magnitudes_end_idx + 1):
            if mP_idx in _variables:
                print ('ModPeak magnitude variable for modPeak values overlaps with existing index!')
                sys.exit(str(_variables_map[mP_idx]))
            _variables.add(mP_idx)
            _variables_map[mP_idx] = ( "ModPeak magnitude variable", mP_idx )


        print ('...dP(A|S)...')
        for sidx, values in decision_probs_idx.items():
            for aidx, idx in values.items():
                desc = '( ' + str(aidx) + ' | ' + str(sidx) + ' )'
                if idx in _variables:
                    print ('Decision probability change variable ' + desc + ' overlaps with existing index!')
                    sys.exit(str(_variables_map[idx]))
                _variables.add(idx)
                _variables_map[idx] = ( "P(A|S) delta variable", desc )


        print ('...reward[s][a][s\']...')
        for sidx, values1 in reward_triples_idx.items():
            for aidx, values2 in values1.items():
                for didx, idx in values2.items():
                    if idx == -1:
                        continue
                    desc = '( ' + str(sidx) + ', ' + str(aidx) + ', ' + str(didx) + ' )'
                    if idx in _variables:
                        print ('Reward triple variable ' + desc + ' overlaps with existing index!')
                        sys.exit(str(_variables_map[idx]))
                    _variables.add(idx)
                    _variables_map[idx] = ( 'Reward triple variable', desc )


        print ('...modPeak[s][a][s\']...')
        for sidx, values1 in modPeak_triples_idx.items():
            for aidx, values2 in values1.items():
                for didx, idx in values2.items():
                    if idx == -1:
                        continue
                    desc = '( ' + str(sidx) + ', ' + str(aidx) + ', ' + str(didx) + ' )'
                    if idx in _variables:
                        print ('modPeak triple variable ' + desc + ' overlaps with existing index!')
                        sys.exit(str(_variables_map[idx]))
                    _variables.add(idx)
                    _variables_map[idx] = ( 'modPeak triple variable', desc )


        print ('Number of variable indices checked = ' + str(len(_variables)))
        print ('Number of variables = ' + str(num_variables))
        assert num_variables == len(_variables), "Number of variables does not match variables indices scanned!"
        print (datetime.datetime.now())

    print('Scaling constraint matrix...')
    #Scaling A Matrix
    rowIdx = A_table_row[0]
    scaleIndices = []
    for i in range(0, len(A_table_row)):
        # scale row
        if A_table_row[i] != rowIdx:
            maxRow = float('-inf')
            for scaleIdx in scaleIndices:
                if abs(A_table_value[scaleIdx]) > maxRow:
                    maxRow = abs(A_table_value[scaleIdx])
            for scaleIdx in scaleIndices:
                A_table_value[scaleIdx] /= maxRow
            b_constants[rowIdx] /= maxRow
            # set new row Idx
            rowIdx = A_table_row[i]
            scaleIndices.clear()
        scaleIndices.append(i)

    #Scaling G Matrix
    rowIdx = G_table_row[0]
    scaleIndices = []
    for i in range(0, len(G_table_row)):
        # scale row
        if G_table_row[i] != rowIdx:
            maxRow = float('-inf')
            for scaleIdx in scaleIndices:
                if abs(G_table_value[scaleIdx]) > maxRow:
                    maxRow = abs(G_table_value[scaleIdx])
            for scaleIdx in scaleIndices:
                G_table_value[scaleIdx] /= maxRow
            h_constants[rowIdx] /= maxRow
            # set new row Idx
            rowIdx = G_table_row[i]
            scaleIndices.clear()
        scaleIndices.append(i)
    print('Scaled')

    print ('Building sparse matrices for optimization...')
    print ('Total number of variables = ', end='')
    print (num_variables)
    print ('A_table_value # entries = ', end='')
    print (len(A_table_value))
    print ('A_table_row # entries = ', end='')
    print (len(A_table_row))
    print ('A_table_column # entries = ', end='')
    print (len(A_table_column))
    assert len(A_table_value) == len(A_table_row), "# entries must be identical!"
    assert len(A_table_value) == len(A_table_column), "# entires must be identical!"
    print ('A matrix number of constraints = ', end='')
    print (A_table_num_constraints)
    A = spmatrix(A_table_value, A_table_row, A_table_column, ( A_table_num_constraints, num_variables ))
    print ('b constants dimension = ', end='')
    print (len(b_constants))
    assert len(b_constants) == A_table_num_constraints, "Dimension must equal number of A matrix constraints!"
    b = matrix(b_constants)

    print ('Cost function coefficients dimension = ', end='')
    print (len(c_vector))
    assert len(c_vector) == num_variables, "Must equal to number of wariables!"
    c = matrix(c_vector)

    print ('G_table_value # entries = ', end='')
    print (len(G_table_value))
    print ('G_table_row # entries = ', end='')
    print (len(G_table_row))
    print ('G_table_column # entries = ', end='')
    print (len(G_table_column))
    assert len(G_table_value) == len(G_table_row), "# entries must be identical!"
    assert len(G_table_value) == len(G_table_column), "# entires must be identical!"
    print ('G matrix number of constraints = ', end='')
    print (G_table_num_constraints)
    G = spmatrix(G_table_value, G_table_row, G_table_column, ( G_table_num_constraints, num_variables ))
    print ('h constants dimension = ', end='')
    print (len(h_constants))
    assert len(h_constants) == G_table_num_constraints, "Dimension must equal number of G matrix constraints!"
    h = matrix(h_constants)
    print (datetime.datetime.now())

    # Optimizing
    print ('Executing optimization...')

    soln = solvers.lp(c, G, h, A, b, solver = 'glpk')
    print (datetime.datetime.now())


    # Extracting answers
    rewards = array(soln['x'])

    print ('Extraction optimization results ---')
    print ('')

    print ('Extracting rewards and updating DTM...')
    rw = list()
    orig_rw = list()
    dw = list()
    for sidx, values1 in graph[0].items():
        for aidx, values2 in values1.items():
            for didx, tidx in values2.items():
                if tidx != -1:
                    # Update dtm triple reward
                    triple = dtm.triples[tidx]
                    orig_rw.append(triple.reward)
                    triple.reward = rewards[reward_triples_idx[sidx][aidx][didx]][0]
                    rw.append(triple.reward)
                    dw.append(orig_rw[-1] - rw[-1])
#    print ('Magnitude delta for P(A|S) = ', end='') [Deprecated]
#    print (rewards[magnitude_idx][0]) [Deprecated]
    print ('--------------------Analysis---------------------------------')
    print ('')
    print ('Original rewards statistics:')
    print ('\t', end='')
    print (stats.describe(orig_rw))
    print ('')
    print ('New rewards statistics:')
    print ('\t', end='')
    print (stats.describe(rw))
    print ('')
    print ('Reward change statistics (original - new):')
    print ('\t', end='')
    print (stats.describe(dw))
    print ('')

    nP = list()
    for sidx, values1 in graph[0].items():
        for aidx, values2 in values1.items():
            for didx, tidx in values2.items():
                if tidx != -1:
                    nP.append(rewards[modPeak_triples_idx[sidx][aidx][didx]][0])
    print ('modPeak statistics:')
    print ('\t', end='')
    print (stats.describe(nP))
    print ('')


    oP = list()
    for sidx, values in decision_probs.items():
        for aidx, value in values.items():
            oP.append(value)
    print ('Original P(A|S) statistics:')
    print ('\t', end='')
    print (stats.describe(oP))
    print ('')

    new_oP = list()
    dP = list()
    for sidx, values in decision_probs_idx.items():
        for aidx, value in values.items():
            dP.append(rewards[value][0])
            new_oP.append(rewards[value][0] + decision_probs[sidx][aidx])
    print ('New P(A|S) statistics:')
    print ('\t', end='')
    print (stats.describe(new_oP))
    print ('')
    print ('delta P(A|S) statistics:')
    print ('\t', end='')
    print (stats.describe(dP))
    print ('')

    uP = list()
    for sidx, values in magnitudes_idx.items():
        for aidx, value in values.items():
            uP.append(rewards[value][0])
    print ('uP(A|S) statistics:')
    print ('\t', end='')
    print (stats.describe(uP))
    print ('')

    print ('-----------------------------------------------------')
    print ('')
    print (datetime.datetime.now())

#    input ('Continue?')

    print ('Extracting P(A|S) and modifying decision probabilities --')
    new_decision_probs_delta = {}
    dtm.decision_probs = {}
    for sidx in range(dtm.num_cs):
        try:
            decision_probs_idx[sidx]
            new_decision_probs_delta[sidx] = {}
            dtm.decision_probs[sidx] = {}
        except KeyError:
            continue
        for aidx in range(dtm.num_a):
            try:
                new_decision_probs_delta[sidx][aidx] = rewards[decision_probs_idx[sidx][aidx]][0]
            except KeyError:
                pass

    for sidx in range(dtm.num_cs):
        try:
            decision_probs_idx[sidx]
        except KeyError:
            continue
        for aidx in range(dtm.num_a):
            try:
                decision_probs[sidx][aidx]
            except KeyError:
                continue
            dtm.decision_probs[sidx][aidx] = decision_probs[sidx][aidx] + new_decision_probs_delta[sidx][aidx]
            if new_decision_probs_delta[sidx][aidx] == 0:
                continue
            print ('P( ' + str(aidx) + ' | ' + str(sidx) + ' ) = ', end='')
            print (decision_probs[sidx][aidx], end=' ')
            print ('-->', end=' ')
            print (decision_probs[sidx][aidx] + new_decision_probs_delta[sidx][aidx], end = '\t')
            print ('delta = ', end='')
            print (new_decision_probs_delta[sidx][aidx])
            if decision_probs[sidx][aidx] + new_decision_probs_delta[sidx][aidx] > 1:
                print ('')
                print ('++++++++++++++++++ERROR++++++')
                for a_idx in range(dtm.num_a):
                    try:
                        decision_probs[sidx][a_idx]
                    except KeyError:
                        continue
                    print ('P( ' + str(a_idx) + ' | ' + str(sidx) + ' ) = ', end='')
                    print (decision_probs[sidx][a_idx], end=' ')
                    print ('-->', end=' ')
                    print (decision_probs[sidx][a_idx] + new_decision_probs_delta[sidx][a_idx])
                print ('++++++++++++++++++ERROR++++++')
                print ('')

    print (datetime.datetime.now())

#    input ('Continue?')


    # Make reports
    print ('')
    print ('Report Generation -----------------------------')
    print ('')
    reports = list()
    print ('Outputing decision probabilities --')
    reports.append(['Decision probabilities --'])
    reports.append(['State index', 'Action index', 'P(A|S)'])
    for sidx in range(dtm.num_cs):
        try:
            dtm.decision_probs[sidx]
        except KeyError:
            continue
        for aidx in range(dtm.num_a):
            try:
                dtm.decision_probs[sidx][aidx]
            except KeyError:
                continue
            reports.append([ sidx, aidx, dtm.decision_probs[sidx][aidx] ])

    print ('Generating trajectory and neighborhood expected values --')
    reports.append([ 'Lower bound trajectory value for a =', rewards[lbt_a_idx][0] ])
    reports.append([ 'Upper bound trajectory value for b =', rewards[ubt_b_idx][0] ])
    reports.append([ 'Lower bound trajectory value for b =', rewards[lbt_b_idx][0] ])
    reports.append([ 'Upper bound trajectory value for c =', rewards[ubt_c_idx][0] ])
    reports.append([ 'Lower bound trajectory value for c =', rewards[lbt_c_idx][0] ])
    reports.append([ 'Upper bound trajectory value for d =', rewards[ubt_d_idx][0] ])
    print ('\tTrajectory probability does not include discount factor.')
    for traj_list, traj_list_name in zip(traj_lists, traj_lists_names):
        analysis = list()
        analysis.append([ 'Trajectory Id', 'Trajectory probability', 'Reward sum', 'Expected linear reward with discount', 'Trajectory probability x Reward sum', 'Trajectory' ])
        for traj_idx, trajectory in enumerate(traj_list):
            try:
                analysis_item = [ trajectory_ids[tuple(trajectory)] ]
            except KeyError:
                sys.exit('Cannot find trajectory <{}>!'.format(trajectory))
#            analysis_item = [ str(traj_idx + 1) ]
            prob = 1.0
            reward = 0
            linear_expected = 0
            discount = 1.0
            for idx in range(0, len(trajectory) - 1, 2):
                trip = dtm.triples[graph[0][trajectory[idx]][trajectory[idx + 1]][trajectory[idx + 2]]]
                prob *= trip.prob
                reward += trip.reward
                linear_expected += trip.reward * trip.prob * discount
                discount *= discount_rate
            analysis_item.append(prob)
            analysis_item.append(reward)
            analysis_item.append(linear_expected)
            analysis_item.append(prob * reward)
            analysis_item.append(trajectory)
            analysis.append(analysis_item)
        reports.append([ 'Report for trajectory list =', traj_list_name ])
        reports.extend(analysis)
    print (datetime.datetime.now())
    return(reports)


# Compute reward triples R(s, a, s') and sets values in dtm
def DTM_doIRL_naive_BKB(dtm, graph, trajectories, m_bound, discount_rate, peak, traj_prod_sum, traj_sum, incremental_prod_sum_aggregates, path_emphasis, merged_states, merged_percentage, nbr_branching, reward_diversity):
    print (datetime.datetime.now())
    action_probs = DTM_computeDecisionProbabilities(dtm, graph, trajectories)
    print (datetime.datetime.now())
    print ('Starting optimization problem for reward triples...')
#    # remove redundant trajectories
#    print ('Eliminating redundant trajectories...')
#    print ('\t' + str(len(trajectories)) + ' trajectories')
#    temp_trajs = list()
#    for traj in trajectories:
#        found = False
#        for traj2 in temp_trajs:
#            if traj == traj2:
#                found = True
#                break
#        if not found:
#            temp_trajs.append(traj)
#    trajectories = temp_trajs
#    print ('\t' + str(len(trajectories)) + ' unique trajectories')

    trajectories = dtm_neighborhood.DTM_makeTrajectoriesProperSuffixes(trajectories)
    print ('\tTotal trajectories + suffixes = ' + str(len(trajectories)))
    print (datetime.datetime.now())

    print ('Building optimization problem...')
    print (datetime.datetime.now())
    ( reward_start_idx, reward_end_idx, rewards_idx, rewards_list ) = DTM_indexRewardTriples(dtm, graph, 0)
    reward_vars = reward_end_idx + 1

    # Formulate constraints

    A_table_value = list() # Storing in three lists corresponding to value, row, column for sparse matrices
    A_table_row = list()
    A_table_column = list()
    A_table_num_constraints = 0 # How many constraints
    b_constants = list() # Corresponds to the constant on each constraint

    # Always start with trajectory vs neighbors constraints (equality constraint)
    neighborhoods = [0] * len(trajectories)
    neighbor_probability = [0] * len(trajectories)
    for traj_idx in range(len(trajectories)):
        neighbor_probability[traj_idx] = list()
    trajectory_probability = [-1] * len(trajectories)
    for traj_idx, trajectory in enumerate(trajectories):
        print ('Trajectory #' + str(traj_idx + 1), end=':')
#        print (trajectory, end = ' ')

        prob = 1.0
        for idx in range(0, len(trajectory) - 1, 2): # compute trajectory probability
            trip = dtm.triples[graph[0][trajectory[idx]][trajectory[idx + 1]][trajectory[idx + 2]]]

########################################################
            prob *= trip.prob # trajectory probability
########################################################

        trajectory_probability[traj_idx] = prob

        # only compute neighbors if flags require
        if traj_prod_sum or traj_sum or incremental_prod_sum_aggregates:
            if nbr_branching == True:
                neighbors = dtm_neighborhood.DTM_makeBranchingNeighborhood(dtm, graph, trajectory, trajectories)
            else:
                neighbors = dtm_neighborhood.DTM_makeNeighborhood(dtm, graph, trajectory, trajectories, m_bound)
        else:
            neighbors = list()

        print ('Nbrs = ' + str(len(neighbors)))
#$        row = [0.0] * reward_vars # here, we sum up the neighbors vs a normalized trajectory
        neighborhoods[traj_idx] = copy.deepcopy(neighbors)
        for nbr in neighbors:
#            print (nbr)
            if traj_prod_sum or traj_sum:
                row = [0.0] * reward_vars # if here, then we generate one trajectory vs nbr constraint each

            if traj_prod_sum: # Use prod sum
                discount = 1.0
                for idx in range(0, len(trajectory) - 1, 2): # set trajectory constraint coefficients
                    pos = rewards_idx[trajectory[idx]][trajectory[idx + 1]][trajectory[idx + 2]]

########################################################
                    row[pos] += -prob * discount # trajectory expected value term
########################################################

                    discount *= discount_rate

                prob = 1.0
                for idx in range(0, len(nbr) - 1, 2): # compute neighbor trajectory probability
                    trip = dtm.triples[graph[0][nbr[idx]][nbr[idx + 1]][nbr[idx + 2]]]

########################################################
                    prob *= trip.prob
########################################################

                neighbor_probability[traj_idx].append(prob)

                discount = 1.0
                for idx in range(0, len(nbr) - 1, 2): # set neighbor trajectory probability
                    pos = rewards_idx[nbr[idx]][nbr[idx + 1]][nbr[idx + 2]]

########################################################
                    row[pos] += prob * discount # neighbor expected value term
########################################################

                    discount *= discount_rate

#                if not (row in table): # Here is for individual contraints per neighbor
                if True:

########################################################
                    b_constants.append(0.0) # Constant offset
########################################################

                    for pos in range(reward_vars):
                        if row[pos] != 0:
                            A_table_value.append(row[pos])
                            A_table_row.append(A_table_num_constraints)
                            A_table_column.append(pos)
                    A_table_num_constraints += 1

            if traj_sum == True: # Use sum of products of prob x reward
                discount = 1.0
                prob = 1.0
                for idx in range(0, len(trajectory) - 1, 2): # compute trajectory probability
                    trip = dtm.triples[graph[0][trajectory[idx]][trajectory[idx + 1]][trajectory[idx + 2]]]
                    pos = rewards_idx[trajectory[idx]][trajectory[idx + 1]][trajectory[idx + 2]]
                    prob *= trip.prob

########################################################
                    row[pos] += -trip.prob * discount * trip.reward
########################################################

                    discount *= discount_rate


                discount = 1.0
                prob = 1.0
                for idx in range(0, len(nbr) - 1, 2): # compute neighbor trajectory probability
                    trip = dtm.triples[graph[0][nbr[idx]][nbr[idx + 1]][nbr[idx + 2]]]
                    pos = rewards_idx[nbr[idx]][nbr[idx + 1]][nbr[idx + 2]]
                    prob *= trip.prob

########################################################
                    row[pos] += trip.prob * discount * trip.reward
########################################################

                    discount *= discount_rate
                neighbor_probability[traj_idx].append(prob)

#                if not (row in table): # Here is for individual contraints per neighbor
                if True:

########################################################
                    b_constants.append(0.0) # Constant offset
########################################################

                    for pos in range(reward_vars):
                        if row[pos] != 0:
                            A_table_value.append(row[pos])
                            A_table_row.append(A_table_num_constraints)
                            A_table_column.append(pos)
                    A_table_num_constraints += 1


    if incremental_prod_sum_aggregates == True:
        print ('Constructing incremental product sum aggregates constraints...')
        print (datetime.datetime.now())
        for traj_idx, trajectory in enumerate(trajectories):
            print ('Trajectory #' + str(traj_idx + 1), end=':')
            print (trajectory, end = ' ')
            if neighborhoods[traj_idx] == 0:
                if nbr_branching == True:
                    neighbors = dtm_neighborhood.DTM_makeBranchingNeighborhood(dtm, graph, trajectory, trajectories)
                else:
                    neighbors = dtm_neighborhood.DTM_makeNeighborhood(dtm, graph, trajectory, trajectories, m_bound)
                neighborhoods[traj_idx] = copy.deepcopy(neighbors)
            else:
                neighbors = neighborhoods[traj_idx]
            print ('Nbrs = ' + str(len(neighbors)))

            if len(neighbor_probability[traj_idx]) != len(neighbors):
                neighbor_probability[traj_idx] = list()
                for nbr in neighbors:
                    prob = 1.0
                    for idx in range(0, len(nbr) - 1, 2): # compute neighbor trajectory probability
                        trip = dtm.triples[graph[0][nbr[idx]][nbr[idx + 1]][nbr[idx + 2]]]
                        prob *= trip.prob
                    neighbor_probability[traj_idx].append(prob)
            if len(neighbors) == 0:
                continue # nothing to make

            row = [0.0] * reward_vars
            traj_prob = trajectory_probability[traj_idx]
            for idx in range(0, len(trajectory) - 1, 2):
                pos = rewards_idx[trajectory[idx]][trajectory[idx + 1]][trajectory[idx + 2]]
                row[pos] -= traj_prob * math.pow(discount_rate, idx / 2) * len(neighbors)
                for nidx,nbr in enumerate(neighbors):
                    if idx + 2 >= len(nbr):
                       continue
                    pos = rewards_idx[nbr[idx]][nbr[idx + 1]][nbr[idx + 2]]
                    row[pos] += neighbor_probability[traj_idx][nidx] * math.pow(discount_rate, idx / 2)
                if idx == 0: # Skip a length of one aggregate
                    continue

                b_constants.append(0.0) # Constant offset
                for pos in range(reward_vars):
                    if row[pos] != 0:
                        A_table_value.append(row[pos])
                        A_table_row.append(A_table_num_constraints)
                        A_table_column.append(pos)
                A_table_num_constraints += 1

    print (datetime.datetime.now())
    A_num_nbr_constraints = A_table_num_constraints
    print ('Trajectory vs Neighbors constraints = ', end='')
    print (A_num_nbr_constraints)

    # Path emphasis basically takes each the prefix of each trajectory and
    #    makes the partial (prod prob) (sum rewards x discount) to be greater than any other paths in the MDP
    #    that have the same prefix except for the last action and state except for prefixes of
    #    other trajectories. Partial prod prob sum rewards include trajectories.
    if path_emphasis == True:
        print ('Constructing path emphasis constraints...')
        (N, prefixes) = DTM_prefixTrajectories(trajectories)
        for length in range(1, N):
            for pref in prefixes[length]:
                print ('Prefix = ', end='')
                print (pref)
                # Compute partial prod sum for pref
                pref_prob = 1.0
                for idx in range(0, len(pref) - 1, 2):
                    trip = dtm.triples[graph[0][pref[idx]][pref[idx + 1]][pref[idx + 2]]]
                    pref_prob *= trip.prob

                penult_sidx = pref[-1]
                penult_prob = trip.prob
                for trip_idx in graph[4][penult_sidx]:
                    trip = dtm.triples[trip_idx]
                    alt_pref = pref[:-2] + [ trip.action, trip.dest ]
                    if alt_pref in prefixes[length]: # skip
                        continue

                    print ('\talternate = ', end='')
                    print (alt_pref)
                    # Compute partial prod sum for alt_pref
                    alt_prob = pref_prob / penult_prob
                    alt_prob *= trip.prob

                    row = [0.0] * reward_vars

                    # set pref coefficients
                    for idx in range(0, len(pref) - 1, 2):
                        pos = rewards_idx[pref[idx]][pref[idx + 1]][pref[idx + 2]]
                        row[pos] -= pref_prob * math.pow(discount_rate, idx / 2)

                    # set alt pref coefficients
                    for idx in range(0, len(alt_pref) - 1, 2):
                        pos = rewards_idx[alt_pref[idx]][alt_pref[idx + 1]][alt_pref[idx + 2]]
                        row[pos] += alt_prob * math.pow(discount_rate, idx / 2)

                    b_constants.append(0.0) # Constant offset
                    for pos in range(reward_vars):
                        if row[pos] != 0:
                            A_table_value.append(row[pos])
                            A_table_row.append(A_table_num_constraints)
                            A_table_column.append(pos)
                    A_table_num_constraints += 1


    print (datetime.datetime.now())
    print ('Trajectory path emphasis constraints = ', end='')
    print (A_table_num_constraints - A_num_nbr_constraints)
    A_num_nbr_constraints = A_table_num_constraints

#    print (datetime.datetime.now())
#    print ('Adding constraints to maximize magnitude of reward values...')
#    for ridx in range(reward_vars):
#
#########################################################
#        A_table_value.append(-1.0) #reward constraint coeff
#        A_table_row.append(A_table_num_constraints)
#        A_table_column.append(ridx)
#        A_table_num_constraints += 1
#
#        b_constants.append(0.0) # Constant offset
#########################################################


    print (datetime.datetime.now())
    print ('Adding in delta+ and delta- for each such constraint...')
    A_num_rows = A_table_num_constraints
    # Add in the unique delta+ and delta- for each trajectory vs neighbors constraints:
    for constraint_idx in range(A_num_rows):

########################################################
        A_table_value.append(1.0) # delta+ coeff in traj vs nbr constraint
        A_table_row.append(constraint_idx)
        A_table_column.append(reward_vars + constraint_idx)
        A_table_value.append(-1.0) # delta- coeff in traj vs nbr constraint
        A_table_row.append(constraint_idx)
        A_table_column.append(reward_vars + constraint_idx + A_num_rows)
########################################################

    print ('A_table_value length = ', end='')
    print (len(A_table_value))
    print ('A_table_row length = ', end='')
    print (len(A_table_row))
    print ('A_table_column length = ', end='')
    print (len(A_table_column))

    print (datetime.datetime.now())
    print ('Adding in naive BKB constraints...')
    # Constraints for naive BKB
    #   1) rewards[s][a][s'] = [ P(A|S) + dP+(A|S) - dP-(A|S) ] * P(S'|A,S) * 2 * peak - peak
    #   2) P(A|S) + dP+(A|S) - dP-(A|S) >= 0
    #   3) \sum_a [ P(A=a|S) + dP+(A=a|S) - dP"(A=a|S) ] <= 1
    #   4) \sum_a [ P(A=a|S) + dP+(A=a|S) - dP"(A=a|S) ] >= 1

    # Column to variable mapping
    #    [ 0 ... reward_vars-1 ] --- reward values for r(s,a,s')
    #    [ reward_vars ... reward_vars + A_num_rows * 2 - 1 ] --- delta+/- for each constraint
    #    [ reward_vars + A_num_rows * 2 ... reward_vars + A_num_rows * 2 + 2 * |A||S| - 1 ] --- dP+(A|S) and dP-(A|S)

    dP_start = reward_vars + A_num_rows * 2 # start of dP+/- pairs

    A_new_num_constraints = A_num_rows
    csxa = dtm.num_cs * dtm.num_a
    for sidx in range(dtm.num_cs):
        for aidx in range(dtm.num_a):
            try:
                action_probs[sidx] # Check if exists
            except KeyError:
                action_probs[sidx] = {}
            try:
                action_probs[sidx][aidx] # Chek if exists
            except KeyError:
                action_probs[sidx][aidx] = 0

            for didx in range(dtm.num_cs):
                tidx = graph[0][sidx][aidx][didx]
                if tidx != -1:
                    trip = dtm.triples[graph[0][sidx][aidx][didx]]
                    if trip.prob == 0:
                        print ('P[' + str(sidx) + '][' + str(aidx) + '][' + str(didx) + '] is zero.')
                    prob = trip.prob * 2.0 * float(peak)
                    cons = prob * action_probs[sidx][aidx] - float(peak)
                    # Constraint 1)
                    #   1) rewards[s][a][s'] = [ P(A|S) + dP+(A|S) - dP-(A|S) ] * P(S'|A,S) * 2 * peak - peak
                    A_table_value.append(1.0) # reward
                    A_table_row.append(A_new_num_constraints)
                    A_table_column.append(rewards_idx[sidx][aidx][didx])
                    A_table_value.append(-prob) # dP+(A|S)
                    A_table_row.append(A_new_num_constraints)
                    A_table_column.append(dP_start + sidx * dtm.num_a + aidx)
                    A_table_value.append(prob) # dP-(A|S)
                    A_table_row.append(A_new_num_constraints)
                    A_table_column.append(dP_start + csxa + sidx * dtm.num_a + aidx)
                    b_constants.append(cons)
                    A_new_num_constraints += 1

    print ('A_table_value length = ', end='')
    print (len(A_table_value))
    print ('A_table_row length = ', end='')
    print (len(A_table_row))
    print ('A_table_column length = ', end='')
    print (len(A_table_column))


    num_vars = dP_start + dtm.num_a * dtm.num_cs * 2
    A_table_num_constraints = A_new_num_constraints

    G_table_value = list()
    G_table_row = list()
    G_table_column = list()
    G_table_num_constraints = 0
    h_constants = list()

    for sidx in range(dtm.num_cs):
        # Constraint 3)
        #   3) \sum_a [ P(A=a|S) + dP+(A=a|S) - dP"(A=a|S) ] <= 1
        tmp_idx3 = G_table_num_constraints
        G_table_num_constraints += 1
        h_constants.append(0.0) # temporary holder

        # Constraint 4)
        #   4) \sum_a [ P(A=a|S) + dP+(A=a|S) - dP"(A=a|S) ] >= 1
        tmp_idx4 = G_table_num_constraints
        G_table_num_constraints += 1
        h_constants.append(0.0) # temporary holder

        sum45 = 0 # Action prob sums

        for aidx in range(dtm.num_a):
            # Constraint 2)
            #   2) P(A|S) + dP+(A|S) - dP-(A|S) >= 0
            G_table_value.append(-1.0) # dP+
            G_table_row.append(G_table_num_constraints)
            G_table_column.append(dP_start + sidx * dtm.num_a + aidx)
            G_table_value.append(1.0) # dP-
            G_table_row.append(G_table_num_constraints)
            G_table_column.append(dP_start + csxa + sidx * dtm.num_a + aidx)
            h_constants.append(action_probs[sidx][aidx]) # P(A|S)
            G_table_num_constraints += 1

            # Constraint 3)
            G_table_value.append(-1.0) # dP+
            G_table_row.append(tmp_idx3)
            G_table_column.append(dP_start + sidx * dtm.num_a + aidx)
            G_table_value.append(1.0) # dP-
            G_table_row.append(tmp_idx3)
            G_table_column.append(dP_start + csxa + sidx * dtm.num_a + aidx)

            # Constraint 4)
            G_table_value.append(1.0) # dP+
            G_table_row.append(tmp_idx4)
            G_table_column.append(dP_start + sidx * dtm.num_a + aidx)
            G_table_value.append(-1.0) # dP-
            G_table_row.append(tmp_idx4)
            G_table_column.append(dP_start + csxa + sidx * dtm.num_a + aidx)

            sum45 += action_probs[sidx][aidx]

        # Constraint 4)
        h_constants[tmp_idx3] = -1.0 + sum45

        # Constraint 5)
        h_constants[tmp_idx4] = 1.0 - sum45



#    print (datetime.datetime.now())
#    print ('Building lower bound constraints for each triple\'s reward...')
    # Add in r(s, a, s') >= -1.0 * peak
#    for ridx in range(reward_vars):

########################################################
#        G_table_value.append(-1.0) #reward constraint coeff
#        G_table_row.append(G_table_num_constraints)
#        G_table_column.append(ridx)
#        G_table_num_constraints += 1
########################################################

########################################################
#        h_constants.append(float(peak))  # Rewards lower bound
#        h_constants.append(-1)  # Rewards lower bound
########################################################


#    print (datetime.datetime.now())
#    print ('Building upper bound constraints for each triple\'s reward...')
    # Add in r(s, a, s') <= peak
#    for ridx in range(reward_vars):

########################################################
#        G_table_value.append(1.0) #reward constraint coeff
#        G_table_row.append(G_table_num_constraints)
#        G_table_column.append(ridx)
#        G_table_num_constraints += 1
########################################################

########################################################
#        h_constants.append(float(peak))  # Rewards lower bound
#        h_constants.append(-1)  # Rewards lower bound
########################################################


    # Add in delta+ and delta- variables >= 0 constraints
    # Add in dP+ and dP- variables >= 0 constraints
    # Define cost function over delta and dP variables

    print (datetime.datetime.now())
    print ('Building cost function and non-negativity constraints for delta+ and delta- variables and dP+(A|S) and dP-(A|S)...')
########################################################
    costs = [0.0] * num_vars    # Initialize all cost coeffs
########################################################

    for idx in range(reward_vars, num_vars):

########################################################
        if idx >= dP_start:
            costs[idx] = 10.0 # dP+/-
        else:
            if idx < reward_vars + A_num_nbr_constraints:
                costs[idx] = -1.0    # Delta+ cost coeffs for nbr constraints
            elif idx < reward_vars + A_num_rows:
                if reward_diversity == False:
                    costs[idx] = -1.0    # Delta+ cost coeffs for reward magnitude
            elif idx < reward_vars + A_num_rows + A_num_nbr_constraints:
                costs[idx] = 1000000.0    # Delta- cost coeffs for nbr constraints
            else:
                if reward_diversity == False:
                    costs[idx] = -1.0    # Delta- cost coeffs for reward magnitude
########################################################

########################################################
        G_table_value.append(-1.0) # +/- constraint coeef
        G_table_row.append(G_table_num_constraints)
        G_table_column.append(idx)
        G_table_num_constraints += 1
########################################################

########################################################
        h_constants.append(0.0)  # +/- lower bound
########################################################


    print (datetime.datetime.now())
    print ('Cost function dimension = ', end='')
    print (len(costs))
    c = matrix(costs)
    print ('A matrix number of constraints = ', end='')
    print (A_table_num_constraints)
    print ('A matrix number of variables = ', end='')
    print (reward_vars + 2 * A_table_num_constraints)
    A = spmatrix(A_table_value, A_table_row, A_table_column, ( A_table_num_constraints, num_vars ))
    print ('b constants dimension = ', end='')
    print (len(b_constants))
    b = matrix(b_constants)
    print ('G matrix number of constraints = ', end='')
    print (G_table_num_constraints)
    if G_table_num_constraints > 0:
        print ('G matrix number of variables = ', end='')
        print (num_vars)
    G = spmatrix(G_table_value, G_table_row, G_table_column, ( G_table_num_constraints, num_vars ))
    print ('h constants dimension = ', end='')
    print (len(h_constants))
    print (datetime.datetime.now())
    h = matrix(h_constants)
    print ('Executing optimization...')
    if reward_diversity == True: # maximize the difference between rewards
        print ('Using reward diversity objective (maximize difference squared)...')
#        n = len(costs)
#        P_table_value = list()
#        P_table_row = list()
#        P_table_column = list()
#        for idx_i in range(reward_vars):
#            for idx_j in range(reward_vars):
#                if idx_j == idx_i:
#                    P_table_value.append(float(2.0 * n) - 2.0) #reward coeff
#                    P_table_row.append(idx_i)
#                    P_table_column.append(idx_j)
#                else:
#                    P_table_value.append(-2.0) #reward coeff
#                    P_table_row.append(idx_i)
#                    P_table_column.append(idx_j)
#        P = spmatrix(P_table_value, P_table_row, P_table_column, ( n, n ))
#        soln = solvers.qp(P, c, G, h, A, b)

        n = len(costs) # including reward_vars
        cT = c.T
        ddf = [0] * n
        for idx_a in range(reward_vars - 1):
            for idx_b in range(idx_a + 1, reward_vars):
                ddf[idx_a] -= 2
                ddf[idx_b] -= 2
        ddfT = matrix(ddf).T
        def F(x=None, z=None):
            if x is None: return 0, matrix(0.0, (n,1))
            d = [ (a - b)**2 for a, b in combinations(x[:reward_vars], 2) ]
            f = cT * x - sum(d)
            df = [0] * n
            for idx_a, a in enumerate(x[:reward_vars - 1]):
                for idx_b, b in enumerate(x[idx_a + 1:reward_vars]):
                    df[idx_a] -= 2 * ( a -  b )
                    df[idx_b + idx_a] -= 2 * ( b - a )
            Df = c + matrix(df)
            if z is None: return f, Df
            H = spdiag(z[0] * ddfT)
            return f, Df, H
        soln = solvers.cp(F, G=G, h=h, A=A, b=b)['x']
    else: # Just regular linear objective
        soln = solvers.lp(c, G, h, A, b, solver = 'glpk')
#    soln = solvers.lp(c, A, b)

    print (datetime.datetime.now())
    print ('Extracting rewards...')
    rewards = array(soln['x'])
    rw = list()
    for sidx in range(dtm.num_cs):
        for aidx in range(dtm.num_a):
            for didx in range(dtm.num_cs):
                tidx = graph[0][sidx][aidx][didx]
                if tidx != -1:
                    triple = dtm.triples[tidx]
                    triple.reward = rewards[rewards_idx[sidx][aidx][didx]][0]
                    rw.append(triple.reward)
    print (datetime.datetime.now())
    print ('Rewards statistics:')
    print ('\t', end='')
    print (stats.describe(rw))
    if num_vars > reward_vars:
        dw = list()
        for idx in range(reward_vars,num_vars):
            dw.append(rewards[idx][0])
        print ('Delta statistics:')
        print ('\t', end='')
        print (stats.describe(dw))

#    input ('Continue?')

    print (datetime.datetime.now())
    print ('Extracting P(A|S)...')
    new_action_probs = {}
    for sidx in range(dtm.num_cs):
        new_action_probs[sidx] = {}
        for aidx in range(dtm.num_a):
            new_action_probs[sidx][aidx] = {}
            new_action_probs[sidx][aidx][0] = rewards[dP_start + sidx * dtm.num_a + aidx][0]
            new_action_probs[sidx][aidx][1] = rewards[dP_start + csxa + sidx * dtm.num_a + aidx][0]

    print (datetime.datetime.now())
    print ('P(A|S) updates:')
    for sidx in range(dtm.num_cs):
        for aidx in range(dtm.num_a):
            p_change = new_action_probs[sidx][aidx][0] - new_action_probs[sidx][aidx][1]
            if p_change == 0: # Only output if there was a change
                continue
            print ('P( ' + str(aidx) + ' | ' + str(sidx) + ' ) = ', end='')
            print (action_probs[sidx][aidx], end=' ')
            print ('-->', end=' ')
            print (action_probs[sidx][aidx] + p_change, end = '\t')
            print ('delta = ', end='')
            print (p_change, end='\t')
            print ('dP+ = ', end='')
            print (new_action_probs[sidx][aidx][0], end='')
            print (' , dP- = ', end='')
            print (new_action_probs[sidx][aidx][1])

#    input ('Continue?')

    print (datetime.datetime.now())
    print ('Generating trajectory and neighborhood expected values --')
    print ('\tTrajectory probability does not include discount factor.')
    analysis = list()
    analysis.append([ 'Trajectory/Nbr Id', 'Trajectory probability', 'Reward sum', 'Expected linear reward with discount', 'Trajectory probability x Reward sum', 'Trajectory' ])
    for traj_idx, trajectory in enumerate(trajectories):
        analysis_item = [ 'Trajectory #' + str(traj_idx + 1), trajectory_probability[traj_idx] ]
        reward = 0
        linear_expected = 0
        discount = 1.0
        for idx in range(0, len(trajectory) - 1, 2):
            trip = dtm.triples[graph[0][trajectory[idx]][trajectory[idx + 1]][trajectory[idx + 2]]]
            reward += trip.reward
            linear_expected += trip.reward * trip.prob * discount
            discount *= discount_rate
        analysis_item.append(reward)
        analysis_item.append(linear_expected)
        analysis_item.append(trajectory_probability[traj_idx] * reward)
        analysis_item.append(trajectory)
        analysis.append(analysis_item)

        for nidx, nbr in enumerate(neighborhoods[traj_idx]):
            analysis_item = [ '\tNeighbor #' + str(nidx + 1), neighbor_probability[traj_idx][nidx] ]
            reward = 0
            linear_expected = 0
            discount = 1.0
            for idx in range(0, len(nbr) - 1, 2):
                trip = dtm.triples[graph[0][nbr[idx]][nbr[idx + 1]][nbr[idx + 2]]]
                reward += trip.reward
                linear_expected += reward * trip.prob * discount
                discount *= discount_rate
            analysis_item.append(reward)
            analysis_item.append(linear_expected)
            analysis_item.append(neighbor_probability[traj_idx][nidx] * reward)
            analysis_item.append(nbr)
            analysis.append(analysis_item)
    return(analysis)


# Compute reward triples R(s, a, s') and sets values in dtm
def DTM_doIRL(dtm, graph, trajectories, m_bound, discount_rate, peak, traj_prod_sum, traj_sum, incremental_prod_sum_aggregates, path_emphasis, merged_states, merged_percentage, nbr_branching, reward_diversity):
    print (datetime.datetime.now())
    print ('Starting optimization problem for reward triples...')
#    # remove redundant trajectories
#    print ('Eliminating redundant trajectories...')
#    print ('\t' + str(len(trajectories)) + ' trajectories')
#    temp_trajs = list()
#    for traj in trajectories:
#        found = False
#        for traj2 in temp_trajs:
#            if traj == traj2:
#                found = True
#                break
#        if not found:
#            temp_trajs.append(traj)
#    trajectories = temp_trajs
#    print ('\t' + str(len(trajectories)) + ' unique trajectories')


    print (datetime.datetime.now())
    print ('Building trajectory suffixes...')
    trajectories = dtm_neighborhood.DTM_makeTrajectoriesProperSuffixes(trajectories)
    print ('\tTotal trajectories + suffixes = ' + str(len(trajectories)))
    print (datetime.datetime.now())


    # Systematically order and initialize reward triples
    #   Only index a variable if triple (s, a, s') exists, i.e., there is a
    #   transition.
    print (datetime.datetime.now())
    print ('Building optimization problem...')
    rewards_idx = [0] * dtm.num_cs
    rewards_list = list()
    count = 0
    for sidx in range(dtm.num_cs):
        rewards_idx[sidx] = [0] * dtm.num_a
        for aidx in range(dtm.num_a):
            rewards_idx[sidx][aidx] = [-1] * dtm.num_cs
            for didx in range(dtm.num_cs):
                tidx = graph[0][sidx][aidx][didx]
                if tidx != -1:
                    rewards_idx[sidx][aidx][didx] = count
                    rewards_list.append( (sidx, aidx, didx) )
                    count += 1
    reward_vars = count # So, far only rewards variables -- delta variables added later
    print ('Number of reward triples with non-zero probability = ', end='')
    print (reward_vars)
    print (datetime.datetime.now())


    # Formulate constraints

    A_table_value = list() # Storing in three lists corresponding to value, row, column for sparse matrices
    A_table_row = list()
    A_table_column = list()
    A_table_num_constraints = 0 # How many constraints
    b_constants = list() # Corresponds to the constant on each constraint

    # Always start with trajectory vs neighbors constraints (equality constraint)
    neighborhoods = [0] * len(trajectories)
    neighbor_probability = [0] * len(trajectories)
    for traj_idx in range(len(trajectories)):
        neighbor_probability[traj_idx] = list()
    trajectory_probability = [-1] * len(trajectories)
    for traj_idx, trajectory in enumerate(trajectories):
        print ('Trajectory #' + str(traj_idx + 1), end=':')
#        print (trajectory, end = ' ')

        prob = 1.0
        for idx in range(0, len(trajectory) - 1, 2): # compute trajectory probability
            trip = dtm.triples[graph[0][trajectory[idx]][trajectory[idx + 1]][trajectory[idx + 2]]]

########################################################
            prob *= trip.prob # trajectory probability
########################################################

        trajectory_probability[traj_idx] = prob

        # only compute neighbors if flags require
        if traj_prod_sum or traj_sum or incremental_prod_sum_aggregates:
            if nbr_branching == True:
                neighbors = dtm_neighborhood.DTM_makeBranchingNeighborhood(dtm, graph, trajectory, trajectories)
            else:
                neighbors = dtm_neighborhood.DTM_makeNeighborhood(dtm, graph, trajectory, trajectories, m_bound)
        else:
            neighbors = list()

        print ('Nbrs = ' + str(len(neighbors)))
#$        row = [0.0] * reward_vars # here, we sum up the neighbors vs a normalized trajectory
        neighborhoods[traj_idx] = copy.deepcopy(neighbors)
        for nbr in neighbors:
#            print (nbr)
            if traj_prod_sum or traj_sum:
                row = [0.0] * reward_vars # if here, then we generate one trajectory vs nbr constraint each

            if traj_prod_sum: # Use prod sum
                discount = 1.0
                for idx in range(0, len(trajectory) - 1, 2): # set trajectory constraint coefficients
                    pos = rewards_idx[trajectory[idx]][trajectory[idx + 1]][trajectory[idx + 2]]

########################################################
                    row[pos] += -prob * discount # trajectory expected value term
########################################################

                    discount *= discount_rate

                prob = 1.0
                for idx in range(0, len(nbr) - 1, 2): # compute neighbor trajectory probability
                    trip = dtm.triples[graph[0][nbr[idx]][nbr[idx + 1]][nbr[idx + 2]]]

########################################################
                    prob *= trip.prob
########################################################

                neighbor_probability[traj_idx].append(prob)

                discount = 1.0
                for idx in range(0, len(nbr) - 1, 2): # set neighbor trajectory probability
                    pos = rewards_idx[nbr[idx]][nbr[idx + 1]][nbr[idx + 2]]

########################################################
                    row[pos] += prob * discount # neighbor expected value term
########################################################

                    discount *= discount_rate

#                if not (row in table): # Here is for individual contraints per neighbor
                if True:

########################################################
                    b_constants.append(0.0) # Constant offset
########################################################

                    for pos in range(reward_vars):
                        if row[pos] != 0:
                            A_table_value.append(row[pos])
                            A_table_row.append(A_table_num_constraints)
                            A_table_column.append(pos)
                    A_table_num_constraints += 1

            if traj_sum == True: # Use sum of products of prob x reward
                discount = 1.0
                prob = 1.0
                for idx in range(0, len(trajectory) - 1, 2): # compute trajectory probability
                    trip = dtm.triples[graph[0][trajectory[idx]][trajectory[idx + 1]][trajectory[idx + 2]]]
                    pos = rewards_idx[trajectory[idx]][trajectory[idx + 1]][trajectory[idx + 2]]
                    prob *= trip.prob

########################################################
                    row[pos] += -trip.prob * discount * trip.reward
########################################################

                    discount *= discount_rate


                discount = 1.0
                prob = 1.0
                for idx in range(0, len(nbr) - 1, 2): # compute neighbor trajectory probability
                    trip = dtm.triples[graph[0][nbr[idx]][nbr[idx + 1]][nbr[idx + 2]]]
                    pos = rewards_idx[nbr[idx]][nbr[idx + 1]][nbr[idx + 2]]
                    prob *= trip.prob

########################################################
                    row[pos] += trip.prob * discount * trip.reward
########################################################

                    discount *= discount_rate
                neighbor_probability[traj_idx].append(prob)

#                if not (row in table): # Here is for individual contraints per neighbor
                if True:

########################################################
                    b_constants.append(0.0) # Constant offset
########################################################

                    for pos in range(reward_vars):
                        if row[pos] != 0:
                            A_table_value.append(row[pos])
                            A_table_row.append(A_table_num_constraints)
                            A_table_column.append(pos)
                    A_table_num_constraints += 1


    if incremental_prod_sum_aggregates == True:
        print ('Constructing incremental product sum aggregates constraints...')
        print (datetime.datetime.now())
        for traj_idx, trajectory in enumerate(trajectories):
            print ('Trajectory #' + str(traj_idx + 1), end=':')
            print (trajectory, end = ' ')
            if neighborhoods[traj_idx] == 0:
                if nbr_branching == True:
                    neighbors = dtm_neighborhood.DTM_makeBranchingNeighborhood(dtm, graph, trajectory, trajectories)
                else:
                    neighbors = dtm_neighborhood.DTM_makeNeighborhood(dtm, graph, trajectory, trajectories, m_bound)
                neighborhoods[traj_idx] = copy.deepcopy(neighbors)
            else:
                neighbors = neighborhoods[traj_idx]
            print ('Nbrs = ' + str(len(neighbors)))

            if len(neighbor_probability[traj_idx]) != len(neighbors):
                neighbor_probability[traj_idx] = list()
                for nbr in neighbors:
                    prob = 1.0
                    for idx in range(0, len(nbr) - 1, 2): # compute neighbor trajectory probability
                        trip = dtm.triples[graph[0][nbr[idx]][nbr[idx + 1]][nbr[idx + 2]]]
                        prob *= trip.prob
                    neighbor_probability[traj_idx].append(prob)
            if len(neighbors) == 0:
                continue # nothing to make

            row = [0.0] * reward_vars
            traj_prob = trajectory_probability[traj_idx]
            for idx in range(0, len(trajectory) - 1, 2):
                pos = rewards_idx[trajectory[idx]][trajectory[idx + 1]][trajectory[idx + 2]]
                row[pos] -= traj_prob * math.pow(discount_rate, idx / 2) * len(neighbors)
                for nidx,nbr in enumerate(neighbors):
                    if idx + 2 >= len(nbr):
                       continue
                    pos = rewards_idx[nbr[idx]][nbr[idx + 1]][nbr[idx + 2]]
                    row[pos] += neighbor_probability[traj_idx][nidx] * math.pow(discount_rate, idx / 2)
                if idx == 0: # Skip a length of one aggregate
                    continue

                b_constants.append(0.0) # Constant offset
                for pos in range(reward_vars):
                    if row[pos] != 0:
                        A_table_value.append(row[pos])
                        A_table_row.append(A_table_num_constraints)
                        A_table_column.append(pos)
                A_table_num_constraints += 1

    print (datetime.datetime.now())
    A_num_nbr_constraints = A_table_num_constraints
    print ('Trajectory vs Neighbors constraints = ', end='')
    print (A_num_nbr_constraints)

    # Path emphasis basically takes each the prefix of each trajectory and
    #    makes the partial (prod prob) (sum rewards x discount) to be greater than any other paths in the MDP
    #    that have the same prefix except for the last action and state except for prefixes of
    #    other trajectories. Partial prod prob sum rewards include trajectories.
    if path_emphasis == True:
        print ('Constructing path emphasis constraints...')
        (N, prefixes) = DTM_prefixTrajectories(trajectories)
        for length in range(1, N):
            for pref in prefixes[length]:
                print ('Prefix = ', end='')
                print (pref)
                # Compute partial prod sum for pref
                pref_prob = 1.0
                for idx in range(0, len(pref) - 1, 2):
                    trip = dtm.triples[graph[0][pref[idx]][pref[idx + 1]][pref[idx + 2]]]
                    pref_prob *= trip.prob

                penult_sidx = pref[-1]
                penult_prob = trip.prob
                for trip_idx in graph[4][penult_sidx]:
                    trip = dtm.triples[trip_idx]
                    alt_pref = pref[:-2] + [ trip.action, trip.dest ]
                    if alt_pref in prefixes[length]: # skip
                        continue

                    print ('\talternate = ', end='')
                    print (alt_pref)
                    # Compute partial prod sum for alt_pref
                    alt_prob = pref_prob / penult_prob
                    alt_prob *= trip.prob

                    row = [0.0] * reward_vars

                    # set pref coefficients
                    for idx in range(0, len(pref) - 1, 2):
                        pos = rewards_idx[pref[idx]][pref[idx + 1]][pref[idx + 2]]
                        row[pos] -= pref_prob * math.pow(discount_rate, idx / 2)

                    # set alt pref coefficients
                    for idx in range(0, len(alt_pref) - 1, 2):
                        pos = rewards_idx[alt_pref[idx]][alt_pref[idx + 1]][alt_pref[idx + 2]]
                        row[pos] += alt_prob * math.pow(discount_rate, idx / 2)

                    b_constants.append(0.0) # Constant offset
                    for pos in range(reward_vars):
                        if row[pos] != 0:
                            A_table_value.append(row[pos])
                            A_table_row.append(A_table_num_constraints)
                            A_table_column.append(pos)
                    A_table_num_constraints += 1


    print (datetime.datetime.now())
    print ('Trajectory path emphasis constraints = ', end='')
    print (A_table_num_constraints - A_num_nbr_constraints)
    A_num_nbr_constraints = A_table_num_constraints

#    print (datetime.datetime.now())
#    print ('Adding constraints to maximize magnitude of reward values...')
#    for ridx in range(reward_vars):
#
#########################################################
#        A_table_value.append(-1.0) #reward constraint coeff
#        A_table_row.append(A_table_num_constraints)
#        A_table_column.append(ridx)
#        A_table_num_constraints += 1
#
#        b_constants.append(0.0) # Constant offset
#########################################################


    print (datetime.datetime.now())
    print ('Adding in delta+ and delta- for each such constraint...')
    A_num_rows = A_table_num_constraints
    # Add in the unique delta+ and delta- for each trajectory vs neighbors constraints:
    for constraint_idx in range(A_num_rows):

########################################################
        A_table_value.append(1.0) # delta+ coeff in traj vs nbr constraint
        A_table_row.append(constraint_idx)
        A_table_column.append(reward_vars + constraint_idx)
        A_table_value.append(-1.0) # delta- coeff in traj vs nbr constraint
        A_table_row.append(constraint_idx)
        A_table_column.append(reward_vars + constraint_idx + A_num_rows)
########################################################

    num_vars = reward_vars + A_num_rows * 2


    G_table_value = list()
    G_table_row = list()
    G_table_column = list()
    G_table_num_constraints = 0
    h_constants = list()


    print (datetime.datetime.now())
    print ('Building lower bound constraints for each triple\'s reward...')
    # Add in r(s, a, s') >= -1.0 * peak
    for ridx in range(reward_vars):

########################################################
        G_table_value.append(-1.0) #reward constraint coeff
        G_table_row.append(G_table_num_constraints)
        G_table_column.append(ridx)
        G_table_num_constraints += 1
########################################################

########################################################
        h_constants.append(float(peak))  # Rewards lower bound
#        h_constants.append(-1)  # Rewards lower bound
########################################################


    print (datetime.datetime.now())
    print ('Building upper bound constraints for each triple\'s reward...')
    # Add in r(s, a, s') <= peak
    for ridx in range(reward_vars):

########################################################
        G_table_value.append(1.0) #reward constraint coeff
        G_table_row.append(G_table_num_constraints)
        G_table_column.append(ridx)
        G_table_num_constraints += 1
########################################################

########################################################
        h_constants.append(float(peak))  # Rewards lower bound
#        h_constants.append(-1)  # Rewards lower bound
########################################################


    # Add in delta+ and delta- variables >= 0 constraints
    # Define cost function over delta variables

    print (datetime.datetime.now())
    print ('Building cost function and non-negativity constraints for delta+ and delta- variables...')
########################################################
    costs = [0.0] * num_vars    # Rewards cost coeff
########################################################

    for idx in range(reward_vars, num_vars):

########################################################
        if idx < reward_vars + A_num_nbr_constraints:
            costs[idx] = -1.0    # Delta+ cost coeffs for nbr constraints
        elif idx < reward_vars + A_num_rows:
            if reward_diversity == False:
                costs[idx] = -1.0    # Delta+ cost coeffs for reward magnitude
        elif idx < reward_vars + A_num_rows + A_num_nbr_constraints:
            costs[idx] = 1.0    # Delta- cost coeffs for nbr constraints
        else:
            if reward_diversity == False:
                costs[idx] = -1.0    # Delta- cost coeffs for reward magnitude
########################################################

########################################################
        G_table_value.append(-1.0) # Delta+/- constraint coeef
        G_table_row.append(G_table_num_constraints)
        G_table_column.append(idx)
        G_table_num_constraints += 1
########################################################

########################################################
        h_constants.append(0.0)  # Delta+/- lower bound
########################################################


    print (datetime.datetime.now())
    print ('Cost function dimension = ', end='')
    print (len(costs))
    c = matrix(costs)
    print ('A matrix number of constraints = ', end='')
    print (A_table_num_constraints)
    print ('A matrix number of variables = ', end='')
    print (reward_vars + 2 * A_table_num_constraints)
    A = spmatrix(A_table_value, A_table_row, A_table_column, ( A_table_num_constraints, num_vars ))
    print ('b constants dimension = ', end='')
    print (len(b_constants))
    b = matrix(b_constants)
    print ('G matrix number of constraints = ', end='')
    print (G_table_num_constraints)
    if G_table_num_constraints > 0:
        print ('G matrix number of variables = ', end='')
        print (num_vars)
    G = spmatrix(G_table_value, G_table_row, G_table_column, ( G_table_num_constraints, num_vars ))
    print ('h constants dimension = ', end='')
    print (len(h_constants))
    print (datetime.datetime.now())
    h = matrix(h_constants)
    print ('Executing optimization...')
    if reward_diversity == True: # maximize the difference between rewards
        print ('Using reward diversity objective (maximize difference squared)...')
#        n = len(costs)
#        P_table_value = list()
#        P_table_row = list()
#        P_table_column = list()
#        for idx_i in range(reward_vars):
#            for idx_j in range(reward_vars):
#                if idx_j == idx_i:
#                    P_table_value.append(float(2.0 * n) - 2.0) #reward coeff
#                    P_table_row.append(idx_i)
#                    P_table_column.append(idx_j)
#                else:
#                    P_table_value.append(-2.0) #reward coeff
#                    P_table_row.append(idx_i)
#                    P_table_column.append(idx_j)
#        P = spmatrix(P_table_value, P_table_row, P_table_column, ( n, n ))
#        soln = solvers.qp(P, c, G, h, A, b)

        n = len(costs) # including reward_vars
        cT = c.T
        ddf = [0] * n
        for idx_a in range(reward_vars - 1):
            for idx_b in range(idx_a + 1, reward_vars):
                ddf[idx_a] -= 2
                ddf[idx_b] -= 2
        ddfT = matrix(ddf).T
        def F(x=None, z=None):
            if x is None: return 0, matrix(0.0, (n,1))
            d = [ (a - b)**2 for a, b in combinations(x[:reward_vars], 2) ]
            f = cT * x - sum(d)
            df = [0] * n
            for idx_a, a in enumerate(x[:reward_vars - 1]):
                for idx_b, b in enumerate(x[idx_a + 1:reward_vars]):
                    df[idx_a] -= 2 * ( a -  b )
                    df[idx_b + idx_a] -= 2 * ( b - a )
            Df = c + matrix(df)
            if z is None: return f, Df
            H = spdiag(z[0] * ddfT)
            return f, Df, H
        soln = solvers.cp(F, G=G, h=h, A=A, b=b)['x']
    else: # Just regular linear objective
        soln = solvers.lp(c, G, h, A, b, solver = 'glpk')
#    soln = solvers.lp(c, A, b)

    print (datetime.datetime.now())
    print ('Extracting rewards...')
    rewards = array(soln['x'])
    rw = list()
    for sidx in range(dtm.num_cs):
        for aidx in range(dtm.num_a):
            for didx in range(dtm.num_cs):
                tidx = graph[0][sidx][aidx][didx]
                if tidx != -1:
                    triple = dtm.triples[tidx]
                    triple.reward = rewards[rewards_idx[sidx][aidx][didx]][0]
                    rw.append(triple.reward)
    print (datetime.datetime.now())
    print ('Rewards statistics:')
    print ('\t', end='')
    print (stats.describe(rw))
    if num_vars > reward_vars:
        dw = list()
        for idx in range(reward_vars,num_vars):
            dw.append(rewards[idx][0])
        print ('Delta statistics:')
        print ('\t', end='')
        print (stats.describe(dw))


    print (datetime.datetime.now())
    print ('Generating trajectory and neighborhood expected values --')
    print ('\tTrajectory probability does not include discount factor.')
    analysis = list()
    analysis.append([ 'Trajectory/Nbr Id', 'Trajectory probability', 'Reward sum', 'Expected linear reward with discount', 'Trajectory probability x Reward sum', 'Trajectory' ])
    for traj_idx, trajectory in enumerate(trajectories):
        analysis_item = [ 'Trajectory #' + str(traj_idx + 1), trajectory_probability[traj_idx] ]
        reward = 0
        linear_expected = 0
        discount = 1.0
        for idx in range(0, len(trajectory) - 1, 2):
            trip = dtm.triples[graph[0][trajectory[idx]][trajectory[idx + 1]][trajectory[idx + 2]]]
            reward += trip.reward
            linear_expected += trip.reward * trip.prob * discount
            discount *= discount_rate
        analysis_item.append(reward)
        analysis_item.append(linear_expected)
        analysis_item.append(trajectory_probability[traj_idx] * reward)
        analysis_item.append(trajectory)
        analysis.append(analysis_item)

        for nidx, nbr in enumerate(neighborhoods[traj_idx]):
            analysis_item = [ '\tNeighbor #' + str(nidx + 1), neighbor_probability[traj_idx][nidx] ]
            reward = 0
            linear_expected = 0
            discount = 1.0
            for idx in range(0, len(nbr) - 1, 2):
                trip = dtm.triples[graph[0][nbr[idx]][nbr[idx + 1]][nbr[idx + 2]]]
                reward += trip.reward
                linear_expected += reward * trip.prob * discount
                discount *= discount_rate
            analysis_item.append(reward)
            analysis_item.append(linear_expected)
            analysis_item.append(neighbor_probability[traj_idx][nidx] * reward)
            analysis_item.append(nbr)
            analysis.append(analysis_item)
    return(analysis)


# Updates a dtm's reward and probability from DTL_IRL rewards_csv_file generated from DTM_IRL
def updateDTM(dtm, rewards_csv_file):
    print ('Updating DTM from rewards file...')
    print ('\tAssuming header row...')
    print ('\tFields equivalent to [ Start state index, Action index, Finish state index, Probability for Triple, Reward for Triple, Prob x Reward ]')
    with open(rewards_csv_file, 'r') as update:
        reader = csv.reader(update)
        lines = list(reader)
        headers = lines[0]
        count = 1 # Start of triples list
        total = len(lines)
        num_triples = len(dtm.triples)
        while count < total:
            if len(lines[count]) != 6:
                break
            try:
                sidx = int(lines[count][0])
                aidx = int(lines[count][1])
                didx = int(lines[count][2])
                prob = float(lines[count][3])
                reward = float(lines[count][4])
            except ValueError:
                break # End of triples list
            for tidx in range(num_triples):
                if dtm.triples[tidx].source == sidx and dtm.triples[tidx].action == aidx and dtm.triples[tidx].dest == didx:
                    dtm.triples[tidx].prob = prob
                    dtm.triples[tidx].reward = reward
                    break
                if tidx == num_triples - 1:
                    print ('Unknown triple (' + str(sidx) + ' , ' + str(aidx) + ' , ' + str(didx) + ') -- ignoring')
            count += 1


# max_changes is number of trajectory alterations
# discount_rate is value of elements later in trajectory
# peak is the max expected value for a training trajectory
#   also bounds max value for a reward triple
# traj_prod_sum == True means to add in prod sum nbr constraints
# traj_sum == True means to add in linear sum nbr constraints
# incremental_prod_sum_aggregate - use the incremental and aggregate prod sum nbr constraints
# path_emphasis == True implies
    # Path emphasis basically takes each the prefix of each trajectory and
    #    makes the partial prod prob sum rewards to be greater than any other paths in the MDP
    #    that have the same prefix except for the last action and state except for prefixes of
    #    other trajectories. Partial prod prob sum rewards include trajectores
# dtm_report_csvfile generates a report on the learned dtm
# evaluate_trajectories_files generates additional evaluations of trajectories as part of the
#   final analysis
def DTM_IRL(dtmfile, trajectories_file, actions_csv, rewards_csv_file, max_changes = 0, discount_rate = 1.0, peak = 100.0, traj_prod_sum = True, traj_sum = False, incremental_prod_sum_aggregate = False, path_emphasis = False, dtm_report_csvfile='', evaluate_trajectories_file='', merged_states = False, merged_percentage = 90.0, nbr_branching = False, reward_diversity = False, trajectories_nbrs_file = '', evaluate_trajectories_nbrs_file=''):
    if isinstance(dtmfile, str):
        print (datetime.datetime.now())
        print ('Loading dtm...')
        dtm = pydtm.pydtm(dtmfile)
        print (datetime.datetime.now())
    else:
        dtm = dtmfile
    print ('Building graph object from dtm...')
    graph = dtm_neighborhood.DTM_makeGraph(dtm)
    print (datetime.datetime.now())

    print ('Loading target trajectories...')
    trajectory_ids = dict()
    trajectories = DTM_loadTrajectoriesFile(dtm, trajectories_file, actions_csv)
    trajs = trajectories
    for i_idx, item in enumerate(trajs):
        try: # Skip if already seen.
            trajectory_ids[tuple(item)]
        except KeyError: # Hash to its position index and filename
            trajectory_ids[tuple(item)] = ( i_idx, trajectories_file )
    trajectories = set()
    [ trajectories.add(tuple(item)) for item in trajs ] # Make hashable
        # Also removed duplicates
    trajs = [ list(item) for item in trajectories ]
    print (datetime.datetime.now())
    if merged_states == True:
        print ('Merging states...')
        (dtm, graph) = dtm_states.DTM_computeMergeStates(dtm, graph, trajs, percent_distant_pairs = merged_percentage)
        print (datetime.datetime.now())
    print ('')
    print ('Number of unique target trajectories = ', end='')
    print (len(trajs))
    print ('')
    print (datetime.datetime.now())

    if trajectories_nbrs_file != '':
        print ('Loading target nbr trajectories...')
        nbr_trajectories = DTM_loadTrajectoriesFile(dtm, trajectories_nbrs_file, actions_csv)
        nbr_trajs = nbr_trajectories
        nbr_trajectories = set()
        [ nbr_trajectories.add(tuple(item)) for item in nbr_trajs ] # Make hashable
        for i_idx, item in enumerate(nbr_trajs):
            try: # Skip if already seen.
                trajectory_ids[tuple(item)]
            except KeyError: # Hash to its position index and filename
                trajectory_ids[tuple(item)] = ( i_idx, trajectories_nbrs_file )
        nbr_trajs = [ list(item) for item in nbr_trajectories ]
        print (datetime.datetime.now())
        if merged_states == True:
            print ('Merging states...')
            (dtm, graph) = dtm_states.DTM_computeMergeStates(dtm, graph, nbr_trajs, percent_distant_pairs = merged_percentage)
            print (datetime.datetime.now())

        # Eliminate overlap with target trajectories
        print ('Eliminating overlaps with target trajectories...')
        nbr_trajectories = nbr_trajectories - trajectories
        nbr_trajs = [ list(item) for item in nbr_trajectories ]
    else:
        nbr_trajectories = set()
        nbr_trajs = list()
    print ('')
    print ('Number of unique target nbr trajectories = ', end='')
    print (len(nbr_trajs))
    print ('')
    print (datetime.datetime.now())

    eval_trajectories = set()
    if evaluate_trajectories_file != '':
        print ('Loading other\'s trajectories...')
        if not isinstance(evaluate_trajectories_file, list):
            eval_trajectories_file_list = [ evaluate_trajectories_file ]
        else:
            eval_trajectories_file_list = evaluate_trajectories_file
        for eval_file in eval_trajectories_file_list:
            print ('\tLoading trajectories for ' + eval_file + '...')
            eval_trajs = DTM_loadTrajectoriesFile(dtm, eval_file, actions_csv)
            for i_idx, item in enumerate(eval_trajs):
                try: # Skip if already seen.
                    trajectory_ids[tuple(item)]
                except KeyError: # Hash to its position index and filename
                    trajectory_ids[tuple(item)] = ( i_idx, eval_file )
            [ eval_trajectories.add(tuple(item)) for item in eval_trajs ]
            print (datetime.datetime.now())

    # Eliminate overlap with target trajectories and target nbr trajectories
    print ('Eliminating overlaps with target trajectories and target nbr trajectories...')
    eval_trajectories = eval_trajectories - trajectories - nbr_trajectories
    eval_trajs = [ list(item) for item in eval_trajectories ]
    print ('')
    print ('Number of unique other\'s trajectories = ', end='')
    print (len(eval_trajs))
    print ('')
    print (datetime.datetime.now())

    if evaluate_trajectories_nbrs_file != '':
        print ('Loading other\'s nbr trajectories...')
        others_nbr_trajectories = DTM_loadTrajectoriesFile(dtm, evaluate_trajectories_nbrs_file, actions_csv)
        others_nbr_trajs = others_nbr_trajectories
        for i_idx, item in enumerate(others_nbr_trajs):
            try: # Skip if already seen.
                trajectory_ids[tuple(item)]
            except KeyError: # Hash to its position index and filename
                trajectory_ids[tuple(item)] = ( i_idx, evaluate_trajectories_nbr_files)
        others_nbr_trajectories = set()
        [ others_nbr_trajectories.add(tuple(item)) for item in others_nbr_trajs ] # Make hashable
        others_nbr_trajs = [ list(item) for item in others_nbr_trajectories ]
        print (datetime.datetime.now())
        if merged_states == True:
            print ('Merging states...')
            (dtm, graph) = dtm_states.DTM_computeMergeStates(dtm, graph, others_nbr_trajs, percent_distant_pairs = merged_percentage)
            print (datetime.datetime.now())

        # Eliminate overlap with target trajectories
        print ('Eliminating overlaps with target trajectories, target nbr trajectories, other\'s trajectories...')
        others_nbr_trajectories = others_nbr_trajectories - trajectories - nbr_trajectories - eval_trajectories
        others_nbr_trajs = [ list(item) for item in others_nbr_trajectories ]
    else:
        others_nbr_trajectories = set()
        others_nbr_trajs = list()
    print ('')
    print ('Number of unique other\'s nbr trajectories = ', end='')
    print (len(others_nbr_trajs))
    print ('')
    print (datetime.datetime.now())


#    print ('Starting IRL with original naive BKB...')
#    analysis = DTM_doIRL_naive_BKB(dtm, graph, trajectories, max_changes, discount_rate, peak, traj_prod_sum, traj_sum, incremental_prod_sum_aggregate, path_emphasis, merged_states = merged_states, merged_percentage = merged_percentage, nbr_branching = nbr_branching, reward_diversity = reward_diversity)

    all_neighbors = nbr_trajectories
    eval_nbr_trajectories = others_nbr_trajectories
    if max_changes > 0:
        num_nbrs_built = 0
        print ('Building c-neighbors from target trajectories...')
        for trajectory in trajectories:
            if nbr_branching == True:
                neighbors = dtm_neighborhood.DTM_makeBranchingNeighborhood(dtm, graph, list(trajectory), trajs)
            else:
                neighbors = dtm_neighborhood.DTM_makeNeighborhood(dtm, graph, list(trajectory), trajs, max_changes)
            num_nbrs_built += len(neighbors)
            nbrs = set()
            [ nbrs.add(tuple(item)) for item in neighbors ]
            nbrs = nbrs - trajectories - eval_trajectories
            all_neighbors |= nbrs
        print ('')
        print ('Number of c-neighbors constructed = ', end='')
        print (num_nbrs_built)
        print ('Number of unique neighbors = ', end='')
        print (len(all_neighbors))
        print ('')
        for i_idx, item in enumerate(all_neighbors):
            try: # Skip if already seen.
                trajectory_ids[tuple(item)]
            except KeyError: # Hash to its position index and filename
                trajectory_ids[tuple(item)] = ( i_idx, "Target's c-neighbors" )
        print (datetime.datetime.now())


        num_nbrs_built = 0
        print ('Building c-neighbors from other\'s trajectories...')
        for trajectory in eval_trajectories:
            if nbr_branching == True:
                neighbors = dtm_neighborhood.DTM_makeBranchingNeighborhood(dtm, graph, list(trajectory), eval_trajs)
            else:
                neighbors = dtm_neighborhood.DTM_makeNeighborhood(dtm, graph, list(trajectory), eval_trajs, max_changes)
            num_nbrs_built += len(neighbors)
            nbrs = set()
            [ nbrs.add(tuple(item)) for item in neighbors ]
            nbrs = nbrs - trajectories - all_neighbors - eval_trajectories
            eval_nbr_trajectories |= nbrs
        print ('')
        print ('Number of other\'s c-neighbors constructed = ', end='')
        print (num_nbrs_built)
        print ('Number of unique other\'s neighbors = ', end='')
        print (len(eval_nbr_trajectories))
        print ('')
        for i_idx, item in enumerate(eval_nbr_trajectories):
            try: # Skip if already seen.
                trajectory_ids[tuple(item)]
            except KeyError: # Hash to its position index and filename
                trajectory_ids[tuple(item)] = ( i_idx, "Other's c-neighbors" )
        print (datetime.datetime.now())

    # change trajectories back to lists
    trajectories = trajs
    all_neighbors = [ list(item) for item in all_neighbors ]
    eval_trajectories = eval_trajs
    eval_nbr_trajectories = [ list(item) for item in eval_nbr_trajectories ]

    print ('Starting new IRL with original naive BKB...')
    print ('')

    analysis = DTM_solveIRL_naive_BKB_others(dtm=dtm, graph=graph, target_trajs=trajectories, target_nbr_trajs=all_neighbors, others_trajs=eval_trajectories, others_nbr_trajs=eval_nbr_trajectories, trajectory_ids = trajectory_ids, discount_rate=discount_rate, peak_reward_magnitude=peak)

    if evaluate_trajectories_file != '':
        if not isinstance(evaluate_trajectories_file, list):
            eval_trajectories_file_list = [ evaluate_trajectories_file ]
        else:
            eval_trajectories_file_list = evaluate_trajectories_file
        print ('Evaluating additional trajectories...')
        for eval_file in eval_trajectories_file_list:
            print ('\tLoading trajectories for ' + eval_file + '...')
            eval_trajectories = DTM_loadTrajectoriesFile(dtm, eval_file, actions_csv)
            print (datetime.datetime.now())
            print ('\tGenerating additional trajectory expected values --')
            print ('\t\tTrajectory probability does not include discount factor.')
            analysis.append([ 'Trajectory list -', eval_file])
            analysis.append([ 'Additional Trajectory Id', 'Trajectory probability', 'Reward sum', 'Expected linear reward with discount', 'Trajectory probability x Reward sum', 'Trajectory' ])
            for traj_idx, trajectory in enumerate(eval_trajectories):
                prob = 1.0
                for idx in range(0, len(trajectory) - 1, 2):
                    trip = dtm.triples[graph[0][trajectory[idx]][trajectory[idx + 1]][trajectory[idx + 2]]]
                    prob *= trip.prob
                try:
                    analysis_item = [ trajectory_ids[tuple(trajectory)], prob ]
                except KeyError:
                    sys.exit("Unable to find tracjector <{}>!".format(trajectory))
#                analysis_item = [ 'Trajectory #' + str(traj_idx + 1), prob ]
                reward = 0
                linear_expected = 0
                discount = 1.0
                for idx in range(0, len(trajectory) - 1, 2):
                    trip = dtm.triples[graph[0][trajectory[idx]][trajectory[idx + 1]][trajectory[idx + 2]]]
                    reward += trip.reward
                    linear_expected += trip.reward * trip.prob * discount
                    discount *= discount_rate
                analysis_item.append(reward)
                analysis_item.append(linear_expected)
                analysis_item.append(prob * reward)
                analysis_item.append(trajectory)
                analysis.append(analysis_item)
            print (datetime.datetime.now())
    print ('Saving IRL analyses to csv...')
    """
    with open(rewards_csv_file, 'w') as rewards_csv:
        writer = csv.writer(rewards_csv)
        rows = list()
        rows.append([ 'Start state index', 'Action index', 'Finish state index', 'Probability for Triple', 'Reward for Triple', 'Prob x Reward' ])
        for sidx, value1 in graph[0].items():
            for aidx, value2 in value1.items():
                for didx, tidx in value2.items():
                    if tidx != -1:
                        triple = dtm.triples[tidx]
                        rows.append([ sidx, aidx, didx, triple.prob, triple.reward, triple.prob * triple.reward ])
        writer.writerows(rows)
        writer.writerows(analysis)
    print (datetime.datetime.now())
    """

    if dtm_report_csvfile != '':
        print ('Generating DTM report...')
        print (datetime.datetime.now())
        DTM_generateReport(dtm, dtm_report_csvfile)

    del trajectory_ids

# Analyze trajectories
def DTM_trajectoryAnalysis(dtmfile, trajectories_file, actions_csv, dist = None):
    print ('Trajectory analysis ---')
    print (datetime.datetime.now())
    if isinstance(dtmfile, str):
        print (datetime.datetime.now())
        print ('Loading dtm...')
        dtm = pydtm.pydtm(dtmfile)
        print (datetime.datetime.now())
    else:
        dtm = dtmfile
    print ('Loading trajectories...')
    trajectories = DTM_loadTrajectoriesFile(dtm, trajectories_file, actions_csv)
    print (datetime.datetime.now())

    print ('Note - Shared states/actions between two trajectories a and b are listed as compound tuples ( ( state/action idx, trajectory a index - 1 ), ( state/action idx, trajectory b index - 1 ) )')
    print ('')
    for t_idx1, traj1 in enumerate(trajectories):
        print ('Trajectory #' + str(t_idx1 + 1) + ' :')
        print ('\t# of states = ' + str(len(traj1) / 2 + 1))
        print ('\t# of actions = ' + str(len(traj1)))
        for t_idx2, traj2 in enumerate(trajectories):
            if t_idx1 == t_idx2:
                continue
            shared_states = list()
            shared_actions = list()
            for e_idx1, e_entry1 in enumerate(traj1):
                for e_idx2, e_entry2 in enumerate(traj2):
                    if e_idx1 % 2 == 0 and e_idx2 % 2 == 0: # state
                        if e_entry1 == e_entry2:
                            shared_states.append( ( ( e_entry1, e_idx1 ), ( e_entry2, e_idx2 ) ) )
                    elif e_idx1 % 2 == 1 and e_idx2 % 2 == 1: # action
                        if e_entry1 == e_entry2:
                            shared_actions.append( ( ( e_entry1, e_idx1 ), ( e_entry2, e_idx2 ) ) )
            print ('\t\t# shared states with Trajectory #' + str(t_idx2 + 1) + ' = ' + str(len(shared_states)) + ' : ', end='')
            print (shared_states, end='')
            if dist == None:
                print('')
            else:
                print ('\tdistance = ', end='')
                d = dtm_states.DTM_computeTrajectoryDistance(traj1, traj2, dist)
                print (d)
            print ('\t\t# shared actions with Trajectory #' + str(t_idx2 + 1) + ' = ' + str(len(shared_actions)) + ' : ', end='')
            print ('')
#            print (shared_actions)
