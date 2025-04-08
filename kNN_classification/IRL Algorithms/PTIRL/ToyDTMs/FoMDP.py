#
# Filename:     FoMDP.py
# Date:         2020-06-14
# Project:      Feature-organizing MDP
# Author:       Eugene Santos Jr.
# Copyright     Eugene Santos Jr.
#
#
import sys
from math import floor
import os
import csv
import copy
import RoMDP
import L_MDP
import parse
import time
import tqdm
#import gurobipy
#from gurobipy import GRB

DEBUG = True
OPTIMIZER = None # BARON or GUROBI
GUROBI_MODEL = None
GUROBI_VARIABLES = None
GUROBI_CONSTRAINT_NAME = None
GUROBI_IntFeasTol = 1e-9 # Integrality feasibility tolerance
GUROBI_INTEGRALS = set() # Set of variables for GUROBI that are INTEGER or BINARY

# You can also ignore masking and let the system group states independent
#   of features by setting simplify_masking to True
#
# Notation:
#   mask        -- "M_{}".format(k)
#                   -- This is the mask for the kth feature
#                   -- k starts from 0
#
#   states      -- "s_{}".format(i)
#                   -- this is the ith state
#                   -- states index from 0
#               -- "s_{}_<{},{}>".format(i, t, b)
#                   -- trajectory t, position b with state i
#               -- "s_{}_{}".format(i, k)
#                   -- This is the flag that feature k is on in state i
#
#   features   -- "num_features"
#                   -- number of features for each state.
#               -- "d_{}_{}_{}".format(i, i', k)
#                   -- if 0, indicates that s_{i} and s_{i'} have the same
#                       feature value for kth feature, otherwise is different
#                   -- depends on selected mask
#               -- "D_{}_{}".format(i, i')
#                   -- if 0, indicates feature values are identical between
#                       s_{i} and s_{i'}
#                   -- depends on selected mask
#   classes     -- "num_classes"
#                   -- number of equivalence classes available
#               -- "q_{}_{}".format(i, j)
#                   -- if 1, indicates that state i has been assigned to
#                       equivalence class j, otherwise not assigned to j
#                   -- j starts from 0
#               -- "Q_{}".format(i)
#                   -- the numerical value for the class that state i has
#                       been assigned to.
#               -- "DB_{}_{}".format(i, i')
#                   -- if 0, then state i and state i' have the same
#                       assignment, otherwise, they are different
#               -- "FQ_{}_{}_{}_{}".format(j, i, j', i')
#                   -- equal to q_{i}_{j} & q_{i'}_{j'}
#   counts      -- "Cq_targ_{}".format(j)
#                   -- # of occurrences of states in class j in
#                       decision-maker trajectories
#               -- "Cq_nontarg_{}".format(j)
#                   -- # of occurrences of states in class j in
#                       non-decision-maker trajectories
#               -- "Cq_targ_{}_{}".format(j, a)
#                   -- # of occurrences of {s, a} for s in class j in
#                       decision-maker trajectories
#               -- "Cq_nontarg_{}_{}".format(j, a)
#                   -- # of occurrences of {s, a} for s in class j in
#                       non-decision-maker trajectories
#               -- "Cq_{}_{}".format(j, a)
#                   -- # of occurrences of {s, a} for s in class j in all
#                       trajectories
#               -- "Cq_{}_{}_{}".format(j, a, j')
#                   -- # of occurrences of {s, a, s'} for s in class j and
#                       s' in j' in all trajectories
#               -- "Pq_targ_{}_{}".format(a, j)
#                   -- "Cq_targ_{}_{}".format(j, a) / "Cq_targ_{}".format(j)
#               -- "Pq_nontarg_{}_{}".format(a, j)
#                   -- "Cq_nontarg_{}_{}".format(j, a)
#                       / "Cq_nontarg_{}".format(j)
#               -- "Pq_{}_{}_{}".format(j', a, j)
#                   -- "Cq_{}_{}_{}".format(j, a, j')
#                       / "Cq_{}_{}".format(j, a)
#               -- "P_targ_{}_{}".format(a, i)
#                   -- probability of action a given state i for decision-maker
#               -- "P_nontarg_{}_{}".format(a, i)
#                   -- probability of action a given state i for
#                       non-decision-maker
#               -- "P_{}_{}_{}".format(i', a, i)
#                   -- probablity of state i' given action a and state i
# rewards       -- "r_{}_{}_{}".format(i, a, i')
#                   -- reward for triple
#               -- "FRV_targ_{}_{}_{}".format(i, a, i')
#                   -- fractional reward for decision-maker
#               -- "FRV_nontarg_{}_{}_{}".format(i, a, i')
#                   -- fractional reward for other decision-maker
#               -- "FRV_maxMagnitude"
#                   -- maximum magnitude for all fractional rewards
#               -- "rq_{}_{}_{}".format(j, a, j')
#   actions     -- "a_{}".format(i)
#                   -- this is the ith action
#                   -- actions index from 0
#               -- "a_{}--({},{})".format(i, a, b)
#                   -- the ath trajectory's bth action which is action i
#   trajectories
#               -- of the form: [ "s_{}_<{}>--({},{})".format(i_0, k, a, 0), "a_{}--({},{})".format(j_0, a, 0), "s_{}_<{}>--({},{})".format(i_1, k, a, 1), "a_{}--({},{})".format(j_1, a, 1), ... ]
#                   -- is finite in length
#               -- "length(a)"
#                   -- is the number of states in the trajectory sequence (number of actions is 1 less)
#                   -- note state and action indexing "b"
#               -- "num_trajectories"
#                   -- is the total number of trajectories
#               -- "posets"
#                   -- is an array of disjoint sets of trajectories
#                   -- Note - this is a total ordering between sets, we can generalize later. index 0 is the dominant poset.
#               -- "num_posets"
#                   -- number of posets
#   hidden values
#               -- values start from 0
#               -- "T"
#                   -- maximum number of hidden values used across each state
#               -- "tau"
#                   -- maximum number of usable hidden values for any state
#
#   rewards     -- "R_[s_{}_<{}>,a_{},s_{}_<{}>]".format(i, k, j, i', k')
#                   -- reward value to transition from state i hidden value k with action j to state i' hidden value k'
#               -- "peak"
#                   -- non-negative real value representing largest magnitude for all reward values

#
# Dictionaries:
#
#   actions_desc_to_idx             -- dictionary from an action string descriptor to integer action index
#   state_desc_to_idx               -- dictionary from a state string descriptor to integer state index
#   var_desc_to_var                 -- dictionary mapping from the descriptor to name (global)
#   var_to_var_desc                 -- dictionary mapping from name to descriptor (global)
#   equation_desc_to_equation       -- dictionary mapping from the descrtiptor to name (global)
#   equation_to_equation_desc       -- dictionary mapping from name to descriptor (global)

var_desc_to_var = dict()
var_to_var_desc = dict()
equation_desc_to_equation = dict()
equation_to_equation_desc = dict()

#
# Other globals
#
#   next_var_idx                    -- next available integer index for vars (global)
#   next_equation_idx               -- next available integer index for equations (global)
next_var_idx = 0
next_equation_idx = 0

#
# Variables: (bounds are establish at time of declaration)
#
#   "M_{}".format(k)
#       -- binary variable (0 = False, 1 = True)
#   "s_{}_{}".format(i, k)
#       -- non-negative real-valued variable bounded by [0, 1]
#   "d_{}_{}_{}".format(i, i', k)
#       -- non-negative real-valued variable bounded by [0, 1]
#   "D_{}_{}".format(i, i')
#       -- non-negative real-valued variable bounded by [0, 1]
#   "q_{}_{}".format(i, j)
#       -- binary variable
#   "Q_{}".format(i)
#       -- non-negative real-valued variable bounded by [1, 2^(num_classes-1)]
#   "DB_{}_{}".format(i, i')
#       -- binary variable
#   "FQ_{}_{}_{}_{}".format(j, i, j', i')
#       -- binary variable
#   "Cq_targ_{}".format(j)
#       -- integer variable
#   "Cq_nontarg_{}".format(j)
#       -- integer variable
#   "Cq_targ_{}_{}".format(j, a)
#       -- integer variable
#   "Cq_nontarg_{}_{}".format(j, a)
#       -- integer variable
#   "Cq_{}_{}".format(j, a)
#       -- integer variable
#   "Cq_{}_{}_{}".format(j, a, j')
#       -- integer variable
#   "Pq_targ_{}_{}".format(a, j)
#       -- non-negative real-valued variable bounded by [0, 1]
#   "Pq_nontarg_{}_{}".format(a, j)
#       -- non-negative real-valued variable bounded by [0, 1]
#   "Pq_{}_{}_{}".format(j', a, j)
#       -- non-negative real-valued variable bounded by [0, 1]
#   "P_targ_{}_{}".format(a, i)
#       -- non-negative real-valued variable bounded by [0, 1]
#   "P_nontarg_{}_{}".format(a, i)
#       -- non-negative real-valued variable bounded by [0, 1]
#   "P_{}_{}_{}".format(i', a, i)
#       -- non-negative real-valued variable bounded by [0, 1]
#   "r_{}_{}_{}".format(i, a, i')
#       -- real-valued variable bounded by [-peak, peak]
#   "FRV_targ_{}_{}_{}".format(i, a, i')
#       -- real-valued variable bounded by [-2 * peak, 2 * peak]
#   "FRV_nontarg_{}_{}_{}".format(i, a, i')
#       -- real-valued variable bounded by [-2 * peak, 2 * peak]
#   "FRV_maxMagnitude"
#       -- non-negative real-valued variable maximum magnitude for all fractional rewards
#   "rq_{}_{}_{}".format(j, a, j')
#       -- real-valued variable bounded by [-peak, peak]
#   "LER_{}".format(t)
#       -- real valued variable corresponding to the linear expected reward for trajectory t (without discounting)
#       -- unbounded
#   "LB_{}".format(u)
#   "UB_{}".format(u)
#       -- real valued variable corresponding to the lower and upper bounds of LER for trajectories in poset u
#   "delta_{}_{}".format(u, u+1)
#       -- real valued variable corresponding to the distance between two posets u and u+1 (assumed total ordering)
#       -- bounded by [1, inf)


#
# Constraints: (unbound indices are assumed universal)
#
#   if "s_{}_{}".format(i, k) == 0: i.e., the value for
#                                       feature k for state i is 0
#       "s_{}_{}".format(i, k) = 0
#   else:
#       "s_{}_{}".format(i, k) - "M_{}".format(k) = 0
#       -- state i feature k
#
#   "d_{}_{}_{}".format(i, i', k) - "s_{}_{}".format(i, k) -
#       "s_{}_{}".format(i', k) <= 0
#   "d_{}_{}_{}".format(i, i', k) - "s_{}_{}".format(i, k) +
#       "s_{}_{}".format(i', k) >= 0
#   "d_{}_{}_{}".format(i, i', k) + "s_{}_{}".format(i, k) -
#       "s_{}_{}".format(i', k) >= 0
#   "d_{}_{}_{}".format(i, i', k) + "s_{}_{}".format(i, k) +
#       "s_{}_{}".format(i', k) <= 2
#       -- states i and i', feature k
#
#   "D_{}_{}".format(i, i') -
#       \sum_{k=0}^{num_features - 1} "d_{}_{}_{}".format(i, i', k) <= 0
#       -- states i and i'
#   "D_{}_{}".format(i, i') - "d_{}_{}_{}".format(i, i', k) >= 0
#       -- states i and i', feature k
#
#   \sum_{j=0}^{num_classes - 1} "q_{}_{}".format(i, j) = 1
#       -- state i
#
#   "Q_{}".format(i) - \sum_{j=0}^{num_classes - 1} 2^j * "q_{}_{}".format(i, j)
#       = 0
#       -- state i
#
#   2^num_classes * "D_{}_{}".format(i, i') - "Q_{}".format(i)
#       + "Q_{}".format(i') >= 0
#   2^num_classes * "D_{}_{}".format(i, i') + "Q_{}".format(i)
#       - "Q_{}".format(i') >= 0
#   "Q_{}".format(i) - "Q_{}".format(i') - "D_{}_{}".format(i, i')
#       + 2^num_classes * "DB_{}_{}".format(i, i') >= 0
#   "Q_{}".format(i') - "Q_{}".format(i) - "D_{}_{}".format(i, i')
#       - 2^num_classes * "DB_{}_{}".format(i, i') >= -2^num_classes
#       -- states i and i'
#
#   "FQ_{}_{}_{}_{}".format(j, i, j', i') - "q_{}_{}".format(i, j) <= 0
#   "FQ_{}_{}_{}_{}".format(j, i, j', i') - "q_{}_{}".format(i', j') <= 0
#   "FQ_{}_{}_{}_{}".format(j, i, j', i') - "q_{}_{}".format(i, j)
#       - "q_{}_{}".format(i', j') >= -1
#       -- states i and i', classes j and j'
#
#   "Cq_targ_{}".format(j) - \sum_{i=0}^{num_states - 1} C_{DM}(i)
#       * "q_{}_{}".format(i, j) = 0
#       -- class j
#   "Cq_targ_{}_{}".format(j, a) - \sum_{i=0}^{num_states - 1} C_{DM}(i, a)
#       * "q_{}_{}".format(i, j) = 0
#       -- class j and action a
#
#   "Cq_nontarg_{}".format(j) - \sum_{i=0}^{num_states - 1} C_{NDM}(i)
#       * "q_{}_{}".format(i, j) = 0
#       -- class j
#   "Cq_nontarg_{}_{}".format(j, a) - \sum_{i=0}^{num_states - 1} C_{NDM}(i, a)
#       * "q_{}_{}".format(i, j) = 0
#
#   "Cq_{}_{}".format(j, a) - \sum{i=0}^{num_states - 1} C(i, a)
#      * "q_{}_{}".format(i, j) = 0
#       -- class j and action a
#   "Cq_{}_{}_{}".format(j, a, j') - \sum{i=0}^{num_states - 1}
#       \sum{i'=0}^{num_states - 1} C(i, a, i')
#       * "FQ_{}_{}_{}_{}".format(j, i, j', i') = 0
#       -- classes j and j' and action a
#       -- Must flip i and i' if i>i' in FQ term
#
#   "Cq_targ_{}".format(j) * "Pq_targ_{}_{}".format(a, j)
#       - "Cq_targ_{}_{}".format(j, a) = 0
#       -- class j and action a
#
#   "Cq_nontarg_{}".format(j) * "Pq_nontarg_{}_{}".format(a, j)
#       - "Cq_nontarg_{}_{}".format(j, a) = 0
#       -- class j and action a
#
#   "Cq_{}_{}".format(j, a) * "Pq_{}_{}_{}".format(j', a, j)
#       - "Cq_{}_{}_{}".format(j, a, j') = 0
#       -- classes j and j' and action a
#
#   "Pq_targ_{}_{}".format(a, j) - "P_targ_{}_{}".format(a, i)
#       - "q_{}_{}".format(i, j) >= -1
#   "P_targ_{}_{}".format(a, i) - "Pq_targ_{}_{}".format(a, j)
#       - "q_{}_{}".format(i, j) >= -1
#       -- actin a, class j, state i
#
#   "Pq_nontarg_{}_{}".format(a, j) - "P_nontarg_{}_{}".format(a, i)
#       - "q_{}_{}".format(i, j) >= -1
#   "P_nontarg_{}_{}".format(a, i) - "Pq_nontarg_{}_{}".format(a, j)
#       - "q_{}_{}".format(i, j) >= -1
#       -- actin a, class j, state i
#
#   "Pq_{}_{}_{}".format(j', a, j) - "P_{}_{}_{}".format(i', a, i)
#       - "q_{}_{}".format(i, j) - "q_{}_{}".format(i', j') >= -2
#   "P_{}_{}_{}".format(i', a, i) - "Pq_{}_{}_{}".format(j', a, j)
#       - "q_{}_{}".format(i, j) - "q_{}_{}".format(i', j') >= -2
#       -- classes j, j', action a, states i, i'
#
#   "rq_{}_{}_{}".format(j, a, j') - "r_{}_{}_{}".format(i, a, i')
#       - 2 * peak * "q_{}_{}".format(i, j) - 2 * peak * "q_{}_{}".format(i', j') >= -4 * peak
#   "r_{}_{}_{}".format(i, a, i') - "rq_{}_{}_{}".format(j, a, j')
#       - 2 * peak * "q_{}_{}".format(i, j) - 2 * peak * "q_{}_{}".format(i', j') >= -4 * peak
#
#   if {s_i, a, s_i'} occurs in a dm trajectory:
#       "r_{}_{}_{}".format(i, a, i') - peak * "P_targ_{}_{}".format(a, i')
#           - "FRV_targ_{}_{}_{}".format(i, a, i') <= 0
#   if {s_i, a, s_i'} does not occur in a dm trajectory:
#       "r_{}_{}_{}".format(i, a, i') + peak * "P_nontarg_{}_{}".format(a, i')
#           - "FRV_nontarg_{}_{}_{}".format(i, a, i') <= 0
#
#   "FRV_targ_{}_{}_{}".format(i, a, i') <= FRV_maxMagnitude
#   -"FRV_targ_{}_{}_{}".format(i, a, i') <= FRV_maxMagnitude
#   "FRV_nontarg_{}_{}_{}".format(i, a, i') <= FRV_maxMagnitude
#   -"FRV_nontarg_{}_{}_{}".format(i, a, i') <= FRV_maxMagnitude
#
#   "LER_{}".format(t) = \sum_{b=0}^{length(t)-1}
#       "P_{}_{}_{}".format(s_{i}_<{t},{b}>, a_{a}_<{t},{b+1}>,
#       s_{i}_<{t},{b+1}>) * "r_{}_{}_{}".format(s_{i}_<{t},{b}>,
#        a_{a}_<{t},{b+1}>, s_{i}_<{t},{b+1}>)
#       -- computes LER for trajectory a
#
#   "LB_{}".format(u) <= UB_{}".format(u)
#
#   "LB_{}".format(u) = "UB_{}".format(u+1) + "delta_{}_{}".format(u, u+1)
#       -- orders posets
#
#   "LB_{}".format(u) <= "LER_{}".format(t)
#   "UB_{}".format(u) >= "LER_{}".format(t)
#   "LB_{}".format(u) <= "LER_{}".format(t)
#   "UB_{}".format(u) >= "LER_{}".format(t)
#       -- only if trajectory t is in posets[u]


#
# Objective function:
#    -- Version 0
#        No objective - feasibility only
#        min 1
#   -- Version 1
#       min T
#       -- Finds the smallest number of hidden values needed to satisfy the constraints and given trajectories
#       -- Underfit can be avoided by choice of lower bound for "delta_{}_{}".format(q, q+1)
#       -- This provides a feasibility guarantee that separation of trajectory collections can occur while not overfitting. Furthermore, we can provide an ANOVA-like analysis that looks at the effects of varying which set of separation constraints are enforced. Plot separation vs. number of hidden states and identify crossover point, i.e., how big of a separation until T changes to the next higher value.
#
#   -- Version 2
#       max \sum_{q=0}{num_posets-1} "delta_{}_{}".format(q, q+1)
#       -- maximizes the distance between posets
#       -- can overfit trajectories

#
# Generate variable strings

def genVariableName(desc, prefix):
    global next_var_idx
    try:
        return (var_desc_to_var[desc])
    except KeyError:
        name = "{}{}".format(prefix, next_var_idx)
        next_var_idx += 1
        var_desc_to_var[desc] = name
        var_to_var_desc[name] = desc
        return (name)

#   "M_{}".format(k)
#       -- binary variable (0 = False, 1 = True)
def genM(feature_idx):
    desc = "M_{}".format(feature_idx)
    return (genVariableName(desc, "M"))

#   "s_{}_{}".format(i, k)
#       -- non-negative real-valued variable bounded by [0, 1]
def gens(state_idx, feature_idx):
    desc = "s_{}_{}".format(state_idx, feature_idx)
    return (genVariableName(desc, "s"))

#   "d_{}_{}_{}".format(i, i', k)
#       -- non-negative real-valued variable bounded by [0, 1]
def gend(state_idx1, state_idx2, feature_idx):
    desc = "d_{}_{}_{}".format(state_idx1, state_idx2, feature_idx)
    return (genVariableName(desc, "d"))

#   "D_{}_{}".format(i, i')
#       -- non-negative real-valued variable bounded by [0, 1]
def genD(state_idx1, state_idx2):
    desc = "D_{}_{}".format(state_idx1, state_idx2)
    return (genVariableName(desc, "D"))

#   "q_{}_{}".format(i, j)
#       -- binary variable
def genq(state_idx, class_idx):
    desc = "q_{}_{}".format(state_idx, class_idx)
    return (genVariableName(desc, "q"))

#   "Q_{}".format(i)
#       -- non-negative real-valued variable bounded by [1, 2^(num_classes-1)]
def genQ(state_idx):
    desc = "Q_{}".format(state_idx)
    return (genVariableName(desc, "Q"))

#   "DB_{}_{}".format(i, i')
#       -- binary variable
def genDB(state_idx1, state_idx2):
    desc = "DB_{}_{}".format(state_idx1, state_idx2)
    return (genVariableName(desc, "DB"))

#   "FQ_{}_{}_{}_{}".format(j, i, j', i')
#       -- binary variable
def genFQ(class_idx1, state_idx1, class_idx2, state_idx2):
    desc = "FQ_{}_{}_{}_{}".format(class_idx1, state_idx1, class_idx2, state_idx2)
    return (genVariableName(desc, "FQ"))

#   "Cq_targ_{}".format(j)
#       -- integer variable
def genCq_targ1(class_idx):
    desc = "Cq_targ1_{}".format(class_idx)
    return (genVariableName(desc, "Cqt1_"))

#   "Cq_nontarg_{}".format(j)
#       -- integer variable
def genCq_nontarg1(class_idx):
    desc = "Cq_nontarg1_{}".format(class_idx)
    return (genVariableName(desc, "Cqn1_"))

#   "Cq_targ_{}_{}".format(j, a)
#       -- integer variable
def genCq_targ2(class_idx, action_idx):
    desc = "Cq_targ_{}_{}".format(class_idx, action_idx)
    return (genVariableName(desc, "Cqt2_"))

#   "Cq_nontarg_{}_{}".format(j, a)
#       -- integer variable
def genCq_nontarg2(class_idx, action_idx):
    desc = "Cq_nontarg_{}_{}".format(class_idx, action_idx)
    return (genVariableName(desc, "Cqn2_"))

#   "Cq_{}_{}".format(j, a)
#       -- integer variable
def genCq1(class_idx, action_idx):
    desc = "Cq_{}_{}".format(class_idx, action_idx)
    return (genVariableName(desc, "Cq1_"))

#   "Cq_{}_{}_{}".format(j, a, j')
#       -- integer variable
def genCq2(class_idx1, action_idx, class_idx2):
    desc = "Cq_{}_{}_{}".format(class_idx1, action_idx, class_idx2)
    return (genVariableName(desc, "Cq2_"))

#   "Pq_targ_{}_{}".format(a, j)
#       -- non-negative real-valued variable bounded by [0, 1]
def genPq_targ(action_idx, class_idx):
    desc = "Pq_targ_{}_{}".format(action_idx, class_idx)
    return (genVariableName(desc, "Pqt"))

#   "Pq_nontarg_{}_{}".format(a, j)
#       -- non-negative real-valued variable bounded by [0, 1]
def genPq_nontarg(action_idx, class_idx):
    desc = "Pq_nontarg_{}_{}".format(action_idx, class_idx)
    return (genVariableName(desc, "Pqn"))

#   "Pq_{}_{}_{}".format(j', a, j)
#       -- non-negative real-valued variable bounded by [0, 1]
def genPq(class_idx2, action_idx, class_idx1):
    desc = "Pq_{}_{}_{}".format(class_idx2, action_idx, class_idx1)
    return (genVariableName(desc, "Pq"))

#   "P_targ_{}_{}".format(a, i)
#       -- non-negative real-valued variable bounded by [0, 1]
def genP_targ(action_idx, state_idx):
    desc = "P_targ_{}_{}".format(action_idx, state_idx)
    return (genVariableName(desc, "Pt"))

#   "P_nontarg_{}_{}".format(a, i)
#       -- non-negative real-valued variable bounded by [0, 1]
def genP_nontarg(action_idx, state_idx):
    desc = "P_nontarg_{}_{}".format(action_idx, state_idx)
    return (genVariableName(desc, "Pn"))

#   "P_{}_{}_{}".format(i', a, i)
#       -- non-negative real-valued variable bounded by [0, 1]
def genP(state_idx2, action_idx, state_idx1):
    desc = "P_{}_{}_{}".format(state_idx2, action_idx, state_idx1)
    return (genVariableName(desc, "P"))

#   "r_{}_{}_{}".format(i, a, i')
#       -- real-valued variable bounded by [-peak, peak]
def genr(state_idx1, action_idx, state_idx2):
    desc = "r_{}_{}_{}".format(state_idx1, action_idx, state_idx2)
    return (genVariableName(desc, "r"))

#   "FRV_targ_{}_{}_{}".format(i, a, i')
#       -- real-valued variable bounded by [-2 * peak, 2 * peak]
def genFRV_targ(state_idx1, action_idx, state_idx2):
    desc = "FRV_targ_{}_{}_{}".format(state_idx1, action_idx, state_idx2)
    return (genVariableName(desc, "FT"))

#   "FRV_nontarg_{}_{}_{}".format(i, a, i')
#       -- real-valued variable bounded by [-2 * peak, 2 * peak]
def genFRV_nontarg(state_idx1, action_idx, state_idx2):
    desc = "FRV_nontarg_{}_{}_{}".format(state_idx1, action_idx, state_idx2)
    return (genVariableName(desc, "FN"))

#   "FRV_maxMagnitude"
#       -- real-valued variable
def genFRV_maxMagnitude():
    desc = "FRV_maxMagnitude"
    return (genVariableName(desc, "FMAX"))

#   "rq_{}_{}_{}".format(j, a, j')
#       -- real-valued variable bounded by [-peak, peak]
def genrq(class_idx1, action_idx, class_idx2):
    desc = "rq_{}_{}_{}".format(class_idx1, action_idx, class_idx2)
    return (genVariableName(desc, "rq"))

#   "LER_{}".format(t)
#       -- real valued variable corresponding to the linear expected reward for trajectory t (without discounting)
#       -- unbounded
def genLER(traj_idx):
    desc = "LER_{}".format(traj_idx)
    return (genVariableName(desc, "LER"))

#   "LB_{}".format(u)
#   "UB_{}".format(u)
#       -- real valued variable corresponding to the lower and upper bounds of LER for trajectories in poset u
def genLB(poset_idx):
    desc = "LB_{}".format(poset_idx)
    return (genVariableName(desc, "LB"))

def genUB(poset_idx):
    desc = "UB_{}".format(poset_idx)
    return (genVariableName(desc, "UB"))

#   "delta_{}_{}".format(u, u+1)
#       -- real valued variable corresponding to the distance between two posets u and u+1 (assumed total ordering)
#       -- bounded by [1, inf)
def gendelta(poset_idx1, poset_idx2):
    desc = "delta_{}_{}".format(poset_idx1, poset_idx2)
    return (genVariableName(desc, "delta"))

#
# Generate variable type -- Optimizer specific (BARON)

def typeM():
    return ("BINARY_VARIABLE")

def types():
    return ("BINARY_VARIABLE")

def typed():
    return ("BINARY_VARIABLE")

def typeD():
    return ("BINARY_VARIABLE")

def typeq():
    return ("BINARY_VARIABLE")

def typeQ():
    return ("POSITIVE_VARIABLE")

def typeDB():
    return ("BINARY_VARIABLE")

def typeFQ():
    return ("BINARY_VARIABLE")

def typeCq_targ1():
    return ("INTEGER_VARIABLE")

def typeCq_nontarg1():
    return ("INTEGER_VARIABLE")

def typeCq_targ2():
    return ("INTEGER_VARIABLE")

def typeCq_nontarg2():
    return ("INTEGER_VARIABLE")

def typeCq1():
    return ("INTEGER_VARIABLE")

def typeCq2():
    return ("INTEGER_VARIABLE")

def typePq_targ():
    return ("POSITIVE_VARIABLE")

def typePq_nontarg():
    return ("POSITIVE_VARIABLE")

def typePq():
    return ("POSITIVE_VARIABLE")

def typeP_targ():
    return ("POSITIVE_VARIABLE")

def typeP_nontarg():
    return ("POSITIVE_VARIABLE")

def typeP():
    return ("POSITIVE_VARIABLE")

def typer():
    return ("VARIABLE")

def typeFRV_targ():
    return ("VARIABLE")

def typeFRV_nontarg():
    return ("VARIABLE")

def typeFRV_maxMagnitude():
    return ("POSITIVE_VARIABLE")

def typerq():
    return ("VARIABLE")

def typeLER():
    return ("VARIABLE")

def typeLB():
    return ("VARIABLE")

def typeUB():
    return ("VARIABLE")

def typedelta():
    return ("POSITIVE_VARIABLE")

#
# Generate variable bounds based on variable strings -- Optimizer specific (BARON)

def lowerboundQ(state_idx):
    if OPTIMIZER == "GUROBI":
        GUROBI_VARIABLES[genQ(state_idx)].lb = 1.0
        return (None)
    return ("{}: 1;".format(genQ(state_idx)))

def upperboundQ(state_idx, num_classes):
    if OPTIMIZER == "GUROBI":
        GUROBI_VARIABLES[genQ(state_idx)].ub = 2 ** (num_classes-1)
        return (None)
    return ("{}: {};".format(genQ(state_idx), 2**(num_classes-1)))

def upperboundPq_targ(action_idx, class_idx):
    if OPTIMIZER == "GUROBI":
        GUROBI_VARIABLES[genPq_targ(action_idx, class_idx)].ub = 1.0
        return (None)
    return ("{}: 1;".format(genPq_targ(action_idx, class_idx)))

def upperboundPq_nontarg(action_idx, class_idx):
    if OPTIMIZER == "GUROBI":
        GUROBI_VARIABLES[genPq_nontarg(action_idx, class_idx)].ub = 1.0
        return (None)
    return ("{}: 1;".format(genPq_nontarg(action_idx, class_idx)))

def upperboundPq(class_idx2, action_idx, class_idx1):
    if OPTIMIZER == "GUROBI":
        GUROBI_VARIABLES[genPq(class_idx2, action_idx, class_idx1)].ub = 1.0
        return (None)
    return ("{}: 1;".format(genPq(class_idx2, action_idx, class_idx1)))

def upperboundP_targ(action_idx, state_idx):
    if OPTIMIZER == "GUROBI":
        GUROBI_VARIABLES[genP_targ(action_idx, state_idx)].ub = 1.0
        return (None)
    return ("{}: 1;".format(genP_targ(action_idx, state_idx)))

def upperboundP_nontarg(action_idx, state_idx):
    if OPTIMIZER == "GUROBI":
        GUROBI_VARIABLES[genP_nontarg(action_idx, state_idx)].ub = 1.0
        return (None)
    return ("{}: 1;".format(genP_nontarg(action_idx, state_idx)))

def upperboundP(state_idx2, action_idx, state_idx1):
    if OPTIMIZER == "GUROBI":
        GUROBI_VARIABLES[genP(state_idx2, action_idx, state_idx1)].ub = 1.0
        return (None)
    return ("{}: 1;".format(genP(state_idx2, action_idx, state_idx1)))

def lowerboundr(state_idx1, action_idx, state_idx2, peak):
    if OPTIMIZER == "GUROBI":
        GUROBI_VARIABLES[genr(state_idx1, action_idx, state_idx2)].lb = -peak
        return (None)
    return ("{}: {};".format(genr(state_idx1, action_idx, state_idx2), -peak))

def upperboundr(state_idx1, action_idx, state_idx2, peak):
    if OPTIMIZER == "GUROBI":
        GUROBI_VARIABLES[genr(state_idx1, action_idx, state_idx2)].ub = peak
        return (None)
    return ("{}: {};".format(genr(state_idx1, action_idx, state_idx2), peak))

def lowerboundFRV_targ(state_idx1, action_idx, state_idx2, peak):
    if OPTIMIZER == "GUROBI":
        GUROBI_VARIABLES[genFRV_targ(state_idx1, action_idx, state_idx2)].lb = - 2 * peak
        return (None)
    return ("{}: {};".format(genFRV_targ(state_idx1, action_idx, state_idx2), - 2 * peak))

def upperboundFRV_targ(state_idx1, action_idx, state_idx2, peak):
    if OPTIMIZER == "GUROBI":
        GUROBI_VARIABLES[genFRV_targ(state_idx1, action_idx, state_idx2)].ub = 2 * peak
        return (None)
    return ("{}: {};".format(genFRV_targ(state_idx1, action_idx, state_idx2),  2 * peak))

def lowerboundFRV_nontarg(state_idx1, action_idx, state_idx2, peak):
    if OPTIMIZER == "GUROBI":
        GUROBI_VARIABLES[genFRV_nontarg(state_idx1, action_idx, state_idx2)].lb = - 2 * peak
        return (None)
    return ("{}: {};".format(genFRV_nontarg(state_idx1, action_idx, state_idx2), - 2 * peak))

def upperboundFRV_nontarg(state_idx1, action_idx, state_idx2, peak):
    if OPTIMIZER == "GUROBI":
        GUROBI_VARIABLES[genFRV_nontarg(state_idx1, action_idx, state_idx2)].ub = 2 * peak
        return (None)
    return ("{}: {};".format(genFRV_nontarg(state_idx1, action_idx, state_idx2),  2 * peak))

def lowerboundrq(class_idx1, action_idx, class_idx2, peak):
    if OPTIMIZER == "GUROBI":
        GUROBI_VARIABLES[genrq(class_idx1, action_idx, class_idx2)].lb = - peak
        return (None)
    return ("{}: {};".format(genrq(class_idx1, action_idx, class_idx2), -peak))

def upperboundrq(class_idx1, action_idx, class_idx2, peak):
    if OPTIMIZER == "GUROBI":
        GUROBI_VARIABLES[genrq(class_idx1, action_idx, class_idx2)].ub = peak
        return (None)
    return ("{}: {};".format(genrq(class_idx1, action_idx, class_idx2), peak))

def lowerbounddelta(poset_idx1, poset_idx2):
    if OPTIMIZER == "GUROBI":
        GUROBI_VARIABLES[gendelta(poset_idx1, poset_idx2)].lb = 1.0
        return (None)
    return ("{}: 1;".format(gendelta(poset_idx1, poset_idx2)))

#
# Generate constraints based on variable strings -- Optimizer specific (BARON)

def genEquationName(desc, prefix):
    global next_equation_idx
    try:
        return (equation_desc_to_equation[desc])
    except KeyError:
        name = "{}{}".format(prefix, next_equation_idx)
        next_equation_idx += 1
        equation_desc_to_equation[desc] = name
        equation_to_equation_desc[name] = desc
        if OPTIMIZER == "GUROBI":
            global GUROBI_CONSTRAINT_NAME
            GUROBI_CONSTRAINT_NAME = name
        return (name)

#   if "s_{}_{}".format(i, k) == 0: i.e., the value for
#                                       feature k for state i is 0
#       "s_{}_{}".format(i, k) = 0
#   else:
#       "s_{}_{}".format(i, k) - "M_{}".format(k) = 0
#       -- state i feature k
def constraints1_name(state_idx, feature_idx):
    desc = "_s1_{}_{}".format(state_idx, feature_idx)
    return (genEquationName(desc, "s1_"))

def constraints1(state_idx, feature_idx):
    if OPTIMIZER == "GUROBI":
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[gens(state_idx, feature_idx)] == 0.0, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} == 0;".format(gens(state_idx, feature_idx)))

def constraints2_name(state_idx, feature_idx):
    desc = "_s2_{}_{}".format(state_idx, feature_idx)
    return (genEquationName(desc, "s2_"))

def constraints2(state_idx, feature_idx):
    if OPTIMIZER == "GUROBI":
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[gens(state_idx, feature_idx)] - GUROBI_VARIABLES[genM(feature_idx)] == 0.0, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} - {} == 0;".format(gens(state_idx, feature_idx), genM(feature_idx)))

#   "d_{}_{}_{}".format(i, i', k) - "s_{}_{}".format(i, k) -
#       "s_{}_{}".format(i', k) <= 0
#   "d_{}_{}_{}".format(i, i', k) - "s_{}_{}".format(i, k) +
#       "s_{}_{}".format(i', k) >= 0
#   "d_{}_{}_{}".format(i, i', k) + "s_{}_{}".format(i, k) -
#       "s_{}_{}".format(i', k) >= 0
#   "d_{}_{}_{}".format(i, i', k) + "s_{}_{}".format(i, k) +
#       "s_{}_{}".format(i', k) <= 2
#       -- states i and i', feature k
def constraintd1_name(state_idx1, state_idx2, feature_idx):
    if OPTIMIZER == "GUROBI":
        sys.exit('constraintd1 not used for GUROBI')
    desc = "_d1_{}_{}_{}".format(state_idx1, state_idx2, feature_idx)
    return (genEquationName(desc, "d1_"))

def constraintd1(state_idx1, state_idx2, feature_idx):
    if OPTIMIZER == "GUROBI":
        sys.exit('constraintd1 not used for GUROBI')
    return ("{} - {} - {} <= 0;".format(gend(state_idx1, state_idx2, feature_idx), gens(state_idx1, feature_idx), gens(state_idx2, feature_idx)))

def constraintd2_name(state_idx1, state_idx2, feature_idx):
    if OPTIMIZER == "GUROBI":
        sys.exit('constraintd2 not used for GUROBI')
    desc = "_d2_{}_{}_{}".format(state_idx1, state_idx2, feature_idx)
    return (genEquationName(desc, "d2_"))

def constraintd2(state_idx1, state_idx2, feature_idx):
    if OPTIMIZER == "GUROBI":
        sys.exit('constraintd2 not used for GUROBI')
    return ("{} - {} + {} >= 0;".format(gend(state_idx1, state_idx2, feature_idx), gens(state_idx1, feature_idx), gens(state_idx2, feature_idx)))

def constraintd3_name(state_idx1, state_idx2, feature_idx):
    if OPTIMIZER == "GUROBI":
        sys.exit('constraintd3 not used for GUROBI')
    desc = "_d3_{}_{}_{}".format(state_idx1, state_idx2, feature_idx)
    return (genEquationName(desc, "d3_"))

def constraintd3(state_idx1, state_idx2, feature_idx):
    if OPTIMIZER == "GUROBI":
        sys.exit('constraintd3 not used for GUROBI')
    return ("{} + {} - {} >= 0;".format(gend(state_idx1, state_idx2, feature_idx), gens(state_idx1, feature_idx), gens(state_idx2, feature_idx)))

def constraintdg_name(state_idx1, state_idx2, feature_idx): # GUROBI only
    if OPTIMIZER != "GUROBI":
        sys.exit('constraintdg only used for GUROBI')
    desc = "_dg_{}_{}_{}".format(state_idx1, state_idx2, feature_idx)
    return (genEquationName(desc, "dg_"))

def constraintdg(state_idx1, state_idx2, feature_idx): # GUROBI only
    if OPTIMIZER != "GUROBI":
        sys.exit('constraintdg only used for GUROBI')
    GUROBI_MODEL.addGenConstrOr(GUROBI_VARIABLES[gend(state_idx1, state_idx2, feature_idx)], [ GUROBI_VARIABLES[gens(state_idx1, feature_idx)], GUROBI_VARIABLES[gens(state_idx2, feature_idx)] ], GUROBI_CONSTRAINT_NAME)
    return (None)

def constraintd4_name(state_idx1, state_idx2, feature_idx):
    desc = "_d4_{}_{}_{}".format(state_idx1, state_idx2, feature_idx)
    return (genEquationName(desc, "d4_"))

def constraintd4(state_idx1, state_idx2, feature_idx):
    if OPTIMIZER == "GUROBI":
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[gend(state_idx1, state_idx2, feature_idx)] + GUROBI_VARIABLES[gens(state_idx1, feature_idx)] + GUROBI_VARIABLES[gens(state_idx2, feature_idx)] <= 2.0, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} + {} + {} <= 2;".format(gend(state_idx1, state_idx2, feature_idx), gens(state_idx1, feature_idx), gens(state_idx2, feature_idx)))


#   "D_{}_{}".format(i, i') -
#       \sum_{k=0}^{num_features - 1} "d_{}_{}_{}".format(i, i', k) <= 0
#       -- states i and i'
#   "D_{}_{}".format(i, i') - "d_{}_{}_{}".format(i, i', k) >= 0
#       -- states i and i', feature k
def constraintD1_name(state_idx1, state_idx2):
    desc = "_D1_{}_{}".format(state_idx1, state_idx2)
    return (genEquationName(desc, "D1_"))

def constraintD1(state_idx1, state_idx2, num_features):
    if OPTIMIZER == "GUROBI":
        s = list()
        for f_idx in range(num_features):
            s.append(GUROBI_VARIABLES[gend(state_idx1, state_idx2, f_idx)])
        GUROBI_MODEL.addGenConstrOr(GUROBI_VARIABLES[genD(state_idx1, state_idx2)], s, GUROBI_CONSTRAINT_NAME)
        return (None)
    s = "{}".format(genD(state_idx1, state_idx2))
    for f_idx in range(num_features):
        s += " - {}".format(gend(state_idx1, state_idx2, f_idx))
    s += " <= 0;"
    return (s)

def constraintD2_name(state_idx1, state_idx2, feature_idx):
    if OPTIMIZER == "GUROBI":
        sys.exit("constraintD2 not used for GUROBI")
    desc = "_D2_{}_{}_{}".format(state_idx1, state_idx2, feature_idx)
    return (genEquationName(desc, "D2_"))

def constraintD2(state_idx1, state_idx2, feature_idx):
    if OPTIMIZER == "GUROBI":
        sys.exit("constraintD2 not used for GUROBI")
    return ("{} - {} >= 0;".format(genD(state_idx1, state_idx2), gend(state_idx1, state_idx2, feature_idx)))

#   \sum_{j=0}^{num_classes - 1} "q_{}_{}".format(i, j) = 1
#       -- state i
def constraintq_name(state_idx):
    desc = "_q_{}".format(state_idx)
    return (genEquationName(desc, "q_"))

def constraintq(state_idx, num_classes):
    if OPTIMIZER == 'GUROBI':
        s = gurobipy.quicksum(GUROBI_VARIABLES[genq(state_idx, c_idx)] for c_idx in range(num_classes))
        GUROBI_MODEL.addConstr(s == 1, GUROBI_CONSTRAINT_NAME)
        return (None)
    s = ""
    for c_idx in range(num_classes):
        s += "{}".format(genq(state_idx, c_idx))
        if c_idx < num_classes - 1:
            s += " + "
    s += " == 1;"
    return (s)

#   "Q_{}".format(i) - \sum_{j=0}^{num_classes - 1} 2^j * "q_{}_{}".format(i, j)
#       = 0
#       -- state i
#   2^num_classes * "D_{}_{}".format(i, i') - "Q_{}".format(i)
#       + "Q_{}".format(i') >= 0
#   2^num_classes * "D_{}_{}".format(i, i') + "Q_{}".format(i)
#       - "Q_{}".format(i') >= 0
#   "Q_{}".format(i) - "Q_{}".format(i') - "D_{}_{}".format(i, i')
#       + 2^num_classes * "DB_{}_{}".format(i, i') >= 0
#   "Q_{}".format(i') - "Q_{}".format(i) - "D_{}_{}".format(i, i')
#       - 2^num_classes * "DB_{}_{}".format(i, i') >= -2^num_classes
#       -- states i and i'
def constraintQ1_name(state_idx):
    desc = "_Q1_{}".format(state_idx)
    return (genEquationName(desc, "Q1_"))

def constraintQ1(state_idx, num_classes):
    if OPTIMIZER == 'GUROBI':
        s = gurobipy.quicksum(2**c_idx * GUROBI_VARIABLES[genq(state_idx, c_idx)] for c_idx in range(num_classes))
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genQ(state_idx)] - s == 0, GUROBI_CONSTRAINT_NAME)
        return (None)
    s = "{}".format(genQ(state_idx))
    for c_idx in range(num_classes):
        s += " - {} * {}".format(2**c_idx, genq(state_idx, c_idx))
    s += " == 0;"
    return (s)

def constraintQ2_name(state_idx1, state_idx2):
    desc = "_Q2_{}_{}".format(state_idx1, state_idx2)
    return (genEquationName(desc, "Q2_"))

def constraintQ2(state_idx1, state_idx2, num_classes):
    if OPTIMIZER == 'GUROBI':
        GUROBI_MODEL.addConstr(2**num_classes * GUROBI_VARIABLES[genD(state_idx1, state_idx2)] - GUROBI_VARIABLES[genQ(state_idx1)] + GUROBI_VARIABLES[genQ(state_idx2)] >= 0, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} * {} - {} + {} >= 0;".format(2**num_classes, genD(state_idx1, state_idx2), genQ(state_idx1), genQ(state_idx2)))

def constraintQ3_name(state_idx1, state_idx2):
    desc = "_Q3_{}_{}".format(state_idx1, state_idx2)
    return (genEquationName(desc, "Q3_"))

def constraintQ3(state_idx1, state_idx2, num_classes):
    if OPTIMIZER == 'GUROBI':
        GUROBI_MODEL.addConstr(2**num_classes * GUROBI_VARIABLES[genD(state_idx1, state_idx2)] + GUROBI_VARIABLES[genQ(state_idx1)] - GUROBI_VARIABLES[genQ(state_idx2)] >= 0, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} * {} + {} - {} >= 0;".format(2**num_classes, genD(state_idx1, state_idx2), genQ(state_idx1), genQ(state_idx2)))

def constraintQ4_name(state_idx1, state_idx2):
    desc = "_Q4_{}_{}".format(state_idx1, state_idx2)
    return (genEquationName(desc, "Q4_"))

def constraintQ4(state_idx1, state_idx2, num_classes):
    if OPTIMIZER == 'GUROBI':
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genQ(state_idx1)] - GUROBI_VARIABLES[genQ(state_idx2)] - GUROBI_VARIABLES[genD(state_idx1, state_idx2)] + 2**num_vlasses * GUROBI_VARIABLES[genDB(state_idx1, state_idx2)] >= 0, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} - {} - {} + {} * {} >= 0;".format(genQ(state_idx1), genQ(state_idx2), genD(state_idx1, state_idx2), 2**num_classes, genDB(state_idx1, state_idx2)))

def constraintQ5_name(state_idx1, state_idx2):
    desc = "_Q5_{}_{}".format(state_idx1, state_idx2)
    return (genEquationName(desc, "Q5_"))

def constraintQ5(state_idx1, state_idx2, num_classes):
    if OPTIMIZER == 'GUROBI':
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genQ(state_idx2)] - GUROBI_VARIABLES[genQ(state_idx1)] - GUROBI_VARIABLES[genD(state_idx1, state_idx2)] - 2**num_vlasses * GUROBI_VARIABLES[genDB(state_idx1, state_idx2)] >=  -2**num_classes, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} - {} - {} - {} * {} >= {};".format(genQ(state_idx2), genQ(state_idx1), genD(state_idx1, state_idx2), 2**num_classes, genDB(state_idx1, state_idx2), -2**num_classes))

#   "FQ_{}_{}_{}_{}".format(j, i, j', i') - "q_{}_{}".format(i, j) <= 0
#   "FQ_{}_{}_{}_{}".format(j, i, j', i') - "q_{}_{}".format(i', j') <= 0
#   "FQ_{}_{}_{}_{}".format(j, i, j', i') - "q_{}_{}".format(i, j)
#       - "q_{}_{}".format(i', j') >= -1
#       -- states i and i', classes j and j'
def constraintFQG_name(class_idx1, state_idx1, class_idx2, state_idx2):
    desc = "_FQG_{}_{}_{}_{}".format(class_idx1, state_idx1, class_idx2, state_idx2)
    return (genEquationName(desc, "FQ1_"))

def constraintFQG(class_idx1, state_idx1, class_idx2, state_idx2):
    if OPTIMIZER != 'GUROBI':
        sys.exit('constraintFQG(...) not usable for non-GUROBI optimizer.')
    if state_idx1 > state_idx2:
        GUROBI_MODEL.addGenConstrAnd(GUROBI_VARIABLES[genFQ(class_idx2, state_idx2, class_idx1, state_idx1)], [ GUROBI_VARIABLES[genq(state_idx1, class_idx1)], GUROBI_VARIABLES[genq(state_idx2, class_idx2)] ], GUROBI_CONSTRAINT_NAME)
    else:
        GUROBI_MODEL.addGenConstrAnd(GUROBI_VARIABLES[genFQ(class_idx1, state_idx1, class_idx2, state_idx2)], [ GUROBI_VARIABLES[genq(state_idx1, class_idx1)], GUROBI_VARIABLES[genq(state_idx2, class_idx2)] ], GUROBI_CONSTRAINT_NAME)
    return (None)

def constraintFQ1_name(class_idx1, state_idx1, class_idx2, state_idx2):
    desc = "_FQ1_{}_{}_{}_{}".format(class_idx1, state_idx1, class_idx2, state_idx2)
    return (genEquationName(desc, "FQ1_"))

def constraintFQ1(class_idx1, state_idx1, class_idx2, state_idx2):
    if OPTIMIZER == 'GUROBI':
        if state_idx1 > state_idx2:
            GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genFQ(class_idx2, state_idx2, class_idx1, state_idx1)] - GUROBI_VARIABLES[genq(state_idx1, class_idx1)] <= 0, GUROBI_CONSTRAINT_NAME)
        else:
            GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genFQ(class_idx1, state_idx1, class_idx2, state_idx2)] - GUROBI_VARIABLES[genq(state_idx1, class_idx1)] <= 0, GUROBI_CONSTRAINT_NAME)
        return (None)
    if state_idx1 > state_idx2:
        return("{} - {} <= 0;".format(genFQ(class_idx2, state_idx2, class_idx1, state_idx1), genq(state_idx1, class_idx1)))
    else:
        return("{} - {} <= 0;".format(genFQ(class_idx1, state_idx1, class_idx2, state_idx2), genq(state_idx1, class_idx1)))

def constraintFQ2_name(class_idx1, state_idx1, class_idx2, state_idx2):
    desc = "_FQ2_{}_{}_{}_{}".format(class_idx1, state_idx1, class_idx2, state_idx2)
    return (genEquationName(desc, "FQ2_"))

def constraintFQ2(class_idx1, state_idx1, class_idx2, state_idx2):
    if OPTIMIZER == 'GUROBI':
        if state_idx1 > state_idx2:
            GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genFQ(class_idx2, state_idx2, class_idx1, state_idx1)] - GUROBI_VARIABLES[genq(state_idx2, class_idx2)] <= 0, GUROBI_CONSTRAINT_NAME)
        else:
            GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genFQ(class_idx1, state_idx1, class_idx2, state_idx2)] - GUROBI_VARIABLES[genq(state_idx2, class_idx2)] <= 0, GUROBI_CONSTRAINT_NAME)
        return (None)
    if state_idx1 > state_idx2:
        return("{} - {} <= 0;".format(genFQ(class_idx2, state_idx2, class_idx1, state_idx1), genq(state_idx2, class_idx2)))
    else:
        return("{} - {} <= 0;".format(genFQ(class_idx1, state_idx1, class_idx2, state_idx2), genq(state_idx2, class_idx2)))

def constraintFQ3_name(class_idx1, state_idx1, class_idx2, state_idx2):
    desc = "_FQ3_{}_{}_{}_{}".format(class_idx1, state_idx1, class_idx2, state_idx2)
    return (genEquationName(desc, "FQ3_"))

def constraintFQ3(class_idx1, state_idx1, class_idx2, state_idx2):
    if OPTIMIZER == 'GUROBI':
        if state_idx1 > state_idx2:
            GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genFQ(class_idx2, state_idx2, class_idx1, state_idx1)] - GUROBI_VARIABLES[genq(state_idx2, class_idx2)] - GUROBI_VARIABLES[genq(state_idx1, class_idx1)] >= -1, GUROBI_CONSTRAINT_NAME)
        else:
            GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genFQ(class_idx1, state_idx1, class_idx2, state_idx2)] - GUROBI_VARIABLES[genq(state_idx2, class_idx2)] - GUROBI_VARIABLES[genq(state_idx1, class_idx1)] >= -1, GUROBI_CONSTRAINT_NAME)
        return (None)
    if state_idx1 > state_idx2:
        return("{} - {} - {} >= -1;".format(genFQ(class_idx2, state_idx2, class_idx1, state_idx1), genq(state_idx1, class_idx1), genq(state_idx2, class_idx2)))
    else:
        return("{} - {} - {} >= -1;".format(genFQ(class_idx1, state_idx1, class_idx2, state_idx2), genq(state_idx1, class_idx1), genq(state_idx2, class_idx2)))

# Cq_targ1:
#   "Cq_targ_{}".format(j) - \sum_{i=0}^{num_states - 1} C_{DM}(i)
#       * "q_{}_{}".format(i, j) = 0
#       -- class j
# Cq_targ2:
#   "Cq_targ_{}_{}".format(j, a) - \sum_{i=0}^{num_states - 1} C_{DM}(i, a)
#       * "q_{}_{}".format(i, j) = 0
#       -- class j and action a
def constraintCq_targ1_name(class_idx):
    desc = "_Cq_targ_{}".format(class_idx)
    return (genEquationName(desc, "Cq_targ1_"))

def constraintCq_targ1(class_idx, num_states, C1_targ):
    if OPTIMIZER == 'GUROBI':
        l = list()
        for s_idx in range(num_states):
            try:
                l.append( ( C1_targ[s_idx], GUROBI_VARIABLES[genq(s_idx, class_idx)] ) )
            except KeyError:
                pass
        s = gurobipy.quicksum( w * v for w, v in l )
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genCq_targ1(class_idx)] - s == 0.0, GUROBI_CONSTRAINT_NAME)
        return (None)
    s = genCq_targ1(class_idx)
    for s_idx in range(num_states):
        try:
            s += " - {} * {}".format(C1_targ[s_idx], genq(s_idx, class_idx))
        except KeyError:
            pass
    s += " == 0;"
    return (s)

def constraintCq_targ2_name(class_idx, action_idx):
    desc = "_Cq_targ_{}_{}".format(class_idx, action_idx)
    return (genEquationName(desc, "Cq_targ2_"))

def constraintCq_targ2(class_idx, action_idx, num_states, C2_targ):
    if OPTIMIZER == 'GUROBI':
        l = list()
        for s_idx in range(num_states):
            try:
                l.append( ( C2_targ[(s_idx, action_idx)], GUROBI_VARIABLES[genq(s_idx, class_idx)] ) )
            except KeyError:
                pass
        s = gurobipy.quicksum( w * v for w, v in l )
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genCq_targ2(class_idx, action_idx)] - s == 0.0, GUROBI_CONSTRAINT_NAME)
        return (None)
    s = genCq_targ2(class_idx, action_idx)
    for s_idx in range(num_states):
        try:
            s += " - {} * {}".format(C2_targ[(s_idx, action_idx)], genq(s_idx, class_idx))
        except KeyError:
            pass
    s += " == 0;"
    return (s)

# Cq_nontarg1:
#   "Cq_nontarg_{}".format(j) - \sum_{i=0}^{num_states - 1} C_{NDM}(i)
#       * "q_{}_{}".format(i, j) = 0
#       -- class j
# Cq_nontarg2:
#   "Cq_nontarg_{}_{}".format(j, a) - \sum_{i=0}^{num_states - 1} C_{NDM}(i, a)
#       * "q_{}_{}".format(i, j) = 0
def constraintCq_nontarg1_name(class_idx):
    desc = "_Cq_nontarg_{}".format(class_idx)
    return (genEquationName(desc, "Cq_nontarg1_"))

def constraintCq_nontarg1(class_idx, num_states, C1_nontarg):
    """
    if OPTIMIZER == 'GUROBI':
        c = list()
        l = list()
        s = gurobipy.LinExpr( -1.0 * GUROBI_VARIABLES[genCq_nontarg1(class_idx)] )
        for s_idx in range(num_states):
            try:
                c.append(C1_nontarg[s_idx])
                l.append(GUROBI_VARIABLES[genq(s_idx, class_idx)])
            except KeyError:
                pass
        s.addTerms(c, l)
        GUROBI_MODEL.addLConstr(lhs = s, sense = GRB.EQUAL, rhs = 0.0, name = GUROBI_CONSTRAINT_NAME)
        return (None)
    """
    s = genCq_nontarg1(class_idx)
    for s_idx in range(num_states):
        try:
            s += " - {} * {}".format(C1_nontarg[s_idx], genq(s_idx, class_idx))
        except KeyError:
            pass
    s += " == 0;"
    return (s)

def constraintCq_nontarg2_name(class_idx, action_idx):
    desc = "_Cq_nontarg_{}_{}".format(class_idx, action_idx)
    return (genEquationName(desc, "Cq_nontarg2_"))

def constraintCq_nontarg2(class_idx, action_idx, num_states, C2_nontarg):
    if OPTIMIZER == 'GUROBI':
        l = list()
        for s_idx in range(num_states):
            try:
                l.append( ( C2_nontarg[(s_idx, action_idx)], GUROBI_VARIABLES[genq(s_idx, class_idx)] ) )
            except KeyError:
                pass
        s = gurobipy.quicksum( w * v for w, v in l )
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genCq_nontarg2(class_idx, action_idx)] - s == 0.0, GUROBI_CONSTRAINT_NAME)
        return (None)
    s = genCq_nontarg2(class_idx, action_idx)
    for s_idx in range(num_states):
        try:
            s += " - {} * {}".format(C2_nontarg[(s_idx, action_idx)], genq(s_idx, class_idx))
        except KeyError:
            pass
    s += " == 0;"
    return (s)

# Cq1:
#   "Cq_{}_{}".format(j, a) - \sum{i=0}^{num_states - 1} C(i, a)
#       * "q_{}_{}".format(i, j) = 0
#       -- class j and action a
# Cq2:
#   "Cq_{}_{}_{}".format(j, a, j') - \sum{i=0}^{num_states - 1}
#       \sum{i'=0}^{num_states - 1} C(i, a, i')
#       * "FQ_{}_{}_{}_{}".format(j, i, j', i') = 0
#       -- classes j and j' and action a
#       -- Must flip i and i' if i>i' in FQ term
def constraintCq1_name(class_idx, action_idx):
    desc = "_Cq_{}_{}".format(class_idx, action_idx)
    return (genEquationName(desc, "Cq1_"))

def constraintCq1(class_idx, action_idx, num_states, C2):
    if OPTIMIZER == 'GUROBI':
        l = list()
        for s_idx in range(num_states):
            try:
                l.append( ( C2[(s_idx, action_idx)], GUROBI_VARIABLES[genq(s_idx, class_idx)] ) )
            except KeyError:
                pass
        s = gurobipy.quicksum( w * v for w, v in l )
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genCq1(class_idx, action_idx)] - s == 0.0, GUROBI_CONSTRAINT_NAME)
        return (None)
    s = genCq1(class_idx, action_idx)
    for s_idx in range(num_states):
        try:
            s += " - {} * {}".format(C2[(s_idx, action_idx)], genq(s_idx, class_idx))
        except KeyError:
            pass
    s += " == 0;"
    return (s)

def constraintCq2_name(class_idx1, action_idx, class_idx2):
    desc = "_Cq_{}_{}_{}".format(class_idx1, action_idx, class_idx2)
    return (genEquationName(desc, "Cq2_"))

def constraintCq2(class_idx1, action_idx, class_idx2, num_states, C3, trajectories, triples):
    if OPTIMIZER == 'GUROBI':
        pairs = set()
        l = list()
        for ( s_idx1, a_idx, s_idx2 ) in triples:
            if s_idx1 == s_idx2 and class_idx1 != class_idx2:
                continue
            if a_idx != action_idx:
                continue
            if ( s_idx1, s_idx2 ) in pairs:
                continue
            if s_idx1 > s_idx2:
                l.append( ( C3[(s_idx1, action_idx, s_idx2)], GUROBI_VARIABLES[genFQ(class_idx2, s_idx2, class_idx1, s_idx1)] ) )
            else:
                l.append( ( C3[(s_idx1, action_idx, s_idx2)], GUROBI_VARIABLES[genFQ(class_idx1, s_idx1, class_idx2, s_idx2)] ) )
        s = gurobipy.quicksum( w * v for w, v in l )
        if len(l) > 0:
            GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genCq2(class_idx1, action_idx, class_idx2)] - s == 0.0, GUROBI_CONSTRAINT_NAME)
        else:
            GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genCq2(class_idx1, action_idx, class_idx2)] == 0.0, GUROBI_CONSTRAINT_NAME)
        return (None)
    s = genCq2(class_idx1, action_idx, class_idx2)
    pairs = set()
    for (s_idx1, a_idx, s_idx2) in triples:
        if s_idx1 == s_idx2 and class_idx1 != class_idx2:
            continue
        if a_idx != action_idx:
            continue
        if ( s_idx1, s_idx2 ) in pairs:
            continue
        if s_idx1 > s_idx2:
            s += " - {} * {}".format(C3[(s_idx1, action_idx, s_idx2)], genFQ(class_idx2, s_idx2, class_idx1, s_idx1))
        else:
            s += " - {} * {}".format(C3[(s_idx1, action_idx, s_idx2)], genFQ(class_idx1, s_idx1, class_idx2, s_idx2))
        pairs.add(( s_idx1, s_idx2 ))
    s += " == 0;"
    del pairs
    return (s)

# Pq_targ:
#   "Cq_targ_{}".format(j) * "Pq_targ_{}_{}".format(a, j)
#       - "Cq_targ_{}_{}".format(j, a) = 0
#       -- class j and action a
def constraintPq_targ_name(action_idx, class_idx):
    desc = "_Pq_targ_{}_{}".format(action_idx, class_idx)
    return (genEquationName(desc, "Pq_targ_"))

def constraintPq_targ(action_idx, class_idx):
    if OPTIMIZER == 'GUROBI':
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genCq_targ1(class_idx)] * GUROBI_VARIABLES[genPq_targ(action_idx, class_idx)] - GUROBI_VARIABLES[genCq_targ2(class_idx, action_idx)] == 0.0, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} * {} - {} == 0;".format(genCq_targ1(class_idx), genPq_targ(action_idx, class_idx), genCq_targ2(class_idx, action_idx)))

# Pq_nontarg:
#   "Cq_nontarg_{}".format(j) * "Pq_nontarg_{}_{}".format(a, j)
#       - "Cq_nontarg_{}_{}".format(j, a) = 0
#       -- class j and action a
def constraintPq_nontarg_name(action_idx, class_idx):
    desc = "_Pq_nontarg_{}_{}".format(action_idx, class_idx)
    return (genEquationName(desc, "Pq_nontarg_"))

def constraintPq_nontarg(action_idx, class_idx):
    if OPTIMIZER == 'GUROBI':
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genCq_nontarg1(class_idx)] * GUROBI_VARIABLES[genPq_nontarg(action_idx, class_idx)] - GUROBI_VARIABLES[genCq_nontarg2(class_idx, action_idx)] == 0.0, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} * {} - {} == 0;".format(genCq_nontarg1(class_idx), genPq_nontarg(action_idx, class_idx), genCq_nontarg2(class_idx, action_idx)))

# Pq:
#   "Cq_{}_{}".format(j, a) * "Pq_{}_{}_{}".format(j', a, j)
#       - "Cq_{}_{}_{}".format(j, a, j') = 0
#       -- classes j and j' and action a
def constraintPq_name(class_idx2, action_idx, class_idx1):
    desc = "_Pq_{}_{}_{}".format(class_idx2, action_idx, class_idx1)
    return (genEquationName(desc, "Pq_"))

def constraintPq(class_idx2, action_idx, class_idx1):
    if OPTIMIZER == 'GUROBI':
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genCq1(class_idx1, action_idx)] * GUROBI_VARIABLES[genPq(class_idx2, action_idx, class_idx1)] - GUROBI_VARIABLES[genCq2(class_idx1, action_idx, class_idx2)] == 0.0, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} * {} - {} == 0;".format(genCq1(class_idx1, action_idx), genPq(class_idx2, action_idx, class_idx1), genCq2(class_idx1, action_idx, class_idx2)))

# P_targ1:
#   "Pq_targ_{}_{}".format(a, j) - "P_targ_{}_{}".format(a, i)
#       - "q_{}_{}".format(i, j) >= -1
# P_targ2:
#   "P_targ_{}_{}".format(a, i) - "Pq_targ_{}_{}".format(a, j)
#       - "q_{}_{}".format(i, j) >= -1
#       -- actin a, class j, state i
def constraintP_targ1_name(action_idx, state_idx, class_idx):
    desc = "_P_targ1_{}_{}_<{}>".format(action_idx, state_idx, class_idx)
    return (genEquationName(desc, "P_targ1_"))

def constraintP_targ1(action_idx, state_idx, class_idx):
    if OPTIMIZER == 'GUROBI':
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genPq_targ(action_idx, class_idx)] - GUROBI_VARIABLES[genP_targ(action_idx, state_idx)] - GUROBI_VARIABLES[genq(state_idx, class_idx)] >= -1.0, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} - {} - {} >= -1;".format(genPq_targ(action_idx, class_idx), genP_targ(action_idx, state_idx), genq(state_idx, class_idx)))

def constraintP_targ2_name(action_idx, state_idx, class_idx):
    desc = "_P_targ2_{}_{}_<{}>".format(action_idx, state_idx, class_idx)
    return (genEquationName(desc, "P_targ2_"))

def constraintP_targ2(action_idx, state_idx, class_idx):
    if OPTIMIZER == 'GUROBI':
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genP_targ(action_idx, state_idx)] - GUROBI_VARIABLES[genPq_targ(action_idx, class_idx)] - GUROBI_VARIABLES[genq(state_idx, class_idx)] >= -1.0, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} - {} - {} >= -1;".format(genP_targ(action_idx, state_idx), genPq_targ(action_idx, class_idx), genq(state_idx, class_idx)))

# P_nontarg1:
#   "Pq_nontarg_{}_{}".format(a, j) - "P_nontarg_{}_{}".format(a, i)
#       - "q_{}_{}".format(i, j) >= -1
# P_nontarg2:
#   "P_nontarg_{}_{}".format(a, i) - "Pq_nontarg_{}_{}".format(a, j)
#       - "q_{}_{}".format(i, j) >= -1
#       -- actin a, class j, state i
def constraintP_nontarg1_name(action_idx, state_idx, class_idx):
    desc = "_P_nontarg1_{}_{}_<{}>".format(action_idx, state_idx, class_idx)
    return (genEquationName(desc, "P_nontarg1_"))

def constraintP_nontarg1(action_idx, state_idx, class_idx):
    if OPTIMIZER == 'GUROBI':
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genPq_nontarg(action_idx, class_idx)] - GUROBI_VARIABLES[genP_nontarg(action_idx, state_idx)] - GUROBI_VARIABLES[genq(state_idx, class_idx)] >= -1.0, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} - {} - {} >= -1;".format(genPq_nontarg(action_idx, class_idx), genP_nontarg(action_idx, state_idx), genq(state_idx, class_idx)))

def constraintP_nontarg2_name(action_idx, state_idx, class_idx):
    desc = "_P_nontarg2_{}_{}_<{}>".format(action_idx, state_idx, class_idx)
    return (genEquationName(desc, "P_nontarg2_"))

def constraintP_nontarg2(action_idx, state_idx, class_idx):
    if OPTIMIZER == 'GUROBI':
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genP_nontarg(action_idx, state_idx)] - GUROBI_VARIABLES[genPq_nontarg(action_idx, class_idx)] - GUROBI_VARIABLES[genq(state_idx, class_idx)] >= -1.0, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} - {} - {} >= -1;".format(genP_nontarg(action_idx, state_idx), genPq_nontarg(action_idx, class_idx), genq(state_idx, class_idx)))

# P1:
#   "Pq_{}_{}_{}".format(j', a, j) - "P_{}_{}_{}".format(i', a, i)
#       - "q_{}_{}".format(i, j) - "q_{}_{}".format(i', j') >= -2
# P2:
#   "P_{}_{}_{}".format(i', a, i) - "Pq_{}_{}_{}".format(j', a, j)
#       - "q_{}_{}".format(i, j) - "q_{}_{}".format(i', j') >= -2
#       -- classes j, j', action a, states i, i'
def constraintP1_name(state_idx2, action_idx, state_idx1, class_idx2, class_idx1):
    desc = "_P1_{}_{}_{}_<{},{}>".format(state_idx2, action_idx, state_idx1, class_idx2, class_idx1)
    return (genEquationName(desc, "P1_"))

def constraintP1(state_idx2, action_idx, state_idx1, class_idx2, class_idx1):
    if OPTIMIZER == 'GUROBI':
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genPq(class_idx2, action_idx, class_idx1)] - GUROBI_VARIABLES[genP(state_idx2, action_idx, state_idx1)] - GUROBI_VARIABLES[genq(state_idx2, class_idx2)] - GUROBI_VARIABLES[genq(state_idx1, class_idx1)] >= -2.0, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} - {} - {} - {} >= -2;".format(genPq(class_idx2, action_idx, class_idx1), genP(state_idx2, action_idx, state_idx1), genq(state_idx2, class_idx2), genq(state_idx1, class_idx1)))

def constraintP2_name(state_idx2, action_idx, state_idx1, class_idx2, class_idx1):
    desc = "_P2_{}_{}_{}_<{},{}>".format(state_idx2, action_idx, state_idx1, class_idx2, class_idx1)
    return (genEquationName(desc, "P2_"))

def constraintP2(state_idx2, action_idx, state_idx1, class_idx2, class_idx1):
    if OPTIMIZER == 'GUROBI':
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genP(state_idx2, action_idx, state_idx1)] - GUROBI_VARIABLES[genPq(class_idx2, action_idx, class_idx1)] - GUROBI_VARIABLES[genq(state_idx2, class_idx2)] - GUROBI_VARIABLES[genq(state_idx1, class_idx1)] >= -2.0, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} - {} - {} - {} >= -2;".format(genP(state_idx2, action_idx, state_idx1), genPq(class_idx2, action_idx, class_idx1), genq(state_idx2, class_idx2), genq(state_idx1, class_idx1)))

# r1:
#   "rq_{}_{}_{}".format(j, a, j') - "r_{}_{}_{}".format(i, a, i')
#       - 2 * peak * "q_{}_{}".format(i, j) - 2 * peak * "q_{}_{}".format(i', j') >= -4 * peak
# r2:
#   "r_{}_{}_{}".format(i, a, i') - "rq_{}_{}_{}".format(j, a, j')
#       - 2 * peak * "q_{}_{}".format(i, j) - 2 * peak * "q_{}_{}".format(i', j') >= -4 * peak
def constraintr1_name(state_idx1, action_idx, state_idx2, class_idx1, class_idx2):
    desc = "r1_{}_{}_{}_<{},{}>".format(state_idx1, action_idx, state_idx2, class_idx1, class_idx2)
    return (genEquationName(desc, "r1_"))

def constraintr1(state_idx1, action_idx, state_idx2, class_idx1, class_idx2, peak):
    if OPTIMIZER == 'GUROBI':
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genrq(class_idx2, action_idx, class_idx1)] - GUROBI_VARIABLES[genr(state_idx1, action_idx, state_idx2)] - 2.0 * peak * GUROBI_VARIABLES[genq(state_idx1, class_idx1)] - 2.0 * peak * GUROBI_VARIABLES[genq(state_idx2, class_idx2)] >= -4.0 * peak, GUROBI_CONSTRAINT_NAME)
        return (None)
    return("{} - {} - {} * {} - {} * {} >= {};".format(genrq(class_idx1, action_idx, class_idx2), genr(state_idx1, action_idx, state_idx2), 2.0 * peak, genq(state_idx1, class_idx1), 2.0 * peak, genq(state_idx2, class_idx2), -4.0 * peak))

def constraintr2_name(state_idx1, action_idx, state_idx2, class_idx1, class_idx2):
    desc = "r2_{}_{}_{}_<{},{}>".format(state_idx1, action_idx, state_idx2, class_idx1, class_idx2)
    return (genEquationName(desc, "r2_"))

def constraintr2(state_idx1, action_idx, state_idx2, class_idx1, class_idx2, peak):
    if OPTIMIZER == 'GUROBI':
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genr(state_idx1, action_idx, state_idx2)] - GUROBI_VARIABLES[genrq(class_idx2, action_idx, class_idx1)] - 2.0 * peak * GUROBI_VARIABLES[genq(state_idx1, class_idx1)] - 2.0 * peak * GUROBI_VARIABLES[genq(state_idx2, class_idx2)] >= -4.0 * peak, GUROBI_CONSTRAINT_NAME)
        return (None)
    return("{} - {} - {} * {} - {} * {} >= {};".format(genr(state_idx1, action_idx, state_idx2), genrq(class_idx1, action_idx, class_idx2), 2.0 * peak, genq(state_idx1, class_idx1), 2.0 * peak, genq(state_idx2, class_idx2), -4.0 * peak))

#   if {s_i, a, s_i'} occurs in a dm trajectory:
#       "r_{}_{}_{}".format(i, a, i') - peak * "P_targ_{}_{}".format(a, i')
#           - "FRV_targ_{}_{}_{}".format(i, a, i') <= 0
#   if {s_i, a, s_i'} does not occur in a dm trajectory:
#       "r_{}_{}_{}".format(i, a, i') + peak * "P_nontarg_{}_{}".format(a, i')
#           - "FRV_nontarg_{}_{}_{}".format(i, a, i') <= 0
def constraintFRV_targ_name(state_idx1, action_idx, state_idx2):
    desc = "_FRV_targ_{}_{}_{}".format(state_idx1, action_idx, state_idx2)
    return (genEquationName(desc, "FT"))

def constraintFRV_targ(state_idx1, action_idx, state_idx2, peak):
    if OPTIMIZER == 'GUROBI':
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genr(state_idx1, action_idx, state_idx2)] - peak * GUROBI_VARIABLES[genP_targ(action_idx, state_idx1)] - GUROBI_VARIABLES[genFRV_targ(state_idx1, action_idx, state_idx2)] <= 0.0, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} - {} * {} - {} <= 0;".format(genr(state_idx1, action_idx, state_idx2), peak, genP_targ(action_idx, state_idx1), genFRV_targ(state_idx1, action_idx, state_idx2)))

def constraintFRV_nontarg_name(state_idx1, action_idx, state_idx2):
    desc = "_FRV_nontarg_{}_{}_{}".format(state_idx1, action_idx, state_idx2)
    return (genEquationName(desc, "FN"))

def constraintFRV_nontarg(state_idx1, action_idx, state_idx2, peak):
    if OPTIMIZER == 'GUROBI':
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genr(state_idx1, action_idx, state_idx2)] - peak * GUROBI_VARIABLES[genP_nontarg(action_idx, state_idx1)] - GUROBI_VARIABLES[genFRV_nontarg(state_idx1, action_idx, state_idx2)] <= 0.0, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} - {} * {} - {} <= 0;".format(genr(state_idx1, action_idx, state_idx2), peak, genP_nontarg(action_idx, state_idx1), genFRV_nontarg(state_idx1, action_idx, state_idx2)))

#   "FRV_targ_{}_{}_{}".format(i, a, i') <= FRV_maxMagnitude
#   -"FRV_targ_{}_{}_{}".format(i, a, i') <= FRV_maxMagnitude
#   "FRV_nontarg_{}_{}_{}".format(i, a, i') <= FRV_maxMagnitude
#   -"FRV_nontarg_{}_{}_{}".format(i, a, i') <= FRV_maxMagnitude
def constraintFRV_maxMagnitude_targ1_name(state_idx1, action_idx, state_idx2):
    desc = "_FRV_maxMagnitude_targ1_{}_{}_{}".format(state_idx1, action_idx, state_idx2)
    return (genEquationName(desc, "FMT1"))

def constraintFRV_maxMagnitude_targ1(state_idx1, action_idx, state_idx2):
    if OPTIMIZER == "GUROBI":
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genFRV_targ(state_idx1, action_idx, state_idx2)] - GUROBI_VARIABLES[genFRV_maxMagnitude()] <= 0.0, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} - {} <= 0".format(genFRV_targ(state_idx1, action_idx, state_idx2), genFRV_maxMagnitude()))

def constraintFRV_maxMagnitude_targ2_name(state_idx1, action_idx, state_idx2):
    desc = "_FRV_maxMagnitude_targ2_{}_{}_{}".format(state_idx1, action_idx, state_idx2)
    return (genEquationName(desc, "FMT2"))

def constraintFRV_maxMagnitude_targ2(state_idx1, action_idx, state_idx2):
    if OPTIMIZER == "GUROBI":
        GUROBI_MODEL.addConstr(- GUROBI_VARIABLES[genFRV_targ(state_idx1, action_idx, state_idx2)] - GUROBI_VARIABLES[genFRV_maxMagnitude()] <= 0.0, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("- {} - {} <= 0".format(genFRV_targ(state_idx1, action_idx, state_idx2), genFRV_maxMagnitude()))

def constraintFRV_maxMagnitude_nontarg1_name(state_idx1, action_idx, state_idx2):
    desc = "_FRV_maxMagnitude_nontarg1_{}_{}_{}".format(state_idx1, action_idx, state_idx2)
    return (genEquationName(desc, "FMNT1"))

def constraintFRV_maxMagnitude_nontarg1(state_idx1, action_idx, state_idx2):
    if OPTIMIZER == "GUROBI":
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genFRV_nontarg(state_idx1, action_idx, state_idx2)] - GUROBI_VARIABLES[genFRV_maxMagnitude()] <= 0.0, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} - {} <= 0".format(genFRV_nontarg(state_idx1, action_idx, state_idx2), genFRV_maxMagnitude()))

def constraintFRV_maxMagnitude_nontarg2_name(state_idx1, action_idx, state_idx2):
    desc = "_FRV_maxMagnitude_nontarg2_{}_{}_{}".format(state_idx1, action_idx, state_idx2)
    return (genEquationName(desc, "FMNT2"))

def constraintFRV_maxMagnitude_nontarg2(state_idx1, action_idx, state_idx2):
    if OPTIMIZER == "GUROBI":
        GUROBI_MODEL.addConstr(- GUROBI_VARIABLES[genFRV_nontarg(state_idx1, action_idx, state_idx2)] - GUROBI_VARIABLES[genFRV_maxMagnitude()] <= 0.0, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("- {} - {} <= 0".format(genFRV_nontarg(state_idx1, action_idx, state_idx2), genFRV_maxMagnitude()))

#   "LER_{}".format(t) = \sum_{b=0}^{length(t)-1}
#       "P_{}_{}_{}".format(s_{i}_<{t},{b}>, a_{a}_<{t},{b+1}>,
#       s_{i}_<{t},{b+1}>) * "r_{}_{}_{}".format(s_{i}_<{t},{b}>,
#        a_{a}_<{t},{b+1}>, s_{i}_<{t},{b+1}>)
#       -- computes LER for trajectory a
def constraintLER_name(traj_idx):
    desc = "_LER_{}".format(traj_idx)
    return (genEquationName(desc, "LER_"))

def constraintLER(traj_idx, trajectory):
    if OPTIMIZER == 'GUROBI':
        s = gurobipy.quicksum( GUROBI_VARIABLES[genP(trajectory[(state_pos + 1) * 2], trajectory[state_pos * 2 + 1], trajectory[state_pos * 2])] * GUROBI_VARIABLES[genr(trajectory[state_pos * 2], trajectory[state_pos * 2 + 1], trajectory[(state_pos + 1) * 2])] for state_pos in range(floor(len(trajectory) / 2)))
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genLER(traj_idx)] - s == 0.0, GUROBI_CONSTRAINT_NAME)
        return (None)
    constraint = genLER(traj_idx)
    for state_pos in range(floor(len(trajectory) / 2)):
        constraint += " - {} * {}".format(genP(trajectory[(state_pos + 1) * 2], trajectory[state_pos * 2 + 1], trajectory[state_pos * 2]), genr(trajectory[state_pos * 2], trajectory[state_pos * 2 + 1], trajectory[(state_pos + 1) * 2]))
    constraint += " == 0;"
    return (constraint)

#   "LB_{}".format(u) = "UB_{}".format(u+1) + "delta_{}_{}".format(u, u+1)
#       -- orders posets
def constraintPoset_name(poset_idx1, poset_idx2):
    desc = "_POSETS_{}_{}".format(poset_idx1, poset_idx2)
    return (genEquationName(desc, "POSETS_"))

def constraintPoset(poset_idx1, poset_idx2):
    if OPTIMIZER == 'GUROBI':
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genLB(poset_idx1)] - GUROBI_VARIABLES[genUB(poset_idx2)] - GUROBI_VARIABLES[gendelta(poset_idx1, poset_idx2)] == 0.0, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} - {} - {} == 0;".format(genLB(poset_idx1), genUB(poset_idx2), gendelta(poset_idx1, poset_idx2)))

#   "LB_{}".format(u) <= UB_{}".format(u)
def constraintPosetLBUB_name(poset_idx):
    desc = "_POSETLBUB_{}".format(poset_idx)
    return (genEquationName(desc, "POSET"))

def constraintPosetLBUB(poset_idx):
    if OPTIMIZER == 'GUROBI':
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genLB(poset_idx)] - GUROBI_VARIABLES[genUB(poset_idx)] <= 0.0, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} - {} <= 0;".format(genLB(poset_idx), genUB(poset_idx)))

#   "LB_{}".format(u) <= "LER_{}".format(t)
#   "UB_{}".format(u) >= "LER_{}".format(t)
#   "LB_{}".format(u) <= "LER_{}".format(t)
#   "UB_{}".format(u) >= "LER_{}".format(t)
#       -- only if trajectory t is in posets[u]
def constraintPosetLB_name(poset_idx, traj_idx):
    desc = "_POSETLB_{}_{}".format(poset_idx, traj_idx)
    return (genEquationName(desc, "POSETLB_"))

def constraintPosetLB(poset_idx, traj_idx):
    if OPTIMIZER == 'GUROBI':
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genLB(poset_idx)] - GUROBI_VARIABLES[genLER(traj_idx)] <= 0.0, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} - {} <= 0;".format(genLB(poset_idx), genLER(traj_idx)))

def constraintPosetUB_name(poset_idx, traj_idx):
    desc = "_POSETUB_{}_{}".format(poset_idx, traj_idx)
    return (genEquationName(desc, "POSETUB_"))

def constraintPosetUB(poset_idx, traj_idx):
    if OPTIMIZER == 'GUROBI':
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genUB(poset_idx)] - GUROBI_VARIABLES[genLER(traj_idx)] >= 0.0, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} - {} >= 0;".format(genUB(poset_idx), genLER(traj_idx)))

#
# Declare all variables -- specific to BARON
def addVariable(var, var_type, binary_variables, integer_variables, positive_variables, variables):
    if var_type == "BINARY_VARIABLE":
        if var in binary_variables:
#            if DEBUG:
#                print("{} already in binary variables!".format(var))
            return (False)
        else:
            binary_variables.add(var)
            return (True)
    if var_type == "INTEGER_VARIABLE":
        if var in integer_variables:
#            if DEBUG:
#                print("{} already in integer variables!".format(var))
            return (False)
        else:
            integer_variables.add(var)
            return (True)
    if var_type == "POSITIVE_VARIABLE":
        if var in positive_variables:
#            if DEBUG:
#                print("{} already in positive variables!".format(var))
            return (False)
        else:
            positive_variables.add(var)
            return (True)
    if var_type == "VARIABLE":
        if var in variables:
#            if DEBUG:
#                print("{} already in variables!".format(var))
            return (False)
        else:
            variables.add(var)
            return (True)
    sys.exit("{} type unknown for variable <{}>".format(var_type, var))

def createDeclarations(var_type, variables):
    global OPTIMIZER
    if OPTIMIZER == "BARON":
        vars = list(variables)
        vars.sort()
        row = "{} {}".format(var_type, vars[0])
        for idx in range(1, len(vars)):
            row += ",\n\t{}".format(vars[idx])
        row += ";"
        del vars
        return (row)
    else:
        global GUROBI_MODEL
        global GUROBI_VARIABLES
        global GUROBI_INTEGRALS
        for var in variables:
            if var_type == "BINARY_VARIABLES":
                v = GUROBI_MODEL.addVar(vtype = GRB.BINARY, name = var)
                GUROBI_MODEL.update()
                GUROBI_INTEGRALS.add(v)
            elif var_type == "INTEGER_VARIABLES":
                v = GUROBI_MODEL.addVar(lb = -GRB.INFINITY, vtype = GRB.INTEGER, name = var)
                GUROBI_MODEL.update()
                GUROBI_INTEGRALS.add(v)
            elif var_type == "POSITIVE_VARIABLES":
                v = GUROBI_MODEL.addVar(lb = 0.0, name = var)
                GUROBI_MODEL.update()
            elif var_type == "VARIABLES":
                v = GUROBI_MODEL.addVar(lb = -GRB.INFINITY, name = var)
                GUROBI_MODEL.update()
            else:
                sys.exit("{} type unknown for variable <{}>".format(var_type, var))
            GUROBI_VARIABLES[var] = v
        return (None)

def declareVariables(num_states, num_actions, num_features, num_classes, num_posets, trajectories, triples, dm_triples, others_triples, simplify_masking):
    # Returns a tuple with the first element a list of output strings each a
    # row and second element is a dictionary mapping each LER variable to the
    # unique associated trajectory.
    rows = list()
    binary_variables = set()
    integer_variables = set()
    positive_variables = set()
    variables = set()
    LER_to_trajectory = dict()

    if not simplify_masking:
    #   "M_{}".format(k)
        if DEBUG:
            print ('...M variables...')
            var_ct = len(var_to_var_desc)
        for f_idx in range(num_features):
            addVariable(genM(f_idx), typeM(), binary_variables, integer_variables, positive_variables, variables)
        if DEBUG:
            print ('......{} declared.'.format(len(var_to_var_desc) - var_ct))

    if not simplify_masking:
    #   "s_{}_{}".format(i, k)
        if DEBUG:
            print ('...s variables...')
            var_ct = len(var_to_var_desc)
        for s_idx in range(num_states):
            for f_idx in range(num_features):
                addVariable(gens(s_idx, f_idx), types(), binary_variables, integer_variables, positive_variables, variables)
        if DEBUG:
            print ('......{} declared.'.format(len(var_to_var_desc) - var_ct))

    if not simplify_masking:
    #   "d_{}_{}_{}".format(i, i', k)
    #       symmetric between states
        if DEBUG:
            print ('...d variables...')
            var_ct = len(var_to_var_desc)
        for s_idx1 in range(num_states-1):
            for s_idx2 in range(s_idx1+1, num_states):
                for f_idx in range(num_features):
                    addVariable(gend(s_idx1, s_idx2, f_idx), typed(), binary_variables, integer_variables, positive_variables, variables)
        if DEBUG:
            print ('......{} declared.'.format(len(var_to_var_desc) - var_ct))

    D_pairs = set()
    if not simplify_masking:
    #   "D_{}_{}".format(i, i')
    #       symmetric between states
        if DEBUG:
            print ('...D variables...')
            var_ct = len(var_to_var_desc)
        for (s_idx1, a_idx1, s_idx2) in tqdm.tqdm(triples):
            for (s_idx3, a_idx2, s_idx4) in triples:
                if a_idx1 != a_idx2:
                    continue
                if s_idx1 > s_idx3:
                    D_pairs.add((s_idx3, s_idx1))
                if s_idx1 < s_idx3:
                    D_pairs.add((s_idx1, s_idx3))
                if s_idx2 > s_idx4:
                    D_pairs.add((s_idx4, s_idx2))
                if s_idx2 < s_idx4:
                    D_pairs.add((s_idx2, s_idx4))
        if DEBUG:
            print ('......{} pairs identified.'.format(len(D_pairs)))
        for (s_idx1, s_idx2) in tqdm.tqdm(D_pairs, total=len(D_pairs), desc="D_pairs"):
            addVariable(genD(s_idx1, s_idx2), typeD(), binary_variables, integer_variables, positive_variables, variables)
        if DEBUG:
            print ('......{} declared.'.format(len(var_to_var_desc) - var_ct))

    #   "q_{}_{}".format(i, j)
    if DEBUG:
        print ('...q variables...')
        var_ct = len(var_to_var_desc)
    for s_idx in range(num_states):
        for c_idx in range(num_classes):
            addVariable(genq(s_idx, c_idx), typeq(), binary_variables, integer_variables, positive_variables, variables)
    if DEBUG:
        print ('......{} declared.'.format(len(var_to_var_desc) - var_ct))

    if not simplify_masking:
    #   "Q_{}".format(i)
        if DEBUG:
            print ('...Q variables...')
            var_ct = len(var_to_var_desc)
        for s_idx in range(num_states):
            addVariable(genQ(s_idx), typeQ(), binary_variables, integer_variables, positive_variables, variables)
        if DEBUG:
            print ('......{} declared.'.format(len(var_to_var_desc) - var_ct))

    if not simplify_masking:
    #   "DB_{}_{}".format(i, i')
    #       -- binary variable
        if DEBUG:
            print ('...DB variables...')
            var_ct = len(var_to_var_desc)
        for (s_idx1, s_idx2) in tqdm.tqdm(D_pairs, total=len(D_pairs), desc="D_pairs"):
            addVariable(genDB(s_idx1, s_idx2), typeDB(), binary_variables, integer_variables, positive_variables, variables)
        if DEBUG:
            print ('......{} declared.'.format(len(var_to_var_desc) - var_ct))

    #   "FQ_{}_{}_{}_{}".format(j, i, j', i')
    #       symmetric between states
    if DEBUG:
        print ('...FQ variables...')
        var_ct = len(var_to_var_desc)
    for (s_idx1, a_idx, s_idx2) in tqdm.tqdm(triples):
        for c_idx1 in range(num_classes):
            for c_idx2 in range(num_classes):
                if s_idx1 == s_idx2 and c_idx1 != c_idx2: # Skip since always invalid
                    continue
                if s_idx1 > s_idx2:
                    addVariable(genFQ(c_idx2, s_idx2, c_idx1, s_idx1), typeFQ(), binary_variables, integer_variables, positive_variables, variables)
                else:
                    addVariable(genFQ(c_idx1, s_idx1, c_idx2, s_idx2), typeFQ(), binary_variables, integer_variables, positive_variables, variables)
    if DEBUG:
        print ('......{} declared.'.format(len(var_to_var_desc) - var_ct))

    #   "Cq_targ_{}".format(j)
    #       -- integer variable
    #   "Cq_nontarg_{}".format(j)
    #       -- integer variable
    if DEBUG:
        print ('...Cq_targ1/Cq_nontarg1 variables...')
        var_ct = len(var_to_var_desc)
    for c_idx in range(num_classes):
        addVariable(genCq_targ1(c_idx), typeCq_targ1(), binary_variables, integer_variables, positive_variables, variables)
        addVariable(genCq_nontarg1(c_idx), typeCq_nontarg1(), binary_variables, integer_variables, positive_variables, variables)
    if DEBUG:
        print ('......{} declared.'.format(len(var_to_var_desc) - var_ct))

    #   "Cq_targ_{}_{}".format(j, a)
    #       -- integer variable
    #   "Cq_nontarg_{}_{}".format(j, a)
    #       -- integer variable
    #   "Cq_{}_{}".format(j, a)
    #       -- integer variable
    #   "Pq_targ_{}_{}".format(a, j)
    #       -- non-negative real-valued variable bounded by [0, 1]
    #   "Pq_nontarg_{}_{}".format(a, j)
    #       -- non-negative real-valued variable bounded by [0, 1]
    if DEBUG:
        print ('...Cq_targ2/Cq_nontarg2/C2/Pq_targ/Pq_nontarg variables...')
        var_ct = len(var_to_var_desc)
    for c_idx in range(num_classes):
        for a_idx in range(num_actions):
            addVariable(genCq_targ2(c_idx, a_idx), typeCq_targ2(), binary_variables, integer_variables, positive_variables, variables)
            addVariable(genCq_nontarg2(c_idx, a_idx), typeCq_nontarg2(), binary_variables, integer_variables, positive_variables, variables)
            addVariable(genCq1(c_idx, a_idx), typeCq1(), binary_variables, integer_variables, positive_variables, variables)
            addVariable(genPq_targ(a_idx, c_idx), typePq_targ(), binary_variables, integer_variables, positive_variables, variables)
            addVariable(genPq_nontarg(a_idx, c_idx), typePq_nontarg(), binary_variables, integer_variables, positive_variables, variables)
    if DEBUG:
        print ('......{} declared.'.format(len(var_to_var_desc) - var_ct))

    #   "Cq_{}_{}_{}".format(j, a, j')
    #       -- integer variable
    #   "Pq_{}_{}_{}".format(j', a, j)
    #       -- non-negative real-valued variable bounded by [0, 1]
    if DEBUG:
        print ('...Cq3/Pq variables...')
        var_ct = len(var_to_var_desc)
    for c_idx1 in range(num_classes):
        for a_idx in range(num_actions):
            for c_idx2 in range(num_classes):
                addVariable(genCq2(c_idx1, a_idx, c_idx2), typeCq2(), binary_variables, integer_variables, positive_variables, variables)
                addVariable(genPq(c_idx2, a_idx, c_idx1), typePq(), binary_variables, integer_variables, positive_variables, variables)
    if DEBUG:
        print ('......{} declared.'.format(len(var_to_var_desc) - var_ct))

    #   "P_targ_{}_{}".format(a, i)
    #       -- non-negative real-valued variable bounded by [0, 1]
    #   "P_nontarg_{}_{}".format(a, i)
    #       -- non-negative real-valued variable bounded by [0, 1]
    #   "P_{}_{}_{}".format(i', a, i)
    #       -- non-negative real-valued variable bounded by [0, 1]
    if DEBUG:
        print ('...P_targ/P_nontarg/P variables...')
        var_ct = len(var_to_var_desc)
    for (s_idx1, a_idx, s_idx2) in tqdm.tqdm(triples):
        addVariable(genP_targ(a_idx, s_idx1), typeP_targ(), binary_variables, integer_variables, positive_variables, variables)
        addVariable(genP_nontarg(a_idx, s_idx1), typeP_nontarg(), binary_variables, integer_variables, positive_variables, variables)
        addVariable(genP(s_idx2, a_idx, s_idx1), typeP(), binary_variables, integer_variables, positive_variables, variables)
    if DEBUG:
        print ('......{} declared.'.format(len(var_to_var_desc) - var_ct))

    #   "r_{}_{}_{}".format(i, a, i')
    #       -- real-valued variable bounded by [-peak, peak]
    #   "FRV_targ_{}_{}_{}".format(i, a, i')
    #       -- real-valued variable bounded by [-2 * peak, 2 * peak]
    #   "FRV_nontarg_{}_{}_{}".format(i, a, i')
    #       -- real-valued variable bounded by [-2 * peak, 2 * peak]
    if DEBUG:
        print ('...r/FRV_targ/FRV_nontarg/FRV_maxMagnitude variables...')
        var_ct = len(var_to_var_desc)
    for (s_idx1, a_idx, s_idx2) in tqdm.tqdm(triples):
        addVariable(genr(s_idx1, a_idx, s_idx2), typer(), binary_variables, integer_variables, positive_variables, variables)
        if (s_idx1, a_idx, s_idx2) in dm_triples:
            addVariable(genFRV_targ(s_idx1, a_idx, s_idx2), typeFRV_targ(), binary_variables, integer_variables, positive_variables, variables)
        if (s_idx1, a_idx, s_idx2) in others_triples:
            addVariable(genFRV_nontarg(s_idx1, a_idx, s_idx2), typeFRV_nontarg(), binary_variables, integer_variables, positive_variables, variables)
    addVariable(genFRV_maxMagnitude(), typeFRV_maxMagnitude(), binary_variables, integer_variables, positive_variables, variables)
    if DEBUG:
        print ('......{} declared.'.format(len(var_to_var_desc) - var_ct))

    #   "rq_{}_{}_{}".format(j, a, j')
    #       -- real-valued variable bounded by [-peak, peak]
    if DEBUG:
        print ('...rq variables...')
        var_ct = len(var_to_var_desc)
    for c_idx1 in range(num_classes):
        for a_idx in range(num_actions):
            for c_idx2 in range(num_classes):
                addVariable(genrq(c_idx1, a_idx, c_idx2), typerq(), binary_variables, integer_variables, positive_variables, variables)
    if DEBUG:
        print ('......{} declared.'.format(len(var_to_var_desc) - var_ct))


#   "LER_{}".format(t)
#       -- real valued variable corresponding to the linear expected reward for trajectory t (without discounting)
#       -- unbounded
    if DEBUG:
        print ('...LER variables...')
        var_ct = len(var_to_var_desc)
    for t_idx, traj in enumerate(trajectories):
        var_name = genLER(t_idx)
        addVariable(var_name, typeLER(), binary_variables, integer_variables, positive_variables, variables)
        LER_to_trajectory[var_name] = ( t_idx, tuple(traj) )
    if DEBUG:
        print ('......{} declared.'.format(len(var_to_var_desc) - var_ct))

#   "LB_{}".format(u)
#   "UB_{}".format(u)
#       -- real valued variable corresponding to the lower and upper bounds of LER for trajectories in poset u
    if DEBUG:
        print ('...LB/UN variables...')
        var_ct = len(var_to_var_desc)
    for p_idx in range(num_posets):
        addVariable(genLB(p_idx), typeLB(), binary_variables, integer_variables, positive_variables, variables)
        addVariable(genUB(p_idx), typeUB(), binary_variables, integer_variables, positive_variables, variables)
    if DEBUG:
        print ('......{} declared.'.format(len(var_to_var_desc) - var_ct))

#   "delta_{}_{}".format(u, u+1)
#       -- real valued variable corresponding to the distance between two posets u and u+1 (assumed total ordering)
#       -- bounded by [1, inf)
    if DEBUG:
        print ('...delta variables...')
        var_ct = len(var_to_var_desc)
    for p_idx in range(num_posets - 1):
        addVariable(gendelta(p_idx, p_idx + 1), typedelta(), binary_variables, integer_variables, positive_variables, variables)
    if DEBUG:
        print ('......{} declared.'.format(len(var_to_var_desc) - var_ct))

    # Generate declarations

    if len(binary_variables) > 0:
        rows.append(createDeclarations("BINARY_VARIABLES", binary_variables))
    if len(integer_variables) > 0:
        rows.append(createDeclarations("INTEGER_VARIABLES", integer_variables))
    if len(positive_variables) > 0:
        rows.append(createDeclarations("POSITIVE_VARIABLES", positive_variables))
    if len(variables) > 0:
        rows.append(createDeclarations("VARIABLES", variables))

    if OPTIMIZER == "GUROBI":
        for row in rows:
            if row != None:
                sys.exit('declareVariables has a non-empty row {} for GUROBI'.format(row))
        rows = list()

    del binary_variables
    del integer_variables
    del positive_variables
    del variables
    if DEBUG:
        print ("Total variables declared = {}".format(len(var_to_var_desc)))
    return (rows, LER_to_trajectory, D_pairs)

#
# Bound variables
def boundVariables(num_states, num_actions, num_features, num_classes, peak, trajectories, num_posets, triples, dm_triples, others_triples, simplify_masking, D_pairs):
    global OPTIMIZER
    # Returns a list of rows detailing the bounds
    rows = list()
    if OPTIMIZER != 'GUROBI':
        rows.append("LOWER_BOUNDS{")

    #   "s_{}_{}".format(i, k)
    #       -- non-negative real-valued variable bounded by [0, 1]
    #   "D_{}_{}".format(i, i')
    #       -- non-negative real-valued variable bounded by [0, 1]
    #   "d_{}_{}_{}".format(i, i', k)
    #       -- non-negative real-valued variable bounded by [0, 1]
    #   "q_{}_{}".format(i, j)
    #       -- binary variable
    #   "Q_{}".format(i)
    #       -- non-negative real-valued variable bounded by [1, 2^(num_classes-1)]
    #   "FQ_{}_{}_{}_{}".format(j, i, j', i')
    #       -- binary variable
    if not simplify_masking:
        for s_idx1 in range(num_states):
            rows.append(lowerboundQ(s_idx1))

    #   "Pq_targ_{}_{}".format(a, j)
    #       -- non-negative real-valued variable bounded by [0, 1]
    #   "Pq_nontarg_{}_{}".format(a, j)
    #       -- non-negative real-valued variable bounded by [0, 1]
    #   "Pq_{}_{}_{}".format(j', a, j)
    #       -- non-negative real-valued variable bounded by [0, 1]

    #   "P_targ_{}_{}".format(a, j)
    #       -- non-negative real-valued variable bounded by [0, 1]
    #   "P_nontarg_{}_{}".format(a, j)
    #       -- non-negative real-valued variable bounded by [0, 1]
    #   "P_{}_{}_{}".format(j', a, j)
    #       -- non-negative real-valued variable bounded by [0, 1]

    #   "r_{}_{}_{}".format(i, a, i')
    #       -- real-valued variable bounded by [-peak, peak]
    #   "FRV_targ_{}_{}_{}".format(i, a, i')
    #       -- real-valued variable bounded by [-2 * peak, 2 * peak]
    #   "FRV_nontarg_{}_{}_{}".format(i, a, i')
    #       -- real-valued variable bounded by [-2 * peak, 2 * peak]
    for (s_idx1, a_idx, s_idx2) in triples:
        rows.append(lowerboundr(s_idx1, a_idx, s_idx2, peak))
        if (s_idx1, a_idx, s_idx2) in dm_triples:
            rows.append(lowerboundFRV_targ(s_idx1, a_idx, s_idx2, peak))
        if (s_idx1, a_idx, s_idx2) in others_triples:
            rows.append(lowerboundFRV_nontarg(s_idx1, a_idx, s_idx2, peak))
    #   "rq_{}_{}_{}".format(j, a, j')
    #       -- real-valued variable bounded by [-peak, peak]
    for c_idx1 in range(num_classes):
        for a_idx in range(num_actions):
            for c_idx2 in range(num_classes):
                rows.append(lowerboundrq(c_idx1, a_idx, c_idx2, peak))
    #   "delta_{}_{}".format(u, u+1)
    #       -- real valued variable corresponding to the distance between
    #           two posets u and u+1 (assumed total ordering)
    #       -- bounded by [1, inf)
    for p_idx in range(num_posets - 1):
        rows.append(lowerbounddelta(p_idx, p_idx + 1))

    if OPTIMIZER != 'GUROBI':
        rows.append("}")
        rows.append(" ")

        rows.append("UPPER_BOUNDS{")

    for s_idx1 in range(num_states):
        if not simplify_masking:
            rows.append(upperboundQ(s_idx1, num_classes))

    for a_idx in range(num_actions):
        for c_idx1 in range(num_classes):
            rows.append(upperboundPq_targ(a_idx, c_idx1))
            rows.append(upperboundPq_nontarg(a_idx, c_idx1))
            for c_idx2 in range(num_classes):
                rows.append(upperboundPq(c_idx2, a_idx, c_idx1))

    for (s_idx1, a_idx, s_idx2) in triples:
        rows.append(upperboundP_targ(a_idx, s_idx1))
        rows.append(upperboundP_nontarg(a_idx, s_idx1))
        rows.append(upperboundP(s_idx2, a_idx, s_idx1))

    for (s_idx1, a_idx, s_idx2) in triples:
        rows.append(upperboundr(s_idx1, a_idx, s_idx2, peak))
        if (s_idx1, a_idx, s_idx2) in dm_triples:
            rows.append(upperboundFRV_targ(s_idx1, a_idx, s_idx2, peak))
        if (s_idx1, a_idx, s_idx2) in others_triples:
            rows.append(upperboundFRV_nontarg(s_idx1, a_idx, s_idx2, peak))

    for c_idx1 in range(num_classes):
        for a_idx in range(num_actions):
            for c_idx2 in range(num_classes):
                rows.append(upperboundrq(c_idx1, a_idx, c_idx2, peak))

    if OPTIMIZER != 'GUROBI':
        rows.append("}")
    if OPTIMIZER == "GUROBI":
        for row in rows:
            if row != None:
                sys.exit('boundVariables returned a non-empty row {} for GUROBI'.format(row))
        rows = list()
    return (rows)

#
# Build constraints
def generateConstraints(num_states, num_actions, num_features, num_classes, states, peak, posets, trajectories, triples, dm_triples, others_triples, simplify_masking, C2, C3, C1_targ, C2_targ, C1_nontarg, C2_nontarg, D_pairs):
    # Returns a list of rows detailing the constraint declarations and constraints -- specific to (BARON)
    equation_names = list()
    equation_names_set = set()
    equations = list()

    if not simplify_masking:
        # Map state index to state tuple
        state_to_tuple = dict()
        for key, value in states.items():
            state_to_tuple[value] = key
        assert num_features == len(state_to_tuple[0]), "num_features does not match state feature vector length"

    if not simplify_masking:
        if DEBUG:
            print ("...s constraints...")
        #   if "s_{}_{}".format(i, k) == 0: i.e., the value for
        #                                       feature k for state i is 0
        # s1:      "s_{}_{}".format(i, k) = 0
        #   else:
        # s2:      "s_{}_{}".format(i, k) - "M_{}".format(k) = 0
        #       -- state i feature k
        num_constraints = 0
        for s_idx in range(num_states):
            for f_idx in range(num_features):
                if state_to_tuple[s_idx][f_idx] == 0:
                    eqn = constraints1_name(s_idx, f_idx)
                    if not eqn in equation_names_set: # skip
                        equation_names_set.add(eqn)
                        equation_names.append(eqn)
                        equations.append(constraints1(s_idx, f_idx))
                        num_constraints += 1
                else:
                    eqn = constraints2_name(s_idx, f_idx)
                    if not eqn in equation_names_set: # skip
                        equation_names_set.add(eqn)
                        equation_names.append(eqn)
                        equations.append(constraints2(s_idx, f_idx))
                        num_constraints += 1
        if DEBUG:
            print ("......{} created.".format(num_constraints))

    if not simplify_masking:
        if DEBUG:
            print ("...d constraints...")
        # d1:  "d_{}_{}_{}".format(i, i') - "s_{}_{}".format(i, k) -
        #       "s_{}_{}".format(i', k) <= 0
        # d2:  "d_{}_{}_{}".format(i, i') - "s_{}_{}".format(i, k) +
        #       "s_{}_{}".format(i', k) >= 0
        # d3:  "d_{}_{}_{}".format(i, i') + "s_{}_{}".format(i, k) -
        #       "s_{}_{}".format(i', k) >= 0
        # d4:  "d_{}_{}_{}".format(i, i') + "s_{}_{}".format(i, k) +
        #       "s_{}_{}".format(i', k) <= 2
        #       -- states i and i', feature k
        num_constraints = 0
        for (s_idx1, s_idx2) in tqdm.tqdm(D_pairs, total=len(D_pairs), desc="D_pairs"):
            for f_idx in range(num_features):
                if OPTIMIZER == 'GUROBI':
                    eqn = constraintdg_name(s_idx1, s_idx2, f_idx)
                    if not eqn in equation_names_set: # skip
                        equation_names_set.add(eqn)
                        equation_names.append(eqn)
                        equations.append(constraintdg(s_idx1, s_idx2, f_idx))
                        num_constraints += 1
                else:
                    eqn = constraintd1_name(s_idx1, s_idx2, f_idx)
                    if not eqn in equation_names_set: # skip
                        equation_names_set.add(eqn)
                        equation_names.append(eqn)
                        equations.append(constraintd1(s_idx1, s_idx2, f_idx))
                        num_constraints += 1
                    eqn = constraintd2_name(s_idx1, s_idx2, f_idx)
                    if not eqn in equation_names_set: # skip
                        equation_names_set.add(eqn)
                        equation_names.append(eqn)
                        equations.append(constraintd2(s_idx1, s_idx2, f_idx))
                        num_constraints += 1
                    eqn = constraintd3_name(s_idx1, s_idx2, f_idx)
                    if not eqn in equation_names_set: # skip
                        equation_names_set.add(eqn)
                        equation_names.append(eqn)
                        equations.append(constraintd3(s_idx1, s_idx2, f_idx))
                        num_constraints += 1
                eqn = constraintd4_name(s_idx1, s_idx2, f_idx)
                if not eqn in equation_names_set: # skip
                    equation_names_set.add(eqn)
                    equation_names.append(eqn)
                    equations.append(constraintd4(s_idx1, s_idx2, f_idx))
                    num_constraints += 1
        if DEBUG:
            print ("......{} created.".format(num_constraints))

    if not simplify_masking:
        if DEBUG:
            print ("...D constraints...")
        # D1:  "D_{}_{}".format(i, i') -
        #       \sum_{k=0}^{num_features - 1} "d_{}_{}_{}".format(i, i', k) <= 0
        #       -- states i and i'
        # D2:  "D_{}_{}".format(i, i') - "d_{}_{}_{}".format(i, i', k) >= 0
        #       -- states i and i', feature k
        num_constraints = 0
        for (s_idx1, s_idx2) in tqdm.tqdm(D_pairs, total=len(D_pairs), desc="D_pairs"):
            eqn = constraintD1_name(s_idx1, s_idx2)
            if not eqn in equation_names_set: # skip
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintD1(s_idx1, s_idx2, num_features))
                num_constraints += 1
            if OPTIMIZER != 'GUROBI':
                for f_idx in range(num_features):
                    eqn = constraintD2_name(s_idx1, s_idx2, f_idx)
                    if not eqn in equation_names_set: # skip
                        equation_names_set.add(eqn)
                        equation_names.append(eqn)
                        equations.append(constraintD2(s_idx1, s_idx2, f_idx))
                        num_constraints += 1
        if DEBUG:
            print ("......{} created.".format(num_constraints))

    if DEBUG:
        print ("...q constraints...")
    # q:  \sum_{j=0}^{num_classes - 1} "q_{}_{}".format(i, j) = 1
    #       -- state i
    num_constraints = 0
    for s_idx in range(num_states):
        eqn = constraintq_name(s_idx)
        if not eqn in equation_names_set: # skip
            equation_names_set.add(eqn)
            equation_names.append(eqn)
            equations.append(constraintq(s_idx, num_classes))
            num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    if not simplify_masking:
        if DEBUG:
            print ("...Q constraints...")
        # Q1:  "Q_{}".format(i)
        #       - \sum_{j=0}^{num_classes - 1} 2^j * "q_{}_{}".format(i, j)
        #       -- state i
        # Q2:  2^num_classes * "D_{}_{}".format(i, i') - "Q_{}".format(i)
        #       + "Q_{}".format(i') >= 0
        # Q3:  2^num_classes * "D_{}_{}".format(i, i') + "Q_{}".format(i)
        #       - "Q_{}".format(i') >= 0
        # Q4:  "Q_{}".format(i) - "Q_{}".format(i') - "D_{}_{}".format(i, i')
        #       + 2^num_classes * "DB_{}_{}".format(i, i') >= 0
        # Q5:  "Q_{}".format(i') - "Q_{}".format(i) - "D_{}_{}".format(i, i')
        #       - 2^num_classes * "DB_{}_{}".format(i, i') >= -2^num_classes
        #       -- states i and i'
        num_constraints = 0
        for s_idx1 in range(num_states):
            eqn = constraintQ1_name(s_idx1)
            if not eqn in equation_names_set: # skip
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintQ1(s_idx1, num_classes))
                num_constraints += 1

        for (s_idx1, s_idx2) in tqdm.tqdm(D_pairs, total=len(D_pairs), desc="D_pairs"):
            eqn = constraintQ2_name(s_idx1, s_idx2)
            if not eqn in equation_names_set: # skip
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintQ2(s_idx1, s_idx2, num_classes))
                num_constraints += 1
            eqn = constraintQ3_name(s_idx1, s_idx2)
            if not eqn in equation_names_set: # skip
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintQ3(s_idx1, s_idx2, num_classes))
                num_constraints += 1
            eqn = constraintQ4_name(s_idx1, s_idx2)
            if not eqn in equation_names_set: # skip
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintQ4(s_idx1, s_idx2, num_classes))
                num_constraints += 1
            eqn = constraintQ5_name(s_idx1, s_idx2)
            if not eqn in equation_names_set: # skip
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintQ5(s_idx1, s_idx2, num_classes))
                num_constraints += 1
        if DEBUG:
            print ("......{} created.".format(num_constraints))

    if DEBUG:
        print ("...FQ constraints...")
    # FQ1:  "FQ_{}_{}_{}_{}".format(j, i, j', i') - "q_{}_{}".format(i, j) <= 0
    # FQ2:  "FQ_{}_{}_{}_{}".format(j, i, j', i') - "q_{}_{}".format(i', j') <= 0
    # FQ3:  "FQ_{}_{}_{}_{}".format(j, i, j', i') - "q_{}_{}".format(i, j)
    #       - "q_{}_{}".format(i', j') >= -1
    #       -- states i and i', classes j and j'
    num_constraints = 0
    for (s_idx1, _, s_idx2) in tqdm.tqdm(triples):
        for c_idx1 in range(num_classes):
            for c_idx2 in range(num_classes):
                if s_idx1 == s_idx2 and c_idx1 != c_idx2: # Skip
                    continue
                if OPTIMIZER == 'GUROBI':
#                if False:
                    eqn = constraintFQG_name(c_idx1, s_idx1, c_idx2, s_idx2)
                    if not eqn in equation_names_set: # skip
                        equation_names_set.add(eqn)
                        equation_names.append(eqn)
                        equations.append(constraintFQG(c_idx1, s_idx1, c_idx2, s_idx2))
                        num_constraints += 1
                else:
                    eqn = constraintFQ1_name(c_idx1, s_idx1, c_idx2, s_idx2)
                    if not eqn in equation_names_set: # skip
                        equation_names_set.add(eqn)
                        equation_names.append(eqn)
                        equations.append(constraintFQ1(c_idx1, s_idx1, c_idx2, s_idx2))
                        num_constraints += 1
                    if s_idx1 != s_idx2:
                        eqn = constraintFQ2_name(c_idx1, s_idx1, c_idx2, s_idx2)
                        if not eqn in equation_names_set: # skip
                            equation_names_set.add(eqn)
                            equation_names.append(eqn)
                            equations.append(constraintFQ2(c_idx1, s_idx1, c_idx2, s_idx2))
                            num_constraints += 1
                    eqn = constraintFQ3_name(c_idx1, s_idx1, c_idx2, s_idx2)
                    if not eqn in equation_names_set: # skip
                        equation_names_set.add(eqn)
                        equation_names.append(eqn)
                        equations.append(constraintFQ3(c_idx1, s_idx1, c_idx2, s_idx2))
                        num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    if DEBUG:
        print ("...Cq_targ constraints...")
    # Cq_targ1:
    #   "Cq_targ_{}".format(j) - \sum_{i=0}^{num_states - 1} C_{DM}(i)
    #       * "q_{}_{}".format(i, j) = 0
    #       -- class j
    # Cq_targ2:
    #   "Cq_targ_{}_{}".format(j, a) - \sum_{i=0}^{num_states - 1} C_{DM}(i, a)
    #       * "q_{}_{}".format(i, j) = 0
    #       -- class j and action a
    num_constraints = 0
    for c_idx in range(num_classes):
        eqn = constraintCq_targ1_name(c_idx)
        if not eqn in equation_names_set: # skip
            equation_names_set.add(eqn)
            equation_names.append(eqn)
            equations.append(constraintCq_targ1(c_idx, num_states, C1_targ))
            num_constraints += 1
        for a_idx in range(num_actions):
            eqn = constraintCq_targ2_name(c_idx, a_idx)
            if not eqn in equation_names_set: # skip
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintCq_targ2(c_idx, a_idx, num_states, C2_targ))
                num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    if DEBUG:
        print ("...Cq_nontarg constraints...")
    # Cq_nontarg1:
    #   "Cq_nontarg_{}".format(j) - \sum_{i=0}^{num_states - 1} C_{NDM}(i)
    #       * "q_{}_{}".format(i, j) = 0
    #       -- class j
    # Cq_nontarg2:
    #   "Cq_nontarg_{}_{}".format(j, a) - \sum_{i=0}^{num_states - 1} C_{NDM}(i, a)
    #       * "q_{}_{}".format(i, j) = 0
    num_constraints = 0
    for c_idx in range(num_classes):
        eqn = constraintCq_nontarg1_name(c_idx)
        if not eqn in equation_names_set: # skip
            equation_names_set.add(eqn)
            equation_names.append(eqn)
            equations.append(constraintCq_nontarg1(c_idx, num_states, C1_nontarg))
            num_constraints += 1
        for a_idx in range(num_actions):
            eqn = constraintCq_nontarg2_name(c_idx, a_idx)
            if not eqn in equation_names_set: # skip
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintCq_nontarg2(c_idx, a_idx, num_states, C2_nontarg))
                num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    if DEBUG:
        print ("...Cq constraints...")
    # Cq1:
    #   "Cq_{}_{}".format(j, a) - \sum{i=0}^{num_states - 1} C(i, a)
    #       * "q_{}_{}".format(i, j) = 0
    #       -- class j and action a
    # Cq2:
    #   "Cq_{}_{}_{}".format(j, a, j') - \sum{i=0}^{num_states - 1}
    #       \sum{i'=0}^{num_states - 1} C(i, a, i')
    #       * "FQ_{}_{}_{}_{}".format(j, i, j', i') = 0
    #       -- classes j and j' and action a
    #       -- Must flip i and i' if i>i' in FQ term
    #       -- Must flip i and i' if i>i' in FQ term
    num_constraints = 0
    for c_idx1 in range(num_classes):
        for a_idx in range(num_actions):
            eqn = constraintCq1_name(c_idx1, a_idx)
            if not eqn in equation_names_set: # skip
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintCq1(c_idx1, a_idx, num_states, C2))
                num_constraints += 1
            for c_idx2 in range(num_classes):
                eqn = constraintCq2_name(c_idx1, a_idx, c_idx2)
                if not eqn in equation_names_set: # skip
                    equation_names_set.add(eqn)
                    equation_names.append(eqn)
                    equations.append(constraintCq2(c_idx1, a_idx, c_idx2, num_states, C3, trajectories, triples))
                    num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    if DEBUG:
        print ("...Pq_targ constraints...")
    # Pq_targ:
    #   "Cq_targ_{}".format(j) * "Pq_targ_{}_{}".format(a, j)
    #       - "Cq_targ_{}_{}".format(j, a) = 0
    #       -- class j and action a
    num_constraints = 0
    for c_idx in range(num_classes):
        for a_idx in range(num_actions):
            eqn = constraintPq_targ_name(a_idx, c_idx)
            if not eqn in equation_names_set: # skip
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintPq_targ(a_idx, c_idx))
                num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    if DEBUG:
        print ("...Pq_nontarg constraints...")
    # Pq_nontarg:
    #   "Cq_nontarg_{}".format(j) * "Pq_nontarg_{}_{}".format(a, j)
    #       - "Cq_nontarg_{}_{}".format(j, a) = 0
    #       -- class j and action a
    num_constraints = 0
    for c_idx in range(num_classes):
        for a_idx in range(num_actions):
            eqn = constraintPq_nontarg_name(a_idx, c_idx)
            if not eqn in equation_names_set: # skip
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintPq_nontarg(a_idx, c_idx))
                num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    if DEBUG:
        print ("...Pq constraints...")
    # Pq:
    #   "Cq_{}_{}".format(j, a) * "Pq_{}_{}_{}".format(j', a, j)
    #       - "Cq_{}_{}_{}".format(j, a, j') = 0
    #       -- classes j and j' and action a
    num_constraints = 0
    for c_idx2 in range(num_classes):
        for a_idx in range(num_actions):
            for c_idx1 in range(num_classes):
                eqn = constraintPq_name(c_idx2, a_idx, c_idx1)
                if not eqn in equation_names_set: # skip
                    equation_names_set.add(eqn)
                    equation_names.append(eqn)
                    equations.append(constraintPq(c_idx2, a_idx, c_idx1))
                    num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    if DEBUG:
        print ("...P_targ constraints...")
    # P_targ1:
    #   "Pq_targ_{}_{}".format(a, j) - "P_targ_{}_{}".format(a, i)
    #       - "q_{}_{}".format(i, j) >= -1
    # P_targ2:
    #   "P_targ_{}_{}".format(a, i) - "Pq_targ_{}_{}".format(a, j)
    #       - "q_{}_{}".format(i, j) >= -1
    #       -- action a, class j, state i
    num_constraints = 0
    for (s_idx, a_idx, _) in triples:
        for c_idx in range(num_classes):
            eqn = constraintP_targ1_name(a_idx, s_idx, c_idx)
            if not eqn in equation_names_set: # skip
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintP_targ1(a_idx, s_idx, c_idx))
                num_constraints += 1
            eqn = constraintP_targ2_name(a_idx, s_idx, c_idx)
            if not eqn in equation_names_set: # skip
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintP_targ2(a_idx, s_idx, c_idx))
                num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    if DEBUG:
        print ("...P_nontarg constraints...")
    # P_nontarg1:
    #   "Pq_nontarg_{}_{}".format(a, j) - "P_nontarg_{}_{}".format(a, i)
    #       - "q_{}_{}".format(i, j) >= -1
    # P_nontarg2:
    #   "P_nontarg_{}_{}".format(a, i) - "Pq_nontarg_{}_{}".format(a, j)
    #       - "q_{}_{}".format(i, j) >= -1
    #       -- action a, class j, state i
    num_constraints = 0
    for (s_idx, a_idx, _) in triples:
        for c_idx in range(num_classes):
            eqn = constraintP_nontarg1_name(a_idx, s_idx, c_idx)
            if not eqn in equation_names_set: # skip
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintP_nontarg1(a_idx, s_idx, c_idx))
                num_constraints += 1
            eqn = constraintP_nontarg2_name(a_idx, s_idx, c_idx)
            if not eqn in equation_names_set: # skip
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintP_nontarg2(a_idx, s_idx, c_idx))
                num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    if DEBUG:
        print ("...P constraints...")
    # P1:
    #   "Pq_{}_{}_{}".format(j', a, j) - "P_{}_{}_{}".format(i', a, i)
    #       - "q_{}_{}".format(i, j) - "q_{}_{}".format(i', j') >= -2
    # P2:
    #   "P_{}_{}_{}".format(i', a, i) - "Pq_{}_{}_{}".format(j', a, j)
    #       - "q_{}_{}".format(i, j) - "q_{}_{}".format(i', j') >= -2
    #       -- classes j, j', action a, states i, i'
    num_constraints = 0
    for (s_idx1, a_idx, s_idx2) in tqdm.tqdm(triples):
        for c_idx1 in range(num_classes):
            for c_idx2 in range(num_classes):
                eqn = constraintP1_name(s_idx2, a_idx, s_idx1, c_idx2, c_idx1)
                if not eqn in equation_names_set: # skip
                    equation_names_set.add(eqn)
                    equation_names.append(eqn)
                    equations.append(constraintP1(s_idx2, a_idx, s_idx1, c_idx2, c_idx1))
                    num_constraints += 1
                eqn = constraintP2_name(s_idx2, a_idx, s_idx1, c_idx2, c_idx1)
                if not eqn in equation_names_set: # skip
                    equation_names_set.add(eqn)
                    equation_names.append(eqn)
                    equations.append(constraintP2(s_idx2, a_idx, s_idx1, c_idx2, c_idx1))
                    num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    if DEBUG:
        print ("...r constraints...")
    # r1:
    #   "rq_{}_{}_{}".format(j, a, j') - "r_{}_{}_{}".format(i, a, i')
    #       - 2 * peak * "q_{}_{}".format(i, j) - 2 * peak * "q_{}_{}".format(i', j') >= -4 * peak
    # r2:
    #   "r_{}_{}_{}".format(i, a, i') - "rq_{}_{}_{}".format(j, a, j')
    #       - 2 * peak * "q_{}_{}".format(i, j) - 2 * peak * "q_{}_{}".format(i', j') >= -4 * peak
    num_constraints = 0
    for (s_idx1, a_idx, s_idx2) in tqdm.tqdm(triples):
        for c_idx1 in range(num_classes):
            for c_idx2 in range(num_classes):
                eqn = constraintr1_name(s_idx1, a_idx, s_idx2, c_idx1, c_idx2)
                if not eqn in equation_names_set: # skip
                    equation_names_set.add(eqn)
                    equation_names.append(eqn)
                    equations.append(constraintr1(s_idx1, a_idx, s_idx2, c_idx1, c_idx2, peak))
                    num_constraints += 1
                eqn = constraintr2_name(s_idx1, a_idx, s_idx2, c_idx1, c_idx2)
                if not eqn in equation_names_set: # skip
                    equation_names_set.add(eqn)
                    equation_names.append(eqn)
                    equations.append(constraintr2(s_idx1, a_idx, s_idx2, c_idx1, c_idx2, peak))
                    num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    if DEBUG:
        print ("...FRV_targ/nontarg constraints...")

    #   if {s_i, a, s_i'} occurs in a dm trajectory:
    #       "r_{}_{}_{}".format(i, a, i') - peak * "P_targ_{}_{}".format(a, i')
    #           - "FRV_targ_{}_{}_{}".format(i, a, i') <= 0
    #   else:
    #       "r_{}_{}_{}".format(i, a, i') + peak * "P_nontarg_{}_{}".format(a, i')
    #           - "FRV_nontarg_{}_{}_{}".format(i, a, i') <= 0
    num_constraints = 0
    for (s_idx1, a_idx, s_idx2) in triples:
        if (s_idx1, a_idx, s_idx2) in dm_triples:
            eqn = constraintFRV_targ_name(s_idx1, a_idx, s_idx2)
            if not eqn in equation_names_set:
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintFRV_targ(s_idx1, a_idx, s_idx2, peak))
                num_constraints += 1
        if (s_idx1, a_idx, s_idx2) in others_triples:
            eqn = constraintFRV_nontarg_name(s_idx1, a_idx, s_idx2)
            if not eqn in equation_names_set:
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintFRV_nontarg(s_idx1, a_idx, s_idx2, peak))
                num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    if DEBUG:
        print ("...FRV_maxMagnitude constraints...")

    #   "FRV_targ_{}_{}_{}".format(i, a, i') <= FRV_maxMagnitude
    #   -"FRV_targ_{}_{}_{}".format(i, a, i') <= FRV_maxMagnitude
    #   "FRV_nontarg_{}_{}_{}".format(i, a, i') <= FRV_maxMagnitude
    #   -"FRV_nontarg_{}_{}_{}".format(i, a, i') <= FRV_maxMagnitude
    num_constraints = 0
    for (s_idx1, a_idx, s_idx2) in triples:
        if (s_idx1, a_idx, s_idx2) in dm_triples:
            eqn = constraintFRV_maxMagnitude_targ1_name(s_idx1, a_idx, s_idx2)
            if not eqn in equation_names_set:
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintFRV_maxMagnitude_targ1(s_idx1, a_idx, s_idx2))
                num_constraints += 1
            eqn = constraintFRV_maxMagnitude_targ2_name(s_idx1, a_idx, s_idx2)
            if not eqn in equation_names_set:
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintFRV_maxMagnitude_targ2(s_idx1, a_idx, s_idx2))
                num_constraints += 1
        if (s_idx1, a_idx, s_idx2) in others_triples:
            eqn = constraintFRV_maxMagnitude_nontarg1_name(s_idx1, a_idx, s_idx2)
            if not eqn in equation_names_set:
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintFRV_maxMagnitude_nontarg1(s_idx1, a_idx, s_idx2))
                num_constraints += 1
            eqn = constraintFRV_maxMagnitude_nontarg2_name(s_idx1, a_idx, s_idx2)
            if not eqn in equation_names_set:
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintFRV_maxMagnitude_nontarg2(s_idx1, a_idx, s_idx2))
                num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    if DEBUG:
        print ("...LER constraints...")
    #   "LER_{}".format(t) = \sum_{b=0}^{length(t)-1}
    #       "P_{}_{}_{}".format(s_{i}_<{t},{b}>, a_{a}_<{t},{b+1}>,
    #       s_{i}_<{t},{b+1}>) * "r_{}_{}_{}".format(s_{i}_<{t},{b}>,
    #        a_{a}_<{t},{b+1}>, s_{i}_<{t},{b+1}>)
    #       -- computes LER for trajectory a
    num_constraints = 0
    for t_idx in range(len(trajectories)):
        eqn = constraintLER_name(t_idx)
        if not eqn in equation_names_set:
            equation_names_set.add(eqn)
            equation_names.append(eqn)
            equations.append(constraintLER(t_idx, trajectories[t_idx]))
            num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    if DEBUG:
        print ("...LB/UB/delta constraints...")

    #   "LB_{}".format(u) <= UB_{}".format(u)
    num_constraints = 0
    for p_idx in range(len(posets)):
        eqn = constraintPosetLBUB_name(p_idx)
        if not eqn in equation_names_set:
            equation_names_set.add(eqn)
            equation_names.append(eqn)
            equations.append(constraintPosetLBUB(p_idx))
            num_constraints += 1

    #   "LB_{}".format(u) = "UB_{}".format(u+1) + "delta_{}_{}".format(u, u+1)
    #       -- orders posets
    for p_idx in range(len(posets) - 1):
        eqn = constraintPoset_name(p_idx, p_idx + 1)
        if not eqn in equation_names_set:
            equation_names_set.add(eqn)
            equation_names.append(eqn)
            equations.append(constraintPoset(p_idx, p_idx + 1))
            num_constraints += 1

    #   "LB_{}".format(u) <= "LER_{}".format(t)
    #   "UB_{}".format(u) >= "LER_{}".format(t)
    #   "LB_{}".format(u) <= "LER_{}".format(t)
    #   "UB_{}".format(u) >= "LER_{}".format(t)
    #       -- only if trajectory t is in posets[u]
    for p_idx, poset in enumerate(posets):
        for t_idx in poset:
            eqn = constraintPosetLB_name(p_idx, t_idx)
            if not eqn in equation_names_set:
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintPosetLB(p_idx, t_idx))
                num_constraints += 1
            eqn = constraintPosetUB_name(p_idx, t_idx)
            if not eqn in equation_names_set:
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintPosetUB(p_idx, t_idx))
                num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))


    if DEBUG:
        print ("Total constraints created = {}".format(len(equation_names)))
    # Form into optimzer format

    rows = list()
    if len(equation_names) == 0:
        return (rows)
    if OPTIMIZER == 'GUROBI':
        for row in rows:
            if row != None:
                sys.exit('generateConstraints has a non-empty row {} for GUROBI'.format(row))
        rows = list()
    else:
        row = "EQUATIONS {}".format(equation_names[0])
        for e_idx in range(1, len(equation_names)):
            row += ","
            rows.append(row)
            row = "\t{}".format(equation_names[e_idx])
        row += ";"
        rows.append(row)
        rows.append("")

        for e_idx in range(len(equations)):
            row = "{}:\t{}".format(equation_names[e_idx], equations[e_idx])
            rows.append(row)

    del equation_names
    del equations
    del equation_names_set
    return (rows)

def constructOptimization(**kwargs):
    global var_desc_to_var
    global var_to_var_desc
    global equation_desc_to_equation
    global equation_to_equation_desc
    global next_var_idx
    global next_equation_idx
    var_desc_to_var = dict()
    var_to_var_desc = dict()
    equation_desc_to_equation = dict()
    equation_to_equation_desc = dict()
    next_var_idx = 0
    next_equation_idx = 0

    num_states = kwargs['num_states']
    states = kwargs['states']
    for key, _ in states.items():
        num_features = len(key)
        break;
    num_actions = kwargs['num_actions']
    num_classes = kwargs['num_classes']
    peak = kwargs['peak']
    num_posets = kwargs['num_posets']
    posets = kwargs['posets']
    t = kwargs['all_trajectories']
    dm_trajectory_indices = kwargs['dm_trajectory_indices']
    others_trajectory_indices = kwargs['others_trajectory_indices']
    triples = kwargs['all_triples']
    dm_triples = kwargs['dm_triples']
    others_triples = kwargs['others_triples']
    global OPTIMIZER
    OPTIMIZER = kwargs['optimizer']
    maxtime = kwargs['maxtime']
    license_fn = kwargs['license_fn']
    simplify_masking = kwargs['simplify_masking']
#   maxtime is maximum time allowed for BARON
#   license_fn is the full path filename to the BARON license
    feasibilityOnly = kwargs['feasibility_only']
    # if True, uses Version 0
    # else uses Version 2

    if DEBUG:
        print ("Building FoMDP optimization problem...")

    if OPTIMIZER == "BARON":
        rows = [ "OPTIONS {", "MaxTime:{};".format(maxtime), "LicName: \"{}\";".format(license_fn), "nlpsol: 10;", "}", " " ]
    elif OPTIMIZER == "GUROBI":
        global GUROBI_MODEL
        GUROBI_MODEL = gurobipy.Model("miqcp")
        GUROBI_MODEL.params.NonConvex = 2
        global GUROBI_VARIABLES
        GUROBI_VARIABLES = dict() # Hash from var to gurobi variable
        rows = list() # Not used for GUROBI
    else:
        sys.exit(".constructOptimization(...) -- can only build for BARON or GYROBI.")

    if DEBUG:
        start_time = time.time()
        print ("Computing state/action frequencies and probabilities...")
    _, _, _, C2, C3, C1_targ, C2_targ, C1_nontarg, C2_nontarg = RoMDP.computePs(t, dm_trajectory_indices, others_trajectory_indices)
    if DEBUG:
        print ("\t...elapsed time = {}".format(time.time() - start_time))
        start_time = time.time()
        print ("Declaring variables...")
    ( new_rows, LER_to_trajectories, D_pairs ) = declareVariables(num_states, num_actions, num_features, num_classes, num_posets, t, triples, dm_triples, others_triples, simplify_masking)
    rows.extend(new_rows)
    rows.append(" ")
    if DEBUG:
        print ("\t...elapsed time = {}".format(time.time() - start_time))
        start_time = time.time()
        print ("Constructing variable bounds...")
    rows.extend(boundVariables(num_states, num_actions, num_features, num_classes, peak, t, num_posets, triples, dm_triples, others_triples, simplify_masking, D_pairs))
    rows.append(" ")
    if DEBUG:
        print ("\t...elapsed time = {}".format(time.time() - start_time))
        start_time = time.time()
        print ("Generating constraints...")
    rows.extend(generateConstraints(num_states, num_actions, num_features, num_classes, states, peak, posets, t, triples, dm_triples, others_triples, simplify_masking, C2, C3, C1_targ, C2_targ, C1_nontarg, C2_nontarg, D_pairs))
    if OPTIMIZER == "BARON":
        rows.append(" ")
    if feasibilityOnly:
        if OPTIMIZER == "BARON":
            rows.append("OBJ: minimize")
            rows.append("\t{};".format(1))
        elif OPTIMIZER == "GUROBI":
            GUROBI_MODEL.setObjective(1.0)
    else:
        if OPTIMIZER == 'BARON':
            rows.append("OBJ: maximize")
            obj = ""
            for p_idx in range(num_posets - 1):
                if p_idx > 0:
                    obj += " + "
                obj += gendelta(p_idx, p_idx + 1)
            rows.append("\t{};".format(obj))
#        rows.append("\t{};".format(genT()))
        elif OPTIMIZER == 'GUROBI':
#            GUROBI_MODEL.setObjective(GUROBI_VARIABLES[genFRV_maxMagnitude()])
#        else:
            # Make poset irreducible
            p_idx_start = -1
            for p_idx in range(num_posets):
                if len(posets[p_idx]) > 0:
                    p_idx_start = p_idx
                    break
            if p_idx_start == -1:
                print ("All posets are empty! No objective set.")
                GUROBI_MODEL.setObjective(0.0)
            else:
                p_idx_end = p_idx_start
                for p_idx in range(num_posets - 1, p_idx_start, -1):
                    if len(posets[p_idx]) > 0:
                        p_idx_end = p_idx
                        break
                obj = gurobipy.quicksum(- GUROBI_VARIABLES[gendelta(p_idx, p_idx + 1)] for p_idx in range(p_idx_start, p_idx_end))
                GUROBI_MODEL.setObjective(obj)
    if DEBUG:
        print ("\t...elapsed time = {}".format(time.time() - start_time))
    return ( rows, LER_to_trajectories, None, None, None, var_to_var_desc )

def reportMaskAndStateClusters(var_to_var_desc, answers, analytics):
    Qs = dict()
    Ms = dict()
    for key, value in answers.items():
        item = var_to_var_desc[key]
#        p = parse.parse("Q_{}", item, case_sensitive=True)
#        if p != None: # Possible match
#            try:
#                s_idx = int(p[0])
#                Qs[s_idx] = value
#                continue
#            except ValueError: # No match
#                pass
#        if float(value) == 1:
        if float(value) >= .9: # For tolerance issues of binary values
            p = parse.parse("q_{}_{}", item, case_sensitive=True)
            if p != None:
                try:
                    s_idx = int(p[0])
                    c_idx = int(p[1])
                    Qs[s_idx] = c_idx
                    continue
                except ValueError:
                    pass
        p = parse.parse("M_{}", item, case_sensitive=True)
        if p == None:
            continue
        try:
            f_idx = int(p[0])
            Ms[f_idx] = value
            continue
        except ValueError:
            pass

    Qs_list = [ None for _ in range(1 + max([ int(key) for key, _ in Qs.items() ])) ]
    for key, value in Qs.items():
        Qs_list[key] = value
    if len(Ms) == 0:
        Ms_list = list()
    else:
        Ms_list = [ None for _ in range(1 + max([ int(key) for key, _ in Ms.items() ])) ]
        for key, value in Ms.items():
            Ms_list[key] = value

    analytics["Maskings"]["__header__"] = ( "Mask", "Feature masking" )
    analytics["Maskings"]["Mask"] = tuple(Ms_list)
    rows = list()
    rows.append([ "Mask", tuple(Ms_list) ])
    rows.append([ "" ])
    analytics["Clusters"]["__header__"] = ( "Key: State Index", "Cluster Id" )
    rows.append([ "State Index", "Cluster Id" ])
    for s_idx, cid in enumerate(Qs_list):
        analytics["Clusters"][s_idx] = cid
        rows.append([ s_idx, cid ])
    print (rows)
    return (rows, Ms_list, Qs_list)
