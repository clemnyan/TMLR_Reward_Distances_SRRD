#
# Filename:     FoMDP_2.py
# Date:         2020-06-22
# Project:      Feature-organizing MDP Ver 2
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

DEBUG = True

# This version reduces the number of variables and constraints at the
#   expense of more terms in the constraints and less linearity.

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
#   features   -- "num_features"
#                   -- number of features for each state.
#               -- "s_{}_{}".format(i, k)
#                   -- This is the flag that feature k is on in state i
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
# rewards       -- "rq_{}_{}_{}".format(j, a, j')
#                   -- reward for triples based on classes
#               -- "FRVq_targ_{}_{}_{}".format(j, a, j')
#                   -- fractional reward for decision-maker
#               -- "FRVq_nontarg_{}_{}_{}".format(j, a, j')
#                   -- fractional reward for other decision-maker
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
#   "q_{}_{}".format(i, j)
#       -- binary variable
#   "Q_{}".format(i)
#       -- non-negative real-valued variable bounded by [1, 2^(num_classes-1)]
#   "DB_{}_{}".format(i, i')
#       -- binary variable
#   "Cq_targ_{}".format(j)
#       -- non-negative real-valued variable
#   "Cq_nontarg_{}".format(j)
#       -- non-negative real-valued variable
#   "Cq_targ_{}_{}".format(j, a)
#       -- non-negative real-valued variable
#   "Cq_nontarg_{}_{}".format(j, a)
#       -- non-negative real-valued variable
#   "Cq_{}_{}".format(j, a)
#       -- non-negative real-valued variable
#   "Cq_{}_{}_{}".format(j, a, j')
#       -- non-negative real-valued variable
#   "Pq_targ_{}_{}".format(a, j)
#       -- non-negative real-valued variable bounded by [0, 1]
#   "Pq_nontarg_{}_{}".format(a, j)
#       -- non-negative real-valued variable bounded by [0, 1]
#   "Pq_{}_{}_{}".format(j', a, j)
#       -- non-negative real-valued variable bounded by [0, 1]
#   "rq_{}_{}_{}".format(j, a, j')
#       -- real-valued variable bounded by [-peak, peak]
#   "FRVq_targ_{}_{}_{}".format(j, a, j')
#       -- real-valued variable bounded by [-2 * peak, 2 * peak]
#   "FRVq_nontarg_{}_{}_{}".format(j, a, j')
#       -- real-valued variable bounded by [-2 * peak, 2 * peak]
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
# s1:       "s_{}_{}".format(i, k) = 0
#   else:
# s2:       "s_{}_{}".format(i, k) - "M_{}".format(k) = 0
#               -- state i feature k
#
# q:    \sum_{j=0}^{num_classes - 1} "q_{}_{}".format(i, j) = 1
#           -- state i
#
# Q1:   "Q_{}".format(i) - \sum_{j=0}^{num_classes - 1} 2^j
#           * "q_{}_{}".format(i, j) = 0
#           -- state i
# Q2:   "Q_{}".format(i') - "Q_{}".format(i)
#           + 2^num_classes * \sum_{k=0}^{num_features-1} { 2
#               * "s_{}_{}".format(i, k) * "s_{}_{}".format(i', k)
#               - "s_{}_{}".format(i, k) - "s_{}_{}".format(i', k) }
#           >= - 2^(num_classes + 1)
# Q3:   "Q_{}".format(i) - "Q_{}".format(i')
#           + 2^num_classes * \sum_{k=0}^{num_features-1} { 2
#               * "s_{}_{}".format(i, k) * "s_{}_{}".format(i', k)
#               - "s_{}_{}".format(i, k) - "s_{}_{}".format(i', k) }
#           >= - 2^(num_classes + 1)
# Q4:   "Q_{}".format(i) - "Q_{}".format(i')
#           - (1/num_classes) * \sum_{k=0}^{num_features-1} { 2
#               * "s_{}_{}".format(i, k) * "s_{}_{}".format(i', k)
#               - "s_{}_{}".format(i, k) - "s_{}_{}".format(i', k) }
#           + 2^num_classes * "DB_{}_{}".format(i, i')
#           >= 2/num_classes
# Q5:   "Q_{}".format(i) - "Q_{}".format(i')
#           - (1/num_classes) * \sum_{k=0}^{num_features-1} { 2
#               * "s_{}_{}".format(i, k) * "s_{}_{}".format(i', k)
#               - "s_{}_{}".format(i, k) - "s_{}_{}".format(i', k) }
#           + 2^num_classes * "DB_{}_{}".format(i, i')
#           >= 2/num_classes
#           -- states i, i'
#
# Cqt1: "Cq_targ_{}".format(j) - \sum_{i=0}^{num_states - 1} C_{DM}(i)
#           * "q_{}_{}".format(i, j) = 0
#           -- class j
# Cqt2: "Cq_targ_{}_{}".format(j, a) - \sum_{i=0}^{num_states - 1} C_{DM}(i, a)
#           * "q_{}_{}".format(i, j) = 0
#           -- class j and action a
#
# Cqn1: "Cq_nontarg_{}".format(j) - \sum_{i=0}^{num_states - 1} C_{NDM}(i)
#           * "q_{}_{}".format(i, j) = 0
#           -- class j
# Cqn2: "Cq_nontarg_{}_{}".format(j, a)
#           - \sum_{i=0}^{num_states - 1} C_{NDM}(i, a)
#           * "q_{}_{}".format(i, j) = 0
#           -- class j and action a
#
# Cq1:  "Cq_{}_{}".format(j, a) - \sum{i=0}^{num_states - 1} C(i, a)
#           * "q_{}_{}".format(i, j) = 0
#           -- class j and action a
#       If j != j':
# Cq2:      "Cq_{}_{}_{}".format(j, a, j') - \sum{i=0}^{num_states - 1}
#               \sum{i'=0, i'!=i}^{num_states - 1} C(i, a, i')
#               * "q_{}_{}".format(i, j) * "q_{}_{}.format(i', j') = 0
#           -- classes j and j' and action a
#       else:
# Cq3:      "Cq_{}_{}_{}".format(j, a, j)
#               - \sum{i=0}^{num_states - 1} C(i, a, i) * "q_{}_{}".format(i, j)
#               = 0
#           -- class j and action a
#
# Pqt:  "Cq_targ_{}".format(j) * "Pq_targ_{}_{}".format(a, j)
#           - "Cq_targ_{}_{}".format(j, a) = 0
#           -- class j and action a
#
# Pqn:  "Cq_nontarg_{}".format(j) * "Pq_nontarg_{}_{}".format(a, j)
#           - "Cq_nontarg_{}_{}".format(j, a) = 0
#           -- class j and action a
#
# Pq:  "Cq_{}_{}".format(j, a) * "Pq_{}_{}_{}".format(j', a, j)
#           - "Cq_{}_{}_{}".format(j, a, j') = 0
#           -- classes j and j' and action a
#
# rqt:  "rq_{}_{}_{}".format(j, a, j') - peak * "Pq_targ_{}_{}".format(a, j)
#           - "FRVq_targ_{}_{}_{}".format(j, a, j') = 0
#
# rqn:  "rq_{}_{}_{}".format(j, a, j') + peak * "Pq_nontarg_{}_{}".format(a, j)
#           - "FRVq_nontarg_{}_{}_{}".format(j, a, j') = 0
#
# LER: "LER_{}".format(t) - \sum_{b=0}^{length(t)-1} \sum_{j=0}^{num_classes)
#           \sum_{j'=0}^{num_classes}
#               "rq_{}_{}_{}".format(j, a_{a}_<{t},{b+1}>, j')
#               * "Pq_{}_{}_{}".format(j', a_{a}_<{t},{b+1}>, j)
#               * "q_{}_{}".format(s_{i}_<{t},{b}>, j)
#               * "q_{}_{}".format(s_{i}_<{t},{b+1}>, j')
#           = 0
#       -- computes LER for trajectory a
#
#       "LB_{}".format(u) <= UB_{}".format(u)
#       "LB_{}".format(u) <= "LER_{}".format(t)
#       "UB_{}".format(u) >= "LER_{}".format(t)
#       "LB_{}".format(u) <= "LER_{}".format(t)
#       "UB_{}".format(u) >= "LER_{}".format(t)
#       -- only if trajectory t is in posets[u]
#
#       "LB_{}".format(u) = "UB_{}".format(u+1) + "delta_{}_{}".format(u, u+1)
#           -- orders posets

#
# Objective function:
#    -- Version 0
#        No objective
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

#   "Cq_targ_{}".format(j)
#       -- non-negative real-valued variable
def genCq_targ1(class_idx):
    desc = "Cq_targ1_{}".format(class_idx)
    return (genVariableName(desc, "Cqt1_"))

#   "Cq_nontarg_{}".format(j)
#       -- non-negative real-valued variable
def genCq_nontarg1(class_idx):
    desc = "Cq_nontarg1_{}".format(class_idx)
    return (genVariableName(desc, "Cqn1_"))

#   "Cq_targ_{}_{}".format(j, a)
#       -- non-negative real-valued variable
def genCq_targ2(class_idx, action_idx):
    desc = "Cq_targ_{}_{}".format(class_idx, action_idx)
    return (genVariableName(desc, "Cqt2_"))

#   "Cq_nontarg_{}_{}".format(j, a)
#       -- non-negative real-valued variable
def genCq_nontarg2(class_idx, action_idx):
    desc = "Cq_nontarg_{}_{}".format(class_idx, action_idx)
    return (genVariableName(desc, "Cqn2_"))

#   "Cq_{}_{}".format(j, a)
#       -- non-negative real-valued variable
def genCq1(class_idx, action_idx):
    desc = "Cq_{}_{}".format(class_idx, action_idx)
    return (genVariableName(desc, "Cq1_"))

#   "Cq_{}_{}_{}".format(j, a, j')
#       -- non-negative real-valued variable
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

#   "rq_{}_{}_{}".format(j, a, j')
#       -- real-valued variable bounded by [-peak, peak]
def genrq(class_idx1, action_idx, class_idx2):
    desc = "rq_{}_{}_{}".format(class_idx1, action_idx, class_idx2)
    return (genVariableName(desc, "rq"))

#   "FRVq_targ_{}_{}_{}".format(j, a, j')
#       -- real-valued variable bounded by [-2 * peak, 2 * peak]
def genFRVq_targ(class_idx1, action_idx, class_idx2):
    desc = "FRVq_targ_{}_{}_{}".format(class_idx1, action_idx, class_idx2)
    return (genVariableName(desc, "FTq"))

#   "FRVq_nontarg_{}_{}_{}".format(j, a, j')
#       -- real-valued variable bounded by [-2 * peak, 2 * peak]
def genFRVq_nontarg(class_idx1, action_idx, class_idx2):
    desc = "FRVq_nontarg_{}_{}_{}".format(class_idx1, action_idx, class_idx2)
    return (genVariableName(desc, "FNq"))

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
    return ("POSITIVE_VARIABLE")

def typeq():
    return ("BINARY_VARIABLE")

def typeQ():
    return ("POSITIVE_VARIABLE")

def typeDB():
    return ("BINARY_VARIABLE")

def typeCq_targ1():
    return ("POSITIVE_VARIABLE")

def typeCq_nontarg1():
    return ("POSITIVE_VARIABLE")

def typeCq_targ2():
    return ("POSITIVE_VARIABLE")

def typeCq_nontarg2():
    return ("POSITIVE_VARIABLE")

def typeCq1():
    return ("POSITIVE_VARIABLE")

def typeCq2():
    return ("POSITIVE_VARIABLE")

def typePq_targ():
    return ("POSITIVE_VARIABLE")

def typePq_nontarg():
    return ("POSITIVE_VARIABLE")

def typePq():
    return ("POSITIVE_VARIABLE")

def typerq():
    return ("VARIABLE")

def typeFRVq_targ():
    return ("VARIABLE")

def typeFRVq_nontarg():
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

def upperbounds(state_idx, feature_idx):
    return ("{}: 1;".format(gens(state_idx, feature_idx)))

def lowerboundQ(state_idx):
    return ("{}: 1;".format(genQ(state_idx)))

def upperboundQ(state_idx, num_classes):
    return ("{}: {};".format(genQ(state_idx), 2**(num_classes-1)))

def upperboundPq_targ(action_idx, class_idx):
    return ("{}: 1;".format(genPq_targ(action_idx, class_idx)))

def upperboundPq_nontarg(action_idx, class_idx):
    return ("{}: 1;".format(genPq_nontarg(action_idx, class_idx)))

def upperboundPq(class_idx2, action_idx, class_idx1):
    return ("{}: 1;".format(genPq(class_idx2, action_idx, class_idx1)))

def lowerboundrq(class_idx1, action_idx, class_idx2, peak):
    return ("{}: {};".format(genrq(class_idx1, action_idx, class_idx2), -peak))

def upperboundrq(class_idx1, action_idx, class_idx2, peak):
    return ("{}: {};".format(genrq(class_idx1, action_idx, class_idx2), peak))

def lowerboundFRVq_targ(class_idx1, action_idx, class_idx2, peak):
    return ("{}: {};".format(genFRVq_targ(class_idx1, action_idx, class_idx2), - 2 * peak))

def upperboundFRVq_targ(class_idx1, action_idx, class_idx2, peak):
    return ("{}: {};".format(genFRVq_targ(class_idx1, action_idx, class_idx2),  2 * peak))

def lowerboundFRVq_nontarg(class_idx1, action_idx, class_idx2, peak):
    return ("{}: {};".format(genFRVq_nontarg(class_idx1, action_idx, class_idx2), - 2 * peak))

def upperboundFRVq_nontarg(class_idx1, action_idx, class_idx2, peak):
    return ("{}: {};".format(genFRVq_nontarg(class_idx1, action_idx, class_idx2),  2 * peak))

def lowerbounddelta(poset_idx1, poset_idx2):
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
        return (name)

#   if "s_{}_{}".format(i, k) == 0: i.e., the value for
#                                       feature k for state i is 0
# s1:       "s_{}_{}".format(i, k) = 0
#   else:
# s2:       "s_{}_{}".format(i, k) - "M_{}".format(k) = 0
#               -- state i feature k
def constraints1_name(state_idx, feature_idx):
    desc = "_s1_{}_{}".format(state_idx, feature_idx)
    return (genEquationName(desc, "s1_"))

def constraints1(state_idx, feature_idx):
    return ("{} == 0;".format(gens(state_idx, feature_idx)))

def constraints2_name(state_idx, feature_idx):
    desc = "_s2_{}_{}".format(state_idx, feature_idx)
    return (genEquationName(desc, "s2_"))

def constraints2(state_idx, feature_idx):
    return ("{} - {} == 0;".format(gens(state_idx, feature_idx), genM(feature_idx)))

# q:    \sum_{j=0}^{num_classes - 1} "q_{}_{}".format(i, j) = 1
#           -- state i
def constraintq_name(state_idx):
    desc = "_q_{}".format(state_idx)
    return (genEquationName(desc, "q"))

def constraintq(state_idx, num_classes):
    s = ""
    for c_idx in range(num_classes):
        s += "{}".format(genq(state_idx, c_idx))
        if c_idx < num_classes - 1:
            s += " + "
    s += " == 1;"
    return (s)

# Q1:   "Q_{}".format(i) - \sum_{j=0}^{num_classes - 1} 2^j
#           * "q_{}_{}".format(i, j) = 0
#           -- state i
# Q2:   "Q_{}".format(i') - "Q_{}".format(i)
#           + 2^num_classes * \sum_{k=0}^{num_features-1} { 2
#               * "s_{}_{}".format(i, k) * "s_{}_{}".format(i', k)
#               - "s_{}_{}".format(i, k) - "s_{}_{}".format(i', k) }
#           >= - 2^(num_classes + 1)
# Q3:   "Q_{}".format(i) - "Q_{}".format(i')
#           + 2^num_classes * \sum_{k=0}^{num_features-1} { 2
#               * "s_{}_{}".format(i, k) * "s_{}_{}".format(i', k)
#               - "s_{}_{}".format(i, k) - "s_{}_{}".format(i', k) }
#           >= - 2^(num_classes + 1)
# Q4:   "Q_{}".format(i) - "Q_{}".format(i')
#           - (1/num_classes) * \sum_{k=0}^{num_features-1} { 2
#               * "s_{}_{}".format(i, k) * "s_{}_{}".format(i', k)
#               - "s_{}_{}".format(i, k) - "s_{}_{}".format(i', k) }
#           + 2^num_classes * "DB_{}_{}".format(i, i')
#           >= 2/num_classes
# Q5:   "Q_{}".format(i) - "Q_{}".format(i')
#           - (1/num_classes) * \sum_{k=0}^{num_features-1} { 2
#               * "s_{}_{}".format(i, k) * "s_{}_{}".format(i', k)
#               - "s_{}_{}".format(i, k) - "s_{}_{}".format(i', k) }
#           + 2^num_classes * "DB_{}_{}".format(i, i')
#           >= 2/num_classes
#           -- states i, i'
def constraintQ1_name(state_idx):
    desc = "_Q1_{}".format(state_idx)
    return (genEquationName(desc, "Q1_"))

def constraintQ1(state_idx, num_classes):
    s = "{}".format(genQ(state_idx))
    for c_idx in range(num_classes):
        s += " - {} * {}".format(2**c_idx, genq(state_idx, c_idx))
    s += " == 0;"
    return (s)

def constraintQ2_name(state_idx1, state_idx2):
    desc = "_Q2_{}_{}".format(state_idx1, state_idx2)
    return (genEquationName(desc, "Q2_"))

def constraintQ2(state_idx1, state_idx2, num_classes, num_features):
    s = "{} - {}".format(genQ(state_idx2), genQ(state_idx1))
    for f_idx in range(num_features):
        s += "+ {} * {} * {} - {} * {} - {} * {}".format(2**(num_classes + 1), gens(state_idx1, f_idx), gens(state_idx2, f_idx), 2**(num_classes), gens(state_idx1, f_idx), 2**(num_classes), gens(state_idx2, f_idx))
    s += " == 0;"
    return (s)

def constraintQ3_name(state_idx1, state_idx2):
    desc = "_Q3_{}_{}".format(state_idx1, state_idx2)
    return (genEquationName(desc, "Q3_"))

def constraintQ3(state_idx1, state_idx2, num_classes, num_features):
    s = "{} - {}".format(genQ(state_idx1), genQ(state_idx2))
    for f_idx in range(num_features):
        s += "+ {} * {} * {} - {} * {} - {} * {}".format(2**(num_classes + 1), gens(state_idx1, f_idx), gens(state_idx2, f_idx), 2**(num_classes), gens(state_idx1, f_idx), 2**(num_classes), gens(state_idx2, f_idx))
    s += " == 0;"
    return (s)

def constraintQ4_name(state_idx1, state_idx2):
    desc = "_Q4_{}_{}".format(state_idx1, state_idx2)
    return (genEquationName(desc, "Q4_"))

def constraintQ4(state_idx1, state_idx2, num_classes, num_features):
    s = "{} - {} + {}".format(genQ(state_idx1), genQ(state_idx2), genDB(state_idx1, state_idx2))
    for f_idx in range(num_features):
        s += " - {} * {} * {} + {} * {} + {} * {}".format(2.0 / float(num_classes), gens(state_idx1, f_idx), gens(state_idx2, f_idx), 1.0 / float(num_classes), gens(state_idx1, f_idx), 1.0 / float(num_classes), gens(state_idx2, f_idx))
    s += " == {};".format(2.0 / float(num_classes))
    return (s)

def constraintQ5_name(state_idx1, state_idx2):
    desc = "_Q5_{}_{}".format(state_idx1, state_idx2)
    return (genEquationName(desc, "Q5_"))

def constraintQ5(state_idx1, state_idx2, num_classes, num_features):
    s = "{} - {} + {}".format(genQ(state_idx2), genQ(state_idx1), genDB(state_idx1, state_idx2))
    for f_idx in range(num_features):
        s += " - {} * {} * {} + {} * {} + {} * {}".format(2.0 / float(num_classes), gens(state_idx1, f_idx), gens(state_idx2, f_idx), 1.0 / float(num_classes), gens(state_idx1, f_idx), 1.0 / float(num_classes), gens(state_idx2, f_idx))
    s += " == {};".format(2.0 / float(num_classes))
    return (s)

# Cqt1: "Cq_targ_{}".format(j) - \sum_{i=0}^{num_states - 1} C_{DM}(i)
#           * "q_{}_{}".format(i, j) = 0
#           -- class j
# Cqt2: "Cq_targ_{}_{}".format(j, a) - \sum_{i=0}^{num_states - 1} C_{DM}(i, a)
#           * "q_{}_{}".format(i, j) = 0
#           -- class j and action a
def constraintCqt1_name(class_idx):
    desc = "_Cq_targ_{}".format(class_idx)
    return (genEquationName(desc, "Cqt1_"))

def constraintCqt1(class_idx, num_states, C1_targ):
    s = genCq_targ1(class_idx)
    for s_idx in range(num_states):
        try:
            if C1_targ[s_idx] > 0:
                s += " - {} * {}".format(C1_targ[s_idx], genq(s_idx, class_idx))
        except KeyError:
            pass
    s += " == 0;"
    return (s)

def constraintCqt2_name(class_idx, action_idx):
    desc = "_Cq_targ_{}_{}".format(class_idx, action_idx)
    return (genEquationName(desc, "Cqt2_"))

def constraintCqt2(class_idx, action_idx, num_states, C2_targ):
    s = genCq_targ2(class_idx, action_idx)
    for s_idx in range(num_states):
        try:
            if C2_targ[(s_idx, action_idx)] > 0:
                s += " - {} * {}".format(C2_targ[(s_idx, action_idx)], genq(s_idx, class_idx))
        except KeyError:
            pass
    s += " == 0;"
    return (s)

# Cqn1: "Cq_nontarg_{}".format(j) - \sum_{i=0}^{num_states - 1} C_{NDM}(i)
#           * "q_{}_{}".format(i, j) = 0
#           -- class j
# Cqn2: "Cq_nontarg_{}_{}".format(j, a)
#           - \sum_{i=0}^{num_states - 1} C_{NDM}(i, a)
#           * "q_{}_{}".format(i, j) = 0
#           -- class j and action a
def constraintCqn1_name(class_idx):
    desc = "_Cq_nontarg_{}".format(class_idx)
    return (genEquationName(desc, "Cqn1_"))

def constraintCqn1(class_idx, num_states, C1_nontarg):
    s = genCq_nontarg1(class_idx)
    for s_idx in range(num_states):
        try:
            if C1_nontarg[s_idx] > 0:
                s += " - {} * {}".format(C1_nontarg[s_idx], genq(s_idx, class_idx))
        except KeyError:
            pass
    s += " == 0;"
    return (s)

def constraintCqn2_name(class_idx, action_idx):
    desc = "_Cq_nontarg_{}_{}".format(class_idx, action_idx)
    return (genEquationName(desc, "Cqn2_"))

def constraintCqn2(class_idx, action_idx, num_states, C2_nontarg):
    s = genCq_nontarg2(class_idx, action_idx)
    for s_idx in range(num_states):
        try:
            if C2_nontarg[(s_idx, action_idx)] > 0:
                s += " - {} * {}".format(C2_nontarg[(s_idx, action_idx)], genq(s_idx, class_idx))
        except KeyError:
            pass
    s += " == 0;"
    return (s)

# Cq1:  "Cq_{}_{}".format(j, a) - \sum{i=0}^{num_states - 1} C(i, a)
#           * "q_{}_{}".format(i, j) = 0
#           -- class j and action a
#       If j != j':
# Cq2:      "Cq_{}_{}_{}".format(j, a, j') - \sum{i=0}^{num_states - 1}
#               \sum{i'=0, i'!=i}^{num_states - 1} C(i, a, i')
#               * "q_{}_{}".format(i, j) * "q_{}_{}.format(i', j') = 0
#           -- classes j and j' and action a
#       else:
# Cq3:      "Cq_{}_{}_{}".format(j, a, j)
#               - \sum{i=0}^{num_states - 1} C(i, a, i) * "q_{}_{}".format(i, j)
#               = 0
#           -- class j and action a
def constraintCq1_name(class_idx, action_idx):
    desc = "_Cq_{}_{}".format(class_idx, action_idx)
    return (genEquationName(desc, "Cq1_"))

def constraintCq1(class_idx, action_idx, num_states, C2):
    s = genCq1(class_idx, action_idx)
    for s_idx in range(num_states):
        try:
            if C2[(s_idx, action_idx)] > 0:
                s += " - {} * {}".format(C2[(s_idx, action_idx)], genq(s_idx, class_idx))
        except KeyError:
            pass
    s += " == 0;"
    return (s)

def constraintCq2_name(class_idx1, action_idx, class_idx2):
    if class_idx1 == class_idx2:
        sys.exit("Constraint Cq2 must have different class indices!")
    desc = "_Cq_{}_{}_{}".format(class_idx1, action_idx, class_idx2)
    return (genEquationName(desc, "Cq2_"))

def constraintCq2(class_idx1, action_idx, class_idx2, num_states, C3, trajectories, triples):
    s = genCq2(class_idx1, action_idx, class_idx2)
    pairs = set()
    for (s_idx1, a_idx, s_idx2) in triples:
        if a_idx != action_idx:
            continue
        if s_idx1 == s_idx2:
            continue
        if ( s_idx1, s_idx2 ) in pairs:
            continue
        s += " - {} * {}".format(C3[(s_idx1, action_idx, s_idx2)], genq(s_idx1, class_idx1), genq(s_idx2, class_idx2))
        pairs.add(( s_idx1, s_idx2 ))
    s += " == 0;"
    del pairs
    return (s)

def constraintCq3_name(class_idx, action_idx):
    desc = "_Cq_{}_{}_{}".format(class_idx, action_idx, class_idx)
    return (genEquationName(desc, "Cq3_"))

def constraintCq3(class_idx, action_idx, num_states, C3):
    s = genCq2(class_idx, action_idx, class_idx)
    for s_idx in range(num_states):
        try:
            s += " - {} * {}".format(C3[(s_idx, action_idx, s_idx)], genq(s_idx, class_idx))
        except KeyError:
            pass
    s += " == 0;"
    return (s)

# Pqt:  "Cq_targ_{}".format(j) * "Pq_targ_{}_{}".format(a, j)
#           - "Cq_targ_{}_{}".format(j, a) = 0
#           -- class j and action a
def constraintPqt_name(action_idx, class_idx):
    desc = "_Pq_targ_{}_{}".format(action_idx, class_idx)
    return (genEquationName(desc, "Pqt"))

def constraintPqt(action_idx, class_idx):
    return ("{} * {} - {} == 0;".format(genCq_targ1(class_idx), genPq_targ(action_idx, class_idx), genCq_targ2(class_idx, action_idx)))

# Pqn:  "Cq_nontarg_{}".format(j) * "Pq_nontarg_{}_{}".format(a, j)
#           - "Cq_nontarg_{}_{}".format(j, a) = 0
#           -- class j and action a
def constraintPqn_name(action_idx, class_idx):
    desc = "_Pq_nontarg_{}_{}".format(action_idx, class_idx)
    return (genEquationName(desc, "Pqn"))

def constraintPqn(action_idx, class_idx):
    return ("{} * {} - {} == 0;".format(genCq_nontarg1(class_idx), genPq_nontarg(action_idx, class_idx), genCq_nontarg2(class_idx, action_idx)))

# Pq:  "Cq_{}_{}".format(j, a) * "Pq_{}_{}_{}".format(j', a, j)
#           - "Cq_{}_{}_{}".format(j, a, j') = 0
#           -- classes j and j' and action a
def constraintPq_name(class_idx2, action_idx, class_idx1):
    desc = "_Pq_{}_{}_{}".format(class_idx2, action_idx, class_idx1)
    return (genEquationName(desc, "Pq"))

def constraintPq(class_idx2, action_idx, class_idx1):
    return ("{} * {} - {} == 0;".format(genCq1(class_idx1, action_idx), genPq(class_idx2, action_idx, class_idx1), genCq2(class_idx1, action_idx, class_idx2)))

# rqt:  "rq_{}_{}_{}".format(j, a, j') - peak * "Pq_targ_{}_{}".format(a, j)
#           - "FRVq_targ_{}_{}_{}".format(j, a, j') = 0
def constraintrqt_name(class_idx1, action_idx, class_idx2):
    desc = "_rqt_{}_{}_{}".format(class_idx1, action_idx, class_idx2)
    return (genEquationName(desc, "rqt"))

def constraintrqt(class_idx1, action_idx, class_idx2, peak):
    return ("{} - {} * {} - {} == 0;".format(genrq(class_idx1, action_idx, class_idx2), peak, genPq_targ(action_idx, class_idx1), genFRVq_targ(class_idx1, action_idx, class_idx2)))

# rqn:  "rq_{}_{}_{}".format(j, a, j') + peak * "Pq_nontarg_{}_{}".format(a, j)
#           - "FRVq_nontarg_{}_{}_{}".format(j, a, j') = 0
def constraintrqn_name(class_idx1, action_idx, class_idx2):
    desc = "_rqn_{}_{}_{}".format(class_idx1, action_idx, class_idx2)
    return (genEquationName(desc, "rqn"))

def constraintrqn(class_idx1, action_idx, class_idx2, peak):
    return ("{} + {} * {} - {} == 0;".format(genrq(class_idx1, action_idx, class_idx2), peak, genPq_nontarg(action_idx, class_idx1), genFRVq_nontarg(class_idx1, action_idx, class_idx2)))

# LER: "LER_{}".format(t) - \sum_{b=0}^{length(t)-1} \sum_{j=0}^{num_classes)
#           \sum_{j'=0}^{num_classes}
#               "rq_{}_{}_{}".format(j, a_{a}_<{t},{b+1}>, j')
#               * "Pq_{}_{}_{}".format(j', a_{a}_<{t},{b+1}>, j)
#               * "q_{}_{}".format(s_{i}_<{t},{b}>, j)
#               * "q_{}_{}".format(s_{i}_<{t},{b+1}>, j')
#           = 0
#       -- computes LER for trajectory a
def constraintLER_name(traj_idx):
    desc = "_LER_{}".format(traj_idx)
    return (genEquationName(desc, "LER_"))

def constraintLER(traj_idx, trajectory, num_classes):
    constraint = genLER(traj_idx)
    triples = dict()
    for state_pos in range(floor(len(trajectory) / 2)):
        s_idx1 = trajectory[state_pos * 2]
        a_idx = trajectory[state_pos * 2 + 1]
        s_idx2 = trajectory[(state_pos + 1) * 2]
        try:
            triples[(s_idx1, a_idx, s_idx2)] += 1
        except KeyError:
            triples[(s_idx1, a_idx, s_idx2)] = 1
    for (s_idx1, a_idx, s_idx2), count in triples.items():
        for c_idx1 in range(num_classes):
            for c_idx2 in range(num_classes):
                constraint += " - {} * {} * {} * {} * {}".format(count, genrq(c_idx1, a_idx, c_idx2), genPq(c_idx2, a_idx, c_idx1), genq(s_idx1, c_idx1), genq(s_idx2, c_idx2))
    constraint += " == 0;"
    return (constraint)

#   "LB_{}".format(u) = "UB_{}".format(u+1) + "delta_{}_{}".format(u, u+1)
#       -- orders posets
def constraintPoset_name(poset_idx1, poset_idx2):
    desc = "_POSETS_{}_{}".format(poset_idx1, poset_idx2)
    return (genEquationName(desc, "POSETS_"))

def constraintPoset(poset_idx1, poset_idx2):
    return ("{} - {} - {} == 0;".format(genLB(poset_idx1), genUB(poset_idx2), gendelta(poset_idx1, poset_idx2)))

#   "LB_{}".format(u) <= UB_{}".format(u)
def constraintPosetLBUB_name(poset_idx):
    desc = "_POSETLBUB_{}".format(poset_idx)
    return (genEquationName(desc, "POSET"))

def constraintPosetLBUB(poset_idx):
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
    return ("{} - {} <= 0;".format(genLB(poset_idx), genLER(traj_idx)))

def constraintPosetUB_name(poset_idx, traj_idx):
    desc = "_POSETUB_{}_{}".format(poset_idx, traj_idx)
    return (genEquationName(desc, "POSETUB_"))

def constraintPosetUB(poset_idx, traj_idx):
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

def createDeclarations(type, variables):
    vars = list(variables)
    vars.sort()
    row = "{} {}".format(type, vars[0])
    for idx in range(1, len(vars)):
        row += ",\n\t{}".format(vars[idx])
    row += ";"
    del vars
    return (row)

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

    D_pairs = set()
    if not simplify_masking:
        if DEBUG:
            print ('...Building D_pairs...')
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

    #   "Cq_targ_{}".format(j)
    #       -- non-negative real-valued variable
    #   "Cq_nontarg_{}".format(j)
    #       -- non-negative real-valued variable
    if DEBUG:
        print ('...Cq_targ1/Cq_nontarg1 variables...')
        var_ct = len(var_to_var_desc)
    for c_idx in range(num_classes):
        addVariable(genCq_targ1(c_idx), typeCq_targ1(), binary_variables, integer_variables, positive_variables, variables)
        addVariable(genCq_nontarg1(c_idx), typeCq_nontarg1(), binary_variables, integer_variables, positive_variables, variables)
    if DEBUG:
        print ('......{} declared.'.format(len(var_to_var_desc) - var_ct))

    #   "Cq_targ_{}_{}".format(j, a)
    #       -- non-negative real-valued variable
    #   "Cq_nontarg_{}_{}".format(j, a)
    #       -- non-negative real-valued variable
    #   "Cq_{}_{}".format(j, a)
    #       -- non-negative real-valued variable
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
    #       -- non-negative real-valued variable
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

    #   "rq_{}_{}_{}".format(j, a, j')
    #       -- real-valued variable bounded by [-peak, peak]
    #   "FRVq_targ_{}_{}_{}".format(j, a, j')
    #       -- real-valued variable bounded by [-2 * peak, 2 * peak]
    #   "FRVq_nontarg_{}_{}_{}".format(j, a, j')
    #       -- real-valued variable bounded by [-2 * peak, 2 * peak]
    if DEBUG:
        print ('...rq/FRVq_targ/FRVq_nontarg variables...')
        var_ct = len(var_to_var_desc)
    for c_idx1 in range(num_classes):
        for a_idx in range(num_actions):
            for c_idx2 in range(num_classes):
                addVariable(genrq(c_idx1, a_idx, c_idx2), typerq(), binary_variables, integer_variables, positive_variables, variables)
                addVariable(genFRVq_targ(c_idx1, a_idx, c_idx2), typeFRVq_targ(), binary_variables, integer_variables, positive_variables, variables)
                addVariable(genFRVq_nontarg(c_idx1, a_idx, c_idx2), typeFRVq_nontarg(), binary_variables, integer_variables, positive_variables, variables)
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
        LER_to_trajectory[var_name] = tuple(traj)
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
    # Returns a list of rows detailing the bounds
    rows = list()
    rows.append("LOWER_BOUNDS{")

    #   "s_{}_{}".format(i, k)
    #       -- non-negative real-valued variable bounded by [0, 1]
    #   "q_{}_{}".format(i, j)
    #       -- binary variable
    #   "Q_{}".format(i)
    #       -- non-negative real-valued variable bounded by [1, 2^(num_classes-1)]
    #   "FQ_{}_{}_{}_{}".format(j, i, j', i')
    #       -- non-negative real-valued variable bounded by [0, 1]
    if not simplify_masking:
        for s_idx in range(num_states):
            rows.append(lowerboundQ(s_idx))

    #   "Pq_targ_{}_{}".format(a, j)
    #       -- non-negative real-valued variable bounded by [0, 1]
    #   "Pq_nontarg_{}_{}".format(a, j)
    #       -- non-negative real-valued variable bounded by [0, 1]
    #   "Pq_{}_{}_{}".format(j', a, j)
    #       -- non-negative real-valued variable bounded by [0, 1]

    #   "rq_{}_{}_{}".format(j, a, j')
    #       -- real-valued variable bounded by [-peak, peak]
    #   "FRVq_targ_{}_{}_{}".format(j, a, j')
    #       -- real-valued variable bounded by [-2 * peak, 2 * peak]
    #   "FRVq_nontarg_{}_{}_{}".format(j, a, j')
    #       -- real-valued variable bounded by [-2 * peak, 2 * peak]
    for c_idx1 in range(num_classes):
        for a_idx in range(num_actions):
            for c_idx2 in range(num_classes):
                rows.append(lowerboundrq(c_idx1, a_idx, c_idx2, peak))
                rows.append(lowerboundFRVq_targ(c_idx1, a_idx, c_idx2, peak))
                rows.append(lowerboundFRVq_nontarg(c_idx1, a_idx, c_idx2, peak))

    #   "delta_{}_{}".format(u, u+1)
    #       -- real valued variable corresponding to the distance between
    #           two posets u and u+1 (assumed total ordering)
    #       -- bounded by [1, inf)
    for p_idx in range(num_posets - 1):
        rows.append(lowerbounddelta(p_idx, p_idx + 1))

    rows.append("}")
    rows.append(" ")

    rows.append("UPPER_BOUNDS{")

    for s_idx in range(num_states):
        if not simplify_masking:
            rows.append(upperboundQ(s_idx, num_classes))
        if not simplify_masking:
            for f_idx1 in range(num_features):
                rows.append(upperbounds(s_idx, f_idx1))

    for a_idx in range(num_actions):
        for c_idx1 in range(num_classes):
            rows.append(upperboundPq_targ(a_idx, c_idx1))
            rows.append(upperboundPq_nontarg(a_idx, c_idx1))
            for c_idx2 in range(num_classes):
                rows.append(upperboundPq(c_idx2, a_idx, c_idx1))

    for c_idx1 in range(num_classes):
        for a_idx in range(num_actions):
            for c_idx2 in range(num_classes):
                rows.append(upperboundrq(c_idx1, a_idx, c_idx2, peak))
                rows.append(upperboundFRVq_targ(c_idx1, a_idx, c_idx2, peak))
                rows.append(upperboundFRVq_nontarg(c_idx1, a_idx, c_idx2, peak))

    rows.append("}")
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
        # s1:       "s_{}_{}".format(i, k) = 0
        #   else:
        # s2:       "s_{}_{}".format(i, k) - "M_{}".format(k) = 0
        #               -- state i feature k
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

    if DEBUG:
        print ("...q constraints...")
    # q:    \sum_{j=0}^{num_classes - 1} "q_{}_{}".format(i, j) = 1
    #           -- state i
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
        # Q1:   "Q_{}".format(i) - \sum_{j=0}^{num_classes - 1} 2^j
        #           * "q_{}_{}".format(i, j) = 0
        #           -- state i
        # Q2:   "Q_{}".format(i') - "Q_{}".format(i)
        #           + 2^num_classes * \sum_{k=0}^{num_features-1} { 2
        #               * "s_{}_{}".format(i, k) * "s_{}_{}".format(i', k)
        #               - "s_{}_{}".format(i, k) - "s_{}_{}".format(i', k) }
        #           >= - 2^(num_classes + 1)
        # Q3:   "Q_{}".format(i) - "Q_{}".format(i')
        #           + 2^num_classes * \sum_{k=0}^{num_features-1} { 2
        #               * "s_{}_{}".format(i, k) * "s_{}_{}".format(i', k)
        #               - "s_{}_{}".format(i, k) - "s_{}_{}".format(i', k) }
        #           >= - 2^(num_classes + 1)
        # Q4:   "Q_{}".format(i) - "Q_{}".format(i')
        #           - (1/num_classes) * \sum_{k=0}^{num_features-1} { 2
        #               * "s_{}_{}".format(i, k) * "s_{}_{}".format(i', k)
        #               - "s_{}_{}".format(i, k) - "s_{}_{}".format(i', k) }
        #           + 2^num_classes * "DB_{}_{}".format(i, i')
        #           >= 2/num_classes
        # Q5:   "Q_{}".format(i) - "Q_{}".format(i')
        #           - (1/num_classes) * \sum_{k=0}^{num_features-1} { 2
        #               * "s_{}_{}".format(i, k) * "s_{}_{}".format(i', k)
        #               - "s_{}_{}".format(i, k) - "s_{}_{}".format(i', k) }
        #           + 2^num_classes * "DB_{}_{}".format(i, i')
        #           >= 2/num_classes
        #           -- states i, i'
        num_constraints = 0
        for s_idx in range(num_states):
            eqn = constraintQ1_name(s_idx)
            if not eqn in equation_names_set: # skip
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintQ1(s_idx, num_classes))
                num_constraints += 1

        for (s_idx1, s_idx2) in tqdm.tqdm(D_pairs, total=len(D_pairs), desc="D_pairs"):
            eqn = constraintQ2_name(s_idx1, s_idx2)
            if not eqn in equation_names_set: # skip
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintQ2(s_idx1, s_idx2, num_classes, num_features))
                num_constraints += 1
            eqn = constraintQ3_name(s_idx1, s_idx2)
            if not eqn in equation_names_set: # skip
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintQ3(s_idx1, s_idx2, num_classes, num_features))
                num_constraints += 1
            eqn = constraintQ4_name(s_idx1, s_idx2)
            if not eqn in equation_names_set: # skip
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintQ4(s_idx1, s_idx2, num_classes, num_features))
                num_constraints += 1
            eqn = constraintQ5_name(s_idx1, s_idx2)
            if not eqn in equation_names_set: # skip
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintQ5(s_idx1, s_idx2, num_classes, num_features))
                num_constraints += 1
        if DEBUG:
            print ("......{} created.".format(num_constraints))

    if DEBUG:
        print ("...Cq_targ constraints...")
    # Cqt1: "Cq_targ_{}".format(j) - \sum_{i=0}^{num_states - 1} C_{DM}(i)
    #           * "q_{}_{}".format(i, j) = 0
    #           -- class j
    # Cqt2: "Cq_targ_{}_{}".format(j, a) - \sum_{i=0}^{num_states - 1} C_{DM}(i, a)
    #           * "q_{}_{}".format(i, j) = 0
    #           -- class j and action a
    num_constraints = 0
    for c_idx in range(num_classes):
        eqn = constraintCqt1_name(c_idx)
        if not eqn in equation_names_set: # skip
            equation_names_set.add(eqn)
            equation_names.append(eqn)
            equations.append(constraintCqt1(c_idx, num_states, C1_targ))
            num_constraints += 1
        for a_idx in range(num_actions):
            eqn = constraintCqt2_name(c_idx, a_idx)
            if not eqn in equation_names_set: # skip
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintCqt2(c_idx, a_idx, num_states, C2_targ))
                num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    if DEBUG:
        print ("...Cq_nontarg constraints...")
    # Cqn1: "Cq_nontarg_{}".format(j) - \sum_{i=0}^{num_states - 1} C_{NDM}(i)
    #           * "q_{}_{}".format(i, j) = 0
    #           -- class j
    # Cqn2: "Cq_nontarg_{}_{}".format(j, a)
    #           - \sum_{i=0}^{num_states - 1} C_{NDM}(i, a)
    #           * "q_{}_{}".format(i, j) = 0
    #           -- class j and action a
    num_constraints = 0
    for c_idx in range(num_classes):
        eqn = constraintCqn1_name(c_idx)
        if not eqn in equation_names_set: # skip
            equation_names_set.add(eqn)
            equation_names.append(eqn)
            equations.append(constraintCqn1(c_idx, num_states, C1_nontarg))
            num_constraints += 1
        for a_idx in range(num_actions):
            eqn = constraintCqn2_name(c_idx, a_idx)
            if not eqn in equation_names_set: # skip
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintCqn2(c_idx, a_idx, num_states, C2_nontarg))
                num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    if DEBUG:
        print ("...Cq constraints...")
    # Cq1:  "Cq_{}_{}".format(j, a) - \sum{i=0}^{num_states - 1} C(i, a)
    #           * "q_{}_{}".format(i, j) = 0
    #           -- class j and action a
    #       If j != j':
    # Cq2:      "Cq_{}_{}_{}".format(j, a, j') - \sum{i=0}^{num_states - 1}
    #               \sum{i'=0, i'!=i}^{num_states - 1} C(i, a, i')
    #               * "q_{}_{}".format(i, j) * "q_{}_{}.format(i', j') = 0
    #           -- classes j and j' and action a
    #       else:
    # Cq3:      "Cq_{}_{}_{}".format(j, a, j)
    #               - \sum{i=0}^{num_states - 1} C(i, a, i) * "q_{}_{}".format(i, j)
    #               = 0
    #           -- class j and action a
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
                if c_idx1 != c_idx2:
                    eqn = constraintCq2_name(c_idx1, a_idx, c_idx2)
                    if not eqn in equation_names_set: # skip
                        equation_names_set.add(eqn)
                        equation_names.append(eqn)
                        equations.append(constraintCq2(c_idx1, a_idx, c_idx2, num_states, C3, trajectories, triples))
                        num_constraints += 1
                else:
                    eqn = constraintCq3_name(c_idx1, a_idx)
                    if not eqn in equation_names_set: # skip
                        equation_names_set.add(eqn)
                        equation_names.append(eqn)
                        equations.append(constraintCq3(c_idx1, a_idx, num_states, C3))
                        num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    if DEBUG:
        print ("...Pq_targ constraints...")
    # Pqt:  "Cq_targ_{}".format(j) * "Pq_targ_{}_{}".format(a, j)
    #           - "Cq_targ_{}_{}".format(j, a) = 0
    #           -- class j and action a
    num_constraints = 0
    for c_idx in range(num_classes):
        for a_idx in range(num_actions):
            eqn = constraintPqt_name(a_idx, c_idx)
            if not eqn in equation_names_set: # skip
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintPqt(a_idx, c_idx))
                num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    if DEBUG:
        print ("...Pq_nontarg constraints...")
    # Pqn:  "Cq_nontarg_{}".format(j) * "Pq_nontarg_{}_{}".format(a, j)
    #           - "Cq_nontarg_{}_{}".format(j, a) = 0
    #           -- class j and action a
    num_constraints = 0
    for c_idx in range(num_classes):
        for a_idx in range(num_actions):
            eqn = constraintPqn_name(a_idx, c_idx)
            if not eqn in equation_names_set: # skip
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintPqn(a_idx, c_idx))
                num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    if DEBUG:
        print ("...Pq constraints...")
    # Pq:  "Cq_{}_{}".format(j, a) * "Pq_{}_{}_{}".format(j', a, j)
    #           - "Cq_{}_{}_{}".format(j, a, j') = 0
    #           -- classes j and j' and action a
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
        print ("...FRVq_targ constraints...")
    # rqt:  "rq_{}_{}_{}".format(j, a, j') - peak * "Pq_targ_{}_{}".format(a, j)
    #           - "FRVq_targ_{}_{}_{}".format(j, a, j') = 0
    num_constraints = 0
    for c_idx1 in range(num_classes):
        for a_idx in range(num_actions):
            for c_idx2 in range(num_classes):
                eqn = constraintrqt_name(c_idx1, a_idx, c_idx2)
                if not eqn in equation_names_set:
                    equation_names_set.add(eqn)
                    equation_names.append(eqn)
                    equations.append(constraintrqt(c_idx1, a_idx, c_idx2, peak))
                    num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    if DEBUG:
        print ("...FRVq_nontarg constraints...")
    # rqn:  "rq_{}_{}_{}".format(j, a, j') + peak * "Pq_nontarg_{}_{}".format(a, j)
    #           - "FRVq_nontarg_{}_{}_{}".format(j, a, j') = 0
    num_constraints = 0
    for c_idx1 in range(num_classes):
        for a_idx in range(num_actions):
            for c_idx2 in range(num_classes):
                eqn = constraintrqn_name(c_idx1, a_idx, c_idx2)
                if not eqn in equation_names_set:
                    equation_names_set.add(eqn)
                    equation_names.append(eqn)
                    equations.append(constraintrqn(c_idx1, a_idx, c_idx2, peak))
                    num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    if DEBUG:
        print ("...LER constraints...")
    # LER: "LER_{}".format(t) - \sum_{b=0}^{length(t)-1} \sum_{j=0}^{num_classes)
    #           \sum_{j'=0}^{num_classes}
    #               "rq_{}_{}_{}".format(j, a_{a}_<{t},{b+1}>, j')
    #               * "Pq_{}_{}_{}".format(j', a_{a}_<{t},{b+1}>, j)
    #               * "q_{}_{}".format(s_{i}_<{t},{b}>, j)
    #               * "q_{}_{}".format(s_{i}_<{t},{b+1}>, j')
    #           = 0
    #       -- computes LER for trajectory a
    num_constraints = 0
    for t_idx in range(len(trajectories)):
        eqn = constraintLER_name(t_idx)
        if not eqn in equation_names_set:
            equation_names_set.add(eqn)
            equation_names.append(eqn)
            equations.append(constraintLER(t_idx, trajectories[t_idx], num_classes))
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
    t = kwargs['all_trajectories']
    triples = kwargs['all_triples']
    dm_triples = kwargs['dm_triples']
    others_triples = kwargs['others_triples']
    optimizer = kwargs['optimizer']
    maxtime = kwargs['maxtime']
    license_fn = kwargs['license_fn']
    simplify_masking = kwargs['simplify_masking']
#   maxtime is maximum time allowed for BARON
#   license_fn is the full path filename to the BARON license

    if DEBUG:
        print ("Building FoMDP_2 optimization problem...")

    if not optimizer == "BARON":
        sys.exit(".constructOptimization(...) -- can only build for BARON.")
    rows = [ "OPTIONS {", "MaxTime:{};".format(maxtime), "LicName: \"{}\";".format(license_fn), "nlpsol: 10;", "}", " " ]


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
    rows.append(" ")
    rows.append("OBJ: minimize")
    rows.append("\t{};".format(1))
#    rows.append("\t{};".format(genT()))
    if DEBUG:
        print ("\t...elapsed time = {}".format(time.time() - start_time))
    return ( rows, LER_to_trajectories, None, None, None, var_to_var_desc )
