#
# Filename:     SsMDP.py
# Date:         2020-03-31
# Project:      State-splitting MDP
# Author:       Eugene Santos Jr.
# Copyright     Eugene Santos Jr.
#
#
import sys
from math import floor
import os
import csv
import copy
import time
#import gurobipy
#from gurobipy import GRB

DEBUG = True
OPTIMIZER = None # BARON or GUROBI
GUROBI_MODEL = None
GUROBI_VARIABLES = None
GUROBI_CONSTRAINT_NAME = None
GUROBI_IntFeasTol = 1e-9 # Integrality feasibility tolerance
GUROBI_INTEGRALS = set() # Set of variables for GUROBI that are INTEGER or BINARY

#
# Notation:
#
#   states      -- "s_{}".format(i)
#                   -- this is the ith state
#                   -- states index from 0
#               -- "s_{}_<{}>".format(i, k)
#                   -- ith state with kth hidden value
#               -- "s_{}_<{}>--({},{})".format(i, k, a, b)
#                   -- the ath trajectory's bth state which is state i mapped to kth hidden value
#               -- "C_[s_{}_<{}>,a_{}]".format(i, k, j)
#                   -- Number of observed state i with hidden value k and action j pairs
#               -- "C__[s_{}_<{}>,a_{},s_{}_<{}>]".format(i, k, j, i', k')
#                   -- Number of observed transitions from the state i with hidden value k through action j to state i' with hidden value k"
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
#   "T"
#       -- non-negative integer variable
#   "s_{}_<{}>--({},{})".format(i, k, a, b)
#       -- binary variable (0 = False, 1 = True)
#   "F[s_{}_<{}>--({},{}),s_{}_<{}>--({},{})]".format(i, k, a, b, i', k', a', b')
#       -- binary variable (0 = False, 1 = True)
#       --  is True if and only if both binary variables are True
#   "C_[s_{}_<{}>,a_{}]".format(i, k, j)
#       -- real valued non-negative variable
#   "C__[s_{}_<{}>,a_{},s_{}_<{}>]".format(i, k, j, i', k')
#       -- real valued non-negative variable
#   "P_[s_{}_<{}> | a_{},s_{}_<{}>]".format(i, k, j, i', k')
#       -- real valued variable that is the conditional probability of transition between [0, 1]
#   "R_[s_{}_<{}>,a_{},s_{}_<{}>]".format(i, k, j, i', k')
#       -- real valued variable
#       -- bounded by [-peak, peak]
#   "RT_[s_{}_<{}>--({},{}),a_{},s_{}_<{}>--({},{})]".format(i, k, a, b, j, i', k', a, b+1)
#       -- real valued variable corresponding to the mapping of the ath trajectories rewards used in expected linear reward computation
#       -- this is created for each trajectory a and each triple in the trajectory
#       -- bounded by [-peak, peak]
#   "LER_{}".format(a)
#       -- real valued variable corresponding to the linear expected reward for trajectory a (without discounting)
#       -- unbounded
#   "LB_{}".format(q)
#   "UB_{}".format(q)
#       -- real valued variable corresponding to the lower and upper bounds of LER for trajectories in poset q
#   "delta_{}_{}".format(q, q+1)
#       -- real valued variable corresponding to the distance between two posets q and q+1 (assumed total ordering)
#       -- bounded by [1, inf)

#
# Constraints: (unbound indices are assumed universal)
#
#   k * "s_{}_<{}>--({},{})".format(i, k, a, b) <= T - 1
#       -- lower bound of used hidden values
#
#   \sum_{k=0}^{tau-1} "s_{}_<{}>--({},{})".format(i, k, a, b) = 1
#       -- constraint that enforces use of exactly one hidden state for ath trajectory bth state which is state i
#
#   "F[s_{}_<{}>--({},{}),s_{}_<{}>--({},{})]".format(i, k, a, b, i', k', a', b') <= "s_{}_<{}>--({},{})".format(i, k, a, b)
#   "F[s_{}_<{}>--({},{}),s_{}_<{}>--({},{})]".format(i, k, a, b, i', k', a', b') <= "s_{}_<{}>--({},{})".format(i', k', a', b')
#   "s_{}_<{}>--({},{})".format(i, k, a, b) + "s_{}_<{}>--({},{})".format(i', k', a', b') - 1 <= F[s_{}_<{}>--({},{}),s_{}_<{}>--({},{})]".format(i, k, a, b, i', k', a', b')
#       -- constraints for managing F[s_{}_<{}>--({},{}),s_{}_<{}>--({},{})]".format(i, k, a, b, i', k', a', b')
#
#   "C_[s_{}_<{}>,a_{}]".format(i, k, j) = \sum_{a,b:"a_{}--({},{})".format(j, a, b) occurs} "s_{}_<{}>--({},{})".format(i, k, a, b)
#       -- counts based on assigned hidden values to trajectory states
#
#   "C__[s_{}_<{}>,a_{},s_{}_<{}>]".format(i, k, j, i', k') = \sum_{a,b:"a_{}--({},{})".format(j, a, b) occur in some triple} "F[s_{}_<{}>--({},{}),s_{}_<{}>--({},{})]".format(i, k, a, b, i', k', a, b+1)
#       -- counts based on occurring triples in the trajectories
#
#   "P_[s_{}_<{}> | a_{},s_{}_<{}>]".format(i, k, j, i', k') = "C__[s_{}_<{}>,a_{},s_{}_<{}>]".format(i, k, j, i', k') / "C_[s_{}_<{}>,a_{}]".format(i, k, j)
#       -- ratio for conditional probability (rewrite to eliminate division)
#
#   "RT_[s_{}_<{}>--({},{}),a_{},s_{}_<{}>--({},{})]".format(i, k, a, b, j, i', k', a, b+1) <= peak * "F[s_{}_<{}>--({},{}),s_{}_<{}>--({},{})]".format(i, k, a, b, i', k', a, b+1)
#   "RT_[s_{}_<{}>--({},{}),a_{},s_{}_<{}>--({},{})]".format(i, k, a, b, j, i', k', a, b+1) >= -peak * "F[s_{}_<{}>--({},{}),s_{}_<{}>--({},{})]".format(i, k, a, b, i', k', a, b+1)
#       -- forces to 0 if not activated based on hidden value assignment
#
#   "RT_[s_{}_<{}>--({},{}),a_{},s_{}_<{}>--({},{})]".format(i, k, a, b, j, i', k', a, b+1) <= "R_[s_{}_<{}>,a_{},s_{}_<{}>]".format(i, k, j, i', k') + (1 - "F[s_{}_<{}>--({},{}),s_{}_<{}>--({},{})]".format(i, k, a, b, i', k', a, b+1)) * peak
#   "RT_[s_{}_<{}>--({},{}),a_{},s_{}_<{}>--({},{})]".format(i, k, a, b, j, i', k', a, b+1) >= "R_[s_{}_<{}>,a_{},s_{}_<{}>]".format(i, k, j, i', k') - (1 - "F[s_{}_<{}>--({},{}),s_{}_<{}>--({},{})]".format(i, k, a, b, i', k', a, b+1)) * peak
#       -- forces to the appropriate reward value based on hidden value assignment
#
#   "LER_{}".format(a) = \sum_{b=0}^{length(a)-1} \sum_{k=0}^{tau-1} \sum_{k'=0}^{tau-1} "P_[s_{}_<{}> | a_{},s_{}_<{}>]".format(i', k', j, i, k) * "RT_[s_{}_<{}>--({},{}),a_{},s_{}_<{}>--({},{})]".format(i, k, a, b, j, i', k', a, b+1)
#       -- computes LER for trajectory a
#
#   "LB_{}".format(u) <= UB_{}".format(u)
#
#   "LB_{}".format(q) = "UB_{}".format(q+1) + "delta_{}_{}".format(q, q+1)
#       -- orders posets
#
#   "LB_{}".format(q) <= "LER_{}".format(a)
#   "UB_{}".format(q) >= "LER_{}".format(a)
#       -- only if trajectory a is in posets[q]

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

def genT():
    desc = "T"
    return (genVariableName(desc, "T"))

def genHiddenState(state_idx, hidden_idx, traj_idx, state_pos):
    desc = "s_{}_<{}>--({},{})".format(state_idx, hidden_idx, traj_idx, state_pos)
    return (genVariableName(desc, "s"))

def genF(state_idx1, hidden_idx1, traj_idx1, state_pos1, state_idx2, hidden_idx2, traj_idx2, state_pos2):
    desc = "F[s_{}_<{}>--({},{}),s_{}_<{}>--({},{})]".format(state_idx1, hidden_idx1, traj_idx1, state_pos1, state_idx2, hidden_idx2, traj_idx2, state_pos2)
    return (genVariableName(desc, "F"))

def genC(state_idx, hidden_idx, action_idx):
    desc = "C_[s_{}_<{}>,a_{}]".format(state_idx, hidden_idx, action_idx)
    return (genVariableName(desc, "C_"))

def genC2(state_idx1, hidden_idx1, action_idx, state_idx2, hidden_idx2):
    desc = "C__[s_{}_<{}>,a_{},s_{}_<{}>]".format(state_idx1, hidden_idx1, action_idx, state_idx2, hidden_idx2)
    return (genVariableName(desc, "C__"))

def genP(state_idx, hidden_idx, action_idx, state_idx_cond, hidden_idx_cond):
    desc = "P_[s_{}_<{}> | a_{},s_{}_<{}>]".format(state_idx, hidden_idx, action_idx, state_idx_cond, hidden_idx_cond)
    return (genVariableName(desc, "P"))

def genR(state_idx1, hidden_idx1, action_idx, state_idx2, hidden_idx2):
    desc = "R_[s_{}_<{}>,a_{},s_{}_<{}>]".format(state_idx1, hidden_idx1, action_idx, state_idx2, hidden_idx2)
    return (genVariableName(desc, "R"))

def genRT(state_idx1, hidden_idx1, traj_idx, state_pos, action_idx, state_idx2, hidden_idx2):
    desc = "RT_[s_{}_<{}>--({},{}),a_{},s_{}_<{}>--({},{})]".format(state_idx1, hidden_idx1, traj_idx, state_pos, action_idx, state_idx2, hidden_idx2, traj_idx, state_pos + 1)
    return (genVariableName(desc, "RT"))

def genLER(traj_idx):
    desc = "LER_{}".format(traj_idx)
    return (genVariableName(desc, "LER"))

def genLB(poset_idx):
    desc = "LB_{}".format(poset_idx)
    return (genVariableName(desc, "LB"))

def genUB(poset_idx):
    desc = "UB_{}".format(poset_idx)
    return (genVariableName(desc, "UB"))

def gendelta(poset_idx1, poset_idx2):
    desc = "delta_{}_{}".format(poset_idx1, poset_idx2)
    return (genVariableName(desc, "delta"))

#
# Generate variable type -- Optimizer specific (BARON)

def typeT():
    return ("INTEGER_VARIABLE")

def typeHiddenState():
    return ("BINARY_VARIABLE")

def typeF():
    return ("BINARY_VARIABLE")

def typeC():
    return ("POSITIVE_VARIABLE")

def typeC2():
    return ("POSITIVE_VARIABLE")

def typeP():
    return ("POSITIVE_VARIABLE")

def typeR():
    return ("VARIABLE")

def typeRT():
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

def lowerboundT():
    if OPTIMIZER == "GUROBI":
        GUROBI_VARIABLES[genT()].lb = 0.0
        return (None)
    return ("{}: 0;".format(genT()))

def upperboundT(tau):
    if OPTIMIZER == "GUROBI":
        GUROBI_VARIABLES[genT()].ub = tau
        return (None)
    return ("{}: {};".format(genT(), tau))

def upperboundP(state_idx, hidden_idx, action_idx, state_idx_cond, hidden_idx_cond):
    if OPTIMIZER == "GUROBI":
        GUROBI_VARIABLES[genP(state_idx, hidden_idx, action_idx, state_idx_cond, hidden_idx_cond)].ub = 1.0
        return (None)
    return ("{}: 1;".format(genP(state_idx, hidden_idx, action_idx, state_idx_cond, hidden_idx_cond)))

def lowerboundR(state_idx1, hidden_idx1, action_idx, state_idx2, hidden_idx2, peak):
    if OPTIMIZER == "GUROBI":
        GUROBI_VARIABLES[genR(state_idx1, hidden_idx1, action_idx, state_idx2, hidden_idx2)].lb = -peak
        return (None)
    return ("{}: {};".format(genR(state_idx1, hidden_idx1, action_idx, state_idx2, hidden_idx2), -peak))

def upperboundR(state_idx1, hidden_idx1, action_idx, state_idx2, hidden_idx2, peak):
    if OPTIMIZER == "GUROBI":
        GUROBI_VARIABLES[genR(state_idx1, hidden_idx1, action_idx, state_idx2, hidden_idx2)].ub = peak
        return (None)
    return ("{}: {};".format(genR(state_idx1, hidden_idx1, action_idx, state_idx2, hidden_idx2), peak))

def lowerbounddelta(poset_idx1, poset_idx2):
    if OPTIMIZER == "GUROBI":
        GUROBI_VARIABLES[gendelta(poset_idx1, poset_idx2)].lb = 1.0
        return(None)
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

    # k * "s_{}_<{}>--({},{})".format(i, k, a, b) <= T - 1
def constraintT_name(state_idx, hidden_idx, traj_idx, state_pos):
    desc = "_T_{}_<{}>--({},{})".format(state_idx, hidden_idx, traj_idx, state_pos)
    return (genEquationName(desc, "T_"))

def constraintT(state_idx, hidden_idx, traj_idx, state_pos):
    if OPTIMIZER == "GUROBI":
        GUROBI_MODEL.addConstr(hidden_idx * GUROBI_VARIABLES[genHiddenState(state_idx, hidden_idx, traj_idx, state_pos)] - GUROBI_VARIABLES[genT()] <= -1, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} * {} - {} <= -1;".format(hidden_idx, genHiddenState(state_idx, hidden_idx, traj_idx, state_pos), genT()))

#   \sum_{k=0}^{tau-1} "s_{}_<{}>--({},{})".format(i, k, a, b) = 1
def constraintState_name(state_idx, traj_idx, state_pos):
    desc = "_s_{}--({},{})".format(state_idx, traj_idx, state_pos)
    return (genEquationName(desc, "s_"))

def constraintState(state_idx, traj_idx, state_pos, tau):
    if OPTIMIZER == "GUROBI":
        s = gurobipy.quicksum(GUROBI_VARIABLES[genHiddenState(state_idx, hidden_idx, traj_idx, state_pos)] for hidden_idx in range(tau))
        GUROBI_MODEL.addConstr(s == 1, GUROBI_CONSTRAINT_NAME)
        return (None)
    constraint = ""
    for hidden_idx in range(tau):
        if hidden_idx > 0:
            constraint += " + "
        constraint += "{}".format(genHiddenState(state_idx, hidden_idx, traj_idx, state_pos))
    constraint += " == 1;"
    return (constraint)

#   "F[s_{}_<{}>--({},{}),s_{}_<{}>--({},{})]".format(i, k, a, b, i', k', a', b') <= "s_{}_<{}>--({},{})".format(i, k, a, b)
def constraintF1_name(state_idx1, hidden_idx1, traj_idx1, state_pos1, state_idx2, hidden_idx2, traj_idx2, state_pos2):
    desc = "_F1_[s_{}_<{}>--({},{}),s_{}_<{}>--({},{})]".format(state_idx1, hidden_idx1, traj_idx1, state_pos1, state_idx2, hidden_idx2, traj_idx2, state_pos2)
    return (genEquationName(desc, "F1_"))

def constraintF1(state_idx1, hidden_idx1, traj_idx1, state_pos1, state_idx2, hidden_idx2, traj_idx2, state_pos2):
    if OPTIMIZER == "GUROBI":
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genF(state_idx1, hidden_idx1, traj_idx1, state_pos1, state_idx2, hidden_idx2, traj_idx2, state_pos2)] - GUROBI_VARIABLES[genHiddenState(state_idx1, hidden_idx1, traj_idx1, state_pos1)] <= 0, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} - {} <= 0;".format(genF(state_idx1, hidden_idx1, traj_idx1, state_pos1, state_idx2, hidden_idx2, traj_idx2, state_pos2), genHiddenState(state_idx1, hidden_idx1, traj_idx1, state_pos1)))

#   "F[s_{}_<{}>--({},{}),s_{}_<{}>--({},{})]".format(i, k, a, b, i', k', a', b') <= "s_{}_<{}>--({},{})".format(i', k', a', b')
def constraintF2_name(state_idx1, hidden_idx1, traj_idx1, state_pos1, state_idx2, hidden_idx2, traj_idx2, state_pos2):
    desc = "_F2_[s_{}_<{}>--({},{}),s_{}_<{}>--({},{})]".format(state_idx1, hidden_idx1, traj_idx1, state_pos1, state_idx2, hidden_idx2, traj_idx2, state_pos2)
    return (genEquationName(desc, "F2_"))

def constraintF2(state_idx1, hidden_idx1, traj_idx1, state_pos1, state_idx2, hidden_idx2, traj_idx2, state_pos2):
    if OPTIMIZER == "GUROBI":
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genF(state_idx1, hidden_idx1, traj_idx2, state_pos1, state_idx2, hidden_idx2, traj_idx2, state_pos2)] - GUROBI_VARIABLES[genHiddenState(state_idx2, hidden_idx2, traj_idx2, state_pos2)] <= 0, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} - {} <= 0;".format(genF(state_idx1, hidden_idx1, traj_idx1, state_pos1, state_idx2, hidden_idx2, traj_idx2, state_pos2), genHiddenState(state_idx2, hidden_idx2, traj_idx2, state_pos2)))

#   "s_{}_<{}>--({},{})".format(i, k, a, b) + "s_{}_<{}>--({},{})".format(i', k', a', b') - 1 <= F[s_{}_<{}>--({},{}),s_{}_<{}>--({},{})]".format(i, k, a, b, i', k', a', b')
def constraintF3_name(state_idx1, hidden_idx1, traj_idx1, state_pos1, state_idx2, hidden_idx2, traj_idx2, state_pos2):
    desc = "_F3_[s_{}_<{}>--({},{}),s_{}_<{}>--({},{})]".format(state_idx1, hidden_idx1, traj_idx1, state_pos1, state_idx2, hidden_idx2, traj_idx2, state_pos2)
    return (genEquationName(desc, "F3_"))

def constraintF3(state_idx1, hidden_idx1, traj_idx1, state_pos1, state_idx2, hidden_idx2, traj_idx2, state_pos2):
    if OPTIMIZER == "GUROBI":
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genHiddenState(state_idx1, hidden_idx1, traj_idx1, state_pos1)] + GUROBI_VARIABLES[genHiddenState(state_idx2, hidden_idx2, traj_idx2, state_pos2)] - GUROBI_VARIABLES[genF(state_idx1, hidden_idx1, traj_idx1, state_pos1, state_idx2, hidden_idx2, traj_idx2, state_pos2)] <= 1, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} + {} - {} <= 1;".format(genHiddenState(state_idx1, hidden_idx1, traj_idx1, state_pos1), genHiddenState(state_idx2, hidden_idx2, traj_idx2, state_pos2), genF(state_idx1, hidden_idx1, traj_idx1, state_pos1, state_idx2, hidden_idx2, traj_idx2, state_pos2)))

#   "C_[s_{}_<{}>,a_{}]".format(i, k, j) = \sum_{a,b:"a_{}--({},{})".format(j, a, b) occurs} "s_{}_<{}>--({},{})".format(i, k, a, b)
def constraintC_name(state_idx, hidden_idx, action_idx):
    desc = "_C_[s_{}_<{}>,a_{}]".format(state_idx, hidden_idx, action_idx)
    return (genEquationName(desc, "C_"))

def constraintC(state_idx, hidden_idx, action_idx, trajectories):
    if OPTIMIZER == "GUROBI":
        l = list()
        for t_idx in range(len(trajectories)):
            for pos in range(0, len(trajectories[t_idx]) - 2, 2):
                if trajectories[t_idx][pos] == state_idx and trajectories[t_idx][pos + 1] == action_idx:
                    l.append(GUROBI_VARIABLES[genHiddenState(state_idx, hidden_idx, t_idx, floor(pos / 2))])
        s = gurobipy.quicksum(t for t in l)
        GUROBI_MODEL.addConstr(s - GUROBI_VARIABLES[genC(state_idx, hidden_idx, action_idx)] == 0, GUROBI_CONSTRAINT_NAME)
        return (None)
    constraint = genC(state_idx, hidden_idx, action_idx)
    for t_idx in range(len(trajectories)):
        for pos in range(0, len(trajectories[t_idx]) - 2, 2):
            if trajectories[t_idx][pos] == state_idx and trajectories[t_idx][pos + 1] == action_idx:
                constraint += " - {}".format(genHiddenState(state_idx, hidden_idx, t_idx, floor(pos / 2)))
    constraint += " == 0;"
    return (constraint)

#   "C__[s_{}_<{}>,a_{},s_{}_<{}>]".format(i, k, j, i', k') = \sum_{a,b:"a_{}--({},{})".format(j, a, b) occur in some triple} "F[s_{}_<{}>--({},{}),s_{}_<{}>--({},{})]".format(i, k, a, b, i', k', a, b+1)
def constraintC2_name(state_idx1, hidden_idx1, action_idx, state_idx2, hidden_idx2):
    desc = "_C__[s_{}_<{}>,a_{},s_{}_<{}>]".format(state_idx1, hidden_idx1, action_idx, state_idx2, hidden_idx2)
    return (genEquationName(desc, "C2_"))

def constraintC2(state_idx1, hidden_idx1, action_idx, state_idx2, hidden_idx2, trajectories):
    if OPTIMIZER == "GUROBI":
        l = list()
        for t_idx in range(len(trajectories)):
            for pos in range(0, len(trajectories[t_idx]) - 2, 2):
                if trajectories[t_idx][pos] == state_idx1 and trajectories[t_idx][pos + 1] == action_idx and trajectories[t_idx][pos + 2] == state_idx2:
                    l.append(GUROBI_VARIABLES[genF(state_idx1, hidden_idx1, t_idx, floor(pos / 2), state_idx2, hidden_idx2, t_idx, floor(pos / 2) + 1)])
        s = gurobipy.quicksum(t for t in l)
        GUROBI_MODEL.addConstr(s - GUROBI_VARIABLES[genC2(state_idx1, hidden_idx1, action_idx, state_idx2, hidden_idx2)] == 0, GUROBI_CONSTRAINT_NAME)
    return (None)
    constraint = genC2(state_idx1, hidden_idx1, action_idx, state_idx2, hidden_idx2)
    for t_idx in range(len(trajectories)):
        for pos in range(0, len(trajectories[t_idx]) - 2, 2):
            if trajectories[t_idx][pos] == state_idx1 and trajectories[t_idx][pos + 1] == action_idx and trajectories[t_idx][pos + 2] == state_idx2:
                constraint += " - {}".format(genF(state_idx1, hidden_idx1, t_idx, floor(pos / 2), state_idx2, hidden_idx2, t_idx, floor(pos / 2) + 1))
    constraint += " == 0;"
    return (constraint)

#   "P_[s_{}_<{}> | a_{},s_{}_<{}>]".format(i, k, j, i', k') = "C__[s_{}_<{}>,a_{},s_{}_<{}>]".format(i, k, j, i', k') / "C_[s_{}_<{}>,a_{}]".format(i, k, j)
def constraintP_name(state_idx, hidden_idx, action_idx, state_idx_cond, hidden_idx_cond):
    desc = "_P_[s_{}_<{}> | a_{},s_{}_<{}>]".format(state_idx, hidden_idx, action_idx, state_idx_cond, hidden_idx_cond)
    return (genEquationName(desc, "P_"))

def constraintP(state_idx, hidden_idx, action_idx, state_idx_cond, hidden_idx_cond):
    if OPTIMIZER == "GUROBI":
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genP(state_idx, hidden_idx, action_idx, state_idx_cond, hidden_idx_cond)] * GUROBI_VARIABLES[genC(state_idx_cond, hidden_idx_cond, action_idx)] - GUROBI_VARIABLES[genC2(state_idx_cond, hidden_idx_cond, action_idx, state_idx, hidden_idx)] == 0, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} * {} - {} == 0;".format(genP(state_idx, hidden_idx, action_idx, state_idx_cond, hidden_idx_cond), genC(state_idx_cond, hidden_idx_cond, action_idx), genC2(state_idx_cond, hidden_idx_cond, action_idx, state_idx, hidden_idx)))

#   "RT_[s_{}_<{}>--({},{}),a_{},s_{}_<{}>--({},{})]".format(i, k, a, b, j, i', k', a, b+1) <= peak * "F[s_{}_<{}>--({},{}),s_{}_<{}>--({},{})]".format(i, k, a, b, i', k', a, b+1)
def constraintRT1_name(state_idx1, hidden_idx1, traj_idx, state_pos, action_idx, state_idx2, hidden_idx2):
    desc = "_RT1_[s_{}_<{}>--({},{}),a_{},s_{}_<{}>--({},{})]".format(state_idx1, hidden_idx1, traj_idx, state_pos, action_idx, state_idx2, hidden_idx2, traj_idx, state_pos + 1)
    return (genEquationName(desc, "RT1_"))

def constraintRT1(state_idx1, hidden_idx1, traj_idx, state_pos, action_idx, state_idx2, hidden_idx2, peak):
    if OPTIMIZER == "GUROBI":
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genRT(state_idx1, hidden_idx1, traj_idx, state_pos, action_idx, state_idx2, hidden_idx2)] - peak * GUROBI_VARIABLES[genF(state_idx1, hidden_idx1, traj_idx, state_pos, state_idx2, hidden_idx2, traj_idx, state_pos + 1)] <= 0, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} - {} * {} <= 0;".format(genRT(state_idx1, hidden_idx1, traj_idx, state_pos, action_idx, state_idx2, hidden_idx2), peak, genF(state_idx1, hidden_idx1, traj_idx, state_pos, state_idx2, hidden_idx2, traj_idx, state_pos + 1)))

#   "RT_[s_{}_<{}>--({},{}),a_{},s_{}_<{}>--({},{})]".format(i, k, a, b, j, i', k', a, b+1) >= -peak * "F[s_{}_<{}>--({},{}),s_{}_<{}>--({},{})]".format(i, k, a, b, i', k', a, b+1)
def constraintRT2_name(state_idx1, hidden_idx1, traj_idx, state_pos, action_idx, state_idx2, hidden_idx2):
    desc = "_RT2_[s_{}_<{}>--({},{}),a_{},s_{}_<{}>--({},{})]".format(state_idx1, hidden_idx1, traj_idx, state_pos, action_idx, state_idx2, hidden_idx2, traj_idx, state_pos + 1)
    return (genEquationName(desc, "RT2_"))

def constraintRT2(state_idx1, hidden_idx1, traj_idx, state_pos, action_idx, state_idx2, hidden_idx2, peak):
    if OPTIMIZER == "GUROBI":
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genRT(state_idx1, hidden_idx1, traj_idx, state_pos, action_idx, state_idx2, hidden_idx2)] + peak * GUROBI_VARIABLES[genF(state_idx1, hidden_idx1, traj_idx, state_pos, state_idx2, hidden_idx2, traj_idx, state_pos + 1)] >= 0, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} + {} * {} >= 0;".format(genRT(state_idx1, hidden_idx1, traj_idx, state_pos, action_idx, state_idx2, hidden_idx2), peak, genF(state_idx1, hidden_idx1, traj_idx, state_pos, state_idx2, hidden_idx2, traj_idx, state_pos + 1)))

#   "RT_[s_{}_<{}>--({},{}),a_{},s_{}_<{}>--({},{})]".format(i, k, a, b, j, i', k', a, b+1) <= "R_[s_{}_<{}>,a_{},s_{},<{}>]".format(i, k, j, i', k') + (1 - "F[s_{}_<{}>--({},{}),s_{}_<{}>--({},{})]".format(i, k, a, b, i', k', a, b+1)) * peak
def constraintRT3_name(state_idx1, hidden_idx1, traj_idx, state_pos, action_idx, state_idx2, hidden_idx2):
    desc = "_RT3_[s_{}_<{}>--({},{}),a_{},s_{}_<{}>--({},{})]".format(state_idx1, hidden_idx1, traj_idx, state_pos, action_idx, state_idx2, hidden_idx2, traj_idx, state_pos + 1)
    return (genEquationName(desc, "RT3_"))

def constraintRT3(state_idx1, hidden_idx1, traj_idx, state_pos, action_idx, state_idx2, hidden_idx2, peak):
    if OPTIMIZER == "GUROBI":
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genRT(state_idx1, hidden_idx1, traj_idx, state_pos, action_idx, state_idx2, hidden_idx2)] - GUROBI_VARIABLES[genR(state_idx1, hidden_idx1, action_idx, state_idx2, hidden_idx2)] + peak * GUROBI_VARIABLES[genF(state_idx1, hidden_idx1, traj_idx, state_pos, state_idx2, hidden_idx2, traj_idx, state_pos + 1)] <= peak, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} - {} + {} * {} <= {};".format(genRT(state_idx1, hidden_idx1, traj_idx, state_pos, action_idx, state_idx2, hidden_idx2), genR(state_idx1, hidden_idx1, action_idx, state_idx2, hidden_idx2), peak, genF(state_idx1, hidden_idx1, traj_idx, state_pos, state_idx2, hidden_idx2, traj_idx, state_pos + 1), peak))

#   "RT_[s_{}_<{}>--({},{}),a_{},s_{}_<{}>--({},{})]".format(i, k, a, b, j, i', k', a, b+1) >= "R_[s_{}_<{}>,a_{},s_{},<{}>]".format(i, k, j, i', k') - (1 - "F[s_{}_<{}>--({},{}),s_{}_<{}>--({},{})]".format(i, k, a, b, i', k', a, b+1)) * peak
def constraintRT4_name(state_idx1, hidden_idx1, traj_idx, state_pos, action_idx, state_idx2, hidden_idx2):
    desc = "_RT4_[s_{}_<{}>--({},{}),a_{},s_{}_<{}>--({},{})]".format(state_idx1, hidden_idx1, traj_idx, state_pos, action_idx, state_idx2, hidden_idx2, traj_idx, state_pos + 1)
    return (genEquationName(desc, "RT4_"))

def constraintRT4(state_idx1, hidden_idx1, traj_idx, state_pos, action_idx, state_idx2, hidden_idx2, peak):
    if OPTIMIZER == "GUROBI":
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genRT(state_idx1, hidden_idx1, traj_idx, state_pos, action_idx, state_idx2, hidden_idx2)] - GUROBI_VARIABLES[genR(state_idx1, hidden_idx1, action_idx, state_idx2, hidden_idx2)] - peak * GUROBI_VARIABLES[genF(state_idx1, hidden_idx1, traj_idx, state_pos, state_idx2, hidden_idx2, traj_idx, state_pos + 1)] <= peak, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} - {} - {} * {} >= -{};".format(genRT(state_idx1, hidden_idx1, traj_idx, state_pos, action_idx, state_idx2, hidden_idx2), genR(state_idx1, hidden_idx1, action_idx, state_idx2, hidden_idx2), peak, genF(state_idx1, hidden_idx1, traj_idx, state_pos, state_idx2, hidden_idx2, traj_idx, state_pos + 1), peak))

#   "LER_{}".format(a) = \sum_{b=0}^{length(a)-1} \sum_{k=0}^{tau-1} \sum_{k'=0}^{tau-1} "P_[s_{}_<{}> | a_{},s_{}_<{}>]".format(i', k', j, i, k) * "RT_[s_{}_<{}>--({},{}),a_{},s_{}_<{}>--({},{})]".format(i, k, a, b, j, i', k', a, b+1)
def constraintLER_name(traj_idx):
    desc = "_LER_{}".format(traj_idx)
    return (genEquationName(desc, "LER_"))

def constraintLER(traj_idx, trajectory, tau):
    if OPTIMIZER == "GUROBI":
        l = list()
        for state_pos in range(floor(len(trajectory) / 2)):
            for hidden_idx1 in range(tau):
                for hidden_idx2 in range(tau):
                    l.append(( GUROBI_VARIABLES[genP(trajectory[(state_pos + 1) * 2], hidden_idx2, trajectory[state_pos * 2 + 1], trajectory[state_pos * 2], hidden_idx1)], GUROBI_VARIABLES[genRT(trajectory[state_pos * 2], hidden_idx1, traj_idx, state_pos, trajectory[state_pos * 2 + 1], trajectory[(state_pos + 1) * 2], hidden_idx2)] ))
        s = gurobipy.quicksum( - w * v for w, v in l )
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genLER(traj_idx)] + s == 0, GUROBI_CONSTRAINT_NAME)
        return (None)
    constraint = genLER(traj_idx)
    for state_pos in range(floor(len(trajectory) / 2)):
        for hidden_idx1 in range(tau):
            for hidden_idx2 in range(tau):
                constraint += " - {} * {}".format(genP(trajectory[(state_pos + 1) * 2], hidden_idx2, trajectory[state_pos * 2 + 1], trajectory[state_pos * 2], hidden_idx1), genRT(trajectory[state_pos * 2], hidden_idx1, traj_idx, state_pos, trajectory[state_pos * 2 + 1], trajectory[(state_pos + 1) * 2], hidden_idx2))
    constraint += " == 0;"
    return (constraint)

#   "LB_{}".format(u) <= UB_{}".format(u)
def constraintPosetLBUB_name(poset_idx):
    desc = "_POSETLBUB_{}".format(poset_idx)
    return (genEquationName(desc, "POSET"))

def constraintPosetLBUB(poset_idx):
    if OPTIMIZER == "GUROBI":
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genLB(poset_idx)] - GUROBI_VARIABLES[genUB(poset_idx)] <= 0, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} - {} <= 0;".format(genLB(poset_idx), genUB(poset_idx)))

#   "LB_{}".format(q) = "UB_{}".format(q+1) + "delta_{}_{}".format(q, q+1)
def constraintPoset_name(poset_idx1, poset_idx2):
    desc = "_POSETS_{}_{}".format(poset_idx1, poset_idx2)
    return (genEquationName(desc, "POSETS_"))

def constraintPoset(poset_idx1, poset_idx2):
    if OPTIMIZER == "GUROBI":
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genLB(poset_idx1)] - GUROBI_VARIABLES[genUB(poset_idx2)] - GUROBI_VARIABLES[gendelta(poset_idx1, poset_idx2)] == 0, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} - {} - {} == 0;".format(genLB(poset_idx1), genUB(poset_idx2), gendelta(poset_idx1, poset_idx2)))

#   "LB_{}".format(q) <= "LER_{}".format(a)
def constraintPosetLB_name(poset_idx, traj_idx):
    desc = "_POSETLB_{}_{}".format(poset_idx, traj_idx)
    return (genEquationName(desc, "POSETLB_"))

def constraintPosetLB(poset_idx, traj_idx):
    if OPTIMIZER == "GUROBI":
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genLB(poset_idx)] - GUROBI_VARIABLES[genLER(traj_idx)] <= 0, GUROBI_CONSTRAINT_NAME)
        return (None)
    return ("{} - {} <= 0;".format(genLB(poset_idx), genLER(traj_idx)))

#   "UB_{}".format(q) >= "LER_{}".format(a)
def constraintPosetUB_name(poset_idx, traj_idx):
    desc = "_POSETUB_{}_{}".format(poset_idx, traj_idx)
    return (genEquationName(desc, "POSETUB_"))

def constraintPosetUB(poset_idx, traj_idx):
    if OPTIMIZER == "GUROBI":
        GUROBI_MODEL.addConstr(GUROBI_VARIABLES[genUB(poset_idx)] - GUROBI_VARIABLES[genLER(traj_idx)] >= 0, GUROBI_CONSTRAINT_NAME)
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

def declareVariables(num_states, num_actions, tau, num_posets, trajectories):
    # Returns a tuple with the first element a list of output strings each a
    # row and second element is a dictionary mapping each LER variable to the
    # unique associated trajectory.
    rows = list()
    binary_variables = set()
    integer_variables = set()
    positive_variables = set()
    variables = set()
    LER_to_trajectory = dict()

    if DEBUG:
        print ("...T variable...")
    addVariable(genT(), typeT(), binary_variables, integer_variables, positive_variables, variables)
    if DEBUG:
        print ("...... 1 declared.")

    if DEBUG:
        print ("...HiddenState variables...")
        var_ct = len(var_to_var_desc)
    for t_idx in range(len(trajectories)):
        traj = trajectories[t_idx]
        for pos in range(0, len(traj), 2):
            for h_idx in range(tau):
                addVariable(genHiddenState(traj[pos], h_idx, t_idx, floor(pos / 2)), typeHiddenState(), binary_variables, integer_variables, positive_variables, variables)
    if DEBUG:
        print ("......{} delared.".format(len(var_to_var_desc) - var_ct))

    if DEBUG:
        print ("...F variables...")
        var_ct = len(var_to_var_desc)
    for t_idx in range(len(trajectories)):
        traj = trajectories[t_idx]
        for pos in range(0, len(traj) - 1, 2):
            s_idx1 = traj[pos]
            a_idx = traj[pos + 1]
            s_idx2 = traj[pos + 2]
            for h_idx1 in range(tau):
                for h_idx2 in range(tau):
                    addVariable(genF(s_idx1, h_idx1, t_idx, floor(pos / 2), s_idx2, h_idx2, t_idx, floor(pos / 2) + 1), typeF(), binary_variables, integer_variables, positive_variables, variables)
    if DEBUG:
        print ("......{} delared.".format(len(var_to_var_desc) - var_ct))

    if DEBUG:
        print ("...C variables...")
        var_ct = len(var_to_var_desc)
    for t_idx in range(len(trajectories)):
        traj = trajectories[t_idx]
        for pos in range(0, len(traj) - 1, 2):
            s_idx = traj[pos]
            a_idx = traj[pos + 1]
            for h_idx in range(tau):
                addVariable(genC(s_idx, h_idx, a_idx), typeC(), binary_variables, integer_variables, positive_variables, variables)
    if DEBUG:
        print ("......{} delared.".format(len(var_to_var_desc) - var_ct))

    if DEBUG:
        print ("...C2 variables...")
        var_ct = len(var_to_var_desc)
    for t_idx in range(len(trajectories)):
        traj = trajectories[t_idx]
        for pos in range(0, len(traj) - 1, 2):
            s_idx1 = traj[pos]
            a_idx = traj[pos + 1]
            s_idx2 = traj[pos + 2]
            for h_idx1 in range(tau):
                for h_idx2 in range(tau):
                    addVariable(genC2(s_idx1, h_idx1, a_idx, s_idx2, h_idx2), typeC2(), binary_variables, integer_variables, positive_variables, variables)
    if DEBUG:
        print ("......{} delared.".format(len(var_to_var_desc) - var_ct))

    if DEBUG:
        print ("...P variables...")
        var_ct = len(var_to_var_desc)
    for t_idx in range(len(trajectories)):
        traj = trajectories[t_idx]
        for pos in range(0, len(traj) - 1, 2):
            s_idx1 = traj[pos]
            a_idx = traj[pos + 1]
            s_idx2 = traj[pos + 2]
            for h_idx1 in range(tau):
                for h_idx2 in range(tau):
                    addVariable(genP(s_idx2, h_idx2, a_idx, s_idx1, h_idx1), typeP(), binary_variables, integer_variables, positive_variables, variables)
    if DEBUG:
        print ("......{} delared.".format(len(var_to_var_desc) - var_ct))

    if DEBUG:
        print ("...R variables...")
        var_ct = len(var_to_var_desc)
    for t_idx in range(len(trajectories)):
        traj = trajectories[t_idx]
        for pos in range(0, len(traj) - 1, 2):
            s_idx1 = traj[pos]
            a_idx = traj[pos + 1]
            s_idx2 = traj[pos + 2]
            for h_idx1 in range(tau):
                for h_idx2 in range(tau):
                    addVariable(genR(s_idx1, h_idx1, a_idx, s_idx2, h_idx2), typeR(), binary_variables, integer_variables, positive_variables, variables)
    if DEBUG:
        print ("......{} delared.".format(len(var_to_var_desc) - var_ct))

    if DEBUG:
        print ("...RT variables...")
        var_ct = len(var_to_var_desc)
    for t_idx in range(len(trajectories)):
        traj = trajectories[t_idx]
        for pos in range(0, len(traj) - 1, 2):
            s_idx1 = traj[pos]
            a_idx = traj[pos + 1]
            s_idx2 = traj[pos + 2]
            for h_idx1 in range(tau):
                for h_idx2 in range(tau):
                    addVariable(genRT(s_idx1, h_idx1, t_idx, floor(pos / 2), a_idx, s_idx2, h_idx2), typeRT(), binary_variables, integer_variables, positive_variables, variables)
    if DEBUG:
        print ("......{} delared.".format(len(var_to_var_desc) - var_ct))

    if DEBUG:
        print ("...LER variables...")
        var_ct = len(var_to_var_desc)
    for t_idx, traj in enumerate(trajectories):
        var_name = genLER(t_idx)
        addVariable(var_name, typeLER(), binary_variables, integer_variables, positive_variables, variables)
        LER_to_trajectory[var_name] = ( t_idx, tuple(traj) )
    if DEBUG:
        print ("......{} delared.".format(len(var_to_var_desc) - var_ct))

    if DEBUG:
        print ("...LB variables...")
        var_ct = len(var_to_var_desc)
    for p_idx in range(num_posets):
        addVariable(genLB(p_idx), typeLB(), binary_variables, integer_variables, positive_variables, variables)
    if DEBUG:
        print ("......{} delared.".format(len(var_to_var_desc) - var_ct))

    if DEBUG:
        print ("...UB variables...")
        var_ct = len(var_to_var_desc)
    for p_idx in range(num_posets):
        addVariable(genUB(p_idx), typeUB(), binary_variables, integer_variables, positive_variables, variables)
    if DEBUG:
        print ("......{} delared.".format(len(var_to_var_desc) - var_ct))

    if DEBUG:
        print ("...delta variables...")
        var_ct = len(var_to_var_desc)
    for p_idx in range(num_posets - 1):
        addVariable(gendelta(p_idx, p_idx + 1), typedelta(), binary_variables, integer_variables, positive_variables, variables)
    if DEBUG:
        print ("......{} delared.".format(len(var_to_var_desc) - var_ct))


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
    return (rows, LER_to_trajectory)

#
# Bound variables
def boundVariables(num_states, num_actions, tau, peak, trajectories, num_posets):
    # Returns a list of rows detailing the bounds
    rows = list()
    if OPTIMIZER != "GUROBI":
        rows.append("LOWER_BOUNDS{")

    rows.append(lowerboundT())
    for t_idx in range(len(trajectories)):
        traj = trajectories[t_idx]
        for pos in range(0, len(traj) - 1, 2):
            s_idx1 = traj[pos]
            a_idx = traj[pos + 1]
            s_idx2 = traj[pos + 2]
            for h_idx1 in range(tau):
                for h_idx2 in range(tau):
                    rows.append(lowerboundR(s_idx1, h_idx1, a_idx, s_idx2, h_idx2, peak))
    for p_idx in range(num_posets - 1):
        rows.append(lowerbounddelta(p_idx, p_idx + 1))
    if OPTIMIZER != "GUROBI":
        rows.append("}")
        rows.append(" ")

        rows.append("UPPER_BOUNDS{")

    rows.append(upperboundT(tau))
    for t_idx in range(len(trajectories)):
        traj = trajectories[t_idx]
        for pos in range(0, len(traj) - 1, 2):
            s_idx1 = traj[pos]
            a_idx = traj[pos + 1]
            s_idx2 = traj[pos + 2]
            for h_idx1 in range(tau):
                for h_idx2 in range(tau):
                    rows.append(upperboundP(s_idx2, h_idx2, a_idx, s_idx1, h_idx1))

    for t_idx in range(len(trajectories)):
        traj = trajectories[t_idx]
        for pos in range(0, len(traj) - 1, 2):
            s_idx1 = traj[pos]
            a_idx = traj[pos + 1]
            s_idx2 = traj[pos + 2]
            for h_idx1 in range(tau):
                for h_idx2 in range(tau):
                    rows.append(upperboundR(s_idx1, h_idx1, a_idx, s_idx2, h_idx2, peak))
    if OPTIMIZER != "GUROBI":
        rows.append("}")
    if OPTIMIZER == "GUROBI":
        for row in rows:
            if row != None:
                sys.exit('boundVariables returned a non-empty row {} for GUROBI'.format(row))
        rows = list()
    return (rows)

#
# Build constraints
def generateConstraints(num_states, num_actions, tau, peak, posets, trajectories):
    # Returns a list of rows detailing the constraint declarations and constraints -- specific to (BARON)
    equation_names = list()
    equation_names_set = set()
    equations = list()

    if DEBUG:
        print ("...T constraints...")
    num_constraints = 0
    for t_idx in range(len(trajectories)):
        traj = trajectories[t_idx]
        for pos in range(0, len(traj), 2):
            s_idx = traj[pos]
            for h_idx in range(1, tau): # skip 0 since it is vacuously true according to the constraint.
                eqn = constraintT_name(s_idx, h_idx, t_idx, floor(pos / 2))
                if eqn in equation_names_set: # skip
                    continue
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintT(s_idx, h_idx, t_idx, floor(pos / 2)))
                num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    if DEBUG:
        print ("...State constraints...")
    num_constraints = 0
    for t_idx in range(len(trajectories)):
        traj = trajectories[t_idx]
        for pos in range(0, len(traj), 2):
            s_idx = traj[pos]
            eqn = constraintState_name(s_idx, t_idx, floor(pos / 2))
            if eqn in equation_names_set: # skip
                continue
            equation_names_set.add(eqn)
            equation_names.append(eqn)
            equations.append(constraintState(s_idx, t_idx, floor(pos / 2), tau))
            num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    if DEBUG:
        print ("...F constraints...")
    num_constraints = 0
    for t_idx in range(len(trajectories)):
        traj = trajectories[t_idx]
        for pos in range(0, len(traj) - 1, 2):
            s_idx1 = traj[pos]
            a_idx = traj[pos + 1]
            s_idx2 = traj[pos + 2]
            for h_idx1 in range(tau):
                for h_idx2 in range(tau):
                    eqn = constraintF1_name(s_idx1, h_idx1, t_idx, floor(pos / 2), s_idx2, h_idx2, t_idx, floor(pos / 2) + 1)
                    if not eqn in equation_names_set:
                        equation_names_set.add(eqn)
                        equation_names.append(eqn)
                        equations.append(constraintF1(s_idx1, h_idx1, t_idx, floor(pos / 2), s_idx2, h_idx2, t_idx, floor(pos / 2) + 1))
                        num_constraints += 1
                    eqn = constraintF2_name(s_idx1, h_idx1, t_idx, floor(pos / 2), s_idx2, h_idx2, t_idx, floor(pos / 2) + 1)
                    if not eqn in equation_names_set:
                        equation_names_set.add(eqn)
                        equation_names.append(eqn)
                        equations.append(constraintF2(s_idx1, h_idx1, t_idx, floor(pos / 2), s_idx2, h_idx2, t_idx, floor(pos / 2) + 1))
                        num_constraints += 1
                    eqn = constraintF3_name(s_idx1, h_idx1, t_idx, floor(pos / 2), s_idx2, h_idx2, t_idx, floor(pos / 2) + 1)
                    if not eqn in equation_names_set:
                        equation_names_set.add(eqn)
                        equation_names.append(eqn)
                        equations.append(constraintF3(s_idx1, h_idx1, t_idx, floor(pos / 2), s_idx2, h_idx2, t_idx, floor(pos / 2) + 1))
                        num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    if DEBUG:
        print ("...C constraints...")
    num_constraints = 0
    for t_idx in range(len(trajectories)):
        traj = trajectories[t_idx]
        for pos in range(0, len(traj) - 1, 2):
            s_idx = traj[pos]
            a_idx = traj[pos + 1]
            for h_idx in range(tau):
                eqn = constraintC_name(s_idx, h_idx, a_idx)
                if not eqn in equation_names_set:
                    equation_names_set.add(eqn)
                    equation_names.append(eqn)
                    equations.append(constraintC(s_idx, h_idx, a_idx, trajectories))
                    num_constraints += 1

    for t_idx in range(len(trajectories)):
        traj = trajectories[t_idx]
        for pos in range(0, len(traj) - 1, 2):
            s_idx1 = traj[pos]
            a_idx = traj[pos + 1]
            s_idx2 = traj[pos + 2]
            for h_idx1 in range(tau):
                for h_idx2 in range(tau):
                    eqn = constraintC2_name(s_idx1, h_idx1, a_idx, s_idx2, h_idx2)
                    if not eqn in equation_names_set:
                        equation_names_set.add(eqn)
                        equation_names.append(eqn)
                        equations.append(constraintC2(s_idx1, h_idx1, a_idx, s_idx2, h_idx2, trajectories))
                        num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    if DEBUG:
        print ("...P constraints...")
    num_constraints = 0
    for t_idx in range(len(trajectories)):
        traj = trajectories[t_idx]
        for pos in range(0, len(traj) - 1, 2):
            s_idx1 = traj[pos]
            a_idx = traj[pos + 1]
            s_idx2 = traj[pos + 2]
            for h_idx1 in range(tau):
                for h_idx2 in range(tau):
                    eqn = constraintP_name(s_idx2, h_idx2, a_idx, s_idx1, h_idx1)
                    if not eqn in equation_names_set:
                        equation_names_set.add(eqn)
                        equation_names.append(eqn)
                        equations.append(constraintP(s_idx2, h_idx2, a_idx, s_idx1, h_idx1))
                        num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    if DEBUG:
        print ("...r constraints...")
    num_constraints = 0
    for t_idx in range(len(trajectories)):
        traj = trajectories[t_idx]
        for pos in range(0, len(traj) - 1, 2):
            s_idx1 = traj[pos]
            a_idx = traj[pos + 1]
            s_idx2 = traj[pos + 2]
            for h_idx1 in range(tau):
                for h_idx2 in range(tau):
                    eqn = constraintRT1_name(s_idx1, h_idx1, t_idx, floor(pos / 2), a_idx, s_idx2, h_idx2)
                    if not eqn in equation_names_set:
                        equation_names_set.add(eqn)
                        equation_names.append(eqn)
                        equations.append(constraintRT1(s_idx1, h_idx1, t_idx, floor(pos / 2), a_idx, s_idx2, h_idx2, peak))
                        num_constraints += 1
                    eqn = constraintRT2_name(s_idx1, h_idx1, t_idx, floor(pos / 2), a_idx, s_idx2, h_idx2)
                    if not eqn in equation_names_set:
                        equation_names_set.add(eqn)
                        equation_names.append(eqn)
                        equations.append(constraintRT2(s_idx1, h_idx1, t_idx, floor(pos / 2), a_idx, s_idx2, h_idx2, peak))
                        num_constraints += 1
                    eqn = constraintRT3_name(s_idx1, h_idx1, t_idx, floor(pos / 2), a_idx, s_idx2, h_idx2)
                    if not eqn in equation_names_set:
                        equation_names_set.add(eqn)
                        equation_names.append(eqn)
                        equations.append(constraintRT3(s_idx1, h_idx1, t_idx, floor(pos / 2), a_idx, s_idx2, h_idx2, peak))
                        num_constraints += 1
                    eqn = constraintRT4_name(s_idx1, h_idx1, t_idx, floor(pos / 2), a_idx, s_idx2, h_idx2)
                    if not eqn in equation_names_set:
                        equation_names_set.add(eqn)
                        equation_names.append(eqn)
                        equations.append(constraintRT4(s_idx1, h_idx1, t_idx, floor(pos / 2), a_idx, s_idx2, h_idx2, peak))
                        num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    if DEBUG:
        print ("...LER constraints...")
    num_constraints = 0
    for t_idx in range(len(trajectories)):
        eqn = constraintLER_name(t_idx)
        if not eqn in equation_names_set:
            equation_names_set.add(eqn)
            equation_names.append(eqn)
            equations.append(constraintLER(t_idx, trajectories[t_idx], tau))
            num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    if DEBUG:
        print ("...LB/UB/delta constraints...")
    num_constraints = 0
    #   "LB_{}".format(u) <= UB_{}".format(u)
    for p_idx in range(len(posets)):
        eqn = constraintPosetLBUB_name(p_idx)
        if not eqn in equation_names_set:
            equation_names_set.add(eqn)
            equation_names.append(eqn)
            equations.append(constraintPosetLBUB(p_idx))

    for p_idx in range(len(posets) - 1):
        eqn = constraintPoset_name(p_idx, p_idx + 1)
        if not eqn in equation_names_set:
            equation_names_set.add(eqn)
            equation_names.append(eqn)
            equations.append(constraintPoset(p_idx, p_idx + 1))

    for p_idx, poset in enumerate(posets):
        for t_idx in poset:
            eqn = constraintPosetLB_name(p_idx, t_idx)
            if not eqn in equation_names_set:
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintPosetLB(p_idx, t_idx))

    for p_idx, poset in enumerate(posets):
        for t_idx in poset:
            eqn = constraintPosetUB_name(p_idx, t_idx)
            if not eqn in equation_names_set:
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintPosetUB(p_idx, t_idx))
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    if DEBUG:
        print ("Total constraints created = {}".format(len(equation_names)))
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
    num_actions = kwargs['num_actions']
    tau = kwargs['tau']
    peak = kwargs['peak']
    num_posets = kwargs['num_posets']
    posets = kwargs['posets']
    t = kwargs['all_trajectories']
#    dm_trajectory_indices = kwargs['dm_trajectory_indices']
#    others_trajectory_indices = kwargs['others_trajectory_indices']
    global OPTIMIZER
    OPTIMIZER = kwargs['optimizer']
    maxtime = kwargs['maxtime']
    license_fn = kwargs['license_fn']
    feasibilityOnly = kwargs['feasibility_only']
# Builds and runs the optimization problem.
#    dm_trajectory_indices and others_trajectory_indices is ignored for now
#   maxtime is maximum time allowed for BARON
#   license_fn is the full path filename to the BARON license

    if DEBUG:
        print ("Building SsMDP optimization problem...")

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
        print ("Declaring variables...")
    ( new_rows, LER_to_trajectories ) = declareVariables(num_states, num_actions, tau, num_posets, t)
    rows.extend(new_rows)
    rows.append(" ")
    if DEBUG:
        print ("\t...elapsed time = {}".format(time.time() - start_time))
        start_time = time.time()
        print ("Constructing variable bounds...")
    rows.extend(boundVariables(num_states, num_actions, tau, peak, t, num_posets))
    rows.append(" ")
    if DEBUG:
        print ("\t...elapsed time = {}".format(time.time() - start_time))
        start_time = time.time()
        print ("Generating constraints...")
    rows.extend(generateConstraints(num_states, num_actions, tau, peak, posets, t))
    if feasibilityOnly:
        if OPTIMIZER == "BARON":
            rows.append(" ")
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
#    rows.append("\t{};".format(genT()))
    return ( rows, LER_to_trajectories, None, None, None, var_to_var_desc )



import L_MDP
import T_MDP

def test(tau=5):
    t1 = [ 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 2, 0, 2, 0, 2, 0, 2 ]
    t2 = [ 1, 0, 1, 1, 2, 0, 2, 0, 2, 0, 2]
    t3 = [ 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 2, 0, 2, 0, 2 ]
    num_posets = 2
    posets = [[0], [1,2]]
    num_states = 2
    num_actions = 2
    peak = 100
    t = [ t1, t2, t3 ]

#    t4 = [ 1, 0, 1, 0, 1, 1, 2 ]
#    t5 = [ 1, 0, 1, 1, 2 ]
#    t6 = [ 1, 0, 1, 0, 1, 0, 1, 1, 2 ]
#    t7 = [ 1, 1, 2 ]
#    num_posets = 3
#    posets = [[1, 2], [0], [3]]
#    num_states = 3
#    num_actions = 2
#    peak = 100
#    t = [ t4, t5, t6, t7 ]
    results = L_MDP.learnMDPabs(num_actions, num_states, "SsMDP-test0-{}".format(tau), tau, peak, num_posets, posets, t, [ 1 ], [ 0, 2 ], constructOptimization, "SsMDP", "GUROBI", "", 10000, "")
#    results = L_MDP.learnMDPabs(num_actions, num_states, "SsMDP-test0-{}".format(tau), tau, peak, num_posets, posets, t, [ 1, 2 ], [ 0, 3 ], constructOptimization, "SsMDP", "BARON", "./baron", 10000, "./baronlice.txt")
    if results[0]:
        transforms = T_MDP.transformBySsMDP(num_states, t, tau, results[2], results[7])
        return (transforms)


    # problems to consider:
    #   - Want only odd numbered cycles or maybe 2 and 5 but not others
    #   - Must always pass through a particular state for every trajectory
    #       - no matter where I start or end, I must pass through X but not repeat any state

def test2(tau=1):
#   1   2   3   4   5
#   6   7   8   9   10
#   11  12  13  14  15
#   16  17  18  19  20
#   21  22  23  24  25
    t1 = [ 1, 0, 7, 0, 13, 0, 19, 0, 25 ]
    t2 = [ 5, 0, 9, 0, 13, 0, 17, 0, 21 ]
    t3 = [ 5, 0, 9, 0, 13, 0, 19, 0, 25 ]
    t4 = [ 25, 0, 19, 0, 13, 0, 7, 0, 1 ]

    t5 = [ 1, 0, 2, 0, 3, 0, 4, 0, 5 ]
    t6 = [ 21, 0, 17, 0, 12, 0, 8, 0, 9 ]
    t7 = [ 1, 0, 7, 0, 13 ]
    t8 = [ 13, 0, 17, 0, 21 ]
    t9 = [ 13, 0, 19, 0, 25 ]
    t10 = [ 5, 0, 9, 0, 13 ]
    num_posets = 2
    posets = [[4, 5, 6, 7, 8, 9], [0, 1, 2, 3]]
    num_states = 25
    num_actions = 2
    peak = 100
    t = [ t1, t2, t3, t4, t5, t6, t7, t8, t9, t10 ]
    rows = constructOptimization(num_states, num_actions, tau, peak, num_posets, posets, t, 10000, "./baronlice.txt")
    with open("tmp2-SsMDP.mnlp", "w") as f:
        for row in rows:
            f.write(row + '\n')
    os.system('./baron tmp2-SsMDP.mnlp')
    answers = extractAnswers("soln2-SsMDP.csv")

def test3(tau = 1,maxtime = 10000):
#   1   2   3   4   5
#   6   7   8   9   10
#   11  12  13  14  15
#   16  17  18  19  20
#   21  22  23  24  25
    t1 = [ 11, 100, 6, 100, 1, 100, 1, 200, 2, 100, 3 ]
    t2 = [ 3, 100, 2, 100, 1, 100, 1, 300, 6, 100, 11 ]

    t3 = [ 11, 100, 6, 100, 1, 100, 1, 300, 6, 100, 11 ]
    t4 = [ 3, 100, 2, 100, 1, 100, 1, 200, 2, 100, 3 ]

    num_posets = 2
    posets = [[0, 1], [2, 3]]
    num_states = 5
    num_actions = 3
    peak = 100
    t = [ t1, t2, t3, t4 ]
    dm_trajectory_indices = posets[0]
    rows = constructOptimization(num_states, num_actions, tau, peak, num_posets, posets, t, maxtime, "./baronlice.txt")
    with open("tmp3-SsMDP.mnlp", "w") as f:
        for row in rows:
            f.write(row + '\n')
    os.system('./baron tmp3-SsMDP.mnlp')
    answers = extractAnswers("soln3-SsMDP.csv")
