#
# Filename:     RoMDP.py
# Date:         2020-05-12
# Project:      Rewards-only MDP
# Author:       Eugene Santos Jr.
# Copyright     Eugene Santos Jr.
#
#
import sys
from math import floor
import os
import csv
import L_MDP
import copy
import time

DEBUG = True

#
# Notation:
# "##" indicates not used in this formulation.
#
#   states      -- "s_{}".format(i)
#                   -- this is the ith state
#                   -- states index from 0
#               -- "s_{}--({},{})".format(i, a, b)
#                   -- the ath trajectory's bth state which is state i
#   actions     -- "a_{}".format(i)
#                   -- this is the ith action
#                   -- actions index from 0
#               -- "a_{}--({},{})".format(i, a, b)
#                   -- the ath trajectory's bth action which is action i
#   trajectories
#               -- of the form: [ "s_{}--({},{})".format(i, a, 0), "a_{}--({},{})".format(i, a, 0), "s_{}--({},{})".format(i, a, 1), "a_{}--({},{})".format(i, a, 1), ... ]
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
#   rewards
#               -- "R_[s_{},a_{},s_{}]".format(i, j, i')
#                   -- reward value to transition from state i hidden value k with action j to state i' hidden value k'
#               -- "peak"
#                   -- non-negative real value representing largest magnitude for all reward values
#               -- "FRV_targ_[s_{},a_{},s_{}]".format(i, j, i')
#                   -- fractional reward value related to target decision-maker bias
#               -- "FRV_nontarg_[s_{},a_{},s_{}]".format(i, j, i')
#                   -- fractional reward value related to nontarget decision-maker bias
#               -- "FRV_maxMagnitude"
#                   -- fractional reward value magnitude max among all FRV_targ and FRV_nontarg above

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
#   "R_[s_{},a_{},s_{}]".format(i, j, i')
#       -- real valued variable corresponding to the reward for s_i, a_j, s_{i'}
#       -- bounded by [-peak, peak]
#   "FRV_targ_[s_{},a_{},s_{}]".format(i, j, i')
#       -- real valued variable corresponding to the target decision-maker fractional reward value for s_i, a_j, s_{i'}
#       -- bounded by [-2 * peak, 2 * peak]
#   "FRV_nontarg_[s_{},a_{},s_{}]".format(i, j, i')
#       -- real valued variable corresponding to the nontarget decision-maker fractional reward value for s_i, a_j, s_{i'}
#       -- bounded by [-2 * peak, 2 * peak]
#   "FRV_maxMagnitude"
#       -- non-negative real valued variable representing largest magnitude among FRV_targs and FRV_nontargs
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
#   "LER_{}".format(a) = \sum_{b=0}^{length(a)-1} "P_[s_{} | a_{},s_{}]".format(i_{b+1}, j_b, i_b) * "R_[s_{},a_{},s_{}]".format(i_b, j_b, i_{b+1})
#       -- computes LER for trajectory a of form [ "s_{}--({},{})".format(i_0, a, 0), "a_{}--({},{})".format(j_0, a, 0), "s_{}--({},{})".format(i_1, a, 1), "a_{}--({},{})".format(j_1, a, 1), ... ]
#
#   "R_[s_{},a_{},s_{}]".format(i, j, i') = peak * P_targ(a_j, s_i) + "FRV_targ_[s_{},a_{},s_{}]".format(i, j, i')
#   "FRV_targ_[s_{},a_{},s_{}]".format(i, j, i') <= "FRV_maxMagnitude"
#   -"FRV_targ_[s_{},a_{},s_{}]".format(i, j, i') <= "FRV_maxMagnitude"
#       -- If {s, a, s'} is a contiguous subsequence of some target decision-maker trajectory
#
#   "R_[s_{},a_{},s_{}]".format(i, j, i') = -peak * P_nontarg(a_j, s_i) + "FRV_nontarg_[s_{},a_{},s_{}]".format(i, j, i')
#   "FRV_nontarg_[s_{},a_{},s_{}]".format(i, j, i') <= "FRV_maxMagnitude"
#   -"FRV_nontarg_[s_{},a_{},s_{}]".format(i, j, i') <= "FRV_maxMagnitude"
#       -- Only if {s, a, s'} is not a contiguous subsequence of any target decision-maker trajectory and
#           is a contiguous subsequence of some non-target decision-maker trajectory

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
#       max \sum_{q=0}{num_posets-1} "delta_{}_{}".format(q, q+1)
#       -- maximizes the distance between posets
#       -- can overfit trajectories

#
# Generate variable strings

def genVariableName(desc, prefix):
    """ Generates global vars just to keep track of things """
    global next_var_idx
    try:
        return (var_desc_to_var[desc])
    except KeyError:
        name = "{}{}".format(prefix, next_var_idx)
        next_var_idx += 1
        var_desc_to_var[desc] = name
        var_to_var_desc[name] = desc
        return (name)

def genR(state_idx1, action_idx, state_idx2):
    desc = "R_[s_{},a_{},s_{}]".format(state_idx1, action_idx, state_idx2)
    return (genVariableName(desc, "R"))

def genFRV_targ(state_idx1, action_idx, state_idx2):
    desc = "FRV_targ_[s_{},a_{},s_{}]".format(state_idx1, action_idx, state_idx2)
    return (genVariableName(desc, "FRV_TARG"))

def genFRV_nontarg(state_idx1, action_idx, state_idx2):
    desc = "FRV_nontarg_[s_{},a_{},s_{}]".format(state_idx1, action_idx, state_idx2)
    return (genVariableName(desc, "FRV_NONTARG"))

def genFRV_maxMagnitude():
    desc = "FRV_maxMagnitude"
    return (genVariableName(desc, "FRV_maxMagnitude"))

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

def typeR():
    return ("VARIABLE")

def typeFRV_targ():
    return ("VARIABLE")

def typeFRV_nontarg():
    return ("VARIABLE")

def typeFRV_maxMagnitude():
    return ("POSITIVE_VARIABLE")

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

def lowerboundR(state_idx1, action_idx, state_idx2, peak):
    return ("{}: {};".format(genR(state_idx1, action_idx, state_idx2), -peak))

def upperboundR(state_idx1, action_idx, state_idx2, peak):
    return ("{}: {};".format(genR(state_idx1, action_idx, state_idx2), peak))

def lowerboundFRV_targ(state_idx1, action_idx, state_idx2, peak):
    return ("{}:{};".format(genFRV_targ(state_idx1, action_idx, state_idx2), -peak * 2))

def upperboundFRV_targ(state_idx1, action_idx, state_idx2, peak):
    return ("{}:{};".format(genFRV_targ(state_idx1, action_idx, state_idx2), peak * 2))

def lowerboundFRV_nontarg(state_idx1, action_idx, state_idx2, peak):
    return ("{}:{};".format(genFRV_nontarg(state_idx1, action_idx, state_idx2), -peak * 2))

def upperboundFRV_nontarg(state_idx1, action_idx, state_idx2, peak):
    return ("{}:{};".format(genFRV_nontarg(state_idx1, action_idx, state_idx2), peak * 2))

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

#   "LER_{}".format(a) = \sum_{b=0}^{length(a)-1} "P_[s_{} | a_{},s_{}]".format(i_{b+1}, j_b, i_b) * "R_[s_{},a_{},s_{}]".format(i_b, j_b, i_{b+1})
def constraintLER_name(traj_idx):
    desc = "_LER_{}".format(traj_idx)
    return (genEquationName(desc, "LER_"))

def constraintLER(traj_idx, trajectory, P):
    constraint = genLER(traj_idx)
    for state_pos in range(floor(len(trajectory) / 2)):
        constraint += " - {} * {}".format(P[(trajectory[(state_pos + 1) * 2], trajectory[state_pos * 2 + 1], trajectory[state_pos * 2])], genR(trajectory[state_pos * 2], trajectory[state_pos * 2 + 1], trajectory[(state_pos + 1) * 2]))
    constraint += " == 0;"
    return (constraint)

#   "R_[s_{},a_{},s_{}]".format(i, j, i') = peak * P_targ(a_j, s_i) + "FRV_targ_[s_{},a_{},s_{}]".format(i, j, i')
def constraintFRV_targ_name(state_idx1, action_idx, state_idx2):
    desc = "_FRV_TARG_{}_{}_{}".format(state_idx1, action_idx, state_idx2)
    return (genEquationName(desc, "FRV_TARG_"))

def constraintFRV_targ(state_idx1, action_idx, state_idx2, peak, p):
    return ("{} - {} == {};".format(genR(state_idx1, action_idx, state_idx2), genFRV_targ(state_idx1, action_idx, state_idx2), peak * p))

#   "FRV_targ_[s_{},a_{},s_{}]".format(i, j, i') <= "FRV_maxMagnitude"
def constraintFRV_targ_magnitude1_name(state_idx1, action_idx, state_idx2):
    desc = "_FRV_TARG_{}_{}_{}_maxMagnitude1".format(state_idx1, action_idx, state_idx2)
    return (genEquationName(desc, "FRV_TARG_maxMagnitude1_"))

def constraintFRV_targ_magnitude1(state_idx1, action_idx, state_idx2):
    return ("{} - {} <= 0;".format(genFRV_targ(state_idx1, action_idx, state_idx2), genFRV_maxMagnitude()))

#   -"FRV_targ_[s_{},a_{},s_{}]".format(i, j, i') <= "FRV_maxMagnitude"
def constraintFRV_targ_magnitude2_name(state_idx1, action_idx, state_idx2):
    desc = "_FRV_TARG_{}_{}_{}_maxMagnitude2".format(state_idx1, action_idx, state_idx2)
    return (genEquationName(desc, "FRV_TARG_maxMagnitude2_"))

def constraintFRV_targ_magnitude2(state_idx1, action_idx, state_idx2):
    return ("-{} - {} <= 0;".format(genFRV_targ(state_idx1, action_idx, state_idx2), genFRV_maxMagnitude()))

#   "R_[s_{},a_{},s_{}]".format(i, j, i') = -peak * P_nontarg(a_j, s_i) + "FRV_nontarg_[s_{},a_{},s_{}]".format(i, j, i')
def constraintFRV_nontarg_name(state_idx1, action_idx, state_idx2):
    desc = "_FRV_NONTARG_{}_{}_{}".format(state_idx1, action_idx, state_idx2)
    return (genEquationName(desc, "FRV_NONTARG_"))

def constraintFRV_nontarg(state_idx1, action_idx, state_idx2, peak, p):
    return ("{} - {} == {};".format(genR(state_idx1, action_idx, state_idx2), genFRV_nontarg(state_idx1, action_idx, state_idx2), -peak * p))

#   "FRV_nontarg_[s_{},a_{},s_{}]".format(i, j, i') <= "FRV_maxMagnitude"
def constraintFRV_nontarg_magnitude1_name(state_idx1, action_idx, state_idx2):
    desc = "_FRV_NONTARG_{}_{}_{}_maxMagnitude1".format(state_idx1, action_idx, state_idx2)
    return (genEquationName(desc, "FRV_NONTARG_maxMagnitude1_"))

def constraintFRV_nontarg_magnitude1(state_idx1, action_idx, state_idx2):
    return ("{} - {} <= 0;".format(genFRV_nontarg(state_idx1, action_idx, state_idx2), genFRV_maxMagnitude()))

#   -"FRV_nontarg_[s_{},a_{},s_{}]".format(i, j, i') <= "FRV_maxMagnitude"
def constraintFRV_nontarg_magnitude2_name(state_idx1, action_idx, state_idx2):
    desc = "_FRV_NONTARG_{}_{}_{}_maxMagnitude2".format(state_idx1, action_idx, state_idx2)
    return (genEquationName(desc, "FRV_NONTARG_maxMagnitude2_"))

def constraintFRV_nontarg_magnitude2(state_idx1, action_idx, state_idx2):
    return ("-{} - {} <= 0;".format(genFRV_nontarg(state_idx1, action_idx, state_idx2), genFRV_maxMagnitude()))

#   "LB_{}".format(u) <= UB_{}".format(u)
def constraintPosetLBUB_name(poset_idx):
    desc = "_POSETLBUB_{}".format(poset_idx)
    return (genEquationName(desc, "POSET"))

def constraintPosetLBUB(poset_idx):
    return ("{} - {} <= 0;".format(genLB(poset_idx), genUB(poset_idx)))

#   "LB_{}".format(q) = "UB_{}".format(q+1) + "delta_{}_{}".format(q, q+1)
def constraintPoset_name(poset_idx1, poset_idx2):
    desc = "_POSETS_{}_{}".format(poset_idx1, poset_idx2)
    return (genEquationName(desc, "POSETS_"))

def constraintPoset(poset_idx1, poset_idx2):
    return ("{} - {} - {} == 0;".format(genLB(poset_idx1), genUB(poset_idx2), gendelta(poset_idx1, poset_idx2)))

#   "LB_{}".format(q) <= "LER_{}".format(a)
def constraintPosetLB_name(poset_idx, traj_idx):
    desc = "_POSETLB_{}_{}".format(poset_idx, traj_idx)
    return (genEquationName(desc, "POSETLB_"))

def constraintPosetLB(poset_idx, traj_idx):
    return ("{} - {} <= 0;".format(genLB(poset_idx), genLER(traj_idx)))

#   "UB_{}".format(q) >= "LER_{}".format(a)
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

def declareVariables(num_states, num_actions, num_posets, trajectories, dm_trajectory_indices, others_trajectory_indices):
    # Returns a tuple with the first element a list of output strings each a
    # row and second element is a dictionary mapping each LER variable to the
    # unique associated trajectory.
    rows = list()
    binary_variables = set()
    integer_variables = set()
    positive_variables = set()
    variables = set()
    LER_to_trajectory = dict()

    for t_idx in range(len(trajectories)):
        traj = trajectories[t_idx]
        for pos in range(0, len(traj) - 1, 2):
            s_idx1 = traj[pos]
            a_idx = traj[pos + 1]
            s_idx2 = traj[pos + 2]
            addVariable(genR(s_idx1, a_idx, s_idx2), typeR(), binary_variables, integer_variables, positive_variables, variables)
            if t_idx in dm_trajectory_indices:
                addVariable(genFRV_targ(s_idx1, a_idx, s_idx2), typeFRV_targ(), binary_variables, integer_variables, positive_variables, variables)
            if t_idx in others_trajectory_indices:
                addVariable(genFRV_nontarg(s_idx1, a_idx, s_idx2), typeFRV_nontarg(), binary_variables, integer_variables, positive_variables, variables)

    addVariable(genFRV_maxMagnitude(), typeFRV_maxMagnitude(), binary_variables, integer_variables, positive_variables, variables)
    print("trajs = ", trajectories)

    for t_idx, traj in enumerate(trajectories):
        var_name = genLER(t_idx)
        print("    var_name = {}, t_idx = {}".format(var_name, t_idx))
        addVariable(var_name, typeLER(), binary_variables, integer_variables, positive_variables, variables)
        LER_to_trajectory[var_name] = tuple(traj)

    for p_idx in range(num_posets):
        addVariable(genLB(p_idx), typeLB(), binary_variables, integer_variables, positive_variables, variables)

    for p_idx in range(num_posets):
        addVariable(genUB(p_idx), typeUB(), binary_variables, integer_variables, positive_variables, variables)

    for p_idx in range(num_posets - 1):
        addVariable(gendelta(p_idx, p_idx + 1), typedelta(), binary_variables, integer_variables, positive_variables, variables)


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
    return (( rows, LER_to_trajectory ))

#
# Bound variables
def boundVariables(num_states, num_actions, peak, trajectories, num_posets, dm_trajectory_indices, others_trajectory_indices, optimizer):
    # Returns a list of rows detailing the bounds
    rows = list()
    if optimizer == "BARON":
        rows.append("LOWER_BOUNDS{")
        for t_idx in range(len(trajectories)):
            traj = trajectories[t_idx]
            for pos in range(0, len(traj) - 1, 2):
                s_idx1 = traj[pos]
                a_idx = traj[pos + 1]
                s_idx2 = traj[pos + 2]
                rows.append(lowerboundR(s_idx1, a_idx, s_idx2, peak))
                if t_idx in dm_trajectory_indices:
                    rows.append(lowerboundFRV_targ(s_idx1, a_idx, s_idx2, peak))
                if t_idx in others_trajectory_indices:
                    rows.append(lowerboundFRV_nontarg(s_idx1, a_idx, s_idx2, peak))
        for p_idx in range(num_posets - 1):
            rows.append(lowerbounddelta(p_idx, p_idx + 1))
        rows.append("}")
        rows.append(" ")

        rows.append("UPPER_BOUNDS{")
        for t_idx in range(len(trajectories)):
            traj = trajectories[t_idx]
            for pos in range(0, len(traj) - 1, 2):
                s_idx1 = traj[pos]
                a_idx = traj[pos + 1]
                s_idx2 = traj[pos + 2]
                rows.append(upperboundR(s_idx1, a_idx, s_idx2, peak))
                if t_idx in dm_trajectory_indices:
                    rows.append(upperboundFRV_targ(s_idx1, a_idx, s_idx2, peak))
                if t_idx in others_trajectory_indices:
                    rows.append(upperboundFRV_nontarg(s_idx1, a_idx, s_idx2, peak))
        rows.append("}")
    if optimizer == "CPLEX":
        rows.append("Bounds")
        # POSITIVE VARIABLES have a default of non-negative in CPLEX
        # Need to take care of "VARIABLE" if not yet bound using "-infinity"
        for t_idx in range(len(trajectories)):
            traj = trajectories[t_idx]
            for pos in range(0, len(traj) - 1, 2):
                s_idx1 = traj[pos]
                a_idx = traj[pos + 1]
                s_idx2 = traj[pos + 2]
                lb = lowerboundR(s_idx1, a_idx, s_idx2, peak)
                ub = upperboundR(s_idx1, a_idx, s_idx2, peak)
                idx = lb.find(":")
                n = lb[:idx]
                lv = lb[idx+1:-1]
                uv = ub[idx+1:-1]
                rows.append("{} <= {} <= {}".format(lv, n, uv))
                if t_idx in dm_trajectory_indices:
                    lb = lowerboundFRV_targ(s_idx1, a_idx, s_idx2, peak)
                    ub = upperboundFRV_targ(s_idx1, a_idx, s_idx2, peak)
                    idx = lb.find(":")
                    n = lb[:idx]
                    lv = lb[idx+1:-1]
                    uv = ub[idx+1:-1]
                    rows.append("{} <= {} <= {}".format(lv, n, uv))
                if t_idx in others_trajectory_indices:
                    lb = lowerboundFRV_nontarg(s_idx1, a_idx, s_idx2, peak)
                    ub = upperboundFRV_nontarg(s_idx1, a_idx, s_idx2, peak)
                    idx = lb.find(":")
                    n = lb[:idx]
                    lv = lb[idx+1:-1]
                    uv = ub[idx+1:-1]
                    rows.append("{} <= {} <= {}".format(lv, n, uv))
        for p_idx in range(num_posets - 1):
            lb = lowerbounddelta(p_idx, p_idx + 1)
            idx = lb.find(":")
            n = lb[:idx]
            lv = lb[idx+1:-1]
            rows.append("{} <= {}".format(lv, n))

        for t_idx, traj in enumerate(trajectories):
            rows.append("-infinity <= {}".format(genLER(t_idx)))

        for p_idx in range(num_posets):
            rows.append("-infinity <= {}".format(genLB(p_idx)))
            rows.append("-infinity <= {}".format(genUB(p_idx)))
        rows.append("End")
    return (rows)

#
# Build constraints
def generateConstraints(num_states, num_actions, peak, posets, trajectories, P, dm_trajectory_indices, others_trajectory_indices, P_targ, P_nontarg, optimizer):
    # Returns a list of rows detailing the constraint declarations and constraints -- specific to (BARON)
    equation_names = list()
    equation_names_set = set()
    equations = list()

    if DEBUG:
        print("...LER constraints...")
    num_constraints = 0
    for t_idx in range(len(trajectories)):
        eqn = constraintLER_name(t_idx)
        if not eqn in equation_names_set:
            equation_names_set.add(eqn)
            equation_names.append(eqn)
            equations.append(constraintLER(t_idx, trajectories[t_idx], P))
            num_constraints += 1
    if DEBUG:
        print ("......{} created.".format(num_constraints))

    if DEBUG:
        print ("...FRV_targ/FRV_nontarg constraints...")

    num_constraints = 0
    for t_idx in range(len(trajectories)):
        traj = trajectories[t_idx]
        for pos in range(0, len(traj) - 1, 2):
            s_idx1 = traj[pos]
            a_idx = traj[pos + 1]
            s_idx2 = traj[pos + 2]
            if t_idx in dm_trajectory_indices:
                eqn = constraintFRV_targ_name(s_idx1, a_idx, s_idx2)
                if not eqn in equation_names_set:
                    equation_names_set.add(eqn)
                    equation_names.append(eqn)
                    equations.append(constraintFRV_targ(s_idx1, a_idx, s_idx2, peak, P_targ[(a_idx, s_idx1)]))
                    num_constraints += 1
                eqn = constraintFRV_targ_magnitude1_name(s_idx1, a_idx, s_idx2)
                if not eqn in equation_names_set:
                    equation_names_set.add(eqn)
                    equation_names.append(eqn)
                    equations.append(constraintFRV_targ_magnitude1(s_idx1, a_idx, s_idx2))
                    num_constraints += 1
                eqn = constraintFRV_targ_magnitude2_name(s_idx1, a_idx, s_idx2)
                if not eqn in equation_names_set:
                    equation_names_set.add(eqn)
                    equation_names.append(eqn)
                    equations.append(constraintFRV_targ_magnitude2(s_idx1, a_idx, s_idx2))
                    num_constraints += 1
            if t_idx in others_trajectory_indices:
                eqn = constraintFRV_nontarg_name(s_idx1, a_idx, s_idx2)
                if not eqn in equation_names_set:
                    equation_names_set.add(eqn)
                    equation_names.append(eqn)
                    equations.append(constraintFRV_nontarg(s_idx1, a_idx, s_idx2, peak, P_nontarg[(a_idx, s_idx1)]))
                    num_constraints += 1
                eqn = constraintFRV_nontarg_magnitude1_name(s_idx1, a_idx, s_idx2)
                if not eqn in equation_names_set:
                    equation_names_set.add(eqn)
                    equation_names.append(eqn)
                    equations.append(constraintFRV_nontarg_magnitude1(s_idx1, a_idx, s_idx2))
                    num_constraints += 1
                eqn = constraintFRV_nontarg_magnitude2_name(s_idx1, a_idx, s_idx2)
                if not eqn in equation_names_set:
                    equation_names_set.add(eqn)
                    equation_names.append(eqn)
                    equations.append(constraintFRV_nontarg_magnitude2(s_idx1, a_idx, s_idx2))
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

    for p_idx in range(len(posets) - 1):
        eqn = constraintPoset_name(p_idx, p_idx + 1)
        if not eqn in equation_names_set:
            equation_names_set.add(eqn)
            equation_names.append(eqn)
            equations.append(constraintPoset(p_idx, p_idx + 1))
            num_constraints += 1

    for p_idx, poset in enumerate(posets):
        for t_idx in poset:
            eqn = constraintPosetLB_name(p_idx, t_idx)
            if not eqn in equation_names_set:
                equation_names_set.add(eqn)
                equation_names.append(eqn)
                equations.append(constraintPosetLB(p_idx, t_idx))
                num_constraints += 1

    for p_idx, poset in enumerate(posets):
        for t_idx in poset:
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

    # Generate for BARON

    rows = list()
    if len(equation_names) == 0:
        return (rows)
    if optimizer == "BARON":
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

def genObjective(num_posets):
#       max \sum_{q=0}{num_posets-1} "delta_{}_{}".format(q, q+1)
#       -- maximizes the distance between posets
#       -- can overfit trajectories
    rows = list()
#    rows.append("OBJ: maximize")
#    obj = ""
#    for p_idx in range(num_posets - 1):
#        if p_idx > 0:
#            obj += " + "
#        obj += gendelta(p_idx, p_idx + 1)
#    rows.append("\t{};".format(obj))
    rows.append("OBJ: minimize")
    rows.append("\t{};".format(genFRV_maxMagnitude()))
    return (rows)

def computePs(trajectories, dm_trajectory_indices, others_trajectory_indices):
    # Computes the world transition probability from the trajectories and creating a dictionary P[(s_idx2, a_idx, s_idx1)]
    # Computes the decision-maker bias for the target from the dm_trajectories and creating a dictionary P_targ[(a_idx, s_idx1)]
    # Computer the decision-maker bias for others from the trajectories not in dm_trajectories and creating a dictionary P_nontarg[(a_idx, s_idx)]
    # Returns tuple (P, P_targ, P_nontarg, C2, C3, C1_targ, C2_targ, C1_nontarg,
    #    C2_nontarg)
    P = dict()
    C3 = dict()
    C2 = dict()
    P_targ = dict()
    P_nontarg = dict()
    C2_targ = dict()
    C2_nontarg = dict()
    C1_targ = dict()
    C1_nontarg = dict()
    for t_idx, traj in enumerate(trajectories):
        for pos in range(0, len(traj) - 1, 2):
            s_idx1 = traj[pos]
            a_idx = traj[pos + 1]
            s_idx2 = traj[pos + 2]
            try:
                C2[(s_idx1, a_idx)] += 1
            except KeyError:
                C2[(s_idx1, a_idx)] = 1
            try:
                C3[(s_idx1, a_idx, s_idx2)] += 1
            except KeyError:
                C3[(s_idx1, a_idx, s_idx2)] = 1
            if t_idx in dm_trajectory_indices:
                try:
                    C1_targ[s_idx1] += 1
                except KeyError:
                    C1_targ[s_idx1] = 1
                try:
                    C2_targ[(s_idx1, a_idx)] += 1
                except KeyError:
                    C2_targ[(s_idx1, a_idx)] = 1
            if t_idx in others_trajectory_indices:
                try:
                    C1_nontarg[s_idx1] += 1
                except KeyError:
                    C1_nontarg[s_idx1] = 1
                try:
                    C2_nontarg[(s_idx1, a_idx)] += 1
                except KeyError:
                    C2_nontarg[(s_idx1, a_idx)] = 1
    for key, value in C3.items():
        (s_idx1, a_idx, s_idx2) = key
        P[( s_idx2, a_idx, s_idx1 )] = C3[key] / C2[(s_idx1, a_idx)] # Note s_idx2 is first coordinate
    for key, value in C2_targ.items():
        ( s_idx1, a_idx ) = key
        P_targ[( a_idx, s_idx1 )] = C2_targ[key] / C1_targ[s_idx1] # Note a_idx is first coordinate
    for key, value in C2_nontarg.items():
        ( s_idx1, a_idx ) = key
        P_nontarg[( a_idx, s_idx1 )] = C2_nontarg[key] / C1_nontarg[s_idx1] # Note a_idx is first coordinate
    return (P, P_targ, P_nontarg, C2, C3, C1_targ, C2_targ, C1_nontarg, C2_nontarg)

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

    print(kwargs)

    num_states = kwargs["num_states"]
    num_actions = kwargs["num_actions"]
    tau = kwargs["tau"]
    peak = kwargs["peak"]
    num_posets = kwargs["num_posets"]
    posets = kwargs["posets"]
    trajectories = kwargs["all_trajectories"]
    dm_trajectory_indices = kwargs["dm_trajectory_indices"]
    others_trajectory_indices = kwargs["others_trajectory_indices"]
    optimizer = kwargs["optimizer"]
    maxtime = kwargs["maxtime"]
    license_fn = kwargs["license_fn"]

# Builds and runs the optimization problem.
#   dm_trajectory_indices are the trajectories to be used for P_{DM}. If None, assumes all trajectories are DM trajectories.
    #   optimizer is "BARON" or "CPLEX" currently
    #   maxtime is maximum time allowed for BARON -- not applicable to CPLEX so set to None
    #   license_fn is the full path filename to the BARON license -- not applicablt to CPLEX so set to None
    #    tau is ignored
    # Returns a tuple where first element is the rows of the constraint problem and the second element is a dictionary mapping LER variable names to trajectories. The next three elements are the computed probabilities. The last element is a copy of the var_to_var_desc.

    if DEBUG:
        print ("Building RoMDP optimization problem...")

    if dm_trajectory_indices == None:
        dm_trajectory_indices = [ idx for idx in range(len(trajectories))]
    if others_trajectory_indices == None:
        others_trajectory_indices = list()
    dm_trajectory_indices = set(dm_trajectory_indices)
    others_trajectory_indices = set(others_trajectory_indices)
    if DEBUG:
        start_time = time.time()
        print ("Computing state/action frequencies and probabilities...")
    P, P_targ, P_nontarg, _, _, _, _, _, _  = computePs(trajectories, dm_trajectory_indices, others_trajectory_indices)
    if DEBUG:
        print ("\t...elapsed time = {}".format(time.time() - start_time))

    if optimizer == "CPLEX": # Generate problem in CPLEX format
        if DEBUG:
            start_time = time.time()
            print ("Declaring variables...")
        ( _, LER_to_trajectories ) = declareVariables(num_states, num_actions, num_posets, trajectories, dm_trajectory_indices, others_trajectory_indices)
        rows = list()
        if DEBUG:
            print ("\t...elapsed time = {}".format(time.time() - start_time))
        newrows = genObjective(num_posets)
        print("     newrows objective = ", newrows)
        for row in newrows:
            row = row.replace("OBJ: ", "")
            row = row.replace("*", "")
            row = row.replace(";", "")
            rows.append(row)
        if DEBUG:
            start_time = time.time()
            print ("Generating constraints...")
        newrows = generateConstraints(num_states, num_actions, peak, posets, trajectories, P, dm_trajectory_indices, others_trajectory_indices, P_targ, P_nontarg, optimizer)
        print("  aaaa  new rows = ", newrows)

        if len(newrows) > 0:
            rows.append("Subject To")
        for row in newrows:
            row = row.replace("==", "=")
            row = row.replace(";", "")
            row = row.replace("*", "")
            rows.append(row)
        if DEBUG:
            print ("\t...elapsed time = {}".format(time.time() - start_time))
            start_time = time.time()
            print ("Constructing variable bounds...")
        newrows = boundVariables(num_states, num_actions, peak, trajectories, num_posets, dm_trajectory_indices, others_trajectory_indices, optimizer)
        rows.extend(newrows)
        if DEBUG:
            print ("\t...elapsed time = {}".format(time.time() - start_time))

    return (( rows, LER_to_trajectories, P, P_targ, P_nontarg, copy.deepcopy(var_to_var_desc) ))


def test3():
#   1   2   3   4   5
#   6   7   8   9   10
#   11  12  13  14  15
#   16  17  18  19  20
#   21  22  23  24  25

    t1 = [ 0, 1, 2, 0, 2]
    t2 = [ 0, 1, 2, 1, 2, 1, 1]

    t3 = [ 0, 0, 2]
    t4 = [ 0, 0, 1, 0, 2]

    num_posets = 2
    posets = [[0, 1], [2, 3]]
    num_states = 3
    num_actions = 2
    peak = 100
    t = [t1, t2, t3, t4]

    rows, LER_to_trajectories, P, P_targ, P_nontarg, var_to_var_desc = constructOptimization(num_states = num_states, \
        num_actions = num_actions, tau = None, peak = peak, \
        num_posets = num_posets, posets = posets, all_trajectories = t, \
        dm_trajectory_indices = posets[0], others_trajectory_indices = posets[1], \
        optimizer = "CPLEX", maxtime = 100000, license_fn = "./baronlice.txt")

    for r in rows:
        print(r)

    print("\n var to var desc")
    print(var_to_var_desc)
    print("\n LER_to_trajs")
    print(LER_to_trajectories)
    print("\n P = ")
    print(P)
    print("\n P_targ = ")
    print(P_targ)
    print("\n P_nontarg")
    print(P_nontarg)

    with open("tmp3-RoMDP.mnlp", "w") as f:
        for row in rows:
            f.write(row + '\n')
    os.system('baron tmp3-RoMDP.mnlp')
    answers = L_MDP.extractAnswers("soln3-RoMDP.csv", var_to_var_desc)
