#
# Filename:     L_MDP.py
# Date:         2020-05-18
# Project:      Learning MDP
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
import FoMDP
import FoMDP_2
import parse
import tqdm
import gzip
import compress_pickle
import cplex
import networkx
#import gurobipy
#from gurobipy import GRB

DEBUG = True

#
# The following are different scenarios for learning MDPs. The general form
#   takes: action-space, state-space, trajectories, and poset specifications
#
# learnMDP(...) - DTM based problem formulation -- loads in from files
# learnMDPabs(...) - all trajectories already converted to integer indices
#   representing states and actions.
#
#   A directory of reports and results files are generated.
#       Note - not all files may be generated or listed here
#           Only relevant files used explictly
#
#       "{}-{}.lp".format(constructDescriptor, optimizer)
#           LP formulation file - format specific to BARON or CPLEX
#           {CPLEX, BARON} for RoMDP
#           Note -- if CPLEX is selected for RoMDP, the docplex API is
#           used directly bipassing the need for test file inputes and
#           CPLEX text file logs and outputs
#       "{}-{}.minp".format(constructDescriptor, optimizer)
#           MINP formulation file - format specific to BARON
#           {BARON} for SsMDP
#       "{}_soln-{}.csv.gz".format(constructDescriptor, optimizer)
#           Optimization formulation variable solutions file
#           {CPLEX, BARON, GUROBI}
#       "{}_mappings-{}.csv.gz".format(constructDescriptor, optimizer)
#           Report of trajectory mappings and solutions
#           {CPLEX, BARON, GUROBI}
#       "{}_probabilities-{}.csv.gz".format(constructDescriptor, optimizer
#           Report of probability mappings for P, P_targ, P_nontarg
#           {CPLEX, BARON, GUROBI}
#       "{}_rewards-{}.csv.gz".format(constructionDescriptor, optimier
#           Report of the reward and fractional reward values including
#           class based rewards for FoMDP
#           {CPLEX, BARON, GUROBI}
#       "{}-{}.sol".format(constructDescriptor, optimizer)
#           CPLEX specific solution file
#           {CPLEX}
#       "{}-{}.cplex".format(constructDescriptor, optimizer)
#           CPLEX execution script
#           {CPLEX}
#       "res.lst"
#           BARON specific solution file
#           {BARON}
#
def learnMDP(action_spec_csvfilename, state_spec_csvfilename, \
    target_traj_list_fn, target_nbr_traj_list_fn, other_nbr_traj_list_fn, \
    other_traj_list_fn, descriptor, tau, peak, constructOptimization, \
    constructDescriptor, optimizer, exe_fn, maxtime, license_fn, num_classes, \
    simplify_masking, feasibility_only):

    #   descriptor is a string descriptor used to create a directory in the current working directory which will
    #       contain the results files -- descriptor must be appropriate for a directory name
    #   see constructOptimization below
    #   constructDescriptor is a string to describe the method of
    #       connstruction such as "RoMDP", "SsMDP", "FoMDP", "FoMDP_2"
    #       it affects file names
    #
    #   optimizer is "BARON", "CPLEX", or "GUROBI" currently
    #       Note that SsMDP can only use BARON or GUROBI
    #        Note that FoMDP is the only user of GUROBI
    #   exe_fn is the full path filename to the matching executable/binary for the optimizer
    #   maxtime is maximum time allowed for BARON -- not applicable to CPLEX so set to None
    #   license_fn is the full path filename to the BARON license -- not applicablt to CPLEX so set to None
    #   num_classes only applies to FoMDP and FoMDP_2
    #    simplify_masking only applies to FoMDP and FoMDP_2
    #    feasbility_only only applies to FoMDP and FoMDP_2
    #
    # Returns the following:
    #
    #   isfeasible, objective_value, answers, P, P_targ, P_nontarg,
    #       LER_to_trajectories, var_to_var_desc, traj_mappings, action_specs,
    #       actions, state_specs, states, target_trajectories,
    #       target_nbr_trajectories, other_nbr_trajectories, other_trajectories
    #       posets, poset_names, actions_values, onehot_actions_encodings,
    #       states_values, onehot_states_encodings
    #
    #   where:
    #       isfeasible (bool) - True if feasible solution found
    #           - Difference from optimality since BARON may not return optimal
    #       objective_value (float) - The objective value of returned solution
    #           - Could also return None if not feasible or other condition
    #       answers (dict) - Maps all variables in formulation to answer or None
    #       P, P_targ, P_nontarg - None if not applicable.
    #       LER_to_trajectories - None if not applicable
    #        var_to_var_desc - maps vars in answers to descriptors
    #           - None if not applicable
    #       traj_mappings is a dictionary mapping each trajectory found to a
    #           set of trajectory filenames - None if not applicable
    #       action_specs and state_specs are lists of strings that need to be
    #           found in the header to build actions and states
    #           - None if not applicable
    #       actions is a dictionary mapping action tuple attributes to an index
    #           - None if not applicable
    #       states is a diectionary mapping state tuple attributes to an index
    #           - None if not applicable
    #       latter 4 are lists of trajectories in indices form
    #       posets is updated based on making the 4 sets disjoint
    #       poset_names is [ 'Target', 'Target-nbr', 'Other-nbr', 'Other' ]
    #       actions_values, onehot_actions_encodings, states_values,
    #           onehot_states_encodings -- see convertTrajectory
    #
    #
    #   This is the central function that formulates the learning problem.
    #   constructOptimization is a function variable specified as follows:
    #       Returns:
    #           ( rows, LER_to_trajectories, P, P_targ, P_nontarg,
    #               var_to_var_desc )
    #
    #       constructOptimization(...) -- must used named arguments
    #           Not all arguments need to be specified and overspecifying
    #           should be valid. Not all arguments are applicable also.
    #
    #           num_states: # of states
    #           states: a dictionary mapping a state (described as a tuple)
    #               to a state index
    #           num_actions: # of actions
    #           actions: a dictionary mapping an action (described as a tuple)
    #               to an action index
    #           tau: upper bound on number of hidden states
    #           peak: positive constant bounding reward values
    #           num_posets: # of posets
    #           posets: list of lists of trajectory indices for training
    #               Note -- posets is simply linear in nature
    #           all_trajectories: a list consisting of all trajectories
    #               Note -- the order in this list corresponds to a trajectory's
    #                   index
    #           dm_trajectory_indices: a list of trajectory indices that
    #               represent the decision-maker
    #               -- indices correspond to those in all_trajectories
    #           others_trajectory_indices: a list of trajectory indices that
    #               represent the other decision-makers
    #               -- indices correspond to those in all_trajectories
    #               Note -- trajectories identified in the others sets
    #                   that are found in the dm sets are still recorded
    #                   though they are only considered in the highest
    #                   ordered poset.
    #           all_triples: the set of unique (s, a, s') contiguous
    #               subsequences that occurs in all_trajectories
    #           dm_triples: the set of unique (s, a, s') contiguous
    #               subsequences that occurs in dm_trajectories
    #           others_triples: the set of unique (s, a, s') contiguous
    #               subsequences that occurs in others_trajectories
    #
    #           optimizer: "CPLEX" or "BARON"
    #               Note -- CPLEX will not run on some L_MDPs but
    #                   BARON can solve all
    #           maxtime: integer in seconds that provides a time bound for
    #               BARON
    #           license_fn: An absolute path filename for the BARON license
    #           num_classes: how many state clusters/equivalence classes
    #               to be considered (FoMDP) and assumed 1HOT encoding.

    master_start_time = time.time()

    if constructDescriptor == "FoMDP" or constructDescriptor == "FoMDP_2":
        onehot = not simplify_masking
    else:
        onehot = False

    # Convert to absolute paths
    action_spec_csvfilename = os.path.abspath(action_spec_csvfilename)
    state_spec_csvfilename = os.path.abspath(state_spec_csvfilename)
    target_traj_list_fn = os.path.abspath(target_traj_list_fn)
    if target_nbr_traj_list_fn != None:
        target_nbr_traj_list_fn = os.path.abspath(target_nbr_traj_list_fn)
    if other_nbr_traj_list_fn != None:
        other_nbr_traj_list_fn = os.path.abspath(other_nbr_traj_list_fn)
    if other_traj_list_fn != None:
        other_traj_list_fn = os.path.abspath(other_traj_list_fn)

    # Load in trajectories.
    if DEBUG:
        start_time = time.time()
        print ("Loading trajectories...")

    (action_specs, actions, state_specs, states, target_trajs, \
    target_nbr_trajs, other_nbr_trajs, other_trajs, traj_mappings, \
    actions_values, onehot_actions_encodings, states_values, \
    onehot_states_encodings) = convertTrajectories(action_spec_csvfilename, \
        state_spec_csvfilename, target_traj_list_fn, target_nbr_traj_list_fn, \
        other_nbr_traj_list_fn, other_traj_list_fn, onehot)

    if DEBUG:
        print ("...completed...elapsed time = {}".format(time.time() - start_time))

    if len(target_trajs) == 0:
        sys.exit("learnMDP(...) -- No target trajectories found!")

    if DEBUG:
        print("Total # of trajectories loaded = {}".format(len(target_trajs) + \
            len(target_nbr_trajs) + len(other_nbr_trajs) + len(other_trajs)))
        print("\t# of target trajectories = {}".format(len(target_trajs)))
        print("\t# of target nbr trajectories = {}".format(len(target_nbr_trajs)))
        print("\t# of other nbr trajectories = {}".format(len(other_nbr_trajs)))
        print("\t# of other trajectories = {}".format(len(other_trajs)))
        print("")

    # Make trajectory lists disjoint
    if DEBUG:
        start_time = time.time()
        print ("Making posets disjoint...")

    (preserved_trajectories_indices, new_trajectory_lists, \
    trajectory_to_location) = disjoinTrajectorySets([ target_trajs, \
            target_nbr_trajs, other_nbr_trajs, other_trajs ])

    if DEBUG:
        print ("...completed...elapsed time = {}".format(time.time() - start_time))

    u_target_trajs = new_trajectory_lists[0]
    u_target_nbr_trajs = new_trajectory_lists[1]
    u_other_nbr_trajs = new_trajectory_lists[2]
    u_other_trajs = new_trajectory_lists[3]
    if DEBUG:
        total_num_trajs = len(u_target_trajs) + len(u_target_nbr_trajs) + \
            len(u_other_nbr_trajs) + len(u_other_trajs)
        print ("Total # of trajs remaining = {}".format(len(u_target_trajs) + \
            len(u_target_nbr_trajs) + len(u_other_nbr_trajs) + len(u_other_trajs)))
        print ("\t# of target trajectories = {}".format(len(u_target_trajs)))
        print ("\t# of target nbr trajectories = {}".format(len(u_target_nbr_trajs)))
        print ("\t# of other nbr trajectories = {}".format(len(u_other_nbr_trajs)))
        print ("\t# of other trajectories = {}".format(len(u_other_trajs)))
        print ("")
        total_min_length = None
        total_max_length = None
        total_avg_length = 0

        for ( l_name, trajs ) in zip([ 'Target Trajectories', \
                'Target Nbr Trajectories', 'Other Nbr Trajectories', \
                'Other Trajectories' ], new_trajectory_lists):

            min_length = None
            max_length = None
            avg_length = 0
            for traj in trajs:
                if min_length == None:
                    min_length = len(traj)
                else:
                    if min_length > len(traj):
                        min_length = len(traj)
                if max_length == None:
                    max_length = len(traj)
                else:
                    if max_length < len(traj):
                        max_length = len(traj)
                avg_length += len(traj)
            total_avg_length += avg_length
            if min_length != None:
                if total_min_length == None:
                    total_min_length = min_length
                else:
                    if total_min_length > min_length:
                        total_min_length = min_length
            if max_length != None:
                if total_max_length == None:
                    total_max_length = max_length
                else:
                    if total_max_length < max_length:
                        total_max_length = max_length
            if len(trajs) == 0:
                avg_length = None
            else:
                avg_length = float(avg_length) / float(len(trajs))

            print ("{} - min length = {} - max length = {} - \
                average length = {}".format(l_name, min_length, max_length, \
                    avg_length))

        if total_num_trajs == 0:
            total_avg_length = None
        else:
            total_avg_length = float(total_avg_length) / float(total_num_trajs)

        print ("Overall - min length = {} - max length = {} - \
            average length = {}".format(total_min_length, total_max_length, \
                total_avg_length))


    num_posets = 0
    posets = list()
    poset = [ idx for idx in range(len(u_target_trajs)) ]
    posets.append(poset)
    num_posets += 1
    dm_trajectory_indices = copy.deepcopy(poset) # Without deepcopy, dm_trajectory simply hashes to the same memory as poset
    all_trajs = copy.deepcopy(u_target_trajs)
    poset = [ idx + len(all_trajs) for idx in range(len(u_target_nbr_trajs)) ]
    posets.append(poset)
    num_posets += 1
    all_trajs.extend(u_target_nbr_trajs)
    dm_trajectory_indices.extend(poset)
    poset = [ idx + len(all_trajs) for idx in range(len(u_other_nbr_trajs)) ]
    posets.append(poset)
    num_posets += 1
    all_trajs.extend(u_other_nbr_trajs)
    poset = [ idx + len(all_trajs) for idx in range(len(u_other_trajs)) ]
    posets.append(poset)
    num_posets += 1
    all_trajs.extend(u_other_trajs)
    trajectory_position = dict()
    for t_idx, traj in enumerate(all_trajs):
        trajectory_position[tuple(traj)] = t_idx

    others_trajectory_indices = list(set([ trajectory_position[tuple(traj)] \
        for traj in other_nbr_trajs ]))
    others_trajectory_indices.extend(list(set([ trajectory_position[tuple(traj)] \
        for traj in other_trajs ])))

    num_states = len(states)
    num_actions = len(actions)

    print ("Elapsed time for processing trajectories = " + str(time.time() - \
        master_start_time))

    if not trajectoriesFeasible(new_trajectory_lists,  [ 'Target', \
        'Target-nbr', 'Other-nbr', 'Other' ], traj_mappings):

        return (False, None, None, None, None, None, None, None, traj_mappings, \
            action_specs, actions, state_specs, states, u_target_trajs, \
            u_target_nbr_trajs, u_other_nbr_trajs, u_other_trajs, posets, \
            [ 'Target', 'Target-nbr', 'Other-nbr', 'Other' ], actions_values, \
            onehot_actions_encodings, states_values, onehot_states_encodings)

    (isfeasible, objective_value, answers, P, P_targ, P_nontarg, \
        LER_to_trajectories, var_to_var_desc ) = _learnMDP(descriptor, \
            num_states, num_actions, tau, peak, num_posets, posets, all_trajs, \
            dm_trajectory_indices, others_trajectory_indices, \
            constructOptimization, constructDescriptor, optimizer, exe_fn, \
            maxtime, license_fn, state_specs, states, action_specs, actions, \
            traj_mappings, actions_values, states_values, num_classes, onehot, \
            onehot_actions_encodings, onehot_states_encodings, \
            simplify_masking, feasibility_only)

    print ("Overall time for solving problem = {}".format(time.time() - \
        master_start_time))

    return (isfeasible, objective_value, answers, P, P_targ, P_nontarg, \
        LER_to_trajectories, var_to_var_desc, traj_mappings, action_specs, \
        actions, state_specs, states, u_target_trajs, u_target_nbr_trajs, \
        u_other_nbr_trajs, u_other_trajs, posets, [ 'Target', 'Target-nbr', \
        'Other-nbr', 'Other' ], actions_values, onehot_actions_encodings, \
        states_values, onehot_states_encodings)

#
# Abstract version using indices
def learnMDPabs(num_actions, num_states, descriptor, tau, peak, num_posets, \
    posets, all_trajectories, dm_trajectory_indices, others_trajectory_indices, \
    constructOptimization, constructDescriptor, optimizer, exe_fn, maxtime, \
    license_fn):

    ( isfeasible, objective_value, answers, P, P_targ, P_nontarg, \
    LER_to_trajectories, var_to_var_desc ) = _learnMDP(descriptor, num_states, \
    num_actions, tau, peak, num_posets, posets, all_trajectories, \
    dm_trajectory_indices, others_trajectory_indices, constructOptimization, \
    constructDescriptor, optimizer, exe_fn, maxtime, license_fn, None, None, \
    None, None, None, None, None, None, None, None, False, None, None)

    return (isfeasible, objective_value, answers, P, P_targ, P_nontarg, \
        LER_to_trajectories, var_to_var_desc)


# This is a shared function that builds the constraint system, optimizes,
#   and extracts the answers.
def _learnMDP(descriptor, num_states, num_actions, tau, peak, num_posets, \
        posets, all_trajectories, dm_trajectory_indices, \
        others_trajectory_indices, constructOptimization, constructDescriptor, \
        optimizer, exe_fn, maxtime, license_fn, state_specs, states, \
        action_specs, actions, traj_mappings, actions_values, states_values, \
        num_classes, onehot, onehot_actions_encodings, onehot_states_encodings, \
        simplify_masking, feasibility_only):

    master_start_time = time.time()

    # Convert to absolute paths
    if optimizer != 'GUROBI':
        exe_fn = os.path.abspath(exe_fn)
    if license_fn != None:
        license_fn = os.path.abspath(license_fn)

    # Make a results directory
    results_dir = "{}-{}-{}".format(descriptor, constructDescriptor, optimizer)
    print ("Creating directory for results {}...".format(results_dir))
    if os.path.exists(results_dir):
        if not os.path.isdir(results_dir):
            sys.exit("learnMDP(...) -- {} is an existing file, \
                cannot create directory.".format(results_dir))
        print ("Using existing directory {} -- will overwrite \
            files.".format(results_dir))
    else:
        os.mkdir(results_dir)
    cwd = os.getcwd()
    os.chdir(results_dir)

    if DEBUG:
        start_time = time.time()
        print ("Extracting (s, a, s') from trajectories...")
    all_triples = extractTriples(all_trajectories)
    dm_triples = extractTriples([ all_trajectories[d_idx] for d_idx in \
        dm_trajectory_indices ])
    others_triples = extractTriples([ all_trajectories[d_idx] for d_idx in \
        others_trajectory_indices ])
    if DEBUG:
        print ("...triples found = {}".format(len(all_triples)))
        print ("...DM triples found = {}".format(len(dm_triples)))
        print ("...Other DMs triples found = {}".format(len(others_triples)))
        print ("\t...elapsed time = {}".format(time.time() - start_time))

    # Build optimization problem
    if constructOptimization == RoMDP.constructOptimization and optimizer == "CPLEX":
        cplex_model = cplex.Cplex()
    else:
        cplex_model = None

    opt_res = constructOptimization(num_states=num_states, states=states, \
        num_actions=num_actions, actions=actions, tau=tau, peak=peak, \
        num_posets=num_posets, posets=posets, \
        all_trajectories=all_trajectories, \
        dm_trajectory_indices=dm_trajectory_indices, \
        others_trajectory_indices=others_trajectory_indices, \
        all_triples=all_triples, dm_triples=dm_triples, \
        others_triples=others_triples, optimizer=optimizer, maxtime=maxtime, \
        license_fn=license_fn, num_classes = num_classes, \
        simplify_masking = simplify_masking, cplex_model = cplex_model, \
        feasibility_only = feasibility_only)


    if constructOptimization == RoMDP.constructOptimization:

        if cplex_model == None:
            opt_file = "{}-{}.lp".format(constructDescriptor, optimizer)
        else:
            opt_file = None

        (rows, LER_to_trajectories, P, P_targ, P_nontarg, var_to_var_desc, \
            cplex_variables) = opt_res

        """
        print("rows")
        print("\n", rows)
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
        print("\n cplex =")
        print(cplex_variables)
        print("optfile = ", opt_file)
        """

    else:

        (rows, LER_to_trajectories, P, P_targ, P_nontarg, var_to_var_desc) = \
            opt_res

    if opt_file != None:

        if optimizer == 'GUROBI':
            GUROBI_MODEL.write("{}-{}.lp.gz".format(constructDescriptor, optimizer))
        else:
            with open (opt_file, "w") as f:
                f.write("\n".join(rows))
                f.close()

    # Output table of variables used in optimizer paired with actual variable descriptor
    var_map = list()
    for var, var_desc in var_to_var_desc.items():
        var_map.append([ var, var_desc ])
    with gzip.open("{}_vars_to_desc-{}.csv.gz".format(constructDescriptor, optimizer), "wt", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(var_map)
        csv_file.close()


    # Run CPLEX to optimize and extract answers
    if optimizer == "CPLEX":
        if cplex_model != None:
            if DEBUG:
                start_time = time.time()
                print ("Solving with CPLEX...")
            cplex_model.solve()
            if DEBUG:
                print ("\t...elapsed time = {}".format(time.time() - start_time))

            isfeasible, objective_value, answers = \
                extractAnswersCPLEXModel(cplex_model, \
                "{}_soln-{}.csv.gz".format(constructDescriptor, optimizer), \
                var_to_var_desc, cplex_variables)
        else:
            os.system("rm {}-{}.sol".format(constructDescriptor, optimizer))
            os.system("echo \"read {}\" > {}-{}.cplex".format(opt_file, \
                constructDescriptor, optimizer))
#            os.system("echo \"set preprocessing presolve no\" >> {}-{}.cplex".format(constructDescriptor, optimizer))
            os.system("echo \"optimize\" >> {}-{}.cplex".format(constructDescriptor, \
                optimizer))
            os.system("echo \"write {}-{}.sol\" >> {}-{}.cplex".format(constructDescriptor, \
                optimizer, constructDescriptor, optimizer))
            os.system("echo \"quit\" >> {}-{}.cplex".format(constructDescriptor, \
                optimizer))
            os.system("{} -f {}-{}.cplex".format(exe_fn, constructDescriptor, \
                optimizer))
            isfeasible, objective_value, answers = extractAnswersCPLEX("{}-{}.sol".format(constructDescriptor, optimizer), "{}_soln-{}.csv.gz".format(constructDescriptor, optimizer), var_to_var_desc)


    print ("Building report and analytics files...")
    analytics = dict()

    # Used to quickly locate a trajectories poset membership
    set_posets = [ set(_poset) for _poset in posets ]
    set_posets_names = [ 'Target', 'Target-nbr', 'Other-nbr', 'Other' ]

    with gzip.open("{}_mappings-{}.csv.gz".format(constructDescriptor, optimizer), "wt") as mappings:
        analytics["Trajectories"] = list()
        if not traj_mappings == None:
            T_ = analytics["Trajectories"]
            rows = list()
            T_.append(( "Trajectory LER", "Trajectory LER Value", "Source", "Trajectory", "Trajectory Index", "Poset" ))
            rows.append([ "Trajectory LER", "Trajectory LER Value", "Source", "Trajectory", "Trajectory Index", "Poset" ])
            for key, _value in LER_to_trajectories.items():
                t_idx, value = _value
                idx = -1
                for p_idx, set_poset in enumerate(set_posets):
                    if t_idx in set_poset:
                        idx = p_idx
                        break
                assert idx != -1, "Trajectory {} - {} not found in any poset!".format(t_idx, tuple(traj_mappings[value]))
                if isfeasible:
                    T_.append([ var_to_var_desc[key], answers[key], tuple(traj_mappings[value]), value, t_idx, set_posets_names[idx] ])
                    rows.append([ var_to_var_desc[key], answers[key], tuple(traj_mappings[value]), value, t_idx, set_posets_names[idx] ])
                else:
                    T_.append([ var_to_var_desc[key], '', tuple(traj_mappings[value]), value, t_idx, set_posets_names[idx] ])
                    rows.append([ var_to_var_desc[key], '', tuple(traj_mappings[value]), value, t_idx, set_posets_names[idx] ])
        analytics["States"] = list()
        if not state_specs == None:
            S_ = analytics["States"]
            rows.append([ "" ])
            if onehot:
                S_.append([ "State Index", tuple(onehot_states_encodings) ])
                rows.append([ "State Index", tuple(onehot_states_encodings) ])
            else:
                S_.append([ "State Index", tuple(state_specs) ])
                rows.append([ "State Index", tuple(state_specs) ])
            for key, value in states.items():
                S_.append([ value, key ])
                rows.append([ value, key ])

        analytics["Actions"] = list()
        if not action_specs == None:
            A_ = analytics["Actions"]
            rows.append([ "" ])
            if onehot:
                A_.append([ "Action Index", tuple(onehot_actions_encodings) ])
                rows.append([ "Action Index", tuple(onehot_actions_encodings) ])
            else:
                A_.append([ "Action Index", tuple(action_specs) ])
                rows.append([ "Action Index", tuple(action_specs) ])
            for key, value in actions.items():
                A_.append([ value, key ])
                rows.append([ value, key ])

        analytics["Maskings"] = dict()
        analytics["Clusters"] = dict()
        if answers != None and (constructDescriptor == "FoMDP" or constructDescriptor == "FoMDP_2"):
            rows.append([ "" ])
            new_rows, Ms_list, Qs_list = FoMDP.reportMaskAndStateClusters(var_to_var_desc, answers, analytics)
            rows.extend(new_rows)

        writer = csv.writer(mappings)
        writer.writerows(rows)
        mappings.close()

    with gzip.open("{}_probabilities-{}.csv.gz".format(constructDescriptor, optimizer), "wt") as probabilities:
        analytics["Probabilities"] = dict()
        P_ = analytics["Probabilities"]
        rows = list()
        if constructDescriptor == "FoMDP_2":
            if len(Qs_list) > 0: # Used self-organizer
                P_["Class Probability"] = list()
                P_CP_ = P_["Class Probability"]
                pq = dict()
                P_CP_.append([ "Class Probability", "Class Tuple", "Probability" ])
                rows.append([ "Class Probability", "Class Tuple", "Probability" ])
                for key, value in answers.items():
                    item = var_to_var_desc[key]
                    p = parse.parse("Pq_{}_{}_{}", item, case_sensitive=True)
                    if p != None:
                        try:
                            c_idx2 = int(p[0])
                            a_idx = int(p[1])
                            c_idx1 = int(p[2])
                            p_tuple = (c_idx2, a_idx, c_idx1)
                            P_CP_.append([ item, p_tuple, value ])
                            rows.append([ item, p_tuple, value ])
                            pq[p_tuple] = value
                        except ValueError:
                            pass
                rows.append([ "" ])
                P_["Class DM Probability"] = list()
                P_CDMP_ = P_["Class DM Probability"]
                pq_targ = dict()
                P_CDMP_.append([ "Class DM Probability", "Class Tuple", "Probability" ])
                rows.append([ "Class DM Probability", "Class Tuple", "Probability" ])
                for key, value in answers.items():
                    item = var_to_var_desc[key]
                    p = parse.parse("Pq_targ_{}_{}", item, case_sensitive=True)
                    if p != None:
                        try:
                            a_idx = int(p[0])
                            c_idx = int(p[1])
                            p_tuple = (a_idx, c_idx)
                            P_CDMP_.append([ item, p_tuple, value ])
                            rows.append([ item, p_tuple, value ])
                            pq_targ[p_tuple] = value
                        except ValueError:
                            pass
                rows.append([ "" ])
                P_["Class Other DMs Probability"] = list()
                P_CODMSP_ = P_["Class Other DMs Probability"]
                pq_nontarg = dict()
                P_CODMSP_.append([ "Class Other DMs Probability", "Class Tuple", "Probability" ])
                rows.append([ "Class Other DMs Probability", "Class Tuple", "Probability" ])
                for key, value in answers.items():
                    item = var_to_var_desc[key]
                    p = parse.parse("Pq_nontarg_{}_{}", item, case_sensitive=True)
                    if p != None:
                        try:
                            a_idx = int(p[0])
                            c_idx = int(p[1])
                            p_tuple = (a_idx, c_idx)
                            P_CODMSP_.append([ item, p_tuple, value ])
                            rows.append([ item, p_tuple, value ])
                            pq_nontarg[p_tuple] = value
                        except ValueError:
                            pass
                rows.append([ "" ])
                P_["State Probability"] = list()
                P_SP_ = P_["State Probability"]
                P_SP_.append([ "State Probability", "Tuple", "Probability" ])
                rows.append([ "State Probability", "Tuple", "Probability" ])
                for (s_idx1, a_idx, s_idx2) in all_triples:
                    P_SP_.append([ "P_{}_{}_{}".format(s_idx2, a_idx, s_idx1), ( s_idx2, a_idx, s_idx1 ), pq[(Qs_list[s_idx2], a_idx, Qs_list[s_idx1])]])
                    rows.append([ "P_{}_{}_{}".format(s_idx2, a_idx, s_idx1), ( s_idx2, a_idx, s_idx1 ), pq[(Qs_list[s_idx2], a_idx, Qs_list[s_idx1])]])
                rows.append([ "" ])
                P_["State DM Probability"] = list()
                P_SDMP_ = P_["State DM Probability"]
                P_SDMP_.append([ "State DM Probability", "Tuple", "Probability" ])
                rows.append([ "State DM Probability", "Tuple", "Probability" ])
                reported = set()
                for (s_idx, a_idx, _) in all_triples:
                    if ( a_idx, s_idx ) in reported:
                        continue
                    P_SDMP_.append([ "P_targ_{}_{}".format(a_idx, s_idx), ( a_idx, s_idx ), pq_targ[(a_idx, Qs_list[s_idx])]])
                    rows.append([ "P_targ_{}_{}".format(a_idx, s_idx), ( a_idx, s_idx ), pq_targ[(a_idx, Qs_list[s_idx])]])
                    reported.add( ( a_idx, s_idx ) )
                rows.append([ "" ])
                P_["State Other DMs Probability"] = list()
                P_SODMP_ = P_["State Other DMs Probability"]
                P_SODMP_.append([ "State Other DMs Probability", "Tuple", "Probability" ])
                rows.append([ "State Other DMs Probability", "Tuple", "Probability" ])
                reported = set()
                for (s_idx, a_idx, _) in all_triples:
                    if ( a_idx, s_idx ) in reported:
                        continue
                    P_SODMP_.append([ "P_nontarg_{}_{}".format(a_idx, s_idx), ( a_idx, s_idx ), pq_nontarg[(a_idx, Qs_list[s_idx])]])
                    rows.append([ "P_nontarg_{}_{}".format(a_idx, s_idx), ( a_idx, s_idx ), pq_nontarg[(a_idx, Qs_list[s_idx])]])
                    reported.add( ( a_idx, s_idx ) )
        else:
            P_["State Probability"] = list()
            P_SP_ = P_["State Probability"]
            P_SP_.append([ "State Probability", "Tuple", "Probability" ])
            P_["State DM Probability"] = list()
            P_SDMP_ = P_["State DM Probability"]
            P_SDMP_.append([ "State DM Probability", "Tuple", "Probability" ])
            P_["State Other DMs Probability"] = list()
            P_SODMP_ = P_["State Other DMs Probability"]
            P_SODMP_.append([ "State Other DMs Probability", "Tuple", "Probability" ])
            if P != None:
                rows.append([ "Symbol", "Tuple", "Probability" ])
                for t, value in P.items():
                    P_SP_.append([ "P_W(s_2| a, s_1)", t, value ])
                    rows.append([ "P_W(s_2| a, s_1)", t, value ])
                rows.append([ "" ])
                for t, value in P_targ.items():
                    P_SDMP_.append([ "P_targ(a | s)", t, value ])
                    rows.append([ "P_targ(a | s)", t, value ])
                rows.append([ "" ])
                for t, value in P_nontarg.items():
                    P_SODMP_.append([ "P_nontarg(a | s)", t, value ])
                    rows.append([ "P_nontarg(a | s)", t, value ])
            elif answers != None:
                for key, value in answers.items():
                    item = var_to_var_desc[key]
                    p = parse.parse("P_{}_{}_{}", item, case_sensitive=True)
                    if p != None:
                        try:
                            s_idx2 = int(p[0])
                            a_idx = int(p[1])
                            s_idx1 = int(p[2])
                            p_tuple = (s_idx2, a_idx, s_idx1)
                            P_SP_.append([ item, p_tuple, value ])
                            rows.append([ item, p_tuple, value ])
                        except ValueError:
                            pass
                for key, value in answers.items():
                    item = var_to_var_desc[key]
                    p = parse.parse("P_targ_{}_{}", item, case_sensitive=True)
                    if p != None:
                        try:
                            a_idx = int(p[0])
                            s_idx = int(p[1])
                            p_tuple = (a_idx, s_idx)
                            P_SDMP_.append([ item, p_tuple, value ])
                            rows.append([ item, p_tuple, value ])
                        except ValueError:
                            pass
                for key, value in answers.items():
                    item = var_to_var_desc[key]
                    p = parse.parse("P_nontarg_{}_{}", item, case_sensitive=True)
                    if p != None:
                        try:
                            a_idx = int(p[0])
                            s_idx = int(p[1])
                            p_tuple = (a_idx, s_idx)
                            P_SODMP_.append([ item, p_tuple, value ])
                            rows.append([ item, p_tuple, value ])
                        except ValueError:
                            pass

        writer = csv.writer(probabilities)
        writer.writerows(rows)
        probabilities.close()

    if answers != None:
        with gzip.open("{}_rewards_{}.csv.gz".format(constructDescriptor, optimizer), "wt") as rewards_fn:
            analytics["Rewards"] = dict()
            R_ = analytics["Rewards"]
            rows = list()
            if constructDescriptor == "FoMDP_2":
                if len(Qs_list) > 0: # Used self-organizer
                    R_["Class Reward"] = list()
                    R_CR_ = R_["Class Reward"]
                    rq = dict()
                    R_CR_.append([ "Class Reward", "Class Tuple", "Value" ])
                    rows.append([ "Class Reward", "Class Tuple", "Value" ])
                    for key, value in answers.items():
                        item = var_to_var_desc[key]
                        p = parse.parse("rq_{}_{}_{}", item, case_sensitive=True)
                        if p != None:
                            try:
                                c_idx1 = int(p[0])
                                a_idx = int(p[1])
                                c_idx2 = int(p[2])
                                r_tuple = (c_idx1, a_idx, c_idx2)
                                R_CR_.append([ item, r_tuple, value ])
                                rows.append([ item, r_tuple, value ])
                                rq[r_tuple] = value
                            except ValueError:
                                pass
                    rows.append([ "" ])
                    R_["Class Fractional DM Reward"] = list()
                    R_FDMR_ = R_["Class Fractional DM Reward"]
                    frvq_targ = dict()
                    R_FDMR_.append([ "Class Fractional DM Reward", "Class Tuple", "Value" ])
                    rows.append([ "Class Fractional DM Reward", "Class Tuple", "Value" ])
                    for key, value in answers.items():
                        item = var_to_var_desc[key]
                        p = parse.parse("FRVq_targ_{}_{}_{}", item, case_sensitive=True)
                        if p != None:
                            try:
                                c_idx1 = int(p[0])
                                a_idx = int(p[1])
                                c_idx2 = int(p[2])
                                r_tuple = (c_idx1, a_idx, c_idx2)
                                R_FDMR_.append([ item, r_tuple, value ])
                                rows.append([ item, r_tuple, value ])
                                frvq_targ[r_tuple] = value
                            except ValueError:
                                pass
                    rows.append([ "" ])
                    R_["Class Fractional Other DMs Reward"] = list()
                    R_FODMR_ = R_["Class Fractional Other DMs Reward"]
                    frvq_nontarg = dict()
                    R_FODMR_.append([ "Class Fractional Other DMs Reward", "Class Tuple", "Value" ])
                    rows.append([ "Class Fractional Other DMs Reward", "Class Tuple", "Value" ])
                    for key, value in answers.items():
                        item = var_to_var_desc[key]
                        p = parse.parse("FRVq_nontarg_{}_{}_{}", item, case_sensitive=True)
                        if p != None:
                            try:
                                c_idx1 = int(p[0])
                                a_idx = int(p[1])
                                c_idx2 = int(p[2])
                                r_tuple = (c_idx1, a_idx, c_idx2)
                                R_FODMR_.append([ item, r_tuple, value ])
                                rows.append([ item, r_tuple, value ])
                                frvq_nontarg[r_tuple] = value
                            except ValueError:
                                pass
                    rows.append([ "" ])
                    R_["Reward"] = list()
                    R_R_ = R_["Reward"]
                    R_R_.append([ "Reward", "Tuple", "Value" ])
                    rows.append([ "Reward", "Tuple", "Value" ])
                    for (s_idx1, a_idx, s_idx2) in all_triples:
                        R_R_.append([ "r_{}_{}_{}".format(s_idx1, a_idx, s_idx2), ( s_idx1, a_idx, s_idx2 ), rq[(Qs_list[s_idx1], a_idx, Qs_list[s_idx2])]])
                        rows.append([ "r_{}_{}_{}".format(s_idx1, a_idx, s_idx2), ( s_idx1, a_idx, s_idx2 ), rq[(Qs_list[s_idx1], a_idx, Qs_list[s_idx2])]])
                    rows.append([ "" ])
                    R_["Fractional DM Reward"] = list()
                    R_FDMR_ = R_["Fractional DM Reward"]
                    R_FDMR_.append([ "Fractional DM Reward", "Tuple", "Value" ])
                    rows.append([ "Fractional DM Reward", "Tuple", "Value" ])
                    for (s_idx1, a_idx, s_idx2) in all_triples:
                        R_FDMR_.append([ "FRV_targ_{}_{}_{}".format(s_idx1, a_idx, s_idx2), ( s_idx1, a_idx, s_idx2 ), frvq_targ[(Qs_list[s_idx1], a_idx, Qs_list[s_idx2])]])
                        rows.append([ "FRV_targ_{}_{}_{}".format(s_idx1, a_idx, s_idx2), ( s_idx1, a_idx, s_idx2 ), frvq_targ[(Qs_list[s_idx1], a_idx, Qs_list[s_idx2])]])
                    rows.append([ "" ])
                    R_["Fractional Other DMs Reward"] = list()
                    R_FODMR_ = R_["Fractional Other DMs Reward"]
                    R_FODMR_.append([ "Fractional Other DMs Reward", "Tuple", "Value" ])
                    rows.append([ "Fractional Other DMs Reward", "Tuple", "Value" ])
                    for (s_idx1, a_idx, s_idx2) in all_triples:
                        R_FODMR_.append([ "FRV_nontarg_{}_{}_{}".format(s_idx1, a_idx, s_idx2), ( s_idx1, a_idx, s_idx2 ), frvq_nontarg[(Qs_list[s_idx1], a_idx, Qs_list[s_idx2])]])
                        rows.append([ "FRV_nontarg_{}_{}_{}".format(s_idx1, a_idx, s_idx2), ( s_idx1, a_idx, s_idx2 ), frvq_nontarg[(Qs_list[s_idx1], a_idx, Qs_list[s_idx2])]])
            else:
                R_["Reward"] = list()
                R_R_ = R_["Reward"]
                R_R_.append([ "Reward", "Tuple", "Value" ])
                R_["Fractional DM Reward"] = list()
                R_FDMR_ = R_["Fractional DM Reward"]
                R_FDMR_.append([ "Fractional DM Reward", "Tuple", "Value" ])
                R_["Fractional Other DMs Reward"] = list()
                R_FODMR_ = R_["Fractional Other DMs Reward"]
                R_FODMR_.append([ "Fractional Other DMs Reward", "Tuple", "Value" ])
                rows.append([ "Reward", "Tuple", "Value" ])
                for key, value in answers.items():
                    item = var_to_var_desc[key]
                    p = parse.parse("r_{}_{}_{}", item, case_sensitive=True)
                    if p != None:
                        try:
                            s_idx1 = int(p[0])
                            a_idx = int(p[1])
                            s_idx2 = int(p[2])
                            r_tuple = (s_idx1, a_idx, s_idx2)
                            R_R_.append([ item, r_tuple, value ])
                            rows.append([ item, r_tuple, value ])
                            continue
                        except ValueError:
                            pass
                    p = parse.parse("R_[s_{},a_{},s_{}]", item, case_sensitive=True)
                    if p != None:
                        try:
                            s_idx1 = int(p[0])
                            a_idx = int(p[1])
                            s_idx2 = int(p[2])
                            r_tuple = ( s_idx1, a_idx, s_idx2 )
                            R_R_.append([ item, r_tuple, value ])
                            rows.append([ item, r_tuple, value ])
                            continue
                        except ValueError:
                            pass
                    p = parse.parse("R_[s_{}_<{}>,a_{},s_{}_<{}>]", item, case_sensitive=True)
                    if p != None:
                        try:
                            s_idx1 = int(p[0])
                            h1 = int(p[1])
                            a_idx = int(p[2])
                            s_idx2 = int(p[3])
                            h2 = int(p[4])
                            r_tuple = ( s_idx1, "-", h1, a_idx, s_idx2, "-", h2)
                            R_R_.append([ item, r_tuple, value ])
                            rows.append([ item, r_tuple, value ])
                            continue
                        except ValueError:
                            pass
                rows.append([ "" ])
                rows.append([ "Fractional DM Reward", "Tuple", "Value" ])
                for key, value in answers.items():
                    item = var_to_var_desc[key]
                    p = parse.parse("FRV_targ_{}_{}_{}", item, case_sensitive=True)
                    if p != None:
                        try:
                            s_idx1 = int(p[0])
                            a_idx = int(p[1])
                            s_idx2 = int(p[2])
                            r_tuple = (s_idx1, a_idx, s_idx2)
                            R_FDMR_.append([ item, r_tuple, value ])
                            rows.append([ item, r_tuple, value ])
                            continue
                        except ValueError:
                            pass
                    p = parse.parse("FRV_targ_[s_{},a_{},s_{}]", item, case_sensitive=True)
                    if p != None:
                        try:
                            s_idx1 = int(p[0])
                            a_idx = int(p[1])
                            s_idx2 = int(p[2])
                            r_tuple = ( s_idx1, a_idx, s_idx2 )
                            R_FDMR_.append([ item, r_tuple, value ])
                            rows.append([ item, r_tuple, value ])
                        except ValueError:
                            pass
                rows.append([ "" ])
                rows.append([ "Fractional Other DM Reward", "Tuple", "Value" ])
                for key, value in answers.items():
                    item = var_to_var_desc[key]
                    p = parse.parse("FRV_nontarg_{}_{}_{}", item, case_sensitive=True)
                    if p != None:
                        try:
                            s_idx1 = int(p[0])
                            a_idx = int(p[1])
                            s_idx2 = int(p[2])
                            r_tuple = (s_idx1, a_idx, s_idx2)
                            R_FODMR_.append([ item, r_tuple, value ])
                            rows.append([ item, r_tuple, value ])
                            continue
                        except ValueError:
                            pass
                    p = parse.parse("FRV_nontarg_[s_{},a_{},s_{}]", item, case_sensitive=True)
                    if p != None:
                        try:
                            s_idx1 = int(p[0])
                            a_idx = int(p[1])
                            s_idx2 = int(p[2])
                            r_tuple = ( s_idx1, a_idx, s_idx2 )
                            R_FODMR_.append([ item, r_tuple, value ])
                            rows.append([ item, r_tuple, value ])
                        except ValueError:
                            pass
            writer = csv.writer(rewards_fn)
            writer.writerows(rows)
            rewards_fn.close()

    # Writing analytics compressed pickle file
    with open("{}_analytics-{}.compressed_pickle".format(constructDescriptor, optimizer), "wb") as analytics_file:
        compress_pickle.dump(analytics, analytics_file, compression="gzip")
        analytics_file.close()

    # Creating graphml of DTM
    try:
        actions_inv = { value: key for key, value in actions.items() }

        graph = networkx.MultiDiGraph()

        if constructOptimization == RoMDP.constructOptimization:
            # Build the nodes
            for key, value in states.items():
                # key is tuple
                # value is state index
                graph.add_node(value, features = str(key))

            # Build the edges
            for ( s_idx2, a_idx, s_idx1 ), value in P.items():
                try:
                    ptarg = P_targ[( a_idx, s_idx1 )]
                except KeyError:
                    ptarg = 0
                try:
                    pnontarg = P_nontarg[( a_idx, s_idx1 )]
                except KeyError:
                    pnontarg = 0
                graph.add_edge(s_idx1, s_idx2, action_index = a_idx, action_features = str(actions_inv[a_idx]), P = value, P_targ = ptarg, P_nontarg = pnontarg)

        elif constructOptimization == FoMDP.constructOptimization:
            Qs_list_inv = dict()
            for key, value in enumerate(Qs_list):
                try:
                    Qs_list_inv[value].append(key)
                except KeyError:
                    Qs_list_inv[value] = [ key ]

            # Build the nodes as clusters
            for cidx, s_idx_s in Qs_list_inv.items():
                graph.add_node(cidx, features = str(s_idx_s))

            probs = dict()
            P_ = analytics["Probabilities"]["State Probability"]
            P_DM_ = analytics["Probabilities"]["State DM Probability"]
            P_O_ = analytics["Probabilities"]["State Other DMs Probability"]
            for pidx in range(1,len(P_)):
                _, ( sidx2, aidx, sidx1 ), p = P_[pidx]
                probs[( Qs_list[sidx2], aidx, Qs_list[sidx1] )] = p

            ptarg = dict()
            for pidx in range(1,len(P_DM_)):
                _, ( aidx, sidx1 ), p = P_DM_[pidx]
                ptarg[( aidx, Qs_list[sidx1] )] = p

            pnontarg = dict()
            for pidx in range(1,len(P_O_)):
                _, ( aidx, sidx1 ), p = P_O_[pidx]
                pnontarg[( aidx, Qs_list[sidx1] )] = p

            # Build the edges
            for ( c_idx2, a_idx, c_idx1 ), p in probs.items():
                try:
                    ptarg[( a_idx, c_idx1 )]
                except KeyError:
                    ptarg[( a_idx, c_idx1 )] = 0
                try:
                    pnontarg[( a_idx, c_idx1 )]
                except KeyError:
                    pnontarg[( a_idx, c_idx1 )] = 0
                graph.add_edge(c_idx1, c_idx2, action_index = a_idx, action_features = str(actions_inv[a_idx]), P = p, P_targ = ptarg[( a_idx, c_idx1 )], P_nontarg = pnontarg[( a_idx, c_idx1 )])

        gmlfile = "{}-{}.graphml.gz".format(constructDescriptor, optimizer)
        networkx.write_graphml(G = graph, path = gmlfile, named_key_ids = True)
    except:
        print ("Unable to create graphml of DTM!")

    # Return to prior working directory
    os.chdir(cwd)
    print ("Optimization and reporting elapsed time = " + str(time.time() - master_start_time))

    return (isfeasible, objective_value, answers, P, P_targ, P_nontarg, LER_to_trajectories, var_to_var_desc)


def extractAnswersBARON(filename, var_to_var_desc):
    print ("Extracting answers --")
    answers = dict()
    with open("res.lst", "r") as results_f:
        flag = True
        while flag:
            line = results_f.readline()
            if line == "":
                break
            if line == "The best solution found is:\n":
                flag = False
        if flag:
            print ('No solution found!')
            return (False, None, None)
        results_f.readline()
        results_f.readline()
        while True:
            line = results_f.readline()
            if line == "\n":
                break
#            print (line, end='')

            info = line.split()
            if len(info) != 4:
                continue
            try:
                print ("Variable {} already assigned to value {}!".format(info[0], answers[info[0]]))
                return (False, None, None)
            except KeyError:
                answers[info[0]] = float(info[2])
#                print ("Varaible {} = {}".format(info[0], answers[info[0]]))
        line = results_f.readline()
        try:
            pos = line.index(':')
        except ValueError:
            print ("Unable to find objective value!")
            return (None)
        if line[:pos] != "The above solution has an objective value of":
            print ("Unable to find objective value!")
            return (False, None, None)
        objective_value = float(line[pos+1:])

    outputs = list()
    for item, value in answers.items():
        outputs.append([ var_to_var_desc[item], value ])
    outputs.sort()
    with gzip.open(filename, "wt", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(outputs)
        csv_file.close()
    del outputs
    return (True, objective_value, answers)


def extractAnswersCPLEXModel(cplex_model, csvoutfilename, var_to_var_desc, cplex_variables):
    isfeasible = cplex_model.solution.get_status() == 1
    if not isfeasible:
        if DEBUG:
            print ("No feasible solution found.")
        return (False, None, dict())
    if DEBUG:
        print ("Extracting answers --")
    objective_value = cplex_model.solution.get_objective_value()
    values = cplex_model.solution.get_values(list(cplex_variables.values()))
    answers = dict(zip(cplex_variables.keys(), values))
    outputs = list()
    for idx, item in enumerate(cplex_variables.keys()):
        outputs.append([ var_to_var_desc[item], values[idx] ])
    outputs.sort()
    with gzip.open(csvoutfilename, "wt", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(outputs)
        csv_file.close()
    del outputs
    return (isfeasible, objective_value, answers)

def extractAnswersCPLEX(cplex_sol, csvoutfilename, var_to_var_desc):
    print ("Extracting answers --")
    answers = dict()
    try:
        with open(cplex_sol, "r") as results_f:
            flag = True
            var_pre = "<variable name=\""
            value_pre = "value=\""
            obj_pre = "objectiveValue=\""
            sol_stat_pre = "solutionStatusValue=\""
            sol_stats_pre = "solutionStatusString=\""
            infeasible = True
            while True:
                line = results_f.readline()
                if line == "": #EOF
                    break
                try: # Look for variable value
                    pos = line.index(var_pre)
                    line = line[pos+len(var_pre):]
                    pos = line.index('"')
                    var = line[:pos]
                    pos = line.index(value_pre)
                    line = line[pos+len(value_pre):]
                    pos = line.index('"')
                    value = float(line[:pos])
                    try:
                        print ("Variable {} already assigned to value {}!".format(var, answers[var]))
                        return (False, None, None)
                    except KeyError:
                        answers[var] = value
                    continue # Next line
                except ValueError:
                    pass
                try:
                    pos = line.index(obj_pre)
                    line = line[pos+len(obj_pre):]
                    pos = line.index('"')
                    objective_value = float(line[:pos])
                    print ("Objective value is {}".format(objective_value))
                    flag = False
                    continue
                except ValueError:
                    pass
                try:
                    pos = line.index(sol_stat_pre)
                    line = line[pos+len(sol_stat_pre):]
                    pos = line.index('"')
                    solution_status_value = int(line[:pos])
                    print ("Solution status value is {}".format(solution_status_value))
                    if solution_status_value == 1 or solution_status_value == 101 or solution_status_value == 102:
                        infeasible = False
                    continue
                except ValueError:
                    pass
                try:
                    pos = line.index(sol_stats_pre)
                    line = line[pos+len(sol_stats_pre):]
                    pos = line.index('"')
                    print ("Solution status string is {}".format(line[:pos]))
                except ValueError:
                    pass
            if flag:
                print ('No solution found!')
                return (False, None, None)
    except FileNotFoundError:
        print ('No solution file found!')
        return (False, None, None)

    outputs = list()
    for item, value in answers.items():
        outputs.append([ var_to_var_desc[item], value ])
    outputs.sort()
    with gzip.open(csvoutfilename, "wt", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(outputs)
        csv_file.close()
    del outputs
    return (infeasible, objective_value, answers)

# Function to load in attribute specifications from a csv file
#   csv file should just be one comma-delimited line of strings
def loadAttributeSpecifications(csvfilename):
    # Returns a list of strings represented the attribute headings from this specification
    if csvfilename[-3:] == '.gz':
        csvfile = gzip.open(csvfilename, 'rt')
    else:
        csvfile = open(csvfilename, 'r')
    rows = list(csv.reader(csvfile))
    #if DEBUG:
        #print (rows)
    csvfile.close()
    return (rows[0])

# Function scans through the trajectories to collect all possible
#   actions and states value
def scanTrajectories(action_specs, state_specs, trajs_fn_list):
    # Returns updated ( actions_values, states_values,
    #   onehot_actions_encodings, onehot_states_encodings )
    actions_values_sets = dict()
    states_values_sets = dict()
    for filename in trajs_fn_list:
        if filename == None:
            continue
        if filename[-3:] == '.gz':
            listfile = gzip.open(filename, 'rt')
        else:
            listfile = open(filename, 'r')
        fn = listfile.readline()
        while fn:
            if fn[len(fn)-1] == '\n':
                fn = fn[:len(fn)-1]
            with open(fn, 'r') as csvfile:
                rows = list(csv.reader(csvfile))
                header = rows.pop(0)
                action_cols = list()
                state_cols = list()
                for a_str in action_specs:
                    try:
                        action_cols.append(header.index(a_str))
                    except ValueError:
                        sys.exit('scanTrajectories(...) -- {} action specifier not found in header {}!'.format(a_str, header))
                for s_str in state_specs:
                    try:
                        state_cols.append(header.index(s_str))
                    except ValueError:
                        sys.exit('scanTrajectories(...) -- {} action specifier not found in header {}!'.format(s_str, header))
                for row in rows: # Process each row to seperate out state and action
                    if len(row) == 0: # skip empty row
                        continue
                    s_strs = list()
                    for s_col in state_cols:
                        try:
                            states_values_sets[header[s_col]].add(row[s_col])
                        except KeyError:
                            states_values_sets[header[s_col]] = set()
                            states_values_sets[header[s_col]].add(row[s_col])
                    for a_col in action_cols:
                        try:
                            actions_values_sets[header[a_col]].add(row[a_col])
                        except KeyError:
                            actions_values_sets[header[a_col]] = set()
                            actions_values_sets[header[a_col]].add(row[a_col])
                csvfile.close()
            fn = listfile.readline()
        listfile.close()

    # Build onehot encoding
    onehot_actions_encodings = list()
    actions_values = dict()
    a_idx = 0
    for a_str in action_specs:
        for v in actions_values_sets[a_str]:
            actions_values[( a_str, v )] = a_idx
            a_idx += 1
            onehot_actions_encodings.append("{}={}".format(a_str, v))
    del actions_values_sets

    onehot_states_encodings = list()
    states_values = dict()
    s_idx = 0
    for s_str in state_specs:
        for v in states_values_sets[s_str]:
            states_values[( s_str, v )] = s_idx
            s_idx += 1
            onehot_states_encodings.append("{}={}".format(s_str, v))
    del states_values_sets

    return ( actions_values, states_values, onehot_actions_encodings, onehot_states_encodings )


# Function to build states and actions from a trajectory csv file if new
#   csv file should have a header describing the attribute name for each column
#       remaining rows are each a comma-delimited list of strings
#   action_specs and state_specs are lists of strings that need to be found in the header to build actions and states
def processTrajectory(csvfilename, states, actions, action_specs, state_specs, onehot, actions_values, states_values):
    # Updates the states and actions dictionary passed in
    # Returns a list representing the state and action indices from converting the trajectory csv file
    #   [ s1, a1, s2, a2, s3, a3, ..., sn ]
    #   corresponding to the dictionaries
    # if onehot == True, tuples representing each state and action are encoded as
    #   onehot accordingly
    if csvfilename[-3:] == '.gz':
        csvfile = gzip.open(csvfilename, 'rt')
    else:
        csvfile = open(csvfilename, 'r')
    rows = list(csv.reader(csvfile))
    header = rows.pop(0)
    action_cols = list()
    state_cols = list()
    for a_str in action_specs:
        try:
            action_cols.append(header.index(a_str))
        except ValueError:
            sys.exit('processTrajectory(...) -- {} action specifier not found in header {}!'.format(a_str, header))
    for s_str in state_specs:
        try:
            state_cols.append(header.index(s_str))
        except ValueError:
            sys.exit('processTrajectory(...) -- {} action specifier not found in header {}!'.format(s_str, header))

    trajectory = list()
    for row in rows: # Process each row to seperate out state and action
        if len(row) == 0: # skip empty row
            continue
        if len(row) != len(header): # Corrupted file
            return (None)
        if onehot:
            s_tuple = [ 0 for _ in range(len(states_values)) ]
            for s_col in state_cols:
                s_tuple[states_values[ (header[s_col], row[s_col]) ]] = 1
            s_tuple = tuple(s_tuple)

            a_tuple = [ 0 for _ in range(len(actions_values)) ]
            for a_col in action_cols:
                a_tuple[actions_values[ (header[a_col], row[a_col]) ]] = 1
            a_tuple = tuple(a_tuple)
        else:
            s_strs = list()
            for s_col in state_cols:
                s_strs.append(row[s_col])
            s_tuple = tuple(s_strs)
            a_strs = list()
            for a_col in action_cols:
                a_strs.append(row[a_col])
            a_tuple = tuple(a_strs)

        try:
            trajectory.append(states[s_tuple])
        except KeyError:
            trajectory.append(len(states))
            states[s_tuple] = len(states)
        try:
            trajectory.append(actions[a_tuple])
        except KeyError:
            trajectory.append(len(actions))
            actions[a_tuple] = len(actions)
    trajectory.pop() # Remove the last action
    csvfile.close()
    return (trajectory)

# Processes the trajectories from a list of filenames
def processTrajectories(traj_lists_fn, poset_name, states, actions, action_specs, state_specs, onehot, actions_values, states_values, traj_mappings):
    # Returns list of trajectories
    trajs = list()
    if traj_lists_fn[-3:] == '.gz':
        trajs_file = gzip.open(traj_lists_fn, 'rt')
    else:
        trajs_file = open(traj_lists_fn, 'r')
    fns = trajs_file.readlines()
    for fn in tqdm.tqdm(fns, total=len(fns), desc=poset_name):
        fn = fn[:-1]
        new_traj = processTrajectory(fn, states, actions, action_specs, state_specs, onehot, actions_values, states_values)
        if new_traj != None:
            trajs.append(new_traj)
            try:
                traj_mappings[tuple(new_traj)].add(fn)
            except KeyError:
                traj_mappings[tuple(new_traj)] = set()
                traj_mappings[tuple(new_traj)].add(fn)
        else:
            print ("Trajectory file in {} corrupted (skipping): {}".format(poset_name, fn))
    trajs_file.close()
    return (trajs)


# Function takes lists of trajectory filenames
#   Each list of trajectories is simply rows of filenames
def convertTrajectories(action_spec_csvfilename, state_spec_csvfilename, target_traj_list_fn, target_nbr_traj_list_fn, other_nbr_traj_list_fn, other_traj_list_fn, onehot = False):
    # Returns ( action_specs, actions, state_specs, states, target trajectories,
    #   target nbr trajectories, other nbr trajectories, other trajectories,
    #   traj_mappings, actions_values, onehot_actions_encodings, states_values,
    #   onehot_states_encodings )
    #   actions is a dictionary mapping action tuple attributes to an index
    #   states is a diectionary mapping state tuple attributes to an index
    #   action_specs and state_specs are lists of strings that need to be found in the header to build actions and states
    #   latter 4 are lists of trajectories in index form
    #   traj_mappings is a dictionary mapping each trajectory found to a set of trajectory filenames
    #       -- multiple filenames since there can be duplication of trajectories between collections.
    #   if onehot is True, then the actions and states are encoded into a binary
    #       feature vector
    #       e.g., Color = Red/Blue/Green is encoded into three binary
    #           features of Color=Red, Color=Blue, Color=Green subvectors
    #       actions_values, onehot_actions_encodings, states_values,
    #           onehot_states_encodings are returned in this case.
    #           actions_values and states_values are disctionaries of the form
    #               (header_name, value) --> index
    #           onehot_xxx_encodings is a list of strings as in the binary feature
    #               example

    actions = dict()
    states = dict()
    traj_mappings = dict()
    if DEBUG:
        print ("Loading action specifications...")
    action_specs = loadAttributeSpecifications(action_spec_csvfilename)
    print ("Action specifications - {}".format(action_specs))
    if DEBUG:
        print ("Loading state specifications...")
    state_specs = loadAttributeSpecifications(state_spec_csvfilename)
    print ("State specifications - {}".format(state_specs))

    if onehot: # Scan and collect all values across the trajectories
        actions_values, states_values, onehot_actions_encodings, onehot_states_encodings = scanTrajectories(action_specs, state_specs,  [ target_traj_list_fn, target_nbr_traj_list_fn, other_nbr_traj_list_fn, other_traj_list_fn])
    else:
        actions_values = dict()
        states_values = dict()
        onehot_actions_encodings = list()
        onehot_states_encodings = list()

    target_trajs = processTrajectories(target_traj_list_fn, "Target trajectories", states, actions, action_specs, state_specs, onehot, actions_values, states_values, traj_mappings)
    print ("Target trajectory files count = {}".format(len(target_trajs)))

    if target_nbr_traj_list_fn != None:
        target_nbr_trajs = processTrajectories(target_nbr_traj_list_fn, "Target nbr trajectories", states, actions, action_specs, state_specs, onehot, actions_values, states_values, traj_mappings)
    else:
        target_nbr_trajs = list()
    print ("Target nbr trajectory files count = {}".format(len(target_nbr_trajs)))

    if other_nbr_traj_list_fn != None:
        other_nbr_trajs = processTrajectories(other_nbr_traj_list_fn, "Other nbr trajectories", states, actions, action_specs, state_specs, onehot, actions_values, states_values, traj_mappings)
    else:
        other_nbr_trajs = list()
    print ("Other nbr trajectory files count = {}".format(len(other_nbr_trajs)))

    if other_traj_list_fn != None:
        other_trajs = processTrajectories(other_traj_list_fn, "Other trajectories", states, actions, action_specs, state_specs, onehot, actions_values, states_values, traj_mappings)
    else:
        other_trajs = list()
    print ("Other trajectory files count = {}".format(len(other_trajs)))

    print ("Total states found = {}".format(len(states)))
    print ("Total actions found = {}".format(len(actions)))
    return (action_specs, actions, state_specs, states, target_trajs, target_nbr_trajs, other_nbr_trajs, other_trajs, traj_mappings, actions_values, onehot_actions_encodings, states_values, onehot_states_encodings)

# Take a list of lists of trajectories and guarantee that they are disjoint
# In the case of a trajectory being in two lists, the list occurring earlier in the trajectory_lists ordering gets
#   precedence and keeps the trajectory while it is removed from the later occurring list.
# Note -- Duplicate trajectories in a single list are also removed.
def disjoinTrajectorySets(trajectory_lists):
    # trajectory_lists is a list of lists of trajectories of the form [ s1, a1, s2, a2, ..., sn ] being integer indices
    # Returns a tuple ( preserved_trajectories_indices, new_trajectory_lists, trajectory_to_location )
    #   -- preserved_trajectories_indices is a newly created list of lists of the position index of trajectories
    #       in a trajectory list that have been kept.
    #   -- new_trajectory_lists is a newly created disjoint list where each list of trajectories is a subsequence
    #       of the corresponding original list; ordering of the trajectories is also preserved.
    #       In the case of duplicate trajectories in a single list, only the first one is kept.
    #   -- trajectory_to_location is a dictionary that maps each trajectry (as a tuple) to a set of tuples
    #       where l_idx is the index of the original trajectory list and pos is the position of this trajectory in
    #       that list; handles mutiple occurrances of a trajectory including repeats is a single list.

    trajectory_to_location = dict()
        # Maps a single trajectory to a set of tuples ( l_idx, pos ) where l_idx is the index of a trajectory list
        #   and pos is the position of this trajectory in that list
    new_trajectory_lists = list()
        # The new disjoint trajectory lists
    preserved_trajectories_indices = list()
        # This is a list of lists of the positions of trajectories that remain in the new_trajectory_lists

    for l_idx, traj_list in enumerate(trajectory_lists):
        new_list = list()
        new_trajectory_lists.append(new_list)
        new_preserved = list()
        preserved_trajectories_indices.append(new_preserved)
        for pos, traj in enumerate(traj_list):
            try:
                trajectory_to_location[tuple(traj)].add(( l_idx, pos ))
            except KeyError:
                trajectory_to_location[tuple(traj)] = set()
                trajectory_to_location[tuple(traj)].add(( l_idx, pos ))
                new_list.append(traj)
                new_preserved.append(pos)

    return ( preserved_trajectories_indices, new_trajectory_lists, trajectory_to_location )

# Extracts and returns a set of (s, a, s') found in the trajectories in index
#   format.
def extractTriples(trajectories):
    triples = set()
    for traj in trajectories:
        for pos in range(0, len(traj) - 1, 2):
            triples.add( ( traj[pos], traj[pos+1], traj[pos+2] ) )
    return (triples)

# Checks to see if a lower poset trajectory happens to be a prefix of a higher
#   poset trajectory. If so, problem is not feasible.
def trajectoriesFeasible(new_trajectory_lists, list_ids, traj_mappings):
    if DEBUG:
        print ("Checking trajectory prefixing...")
    flag = True
    prefix = dict()
    for traj in new_trajectory_lists[0]:
        _traj = tuple(traj)
        for pos in range(2, len(traj) - 1, 2):
            __traj = tuple(traj[:pos])
            try:
                prefix[__traj].add((0, _traj))
            except KeyError:
                prefix[__traj] = set()
                prefix[__traj].add((0, _traj))

    for idx in range(1, 4):
        new_prefix = dict()
        for traj in new_trajectory_lists[idx]:
            _traj = tuple(traj)
            try:
                for (lidx, t) in prefix[_traj]: # Found one
                    flag = False
                    print ("{} in {} is a prefix of {} in {}".format(traj_mappings[_traj], list_ids[idx], traj_mappings[t], list_ids[lidx]))
            except KeyError:
                pass
            for pos in range(2, len(traj) - 1, 2):
                __traj = tuple(traj[:pos])
                try:
                    new_prefix[__traj].add((idx, _traj))
                except KeyError:
                    new_prefix[__traj] = set()
                    new_prefix[__traj].add((idx, _traj))

        prefix.update(new_prefix)
        del new_prefix
    del prefix
    return (flag)


import RoMDP

def runs_RoMDP():
    answers = learnMDP("Starcraft/doc_s_files/team/actions.csv", "Starcraft/doc_s_files/team/attributes.csv", "Starcraft/doc_s_files/team/list.files", None, None, "Starcraft/doc_s_files/others/list.files", "SC", None, 10, RoMDP.constructOptimization, "RoMDP", "CPLEX", "./cplex", None, None, None, None, None)
    answers = learnMDP("Starcraft/doc_s_files/team/actions.csv", "Starcraft/doc_s_files/team/attributes.csv", "Starcraft/doc_s_files/team/list.files", None, None, "Starcraft/doc_s_files/others/list.files", "SC", None, 10, RoMDP.constructOptimization, "RoMDP", "BARON", "./baron", 10000, "./baronlice.txt", None, None, None)
    answers = learnMDP("TAG/C1/Real/actions.csv", "TAG/C1/Real/attributes.csv", "TAG/C1/Real/list.files", "TAG/C1/Synthetic/list.files", "TAG/All/Synthetic/list.files", "TAG/All/Real/list.files", "TAG-C1", None, 1, RoMDP.constructOptimization, "RoMDP", "CPLEX", "./cplex", None, None, None, None, None)
    answers = learnMDP("TAG/C1/Real/actions.csv", "TAG/C1/Real/attributes.csv", "TAG/C1/Real/list.files", "TAG/C1/Synthetic/list.files", "TAG/All/Synthetic/list.files", "TAG/All/Real/list.files", "TAG-C1", None, 1, RoMDP.constructOptimization, "RoMDP", "BARON", "./baron", 10000, "./baronlice.txt", None, None, None)

def runs_1_RoMDP():
    answers = learnMDP("novice/actions.csv", "novice/attributes.csv", "novice/list-one.files", None, None, "planner/list-one.files", "Novice-Planner-1", 1, 10, RoMDP.constructOptimization, "RoMDP", "CPLEX", "./cplex", None, None, None, None, None)

def runs_short_RoMDP():
    answers = learnMDP("novice/actions.csv", "novice/attributes.csv", "novice/list-short.files", None, None, "planner/list-short.files", "Novice-Planner-short", 1, 10, RoMDP.constructOptimization, "RoMDP", "CPLEX", "./cplex", None, None, None, None, None)


def runs_FoMDP(num_classes = 2, simplify_masking = False, feasibility_only = True):
    answers = learnMDP("novice/actions.csv", "novice/attributes.csv", "novice/list-one.files", None, None, "planner/list-one.files", "Novice-Planner-one-{}-{}".format(num_classes, simplify_masking), 1, 10, FoMDP.constructOptimization, "FoMDP", "BARON", "./baron", 10000, "./baronlice.txt", num_classes, simplify_masking, feasbility_only)
    answers = learnMDP("novice/actions.csv", "novice/attributes.csv", "novice/list-short.files", None, None, "planner/list-short.files", "Novice-Planner-short-{}-{}".format(num_classes, simplify_masking), 1, 10, FoMDP.constructOptimization, "FoMDP", "BARON", "./baron", 10000, "./baronlice.txt", num_classes, simplify_masking, feasbility_only)
    answers = learnMDP("novice/actions.csv", "novice/attributes.csv", "novice/list.files", None, None, "planner/list.files", "Novice-Planner-{}-{}".format(num_classes, simplify_masking), 1, 10, FoMDP.constructOptimization, "FoMDP", "BARON", "./baron", 10000, "./baronlice.txt", num_classes, simplify_masking, feasbility_only)
    answers = learnMDP("Starcraft/doc_s_files/team/actions.csv", "Starcraft/doc_s_files/team/attributes.csv", "Starcraft/doc_s_files/team/list.files", None, None, "Starcraft/doc_s_files/others/list.files", "SC", None, 10, FoMDP.constructOptimization, "FoMDP", "BARON", "./baron", 10000, "./baronlice.txt", num_classes, simplify_masking, feasbility_only)

import SsMDP
import T_MDP

def runs_SsMDP():
    answers = learnMDP("novice/actions.csv", "novice/attributes.csv", "novice/list.files", None, None, "planner/list.files", "Novice-Planner", 1, 10, SsMDP.constructOptimization, "SsMDP", "BARON", "./baron", 1000, "./baronlice.txt", None, None, None)
    answers = learnMDP("Starcraft/doc_s_files/team/actions.csv", "Starcraft/doc_s_files/team/attributes.csv", "Starcraft/doc_s_files/team/list.files", None, None, "Starcraft/doc_s_files/others/list.files", "SC", 1, 10, SsMDP.constructOptimization, "SsMDP", "BARON", "./baron", 100, "./baronlice.txt", None, None, None)
    answers = learnMDP("TAG/C1/Real/actions.csv", "TAG/C1/Real/attributes.csv", "TAG/C1/Real/list.files", "TAG/C1/Synthetic/list.files", "TAG/All/Synthetic/list.files", "TAG/All/Real/list.files", "TAG-C1", 1, 1, SsMDP.constructOptimization, "SsMDP", "BARON", "./baron", 100, "./baronlice.txt", None, None, None)


def transformSsMDP(tau = 1):
    isfeasible, objective_value, answers, P, P_targ, P_nontarg, LER_to_trajectories, var_to_var_desc, traj_mappings, action_specs, actions, state_specs, states, target_trajs, target_nbr_trajs, other_nbr_trajs, other_trajs, posets, poset_names = learnMDP("novice/actions.csv", "novice/attributes.csv", "novice/list.files", None, None, "planner/list.files", "Novice-Planner", tau, 10, SsMDP.constructOptimization, "SsMDP", "BARON", "./baron", 10000, "./baronlice.txt", None, None, None)
    new_trajectories, state_map, state_map_inv, state_index_map = T_MDP.transformBySsMDP(len(states), target_trajs + target_nbr_trajs + other_nbr_trajs + other_trajs, tau, answers, var_to_var_desc)
    T_MDP.transformedTrainingFilesFromSsMDP(action_specs, actions, state_specs, states, state_map, state_map_inv, state_index_map, new_trajectories, posets, poset_names, "Novice-Planner-BARON/Transformed")
