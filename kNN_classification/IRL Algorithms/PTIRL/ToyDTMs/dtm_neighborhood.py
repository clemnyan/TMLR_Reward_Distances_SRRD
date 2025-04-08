#!/usr/bin/python3
#
# Name: dtm_neightborhood.py
# Author: Eugene Santos Jr.
# Date: 2017-11-12
# Project: DTM
# Copyright: Eugene Santos Jr.
#
import pydtm
import copy
import datetime
import itertools
import random

# Makes an efficient graphical representation of a dtm
def DTM_makeGraph(dtm):
    graph = {}
    # 0 - indexed by [source1][action][destination2]
    # 1 - [s][a] - list of triples with [s][a][d]
    # 2 - [s][d] - list of triples with [s][a][d]
    # 3 - [d][s] - list of triples with [s][a][d]
    # 4 - [s] - list of triples with [s][a][d]
    graph[0] = {}
    graph[1] = {}
    graph[2] = {}
    graph[3] = {}
    graph[4] = {}
    
    for idx, trip in enumerate(dtm.triples):
        try:
            graph[0][trip.source]
        except KeyError:
            graph[0][trip.source] = {}
            graph[1][trip.source] = {}
            graph[2][trip.source] = {}
            graph[4][trip.source] = list()
        graph[4][trip.source].append(idx)

        try:
            graph[0][trip.source][trip.action]
        except KeyError:
            graph[0][trip.source][trip.action] = {}
            graph[1][trip.source][trip.action] = list()
        graph[0][trip.source][trip.action][trip.dest] = idx
        graph[1][trip.source][trip.action].append(idx)

        try:
            graph[3][trip.dest]
        except KeyError:
            graph[3][trip.dest] = {}
        try:
            graph[3][trip.dest][trip.source]
        except KeyError:
            graph[3][trip.dest][trip.source] = list()
        graph[3][trip.dest][trip.source].append(idx)

        try:
            graph[2][trip.source][trip.dest]
        except KeyError:
            graph[2][trip.source][trip.dest] = list()
        graph[2][trip.source][trip.dest].append(idx)
    return (graph)


def DTM_computePaths(dtm, path_length, start_idx):
    results = list()
    queue = [ [ start_idx ] ]
    while (len(queue) > 0):
        item = queue[0]
        queue = queue[1:]
        item_length = int(len(item) / 3)
        if item_length == path_length: # done
            results.append(item)
            continue
        try:
            for aidx, values1 in graph[0][item[-1]].items():
                for didx, idx in values1.items():
                    if idx == -1:
                        continue
                    new_item = copy.deepcopy(item)
                    new_item.append(aidx)
                    new_item.append(didx)
                    queue.append(new_item)
        except KeyError:
            pass
    return(results)

# Compute distance between two trajectories
#   Returns -1 if lengths are different or starts and ends do not match
def DTM_trajectoryDistance(trajectory1, trajectory2):
    trajectory_length = len(trajectory1)
    if trajectory_length != len(trajectory2):
        return(-1)
    if trajectory1[0] != trajectory2[0]:
        return(-1)
    if trajectory1[-1] != trajectory2[-1]:
        return(-1)
    count = 0
    for idx in range(1,trajectory_length-1):
        if trajectory1[idx] != trajectory2[idx]:
            count += 1
            if idx % 2 == 1: # an action
                idx += 1
    return (count)

# Builds the neighborhood for a trajectory exluding those found in trajectories
#   graph contains the graphical represetation of the dtm.
# Starting state does NOT change in neighborhood
# Also, changes that are seperated by more than 2 positions are skipped as they do not provide additional information and are
#   reduendant with respect to perturbations from shorter combinations.
def DTM_makeNeighborhood(dtm, graph, trajectory, trajectories, m_bound):
# iterate through combinations of positions to change their values
    positions = list()
    for p in range(1, len(trajectory)):
        positions.append(str(p))
    positions = tuple(positions)
    traj_length = len(trajectory)
    neighborhood = list()
    for changes in range(1, m_bound): # building perturbation indices
        combos = list(itertools.combinations(positions, changes))
        for combo in combos:
    		# Skip combo if the spread between consecutive positions is greater than 2
            if len(combo) > 1:
                if max([ int(b) - int(a) for b,a in zip(combo[1:], combo[0:-1]) ]) > 2:
                    continue # skipping
            # eliminate all i in combo if i-1 in combo and is an action
            fixed = [ True ] * traj_length
            temp = [ int(val) for val in combo ]
            combo = list()
            for val in temp:
                fixed[val] = False
                if (val % 2 == 1) or not((val - 1) in temp):
                    combo.append(val)
#            print (fixed)

            length = len(combo)
#            print ('Combo -', end=' ')
#            print (combo)
            counter = [0] * length 
            for i in range(length):
                counter[i] = list()
            new_trajectory = copy.deepcopy(trajectory)
            position = 0
            while (position != -1): # Stopping condition
                not_viable = -1
                for perturbation in range(position, length):
                    t = combo[perturbation] # set up perturbation
                    if len(counter[perturbation]) == 0: # must setup and propagate forward
                        if t % 2 == 0: # a state
                            try:
                                counter[perturbation] = graph[1][new_trajectory[t - 2]][new_trajectory[t - 1]]
                                if len(counter[perturbation]) == 0: # no viable assignment
                                    not_viable = perturbation # need to later clear out from here to end
                                    break
                            except KeyError:
                                counter[perturbation] = list()
                                not_viable = perturbation # need to later clear out from here to end
                                break
                        else: # an action
                            if fixed[t + 1] == False: # following state is perturbable
                                try:
                                    counter[perturbation] = graph[4][new_trajectory[t - 1]]
                                except KeyError:
                                    counter[perturbation] = list()
                            else: # following state is fixed
                                try:
                                    counter[perturbation] = graph[2][new_trajectory[t - 1]][new_trajectory[t + 1]]
                                except KeyError:
                                    counter[perturbation] = list()
#                            print (graph[2][new_trajectory[t - 1]][new_trajectory[t + 1]])
                            if len(counter[perturbation]) == 0: # no viable assignment
                                not_viable = perturbation # need to later clear out from here to end
                                break

                    # Set up for the current perturbation
                    trip_idx = counter[perturbation][0]
                    trip = dtm.triples[trip_idx]
#                    print (t, end='=')
#                    print (len(counter[perturbation]))
#                    print (trip.source, end='+')
#                    print (trip.action, end='+')
#                    print (trip.dest)
                    if t % 2 == 0: # a state
                        new_trajectory[t] = trip.dest
                    else: # an action
                        new_trajectory[t] = trip.action
                        if fixed[t + 1] == False:
                            new_trajectory[t + 1] = trip.dest
#                print(new_trajectory)
#                print (not_viable)

                if not_viable != -1: # Clear to the end
                    for i in range(not_viable, length):
                        counter[i] = list()
                    position = not_viable - 1
                    if position == -1:
                        break # done
                else: # check viability of perturbation
                    viable = True
                    for i in range(0, traj_length - 2, 2):
                        try:
                            if graph[0][new_trajectory[i]][new_trajectory[i+1]][new_trajectory[i+2]] == -1:
                               viable = False
                               break
                        except KeyError:
                            viable = False
                            break
                    if viable:
                        if (new_trajectory != trajectory) and not(new_trajectory in neighborhood) and not(new_trajectory in trajectories):
                            neighborhood.append(new_trajectory)
#                            print (new_trajectory)
                        new_trajectory = copy.deepcopy(new_trajectory)
                    position = length - 1
                               
                # next perturbation
#                print (position)
                while True:
                    counter[position] = counter[position][1:]
#                    print ('Counter = ', end='')
#                    print (len(counter[position]))
                    if len(counter[position]) == 0: # empty
                        if position == 0:
                            position = -1 # done
                            break
                        counter[position] = list()
                        position -= 1
                    else:
                        break
    return (neighborhood)


def DTM_makeNeighborhoodOld(dtm, graph, trajectory, trajectories, m_bound):
# iterate through combinations of positions to change their values
    positions = list()
    for p in range(1, len(trajectory)):
        positions.append(str(p))
    positions = tuple(positions)
    neighborhood = list()
    for changes in range(1, m_bound):
        combos = list(itertools.combinations(positions, changes))
        for combo in combos:
            length = len(combo)
            temp = [ int(val) for val in combo ]
            combo = temp
#            print ('Combo -', end='')
#            print (combo)
            counter = [0] * length
            while True:
#                print (counter)
                new_trajectory = copy.deepcopy(trajectory)
                if len([val for idx, val in enumerate(counter) if val != trajectory[combo[idx]]]) != 0: # is different from original
                    for idx in range(length): 
                        new_trajectory[combo[idx]] = counter[idx]
#                    print (new_trajectory)
                    viable = True
                    for idx in range(length):
                        t = combo[idx]
#                        print (t)
                        if t % 2 == 0: # a state
                            try:
                                if graph[0][new_trajectory[t-2]][new_trajectory[t-1]][new_trajectory[t]] == -1:
                                    viable = False
                                    break
                            except KeyError:
                                viable = False
                                break
                            if t < len(trajectory) - 1:
                                try:
                                    if graph[0][new_trajectory[t]][new_trajectory[t+1]][new_trajectory[t+2]] == -1:
                                        viable = False
                                        break
                                except KeyError:
                                    viable = False
                                    break
                        else: # an action
                            try:
                                if graph[0][new_trajectory[t-1]][new_trajectory[t]][new_trajectory[t+1]] == -1:
                                    viable = False
                                    break
                            except KeyError:
                                viable = False
                                break
#                    print(viable)
                    if viable:
                        if not (new_trajectory in trajectories):
                            neighborhood.append(copy.deepcopy(new_trajectory))
                p = 0
                while True:
                    counter[p] += 1
                    if combo[p] % 2 == 0: # a state
                        limit = dtm.num_cs
                    else: # an action
                        limit = dtm.num_a
                    if counter[p] == limit:
                        counter[p] = 0
                        p += 1
                    else:
                        break
                    if p == length:
                        break
                if p == length: # done
                    break
    return (neighborhood)


def DTM_makeNeighborhoodOlder(dtm, graph, trajectory, trajectories, m_bound):
    neighborhood = list()
    if m_bound < 1:
        return(neighborhood)
    trajectory_length = len(trajectory)
    queue = [ ( [ trajectory[0] ], 0) ]
    while (len(queue) > 0):
        item = queue[0]
#        print ('Popping -- ', end='')
#        print (item)
        queue = queue[1:]
        item_trajectory = item[0]
        item_pos = len(item_trajectory)
        if item_pos == trajectory_length: # check to see if valid
            if item_trajectory[-1] == trajectory[-1]:
                if item_trajectory in trajectories: # skip
                    continue
#                print (item_trajectory)
#                print (item[1])
                neighborhood.append(item_trajectory)
            continue
        item_count = item[1]
#        print ('------', end=' ')
#        print (item_trajectory[-1])
        for aidx in range(dtm.num_a):
            for sidx in range(dtm.num_cs):
                try:
                    if graph[0][item_trajectory[-1]][aidx][sidx] != -1:
                        new_trajectory = copy.deepcopy(item_trajectory)
                        new_trajectory.append(aidx)
                        new_trajectory.append(sidx)
#                        print (new_trajectory)
                        if aidx != trajectory[item_pos] or sidx != trajectory[item_pos + 1]:
                            if item_count + 1 > m_bound: # no longer considered
                                continue 
                            new_item = ( new_trajectory, item_count + 1 )
                        else:
                            new_item = ( new_trajectory, item_count )
                        queue.append(new_item)
                except KeyError:
                    pass
    return (neighborhood)

# This generates a random (weighted by probability) trajectory of at most target_length states
def DTM_generateTrajectory(dtm, graph, start_state_idx, target_length):
    # Returns a trajectory of (s, a, s', ...)
    new_traj = list()
    new_traj.append(start_state_idx)
    while len(new_traj) < target_length * 2 - 1:
        last_state = new_traj[-1]
#        print ('last_state = ', end='')
#        print (last_state)
        try:
            if len(graph[4][last_state]) == 0: # must stop, cannot generate anymore
                break
        except KeyError:
            break
        # first, randomly select an action from the available ones
#        print ('graph[4][last_state] = ', end='')
#        print (graph[4][last_state])
        actions = list(set([ dtm.triples[idx].action for idx in graph[4][last_state] ]))
        choice = random.randrange(len(actions))
        new_traj.append(actions[choice])
        dart = random.choice(graph[1][last_state][actions[choice]])
        dest = dtm.triples[dart].dest
        new_traj.append(dest)
    return(new_traj)

# This generates neighborhoods based on branching out from the given trajectory at each possible
#   brnaching location and then generate a random trajectory to fill in from that point
def DTM_makeBranchingNeighborhood(dtm, graph, trajectory, trajectories):    
    # Returns a list of new trajectories
    new_trajs = list()
    for pos, idx in enumerate(trajectory):
        if pos % 2 == 1: # an action
            # check if branching possible
            try:
                if len(graph[1][trajectory[pos-1]][trajectory[pos]]) > 1: # there is an alternative
                    traj = trajectory[:pos + 1]
                    print ('pos = ' + str(pos) + '\tpartial traj = ', end='')
                    print (traj)
                    mass = (1.0 - dtm.triples[graph[0][trajectory[pos-1]][trajectory[pos]][trajectory[pos+1]]].prob)
                    if mass == 0: # No viable non-zero probability alternative path
                        continue
                    dart = random.random() / mass
                    tidxs = list(set(graph[1][trajectory[pos-1]][trajectory[pos]]) - set([ graph[0][trajectory[pos-1]][trajectory[pos]][trajectory[pos+1]] ]))
                    print ('tidxs = ', end='')
                    print (tidxs)
                    dart = random.choice(tidxs)
                    dest = dtm.triples[dart].dest
                    suffix = DTM_generateTrajectory(dtm, graph, dest, len(trajectory) - pos - 1)
#                    print ('suffix = ', end='')
#                    print (suffix)
                    traj += suffix
#                    print (traj)
                    new_trajs.append(traj)
            except KeyError:
                pass
    return (new_trajs)


# Builds all the proper suffixes for a trajectory with states as starts
def DTM_makeTrajectoryProperSuffixes(trajectory):
    suffixes = set()
    for idx in range(2, len(trajectory), 2):
        suffixes.add(trajectory[idx:])
    return (suffixes)


# Build all the proper suffixes for a set of trajectories with states as starts
def DTM_makeTrajectoriesProperSuffixes(trajectories):
    # Returns list of trajectory sequences
    print ('Building trajectory proper suffixes...')
    new_trajs = set()
    for traj in trajectories:
        if tuple(traj) in new_trajs:
            continue
        new_trajs.add(tuple(traj))
        new_trajs |= DTM_makeTrajectoryProperSuffixes(tuple(traj))
    return ([ list(traj) for traj in new_trajs ])
    

def test(dtmfile):
    print(datetime.datetime.now())
    print ('Loading dtm...')
    dtm = pydtm.pydtm(dtmfile)
    print(datetime.datetime.now())
    print ('Building graph object from dtm...')
    graph = DTM_makeGraph(dtm)
    print(datetime.datetime.now())
    print ('Computing neighborhood for trajectory...')
    traj = [ 0,0,0,0,0,0,0,0,0 ]
    nei = DTM_makeNeighborhood(dtm, graph, traj, [], 5)
    print(nei)
    print(datetime.datetime.now())


# Given 4 trajectory sets:
#   (a) Observed target trajectories
#   (b) Target neighborhood trajectories
#   (c) Observed non-target trajectories
#   (d) Non-target neighborhood trajectories
# perform the following operations in order:
#   (c) = (c) - (a) - (b) - (d)
#   (b) = (b) - (a) - (d)
#   (d) = (d) - (a)
def DTM_filterTrajectorySets(observed_target_trajs, target_nbr_trajs, observed_others_trajs, others_nbr_trajs):
    # Returns ( new_observed_target_trajs, new_target_nbr_trajs, new_objserved_others_trajs, new_others_nbr_trajs )
    print ('Filtering observed trajectories and neighborhood trajectories from both target and others users...')
    new_observed_target_trajs = copy.deepcopy(observed_target_trajs)
    new_others_nbr_trajs = others_nbr_trajs - observed_target_trajs - target_nbr_trajs - observed_others_trajs
    new_target_nbr_trajs = target_nbr_trajs - observed_target_trajs - observed_others_trajs
    new_observed_others_trajs = observed_others_trajs - observed_target_trajs
    return ( ( new_observed_target_trajs, new_target_nbr_trajs, new_observed_others_trajs, new_others_nbr_trajs ) )
