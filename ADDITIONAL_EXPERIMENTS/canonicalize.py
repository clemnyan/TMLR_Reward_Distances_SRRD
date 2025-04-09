""" This code runs EPIC, SARD, and DARD """

import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import numpy as np
import sys
from value_iteration import *
import img_utils
from utils import *
import pandas as pd
from copy import deepcopy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from collections import deque
import pickle
import time
import math
from scipy import stats
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity
from numpy.random import randn, seed
import multiprocessing


# UTILS
def generate_trajs(f_map, grid_size, num_trajs, policy, LEN_LIMIT):
    """
    we will assume an n*n grid
    # This code generates Trajectories from a given policy
    """
    seed()

    f_map = {tuple(f_map[i]):i for i in range(len(f_map))}
    p_dist = [sum(policy[:i+1]) for i in range(len(policy))]
    possible_starts = [(0, 0), (3, 0), (0, 2), (1, 2)]
    action = [0, 1, 2, 3]  # corresponds with ['up', 'right', 'down', 'left']

    store = []
    count = 1

    while count <= num_trajs:

        curr = possible_starts[np.random.randint(len(possible_starts))]
        x, y = curr # get coordinates
        T = []

        while (x, y) != (grid_size - 1, grid_size - 1):
            is_success = False

            while not is_success:
                val = np.random.randint(101)
                for i, v in enumerate(p_dist):
                    if val <= v:
                        direction = action[i]
                        break;

                new_x, new_y = x, y
                if direction == 0: # try up
                    new_x, new_y = x - 1, y
                elif direction == 1: # try right
                    new_x, new_y = x, y + 1
                elif direction == 2: # try down
                    new_x, new_y = x + 1, y
                elif direction == 3:  # try left
                    new_x, new_y = x, y -1

                if (0 <= new_x <= grid_size -1) and (0 <= new_y <= grid_size -1):
                    is_success = True
                    T.append((f_map[(x, y)], direction))
                    x, y = new_x, new_y

        T.append((f_map[(x, y)], direction))

        if LEN_LIMIT[0] <= len(T) <= LEN_LIMIT[1]:
            store.append(T)
            count += 1

    return store


def policy_rollover (state_rewards, grid_size, num_trajs, policy, \
        shaped_graph, type):
    """
    we will assume an n*n grid
    # This code generates Trajectories from a given 4-directional policy
    """
    seed()

    shaped_mp = {tuple(i[:3]):i[3] for i in shaped_graph}

    p_dist = [sum(policy[:i+1]) for i in range(len(policy))]
    possible_starts = [(0, 0)]
    action = [0, 1, 2, 3]  # corresponds with ['up', 'right', 'down', 'left']

    store, shaped_store = [], []   #store for shaped and unshaped rewards
    count = 1
    MAX_SIZE = 60  #Maximum poaaible length of a trajectory

    while count <= num_trajs:

        curr = possible_starts[np.random.randint(len(possible_starts))]
        curr_len, T = 0, []
        x, y = curr # get coordinates

        while (curr_len < MAX_SIZE) and ((x, y) != (grid_size - 1, \
                grid_size - 1)):

            is_success = False
            while not is_success:
                val = np.random.randint(101)
                for i, v in enumerate(p_dist):
                    if val <= v:
                        direction = action[i]
                        break;

                new_x, new_y = x, y
                if direction == 0: # try up
                    new_x, new_y = x - 1, y
                elif direction == 1: # try right
                    new_x, new_y = x, y + 1
                elif direction == 2: # try down
                    new_x, new_y = x + 1, y
                elif direction == 3:  # try left
                    new_x, new_y = x, y -1


                if type == 'random_rollover':
                    """
                    Means we gonna take a random action to a random state
                        # will do this 50% of the time.
                    """
                    if np.random.randint(0, 100) <=70:
                        direction = 4
                        unfeasible = False

                        while unfeasible == False:
                            new_x, new_y = np.random.randint(0, grid_size), \
                                np.random.randint(0, grid_size)
                            # only considering unfeasible transitions here
                            if (new_x, new_y) not in ((x-1, y), (x, y+1), \
                                    (x+1, y), (x, y-1)):

                                unfeasible = True

                if (0 <= new_x <= grid_size -1) and (0 <= new_y <= grid_size -1):
                    is_success = True
                    T.append(((x, y), direction))
                    x, y = new_x, new_y
                    curr_len += 1

        T.append(((x, y), direction))

        # Now add these trajectories to the appropriate stores and append
        unshaped_T, shaped_T = [], []

        for i in range(len(T) -1):
            s, act = T[i]
            sp, _ = T[i+1]
            shaped_T.append((s, act, sp, state_rewards[sp]))
            unshaped_T.append((s, act, sp, shaped_mp[(s, act, sp)]))

        # now append to store
        store.append(shaped_T)
        shaped_store.append(unshaped_T)
        count += 1

    return store, shaped_store



def generate_triples (rewards, f_map, grid_size):
    # generates reward triples in the form r(s, a, s_p)
    r_map = {tuple(f_map[i]):(i, rewards[i]) for i in range(len(f_map))}

    store = []
    for i in range(grid_size):
        for j in range(grid_size):

            rew_from = r_map[(i, j)]

            # try up action
            x, y = i - 1, j
            if 0 <= x <= grid_size -1:
                rew_to = r_map[(x, y)]
                # in the form (s, a, s_p, r(s, a, s_p))
                store.append([rew_from[0], 0, rew_to[0], rew_to[1]])

            # try right action
            x, y = i, j + 1
            if 0 <= y <= grid_size -1:
                rew_to = r_map[(x, y)]
                # in the form (s, a, s_p, r(s, a, s_p))
                store.append([rew_from[0], 1, rew_to[0], rew_to[1]])

            # try down action
            x, y = i + 1, j
            if 0 <= x <= grid_size -1:
                rew_to = r_map[(x, y)]
                # in the form (s, a, s_p, r(s, a, s_p))
                store.append([rew_from[0], 2, rew_to[0], rew_to[1]])

            # try left action
            x, y = i, j - 1
            if 0 <= y <= grid_size - 1:
                rew_to = r_map[(x, y)]
                # in the form (s, a, s_p, r(s, a, s_p))
                store.append([rew_from[0], 3, rew_to[0], rew_to[1]])

    return store

def group_by_origin (rew_triples):
    mp = {}

    for r in rew_triples:
        if r[0] not in mp:
            mp[r[0]] = [r]
        else:
            mp[r[0]].append(r)

    return mp


def canonicalize_epic(triples, gamma, shrink, index_zeros):

    D = len(triples)    # This is the coverage

    # find the total number of states and actions to compute Ds *Da
    Ds, Da = set(), set()
    for t in triples:
        s, a, sp, r = t
        Ds.add(s)
        Ds.add(sp)
        Da.add(a)

    Da = 1
    N = len(Ds)*Da

    mp = group_by_origin(triples)
    rew_map = {tuple(i[:3]):i[3] for i in triples}

    # compute reward shift - done for all the triples
    shift_rew = 0
    for x in mp:
        for r in mp[x]:
            shift_rew += r[3]
    shift_rew /= (N*N)
    #shift_rew /= (state_size * action_size * state_size)

    new_triples = [] # where we store canonicalized rewards
    for triple in triples:
        # we want to compare with the scenario where rewards have not been seen
        s1, a, s2, _ = triple
        # compute expected next
        exp_next = 0
        for x in mp:
            if x == s2:
                for r in mp[x]:
                    exp_next += r[3]
        exp_next /= N

        # compute expected from
        exp_from = 0
        for x in mp:
            if x == s1:
                for r in mp[x]:
                    exp_from += r[3]

        exp_from /= N
        #exp_from /= (state_size * action_size)
        # canocicalize here
        can_rew = rew_map[(s1, a, s2)] + gamma*exp_next - exp_from - \
                gamma*shift_rew
        new_triples.append((s1, a, s2, can_rew))

    indices = set(index_zeros)
    if shrink:  # if we should shrink based on index
        new_arr = []
        for i, r in enumerate(new_triples):
            if i in indices:
                new_arr.append(r)
        return new_arr
    else:
        return new_triples


def canonicalize_dard (triples, weights_mp, gamma, shrink, index_zeros):
    """ Need to write this tonight, then compare with the  other algorithms """
    """ The goal should be to deploy with the DRONE DATA SET """
    """ First finish analysis on the drone scenario, then incorporate to the SC2 method """

    D = len(triples)    # This is the coverage

    # find the total number of states and actions to compute Ds *Da
    Ds, Da = set(), set()
    for t in triples:
        s, a, sp, r = t
        Ds.add(s)
        Ds.add(sp)
        Da.add(a)

    Da = 1
    N = len(Ds)*Da

    mp = group_by_origin(triples)
    rew_map = {tuple(i[:3]):i[3] for i in triples}

    # compute reward shift - done for all the triples
    shift_rew, shift_count, shift_wt = 0, 0, 0
    for x in mp:
        for r in mp[x]:
            wt = weights_mp[tuple(r[:3])]
            shift_rew += (r[3]*wt)
            shift_count += 1
            shift_wt += wt

    if shift_wt != 0:
        shift_rew *= (shift_count/shift_wt)
    shift_rew /= (N*N)

    new_triples = [] # where we store canonicalized rewards
    for triple in triples:
        # we want to compare with the scenario where rewards have not been seen
        s1, a, s2, _ = triple
        # compute expected next
        exp_next, next_count, next_wt = 0, 0, 0
        for x in mp:
            if x == s2:
                for r in mp[x]:
                    wt = weights_mp[tuple(r[:3])]
                    exp_next += (r[3] * wt)
                    next_count += 1
                    next_wt += wt

        if next_wt != 0:
            exp_next *= (next_count/next_wt)
        exp_next /= N

        # compute expected from
        exp_from, from_count, from_wt = 0, 0, 0
        for x in mp:
            if x == s1:
                for r in mp[x]:
                    wt = weights_mp[tuple(r[:3])]
                    exp_from += (r[3]*wt)
                    from_count += 1
                    from_wt += wt

        if from_wt != 0:
            exp_from *= (from_count/from_wt)
        exp_from /= N
        #exp_from /= (state_size * action_size)
        # canocicalize here
        can_rew = rew_map[(s1, a, s2)] + gamma*exp_next - exp_from - \
                gamma*shift_rew

        new_triples.append((s1, a, s2, can_rew))

    indices = set(index_zeros)
    if shrink:  # if we should shrink based on index
        new_arr = []
        for i, r in enumerate(new_triples):
            if i in indices:
                new_arr.append(r)
        return new_arr
    else:
        return new_triples

def helper_get_mean(s, X1, trans_mp, CONST):
    E_X1_s, count_X1_s = 0, CONST   # get E_X1_s
    for state in X1:
        if state in trans_mp:
            if s in trans_mp[state]:
                for t1 in trans_mp[state][s]:
                    E_X1_s += t1[3]
                    count_X1_s += 1
    E_X1_s /= count_X1_s
    return E_X1_s

def get_mean_helper_two (s, X2, trans_mp, CONST):
    E_s_X2, count_s_x2 = 0, CONST  #get E_s_X2
    if s in trans_mp:
        for ns in trans_mp[s]:
            X2.add(ns)
            if ns in trans_mp[s]:
                for t1 in trans_mp[s][ns]:
                    E_s_X2 += t1[3]
                    count_s_x2 += 1
    E_s_X2 /= count_s_x2
    return E_s_X2, X2


def get_trans_mp (triples):
    X, Xp, E_X_Xp = set(), set(), 0
    # create a map of all the possible tranistions such that you have"
    #{key :{next_key:transition}
    trans_mp = {}
    for t in triples:
        s, a, sp, r = t
        # Easy computations
        E_X_Xp += r
        X.add(s)
        Xp.add(sp)
        # if s not in trans_map, insert it
        if s not in trans_mp:
            trans_mp[s] = {}
        # get the map containing s as key
        next_mp = trans_mp[s]
        # check if next_state is in next_mp:
        if sp not in next_mp:
            next_mp[sp] = set()
        # now add to next_mp
        next_mp[sp].add((s, a, sp, r))

    E_X_Xp /= len(triples)
    return trans_mp, E_X_Xp, X, Xp


def canonicalize_sard(triples, gamma, shrink, index_zeros):
    """ This is the SARD metric: Sparsity Agnostic Reward Distance """

    CONST = 0.000000000000001
    trans_mp, E_X_Xp, X, Xp = get_trans_mp (triples)
    new_triples = []

    for t in triples:
        s, a, sp, r = t
        X1, X2 = set(), set()

        E_sp_X1, X1 = get_mean_helper_two (sp, X1, trans_mp, CONST)
        E_s_X2, X2 = get_mean_helper_two (s, X2, trans_mp, CONST)

        E_X_s = helper_get_mean(s, X, trans_mp, CONST)
        E_Xp_s = helper_get_mean(s, Xp, trans_mp, CONST)
        E_X2_s = helper_get_mean(s, X2, trans_mp, CONST)
        E_X1_s = helper_get_mean(s, X1, trans_mp, CONST)

        # canonical equation
        cr = r + gamma*E_sp_X1 - E_s_X2 - gamma*E_X_Xp + (gamma**2)*E_X1_s - \
                (gamma**2)*E_Xp_s + gamma*E_X_s - gamma*E_X2_s
        new_triples.append((s, a, sp, cr))

    indices = set(index_zeros)
    if shrink:  # if we should shrink based on index
        new_arr = []
        for i, r in enumerate(new_triples):
            if i in indices:
                new_arr.append(r)
        return new_arr
    else:
        return new_triples




def canonicalize_sard_1(triples, gamma, shrink, index_zeros):
    """ This is the SARD metric: Sparsity Agnostic Reward Distance """

    CONST = 0.000000000000001
    trans_mp, E_X_Xp, X, Xp = get_trans_mp (triples)
    new_triples = []

    for t in triples:
        s, a, sp, r = t
        X1, X2 = set(), set()

        E_sp_X1, X1 = get_mean_helper_two (sp, X1, trans_mp, CONST)
        E_s_X2, X2 = get_mean_helper_two (s, X2, trans_mp, CONST)

        E_X_s = helper_get_mean(s, X, trans_mp, CONST)
        E_Xp_s = helper_get_mean(s, Xp, trans_mp, CONST)
        E_X2_s = helper_get_mean(s, X2, trans_mp, CONST)
        E_X1_s = helper_get_mean(s, X1, trans_mp, CONST)

        # canonical equation
        cr = r + gamma*E_sp_X1 - E_s_X2 - gamma*E_X_Xp + (gamma**2)*E_X1_s - \
                (gamma**2)*E_Xp_s + gamma*E_X_s - gamma*E_X2_s
        new_triples.append((s, a, sp, cr))

    indices = set(index_zeros)
    if shrink:  # if we should shrink based on index
        new_arr = []
        for i, r in enumerate(new_triples):
            if i in indices:
                new_arr.append(r)
        return new_arr
    else:
        return new_triples
