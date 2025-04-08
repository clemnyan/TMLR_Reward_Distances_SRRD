'''
Implementation of maximum entropy inverse reinforcement learning in
  Ziebart et al. 2008 paper: Maximum Entropy Inverse Reinforcement Learning
  https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf

Acknowledgement:
  This implementation is largely influenced by Matthew Alger's maxent implementation here:
  https://github.com/MatthewJA/Inverse-Reinforcement-Learning/blob/master/irl/maxent.py

By Yiren Lu (luyirenmax@gmail.com), May 2017
'''
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
import bouncing_balls as bb


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


def get_random_trajs(policy, grid_size, type, shaped_mp, \
        num_trajs, MAX_SIZE):

    p_dist = [sum(policy[:i+1]) for i in range(len(policy))]
    possible_starts = [(0, 0)]
    action = [0, 1, 2, 3]  # corresponds with ['up', 'right', 'down', 'left']

    store = []   #store for shaped and unshaped rewards
    count = 1

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
        shaped_T = []

        for i in range(len(T) -1):
            s, act = T[i]
            sp, _ = T[i+1]

            shaped_T.append((s, act, sp, shaped_mp[(s, act, sp)]))

        # now append to store
        store.append(shaped_T)
        count += 1

    return store

def policy_rollover(grid_size, num_trajs, policy, \
        shaped_graph, type, MAX_SIZE):
    """
    we will assume an n*n grid
    # This code generates Trajectories from a given 4-directional policy
    """
    seed()
    shaped_mp = {tuple(i[:3]):i[3] for i in shaped_graph}
    store = get_random_trajs(policy, grid_size, type, \
        shaped_mp, num_trajs, MAX_SIZE)
    return store


def policy_rollover_bb(grid_size, NUM_TRAJS, old_policy, \
        unshaped_graph, shaped_graph, rollover_type, MIN_DIST, LEN_LIMIT):
    """
    we will assume an n*n grid
    # This code generates Trajectories from a given 4-directional policy
    """
    seed()

    if rollover_type == "random_rollover":
        policy = old_policy + [100]   # now have a higher ratio of choosing random actions
    else:
        policy = old_policy + [0]

    # create new game
    num_obstacles = 3
    new_game = bb.Game(num_obstacles, policy, grid_size, MIN_DIST, LEN_LIMIT)

    # store shaped and unshaped rewards
    unshaped_store, shaped_store  = [], []

    for k in range(NUM_TRAJS):
        shaped_temp, unshaped_temp = [], []

        T = new_game.play_game()
        for k in T:
            s, a, sp = k
            r = shaped_graph[k]
            shaped_temp.append((s, a, sp, r))

            # now we should shape the reward
            #r1 = state_rewards[sp] + 2*state_rewards[s]
            r1 = unshaped_graph[k]
            unshaped_temp.append((s, a, sp, r1))

        unshaped_store.append(unshaped_temp)
        shaped_store.append(shaped_temp)

    # generate games using bb domain

    return unshaped_store, shaped_store



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



def canonicalize_epic(BM_threshold, triples, gamma, shrink, index_zeros):

    CONST = 0.00000000000001
    # first, compute BM

    B_M = set()
    trans_mp = {}

    for s, a, sp, r in triples:
        B_M.add(s)
        #B_M.add(a)  # a = 1 in deterministic environments
        B_M.add(sp)
        trans_mp[(s, sp)] = r

    if len(B_M) > BM_threshold:
        B_M = random.sample(B_M, BM_threshold)

    N_M = len(B_M) + CONST  # incase of zeros

    # compute N_M^2
    E_X_XP = [trans_mp[(x, xp)] for x in B_M for xp in B_M if (x, xp) in trans_mp]
    E_X_XP = sum(E_X_XP)/(N_M**2)

    new_triples = []
    for s, a, sp, r in triples:
        E_s_XP = [trans_mp[(s, x)] for x in B_M if (s, x) in trans_mp]
        E_sp_XP = [trans_mp[(sp, x)] for x in B_M if (sp, x) in trans_mp]

        E_s_XP = np.sum(E_s_XP)/N_M
        E_sp_XP = np.sum(E_sp_XP)/N_M

        cr = r + gamma*E_sp_XP - E_s_XP - E_X_XP
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




def canonicalize_dard(BM_threshold, triples, gamma, shrink, index_zeros):

    CONST = 0.00000000000001
    # first, compute BM

    B = set()
    trans_mp = {}

    for s, a, sp, r in triples:
        B.add(s)
        #B_M.add(a)  # a = 1 in deterministic environments
        B.add(sp)
        trans_mp[(s, sp)] = r

    if len(B) > BM_threshold:
        B = random.sample(B, BM_threshold)

    new_triples = []
    for s, a, sp, r in triples:

        B_M = [x for x in B if (s, x) in trans_mp]

        N_M = len(B_M) + CONST  # incase of zeros

        # compute N_M^2
        E_X1_X2 = [trans_mp[(x, xp)] for x in B_M for xp in B_M if (x, xp) in trans_mp]
        E_X1_X2 = sum(E_X1_X2)/(N_M**2)

        E_s_X1 = [trans_mp[(s, x)] for x in B_M if (s, x) in trans_mp]
        E_sp_X2 = [trans_mp[(sp, x)] for x in B_M if (sp, x) in trans_mp]

        E_s_X1 = np.sum(E_s_X1)/N_M
        E_sp_X2 = np.sum(E_sp_X2)/N_M

        cr = r + gamma*E_sp_X2 - E_s_X1 - E_X1_X2
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


def predict_reward(s, sp, model):

    x, y = s
    x1, y1 = sp

    new_x = [[x, y, x1, y1]]
    r = model.predict(new_x)[0][0]
    return r


def canonicalize_epic_regress(triples, gamma, model):
    """ Regress values that are unknown """
    trans_mp, X3, X4 = get_rew_map(triples)

    X = set()
    for i in X3:
        X.add(i)
    for i in X4:
        X.add(i)

    # compute reward shift - done for all the triples
    shift_rew = []
    mp2 = {}

    for s in X:
        temp_s = []
        for sp in X:
            if (s, sp) in trans_mp:
                shift_rew.append(trans_mp[(s, sp)])
                temp_s.append(trans_mp[(s, sp)])
            else:
                # lets use the model to predict reward
                r_pred = predict_reward(s, sp, model)
                shift_rew.append(r_pred)
                temp_s.append(r_pred)
                trans_mp[(s, sp)] = r_pred  # alsp update trans_mp

        mp2[s] = np.mean(temp_s)

    all_mean = np.mean(shift_rew)

    new_triples = []
    for t in triples:
        s, a, sp, r = t

        new_r = r + gamma*mp2[sp] - mp2[s] - gamma*all_mean

        new_triples.append([s, sp, new_r])

    return new_triples




def canonicalize_u_epic(triples, gamma, shrink, index_zeros):
    """ This one uses unbiased estimates """
    CONST = 0.00000000000001
    D = len(triples)    # This is the coverage

    indices = set(index_zeros)

    # find the total number of states and actions to compute Ds *Da
    Ds, Da = set(), set()
    for t in triples:
        s, a, sp, r = t
        Ds.add(s)
        Ds.add(sp)
        Da.add(a)

    mp = group_by_origin(triples)
    rew_map = {tuple(i[:3]):i[3] for i in triples}

    # compute reward shift - done for all the triples
    shift_rew = []
    for x in mp:
        for r in mp[x]:
            shift_rew.append(r[3])
    shift_rew = sum(shift_rew)/(len(shift_rew) + CONST)

    new_triples = [] # where we store canonicalized rewards
    for triple in triples:
        # we want to compare with the scenario where rewards have not been seen
        s1, a, s2, _ = triple
        # compute expected next
        exp_next = []
        for x in mp:
            if x == s2:
                for r in mp[x]:
                    exp_next.append(r[3])
        exp_next = sum(exp_next)/(len(exp_next) + CONST)

        # compute expected from
        exp_from = []
        for x in mp:
            if x == s1:
                for r in mp[x]:
                    exp_from.append(r[3])
        exp_from = sum(exp_from)/(len(exp_from) + CONST)

        #exp_from /= (state_size * action_size)
        # canocicalize here
        can_rew = rew_map[(s1, a, s2)] + gamma*exp_next - exp_from - \
                gamma*shift_rew
        new_triples.append((s1, a, s2, can_rew))

    if shrink:  # if we should shrink based on index
        new_arr = []
        for i, r in enumerate(new_triples):
            if i in indices:
                new_arr.append(r)
        return new_arr
    else:
        return new_triples


def helper_get_mean(X, X1, trans_mp, CONST):

    def op1():
        E_X = 0

        for s in X:
            if s in trans_mp:
                for s1 in trans_mp[s]:
                    if s1 in X1:
                        for t in trans_mp[s][s1]:
                            E_X += t[3]

        count1 = CONST + len(X) * len(X1)
        E_X /= count1

        return E_X

    return op1()


def u_helper_get_mean(X, X1, trans_mp, CONST):

    def op1():
        E_X, count = 0, CONST

        for s in X:
            if s in trans_mp:
                for s1 in trans_mp[s]:
                    if s1 in X1:
                        for t in trans_mp[s][s1]:
                            E_X += t[3]
                            count += 1
        E_X /= count
        return E_X

    return op1()



def get_trans_mp (triples):
    CONST = 0.00000000000001
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

    #E_X_Xp /= (len(triples) + CONST)
    const1 = len(X) * len(Xp) + CONST
    E_X_Xp /= const1

    return trans_mp, E_X_Xp, X, Xp



def get_rew_map(triples):
    """ Returns an unbiased estimate """

    X, Xp = set(), set()
    # create a map of all the possible tranistions such that you have"
    #{key :{next_key:transition}
    trans_mp = {}
    for t in triples:
        s, a, sp, r = t
        # Easy computations
        trans_mp[(s, sp)] = r
        X.add(s)
        Xp.add(sp)

    return trans_mp, X, Xp


def u_get_trans_mp (triples):
    """ Returns an unbiased estimate """

    CONST = 0.00000000000001
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

    E_X_Xp /= (len(triples) + CONST)

    return trans_mp, E_X_Xp, X, Xp



def helper_get_sets (X1, trans_mp, CONST, cache_mp):

    X5, E_X1_X5 = set(), 0

    for s in X1:
        temp_X5, sum_X1_X5 = set(), 0
        if s not in cache_mp:
            if s in trans_mp:
                for ns in trans_mp[s]:
                    temp_X5.add(ns)
                    X5.add(ns)
                    #if ns in trans_mp[s]:
                    for t1 in trans_mp[s][ns]:
                        E_X1_X5 += t1[3]
                        sum_X1_X5 += t1[3]

            cache_mp[s] = [temp_X5, sum_X1_X5]
        else:
            # Retrieve values from cache
            cached_temp_X5, cached_sum_X1_X5 = cache_mp[s]
            X5.update(cached_temp_X5)
            E_X1_X5 += cached_sum_X1_X5

    count1 = CONST + len(X1) * len(X5)
    E_X1_X5 /= count1

    return E_X1_X5, X5


def u_helper_get_sets (X1, trans_mp, CONST, cache_mp):

    """ Uses unbiased estimate implemtation """

    X5, E_X1_X5, count = set(), 0, CONST

    for s in X1:
        temp_X5, sum_X1_X5, count_X1_X5 = set(), 0, 0
        if s not in cache_mp:
            if s in trans_mp:
                for ns in trans_mp[s]:
                    temp_X5.add(ns)
                    X5.add(ns)
                    #if ns in trans_mp[s]:
                    for t1 in trans_mp[s][ns]:
                        E_X1_X5 += t1[3]
                        sum_X1_X5 += t1[3]
                        count += 1
                        count_X1_X5 += 1

            cache_mp[s] = [temp_X5, sum_X1_X5, count_X1_X5]
        else:
            # Retrieve values from cache
            cached_temp_X5, cached_sum_X1_X5, cached_count_X1_X5 = cache_mp[s]
            X5.update(cached_temp_X5)
            E_X1_X5 += cached_sum_X1_X5
            count += cached_count_X1_X5

    E_X1_X5 /= count
    return E_X1_X5, X5

def canonicalize_u_srrd(triples, gamma, shrink, index_zeros):
    """
    This is the SRRD metric: Sparsity Resilient Reward Distance

    Implements SRRD using unbiased estimates.
    """
    CONST = 0.000000000000001
    cache_mp = {}   # to catch sums in case
    trans_mp, E_X3_X4, X3, X4 = u_get_trans_mp(triples)

    new_triples = []

    for t in triples:
        s, a, sp, r = t

        E_sp_X1, X1 = u_helper_get_sets ([sp], trans_mp, CONST, cache_mp)
        E_s_X2, X2 = u_helper_get_sets ([s], trans_mp, CONST, cache_mp)
        E_X1_X5, X5 = u_helper_get_sets (X1, trans_mp, CONST, cache_mp)
        E_X2_X6, X6 = u_helper_get_sets (X2, trans_mp, CONST, cache_mp)

        E_X3_X6 = u_helper_get_mean(X3, X6, trans_mp, CONST)
        E_X4_X5 = u_helper_get_mean(X4, X5, trans_mp, CONST)

        # canonical equation
        cr = r + gamma*E_sp_X1 - E_s_X2 - gamma*E_X3_X4 + (gamma**2)*E_X1_X5 - \
                (gamma**2)*E_X4_X5 + gamma*E_X3_X6 - gamma*E_X2_X6
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


def canonicalize_VAL(triples, gamma, shrink, index_zeros):
    """ This is the SARD metric: Sparsity Agnostic Reward Distance """


    CONST = 0.000000000000001
    cache_mp = {}   # to catch sums in case
    trans_mp, _, _, _ = get_trans_mp(triples)

    new_triples = []

    for t in triples:
        s, a, sp, r = t

        E_sp_X1, X1 = helper_get_sets ([sp], trans_mp, CONST, cache_mp)
        E_s_X2, X2 = helper_get_sets ([s], trans_mp, CONST, cache_mp)
        E_X1_X5, X5 = helper_get_sets (X1, trans_mp, CONST, cache_mp)
        E_X2_X6, X6 = helper_get_sets (X2, trans_mp, CONST, cache_mp)

        E_X3_X6 = helper_get_mean(X3, X6, trans_mp, CONST)
        E_X4_X5 = helper_get_mean(X4, X5, trans_mp, CONST)

        # canonical equation
        cr = r + gamma*E_sp_X1 - E_s_X2 - gamma*E_X3_X4 + (gamma**2)*E_X1_X5 - \
                (gamma**2)*E_X4_X5 + gamma*E_X3_X6 - gamma*E_X2_X6
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

def canonicalize_VAL(triples, gamma, shrink, index_zeros):
    R = {(i[0], i[1], i[2]):i[3] for i in triples}

    policy_mp = {}

    for t in triples:
        s, a, sp, r = t

        if s not in policy_mp:
            policy_mp[s] = set()

        policy_mp[s].add((a, sp))
        if sp not in policy_mp:
            policy_mp[sp] = set()

    #print(policy_mp)
    # now, lets run value iteration for each given state
    V = {i:0 for i in policy_mp}

    diff = sys.maxsize

    k = 0
    while diff > 10:
        #print("iteration = ", k, diff)
        k+=1
        oldV = deepcopy(V)

        for s in policy_mp:
            if s in policy_mp:
                next_action_states = policy_mp[s]

                nV = 0
                for x in next_action_states:
                    a, sp = x
                    nV += 1/(len(next_action_states)) * (R[(s, a, sp)] + gamma*V[sp])   #assumes a uniform policy

            V[s] = nV

        diff = np.sqrt(sum([(V[i] - oldV[i])**2 for i in V]))

    #print(V)
    new_triples = []

    for t in triples:
        s, a, sp, r = t
        nr = r + gamma * V[sp] - V[s]
        new_triples.append((s, a, sp, nr))

    indices = set(index_zeros)
    if shrink:  # if we should shrink based on index
        new_arr = []
        for i, r in enumerate(new_triples):
            if i in indices:
                new_arr.append(r)
        return new_arr
    else:
        return new_triples




def canonicalize_srrd(BM_threshold, triples, gamma, shrink, index_zeros):

    CONST = 0.00000000000001
    # first, compute BM

    BM = set()
    trans_mp = {}
    for s, a, sp, r in triples:
        BM.add(s)
        #B_M.add(a)  # a = 1 in deterministic environments
        BM.add(sp)
        trans_mp[(s, sp)] = r

    # randomly select B_M to be 50 max
    if len(BM) > BM_threshold:
        BM = random.sample(BM, BM_threshold)

    X3, X4 = set(), set()
    for s, a, sp, r in triples:
        if s in BM:
            X3.add(s)
        if sp in BM:
            X4.add(sp)

    # get E_X3_X4
    X3_X4 = [trans_mp[(x, xp)] for x in X3 for xp in X4 if (x, xp) in trans_mp]

    N3, N4 = len(X3) + CONST, len(X4) + CONST
    N3_N4 = N3*N4

    E_X3_X4 = sum(X3_X4)/N3_N4

    new_triples = []
    for s, a, sp, r in triples:

        X1, X2 = set(), set()
        sp_X1, s_X2 = [], []
        for x in BM:
            if (sp, x) in trans_mp:
                X1.add(x)
                sp_X1.append(trans_mp[(sp, x)])

            if (s, x) in trans_mp:
                X2.add(x)
                s_X2.append(trans_mp[(s, x)])

        X5, X6 = set(), set()
        X1_X5, X2_X6 = [], []
        for x in BM:
            for xs in X1:
                if (xs, x) in trans_mp:
                    X5.add(x)
                    X1_X5.append(trans_mp[(xs, x)])

            for xs in X2:
                if (xs, x) in trans_mp:
                    X6.add(x)
                    X2_X6.append(trans_mp[(xs, x)])

        X3_X6, X4_X5 = [], []

        for x in X3:
            for xs in X6:
                if (x, xs) in trans_mp:
                    X3_X6.append(trans_mp[(x, xs)])

        for x in X4:
            for xs in X5:
                if (x, xs) in trans_mp:
                    X4_X5.append(trans_mp[(x, xs)])

        N1 = len(X1) + CONST  # incase of zeros
        N2 = len(X2) + CONST  # incase of zeros
        N5 = len(X5) + CONST  # incase of zeros
        N6 = len(X6) + CONST  # incase of zeros

        E_s_X2 = sum(s_X2)/N2
        E_sp_X1 = sum(sp_X1)/N1
        E_X1_X5 = sum(X1_X5)/(N1*N5)
        E_X2_X6 = sum(X2_X6)/(N2*N6)
        E_X3_X6 = sum(X3_X6)/(N3*N6)
        E_X4_X5 = sum(X4_X5)/(N4*N5)

        cr = r + gamma*E_sp_X1 - E_s_X2 - gamma*E_X3_X4 + (gamma**2)*E_X1_X5 - \
                (gamma**2)*E_X4_X5 + gamma*E_X3_X6 - gamma*E_X2_X6

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


def populate(X3, X4, mp_s_sp, model, CONST):

    E_X3_X4 = []
    for x in X3:
        for x1 in X4:
            if (x, x1) in mp_s_sp:
                E_X3_X4.append(mp_s_sp[(x, x1)])
            else:
                E_X3_X4.append(predict_reward(x, x1, model))

    return sum(E_X3_X4)/(len(E_X3_X4) + CONST)

def get_next_set(mp, X1):
    X2 = set()
    for m in mp:
        s, sp = m
        if s in X1:
            X2.add(sp)
    return X2

def canonicalize_srrd_regress(triples, gamma, model):
    """ Regress values that are unknown """

    mp_s_sp, X3, X4 = get_rew_map(triples)

    CONST = 0.0000000000000001
    cache_mp = {}

    new_triples = []

    E_X3_X4 = populate(X3, X4, mp_s_sp, model, CONST)

    for t in triples:
        s, a, sp, r = t

        X1 = get_next_set(mp_s_sp, [sp])
        X2 = get_next_set(mp_s_sp, [s])
        X5 = get_next_set(mp_s_sp, X1)
        X6 = get_next_set(mp_s_sp, X2)

        E_sp_X1 = populate([sp], X1, mp_s_sp, model, CONST)

        E_s_X2 = populate([s], X2, mp_s_sp, model, CONST)

        E_X1_X5 = populate(X1, X5, mp_s_sp, model, CONST)

        E_X4_X5 = populate(X4, X5, mp_s_sp, model, CONST)

        E_X3_X6 = populate(X3, X6, mp_s_sp, model, CONST)

        E_X2_X6 = populate(X2, X6, mp_s_sp, model, CONST)
        # canonical equation
        cr = r + gamma*E_sp_X1 - E_s_X2 - gamma*E_X3_X4 + (gamma**2)*E_X1_X5 - \
                (gamma**2)*E_X4_X5 + gamma*E_X3_X6 - gamma*E_X2_X6
        new_triples.append([s, sp, cr])

    return new_triples




def canonicalize_dard_regress(triples, gamma, model):
    """ Regress values that are unknown """

    mp_s_sp, _ , _ = get_rew_map(triples)

    CONST = 0.0000000000000001
    new_triples = []

    for t in triples:
        s, a, sp, r = t

        X2 = get_next_set(mp_s_sp, [sp])
        X1 = get_next_set(mp_s_sp, [s])

        all_trans,  actions = set(), set()

        E_s_X1 = populate([s], X1, mp_s_sp, model, CONST)

        E_sp_X2 = populate([sp], X2, mp_s_sp, model, CONST)

        E_X1_X2 = populate(X1, X2, mp_s_sp, model, CONST)

        cr = r + gamma*E_sp_X2 - E_s_X1 - gamma*E_X1_X2
        new_triples.append([s, sp, cr])

    return new_triples



def canonicalize_u_dard(triples, weights_mp, gamma, shrink, index_zeros):
    """ This is the DARD metric, uses unbiased estimates  """

    CONST = 0.0000000000000001
    trans_mp, _, _, _ = get_trans_mp(triples)
    cache_mp = {}
    new_triples = []

    for t in triples:
        s, a, sp, r = t

        #C1, C2, C3 = CONST, CONST, CONST
        _, X2 = helper_get_sets ([sp], trans_mp, CONST, cache_mp)
        _, X1 = helper_get_sets ([s], trans_mp, CONST, cache_mp)

        all_trans, E_X1_X2,  actions = set(), 0, set()

        N_ALL = CONST
        for x in X1:
            if x in trans_mp:
                for y in X2:
                    if y in trans_mp[x]:
                        for trans in trans_mp[x][y]:
                            s1, a1, sp1, r1 = trans
                            E_X1_X2 += r1
                            actions.add(a1)
                            N_ALL += 1

        E_sp_X2, E_s_X1, N_2, N_1 = 0, 0, CONST, CONST

        for x in X2:
            all_trans.add(x)
            if x in trans_mp[sp]:
                for trans in trans_mp[sp][x]:
                    s2, a2, sp2, r2 = trans
                    E_sp_X2 += r2
                    N_2 += 1

        for x in X1:
            all_trans.add(x)
            if x in trans_mp[s]:
                for trans in trans_mp[s][x]:
                    s3, a3, sp3, r3 = trans
                    E_s_X1 += r3
                    N_1 += 1

        cr = r + (gamma/N_2) *E_sp_X2 - (1/N_1)*E_s_X1 - (gamma/N_ALL)*E_X1_X2

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




def compare_rews(irl_rewards, can_original, sparse_epic, sparse_dard, \
    sparse_sard, index_zeros):
    """
    In this method, the goal is to compare each reward values extracted by
    each canonical methods.
    """
    #print(len(irl_rewards), len(can_original), len(sparse_epic), len(sparse_dard))

    indices = set(index_zeros)
    new_rewards = []

    for i, rew in enumerate(irl_rewards):
        if i in indices:
            new_rewards.append(rew)

    arr = []
    for i in range(len(new_rewards)):
        arr.append([new_rewards[i][3], can_original[i][3], \
            sparse_epic[i][3], sparse_dard[i][3], sparse_sard[i][3]])

    df = pd.DataFrame(arr, columns = ['original', 'canonical_orig', \
            'sample_epic', 'sample_dard', 'sample_sard'])

    return df, new_rewards



def compare_rews_new(irl_rewards, can_original, epic_1, dard_1, srrd_1,  \
    epic_2, dard_2, srrd_2, index_zeros):
    """
    In this method, the goal is to compare each reward values extracted by
    each canonical methods.
    """
    #print(len(irl_rewards), len(can_original), len(sparse_epic), len(sparse_dard))

    indices = set(index_zeros)
    new_rewards = []

    for i, rew in enumerate(irl_rewards):
        if i in indices:
            new_rewards.append(rew)

    arr = []
    for i in range(len(new_rewards)):
        arr.append([new_rewards[i][3], can_original[i][3], \
            epic_1[i][3], dard_1[i][3], srrd_1[i][3], \
            epic_2[i][3], dard_2[i][3], srrd_2[i][3]])

    df = pd.DataFrame(arr, columns = ['original', 'canonical_orig', \
            'epic_1', 'dard_1', 'srrd_1', 'epic_2', 'dard_2', 'srrd_2'])

    return df, new_rewards


def compare_new_rews (irl_rewards, can_zeroed, can_zeroed_approx, index_zeros):

    indices = set(index_zeros)
    new_rewards = []

    for i, rew in enumerate(irl_rewards):
        if i in indices:
            new_rewards.append(rew)

    arr = []
    for i in range(len(new_rewards)):
        arr.append([new_rewards[i][3], can_zeroed[i][3], \
                can_zeroed_approx[i][3]])

    df = pd.DataFrame(arr, columns = ['original', 'sample_epic', 'sample_approx'])
    return df, new_rewards


def process_trajs_regress(irl_rewards, trajs):
    """
    Collect trajectories, then perform regression to predict values
    """
    # First remove rewards here and then train a model
    rewards, rewards_linear = deepcopy(irl_rewards), deepcopy(irl_rewards)
    # reward stats
    trans_set = set()

    for T in trajs:
        for t in T:
            trans_set.add(t)

    rewards_zeroed, index_store = [], []
    for index, reward in enumerate(rewards):
        tup = tuple(reward)
        if tup in trans_set:
            index_store.append(index)
            rewards_zeroed.append(reward)
        else:
            # We haven't seen this transition hence we want to
            rewards_linear[index][3] = None

    # now train a network to predict rewards from shaped_rewards_zeroed
    model = perform_regression(rewards_zeroed)
    return rewards_zeroed, rewards_linear, index_store, model



def process_trajs(irl_rewards, trajs, grid_size, action_size):
    """
    We collect the trajectories we sample and find all possible transitions
    """
    # First remove rewards here and then train a model
    rewards, rewards_linear = deepcopy(irl_rewards), deepcopy(irl_rewards)
    # reward stats
    trans_set = set()

    for T in trajs:
        for t in T:
            trans_set.add(t)

    rewards_zeroed, index_store = [], []
    for index, reward in enumerate(rewards):
        tup = tuple(reward)
        if tup in trans_set:
            index_store.append(index)
            rewards_zeroed.append(reward)
        else:
            # We haven't seen this transition hence we want to
            rewards_linear[index][3] = None

    return rewards_zeroed, rewards_linear, index_store



def process_trajs_bb(irl_rewards, trajs, grid_size, action_size):
    """
    We collect the trajectories we sample and find all possible transitions
    """
    # First remove rewards here and then train a model
    rewards, rewards_linear = deepcopy(irl_rewards), deepcopy(irl_rewards)

    # reward stats
    trans_set = set()

    for T in trajs:
        for t in T:
            trans_set.add(t[:3])

    rewards_zeroed, index_store = [], []
    for index, reward in enumerate(rewards):
        tup = tuple(reward[:3])
        if tup in trans_set:
            index_store.append(index)
            rewards_zeroed.append(reward)
        else:
            # We haven't seen this transition hence we want to
            rewards_linear[index][3] = None
    return rewards_zeroed, rewards_linear, index_store



def process_with_index (irl_rewards, index_store):
    """
    We collect the trajectories we sample and find all possible transitions
    """
    # First remove rewards here and then train a model
    rewards, rewards_linear = deepcopy(irl_rewards), deepcopy(irl_rewards)
    index_set = set(deepcopy(index_store))

    # iterate through indices
    rewards_zeroed = []
    for index, reward in enumerate(rewards):
        if index in index_set:
            rewards_zeroed.append(reward)
        else:
            rewards_linear[index][3] = None

    return rewards_zeroed



def perform_regression (rewards):
    # we want to fit a regression model for the state, action, next state triple

    # now rewrite rewards vector in one-hot encoded form
    X, Y = [], []
    for r in rewards:
        s, a, sp, rew = r
        x1, y1 = s
        x2, y2 = sp

        X.append([x1, y1, x2 ,y2])
        Y.append(rew)

    X = np.array(X)
    X = X.reshape(len(X), len(X[0]))
    Y = np.array(Y)
    Y = Y.reshape(len(Y), 1)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    """
    input_size = X_train.shape[1]

    # Define the model architecture
    model = tf.keras.Sequential([
      tf.keras.layers.Dense(64, activation='relu', input_shape=(input_size,)),])
    for l in range(10):
        model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    # Train the model
    history = model.fit(X_train, y_train, epochs=300, validation_data=(X_test, y_test), verbose = 0)

    # Evaluate the model
    reg_score = r2_score(y_test, model.predict(X_test))
    """
    # linear regression
    #print(len(X_train), len(X_test))
    reg3 = LinearRegression().fit(X_train, y_train)
    #reg_score3 = reg3.score(X_test, y_test)


    #mse = mean_squared_error(reg3.predict(X_test), y_test)

    #if mode == 'b':
    #    print("mse = ", mse)
    model = reg3
    """
    for r in rewards_linear:
        s, a, sp, rew = r
        if rew is None:
            x, y = s
            x1, y1 = sp
            X = [[x, y, x1, y1]]
            val = model.predict(X)[0][0]
            r[3] = val
    """
    return model



def fully_connected_mdp (num_states, relationship):

    states = {}   # so states contains the index and the representation etc
    state_rewards = {} # gives rewards per state

    for i in range(num_states):
        for j in range(num_states):
            states[(i, j)] = 0 # 0 is the reward

    graph = []
    if relationship == 'linear':
        # parameters for a linear relationship
        a, b, c = np.random.randint(-100, 100), np.random.randint(-100, 100), np.random.randint(-100, 100)

        for s in states:
            x, y = s
            states[s] = a*x + b*y + c # update reward for all states
            #states[s][1] = a*(x**2) + b*(y**2) + (x*y) + c # update reward for all states

    elif relationship == 'polynomial':

        a, b, c = np.random.randint(-5, 5), np.random.randint(-5, 5), np.random.randint(-100, 100)
        degree = [i for i in range(np.random.randint(4, 8))]
        for s in states:
            x, y = s
            for deg in degree:
                states[s] += (a*(x**deg) + b*(y**deg) + c)/100

    elif relationship == 'sinusoidal':

        a, b, c = np.random.randint(-5, 5), np.random.randint(-5, 5), np.random.randint(-100, 100)
        for s in states:
            x, y = s
            states[s] = a*math.sin(x) + b*math.sin(y) + c

    elif relationship == 'random':
        for s in states:
            states[s] = np.random.randint(-1000, 1000)

    # now connect all edges
    for s in states:
        for i, next_s in enumerate(states):
            # assume action doesnt matter in canonicalization so will mantain it for now as 0\
            x, y = s
            x1, y1 = next_s

            direction = 4 # direction = 4 means that the transition is unfeasible realistically

            feasible_trans = {(x-1, y): 0, (x, y+1):1, (x+1, y):2, (x, y-1):3}

            """
            if (x1, y1) in feasible_trans:
                direction = feasible_trans[(x1, y1)]

            graph.append([s, direction, next_s, states[next_s] + 2*states[s]]) # linear reward inserted
            """

            if (x1, y1) in feasible_trans:
                direction = feasible_trans[(x1, y1)]

            action_feat = (x1-x)*np.random.randint(-5, 5) + (y1 -y)*np.random.randint(-5, 5)

            graph.append([s, direction, next_s, states[next_s] + action_feat + states[s]]) # linear reward inserted



    return graph, states


def fully_connected_bb(num_states, relationship):

    states = {}   # so states contains the index and the representation etc
    state_rewards = {} # gives rewards per state

    for i in range(num_states):
        for j in range(num_states):
            states[(i, j)] = 0 # 0 is the reward

    graph = []
    if relationship == 'linear':
        # parameters for a linear relationship
        a, b, c = np.random.randint(-100, 100), np.random.randint(-100, 100), np.random.randint(-100, 100)

        for s in states:
            x, y = s
            states[s] = a*x + b*y + c # update reward for all states

    elif relationship == 'polynomial':

        a, b, c = np.random.randint(-5, 5), np.random.randint(-5, 5), np.random.randint(-100, 100)
        degree = [i for i in range(np.random.randint(4, 8))]
        for s in states:
            x, y = s
            for deg in degree:
                states[s] += (a*(x**deg) + b*(y**deg) + c)/100

    elif relationship == 'sinusoidal':

        a, b, c = np.random.randint(-5, 5), np.random.randint(-5, 5), np.random.randint(-100, 100)
        for s in states:
            x, y = s
            states[s] = a*(math.sin(x)) + b*(math.sin(y)) + c

    elif relationship == 'random':
        for s in states:
            states[s] = np.random.randint(-1000, 1000)

    # now connect all edges
    for s in states:
        for i, next_s in enumerate(states):
            # assume action doesnt matter in canonicalization so will mantain it for now as 0\
            x, y = s
            x1, y1 = next_s

            direction = 8 # direction = 4 means that the transition is unfeasible realistically

            feasible_trans = {(x, y - 1):0, (x + 1, y - 1):1, (x + 1, y):2,
                    (x + 1, y + 1):3, (x, y + 1):4, (x - 1, y + 1):5,
                    (x - 1, y):6, (x - 1, y - 1):7}

            if (x1, y1) in feasible_trans:
                direction = feasible_trans[(x1, y1)]

            action_feat = (x1-x)*np.random.randint(-5, 5) + (y1 -y)*np.random.randint(-5, 5)

            graph.append([s, direction, next_s, states[next_s] + 2*action_feat + states[s]]) # linear reward inserted

    graph = {(s, a, sp): r for s, a, sp, r in graph if a <= 8}   # graph is a dictionary

    return graph, states


def induce_shaping(graph, states, gamma, terminal, relationship, scale, \
        noise_factor):

    shaped_graph, potential, state_mp = deepcopy(graph), deepcopy(states), \
        deepcopy(states)

    for n in potential:
        potential[n] = 0

    if relationship == 'linear':
        # parameters for a linear relationship
        a, b, c = np.random.randint(-100, 100), np.random.randint(-100, 100), \
            np.random.randint(-100, 100)

        # now induce shaping effects to the states
        for s in potential:
            #if s not in terminal:
            x, y = s
            potential[s] = scale * (a*x + b*y + c)   # shaping based on linear reward

    elif relationship == 'polynomial':

        a, b, c = np.random.randint(-5, 5), np.random.randint(-5, 5), \
            np.random.randint(-100, 100)
        degree = [i for i in range(np.random.randint(4, 8))]

        for s in potential:
            #if s not in terminal:
            x, y = s
            for deg in degree:
                potential[s] += scale/10 * (a*(x**deg) + b*(y**deg) + c)/10

    elif relationship == 'sinusoidal':

        a, b, c = np.random.randint(-10, 10), np.random.randint(-10, 10), \
            np.random.randint(-100, 100)

        for s in potential:
            #if s not in terminal:
            x, y = s
            potential[s] = scale *(a*(math.sin(x)) + b*(math.sin(y)) + c)

    elif relationship == 'random':
        for s in potential:
            #if s not in terminal:
            potential[s] = scale * (np.random.randint(-1000, 1000))

    # update graph based on shaping
    for node in shaped_graph:
        s, a, sp, r = node

        node[3] = r + gamma*potential[sp] - potential[s]

        #node[3] = node[3] + noise_factor * random.uniform(-np.abs(node[3]),\
        #    np.abs(node[3]))

    return shaped_graph, potential


def induce_shaping_bb(graph, states, gamma, terminal, relationship, scale):

    shaped_graph, potential, state_mp = deepcopy(graph), deepcopy(states), \
        deepcopy(states)

    for n in potential:
        potential[n] = 0

    if relationship == 'linear':
        # parameters for a linear relationship
        a, b, c = np.random.randint(-100, 100), np.random.randint(-100, 100), np.random.randint(-100, 100)

        # now induce shaping effects to the states
        for s in potential:
            #if s not in terminal:
            x, y = s
            potential[s] = scale * (a*x + b*y + c)   # shaping based on linear reward

    elif relationship == 'polynomial':

        a, b, c = np.random.randint(-5, 5), np.random.randint(-5, 5), np.random.randint(-100, 100)
        degree = [i for i in range(np.random.randint(4, 8))]

        for s in potential:
            #if s not in terminal:
            x, y = s
            for deg in degree:
                potential[s] += scale/10 * (a*(x**deg) + b*(y**deg) + c)/10

    elif relationship == 'sinusoidal':

        a, b, c = np.random.randint(-5, 5), np.random.randint(-5, 5), np.random.randint(-100, 100)
        for s in potential:
            #if s not in terminal:
            x, y = s
            potential[s] = scale *(a*(math.sin(x)) + b*(math.sin(y)) + c)

    elif relationship == 'random':
        for s in potential:
            #if s not in terminal:
            potential[s] = scale * (np.random.randint(-1000, 1000))

    # update graph based on shaping
    for node in shaped_graph:
        s, a, sp = node
        r = shaped_graph[node]
        shaped_graph[node] = r + gamma*potential[sp] - potential[s]

    return shaped_graph, potential


def run_trial_compare(args):
    """ Compares unbiased vs biased estimate approaches """

    seed()
    D_EPIC = []

    #try:
    index, gamma, grid_size, scale, relationship, state_rewards, policy, \
        action_size, rollover_type, noise_factor, NUM_TRAJS, irl_rewards, \
        shaped_irl_rewards, full_indices, MAX_SIZE, arg_sizes = args

    alpha = time.time()

    # run each independent trial here!
    Nm = set() # coverage and number of states Ds

    trajs = policy_rollover(grid_size, \
        NUM_TRAJS, policy, irl_rewards, rollover_type, MAX_SIZE)

    shaped_trajs = policy_rollover(grid_size, \
        NUM_TRAJS, policy, shaped_irl_rewards, rollover_type, MAX_SIZE)

    rewards_zeroed, rewards_linear, index_zeros = \
        process_trajs(irl_rewards, trajs, grid_size, action_size)

    if full_indices:
        shaped_rewards_linear, shaped_index_zeros = rewards_linear, index_zeros
        shaped_rewards_zeroed = process_with_index(shaped_irl_rewards, \
            index_zeros)
    else:
        shaped_rewards_zeroed, shaped_rewards_linear, shaped_index_zeros = \
            process_trajs(shaped_irl_rewards, shaped_trajs, grid_size, action_size)

    for t in rewards_zeroed:
        s, a, sp, r = t
        Nm.add(s)
        Nm.add(sp)

    BM_threshold = 100

    can_original = canonicalize_epic(BM_threshold, irl_rewards, gamma, True, \
        index_zeros)
    shaped_can_original = canonicalize_epic(BM_threshold, shaped_irl_rewards, \
        gamma, True, shaped_index_zeros)

    # dealing with epic on sparse reward samples
    epic_1 = canonicalize_epic(BM_threshold, rewards_zeroed, gamma, False, \
        index_zeros)
    shaped_epic_1 = canonicalize_epic(BM_threshold, shaped_rewards_zeroed, \
        gamma, False, shaped_index_zeros)

    # dealing with epic on sparse reward samples
    epic_2 = canonicalize_u_epic(rewards_zeroed, gamma, False, \
        index_zeros)
    shaped_epic_2 = canonicalize_u_epic(shaped_rewards_zeroed, \
        gamma, False, shaped_index_zeros)


    srrd_1 = canonicalize_srrd(BM_threshold, rewards_zeroed, gamma, False, \
        index_zeros)
    shaped_srrd_1 = canonicalize_srrd(BM_threshold, shaped_rewards_zeroed, \
        gamma, False, shaped_index_zeros)

    srrd_2 = canonicalize_u_srrd(rewards_zeroed, gamma, False, \
        index_zeros)
    shaped_srrd_2 = canonicalize_u_srrd(shaped_rewards_zeroed, \
        gamma, False, shaped_index_zeros)


    dard_1 = canonicalize_dard(BM_threshold, rewards_zeroed, \
        gamma, False, index_zeros)
    shaped_dard_1 = canonicalize_dard(BM_threshold, shaped_rewards_zeroed, \
        gamma, False, shaped_index_zeros)

    dard_2 = canonicalize_u_dard(rewards_zeroed, None, \
        gamma, False, index_zeros)
    shaped_dard_2 = canonicalize_u_dard(shaped_rewards_zeroed, \
        None, gamma, False, shaped_index_zeros)


    i_zero_mp = {j:i for i, j in enumerate(index_zeros)}
    i_shaped_mp = {j:i for i, j in enumerate(shaped_index_zeros)}

    index_inter = list(set(index_zeros) & set(shaped_index_zeros))

    D = (len(index_inter)/len(rewards_linear))*100

    no_shaping = [i_zero_mp[j] for j in index_inter]
    shaping_list = [i_shaped_mp[j] for j in index_inter]

    can_original = [can_original[i] for i in no_shaping]
    epic_1 = [epic_1[i] for i in no_shaping]
    dard_1 = [dard_1[i] for i in no_shaping]
    srrd_1 = [srrd_1[i] for i in no_shaping]
    epic_2 = [epic_2[i] for i in no_shaping]
    dard_2 = [dard_2[i] for i in no_shaping]
    srrd_2 = [srrd_2[i] for i in no_shaping]

    shaped_can_original = [shaped_can_original[i] for i in shaping_list]
    shaped_epic_1 = [shaped_epic_1[i] for i in shaping_list]
    shaped_dard_1 = [shaped_dard_1[i] for i in shaping_list]
    shaped_srrd_1 = [shaped_srrd_1[i] for i in shaping_list]
    shaped_epic_2 = [shaped_epic_2[i] for i in shaping_list]
    shaped_dard_2 = [shaped_dard_2[i] for i in shaping_list]
    shaped_srrd_2 = [shaped_srrd_2[i] for i in shaping_list]

    df1, new_rewards = compare_rews_new(irl_rewards, can_original, epic_1, dard_1, \
        srrd_1,  epic_2, dard_2, srrd_2, index_inter)

    sl_df1, sl_new_rewards = compare_rews_new(shaped_irl_rewards, \
        shaped_can_original, shaped_epic_1, shaped_dard_1, shaped_srrd_1,  \
        shaped_epic_2, shaped_dard_2, shaped_srrd_2, index_inter)

    try:
        # distance on fully connected rewards
        orig_v_shaped = stats.pearsonr(df1['original'], sl_df1['original'])[0]

        # sample rewards without canonicalization
        can_vs_sh_can = stats.pearsonr(df1['canonical_orig'], sl_df1['canonical_orig'])[0]

        # sample rewards with epic canicalization
        epic1_v_sh_epic1 = stats.pearsonr(df1['epic_1'], sl_df1['epic_1'])[0]
        # sample rewards with dard canonicalization
        dard1_v_sh_dard1 = stats.pearsonr(df1['dard_1'], sl_df1['dard_1'])[0]
        # sample rewards with sard canonicalization
        srrd1_v_sh_srrd1 = stats.pearsonr(df1['srrd_1'], sl_df1['srrd_1'])[0]

        # sample rewards with epic canicalization
        epic2_v_sh_epic2 = stats.pearsonr(df1['epic_2'], sl_df1['epic_2'])[0]
        # sample rewards with dard canonicalization
        dard2_v_sh_dard2 = stats.pearsonr(df1['dard_2'], sl_df1['dard_2'])[0]
        # sample rewards with sard canonicalization
        srrd2_v_sh_srrd2 = stats.pearsonr(df1['srrd_2'], sl_df1['srrd_2'])[0]


        #print(np.sqrt(1 - orig_v_shaped), np.sqrt(1 - can_vs_sh_can) )
        d1 = (1/np.sqrt(2))*np.sqrt(1 - orig_v_shaped)
        d2 = (1/np.sqrt(2))*np.sqrt(1 - can_vs_sh_can)
        d3 = (1/np.sqrt(2))*np.sqrt(1 - epic1_v_sh_epic1)
        d4 = (1/np.sqrt(2))*np.sqrt(1 - dard1_v_sh_dard1)
        d5 = (1/np.sqrt(2))*np.sqrt(1 - srrd1_v_sh_srrd1)
        d6 = (1/np.sqrt(2))*np.sqrt(1 - epic2_v_sh_epic2)
        d7 = (1/np.sqrt(2))*np.sqrt(1 - dard2_v_sh_dard2)
        d8 = (1/np.sqrt(2))*np.sqrt(1 - srrd2_v_sh_srrd2)

        scl = 'high'

        D_EPIC.append([grid_size, scl, relationship, rollover_type, noise_factor, NUM_TRAJS, \
            d1, d2, d3, d4, d5, d6, d7, d8, D, len(Nm)])

        print("run trial = {}/{} and time_taken = {}".format(index, arg_sizes, \
            time.time() - alpha ))
    except:
        print("Error encountered")

    return D_EPIC


def run_trial_regression(args):
    """ Run a single trial of reward canonicalization """

    seed()
    D_EPIC = []

    index, gamma, grid_size, scale, relationship, state_rewards, policy, \
        action_size, rollover_type, noise_factor, NUM_TRAJS, irl_rewards, \
        shaped_irl_rewards, MAX_SIZE, arg_sizes = args

    alpha = time.time()

    # run each independent trial here!
    Nm = set() # coverage and number of states Ds

    trajs = policy_rollover(grid_size, NUM_TRAJS, policy, irl_rewards, \
        rollover_type, MAX_SIZE)

    shaped_trajs = policy_rollover(grid_size, NUM_TRAJS, policy, \
        shaped_irl_rewards, rollover_type, MAX_SIZE)

    rewards_zeroed, rewards_linear, index_zeros, model = \
        process_trajs_regress(irl_rewards, trajs)

    _, _, _, shaped_model = process_trajs_regress(shaped_irl_rewards, shaped_trajs)

    shaped_rewards_linear, shaped_index_zeros = rewards_linear, index_zeros
    shaped_rewards_zeroed = process_with_index(shaped_irl_rewards, index_zeros)

    for t in rewards_zeroed:
        s, a, sp, r = t
        Nm.add(s)
        Nm.add(sp)

    def get_pearson_distance (A, B, is_action):
        """
        if is_action:
            R = [i[3] for i in A if i[3] != (grid_size-1, grid_size-1)]
            R_S = [i[3] for i in B if i[3] != (grid_size-1, grid_size-1)]
        else:
            R = [i[2] for i in A if i[1] != (grid_size-1, grid_size-1)]
            R_S = [i[2] for i in B if i[1] != (grid_size-1, grid_size-1)]
        """
        if is_action:
            R = [i[3] for i in A]
            R_S = [i[3] for i in B]
        else:
            R = [i[2] for i in A]
            R_S = [i[2] for i in B]

        rho = stats.pearsonr(R, R_S)[0]
        return np.sqrt(1-rho)*(1/np.sqrt(2))


    # dealing with epic on sparse reward samples
    sparse_epic = canonicalize_epic_regress(rewards_zeroed, gamma, model)
    shaped_sparse_epic = canonicalize_epic_regress(shaped_rewards_zeroed, \
        gamma, shaped_model)

    # dealing with dard, need weights to factor in transition probs
    sparse_dard = canonicalize_dard_regress(rewards_zeroed, gamma, model)
    shaped_sparse_dard = canonicalize_dard_regress(shaped_rewards_zeroed, \
        gamma, shaped_model)

    sparse_srrd = canonicalize_srrd_regress(rewards_zeroed, gamma, model)
    shaped_sparse_srrd = canonicalize_srrd_regress(shaped_rewards_zeroed, \
        gamma, shaped_model)

    #p_direct = get_pearson_distance(rewards_zeroed, shaped_rewards_zeroed, True)
    p_epic_regress = get_pearson_distance(sparse_epic, shaped_sparse_epic, False)
    p_dard_regress = get_pearson_distance(sparse_dard, shaped_sparse_dard, False)
    p_srrd_regress = get_pearson_distance(sparse_srrd, shaped_sparse_srrd, False)

    can_original = canonicalize_epic(irl_rewards, gamma, True, index_zeros)
    shaped_can_original = canonicalize_epic(shaped_irl_rewards, gamma, \
        True, shaped_index_zeros)

    # dealing with epic on sparse reward samples
    sparse_epic = canonicalize_epic(rewards_zeroed, gamma, False, \
        index_zeros)
    shaped_sparse_epic = canonicalize_epic(shaped_rewards_zeroed, \
        gamma, False, shaped_index_zeros)

    sparse_sard = canonicalize_srrd(rewards_zeroed, gamma, False, \
        index_zeros)
    shaped_sparse_sard = canonicalize_srrd(shaped_rewards_zeroed, \
        gamma, False, shaped_index_zeros)

    # dealing with dard, need weights to factor in transition probs
    weights_mp = None
    sparse_dard = canonicalize_dard(rewards_zeroed, weights_mp, \
        gamma, False, index_zeros)
    shaped_sparse_dard = canonicalize_dard(shaped_rewards_zeroed, \
        weights_mp, gamma, False, shaped_index_zeros)


    i_zero_mp = {j:i for i, j in enumerate(index_zeros)}
    i_shaped_mp = {j:i for i, j in enumerate(shaped_index_zeros)}

    index_inter = list(set(index_zeros) & set(shaped_index_zeros))

    D = (len(index_inter)/len(rewards_linear))*100

    no_shaping = [i_zero_mp[j] for j in index_inter]
    shaping_list = [i_shaped_mp[j] for j in index_inter]

    can_original = [can_original[i] for i in no_shaping]
    sparse_epic = [sparse_epic[i] for i in no_shaping]
    sparse_dard = [sparse_dard[i] for i in no_shaping]
    sparse_sard = [sparse_sard[i] for i in no_shaping]

    shaped_can_original = [shaped_can_original[i] for i in shaping_list]
    shaped_sparse_epic = [shaped_sparse_epic[i] for i in shaping_list]
    shaped_sparse_dard = [shaped_sparse_dard[i] for i in shaping_list]
    shaped_sparse_sard = [shaped_sparse_sard[i] for i in shaping_list]

    df1, new_rewards = compare_rews(irl_rewards, can_original, sparse_epic, \
        sparse_dard, sparse_sard, index_inter)

    sl_df1, sl_new_rewards = compare_rews(shaped_irl_rewards, \
        shaped_can_original, shaped_sparse_epic, shaped_sparse_dard, \
        shaped_sparse_sard, index_inter)

    # distance on fully connected rewards
    orig_v_shaped = stats.pearsonr(df1['original'], sl_df1['original'])[0]

    # sample rewards without canonicalization
    can_vs_sh_can = stats.pearsonr(df1['canonical_orig'], sl_df1['canonical_orig'])[0]

    # sample rewards with epic canicalization
    epic_v_sh_epic = stats.pearsonr(df1['sample_epic'], sl_df1['sample_epic'])[0]

    # sample rewards with dard canonicalization
    dard_v_sh_dard = stats.pearsonr(df1['sample_dard'], sl_df1['sample_dard'])[0]

    # sample rewards with sard canonicalization
    sard_v_sh_sard = stats.pearsonr(df1['sample_sard'], sl_df1['sample_sard'])[0]

    #print(np.sqrt(1 - orig_v_shaped), np.sqrt(1 - can_vs_sh_can) )
    d1 = (1/np.sqrt(2))*np.sqrt(1 - orig_v_shaped)
    d2 = (1/np.sqrt(2))*np.sqrt(1 - can_vs_sh_can)
    d3 = (1/np.sqrt(2))*np.sqrt(1 - epic_v_sh_epic)
    d4 = (1/np.sqrt(2))*np.sqrt(1 - dard_v_sh_dard)
    d5 = (1/np.sqrt(2))*np.sqrt(1 - sard_v_sh_sard)

    scl = 'high'


    # Now need to update this to collect prior regressed data as well
    D_EPIC.append([grid_size, scl, relationship, rollover_type, noise_factor, \
        NUM_TRAJS, d1, d2, d3, d4, d5, p_epic_regress, p_dard_regress, \
        p_srrd_regress, D, len(Nm)])

    print("run trial = {}/{} and time_taken = {}".format(index, arg_sizes, \
        time.time() - alpha ))

    #except:
    #    print("Error encountered")

    return D_EPIC




def run_trial(args):
    """ Run a single trial of reward canonicalization """

    seed()
    D_EPIC = []

    #try:
    index, gamma, grid_size, scale, relationship, state_rewards, policy, \
        action_size, rollover_type, noise_factor, NUM_TRAJS, irl_rewards, \
        shaped_irl_rewards, full_indices, MAX_SIZE, arg_sizes, unbiased = args

    alpha = time.time()

    # run each independent trial here!
    Nm = set() # coverage and number of states Ds

    trajs = policy_rollover(grid_size, NUM_TRAJS, policy, irl_rewards, \
        rollover_type, MAX_SIZE)

    shaped_trajs = policy_rollover(grid_size, NUM_TRAJS, policy, \
        shaped_irl_rewards, rollover_type, MAX_SIZE)

    rewards_zeroed, rewards_linear, index_zeros = \
        process_trajs(irl_rewards, trajs, grid_size, action_size)

    if full_indices:
        shaped_rewards_linear, shaped_index_zeros = rewards_linear, index_zeros
        shaped_rewards_zeroed = process_with_index(shaped_irl_rewards, \
            shaped_trajs, index_zeros, grid_size, action_size, rollover_type)
    else:
        shaped_rewards_zeroed, shaped_rewards_linear, shaped_index_zeros = \
            process_trajs(shaped_irl_rewards, shaped_trajs, grid_size, action_size)

    for t in rewards_zeroed:
        s, a, sp, r = t
        Nm.add(s)
        Nm.add(sp)

    BM_threshold = 50

    if unbiased:
        can_original = canonicalize_u_epic(irl_rewards, gamma, True, index_zeros)
        shaped_can_original = canonicalize_u_epic(shaped_irl_rewards, gamma, \
            True, shaped_index_zeros)

        """
        # dealing with epic on sparse reward samples
        sparse_epic = canonicalize_u_epic(rewards_zeroed, gamma, False, \
            index_zeros)
        shaped_sparse_epic = canonicalize_u_epic(shaped_rewards_zeroed, \
            gamma, False, shaped_index_zeros)
        """

        # dealing with epic on sparse reward samples
        sparse_epic = canonicalize_u_epic(rewards_zeroed, gamma, False, \
            index_zeros)
        shaped_sparse_epic = canonicalize_u_epic(shaped_rewards_zeroed, \
            gamma, False, shaped_index_zeros)

        sparse_sard = canonicalize_u_srrd(rewards_zeroed, gamma, False, \
            index_zeros)
        shaped_sparse_sard = canonicalize_u_srrd(shaped_rewards_zeroed, \
            gamma, False, shaped_index_zeros)

        # dealing with dard, need weights to factor in transition probs
        weights_mp = None
        sparse_dard = canonicalize_u_dard(rewards_zeroed, weights_mp, \
            gamma, False, index_zeros)
        shaped_sparse_dard = canonicalize_u_dard(shaped_rewards_zeroed, \
            weights_mp, gamma, False, shaped_index_zeros)

    else:
        """ Uses original definitions, from the papers """
        can_original = canonicalize_epic(BM_threshold, irl_rewards, gamma, \
            True, index_zeros)
        shaped_can_original = canonicalize_epic(BM_threshold, \
            shaped_irl_rewards, gamma, True, shaped_index_zeros)

        # dealing with epic on sparse reward samples
        sparse_epic = canonicalize_epic(BM_threshold, rewards_zeroed, gamma, \
            False, index_zeros)
        shaped_sparse_epic = canonicalize_epic(BM_threshold, \
            shaped_rewards_zeroed, gamma, False, shaped_index_zeros)

        sparse_sard = canonicalize_srrd(BM_threshold, rewards_zeroed, gamma, \
            False, index_zeros)
        shaped_sparse_sard = canonicalize_srrd(BM_threshold, \
            shaped_rewards_zeroed, gamma, False, shaped_index_zeros)

        # dealing with dard, need weights to factor in transition probs
        sparse_dard = canonicalize_dard(BM_threshold, rewards_zeroed, \
            gamma, False, index_zeros)
        shaped_sparse_dard = canonicalize_dard(BM_threshold, \
            shaped_rewards_zeroed, gamma, False, shaped_index_zeros)


    i_zero_mp = {j:i for i, j in enumerate(index_zeros)}
    i_shaped_mp = {j:i for i, j in enumerate(shaped_index_zeros)}

    index_inter = list(set(index_zeros) & set(shaped_index_zeros))

    D = (len(index_inter)/len(rewards_linear))*100

    no_shaping = [i_zero_mp[j] for j in index_inter]
    shaping_list = [i_shaped_mp[j] for j in index_inter]

    can_original = [can_original[i] for i in no_shaping]
    sparse_epic = [sparse_epic[i] for i in no_shaping]
    sparse_dard = [sparse_dard[i] for i in no_shaping]
    sparse_sard = [sparse_sard[i] for i in no_shaping]

    shaped_can_original = [shaped_can_original[i] for i in shaping_list]
    shaped_sparse_epic = [shaped_sparse_epic[i] for i in shaping_list]
    shaped_sparse_dard = [shaped_sparse_dard[i] for i in shaping_list]
    shaped_sparse_sard = [shaped_sparse_sard[i] for i in shaping_list]

    df1, new_rewards = compare_rews(irl_rewards, can_original, sparse_epic, \
        sparse_dard, sparse_sard, index_inter)

    sl_df1, sl_new_rewards = compare_rews(shaped_irl_rewards, \
        shaped_can_original, shaped_sparse_epic, shaped_sparse_dard, \
        shaped_sparse_sard, index_inter)

    # distance on fully connected rewards
    orig_v_shaped = stats.pearsonr(df1['original'], sl_df1['original'])[0]

    # sample rewards without canonicalization
    can_vs_sh_can = stats.pearsonr(df1['canonical_orig'], sl_df1['canonical_orig'])[0]

    # sample rewards with epic canicalization
    epic_v_sh_epic = stats.pearsonr(df1['sample_epic'], sl_df1['sample_epic'])[0]

    # sample rewards with dard canonicalization
    dard_v_sh_dard = stats.pearsonr(df1['sample_dard'], sl_df1['sample_dard'])[0]

    # sample rewards with sard canonicalization
    sard_v_sh_sard = stats.pearsonr(df1['sample_sard'], sl_df1['sample_sard'])[0]

    #print(np.sqrt(1 - orig_v_shaped), np.sqrt(1 - can_vs_sh_can) )
    d1 = (1/np.sqrt(2))*np.sqrt(1 - orig_v_shaped)
    d2 = (1/np.sqrt(2))*np.sqrt(1 - can_vs_sh_can)
    d3 = (1/np.sqrt(2))*np.sqrt(1 - epic_v_sh_epic)
    d4 = (1/np.sqrt(2))*np.sqrt(1 - dard_v_sh_dard)
    d5 = (1/np.sqrt(2))*np.sqrt(1 - sard_v_sh_sard)

    scl = 'high'
    D_EPIC.append([grid_size, scl, relationship, rollover_type, noise_factor, NUM_TRAJS, \
        d1, d2, d3, d4, d5, D, len(Nm)])

    print("run trial = {}/{} and time_taken = {}".format(index, arg_sizes, \
        time.time() - alpha ))

    #except:
    #    print("Error encountered")

    return D_EPIC





def run_trial_bb(args):
    """ Run a single trial of reward canonicalization """

    seed()
    D_EPIC = []

    index, gamma, grid_size, scale, relationship, state_rewards, policy, \
        action_size, rollover_type, NUM_TRAJS, irl_rewards, \
        shaped_irl_rewards, full_indices, MIN_DIST, LEN_LIMIT, unbiased = args


    print("running trial = ", index)
    alpha = time.time()

    Nm = set() # coverage and number of states Ds

    # sample the ground truth reward, from a distribution of starting states and
    # returns a shaped reward and an unshaped reward after traversing graph using given policy
    trajs, shaped_trajs = policy_rollover_bb(grid_size, NUM_TRAJS, policy, \
        irl_rewards, shaped_irl_rewards, rollover_type, MIN_DIST, LEN_LIMIT)

    s_keys = sorted([k for k in irl_rewards])
    irl_rewards = [list(k)+[irl_rewards[k]] for k in s_keys]
    shaped_irl_rewards = [list(k)+[shaped_irl_rewards[k]] for k in s_keys]

    rewards_zeroed, rewards_linear, index_zeros = \
        process_trajs_bb(irl_rewards, trajs, grid_size, action_size)

    if full_indices:
        shaped_rewards_linear, shaped_index_zeros = rewards_linear, index_zeros
        shaped_rewards_zeroed = process_with_index(shaped_irl_rewards, \
            shaped_trajs, index_zeros, grid_size, action_size, rollover_type)

    else:
        shaped_rewards_zeroed, shaped_rewards_linear, shaped_index_zeros = \
            process_trajs_bb(shaped_irl_rewards, shaped_trajs, grid_size, action_size)

    for t in rewards_zeroed:
        s, a, sp, r = t
        Nm.add(s)
        Nm.add(sp)

    if unbiased:
        can_original = canonicalize_u_epic(irl_rewards, gamma, True, index_zeros)
        shaped_can_original = canonicalize_u_epic(shaped_irl_rewards, gamma, \
            True, shaped_index_zeros)

        # dealing with epic on sparse reward samples
        sparse_epic = canonicalize_u_epic(rewards_zeroed, gamma, False, \
            index_zeros)
        shaped_sparse_epic = canonicalize_u_epic(shaped_rewards_zeroed, \
            gamma, False, shaped_index_zeros)

        sparse_sard = canonicalize_u_srrd(rewards_zeroed, gamma, False, \
            index_zeros)
        shaped_sparse_sard = canonicalize_u_srrd(shaped_rewards_zeroed, \
            gamma, False, shaped_index_zeros)

        # dealing with dard, need weights to factor in transition probs
        weights_mp = None
        sparse_dard = canonicalize_u_dard(rewards_zeroed, weights_mp, \
            gamma, False, index_zeros)
        shaped_sparse_dard = canonicalize_u_dard(shaped_rewards_zeroed, \
            weights_mp, gamma, False, shaped_index_zeros)
    else:
        can_original = canonicalize_epic(irl_rewards, gamma, True, index_zeros)
        shaped_can_original = canonicalize_epic(shaped_irl_rewards, gamma, \
            True, shaped_index_zeros)

        # dealing with epic on sparse reward samples
        sparse_epic = canonicalize_epic(rewards_zeroed, gamma, False, \
            index_zeros)
        shaped_sparse_epic = canonicalize_epic(shaped_rewards_zeroed, \
            gamma, False, shaped_index_zeros)

        sparse_sard = canonicalize_srrd(rewards_zeroed, gamma, False, \
            index_zeros)
        shaped_sparse_sard = canonicalize_srrd(shaped_rewards_zeroed, \
            gamma, False, shaped_index_zeros)

        # dealing with dard, need weights to factor in transition probs
        weights_mp = None
        sparse_dard = canonicalize_dard(rewards_zeroed, weights_mp, \
            gamma, False, index_zeros)
        shaped_sparse_dard = canonicalize_dard(shaped_rewards_zeroed, \
            weights_mp, gamma, False, shaped_index_zeros)

    i_zero_mp = {j:i for i, j in enumerate(index_zeros)}
    i_shaped_mp = {j:i for i, j in enumerate(shaped_index_zeros)}

    index_inter = list(set(index_zeros) & set(shaped_index_zeros))
    D = (len(index_inter)/len(rewards_linear))*100

    no_shaping = [i_zero_mp[j] for j in index_inter]
    shaping_list = [i_shaped_mp[j] for j in index_inter]

    can_original = [can_original[i] for i in no_shaping]
    sparse_epic = [sparse_epic[i] for i in no_shaping]
    sparse_dard = [sparse_dard[i] for i in no_shaping]
    sparse_sard = [sparse_sard[i] for i in no_shaping]

    shaped_can_original = [shaped_can_original[i] for i in shaping_list]
    shaped_sparse_epic = [shaped_sparse_epic[i] for i in shaping_list]
    shaped_sparse_dard = [shaped_sparse_dard[i] for i in shaping_list]
    shaped_sparse_sard = [shaped_sparse_sard[i] for i in shaping_list]

    df1, new_rewards = compare_rews(irl_rewards, can_original, sparse_epic, \
        sparse_dard, sparse_sard, index_inter)

    sl_df1, sl_new_rewards = compare_rews(shaped_irl_rewards, \
        shaped_can_original, shaped_sparse_epic, shaped_sparse_dard, \
        shaped_sparse_sard, index_inter)

    # distance on fully connected rewards
    orig_v_shaped = stats.pearsonr(df1['original'], sl_df1['original'])[0]

    # sample rewards without canonicalization
    can_vs_sh_can = stats.pearsonr(df1['canonical_orig'], sl_df1['canonical_orig'])[0]

    # sample rewards with epic canicalization
    epic_v_sh_epic = stats.pearsonr(df1['sample_epic'], sl_df1['sample_epic'])[0]

    # sample rewards with dard canonicalization
    dard_v_sh_dard = stats.pearsonr(df1['sample_dard'], sl_df1['sample_dard'])[0]

    # sample rewards with sard canonicalization
    sard_v_sh_sard = stats.pearsonr(df1['sample_sard'], sl_df1['sample_sard'])[0]

    #print(np.sqrt(1 - orig_v_shaped), np.sqrt(1 - can_vs_sh_can) )
    d1 = (1/np.sqrt(2))*np.sqrt(1 - orig_v_shaped)
    d2 = (1/np.sqrt(2))*np.sqrt(1 - can_vs_sh_can)
    d3 = (1/np.sqrt(2))*np.sqrt(1 - epic_v_sh_epic)
    d4 = (1/np.sqrt(2))*np.sqrt(1 - dard_v_sh_dard)
    d5 = (1/np.sqrt(2))*np.sqrt(1 - sard_v_sh_sard)

    scl = 'high'
    D_EPIC.append([grid_size, scl, relationship, rollover_type, NUM_TRAJS, \
        d1, d2, d3, d4, d5, D, len(Nm)])

    return D_EPIC

def compare_shaping_ratios(irl_rewards, dard_can_original, dard_shaped_sample, \
    epic_can_original, epic_shaped_sample, sard_can_original, \
    sard_shaped_sample, index_zeros):
    """
    In this method, the goal is to compare each reward values extracted by
    each canonical methods.
    """

    indices = set(index_zeros)
    new_rewards = []

    for i, rew in enumerate(irl_rewards):
        if i in indices:
            new_rewards.append(rew)

    print("\n\n irl_rewards = ", new_rewards)

    arr = []
    for i in range(len(new_rewards)):
        arr.append([new_rewards[i][3], can_original[i][3], \
            sparse_epic[i][3], sparse_dard[i][3], sparse_sard[i][3]])

    df = pd.DataFrame(arr, columns = ['original', 'canonical_orig', \
            'sample_epic', 'sample_dard', 'sample_sard'])

    return df, new_rewards


def compute_epic_shaping(irl_rewards, shaped_index_zeros, shaped_rewards_zeroed, \
    shaped_irl_rewards, gamma, rewards_zeroed, rewards_linear, index_zeros, \
    scale, grid_size, relationship, rollover_type, NUM_TRAJS):

    Nm = set()

    for t in rewards_zeroed:
        s, a, sp, r = t
        Nm.add(s)
        Nm.add(sp)

    # run for dard
    weights_mp = None
    #dard_can_original = canonicalize_dard(irl_rewards, weights_mp, gamma, True,\
    #    index_zeros)
    dard_shaped_sample = canonicalize_dard(rewards_zeroed, weights_mp,gamma, \
        False, shaped_index_zeros)

    # run for epic
    can_original = canonicalize_epic(irl_rewards, gamma, True, index_zeros)
    epic_shaped_sample = canonicalize_epic(rewards_zeroed, gamma,  False, \
        shaped_index_zeros)

    # run for sard
    #sard_can_original = canonicalize_sard(irl_rewards, gamma, True, index_zeros)
    sard_shaped_sample = canonicalize_sard(rewards_zeroed, gamma,  False, \
        shaped_index_zeros)

    i_zero_mp = {j:i for i, j in enumerate(index_zeros)}
    i_shaped_mp = {j:i for i, j in enumerate(shaped_index_zeros)}

    index_inter = list(set(index_zeros) & set(shaped_index_zeros))
    D = (len(index_inter)/len(rewards_linear))*100

    no_shaping = [i_zero_mp[j] for j in index_inter]
    shaping_list = [i_shaped_mp[j] for j in index_inter]

    #dard_can_original = {dard_can_original[i][:3]:dard_can_original[i][3] \
    #    for i in no_shaping}
    can_original = {can_original[i][:3]:can_original[i][3] \
        for i in no_shaping}
    #sard_can_original = {sard_can_original[i][:3]:sard_can_original[i][3] \
    #    for i in no_shaping}

    dard_shaped_sample = {dard_shaped_sample[i][:3]:dard_shaped_sample[i][3] \
        for i in shaping_list}
    epic_shaped_sample = {epic_shaped_sample[i][:3]:epic_shaped_sample[i][3] \
        for i in shaping_list}
    sard_shaped_sample = {sard_shaped_sample[i][:3]:sard_shaped_sample[i][3] \
        for i in shaping_list}

    keys = sorted(list(can_original.keys()))

    can_original = np.array([can_original[m] for m in keys])
    dard_sample = np.array([dard_shaped_sample[m] for m in keys])
    epic_sample = np.array([epic_shaped_sample[m] for m in keys])
    sard_sample = np.array([sard_shaped_sample[m] for m in keys])

    dard_diff = np.log(np.mean(np.abs((dard_sample - can_original)/can_original)))
    epic_diff = np.log(np.mean(np.abs((epic_sample - can_original)/can_original)))
    sard_diff = np.log(np.mean(np.abs((sard_sample - can_original)/can_original)))

    scl = 'high'

    return [grid_size, scl, relationship, rollover_type, NUM_TRAJS, \
        epic_diff, dard_diff, sard_diff, D]


def execute_process(args):

    grid_sizes, gamma, policy, rollover_arr, trajs_arr, relationships, \
    ITERATIONS, scale_arr, full_indices, MAX_SIZE, noise_arr, \
    unbiased = deepcopy(args)

    arg_arr = []
    count_index = 0

    for noise_factor in noise_arr:
        for grid_size in grid_sizes:
            for relationship in relationships:
                action_size, terminal = 1, set([(grid_size - 1, \
                    grid_size - 1)])

                for iteration in range(ITERATIONS):
                    irl_rewards, state_rewards = \
                        fully_connected_mdp(grid_size, relationship)

                    scale = np.random.randint(scale_arr[0], scale_arr[1])/scale_arr[2]

                    shaped_irl_rewards, potential = \
                        induce_shaping(irl_rewards, state_rewards, gamma, \
                        terminal, relationship, scale, noise_factor)

                    for rollover_type in rollover_arr:
                        for num_trial in trajs_arr:

                            arg_sizes = len(trajs_arr) * len(rollover_arr) \
                                * ITERATIONS * len(relationships) * \
                                len(grid_sizes) * len(noise_arr)

                            argument = [count_index, gamma, grid_size, \
                                scale, relationship, state_rewards, \
                                policy, action_size, rollover_type, \
                                noise_factor, num_trial, irl_rewards, \
                                shaped_irl_rewards, full_indices, MAX_SIZE,
                                arg_sizes, unbiased]

                            arg_arr.append(argument)
                            count_index += 1
                            #m = run_trial(argument)
                            #print(m)
    d_epic = []
    print("len arg_arr = ", len(arg_arr))
    # Initialize multiprocessing pool outside the loop
    with multiprocessing.Pool() as pool:
        t_a = time.time()
        # Use the existing pool for map
        res = pool.map(run_trial, arg_arr)

        for r in res:
            for r_val in r:
                d_epic.append(r_val)
        print("time to process =", time.time() - t_a)

    df = pd.DataFrame(d_epic, columns = ['grid_size', 'scale', 'relationship', \
        'rollover', 'noise_factor', 'num_trajs', 'original', 'canonical', \
        'epic', 'dard', 'sard', 'D', 'Nm'])

    return df



def execute_regress(args):

    grid_sizes, gamma, policy, rollover_arr, trajs_arr, relationships, \
    ITERATIONS, scale_arr, MAX_SIZE, noise_arr  = deepcopy(args)

    arg_arr = []
    count_index = 0

    for noise_factor in noise_arr:
        for grid_size in grid_sizes:
            for relationship in relationships:
                action_size, terminal = 1, set([(grid_size - 1, \
                    grid_size - 1)])

                for iteration in range(ITERATIONS):
                    irl_rewards, state_rewards = \
                        fully_connected_mdp(grid_size, relationship)

                    scale = np.random.randint(scale_arr[0], scale_arr[1])/scale_arr[2]

                    shaped_irl_rewards, potential = \
                        induce_shaping(irl_rewards, state_rewards, gamma, \
                        terminal, relationship, scale, noise_factor)

                    for rollover_type in rollover_arr:
                        for num_trial in trajs_arr:

                            arg_sizes = len(trajs_arr) * len(rollover_arr) \
                                * ITERATIONS * len(relationships) * \
                                len(grid_sizes) * len(noise_arr)

                            argument = [count_index, gamma, grid_size, \
                                scale, relationship, state_rewards, \
                                policy, action_size, rollover_type, \
                                noise_factor, num_trial, irl_rewards, \
                                shaped_irl_rewards, MAX_SIZE,
                                arg_sizes]

                            arg_arr.append(argument)
                            count_index += 1
                            #m = run_trial_regression(argument)
                            #print(m)
                            #sys.exit()
    d_epic = []
    print("len arg_arr = ", len(arg_arr))
    # Initialize multiprocessing pool outside the loop
    with multiprocessing.Pool() as pool:
        t_a = time.time()
        # Use the existing pool for map
        res = pool.map(run_trial_regression, arg_arr)

        for r in res:
            for r_val in r:
                d_epic.append(r_val)
        print("time to process =", time.time() - t_a)

    df = pd.DataFrame(d_epic, columns = ['grid_size', 'scale', 'relationship', \
        'rollover', 'noise_factor', 'num_trajs', 'original', 'canonical', \
        'epic', 'dard', 'srrd', "epic_r", "dard_r", "srrd_r", 'D', 'Nm'])

    return df




def execute_process_compare(args):

    grid_sizes, gamma, policy, rollover_arr, trajs_arr, relationships, \
    ITERATIONS, scale_arr, full_indices, MAX_SIZE, noise_arr = deepcopy(args)

    arg_arr = []
    count_index = 0

    for noise_factor in noise_arr:
        for grid_size in grid_sizes:
            for relationship in relationships:
                action_size, terminal = 1, set([(grid_size - 1, \
                    grid_size - 1)])

                for iteration in range(ITERATIONS):
                    irl_rewards, state_rewards = \
                        fully_connected_mdp(grid_size, relationship)

                    scale = np.random.randint(scale_arr[0], scale_arr[1])/scale_arr[2]

                    shaped_irl_rewards, potential = \
                        induce_shaping(irl_rewards, state_rewards, gamma, \
                        terminal, relationship, scale, noise_factor)

                    for rollover_type in rollover_arr:
                        for num_trial in trajs_arr:

                            arg_sizes = len(trajs_arr) * len(rollover_arr) \
                                * ITERATIONS * len(relationships) * \
                                len(grid_sizes) * len(noise_arr)

                            argument = [count_index, gamma, grid_size, \
                                scale, relationship, state_rewards, \
                                policy, action_size, rollover_type, \
                                noise_factor, num_trial, irl_rewards, \
                                shaped_irl_rewards, full_indices, MAX_SIZE,
                                arg_sizes]

                            arg_arr.append(argument)
                            count_index += 1
                            #m = run_trial_compare(argument)
                            #print(m)
    d_epic = []
    print("len arg_arr = ", len(arg_arr))
    # Initialize multiprocessing pool outside the loop
    with multiprocessing.Pool() as pool:
        t_a = time.time()
        # Use the existing pool for map
        res = pool.map(run_trial_compare, arg_arr)

        for r in res:
            for r_val in r:
                d_epic.append(r_val)
        print("time to process =", time.time() - t_a)

    df = pd.DataFrame(d_epic, columns = ['grid_size', 'scale', 'relationship', \
        'rollover', 'noise_factor', 'num_trajs', 'original', 'canonical', \
        'epic1', 'dard1', 'srrd1', 'epic2', 'dard2', 'srrd2', 'D', 'Nm'])

    return df


def execute_process_bb(args):

    grid_sizes, gamma, policy, rollover_arr, trajs_arr, relationships, \
        ITERATIONS, scale_arr, full_indices, MIN_DIST, LEN_LIMIT, \
        induce_random, random_scale, unbiased = deepcopy(args)

    arg_arr = []
    count_index = 0

    for grid_size in grid_sizes:
        for relationship in relationships:

            action_size, terminal = 1, set([(grid_size - 1, grid_size - 1)])

            for iteration in range(ITERATIONS):
                irl_rewards, state_rewards = fully_connected_bb(grid_size, \
                    relationship)   # linear rewards

                scale = np.random.randint(scale_arr[0], scale_arr[1])/scale_arr[2]

                shaped_irl_rewards, potential = induce_shaping_bb(irl_rewards,\
                    state_rewards, gamma, terminal, relationship, scale)

                for rollover_type in rollover_arr:
                    for num_trial in trajs_arr:

                        argument = [count_index, gamma, grid_size, scale, \
                            relationship, state_rewards, policy, \
                            action_size, rollover_type, num_trial, \
                            irl_rewards, shaped_irl_rewards, full_indices, \
                            MIN_DIST, LEN_LIMIT, unbiased]

                        arg_arr.append(argument)
                        count_index += 1
                        #m = run_trial_bb(argument)
                        #print(m)

    d_epic = []
    print("len arg_arr = ", len(arg_arr))
    #sys.exit()
    # Initialize multiprocessing pool outside the loop
    with multiprocessing.Pool() as pool:
        t_a = time.time()
        # Use the existing pool for map
        res = pool.map(run_trial_bb, arg_arr)

        for r in res:
            for r_val in r:
                d_epic.append(r_val)
        print("time to process =", time.time() - t_a)

    df = pd.DataFrame(d_epic, columns = ['grid_size', 'scale', 'relationship', \
        'rollover', 'num_trajs', 'original', 'canonical', 'epic', 'dard', 'sard', \
        'D', 'Nm'])

    return df

def normalize(vals):
  """
  normalize to (0, max_val)
  input:
    vals: 1d array
  """
  min_val = np.min(vals)
  max_val = np.max(vals)
  return (vals - min_val) / (max_val - min_val)


def perform_maxent(f_map, trajs_store, GRIDSIZE, ACTION_LEN, GAMMA, \
        LEARNING_RATE, N_ITERS):

    STATES_LEN = GRIDSIZE*GRIDSIZE    # number of states
    # get the feature map
    feat_map = np.array([list(f_map[i]) for i in range(STATES_LEN)])

    # now get the probability matrix of a transition
    P_a = np.zeros((STATES_LEN, STATES_LEN, ACTION_LEN))

    # fill P_a based on state visitation frequencies from trajs_store
    trans_mp = {}
    for trajs in trajs_store:

        for i in range(len(trajs) -1):
            triple1, triple2 = trajs[i], trajs[i+1]
            s1, a = triple1
            s2, _ = triple2
            if (s1, a) not in trans_mp:
                trans_mp[(s1, a)] = np.zeros(STATES_LEN)
            trans_mp[(s1, a)][s2] += 1

    for t in trans_mp:
        T = trans_mp[t]
        T /= (sum(T) + 0.000000000000000001)

        # now, we need to populate P_a
        s, a = t
        for s_p in range(len(T)):
            P_a[s, s_p, a] = T[s_p]

    rewards = maxent_irl(feat_map, P_a, GAMMA, trajs_store, \
            LEARNING_RATE, N_ITERS)

    # get a list of the rewards
    rew_map = set()
    rew_list = []
    for T in trajs_store:
        for i in range(len(T) - 1):
            s_prev, s_curr =  T[i], T[i+1]
            s, a = s_prev
            sp, _ = s_curr

            curr_rew = (s, a, sp, rewards[sp])
            if curr_rew not in rew_map:
                rew_map.add(curr_rew)
                rew_list.append(curr_rew)

    return rew_list


def get_consistent_rep (EPIC_res):
    # Here we want to index all the visited triples then plot a curve of the extracted rewards
    all_triples = set()
    for T in EPIC_res:
        for s in T:
            all_triples.add(s[:3])

    all_triples = list(all_triples)
    all_triples.sort()

    triple_mp = {}
    for i, element in enumerate(all_triples):
        triple_mp[element] = i

    # now re-write accounting for all the seen triples
    rew_store = []
    for T in EPIC_res:
        t_rew = np.zeros(len(triple_mp))

        for s in T:
            t_rew[triple_mp[s[:3]]] = s[3]
        rew_store.append(t_rew)

    return rew_store


def compute_distances (store):
    distances = []
    # The first dist is for the uniform policy
    uniform = store[0]
    # compute other distances
    for i in range(len(store)):
        curr_policy = store[i]
        dist = (1 - pearsonr(uniform, curr_policy)[0])**0.5
        distances.append(dist)
    # return distances
    return distances

def generate_gridworld_trajs(grid_size, f_map, num_trajs, policy, LEN_LIMIT):
    """
    we will assume an n*n grid
    # This code generates Trajectories from a given policy
    """
    seed()

    #f_map = {tuple(f_map[i]):i for i in range(len(f_map))}
    p_dist = [sum(policy[:i+1]) for i in range(len(policy))]
    possible_starts = [(0, 0), (3, 0), (0, 2), (1, 2)]
    action = [0, 1, 2, 3]  # corresponds with ['up', 'right', 'down', 'left']

    weight_mp, store, count = {}, [], 1

    while count <= num_trajs:
        curr = possible_starts[np.random.randint(len(possible_starts))]
        x, y = curr # get coordinates
        T = []

        while (x, y) != (grid_size - 1, grid_size - 1) and len(T) < LEN_LIMIT[1] - 1:   # -1 to offset things
            #print("len T = ", len(T))
            is_success = False

            curr_policy = deepcopy(policy)
            p_index = [i for i in range(len(curr_policy))]

            while not is_success:

                p_dist = [sum(curr_policy[:i+1]) for i in range(len(p_index))]

                #print(p_index, p_dist)

                val = np.random.randint(p_dist[-1] + 1)

                #print("val = ", val)
                for i, v in enumerate(p_dist):

                    if val <= v:
                        direction = action[p_index[i]]
                        # so that we can easility identify next action according to policy distribution
                        del p_dist[i]
                        del p_index[i]
                        del curr_policy[i]
                        break;

                #print("modified direction = ", direction)
                new_x, new_y = x, y
                if direction == 0: # try up
                    new_x, new_y = x - 1, y
                elif direction == 1: # try right
                    new_x, new_y = x, y + 1
                elif direction == 2: # try down
                    new_x, new_y = x + 1, y
                elif direction == 3:  # try left
                    new_x, new_y = x, y -1
                #print("newx = {} and newy = {}".format(new_x, new_y))

                if (0 <= new_x <= grid_size -1) and (0 <= new_y <= grid_size -1):
                    is_success = True

                    # add to trajectories here
                    transition = ((x, y), direction, (new_x, new_y))
                    T.append(transition)
                    if transition not in weight_mp:
                        weight_mp[transition] = 1
                    else:
                        weight_mp[transition] += 1

                    x, y = new_x, new_y
                    #print("success is here")
        if (x, y) == (grid_size -1, grid_size -1):
            T.append(((x, y), direction, (x, y)))

        if LEN_LIMIT[0] <= len(T):
            store.append(T)
            count += 1

    return store, weight_mp




def get_gw_manual_rewards(grid_size, reverse_mp, fmap, policy, num_trajs, \
    LEN_LIMIT):

    store, weight_mp = generate_gridworld_trajs(grid_size, fmap, num_trajs, \
        policy, LEN_LIMIT)

    #new_store = deepcopy(store)
    new_store = []

    for Trajs in store:

        temp_store = []

        # randomly generate coefficients for reward shaping
        phi_x, phi_y = np.random.randint(1, grid_size//2), np.random.randint(1, grid_size//2)
        shaping_f = lambda x, y: phi_x*x + phi_y*y
        visited = set()    # this map marks trajectories that have been previously seen

        for T in Trajs:
            if T not in visited:
                (x, y), a, (x_p, y_p) = T
                # potential based shaping " R' = R + \gamma*phi(s') - phi(s)"
                # assumed gamma is 0.8
                R_shaped = (x + y) + 0.8*shaping_f(x_p, y_p) - shaping_f(x, y)
                temp_store.append(list(T) + [R_shaped]) # manually specified reward depends on length and feature x + y
                visited.add(T) # mark visited

        new_store.append(temp_store)

    return new_store, weight_mp


def compare_rewards (rA, rB):
    gamma = 0.8
    [rewA, rewA_weights], [rewB, rewB_weights] = rA, rB

    # first, ensure that we have the same state space for both rewA and rewB
    set_mp, set_a, set_b, X_a, X_b= set(), {}, {}, [], []

    for r in rewA:
        set_a[tuple(r[:-1])] = r[3]
        set_mp.add(tuple(r[:-1]))

        f = list(r[0]) + [r[1]] + list(r[2]) + [r[3]]
        X_a.append(f)

    for r in rewB:
        set_b[tuple(r[:-1])] = r[3]
        set_mp.add(tuple(r[:-1]))

        f = list(r[0]) + [r[1]] + list(r[2]) + [r[3]]
        X_b.append(f)

    X_a, X_b = np.asarray(X_a), np.asarray(X_b)

    # train linear regression models
    reg_a = LinearRegression().fit(X_a[:, :5], X_a[:, 5])
    reg_b = LinearRegression().fit(X_b[:, :5], X_b[:, 5])

    X1_a, X1_b = [], []
    # need to update this
    for r in set_mp:
        # deal with a
        trans_r = list(r[0]) + [r[1]] + list(r[2])

        # first fix the weights
        if r not in rewA_weights:
            rewA_weights[r] = 0.0000000000000001  # add a very small number
        if r not in rewB_weights:
            rewB_weights[r] = 0.0000000000000001

        if r in set_a:
            v = trans_r + [set_a[r]]
        else:
            a = reg_a.predict(np.array(trans_r).reshape(1, len(trans_r)))
            v = trans_r + list(a)

        tv = ((v[0], v[1]), v[2], (v[3], v[4]), v[5])
        X1_a.append(tv)

        # deal with b
        if r in set_b:
            v = trans_r + [set_b[r]]
        else:
            a = reg_b.predict(np.array(trans_r).reshape(1, len(trans_r)))
            v = trans_r + list(a)

        tv = ((v[0], v[1]), v[2], (v[3], v[4]), v[5])
        X1_b.append(tv)

    # next, we meed to execute the comparisons
    # do epic
    epic_a = canonicalize_epic(X1_a, gamma, False, [])
    epic_b = canonicalize_epic(X1_b, gamma, False, [])

    # do dard
    dard_a = canonicalize_dard(X1_a, rewA_weights, gamma, False, [])
    dard_b = canonicalize_dard(X1_b, rewB_weights, gamma, False, [])

    # do sard
    sard_a = canonicalize_sard(X1_a, gamma, False, [])
    sard_b = canonicalize_sard(X1_b, gamma, False, [])

    # compute the pearson distance and return ans
    X1_a, X1_b, epic_a, epic_b, dard_a, dard_b, sard_a, sard_b = \
        np.array(X1_a), np.array(X1_b), np.array(epic_a), np.array(epic_b), \
        np.array(dard_a), np.array(dard_b), np.array(sard_a), np.array(sard_b)

    X_dist = np.sqrt(1 - stats.pearsonr(X1_a[:, 3], X1_b[:, 3])[0])/np.sqrt(2)
    X_epic = np.sqrt(1 - stats.pearsonr(epic_a[:, 3], epic_b[:, 3])[0])/np.sqrt(2)
    X_dard = np.sqrt(1 - stats.pearsonr(dard_a[:, 3], dard_b[:, 3])[0])/np.sqrt(2)
    X_sard = np.sqrt(1 - stats.pearsonr(sard_a[:, 3], sard_b[:, 3])[0])/np.sqrt(2)

    return X_dist, X_epic, X_dard, X_sard


def gw_manual_comparisons (grid_size, num_trajs, LEN_LIMIT, \
    fmap, reverse_mp, policies):

    df_store = []

    for trial in range(5):

        temp_mean, temp_cv = [], []

        # select random policy and generate random trajectory
        policy_index = np.random.randint(len(policies))

        # simulated 200 trajectories and will select 1 for comparions, simulated
        # multiple trajectories inorder to get accurate weight_mp needed for DARD
        rew_random, random_weights = get_gw_manual_rewards(grid_size, reverse_mp,\
                fmap, policies[policy_index], num_trajs, LEN_LIMIT)

        rew_random  = rew_random[0]   #initialize the reward that we will compare with

        print("trial  = ", trial)
        for i, policy in enumerate(policies):

            print(" generating rewards for index = ", i)
            rews, trial_weights = get_gw_manual_rewards(grid_size, reverse_mp, \
                    fmap, policy, num_trajs, LEN_LIMIT)

            temp_arr = []
            for j, rew in enumerate(rews):
                X_dist, X_epic, X_dard, X_sard = compare_rewards([rew_random, \
                    random_weights], [rew, trial_weights])

                temp_arr.append([X_dist, X_epic, X_dard, X_sard])

            temp_arr = np.array(temp_arr)

            cv_dist = np.std(temp_arr[:, 0])/np.mean(temp_arr[:, 0])
            cv_epic = np.std(temp_arr[:, 1])/np.mean(temp_arr[:, 1])
            cv_dard = np.std(temp_arr[:, 2])/np.mean(temp_arr[:, 2])
            cv_sard = np.std(temp_arr[:, 3])/np.mean(temp_arr[:, 3])

            temp_mean.append([np.mean(temp_arr[:, 0]), np.mean(temp_arr[:, 1]),\
                np.mean(temp_arr[:, 2]), np.mean(temp_arr[:, 3])])

            temp_cv.append([cv_dist, cv_epic, cv_dard, cv_sard])
            # aggregate resuts tomorrow!!

        temp_mean, temp_cv = np.array(temp_mean), np.array(temp_cv)

        intra_cv = [np.mean(temp_cv[:, 0]), np.mean(temp_cv[:, 1]), \
                    np.mean(temp_cv[:, 2]), np.mean(temp_cv[:, 3])]

        inter_cv = [np.std(temp_mean[:, 0])/np.mean(temp_mean[:, 0]), \
                    np.std(temp_mean[:, 1])/np.mean(temp_mean[:, 1]), \
                    np.std(temp_mean[:, 2])/ np.mean(temp_mean[:, 2]), \
                    np.std(temp_mean[:, 3])/ np.mean(temp_mean[:, 3])]

        print("intra_cv = ", intra_cv)
        print("inter_cv = ", inter_cv)
        df_store.append(intra_cv + inter_cv)

    df = pd.DataFrame(df_store, columns = ["intra_cv_dist", "intra_cv_epic", \
        "intra_cv_dard",  "intra_cv_sard", "inter_cv_dist", "inter_cv_epic", \
        "inter_cv_dard", "inter_cv_sard"])

    return df




#policies = policies[47:]
# generate traejctories and their corresponding labels

def get_transition_matrix (fmap, grid_size, trajs_store):
    # create transition matrix P_a
    P_a = np.zeros((grid_size*grid_size, grid_size*grid_size, 4))

    next_mp = {}
    store = []

    for traj in trajs_store:
        new_trajs = []
        for s, a, sp in traj:

            if fmap[sp] not in next_mp:
                next_mp[fmap[sp]] = [None, None, None, None]

            next_mp[fmap[sp]][a] = fmap[s]

            new_trajs.append([fmap[s], a, fmap[sp]])
            P_a[fmap[s], fmap[sp], a] = 1
        store.append(new_trajs)
    return P_a, next_mp, store


def rewrite_reward (trajs, fmap, rew):
    # now rewrite rewards in the form (s, a, sp, r)
    temp_rew = set()
    for traj in trajs:
        for s, a, sp in traj:
            temp_rew.add((s, a, sp, rew[fmap[sp]]))

    return temp_rew


def gw_maxent_rewards (policies, grid_size, fmap, reverse_mp, \
        num_trajs, num_sets, maxent_iterations, LEN_LIMIT, gamma):

    df_store = []
    numTrials = 1
    for trial in range(numTrials):
        print("running trial {} out of {}".format(trial, numTrials))

        temp_mean, temp_cv = [], []

        # generate dummy trajectories for policy - [25, 25, 25, 25]
        default_trajs, weights_A = generate_gridworld_trajs(grid_size, fmap, \
            num_trajs, [25, 25, 25, 25], LEN_LIMIT)

        feat_map = np.zeros((grid_size*grid_size, 2))
        for k, v in fmap.items():
            feat_map[v] = k

        # get transition matrix for default trajs that we will compare with
        default_P_a, default_next_mp, default_store = \
            get_transition_matrix (fmap, grid_size, default_trajs)

        # now get the reward for the trajectory we will be comparing with
        default_rew = maxent_irl(feat_map, default_P_a, gamma, default_store, 0.8, \
            maxent_iterations, default_next_mp)

        rewA = rewrite_reward (default_trajs, fmap, default_rew)

        # now generate rewards for all the other trajectories
        for p_iter, policy in enumerate(policies):
            print("working on policy = {} out of {}".format(p_iter, len(policy)))
            temp_arr = []

            for run_iter in range(num_sets):
                #print("running {} out of {}".format(run_iter, num_sets))
                trajs, weights_B = generate_gridworld_trajs(grid_size, fmap, \
                    num_trajs, policy, LEN_LIMIT)

                P_a, next_mp, store = get_transition_matrix(fmap, grid_size, trajs)

                rew = maxent_irl(feat_map, P_a, gamma, store, 0.8, \
                    maxent_iterations, next_mp)

                rewB = rewrite_reward (trajs, fmap, rew)

                X_dist, X_epic, X_dard, X_sard = compare_rewards([rewA, weights_A],\
                    [rewB, weights_B])

                temp_arr.append([X_dist, X_epic, X_dard, X_sard])
                #print("policy = {} and  X_dist = {} and X_epic = {} and X_dard = {} and X_sard = {}".format(policy, X_dist, X_epic, X_dard, X_sard))
                #return P_a, feat_map, store


            temp_arr = np.array(temp_arr)
            cv_dist = np.std(temp_arr[:, 0])/np.mean(temp_arr[:, 0])
            cv_epic = np.std(temp_arr[:, 1])/np.mean(temp_arr[:, 1])
            cv_dard = np.std(temp_arr[:, 2])/np.mean(temp_arr[:, 2])
            cv_sard = np.std(temp_arr[:, 3])/np.mean(temp_arr[:, 3])


            temp_mean.append([np.mean(temp_arr[:, 0]), np.mean(temp_arr[:, 1]),\
                np.mean(temp_arr[:, 2]), np.mean(temp_arr[:, 3])])

            temp_cv.append([cv_dist, cv_epic, cv_dard, cv_sard])

            print(policy)
            for pdist in temp_arr:
                print(pdist)
            print("\n\n")
            #sys.exit()


        temp_mean, temp_cv = np.array(temp_mean), np.array(temp_cv)

        intra_cv = [np.mean(temp_cv[:, 0]), np.mean(temp_cv[:, 1]), \
                    np.mean(temp_cv[:, 2]), np.mean(temp_cv[:, 3])]

        inter_cv = [np.std(temp_mean[:, 0])/np.mean(temp_mean[:, 0]), \
                    np.std(temp_mean[:, 1])/np.mean(temp_mean[:, 1]), \
                    np.std(temp_mean[:, 2])/ np.mean(temp_mean[:, 2]), \
                    np.std(temp_mean[:, 3])/ np.mean(temp_mean[:, 3])]

        print("intra_cv = ", intra_cv)
        print("inter_cv = ", inter_cv)
        df_store.append(intra_cv + inter_cv)

    df = pd.DataFrame(df_store, columns = ["intra_cv_dist", "intra_cv_epic", \
        "intra_cv_dard",  "intra_cv_sard", "inter_cv_dist", "inter_cv_epic", \
        "inter_cv_dard", "inter_cv_sard"])

    df.to_csv("inter_vs_intra__5__" + str(time.time()))
    return df
