import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import numpy as np
import sys
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


def group_by_origin (rew_triples):
    mp = {}

    for r in rew_triples:
        if r[0] not in mp:
            mp[r[0]] = [r]
        else:
            mp[r[0]].append(r)

    return mp


def canonicalize_epic(triples, gamma, shrink, index_zeros):
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

    Da = 1
    N = len(Ds)*Da + CONST
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

    E_X_Xp /= (len(triples) + CONST)
    return trans_mp, E_X_Xp, X, Xp

def helper_get_sets_a(X1, trans_mp, CONST, cache_mp):
    X5, E_X1_X5, count = set(), 0, CONST

    for s in X1:
        if s in trans_mp:
            for ns in trans_mp[s]:
                X5.add(ns)
                #if ns in trans_mp[s]:
                for t1 in trans_mp[s][ns]:
                    E_X1_X5 += t1[3]
                    count += 1

    E_X1_X5 /= count
    return E_X1_X5, X5


def helper_get_sets (X1, trans_mp, CONST, cache_mp):
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


def canonicalize_sard(triples, gamma, shrink, index_zeros):
    """ This is the SARD metric: Sparsity Agnostic Reward Distance """

    a1 = time.time()

    CONST = 0.000000000000001
    cache_mp = {}   # to catch sums in case
    trans_mp, E_X3_X4, X3, X4 = get_trans_mp(triples)

    #a2 = time.time()
    #print("time to get x3, x4 = ", a2 -a1)

    new_triples = []

    for t in triples:
        s, a, sp, r = t
        #print("s = {} and sp = {}".format(s, sp))
        #print("")
        #a3 = time.time()
        E_sp_X1, X1 = helper_get_sets ([sp], trans_mp, CONST, cache_mp)
        E_s_X2, X2 = helper_get_sets ([s], trans_mp, CONST, cache_mp)
        E_X1_X5, X5 = helper_get_sets (X1, trans_mp, CONST, cache_mp)
        E_X2_X6, X6 = helper_get_sets (X2, trans_mp, CONST, cache_mp)
        a7 = time.time()

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


def canonicalize_dard(triples, weights_mp, gamma, shrink, index_zeros):
    """ This is the SARD metric: Sparsity Agnostic Reward Distance """

    CONST = 0.0000000000000001
    trans_mp, _, _, _ = get_trans_mp(triples)
    cache_mp = {}
    new_triples = []

    # compute na, ns
    # then estimate things

    #print("triples = ", triples)

    for t in triples:
        s, a, sp, r = t

        #C1, C2, C3 = CONST, CONST, CONST
        _, X2 = helper_get_sets ([sp], trans_mp, CONST, cache_mp)
        _, X1 = helper_get_sets ([s], trans_mp, CONST, cache_mp)

        all_trans, E_X1_X2,  actions = set(), 0, set()

        for x in X1:
            if x in trans_mp:
                for y in X2:
                    if y in trans_mp[x]:
                        for trans in trans_mp[x][y]:
                            s1, a1, sp1, r1 = trans
                            E_X1_X2 += r1
                            #all_trans.add((s1, a1, sp1))
                            actions.add(a1)
                            #C3 += 1

        E_sp_X2, E_s_X1 = 0, 0

        for x in X2:
            all_trans.add(x)
            if x in trans_mp[sp]:
                for trans in trans_mp[sp][x]:
                    s2, a2, sp2, r2 = trans
                    E_sp_X2 += r2
                    #C1 += 1

        for x in X1:
            all_trans.add(x)
            if x in trans_mp[s]:
                for trans in trans_mp[s][x]:
                    s3, a3, sp3, r3 = trans
                    E_s_X1 += r3
                    #C2 += 1
        #print("")
        #print("all_trans = ", all_trans)
        #print("")
        #sys.exit()
        NT = len(all_trans)
        NA = 1

        cr = r + gamma/(NA*NT + CONST) *E_sp_X2 - 1/(NA*NT + CONST)*E_s_X1 - gamma/((NA*NT)**2+CONST)*E_X1_X2
        #cr = r + (gamma/C1) *E_sp_X2 - (1/C2)*E_s_X1 - (gamma/C3)*E_X1_X2

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


def process_with_index (irl_rewards, trajs, index_store, grid_size, \
        action_size, rollover_type):
    """
    We collect the trajectories we sample and find all possible transitions

    We also perform linear regression based on training sample from the
    rewards we get after the appropriate removal.

    The r**2 score is based on the testing set based on a training to testing
    ration of 60:40
    """

    # First remove rewards here and then train a model
    rewards, rewards_linear = deepcopy(irl_rewards), deepcopy(irl_rewards)
    index_set = set(deepcopy(index_store))

    # reward stats
    reward_arr, trans_set, from_states, to_states = [], set(), set(), set()

    for T in trajs:
        rew = 0
        for t in T:
            rew += t[3]
            trans_set.add(t)
        reward_arr.append(rew)

    # iterate through indices
    rewards_zeroed = []
    for index, reward in enumerate(rewards):
        if index in index_set:
            rewards_zeroed.append(reward)
        else:
            rewards_linear[index][3] = None

    return rewards_zeroed



def perform_regression (rewards, grid_size, action_size, mode, rollover_type):
    # we want to fit a regression model for the state, action, next state triple
    v = np.diag([1 for i in range(grid_size)])

    actions = np.diag([0 for i in range(action_size)])

    # now rewrite rewards vector in one-hot encoded form
    X, Y = [], []
    for r in rewards:
        s, a, sp, rew = r
        x1, y1 = s
        x2, y2 = sp

        #m = list(v[x1]) + list(v[y1]) + list(v[x2]) + list(v[y2])
        #X.append(m)
        #X.append([x1, y1, 0, x2 ,y2])
        X.append([x1, y1, x2 ,y2])
        Y.append(rew)

    X = np.array(X)
    X = X.reshape(len(X), len(X[0]))
    Y = np.array(Y)
    Y = Y.reshape(len(Y), 1)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # linear regression
    #print(len(X_train), len(X_test))
    reg3 = LinearRegression().fit(X_train, y_train)
    reg_score3 = reg3.score(X_test, y_test)


    mse = mean_squared_error(reg3.predict(X_test), y_test)

    #if mode == 'b':
    #    print("mse = ", mse)
    model = reg3

    return model, v, actions, [reg_score3, 1, len(X_test)], len(X_train), len(X_test)



def fully_connected_mdp (num_states, relationship):

    states = {}   # so states contains the index and the representation etc
    state_rewards = {} # gives rewards per state

    for i in range(num_states):
        for j in range(num_states):
            states[(i, j)] = 0 # 0 is the reward

    graph = []
    if relationship == 'linear':
        # parameters for a linear relationship
        a, b, c = np.random.randint(1, 100), np.random.randint(1, 100), np.random.randint(1, 100)

        for s in states:
            x, y = s
            states[s] = a*x + b*y + c # update reward for all states
            #states[s][1] = a*(x**2) + b*(y**2) + (x*y) + c # update reward for all states

    elif relationship == 'polynomial':

        a, b, c = np.random.randint(1, 5), np.random.randint(1, 5), np.random.randint(0, 100)/100
        degree = [i for i in range(np.random.randint(2, 5))]
        for s in states:
            x, y = s
            for deg in degree:
                states[s] += a*((x/10)**deg) + b*((y/10)**deg) + c

    elif relationship == 'sinusoidal':

        a, b, c = np.random.randint(1, 5), np.random.randint(1, 5), np.random.randint(0, 100)/100
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

            direction = 4 # direction = 4 means that the transition is unfeasible realistically

            feasible_trans = {(x-1, y): 0, (x, y+1):1, (x+1, y):2, (x, y-1):3}

            if (x1, y1) in feasible_trans:
                direction = feasible_trans[(x1, y1)]

            graph.append([s, direction, next_s, states[next_s] + 2*states[s]]) # linear reward inserted

    return graph, states


def induce_shaping(graph, states, gamma, terminal, relationship, scale):

    shaped_graph, potential, state_mp = deepcopy(graph), deepcopy(states), deepcopy(states)

    for n in potential:
        potential[n] = 0

    if relationship == 'linear':
        # parameters for a linear relationship
        a, b, c = np.random.randint(1, 100), np.random.randint(1, 100), np.random.randint(1, 100)

        # now induce shaping effects to the states
        for s in potential:
            #if s not in terminal:
            x, y = s
            potential[s] = scale * (a*x + b*y + c)   # shaping based on linear reward

    elif relationship == 'polynomial':

        a, b, c = np.random.randint(1, 5), np.random.randint(1, 5), np.random.randint(0, 100)/100
        degree = [i for i in range(np.random.randint(2, 5))]

        for s in potential:
            #if s not in terminal:
            x, y = s
            for deg in degree:
                potential[s] += scale * (a*((x/10)**deg) + b*((y/10)**deg) + c)

    elif relationship == 'sinusoidal':

        a, b, c = np.random.randint(1, 5), np.random.randint(1, 5), np.random.randint(0, 100)/100
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

    return shaped_graph, potential




def run_trial(args):
    """ Run a single trial of reward canonicalization """

    seed()
    D_EPIC = []

    try:
        index, gamma, grid_size, scale, relationship, state_rewards, policy, \
            action_size, rollover_type, NUM_TRAJS, irl_rewards, \
            shaped_irl_rewards, full_indices, MAX_SIZE = args

        print("running trial = ", index)
        alpha = time.time()

        # run each independent trial here!
        Nm = set() # coverage and number of states Ds

        #a11 = time.time()

        # sample the ground truth reward, from a distribution of starting states and
        # returns a shaped reward and an unshaped reward after traversing graph using given policy
        trajs, shaped_trajs = policy_rollover(state_rewards, grid_size, \
            NUM_TRAJS, policy, shaped_irl_rewards, rollover_type, MAX_SIZE)

        #a12 = time.time()
        #print("time to generate full trajs = ", a12 - a11)
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

        can_original = canonicalize_epic(irl_rewards, gamma, True, index_zeros)
        shaped_can_original = canonicalize_epic(shaped_irl_rewards, gamma, \
            True, shaped_index_zeros)

        # dealing with epic on sparse reward samples
        sparse_epic = canonicalize_epic(rewards_zeroed, gamma, False, \
            index_zeros)
        shaped_sparse_epic = canonicalize_epic(shaped_rewards_zeroed, \
            gamma, False, shaped_index_zeros)

        sparse_sard = canonicalize_sard(rewards_zeroed, gamma, False, \
            index_zeros)
        shaped_sparse_sard = canonicalize_sard(shaped_rewards_zeroed, \
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

    except:
        print("Error encountered")

    return D_EPIC

def execute_process(args):

    grid_sizes, gamma, policy, rollover_arr, trajs_arr, relationships, \
        ITERATIONS, scale_arr, full_indices, MAX_SIZE = deepcopy(args)

    arg_arr = []
    count_index = 0

    for grid_size in grid_sizes:
        for relationship in relationships:
            for scale_arg in scale_arr:

                action_size, terminal = 1, set([(grid_size - 1, grid_size - 1)])

                for iteration in range(ITERATIONS):
                    irl_rewards, state_rewards = fully_connected_mdp(grid_size, relationship)   # linear rewards

                    scale = np.random.randint(10, 30)/10

                    shaped_irl_rewards, potential = induce_shaping(irl_rewards, state_rewards, \
                        gamma, terminal, relationship, scale)

                    for rollover_type in rollover_arr:
                        for num_trial in trajs_arr:
                            argument = [count_index, gamma, grid_size, scale, \
                                relationship, state_rewards, policy, \
                                action_size, rollover_type, num_trial, \
                                irl_rewards, shaped_irl_rewards, full_indices, \
                                MAX_SIZE]

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
        'rollover', 'num_trajs', 'original', 'canonical', 'epic', 'dard', 'sard', \
        'D', 'Nm'])

    return df


def compute_featmp(GRIDSIZE, feat_type):
    STATES_LEN = GRIDSIZE*GRIDSIZE
    num_states = [i for i in range(STATES_LEN)][::-1]

    if feat_type == 'original':    # original xy features
        fmap = {(x, y):num_states.pop() for x in range(GRIDSIZE) for y in \
             range(GRIDSIZE)}

    elif feat_type == 'binary':   # binary features

        def int_to_bin(number):
            binary = bin(number)[2:].zfill(5)  # Convert number to binary and pad with leading zeros
            binary_list = [int(bit) for bit in binary]  # Convert each character in binary to an integer
            return binary_list

        fmap = {tuple(int_to_bin(x) + int_to_bin(y)) : num_states.pop() \
                 for x in range(GRIDSIZE) for y in range(GRIDSIZE)}

    #elif feat_type == 'onehot':
    reverse_mp = {fmap[i]:i for i in fmap}
    return fmap, reverse_mp

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

    store, count = [], 1

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

                    x, y = new_x, new_y
                    #print("success is here")
        if (x, y) == (grid_size -1, grid_size -1):
            T.append(((x, y), direction, (x, y)))

        if LEN_LIMIT[0] <= len(T):
            store.append(T)
            count += 1

    return store
