import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import numpy as np
import sys
import pandas as pd
from copy import deepcopy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from collections import deque, Counter
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
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import IRL_UTILS.maxent_utils as mu



def group_by_origin (rew_triples):
    mp = {}

    for r in rew_triples:
        if r[0] not in mp:
            mp[r[0]] = [r]
        else:
            mp[r[0]].append(r)

    return mp


def canonicalize_epic(triples, gamma):
    CONST = 0.00000000000001
    D = len(triples)    # This is the coverage

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


def canonicalize_sard(triples, gamma):
    """ This is the SARD metric: Sparsity Agnostic Reward Distance """

    CONST = 0.000000000000001
    cache_mp = {}   # to catch sums in case
    trans_mp, E_X3_X4, X3, X4 = get_trans_mp(triples)

    new_triples = []

    for t in triples:
        s, a, sp, r = t

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

    return new_triples


def canonicalize_dard(triples, gamma):
    """ This is the SARD metric: Sparsity Agnostic Reward Distance """

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

        NT = len(all_trans)
        NA = 1

        cr = r + gamma/(NA*NT + CONST) *E_sp_X2 - 1/(NA*NT + CONST)*E_s_X1 - gamma/((NA*NT)**2+CONST)*E_X1_X2
        #cr = r + (gamma/C1) *E_sp_X2 - (1/C2)*E_s_X1 - (gamma/C3)*E_X1_X2
        new_triples.append((s, a, sp, cr))

    return new_triples


def perform_regression (rewards, grid_size):
    # we want to fit a regression model for the state, action, next state triple

    X_feat = np.diag(np.ones(grid_size))
    Y_feat = np.diag(np.ones(grid_size))

    # now rewrite rewards vector in one-hot encoded form
    X, Y = [], []
    for r in rewards:
        s, a, sp, rew = r
        x1, y1 = s
        x2, y2 = sp

        X.append(list(X_feat[x1]) + list(Y_feat[y1]) + list(X_feat[x2]) + list(Y_feat[y2]))
        #X.append([x1, y1, x2 ,y2])
        Y.append(rew)

    X = np.array(X)
    X = X.reshape(len(X), len(X[0]))
    Y = np.array(Y)
    Y = Y.reshape(len(Y), 1)

    reg3 = LinearRegression().fit(X, Y)
    return reg3.coef_[0]


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

def manual_canonicalization(grid_rewards, state_rewards, grid_size, gamma):

    # These rewards are manually defined rather than from IRL
    TRAJS_DIR = "TRAJECTORIES"

    terminal = set([(grid_size - 1, grid_size - 1)])
    scale = np.random.randint(10, 30)/10
    relationship = "random"

    # read all files from TRAJECTORIES and re-write with linear rewards
    store_orig, store_epic, store_dard, store_sard, reward_labels = [], [], [], [], []
    label_mp = {}

    for fname in os.listdir(TRAJS_DIR):
        # now read the files in here
        print("fname = ", fname)
        with open(TRAJS_DIR + "/" + fname, "rb") as f:
            trajs_store = pickle.load(f)

        # to store labels
        label = tuple([int(i) for i in fname.replace(".pkl", '').split("_")])
        if label not in label_mp:
            label_mp[label] = len(label_mp)

        for traj in trajs_store:

            rew = induce_shaping(grid_rewards, traj, state_rewards, \
                gamma, terminal, relationship, scale)

            r_orig = rew
            r_epic = canonicalize_epic(rew, gamma)
            r_dard = canonicalize_dard(rew, gamma)
            r_sard = canonicalize_sard(rew, gamma)

            # original trajs, not canonicalized
            store_orig.append(r_orig)
            # perform epic canonicalization
            store_epic.append(r_epic)
            # perform dard canonicalization
            store_dard.append(r_dard)
            # perform sard canonicalization
            store_sard.append(r_sard)
            # append label
            reward_labels.append(label_mp[label])

    return store_orig, store_epic, store_dard, store_sard, reward_labels



def maxent_canonicalization(grid_size, gamma):

    # These rewards are manually defined rather than from IRL
    TRAJS_DIR = "TRAJECTORIES"

    # read all files from TRAJECTORIES and re-write with linear rewards
    store_orig, store_epic, store_dard, store_sard, reward_labels = [], [], [], [], []
    label_mp = {}
    """
    for fname in os.listdir(TRAJS_DIR):
        # now read the files in here
        with open(TRAJS_DIR + "/" + fname, "rb") as f:
            trajs_store = pickle.load(f)

        # to store labels
        label = tuple([int(i) for i in fname.replace(".pkl", '').split("_")])
        if label not in label_mp:
            label_mp[label] = len(label_mp)
    """
    for fname in os.listdir(TRAJS_DIR):
        print("working on ", fname)
        # now read the files in here
        with open(TRAJS_DIR + "/" + fname, "rb") as f:
            trajs_store = pickle.load(f)

        # to store labels
        label = tuple([int(i) for i in fname.replace(".pkl", '').split("_")])
        if label not in label_mp:
            label_mp[label] = len(label_mp)

        chunk_size = 5
        list_chunks = [trajs_store[i:i + chunk_size] for i in \
            range(0, len(trajs_store), chunk_size)]

        for traj in list_chunks:

            # setup for maxent
            observation_matrix, state_mp, action_mp, env_single, om, \
                expert_trajs = mu.fix_for_irl(traj)

            # perform maxent here
            rews = mu.perform_Maxent(env_single, om, observation_matrix, \
                is_linear = "nonlinear")

            # rewrite traj_set
            rew = []
            for T in traj:
                for t in T:
                    s, a, sp = t
                    r = rews[state_mp[s]] * rews[state_mp[sp]]
                    rew.append((s, a, sp, r))

            r_orig = rew
            r_epic = canonicalize_epic(rew, gamma)
            r_dard = canonicalize_dard(rew, gamma)
            r_sard = canonicalize_sard(rew, gamma)

            # original trajs, not canonicalized
            store_orig.append(r_orig)
            # perform epic canonicalization
            store_epic.append(r_epic)
            # perform dard canonicalization
            store_dard.append(r_dard)
            # perform sard canonicalization
            store_sard.append(r_sard)
            # append label
            reward_labels.append(label_mp[label])

    return store_orig, store_epic, store_dard, store_sard, reward_labels


def airl_canonicalization(grid_size, gamma):

    # These rewards are manually defined rather than from IRL
    TRAJS_DIR = "TRAJECTORIES"

    # read all files from TRAJECTORIES and re-write with linear rewards
    store_orig, store_epic, store_dard, store_sard, reward_labels = [], [], [], [], []
    label_mp = {}

    for fname in os.listdir(TRAJS_DIR):
        print("working on ", fname)
        # now read the files in here
        with open(TRAJS_DIR + "/" + fname, "rb") as f:
            trajs_store = pickle.load(f)

        # to store labels
        label = tuple([int(i) for i in fname.replace(".pkl", '').split("_")])
        if label not in label_mp:
            label_mp[label] = len(label_mp)

        chunk_size = 5
        list_chunks = [trajs_store[i:i + chunk_size] for i in \
            range(0, len(trajs_store), chunk_size)]

        for traj in list_chunks:

            # setup for maxent
            observation_matrix, state_mp, action_mp, env_single, om, \
                expert_trajs = mu.fix_for_irl(traj)

            # perform airl here
            rews = mu.perform_AIRL(env_single, expert_trajs)

            # rewrite traj_set
            rew = []
            for T in traj:
                for t in T:
                    s, a, sp = t
                    r = rews[state_mp[s]] * rews[state_mp[sp]]
                    rew.append((s, a, sp, r))

            r_orig = rew
            r_epic = canonicalize_epic(rew, gamma)
            r_dard = canonicalize_dard(rew, gamma)
            r_sard = canonicalize_sard(rew, gamma)

            # original trajs, not canonicalized
            store_orig.append(r_orig)
            # perform epic canonicalization
            store_epic.append(r_epic)
            # perform dard canonicalization
            store_dard.append(r_dard)
            # perform sard canonicalization
            store_sard.append(r_sard)
            # append label
            reward_labels.append(label_mp[label])

    return store_orig, store_epic, store_dard, store_sard, reward_labels


def ptirl_canonicalization(grid_size, gamma):

    # These rewards are manually defined rather than from IRL
    TRAJS_DIR = "TRAJECTORIES"

    label_mp = {}
    # read all files from TRAJECTORIES and re-write with linear rewards
    store_orig, store_epic, store_dard, store_sard, \
        reward_labels = [], [], [], [], []

    # open ptirl_rewards
    with open("PTIRL_UTILS/ptirl_rewards.pkl", "rb") as f:
        rewards = pickle.load(f)

    for index, element in enumerate(rewards):
        print("working on index = {} out of {}".format(index, len(rewards)))
        rew, label = element


        if str(label) not in label_mp:
            label_mp[str(label)] = len(label_mp)
        new_label = label_mp[str(label)]

        rew = [k for k in set(rew)]

        r_orig = rew
        r_epic = canonicalize_epic(rew, gamma)
        r_dard = canonicalize_dard(rew, gamma)
        r_sard = canonicalize_sard(rew, gamma)

        # append to main store
        store_orig.append(r_orig)
        store_epic.append(r_epic)
        store_dard.append(r_dard)
        store_sard.append(r_sard)
        reward_labels.append(new_label)

    return store_orig, store_epic, store_dard, store_sard, reward_labels




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

            direction = 8 # direction = 4 means that the transition is unfeasible realistically

            feasible_trans = {(x, y - 1):0, (x + 1, y - 1):1, (x + 1, y):2,
                    (x + 1, y + 1):3, (x, y + 1):4, (x - 1, y + 1):5,
                    (x - 1, y):6, (x - 1, y - 1):7}

            if (x1, y1) in feasible_trans:
                direction = feasible_trans[(x1, y1)]

            graph.append([s, direction, next_s, states[next_s] + 2*states[s]]) # linear reward inserted

    graph = {(s, a, sp): r for s, a, sp, r in graph if a < 8}   # graph is a dictionary
    return graph, states


def induce_shaping(graph, trajectory, states, gamma, terminal, relationship, scale):

    """ Induce shaping to a given trajectory """
    shaped_graph, potential = deepcopy(graph), deepcopy(states)

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

    # insert shaped rewards into trajectory
    new_traj = set()
    for s, a, sp in trajectory:
        try:
            #rew = shaped_graph[(s, a, sp)] + np.random.randint(-100, 100)
            rew = shaped_graph[(s, a, sp)] + gamma*potential[sp] - potential[s]
            new_traj.add((s, a, sp, rew))
        except:
            print("failed transition = {}", (s, a, sp))


    return list(new_traj)


def compute_dist(r_a, r_b):

    # create maps of the rewards
    r1 = {(s, a, sp):r for s, a, sp, r in r_a}
    r2 = {(s, a, sp):r for s, a, sp, r in r_b}

    # find intersection
    rews = []
    for t in r1:
        if t in r2:
            rews.append([r1[t], r2[t]])

    if len(rews) > 30:
        rews = np.array(rews)

        # compute distance here
        corr = stats.pearsonr(rews[:, 0], rews[:, 1])[0]
        d1 = (1/np.sqrt(2))*np.sqrt(1 - corr)

        return d1
    else:
        return 2  # means we have no intersection,worst case


def getcounts(reward_store, reward_labels, sample_points):

    store = {}

    for i in sample_points:
        r_val = reward_store[i]
        #print("working on i = ", i)
        dists = []
        #print("working on {}/{}".format(i, len(reward_store)))
        for j, r_val1 in enumerate(reward_store):
            d = compute_dist(r_val, r_val1)
            dists.append((d, reward_labels[i], reward_labels[j]))

        dists = Counter([k[2] for k in sorted(dists)[:20]])
        store[(i, reward_labels[i])] = sorted(dists.items(), \
            key = lambda x:x[1], reverse = True)

    preds = []
    for k in store:
        try:
            preds.append([k[1], store[k][0][0]])
        except:
            pass

    preds = np.array(preds)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    acc = accuracy_score(preds[:, 0], preds[:, 1])
    pre = precision_score(preds[:, 0], preds[:, 1], average = 'macro')
    rec = recall_score(preds[:, 0], preds[:, 1], average = 'macro')
    f1_s = f1_score(preds[:, 0], preds[:, 1], average = "macro")

    return acc, pre, rec, f1_s
