import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")

import random
sys.path.append("../../CURRENT_IRL")
# trying to use the seals module to create an env
from seals_dir.src.seals.base_envs import TabularModelPOMDP
from typing import Optional, Tuple
import seals_dir.src.seals.base_envs as base_envs
import os
import multiprocessing
import scipy
import time
import pickle
from scipy.stats import pearsonr
from copy import deepcopy
from functools import partial
from stable__baselines3.stable__baselines3.common.vec_env import DummyVecEnv

from imitation.src.imitation.algorithms.mce_irl import (
    MCEIRL,
    mce_occupancy_measures,
    mce_partition_fh,
    TabularPolicy,
    )

import torch as th
import seaborn as sns
from imitation.src.imitation.data import rollout
from imitation.src.imitation.rewards import reward_nets
from stable__baselines3.stable__baselines3 import PPO
from stable__baselines3.stable__baselines3.ppo import MlpPolicy
from imitation.src.imitation.algorithms.adversarial.airl import AIRL
from sklearn.linear_model import LinearRegression


# Reset the RNG to its initial state
random.seed()


def get_state_action_mp (mp, state_mp, action_mp):

    store = []
    for k in mp:
        traj = mp[k]

        temp = []
        for s, a, sp in traj:
            t_s, t_a, t_sp = tuple(s), tuple(a), tuple(sp)

            if t_s not in state_mp:
                state_mp[t_s] = len(state_mp) 
            if t_a not in action_mp:
                action_mp[t_a] = len(action_mp) 
            if t_sp not in state_mp:
                state_mp[t_sp] = len(state_mp)
                
            temp.append([state_mp[t_s], action_mp[t_a], state_mp[t_sp]])
            
        store.append(temp)
    return store 


def perform_PCA(X, n_components):

    # Standardize the data (important for PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Explained variance ratio
    explained_variance = pca.explained_variance_ratio_

    return pca, 1 - sum(explained_variance)

def traverse(FILENAME, store):

    f = h5py.File(FILENAME, "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]


    total_samples = 0

    count = 0

    trajs = []

    for ind in range(min(len(demos), 200)):
        ep = demos[ind]

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        actions = f["data/{}/actions".format(ep)][()]
        rewards = f["data/{}/rewards".format(ep)][()]

        temp  = []

        for i in range(1, len(states)):
            s_i = states[i-1]
            s_j = states[i]
            temp.append([s_i, 1, s_j])

        trajs.append(temp)

        store += list(states)

    return trajs


def compress_trajs (pca, traj_set, state_mp):

    scaler = StandardScaler()

    res = []
    for trajs in traj_set:  # 200

        x, xp = [], []
        for s, a, sp in trajs:  # 58 (each trajectory)

            x.append(s)
            xp.append(sp)

        x = np.round(pca.fit_transform(scaler.fit_transform(x)), 1)
        xp = np.round(pca.fit_transform(scaler.fit_transform(xp)), 1)

        temp = []
        for i in range(len(x)):
            f1 = tuple(float(b) for b in x[i])
            f2 = tuple(float(c) for c in xp[i])

            if f1 not in state_mp:
                state_mp[f1] = len(state_mp)
            if f2 not in state_mp:
                state_mp[f2] = len(state_mp)

            temp.append((state_mp[f1], 1, state_mp[f2]))

        res.append(temp)

    return res

def get_trajs (FILENAME_ph, FILENAME_mg, n_features):
    store, state_mp = [], {}

    t_ph = traverse(FILENAME_ph, store)
    t_mg = traverse(FILENAME_mg, store)

    pca, var_ratio = perform_PCA(store, n_features)

    t_ph = compress_trajs(pca, t_ph, state_mp)
    t_mg = compress_trajs(pca, t_mg, state_mp)

    return t_ph, t_mg, state_mp






def new_representation(T, S_MP, fixed_Trajs = True):

    # T is the original trajectories represented as integers
    # X_len is the number of X features
    # Y len is the number of Y features

    horizon = 0
    state_mp, action_mp, store = {}, {}, []

    triples = set()

    for traj in T:
        new_traj = []
        horizon = max(horizon, len(traj))

        for s1, a, sp1 in traj:

            s, sp = S_MP[s1], S_MP[sp1]
            a = 0
            if s not in state_mp:
                state_mp[s] = len(state_mp)
            if a not in action_mp:
                action_mp[a] = a #len(action_mp)
            if sp not in state_mp:
                state_mp[sp] = len(state_mp)
            ns, na, nsp = state_mp[s], action_mp[a], state_mp[sp]
            new_traj.append([ns, na, nsp])

            triples.add((s, a, sp))

        store.append(new_traj)

    # get transition matrix
    t_matrix = np.zeros((len(state_mp), len(action_mp), len(state_mp)))

    # policy matrix
    pi = np.zeros((len(state_mp), len(action_mp)))

    init_state_dist = np.zeros(len(state_mp))

    for traj in store:
        for i, val in enumerate(traj):
            s, a, sp = val
            if i == 0:
                init_state_dist[s] += 1
            # update transition matrix here
            t_matrix[s, a, sp] += 1
            pi[s, a] += 1

    # normalize t_matrix
    for s in range(len(t_matrix)):

        if sum(pi[s]) == 0:
            #pi[s][:] = 1/len(action_mp)
            pass
        else:
            pi[s][:] /= sum(pi[s])

        for a in action_mp.values():
            #for a in range(len(t_matrix[s])):
            row_sum = sum(t_matrix[s, a])
            if row_sum != 0:
                t_matrix[s, a] /= row_sum
            else:
                #pass
                t_matrix[s, a] = 1/len(state_mp)

    new_pi = np.tile(pi, (horizon, 1))
    pi = new_pi.reshape(horizon, len(state_mp), len(action_mp))

    # get reward matrix
    r_matrix = np.zeros(len(state_mp))

    init_state_dist /= sum(init_state_dist)

    if fixed_Trajs == True:
        observation_matrix = np.zeros((len(state_mp), 2*len(S_MP[0])))
    else:
        observation_matrix = np.zeros((len(state_mp), len(S_MP[0])))

    #observation_matrix = np.zeros((len(state_mp), X_LEN + Y_LEN))
    for k in state_mp:
        if fixed_Trajs == True:
            index = state_mp[k]
            observation_matrix[index][:] = list(k)
            #observation_matrix[index][:] = list(X_one_hot[x]) + list(Y_one_hot[y])
        else:
            index = state_mp[k]
            observation_matrix[index][:] = list(k)

    # create a tabular environment
    env_creator = partial(TabularModelPOMDP, transition_matrix=t_matrix, \
        observation_matrix = observation_matrix, reward_matrix = r_matrix, \
        horizon = horizon, initial_state_dist = init_state_dist,
    )

    env_single = env_creator()
    state_env_creator = lambda: base_envs.ExposePOMDPStateWrapper(env_creator())
    # This is just a vectorized environment because `generate_trajectories` expects one
    state_venv = DummyVecEnv([state_env_creator] * 1)

    #_, _, pi = mce_partition_fh(env_single)
    _, om = mce_occupancy_measures(env_single, pi=pi)

    expert_trajs = rollout.generate_without_policy(
        traj_store = store,
    )

    return action_mp, state_mp, t_matrix, r_matrix, horizon, init_state_dist, \
        env_single, state_venv, pi, om, expert_trajs, observation_matrix, triples



def perform_AIRL(observation_matrix, env_single, expert_trajs, traj, \
    airl_iterations = 20000, is_linear = "nonlinear"):

    state_env_creator = lambda: base_envs.ExposePOMDPStateWrapper(env_single)
    state_venv = DummyVecEnv([state_env_creator] * 1)

    if is_linear == "linear":
        airl_reward_net = reward_nets.BasicRewardNet(env_single.observation_space, \
            env_single.action_space, hid_sizes=[], use_action=False, \
            use_done=False, use_next_state=False)
    else:  # this is the nonlinear part
        airl_reward_net = reward_nets.BasicRewardNet(env_single.observation_space, \
            env_single.action_space, hid_sizes=[256, 128], use_action=False, \
            use_done=False, use_next_state=True)

    SEED = 42

    learner = PPO(env=env_single, policy=MlpPolicy, batch_size=16, \
        learning_rate=0.0001, n_epochs=5, seed=SEED, \
        observation_matrix = env_single.observation_matrix)

    airl_trainer = AIRL(demonstrations=expert_trajs, demo_batch_size=16, \
        gen_replay_buffer_capacity=2048, n_disc_updates_per_round=4, \
        venv=env_single, gen_algo=learner, reward_net=airl_reward_net,)

    airl_trainer.train(airl_iterations)

    return airl_reward_net

def perform_Maxent(state_mp, observation_matrix, env_single, pi, \
        trajs, LIN_EPS, is_linear = "nonlinear"):

    state_env_creator = lambda: base_envs.ExposePOMDPStateWrapper(env_single)
    state_venv = DummyVecEnv([state_env_creator] * 1)

    if is_linear == "linear":
        reward_net = reward_nets.BasicRewardNet(env_single.observation_space, \
            env_single.action_space, hid_sizes=[], use_action=False, \
            use_done=False, use_next_state=False)
    else:  # this is the nonlinear part
        reward_net = reward_nets.BasicRewardNet(env_single.observation_space, \
            env_single.action_space, hid_sizes=[256, 128], use_action=False, \
            use_done=False, use_next_state=False)

    all_states = th.tensor(env_single.observation_matrix, dtype = th.float32)

    # training on analytically computed occupancy measures
    _, om = mce_occupancy_measures(env_single, pi=pi)

    #print("om = \n")
    #print(om)

    mce_irl = MCEIRL(om, env_single, reward_net, linf_eps = LIN_EPS, \
        log_interval=50, optimizer_kwargs={"lr": 0.001}, \
        rng = np.random.default_rng(seed=np.random.randint(0, 10000)))

    occ_measure = mce_irl.train(max_iter = 400)

    return reward_net


def perform_intrinsic(S_MP, T):
    X, Y = [], []

    for traj in T:
        for s, a, sp, r in traj:

            # Flatten or concatenate state, action, next_state into a single feature vector
            # Assuming s, a, sp are all numerical and can be concatenated directly
            feature = np.concatenate((np.array(S_MP[s]).flatten(), \
                np.array([a]).flatten(), np.array(S_MP[sp]).flatten()))

            X.append(feature)
            Y.append(r)

        X = np.array(X)
        Y = np.array(Y)

    # Train the reward model using Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, Y)

    return model


def get_reward_models (traj_arr, iterations, state_mp, action_mp, TRAJ_DIV, rew_type):

    S_MP = {j:i for i, j in state_mp.items()}
    A_MP = {j:i for i, j in action_mp.items()}

    REW_MP = {}

    for i in range(len(traj_arr)):

        T_set = traj_arr[i]

        if i not in REW_MP:
            REW_MP[i] = []

        L = len(T_set) // TRAJ_DIV

        for j in range(L):

            try:

                T = T_set[j*TRAJ_DIV: (j+1)*TRAJ_DIV]

                action_mp, state_mp, t_matrix, r_matrix, horizon, init_state_dist, \
                env_single, state_venv, pi, om, expert_trajs, observation_matrix, \
                    traj_triples = new_representation(T, S_MP, fixed_Trajs = False)

                if rew_type == "airl":
                    airl_iterations = iterations
                    rews = perform_AIRL(observation_matrix, env_single, expert_trajs, \
                        traj_triples, airl_iterations, "nonlinear")

                elif rew_type == "maxent":
                    maxent_iterations = iterations

                    LIN_EPS = 1
                    rews = perform_Maxent(state_mp, observation_matrix, env_single, \
                        pi, traj_triples, LIN_EPS, "nonlinear")

                elif rew_type == "intrinsic":
                    rews = perform_intrinsic(S_MP, T)

                REW_MP[i].append((rews, traj_triples))
            except:
                print("data error ")

    return REW_MP


def extract_states(trajs):

    all_states = set()
    for s, a, sp in trajs:
        all_states.add((s, sp))
    return all_states


def canonicalize_rewards(R1, T11, T22, model, gamma):

    trajs = [(tuple(T11[i].tolist()), 1, tuple(T22[i].tolist()), R1[i]) for i in range(len(T11))]

    BM_threshold = 50
    nr_epic = canonicalize_u_epic(BM_threshold, trajs, gamma)
    nr_dard = canonicalize_u_dard(BM_threshold, trajs, gamma)
    nr_srrd = canonicalize_u_srrd(BM_threshold, trajs, gamma)

    return nr_epic, nr_dard, nr_srrd


def get_instrinsic(X, X_P):

    potential = {}
    gamma = 0.7

    T = []
    for i in range(len(X)):

        t1, t2 = tuple(X[i]), tuple(X_P[i])

        if t1 not in potential:
            potential[t1] = np.random.randint(0, 50)

        if t2 not in potential:
            potential[t2] = np.random.randint(0, 50)

        m = list(X[i]) + list(X_P[i])
        R = np.linalg.norm(m) + gamma*potential[t2] - potential[t1]

        T.append(R)

    return T


def compute_dist(REW_MP, label1, i1, label2, i2, rew_type, gamma):

    model1, T1 = REW_MP[label1][i1]
    model2, T2 = REW_MP[label2][i2]

    T1 = extract_states(T1)
    T2 = extract_states(T2)

    all_T = np.array(list(T1.union(T2)))

    T11 = th.as_tensor(np.array(all_T[:, 0]), dtype = th.float32)
    T22 = th.as_tensor(np.array(all_T[:, 1]), dtype = th.float32)

    if rew_type == "maxent":

        R1 = model1(T22, None, None, None).detach().numpy()
        R2 = model2(T22, None, None, None).detach().numpy()

    if rew_type == "airl":

        R1 = model1(T11, None, T22, None).detach().numpy()
        R2 = model2(T11, None, T22, None).detach().numpy()

    elif rew_type == "intrinsic":
        # will differ due to potential shaping
        t1, t2 = all_T[:, 0], all_T[:, 1]

        X = []
        for i in range(len(t1)):
            feature = np.concatenate((np.array(t1[i]).flatten(), \
                np.array([1]).flatten(), np.array(t2[i]).flatten()))
            X.append(feature)

        X = np.array(X)

        R1 = model1.predict(X)
        R2 = model2.predict(X)

        #R1 = get_instrinsic(all_T[:, 0], all_T[:, 1])
        #R2 = get_instrinsic(all_T[:, 0], all_T[:, 1])

    R1_epic, R1_dard, R1_srrd = canonicalize_rewards(R1, T11, T22, model1, gamma)
    R2_epic, R2_dard, R2_srrd = canonicalize_rewards(R2, T11, T22, model2, gamma)

    d_R = np.sqrt((1 - pearsonr(R1, R2)[0])/2)
    d_R_epic = np.sqrt((1 - pearsonr(R1_epic, R2_epic)[0])/2)
    d_R_dard = np.sqrt((1 - pearsonr(R1_dard, R2_dard)[0])/2)
    d_R_srrd = np.sqrt((1 - pearsonr(R1_srrd, R2_srrd)[0])/2)

    return d_R, d_R_epic, d_R_dard, d_R_srrd

def return_top(d_arr, description):
    res1 = sum([1 for i, j in d_arr[:15] if j == True])/15*100
    res2 = sum([1 for i, j in d_arr[:20] if j == True])/20*100
    res3 = sum([1 for i, j in d_arr[:25] if j == True])/25*100
    print(description + "   ", res1, res2, res3)
    return res2


def get_dist(keys, REW_MP, agent_keys, all_keys, rew_type, gamma):

    d_arr, d_arr_epic, d_arr_dard, d_arr_srrd = [], [], [], []

    print("\n agent keys")
    print(agent_keys)
    print("\n")
    print(all_keys)
    print("")

    for k1, i1 in agent_keys:
        for k2, i2 in all_keys:

            if (k1, i1) != (k2, i2):
            
                d = compute_dist(REW_MP, keys[k1], i1, keys[k2], i2, rew_type, gamma)

                d_arr.append((d[0], k1 == k2))
                d_arr_epic.append((d[1], k1 == k2))
                d_arr_dard.append((d[2], k1 == k2))
                d_arr_srrd.append((d[3], k1 == k2))

            #print(d)

    d_arr.sort()
    d_arr_epic.sort()
    d_arr_dard.sort()
    d_arr_srrd.sort()

    print("\n d_arr srrd sorted = ", d_arr_srrd[:20], " ", len(d_arr_srrd), "\n")

    r_norm = return_top(d_arr, "normal")
    r_epic = return_top(d_arr_epic, "epic")
    r_dard = return_top(d_arr_dard, "dard")
    r_srrd = return_top(d_arr_srrd, "srrd")

    return r_norm, r_epic, r_dard, r_srrd


def compute_distances(REW_MP, rew_type, gamma):

    keys = list(REW_MP.keys())

    print("keys = ", keys, len(keys))

    L = len(keys)


    res = []
    for i in range(L-1, -1, -1):
        print("label = ", i)
        # gonna do a sampling technique
        within_keys = [(i, k1) for k1 in range(len(REW_MP[keys[i]]))]

        all_keys = random.sample(within_keys, min(len(within_keys), 10))
        agent_keys = deepcopy(all_keys)

        other_keys = []
        for j in range(L-1, -1, -1):
            if j != i:
                other_keys += [(j, k) for k in range(len(REW_MP[keys[j]]))]

        # now sample 30 items from m
        other_keys = random.sample(other_keys, min(len(other_keys), 10*(L-1)))

        all_keys += other_keys   #should have a size of 30

        print("all keys = ", all_keys, "  length = ", len(all_keys))
        d = get_dist(keys, REW_MP, agent_keys, all_keys, rew_type, gamma)
        res.append(d)

    return res


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

def group_by_origin (rew_triples):
    mp = {}

    for r in rew_triples:
        if r[0] not in mp:
            mp[r[0]] = [r]
        else:
            mp[r[0]].append(r)
    return mp

def canonicalize_u_epic(BM_threshold, triples, gamma):
    """ This one uses unbiased estimates """

    CONST = 0.0000000000001
    D = len(triples)    # This is the coverage

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

        new_triples.append(can_rew)

    return np.array(new_triples)


def canonicalize_u_dard(BM_threshold, triples, gamma):
    """ This is the DARD metric, uses unbiased estimates  """

    CONST = 0.0000000000001
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

        new_triples.append(cr)

    return np.array(new_triples)


def canonicalize_u_srrd(BM_threshold, triples, gamma):
    """
    This is the SRRD metric: Sparsity Resilient Reward Distance

    Implements SRRD using unbiased estimates.
    """
    CONST = 0.0000000000001
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

        new_triples.append(cr)

    return np.array(new_triples)



def u_get_trans_mp (triples):
    """ Returns an unbiased estimate """

    CONST = 0.0000000000001
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
    CONST = 0.0000000000001
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



def perform_PCA(X, n_components):

    # Standardize the data (important for PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Explained variance ratio
    explained_variance = pca.explained_variance_ratio_

    return pca, scaler, 1 - sum(explained_variance)
