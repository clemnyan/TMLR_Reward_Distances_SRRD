# In this code, we want to run adversarial IRL for the drone dataset
import os
import pandas as pd
import numpy as np
import sys
import pickle
import gzip

sys.path.append("../../CURRENT_IRL")
from collections import Counter
from functools import partial

import warnings
# Ignore all warnings (not recommended)
warnings.filterwarnings("ignore")

# trying to use the seals module to create an env
from seals_dir.src.seals.base_envs import TabularModelPOMDP
import seals_dir.src.seals.base_envs as base_envs
from copy import deepcopy
from stable__baselines3.stable__baselines3.common.vec_env import DummyVecEnv

from imitation.src.imitation.algorithms.mce_irl import (
    MCEIRL,
    mce_occupancy_measures,
    mce_partition_fh,
    TabularPolicy,
)

import torch as th
import matplotlib.pyplot as plt
import seaborn as sns
from imitation.src.imitation.data import rollout
from imitation.src.imitation.rewards import reward_nets
from stable__baselines3.stable__baselines3 import PPO
from stable__baselines3.stable__baselines3.ppo import MlpPolicy
from imitation.src.imitation.algorithms.adversarial.airl import AIRL

def find_one_hot_keys(MAIN_DIR, PICKLED_DIR, STATE_ACTION_MAPDIR, total_num):
    """ Find limits for one-hot-encoding from the entire dataset """

    # check if we have test sample points stored already
    SAMPLE_PKL = PICKLED_DIR + '/sample_points.pkl'

    if not os.path.isdir(PICKLED_DIR):
        os.mkdir(PICKLED_DIR)

        with gzip.open(STATE_ACTION_MAPDIR, 'rb') as f:
            state_mp, action_mp = pickle.load(f)
            s_k = np.array(list(state_mp.keys()))   #state keys
            max_keys = [max(s_k[:, i]) for i in range(len(s_k[0]))]   #max_vals

        max_keys = [np.arange(i) + 1 for i in max_keys]
        arr_set = set()
        while len(arr_set) < total_num:
            val = tuple([arr[np.random.randint(0, len(arr))] for arr in max_keys])
            if val not in arr_set:
                arr_set.add(val)

        arr_set = [list(i) for i in arr_set]
        with open(SAMPLE_PKL, "wb") as f:
            pickle.dump([max_keys, arr_set], f)

    else:
        with open(SAMPLE_PKL, "rb") as f:
            max_keys, arr_set = pickle.load(f)

    return max_keys, sorted(arr_set)


def fix_for_irl (trajs):
    """
    The goal in this method is to preprocess a group of trajectories in form
    of a pandas dataframe and format to ensure that we get:
        - observation matrix: relating to each state and associated observation
        - env_single: Virtual environment to run IRL (adversarial and maxent)
        - expert trajs: input trajetories assuming they are derived from an expert
    """
    # t_mp - is the transition matrix
    t_mp, state_mp, action_mp, traj_store, horizons = {}, {}, {}, [], []
    init_mp = Counter()  # initial distribution of states
    traj_store = []

    for traj in trajs:
        horizons.append(len(traj))
        temp_traj = []

        for i, t in enumerate(traj):
            # get the transition
            s, a, sp = t

            if s not in state_mp:
                state_mp[s] = len(state_mp)
            if a not in action_mp:
                action_mp[a] = len(action_mp)
            if sp not in state_mp:
                state_mp[sp] = len(state_mp)

            temp_traj.append([state_mp[s], action_mp[a], state_mp[sp]])

            if i == 1:
                init_mp[state_mp[s]] += 1

            transition = (s, a, sp)
            if transition not in t_mp:
                t_mp[transition] = 1
            else:
                t_mp[transition] += 1

            traj_store.append(temp_traj)

    # horizon
    horizon = np.mean(horizons, dtype = np.int64)

    # create a transition_matrix
    t_matrix = np.zeros((len(state_mp), len(action_mp), len(state_mp)))
    pi = np.zeros((len(state_mp), len(action_mp)))   # policy

    init_state_dist = np.zeros(len(state_mp))
    for s in init_mp:
        init_state_dist[s] = init_mp[s]

    # normalize init states
    init_state_dist /= sum(init_state_dist)

    for t in t_mp:
        s, a, sp = state_mp[t[0]], t[1], state_mp[t[2]]
        t_matrix[s, a, sp] += t_mp[t]
        pi[s, a] += t_mp[t]

    # normalize the t_matrix
    for s in range(len(t_matrix)):
        if sum(pi[s]) == 0:
            pi[s][:] = 1/len(action_mp)
        else:
            pi[s][:] /= sum(pi[s])

        for a in action_mp:
            #for a in range(len(t_matrix[s])):
            row_sum = sum(t_matrix[s, a])
            if row_sum != 0:
                t_matrix[s, a] /= row_sum
            else:
                t_matrix[s, a] = 1/len(state_mp)

    new_pi = np.tile(pi, (horizon, 1))
    pi = new_pi.reshape(horizon, len(state_mp), len(action_mp))

    # initialize reward matrix
    r_matrix = np.zeros(len(state_mp))

    # create observation_matrix
    observation_matrix = np.zeros((len(state_mp), 2))
    for ns, s in state_mp.items():
        observation_matrix[s][:] = ns

    # create a tabular environement
    env_creator = partial(TabularModelPOMDP, transition_matrix=t_matrix, \
        observation_matrix = observation_matrix, reward_matrix = r_matrix, \
        horizon = horizon, initial_state_dist = init_state_dist)

    env_single = env_creator()

    _, om = mce_occupancy_measures(env_single, pi=pi)
    # will need to fix store here, so that I can rewrite trajectories
    expert_trajs = rollout.generate_without_policy(traj_store = traj_store)
    # now create observation matrix based on state_mp
    return observation_matrix, state_mp, action_mp, env_single, om, expert_trajs


def perform_Maxent(env_single, om, observation_matrix, is_linear = "nonlinear"):

    # create a virtual environemnt
    state_env_creator = lambda: base_envs.ExposePOMDPStateWrapper(env_single)
    state_venv = DummyVecEnv([state_env_creator] * 1)

    # get all states
    all_states = th.tensor(observation_matrix, dtype = th.float32)

    if is_linear == "linear":
        reward_net = reward_nets.BasicRewardNet(env_single.observation_space, \
            env_single.action_space, hid_sizes=[], use_action=False, \
            use_done=False, use_next_state=False)
    else:  # this is the nonlinear part
        reward_net = reward_nets.BasicRewardNet(env_single.observation_space, \
            env_single.action_space, hid_sizes=[256, 64], use_action=False, \
            use_done=False, use_next_state=False)

    mce_irl = MCEIRL(
        om,
        env_single,
        reward_net,
        log_interval=20,
        optimizer_kwargs={"lr": 0.001},
        rng=np.random.default_rng(),
    )

    maxent_rewards = reward_net(th.as_tensor(observation_matrix, \
        dtype = th.float32), None, None, None)

    maxent_rewards = maxent_rewards.detach().numpy()
    return maxent_rewards


def perform_AIRL(env_single, expert_trajs, ITERATIONS = 5000, \
        is_linear = "nonlinear"):

    state_env_creator = lambda: base_envs.ExposePOMDPStateWrapper(env_single)
    state_venv = DummyVecEnv([state_env_creator] * 1)

    if is_linear == "linear":
        airl_reward_net = reward_nets.BasicRewardNet(env_single.observation_space, \
            env_single.action_space, hid_sizes=[], use_action=False, \
            use_done=False, use_next_state=False)
    else:  # this is the nonlinear part
        airl_reward_net = reward_nets.BasicRewardNet(env_single.observation_space, \
            env_single.action_space, hid_sizes=[256, 128, 64], use_action=False, \
            use_done=False, use_next_state=False)

    SEED = np.random.randint(10, 1000)

    learner = PPO(env=env_single, policy=MlpPolicy, batch_size=16, \
        learning_rate=0.001, n_epochs=20, seed=SEED, \
        observation_matrix = env_single.observation_matrix)

    airl_trainer = AIRL(demonstrations=expert_trajs, demo_batch_size=16, \
        gen_replay_buffer_capacity=2048, n_disc_updates_per_round=4, \
        venv=env_single, gen_algo=learner, reward_net=airl_reward_net,)

    airl_trainer.train(ITERATIONS)

    #airl_rewards = airl_reward_net(th.as_tensor(env_single.observation_matrix, \
    #    dtype = th.float32), None, None, None)
    airl_rewards = airl_reward_net(th.as_tensor(env_single.observation_matrix, \
        dtype = th.float32), None, None, None)

    return airl_rewards.detach().numpy()
