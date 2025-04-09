import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append('irl-imitation')

from reward_distance_utils import execute_process
import pandas as pd
import numpy as np
from copy import deepcopy
import pickle
import time
from scipy import stats
import multiprocessing
from numpy.random import randn, seed


def init_pool_processes():
    seed()

#relationships = ['sinusoidal']
relationships = ['linear', 'polynomial', 'sinusoidal', 'random']
grid_sizes = [20]

#rollover_arr = ['rollover']
rollover_arr = ['rollover', 'random_rollover']

trajs_arr = [1, 2, 3, 4, 5, 6, 7, 8, 8, 10, 12, 13, 15, 17, 19, 20, 22, 25, \
    27, 30, 33, 35, 40, 45, 50, 60, 70, 80, 90, 100, 200, 400, 600, 800, 1000]

ITERATIONS = 200

policy = [25, 25, 25, 25]

gamma = 0.7

scale_arr = [100, 200, 10]   #range for reward shaping [(lower, upper, divisor)]

unbiased = False # whether we use canonicaliztion method based on unbiased estimates or not
full_indices = False
MAX_TRAJ_SIZE = 200
# instead of potential shaping, add some form of randomness to shaping
noise_factor = [0]  # scale factor we work with for this functionality

df = pd.DataFrame([], columns = ['grid_size', 'scale', 'relationship', \
    'rollover', 'noise_factor', 'num_trajs', 'original', 'canonical', 'epic', \
    'dard', 'sard', 'D', 'Nm'])

res_df = execute_process([grid_sizes, gamma, policy, rollover_arr, trajs_arr, \
    relationships, ITERATIONS, scale_arr, full_indices, MAX_TRAJ_SIZE, \
    noise_factor, unbiased])

df = pd.concat([df, res_df])

with open('epic_' + rollover_arr[0] + '.pkl', 'wb') as f:
    pickle.dump(df, f)
