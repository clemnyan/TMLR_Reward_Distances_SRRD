import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append('irl-imitation')

from sklearn.gaussian_process import GaussianProcessRegressor
from maxent_irl import execute_process_compare
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
relationships = ['polynomial', 'random']
grid_sizes = [8]

#rollover_arr = ['rollover']
rollover_arr = ['random_rollover'] #, 'random_rollover']

#trajs_arr = [1, 2, 3, 4, 5, 6, 7, 8, 8, 10, 12, 13, 15, 17, 19, 20, 22, 25, \
#    27, 30, 33, 35, 40, 45, 50, 60, 70, 80, 90, 100, 200, 400, 600, 800, 1000]
#trajs_arr  = [1, 5, 10, 20, 50, 75, 100]

#trajs_arr = [5, 10, 20, 30, 40, 50, 60, 80, 100]
trajs_arr = [3, 5, 7, 9, 12, 15, 19, 22, 25, 30, 35, 40, 50, 70, 90, 100, 200, 300, 400, 500]

ITERATIONS = 20
#ITERATIONS = 20
#ITERATIONS = 10
policy = [5, 5, 45, 45]
#policy = [25, 25, 25, 25]

gamma = 0.5

scale_arr = [100, 200, 10]   #range for reward shaping [(lower, upper, divisor)]

full_indices = True
MAX_TRAJ_SIZE = 200
# instead of potential shaping, add some form of randomness to shaping
noise_factor = [0]  # scale factor we work with for this functionality

df = pd.DataFrame([], columns = ['grid_size', 'scale', 'relationship', \
    'rollover', 'noise_factor', 'num_trajs', 'original', 'canonical', 'epic1', \
    'dard1', 'srrd1', 'epic2', 'dard2', 'srrd2', 'D', 'Nm'])

res_df = execute_process_compare([grid_sizes, gamma, policy, rollover_arr, \
    trajs_arr, relationships, ITERATIONS, scale_arr, full_indices, \
    MAX_TRAJ_SIZE, noise_factor])

df = pd.concat([df, res_df])

with open('a_epic_compare_' + rollover_arr[0] + '.pkl', 'wb') as f:
    pickle.dump(df, f)
