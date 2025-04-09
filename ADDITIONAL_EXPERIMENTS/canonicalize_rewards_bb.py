import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append('irl-imitation')

from maxent_irl import execute_process_bb
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
#relationships = ['linear', 'polynomial', 'sinusoidal', 'random']
relationships = ['polynomial', 'sinusoidal', 'random']
grid_sizes = [15]

#rollover_arr = ['rollover']
rollover_arr = ['rollover', 'random_rollover']

#trajs_arr = [1, 2, 3, 4, 5, 6, 7, 8, 8, 10, 12, 13, 15, 17, 19, 20, 22, 25, \
#    27, 30, 33, 35, 40, 45, 50, 60, 70, 80, 90, 100, 200, 400, 600, 800, 1000]
#trajs_arr  = [1, 5, 10, 20, 50, 75, 100]

#trajs_arr = [5, 10, 20, 30, 40, 50, 60, 80, 100] #, 150, 200, 250, 300]
trajs_arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 19, 20, 22, 25, \
    27, 30, 33, 35, 40, 45, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]

ITERATIONS = 30
#ITERATIONS = 20
#ITERATIONS = 10
#policy = [13, 12, 12, 12, 12, 13, 13, 13]
policy = [10, 10, 15, 20, 15, 10, 10, 10]
gamma = 0.8

scale_arr = [100, 200, 10]   #range for reward shaping [(lower, upper, divisor)]

unbiased = True # whether we use canonicaliztion method based on unbiased estimates or not
full_indices = False
MIN_DIST = 2
LEN_LIMIT = 200
# instead of potential shaping, add some form of randomness to shaping
noise_factor, random_scale, induce_random = [0], [0], False  # scale factor we work with for this functionality

df = pd.DataFrame([], columns = ['grid_size', 'scale', 'relationship', \
    'rollover', 'num_trajs', 'original', 'canonical', 'epic', 'dard', 'sard', \
    'D', 'Nm'])

res_df = execute_process_bb([grid_sizes, gamma, policy, rollover_arr, \
    trajs_arr, relationships, ITERATIONS, scale_arr, full_indices, \
    MIN_DIST, LEN_LIMIT, induce_random, random_scale, unbiased])

df = pd.concat([df, res_df])

with open('epic_b1_' + rollover_arr[0] + '.pkl', 'wb') as f:
    pickle.dump(df, f)
