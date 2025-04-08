import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append('irl-imitation')

from maxent_irl import execute_regress
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
relationships = ['polynomial', 'sinusoidal', 'random']
grid_sizes = [10]

#rollover_arr = ['rollover']
rollover_arr = ['rollover']

#trajs_arr = [20] #22, 25, \
#    27, 30, 33, 35, 40, 45, 50, 60, 70, 80, 90, 100, 200, 400, 600, 800, 1000]
#trajs_arr  = [1, 5, 10, 20, 50, 75, 100]

trajs_arr = [1,3,5,7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 33, 37, 40, 45, 50, 60, 65, 70, 75, 80, 85, 90, 95, 100] #, 200, 300, 400, 500]

ITERATIONS = 15
#ITERATIONS = 20
#ITERATIONS = 10
policy = [25, 25, 25, 25]

gamma = 0.7

scale_arr = [50, 100, 10]   #range for reward shaping [(lower, upper, divisor)]

MAX_TRAJ_SIZE = 200
# instead of potential shaping, add some form of randomness to shaping
noise_factor = [0]  # scale factor we work with for this functionality

df = pd.DataFrame([], columns = ['grid_size', 'scale', 'relationship', \
    'rollover', 'noise_factor', 'num_trajs', 'original', 'canonical', \
    'epic', 'dard', 'srrd', "epic_r", "dard_r", "srrd_r", 'D', 'Nm'])


res_df = execute_regress([grid_sizes, gamma, policy, rollover_arr, trajs_arr, \
    relationships, ITERATIONS, scale_arr, MAX_TRAJ_SIZE, noise_factor])

df = pd.concat([df, res_df])

with open('epic6_' + rollover_arr[0] + '.pkl', 'wb') as f:
    pickle.dump(df, f)
