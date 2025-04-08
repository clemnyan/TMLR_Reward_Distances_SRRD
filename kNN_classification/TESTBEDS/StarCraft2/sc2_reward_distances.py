import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append('../irl-imitation')

from maxent_irl import canonicalize_epic, local_epic
import pandas as pd
import numpy as np
from copy import deepcopy
import pickle
import time
from scipy import stats
import multiprocessing
from numpy.random import randn, seed
from utilities import *

def init_pool_processes():
    seed()

REW_PATH = "SC2/ptirl_rewards.pkl"

# extract rewards
with open(REW_PATH, 'rb') as f:
    rew_mp = pickle.load(f)

arr_set = re_write_rewards(rew_mp)
res_common, res_all = {}, {}

for i, args in enumerate(arr_set):
    print("finished running arg number = {}/{}".format(i, len(arr_set)))
    with multiprocessing.Pool() as pool:
        result = pool.map(sc2_distance_trial, args)
        pool.close()
        pool.join()

        for r in result:
            r_label, r_common, r_all = r
            if r_label not in res_common:
                res_common[r_label] = [list(r_common)]
                res_all[r_label] = [list(r_all)]
            else:
                res_common[r_label].append(list(r_common))
                res_all[r_label].append(list(r_all))
