import random
import numpy as np
import sys
import pandas as pd
import pickle
import time
import math
import os
from scipy import stats
import multiprocessing

import canonical_utilities as cn

GRIDSIZE = 20
FEAT_TYPE = "original"
NUM_TRAJS = 100    # number of trajectories
LEN_LIMIT = [100, 200]

# get feature map
fmap, _ = cn.compute_featmp(GRIDSIZE, FEAT_TYPE)

POLICIES = [[25, 25, 25, 25], [5, 5, 5, 85], [85, 5, 5, 5],
            [5, 85, 5, 5], [5, 5, 85, 5], [5, 15, 30, 55],
            [55, 30, 15, 5], [15, 5, 55, 30], [5, 55, 30, 15],
            [15, 30, 5, 55]]

# place to store grid trajectories
TRAJ_DIR = "TRAJECTORIES"
if not os.path.isdir(TRAJ_DIR):
    os.mkdir(TRAJ_DIR)

# now store the trajectories from each policy
for policy in POLICIES:
    # generate trajectories
    traj_store = cn.generate_gridworld_trajs(GRIDSIZE, fmap, NUM_TRAJS, policy, LEN_LIMIT)
    print("finished working on ", policy)

    f_name = str(policy).replace("[", '').replace("]", '').replace(", ", '_') + ".pkl"

    # store these trajectories here
    with open(TRAJ_DIR + "/" + f_name, "wb") as f:
        pickle.dump(traj_store, f)
