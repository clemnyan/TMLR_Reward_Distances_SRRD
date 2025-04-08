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
import bouncing_balls as bb
import canonical_utilities as cn


num_obstacles = 3
grid_size = 20
MIN_DIST = 3
LEN_LIMITS = 350
NUM_TRAJS = 100    # number of trajectories

policies = [[12, 12, 12, 12, 13, 13, 13, 13, 0],
        [5, 5, 25, 25, 25, 5, 5, 5, 0],
        [25, 25, 25, 5, 5, 5, 5, 5, 0],
        [5, 5, 5, 5, 5, 25, 25, 25, 0],
        [5, 5, 65, 5, 5, 5, 5, 5, 0],
        [5, 5, 5, 65, 5, 5, 5, 5, 0],
        [5, 5, 5, 5, 65, 5, 5, 5, 0],
        [5, 25, 5, 25, 5, 25, 5, 5, 0],
        [20, 5, 20, 5, 20, 5, 20, 5, 0],
        [5, 20, 5, 20, 5, 20, 5, 20, 0]]

#policies = [[5, 5, 25, 25, 25, 5, 5, 5, 0]]

# place to store grid trajectories
TRAJ_DIR = "TRAJECTORIES"
if not os.path.isdir(TRAJ_DIR):
    os.mkdir(TRAJ_DIR)

# now store the trajectories from each policy
for policy in policies:

    # create new game
    new_game = bb.Game(num_obstacles, policy, grid_size, MIN_DIST, LEN_LIMITS)

    # generate trajectories
    traj_store = []

    for k in range(NUM_TRAJS):
        T = new_game.play_game()
        traj_store.append(T)

    for t in traj_store:
        print("len = ", len(t))
        
    print("finished working on ", policy)
    f_name = str(policy).replace("[", '').replace("]", '').replace(", ", '_') + ".pkl"
    # store these trajectories here
    with open(TRAJ_DIR + "/" + f_name, "wb") as f:
        pickle.dump(traj_store, f)
