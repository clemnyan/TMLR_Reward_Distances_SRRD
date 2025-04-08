import numpy as np
import pandas as pd
import os
import sys
import time
import pickle
import shutil
from copy import deepcopy
import multiprocessing


DATA, PROCESSED = "../TRAJECTORIES", "NEW_DATA"
state_mp, action_mp = {}, {}

if not os.path.isdir(PROCESSED):
    os.mkdir(PROCESSED)

for f_name in os.listdir(DATA):

    # create folder in PROCESSED
    new_dir = PROCESSED + "/" + f_name.replace(".pkl", "")
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)

    # load files to put in newdir
    with open(DATA + "/" + f_name, "rb") as f:
        trajs = pickle.load(f)

    # put into newdir
    for index, T in enumerate(trajs):
        new_T = []
        for s, a, sp in T:
            if s not in state_mp:
                state_mp[s] = len(state_mp)
            if a not in action_mp:
                action_mp[a] = len(action_mp)
            new_T.append((state_mp[s], action_mp[a]))

        df = pd.DataFrame(new_T, columns = ["state", "action"])
        file_name = new_dir + "/Trajectory_"+str(index)+".csv"
        df.to_csv(file_name)


with open("state_action_mps.pkl", "wb") as f:
    pickle.dump([state_mp, action_mp], f)
