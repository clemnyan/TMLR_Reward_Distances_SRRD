import pandas as pd
import cv2
import sys
import io
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
from utils import *


BASE_SCREEN_DIR = "atari_v1/screens/revenge/"
BASE_TRAJS_DIR = "atari_v1/trajectories/revenge/"


MP_DIR = {}
for skill in ["novice", "medium", "expert"]:
    MP_DIR[skill] = []

count, div = 0, 10
all_pixels = []

for DIR in os.listdir(BASE_SCREEN_DIR):

    trajectory = BASE_TRAJS_DIR + DIR + ".txt"
    img_arr = BASE_SCREEN_DIR + DIR

    with open(trajectory, 'r') as f:
        lines = f.readlines()

    data = pd.read_csv(io.StringIO(''.join(lines[1:])))

    L = len(data)

    score_arr, s_pixels, sp_pixels = [], [], []

    last_score = 0
    for i in range(L//div):

        try:
            s, sp = data.iloc[i*div], data.iloc[(i+1)*div]
            last_score = sp["score"]

            path_s = img_arr + "/" + str(s['frame']) + ".png"
            path_sp = img_arr + "/" + str(sp['frame']) + ".png"

            score_arr.append(sp['score'])

            img_s = cv2.imread(path_s, cv2.IMREAD_GRAYSCALE).flatten()
            img_sp = cv2.imread(path_sp, cv2.IMREAD_GRAYSCALE).flatten()

            x_s, x_sp = img_s.tolist(), img_sp.tolist()

            if np.random.randint(0, 100) <= 20:
                all_pixels.append(x_s)
                all_pixels.append(x_sp)   # could ignore this to save time

            s_pixels.append(x_s)
            sp_pixels.append(x_sp)
        except:
            print("Error encountered")

    print("completed run = ", count)
    count += 1

    if last_score <= 1000:
        MP_DIR["novice"].append([s_pixels, sp_pixels, score_arr, last_score])
    elif last_score <= 2500:
        MP_DIR["medium"].append([s_pixels, sp_pixels, score_arr, last_score])
    else:
        MP_DIR["expert"].append([s_pixels, sp_pixels, score_arr, last_score])

    if count == 300:
        break;

n_components = 2
pca, scaler, var = perform_PCA(all_pixels, n_components)
print("finished perfoming PCA")

REW_MP, state_mp = {}, {}

for d in MP_DIR:
    print("d = ", d)
    count = 0
    for s_arr, sp_arr, r_arr, _ in MP_DIR[d]:
        print("      count = {} out of {}".format(count, len(MP_DIR[d])))
        count += 1
        temp = []
        a1 = pca.fit_transform(scaler.fit_transform(s_arr)).astype(int)
        a2 = pca.fit_transform(scaler.fit_transform(sp_arr)).astype(int)

        for i in range(len(a1)):
            t1, t2 = tuple(a1[i]), tuple(a2[i])
            if t1 not in state_mp:
                state_mp[t1] = len(state_mp)
            if t2 not in state_mp:
                state_mp[t2] = len(state_mp)

            temp.append((state_mp[t1], 1, state_mp[t2], r_arr[i]))

        if d not in REW_MP:
            REW_MP[d] = []

        REW_MP[d].append(temp)

DATA = "DATA"

if not os.path.isdir(DATA):
    os.mkdir(DATA)

with open(DATA + "/rew_state_mp.pkl", "wb") as f:
    pickle.dump([REW_MP, state_mp], f)
