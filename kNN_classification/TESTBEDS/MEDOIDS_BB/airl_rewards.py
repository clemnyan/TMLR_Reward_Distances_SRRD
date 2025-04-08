import pickle
import os
import numpy as np
import canonical_utilities as cn
import train_classifier as tc
from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
import random
import sys
import pandas as pd
import time


method = "airl"

grid_size, gamma = 20, 0.05


store_orig, store_epic, store_dard, store_sard, reward_labels = \
    cn.airl_canonicalization(grid_size, gamma)


df_labels = ['orig_acc', 'orig_prec', 'orig_rec', 'orig_f1', \
        'epic_acc', 'epic_prec', 'epic_rec', 'epic_f1', \
        'dard_acc', 'dard_prec', 'dard_rec', 'dard_f1', \
        'sard_acc', 'sard_prec', 'sard_rec', 'sard_f1']

results = []

for k in range(100):
    t1 = time.time()
    print("im here boss")
    sample_points = list(np.arange(len(reward_labels)))
    random.shuffle(sample_points)
    sample_points = sample_points[:100]   # this helps to handle large data

    orig = cn.getcounts(store_orig, reward_labels, sample_points)
    epic = cn.getcounts(store_epic, reward_labels, sample_points)
    dard = cn.getcounts(store_dard, reward_labels, sample_points)
    sard = cn.getcounts(store_sard, reward_labels, sample_points)

    res = orig + epic + dard + sard

    results.append(res)
    print("orig")
    print(orig)
    print("epic")
    print(epic)
    print("dard")
    print(dard)
    print("sard")
    print(sard)
    print("")
    print("total time = ", time.time() - t1)

df = pd.DataFrame(results, columns = df_labels)
