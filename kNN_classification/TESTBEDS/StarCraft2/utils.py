import os
import pickle
import gzip
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
import os
import pandas as pd
import seaborn as sns
import multiprocessing
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from copy import deepcopy
from collections import Counter


def create_XY(rewards_store, class_len):
    """ Create X and Y inputs for the classifier """
    X, Y = [], []
    for g_args in rewards_store:
        r_set, l_set = g_args

        X.append(r_set)
        Y.append(l_set)

    encoding = np.diag(np.ones(class_len))
    Y = [encoding[i] for i in Y]
    return np.array(X), np.array(Y)

def split_qvals (rewards_arr, state_map, action_map):
    """ Represent the reward functions as state features """
    coeff_arr = []
    for rewards in rewards_arr:
        fstore, qstore = [], []
        for rew_triple in rewards:
            triple, t_reward = rew_triple
            s, a, ns = triple
            arr = list(state_map[s]) + [action_map[a]] + list(state_map[ns])
            fstore.append(arr)
            qstore.append(t_reward)
        # fit Linear Models
        reg = LinearRegression().fit(np.array(fstore), np.ravel(qstore))
        coeff_arr.append(reg.coef_)
    return coeff_arr

def process_path (args):
    dir_name, state_mp, action_mp = args

    res_store = []

    count = 0
    for file in os.listdir(dir_name):
        print("file num = ", count)
        count += 1

        #try:
        res = None
        with gzip.open(dir_name + '/' + file, 'rb') as f_name:
            res = pickle.load(f_name)

            res_store.append(res)

    return res_store

def divide_arr (arr_list, factor):
    """ divide the array using the given factor """
    full_len = len(arr_list)
    div_len = int(full_len/factor)
    i = -1

    ans_list = []
    for i in range(div_len):
        ans_list.append(arr_list[factor*i: factor*(i+1)])
    extra = arr_list[factor*(i+1): full_len]
    if (len(extra) != 0):
        ans_list.append(arr_list[factor*(i+1): full_len])
    return ans_list


def unravel_trajs (TRAJ_LOC, p_len, f_name):

    # get state map and action map
    #sa_dir = "state_action_mps.pkl"
    #state_mp, action_mp = None, None

    #with open(sa_dir, 'rb') as f:
    #    state_mp, action_mp = pickle.load(f)
    #    state_mp = {state_mp[i]:i for i in state_mp}
    #    action_mp = {action_mp[i]:i for i in action_mp}

    TRAJ_LOC += "/res_store_folder"
    #First we need to load the files that have been completed here

    dir_name = TRAJ_LOC + '/' + os.listdir(TRAJ_LOC)[p_len]
    print('file dir = {} and processing num = {}'.format(dir_name, p_len))

    res_store = process_path([dir_name, None, None])
    return res_store

def get_class_mp(CLASS_LEN):
    # useful for multilabel learning

    class_arr = ['adept1', 'adept2', 'adept3', 'adept4', \
                'voidray1', 'voidray2', 'voidray3', 'voidray4', \
                'phoenix1', 'phoenix2', 'phoenix3', 'phoenix4', \
                'stalker1', 'stalker2', 'stalker3', 'stalker4']   # 16 classes defined

    class_mp = {}
    for i in range(CLASS_LEN):
        for c in class_arr:
            class_mp[(c, i)] = len(class_mp)

    # class_mp contains: key = class, val = index
    return class_mp


def get_class (class_mp, CLASS_LEN, label, type):
    if type == 'dir':
        label = label.split('res_')[1]
    else:
        label = label.split('win_')[1]

    label_arr = label.split('_')
    class_arr = [0 for i in range(len(class_mp))]

    for i in range(CLASS_LEN):
        i1 = 2*i
        i2 = 2*i+1

        if label_arr[i1] == 'pheonix':  # fix on dir label
            label_arr[i1] = 'phoenix'

        new_lab = label_arr[i1] + label_arr[i2]
        class_arr[class_mp[(new_lab, i)]] = 1

    return class_arr



def compute_stats(res_ans):

    # get the class_mp of all predictions
    label_mp, confusion_mp, acc_mp = {}, {}, {}

    for r in res_ans:
        tup_p = tuple(sorted([i[0] for i in r[0]]))  # prediction
        tup_r = tuple(sorted([i[0] for i in r[1]]))   # true

        if tup_r not in label_mp:
            label_mp[tup_r] = len(label_mp)
            confusion_mp[label_mp[tup_r]] = []
            acc_mp[tup_r] = [0, 0, 0, 0]  # correct, incorrect, u_style, u_type

    print("done here")
    for r in res_ans:
        tup_r = tuple(sorted([i[0] for i in r[1]]))   # true
        tup_p = tuple(sorted([i[0] for i in r[0]]))  # prediction

        cr, cp = Counter(tup_r), Counter(tup_p)
        if cr == cp:  # exact match
            acc_mp[tup_r][0] += 1
            acc_mp[tup_r][2] += 1
            acc_mp[tup_r][3] += 1
            confusion_mp[label_mp[tup_r]].append(label_mp[tup_p])
        else:

            if tup_p in label_mp:
                confusion_mp[label_mp[tup_r]].append(label_mp[tup_p])
            else:
                confusion_mp[label_mp[tup_r]].append(-1)
            acc_mp[tup_r][1] += 1

            # u_style match
            count = 0
            for u in cr:
                if u in cp:
                    count += min(cr[u], cp[u])
            acc_mp[tup_r][2] += (count/float(10))

            # u_type match
            t_r, t_p = [0]*4, [0]*4
            mp = {'stalker':0 ,'voidray':1, 'adept':2, 'phoenix':3}  #keys are indices
            for i in range(len(tup_r)):
                for u in mp:
                    if u in tup_r[i]:
                        t_r[mp[u]] += 1
                    if u in tup_p[i]:
                        t_p[mp[u]] += 1

            count = 0
            for i in range(len(t_r)):
                count += min(t_r[i], t_p[i])
            acc_mp[tup_r][3] += (count/float(10))


            actual, pred = [], []
            for l in confusion_mp:
                for x in confusion_mp[l]:
                    actual.append(l)
                    pred.append(x)
            actual = np.array(actual)
            pred = np.array(pred)

    print("done this part")
    for p in acc_mp:
        acc, wrong, style, un = acc_mp[p]
        sum_v = acc + wrong
        acc_mp[p] = [100*acc/float(sum_v), 100*style/float(sum_v), 100*un/float(sum_v)]

    print("done here too")
    m = classification_report(actual, pred, labels = list(set(actual)))

    arr = np.array(list(acc_mp.values()))
    return m , arr
