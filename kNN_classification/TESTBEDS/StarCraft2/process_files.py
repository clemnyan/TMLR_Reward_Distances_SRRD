import numpy as np
import pandas as pd
import os
import sys
import time
import pickle
import shutil
from copy import deepcopy
import multiprocessing


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


def process_files (dir):

    RAW_DIR = "../../New_SC2_decentralized/DATA/RAW_GAME_DATA/"
    NEW_DIR = 'SC2/PROCESSED_DATA/'

    attack_units = ['Voidray', 'Adept', 'Stalker', 'Phoenix']
    state_mp, f_arr, action_mp, scale_factor = {}, [], {}, 1

    def name_units (x):
        ans = [0, 0, 0]
        for i in x:
            if i == 'background':
                ans[0] += 1/scale_factor
            elif i == 'enemy':
                ans[1] += 1/scale_factor
            elif i == 'attack':
                ans[2] += 1/scale_factor
        return ans

    def expand_units (x):
        return pd.Series([x[0], x[1], x[2]])

    def encode_sa (x):
        s_arr = (x['other'], x['enemy'], x['attack'])
        #s_arr = (x['x_pos'], x['y_pos'], x['other'], x['enemy'], x['attack'])
        a_arr = (x['is_attacking'])

        if s_arr not in state_mp:
            state_mp[s_arr] = len(state_mp)

        if a_arr not in action_mp:
            action_mp[a_arr] = len(action_mp)

        return pd.Series([state_mp[s_arr], action_mp[a_arr]])



    #try:
    a1 = time.time()
    m = os.listdir(RAW_DIR + dir)

    for file in m:
        df = pd.read_csv(RAW_DIR + dir + '/' + file)
        # only encode enemy and ally units
        df = df[(df['is_mine'] == True) | (df['is_enemy'] == True)]
        # set xpos and ypos to 0 if not in attack units (only looking at attack units pos)
        df['x_pos'] = np.where((~(df['name'].isin (attack_units))), 0, df['x_pos'])
        df['y_pos'] = np.where((~(df['name'].isin (attack_units))), 0, df['y_pos'])
        df['name'] = np.where((df['is_enemy'] == True), 'enemy', df['name'])
        df['name'] = np.where((df['is_mine'] == True) & \
                (~(df['name'].isin (attack_units))), 'background', df['name'])
        df['name'] = np.where((df['is_mine'] == True) & \
                (df['name'].isin (attack_units)), 'attack', df['name'])

        # make other columns integers (true or false col)
        df = df.groupby('iteration').agg(list)
        df['is_attacking'] = df['is_attacking'].apply(lambda x: sum([i for i in \
                x if i == True])/scale_factor).astype('int')
        df['x_pos'] = df['x_pos'].apply(lambda x: int(np.mean(x)))
        df['y_pos'] = df['y_pos'].apply(lambda x: int(np.mean(x)))
        df['name'] = df['name'].apply(name_units)
        df[['other', 'enemy', 'attack']] = df['name'].apply(expand_units).astype('int')
        #df[['state', 'action']] = df[['x_pos', 'y_pos', 'is_attacking', \
        #            'other', 'enemy', 'attack']].apply(encode_sa, axis = 1)
        df[['state', 'action']] = df[['is_attacking', 'other', 'enemy', \
                'attack']].apply(encode_sa, axis = 1)


        df = df.drop(columns = ['x_pos', 'y_pos', 'is_attacking', 'other', \
                    'enemy', 'attack', 'name', 'is_mine', 'is_enemy'])

        """
        df = df.drop(columns = ['x_pos', 'y_pos', 'is_attacking', 'other', \
                    'enemy', 'attack', 'name', 'is_mine', 'is_enemy', 'score'])
        """

        df['score'] = df['score'].apply(lambda x: np.sum(x)/1000)
        #df['score'] = df['score'].apply(lambda x: np.sum([i for i in \
        #    x.replace('[', '').replace(']', '').replace(' ', '').split(',')]))

        # create csv of processed data
        second_dir = NEW_DIR + dir + '/'
        if not os.path.isdir(second_dir):
            os.mkdir(second_dir)

        f = 'Trajectory' + file.split('.')[0].split('_')[1]
        f_name = second_dir + f + '.csv.gz'
        if not os.path.isfile(f_name):
            df.to_csv(f_name, compression='gzip')
            f_arr.append(f_name)

    b1 = time.time()
    print("elapsed time = ", b1 - a1)
    return [f_arr, state_mp, action_mp]


def re_write_mp (store):

    new_dir = "SC2/NEW_PROCESSED_FILES/"

    # reset and remake
    if os.path.isdir(new_dir):
        shutil.rmtree(new_dir)

    os.mkdir(new_dir)

    state_mp, action_mp = {}, {}

    for arr in store:
        prev_arr, s_mp, a_mp = arr
        # reverse map
        s_mp = {s_mp[i]:i for i in s_mp}
        a_mp = {a_mp[i]:i for i in a_mp}

        def new_sa_encoding(x):
            original_s, original_a = s_mp[x['state']], a_mp[x['action']]

            if original_s not in state_mp:
                state_mp[original_s] = len(state_mp)
            if original_a not in action_mp:
                action_mp[original_a] = len(action_mp)

            return pd.Series([state_mp[original_s], action_mp[original_a]])

        for prev_dir in prev_arr:
            df = pd.read_csv(prev_dir)
            folder_name, file_name = prev_dir.split('SC2/PROCESSED_DATA/')[1].split('/')
            if not os.path.isdir(new_dir + folder_name):
                os.mkdir(new_dir + folder_name)

            new_name = new_dir + prev_dir.split('SC2/PROCESSED_DATA/')[1]
            df[['state', 'action']] = df[['state', 'action']].apply(new_sa_encoding, axis = 1)
            df.to_csv(new_name, compression = 'gzip')

    return state_mp, action_mp








# if we already have a processed file
PROCESSED_DATA = "SC2/PROCESSED_DATA"
STATE_ACTION_MPS = "state_action_mps.pkl"
RAW_DATA = "../../New_SC2_decentralized/DATA/RAW_GAME_DATA/"

# reset
if os.path.isdir(PROCESSED_DATA):
    shutil.rmtree(PROCESSED_DATA)
if os.path.isfile(STATE_ACTION_MPS):
    os.remove(STATE_ACTION_MPS)

# create files here
os.mkdir(PROCESSED_DATA)

# now do a run
dirs = os.listdir(RAW_DATA)

store = []
#for dir_arr in divide_arr(dirs, len(dirs)//2 + 1):
for dir_arr in divide_arr(dirs, 30):

    res = multiprocessing.Pool().map(process_files, dir_arr)

    for r in res:
        if r != None:
            store.append(r)
            print("len store = ", len(r))


state_mp, action_mp = re_write_mp(store)
# DUMP FILES
print("len states_mp = {} and len action_mp = {}".format(len(state_mp), len(action_mp)))

with open('SC2/state_action_mps.pkl', 'wb') as f:
    pickle.dump([state_mp, action_mp], f)
