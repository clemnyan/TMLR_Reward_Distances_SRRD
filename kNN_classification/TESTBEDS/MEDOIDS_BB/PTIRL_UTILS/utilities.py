import os
import sys
import numpy as np
import time
import subprocess
import pickle
import shutil
import multiprocessing
import pandas as pd
from itertools import combinations as comb
import gzip
import pickle as cPickle
import keras
from keras.utils import to_categorical
from collections import Counter
import os
import pandas as pd

# need to edit this then make a run
def make_run(TEMP_DIR, dir_mp):
    if os.path.isdir(TEMP_DIR):
        if not os.path.isdir(TEMP_DIR + "/PROGRESS"):
            os.mkdir(TEMP_DIR + "/PROGRESS")

    # LEN_ALL_FILES => length of all files
    LEN_ALL_FILES = len(os.listdir(TEMP_DIR + "/g_args_store"))
    # LEN_PROGRESS => Length of files that are currently or have been processed
    LEN_PROGRESS = len(os.listdir(TEMP_DIR + "/PROGRESS"))

    # if we havent ran all existing files in progress
    if (LEN_PROGRESS < LEN_ALL_FILES):
        print("all = {}, progress = {}".format(LEN_ALL_FILES, LEN_PROGRESS))
        # write pogress check file
        f_name = TEMP_DIR + '/PROGRESS/file_' + str(LEN_PROGRESS) + '.p'

        with open(f_name, "wb") as f:
            pickle.dump('', f)
        # compute PTIRL rewards which are located in in TEMPFILES_res_store_folder
        ans = perform_PTIRL(TEMP_DIR, dir_mp, LEN_PROGRESS)
        return ans
    else:
        print("all files have been accessed")



def perform_PTIRL(TEMP_DIR, dir_mp, filename):
    """
    Compute PT-IRL algorithm to obtain rewards using the combination of
    trajectories that are located in the g_args_store directory.
    """
    # load input arguments
    file_name = TEMP_DIR + '/g_args_store/g_args_' + str(filename) + '.p'
    with open(file_name, "rb") as f:
        g = pickle.load(f)
    print("Total number of samples = ", len(g))
    main_arr = []
    for count, p in enumerate(g):
        a = time.time()
        print("\nWorking on sample number = ", count, "\n")
        p += [dir_mp]
        count += 1

        temp_res = cneigh_classifier(p)

        dir_name_new = TEMP_DIR + "/res_store_folder/res_store_new_" + str(filename)
        if not os.path.isdir(dir_name_new):
            os.mkdir(dir_name_new)

        out_put_name = dir_name_new + "/res_" + str(count) + '_'  + str(time.time()) + " _.pk1.gz"
        with gzip.open(out_put_name, "wb") as f:
            cPickle.dump(temp_res, f)

        print("Time after IRL = ", abs(time.time() - a))
        main_arr.append(temp_res)

    return main_arr

def cneigh_classifier (args):
    """
    In this method, i want to map rewards to the fitted weights, and then \
    map the fitted weights to the original attributes
    """
    alpha = time.time()

    #warnings.filterwarnings("ignore")
    dir1, dir2, dir3, method, div_len, n_process, save_dir, dir_mp = args[0], args[1],\
        args[2], args[3], args[4], args[5], args[6], args[7]

    DATA_DIR = 'NEW_DATA/'
    dir1 = DATA_DIR + dir_mp[dir1]
    dir2 = DATA_DIR + dir_mp[dir2]
    dir3 = DATA_DIR + dir_mp[dir3]

    dirs_1, dirs_2, dirs_3, min_len = get_dir_runs(dir1, dir2, dir3, \
        div_len, save_dir)

    d_stores = [[[dirs_1[i], dirs_2[i], dirs_3[i]], method, save_dir] \
        for i in range(min_len)]
    main_res = []
    args_group = divide_arr(d_stores, n_process) # decide how many multiprocessing runs
    print("len d_stores = {} and len args_group = {}".format(len(d_stores), len(args_group)))
    for i, k_arr in enumerate(args_group):
        print("Running multiprocessing args set = {} ".format(i+1))

        #m = do_Qlearning(k_arr[0])
        #sys.exit()
        Ppool = multiprocessing.Pool()
        results = Ppool.map(do_Qlearning, k_arr)
        Ppool.close()
        Ppool.join()

        for j, rew_mp in enumerate(results):
            main_res.append(rew_mp)

    try:
        print("deleted dir = ", save_dir)
        shutil.rmtree(save_dir)
    except:
        print("error removing dir")

    print("IRL TIME = ", time.time() - alpha)
    return [[main_res, dir1]]   # dir1[1] is the class of attributes


def copy_transfer(args):
    """ Move files from source to dest """
    src, dest = args
    subprocess.check_call(['cp', src, dest])

def get_dir_runs (dir1, dir2, dir3, val_len, save_dir):
    """ Create directory runs for IRL using trajectories stored in dir1 ... dir3 """
    print(os.getcwd())
    print(dir1)
    print(dir2)
    print(dir3)

    # make dir to save files
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    f1, f2, f3  = os.listdir(dir1), os.listdir(dir2), os.listdir(dir3)
    min_len = (min(len(f1), len(f2), len(f3))//val_len) * val_len   # ensure that arrays are multiples of val_len
    num_splits = min_len // val_len

    # split arrays based on val_len
    f1 = np.array_split(f1[:min_len], num_splits)
    f2 = np.array_split(f2[:min_len], num_splits)
    f3 = np.array_split(f3[:min_len], num_splits)

    # create new dirs
    dir_arr1 = ['']*len(f1)
    dir_arr2 = ['']*len(f2)
    dir_arr3 = ['']*len(f3)

    dir_args = []
    for i in range(len(f1)):
        a = time.time()
        pad1 = save_dir + '/dir_cneigh_f1_' + str(time.time()) + '_' + str(i)
        pad2 = save_dir + '/dir_cneigh_f2_' + str(time.time()) + '_' + str(i)
        pad3 = save_dir + '/dir_cneigh_f3_' + str(time.time()) + '_' + str(i)
        # create these folders
        os.mkdir(pad1)
        os.mkdir(pad2)
        os.mkdir(pad3)

        # copy elements into these dirs
        for k in range(len(f1[i])):
            dir_args.append([dir1 + '/' + f1[i][k], pad1])
            dir_args.append([dir2 + '/' + f2[i][k], pad2])
            dir_args.append([dir3 + '/' + f3[i][k], pad3])

        dir_arr1[i] = pad1
        dir_arr2[i] = pad2
        dir_arr3[i] = pad3

    Ppool = multiprocessing.Pool()
    Ppool.map(copy_transfer, dir_args)
    Ppool.close()
    Ppool.join()
    return dir_arr1, dir_arr2, dir_arr3, num_splits


def getParameters(dir_store, method, save_dir):
    dir_name = dir_store[0] + '/irl_res_' + method

    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    # move trajs here!
    # move dir_0 to team directory
    if not os.path.isdir(dir_name + '/team'):
        os.mkdir(dir_name + '/team')

        for f in os.listdir(dir_store[0]):
            if 'Trajectory' in f:
                from_file = dir_store[0] + '/' + f
                z = f.split('.gz')[0]
                df = pd.read_csv(from_file)
                new_file = dir_name + '/team/' +  z
                df.to_csv(new_file)

                to_file = dir_name + '/team'
                name_df2 = to_file + '/actions.csv'
                name_df3 = to_file + '/attributes.csv'

                # make actions.csv and attributes.csv
                df2 = pd.DataFrame([], columns = ['action'])
                df2.to_csv(name_df2, index = False)
                df3 = pd.DataFrame([], columns = ['state'])
                df3.to_csv(name_df3, index = False)

    if not os.path.isdir(dir_name + '/others'):
        os.mkdir(dir_name + '/others')

        for x in range(2):
            for f in os.listdir(dir_store[x + 1]):
                from_file = dir_store[x + 1] + '/' + f

                df = pd.read_csv(from_file)
                rand_val = str(np.random.randint(0, 1000000)) + str(np.random.randint(0, 1000000))
                new_file = dir_name + '/others/' + f.split('.')[0] + rand_val + '.csv'
                df.to_csv(new_file)

    d1 = [dir_store, dir_name, method, save_dir]
    rew_new = perform_irl_run(d1)
    return rew_new

def do_Qlearning (args):
    """ Perform Q-learning """
    dir_store, method, save_dir = args[0], args[1], args[2]
    rew_map = getParameters(dir_store, method, save_dir)
    return rew_map

def perform_irl_run (args):
    """ Make IRL Run to get rewards using the cneighbor method """
    dir_store, dir_name, method, save_dir = args[0], args[1], args[2], args[3]
    # Permissions
    cwd = os.getcwd()
    os.chdir("../../../")

    subprocess.check_call(['chmod', 'u+x', './doc_irl_2.sh'])
    subprocess.check_call(['./doc_irl_2.sh', dir_name+'/team', \
        dir_name+'/others', '1', '0.9', '10'], stdout = open(os.devnull, 'w'), \
        stderr=subprocess.STDOUT, close_fds = True)
    # load irl results from dir_name

    os.chdir(cwd)
    # NEED TO WORK ON THIS PART!!
    rew_new = load_RoMDP(dir_name + '/team')
    return rew_new


def load_RoMDP (dir_name):
    dir_name1 = str(dir_name) + '/saved_results-RoMDP-CPLEX' + \
        '/RoMDP_soln-CPLEX.csv.gz'
    # read states and action map
    dir_name2 = str(dir_name) + '/saved_results-RoMDP-CPLEX' + \
        '/RoMDP_mappings-CPLEX.csv.gz'

    df2 = pd.read_csv(dir_name2, low_memory = False)

    # read the mappings of the dtm
    start_index, start_index1, tick, states_map, actions_map = 0, 0, 0, {}, {}

    for i in range(len(df2)):

        df_val = df2.iloc[i]
        if (df_val[0] == 'State Index'):
            start_index, tick = i, 1
        elif (df_val[0] == 'Action Index'):
            start_index1, tick  = i, 1

        if (tick != 1):
            if (start_index > start_index1): # implies that we are reading  states
                states_map[int(df_val[0])] = int(df_val[1].replace("(", '').\
                    replace(",)", '').replace("'", ''))
            elif (start_index < start_index1):
                actions_map[int(df_val[0])] = int(df_val[1].replace("('", '').\
                    replace(",)", '').replace("'", ''))
        tick = 0

    c = time.time()
    # read the rewards file
    df1 = pd.read_csv(dir_name1, low_memory = False)

    df1_store = []
    for i in range(len(df1)):
        if ('R_[s' in df1.iloc[i][0]):
            triple = df1.iloc[i]
            triple_arr = triple[0].replace(']', '').replace('s_', '').\
                replace('a_', '').split('[')[1].split(',')
            # convert to original trajectory form
            triple_arr = [states_map[int(triple_arr[0])], \
                          actions_map[int(triple_arr[1])], \
                          states_map[int(triple_arr[2])]]
            df1_store.append([triple_arr, triple[1]])
    return df1_store

def set_up_directories (DATA_DIR, TEMP_DIR, save_dir, filter_classes, num_trajs):
    m = os.listdir(DATA_DIR)
    print("OVERALL NUMBER OF CLASSES = ", len(m))
    #sort list of dirs
    m.sort()
    # create map of directories
    dir_mp = {i:m[i] for i in range(len(m))}

    # set up here
    if not os.path.isdir(TEMP_DIR):
        #set up individual files for the PT-IRL algorithm
        setup_PTIRL(num_trajs, filter_classes, DATA_DIR, TEMP_DIR, dir_mp, \
            save_dir)
        # delete the progress check file
        if os.path.isdir(TEMP_DIR + "/PROGRESS"):
            shutil.rmtree(TEMP_DIR + "/PROGRESS")

    return dir_mp


def setup_PTIRL (num_trajs, selected_classes, DATA_DIR, TEMP_DIR, dir_mp, \
        save_dir):
    """
    Set up folders to perform PTIRL
    To do so we need:
    1) /g_args_store - directory to store all inputs and parameters for irl
    2) ./res_store - will have results that are obtained
    """
    # parameters
    bot_classes, method, n_process = os.listdir(DATA_DIR), 'new', 20
    other_dirs = [list(i) for i in comb(selected_classes, 2)]

    # organize files in the form of target and other trajectories
    p = []
    for i in dir_mp:
        for j in other_dirs:
            p.append([i] + j)

    print('NUMBER OF COMBINATION DIRS = ', len(p))
    g = []
    for i in range(len(p)):
        a1, b1 = str(np.random.randint(1, 1000000)), \
            str(np.random.randint(1, 1000000))

        save_dir_label = save_dir + str(time.time()) + '_' + a1 + '_' +  b1 + \
            '_' + str(i)

        g.append([p[i][0], p[i][1], p[i][2], method, num_trajs, n_process, \
            save_dir_label])

    if not os.path.isdir(TEMP_DIR):
        os.mkdir(TEMP_DIR)
    if os.path.isdir(TEMP_DIR + '/g_args_store'):
        subprocess.check_call(['rm', '-rf', TEMP_DIR + '/g_args_store'])
    if os.path.isdir(TEMP_DIR + '/res_store_folder'):
        subprocess.check_call(['rm', '-rf', TEMP_DIR + '/res_store_folder'])

    # make new directory
    os.mkdir(TEMP_DIR + '/g_args_store')
    os.mkdir(TEMP_DIR + '/res_store_folder')

    # now make the g_args files
    g_arr = divide_arr(g, 20)

    for i in range(len(g_arr)):
        file_name = TEMP_DIR + '/g_args_store/g_args_' + str(i) + '.p'
        pickle.dump(g_arr[i], open(file_name, "wb"))


def gen_seq_data (num_segments):
    DATA_DIR = '../DATA/NEW_PROCESSED_FILES'

    with open('../DATA/state_action_mps.pkl', 'rb') as f:
        state_mp, action_mp = pickle.load(f)

    state_mp = {state_mp[i]:i for i in state_mp}
    action_mp = {action_mp[i]:i for i in action_mp}
    s_len, a_len = len(state_mp[0]), len(action_mp)

    m = os.listdir(DATA_DIR)
    print("OVERALL NUMBER OF CLASSES = ", len(m))
    #sort list of dirs
    m.sort()
    # create map of directories
    dir_mp = {i:m[i] for i in range(len(m))}
    trans_store, label_store  = [], []

    for iter, dir in enumerate(m):
        dirs  = os.listdir(DATA_DIR + '/' + dir)
        for d in dirs:   # csv file is d
            pd_file = pd.read_csv(DATA_DIR + '/' + dir + '/' + d)
            pd_arr = np.array_split(pd_file, num_segments)

            f_store = []
            for p_df in pd_arr:   # n segments of the csv file

                c_s, c_a = Counter(p_df['state']), Counter(p_df['action'])
                s_arr, a_arr = np.zeros(s_len), np.zeros(a_len)

                for x in c_s:
                    s_arr += c_s[x]*np.array(state_mp[x])
                for x in c_a:
                    a_arr[action_mp[x]] = c_a[x]

                n_arr = list(s_arr) + list(a_arr)
                f_store.append(n_arr)

            trans_store.append(np.array(f_store))   # this is the csv level
            label_store.append(iter)

    label_store =  to_categorical(label_store, num_classes=len(dir_mp))
    return trans_store, label_store, dir_mp

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
