import os
import pickle
import gzip
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
import tensorflow as tf
import keras
import os
import pandas as pd
import seaborn as sns
import multiprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from copy import deepcopy
from tensorflow.python.ops import math_ops
from tensorflow.keras import backend as K
from collections import Counter




def feed_forward_classifier (input_args):
    """ Train classifer to identify attributes from feature weights """
    x_input, y_out, TEMP_DIR = input_args

    def customLoss (y_true, y_pred):

        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
        tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
        fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
        fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

        p = tp / (tp + fp + K.epsilon())
        r = tp / (tp + fn + K.epsilon())

        f1 = 2*p*r / (p+r+K.epsilon())
        f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
        return 1 - K.mean(f1)


    def w_BLoss(target, output):
        """Calculate weighted binary crossentropy between an output tensor and a target tensor.
        # Arguments
            target: A tensor with the same shape as `output`.
            output: A tensor.
            from_logits: Whether `output` is expected to be a logits tensor.
                By default, we consider that `output`
                encodes a probability distribution.
        # Returns
            A tensor.
        """

        output = tf.convert_to_tensor(output)
        target = tf.cast(target, output.dtype)

        _epsilon = tf.convert_to_tensor(K.epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.math.log(output / (1 - output))

        return tf.nn.weighted_cross_entropy_with_logits(target, output, 2)


    KERNEL_INITIALIZER, OPTIMIZER = 'he_uniform', 'adamax'
    #ACTIVATION, LOSS = 'selu', 'mse'
    ACTIVATION, LOSS = 'selu', 'binary_crossentropy'

    input_len, out_len = len(x_input[0]), len(y_out[0])

    #in_LAYERS = [4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1]
    #in_LAYERS = [2, 2, 2, 1, 1]
    #in_LAYERS = [2]

    in_LAYERS = [30, 30, 30, 30, 20, 20, 20, 20, 20, 20]
    #in_LAYERS = [50, 50, 50, 30, 30, 30, 30, 30, 30, 10, 10, 10, 10, 10, 10]
    #in_LAYERS = [100, 100, 50, 50, 20, 20]

    out_LAYERS = [1]

    optimizer = tf.keras.optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999)
    DROPOUT = 0.1

    model = Sequential()
    # Define feed forward model
    model.add(Dense(int(input_len), activation = ACTIVATION,
        kernel_initializer = KERNEL_INITIALIZER, input_dim = input_len))
    for i in in_LAYERS:
        model.add(Dense(int(input_len * i), activation = ACTIVATION))
        model.add(Dropout(DROPOUT))
    for i in out_LAYERS:
        model.add(Dense(int(out_len * i), activation = ACTIVATION))
        model.add(Dropout(DROPOUT))

    #model.add(Dense(out_len, activation = 'softmax'))
    model.add(Dense(out_len, activation = 'sigmoid'))

    # compile model
    #model.compile(loss = LOSS, optimizer = optimizer, metrics = ['accuracy'])
    #model.compile(loss = LOSS, optimizer = optimizer, \
    #        metrics=[tf.keras.metrics.BinaryAccuracy()])

    #model.compile(loss = customLoss, optimizer = optimizer, \
    #        metrics=[tf.keras.metrics.BinaryAccuracy()])

    model.compile(loss = w_BLoss, optimizer = optimizer, \
            metrics=[tf.keras.metrics.BinaryAccuracy()])

    model.summary()

    TEST_SIZE = 0.4
    X_train, X_test, Y_train, Y_test = train_test_split(x_input, y_out,
        test_size=TEST_SIZE, random_state=49)

    x_train = np.array([np.array(i) for i in X_train])
    y_train = np.array([np.array(i) for i in Y_train])
    x_test = np.array([np.array(i) for i in X_test])
    y_test = np.array([np.array(i) for i in Y_test])

    print("part b")
    history = model.fit(x_train, y_train, batch_size = 64,  epochs = 400,
        validation_data = (x_test, y_test))

    print("finished training")
    v_acc = history.history['val_binary_accuracy']
    t_loss = history.history['loss']
    t_acc = history.history['binary_accuracy']
    v_loss = history.history['val_loss']

    plt.rc('font', size=15)          # controls default text sizes
    plt.rc('axes', titlesize=15)     # fontsize of the axes title
    plt.rc('axes', labelsize=15)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
    plt.rc('legend', fontsize=15)    # legend fontsize
    plt.rc('figure', titlesize=15)   # fontsize of the figure title

    fig, axes = plt.subplots(1, 2, figsize = (14, 5))
    plt.subplots_adjust(hspace=0.7)
    axes[0].plot(t_loss, '-*', color = 'blue', label = 'training')
    axes[0].plot(v_loss, '-*', color = 'red', label = 'validation')
    axes[0].set_xlabel('iteration')
    axes[0].set_ylabel('loss')
    axes[0].set_title("classifier loss")
    axes[0].legend(loc="best", ncol=3, prop={'size': 12})

    axes[1].plot(t_acc, '-*', color = 'blue', label = 'training')
    axes[1].plot(v_acc, '-*', color = 'red', label = 'validation')
    axes[1].set_xlabel('iteration')
    axes[1].set_ylabel('accuracy')
    axes[1].set_title("classifier acccuracy")
    axes[1].legend(loc="best", ncol=3, prop={'size': 12})

    if not (os.path.isdir(TEMP_DIR + '/network_results')):
        os.mkdir(TEMP_DIR + '/network_results')
    if not (os.path.isdir(TEMP_DIR + '/my_models')):
        os.mkdir(TEMP_DIR + '/my_models')
    if not (os.path.isdir(TEMP_DIR + '/confusion_matrices')):
        os.mkdir(TEMP_DIR + '/confusion_matrices')

    rand_val = str(np.random.randint(1, 1000))

    filename = str([KERNEL_INITIALIZER, ACTIVATION, in_LAYERS, out_LAYERS, \
        OPTIMIZER, TEST_SIZE, LOSS, rand_val])

    plt.savefig(TEMP_DIR + '/network_results/'+ filename + '_.png')

    model_name = TEMP_DIR + '/my_models/feed_forward_model_' + rand_val + '_.h5'
    model.save(model_name)


    # train x_input on the model
    #results, res_arr = [], []
    results = []
    pred = model.predict(x_test)


    CLASS_LEN = 10

    for i in range(len(pred)):
        ans = []

        for j in range(CLASS_LEN): # 10 is the number of classes
            max_arg = np.argmax(pred[i][16*j: 16*(j + 1)])
            temp = [0]*16
            temp[max_arg] = 1
            ans += temp

        results.append([np.array(ans), y_test[i]])

    """
    results = np.array(results)
    print("\nModel details\n")
    print(filename)
    print(model_name)

    # plot confusion matrices
    df1 = getConfusion(results)
    # clear figure and plot the heatmap

    plt.clf()
    fig, axes = plt.subplots(figsize = (30, 30))
    m = sns.heatmap(df1, annot=True, fmt="d", cmap = 'YlOrBr')

    #plt.title('Prediction Accuracies')
    plt.ylabel('predicted class')
    plt.xlabel('actual class')
    plt.savefig(TEMP_DIR + '/confusion_matrices/'+ filename + '_.jpg')

    count = 0
    for i in range(len(df1)):
        vec = df1.iloc[i]
        if (i == count):
            ratio = vec[count]/sum(vec)
            print(i, "ratio = ", ratio)
        count += 1
    #return results, res_arr, history, model
    print("")
    #print(df1)
    """
    print("")
    print("length of training set = ", len(X_train))
    print("length of testing set = ", len(X_test))
    return results, model_name, model


def getConfusion (arr):
    """ Get confusion matrix for the predictions that where made """
    arr_size = len(arr[0][0][0])
    pred_map = {i:np.array([0 for k in range(arr_size)]) for i in range(arr_size)}

    for m in arr:
        orig_index, pred_index = np.argmax(m[1][0]), np.argmax(m[0][0])
        mod_map = pred_map[orig_index]
        mod_map[pred_index] = mod_map[pred_index] + 1

    return pd.DataFrame([pred_map[i] for i in range(arr_size)])


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

def split_qvals (rewards_arr, state_map, action_map, label, rewards_store):
    """ Represent the reward functions as state features """

    all_stores = []
    for rewards in rewards_arr:
        store = []
        for rew_triple in rewards:
            triple, t_reward = rew_triple
            s, a, ns = triple
            store.append((state_map[s], action_map[a], state_map[ns], t_reward))

        rewards_store.append([store, label])
    return all_stores


def process_path (args):
    dir_name, state_mp, action_mp, rewards_store = args

    count = 0
    for file in os.listdir(dir_name):
        print("file num = ", count)
        count += 1

        #try:
        res = None
        with gzip.open(dir_name + '/' + file, 'rb') as f_name:
            res = pickle.load(f_name)
            res_label = res[0][1]
            rew = split_qvals(res[0][0], state_mp, action_mp, res_label, rewards_store)


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


def unravel_trajs (TRAJ_LOC, p_len, f_name, rewards_store):

    # get state map and action map
    sa_dir = "state_action_mps.pkl"
    state_mp, action_mp = None, None

    with open(sa_dir, 'rb') as f:
        state_mp, action_mp = pickle.load(f)
        state_mp = {state_mp[i]:i for i in state_mp}
        action_mp = {action_mp[i]:i for i in action_mp}

    TRAJ_LOC += "/res_store_folder"
    #First we need to load the files that have been completed here

    dir_name = TRAJ_LOC + '/' + os.listdir(TRAJ_LOC)[p_len]
    print('file dir = {} and processing num = {}'.format(dir_name, p_len))

    process_path([dir_name, state_mp, action_mp, rewards_store])

    with open(f_name, 'wb') as f:
        pickle.dump([], f)


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
