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
    x_input, y_out, EPOCHS = input_args

    x_input = np.array([np.array(i) for i in x_input])
    y_out = np.array([np.array(i) for i in y_out])

    KERNEL_INITIALIZER, OPTIMIZER = 'he_uniform', 'adamax'
    #ACTIVATION, LOSS = 'selu', 'mse'
    ACTIVATION, LOSS = 'relu', 'categorical_crossentropy'

    input_len, out_len = len(x_input[0]), len(y_out[0])

    #in_LAYERS = [4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1]
    #in_LAYERS = [8,8,8,8,4, 4,4, 4, 4, 4]
    in_LAYERS = [1]

    #in_LAYERS = [30, 30, 30, 30, 20, 20, 20, 20, 20, 20]
    #in_LAYERS = [50, 50, 50, 30, 30, 30, 30, 30, 30, 10, 10, 10, 10, 10, 10]
    #in_LAYERS = [100, 100, 50, 50, 20, 20]
    out_LAYERS = [1]

    optimizer = tf.keras.optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999)

    model = Sequential()
    # Define feed forward model
    model.add(Dense(int(input_len), activation = ACTIVATION,
        kernel_initializer = KERNEL_INITIALIZER, input_dim = input_len))
    for i in in_LAYERS:
        model.add(Dense(int(input_len * i), activation = ACTIVATION))
    for i in out_LAYERS:
        model.add(Dense(int(out_len * i), activation = ACTIVATION))

    model.add(Dense(out_len, activation = 'softmax'))
    #model.add(Dense(out_len, activation = 'sigmoid'))

    model.compile(loss=LOSS, optimizer = optimizer, \
            metrics=[tf.keras.metrics.CategoricalAccuracy()])

    #model.summary()

    TEST_SIZE = 0.4
    X_train, X_test, Y_train, Y_test = train_test_split(x_input, y_out,
        test_size=TEST_SIZE, random_state=49)

    x_train = np.array([np.array(i) for i in X_train])
    y_train = np.array([np.array(i) for i in Y_train])
    x_test = np.array([np.array(i) for i in X_test])
    y_test = np.array([np.array(i) for i in Y_test])

    history = model.fit(x_train, y_train, batch_size = 64,  epochs = EPOCHS,
        validation_data = (x_test, y_test), verbose = 1)

    print(model.predict(x_test))
    #print("finished training")
    v_acc = history.history['val_categorical_accuracy']
    t_loss = history.history['loss']
    t_acc = history.history['categorical_accuracy']
    v_loss = history.history['val_loss']

    return max(v_acc)
