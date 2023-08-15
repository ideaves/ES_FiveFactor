# multivariate multi-step encoder-decoder lstm example
import tensorflow as tf
import pandas as pd
import time

from pathlib import Path

from numpy import array
from numpy import ndarray
from numpy import hstack
from numpy import mean
from numpy import var
from numpy import exp
from numpy import log
from numpy import std
from numpy import reshape

from numpy import transpose
import numpy as np
import scipy
from scipy import stats

from datetime import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tensorflow import keras
from keras.models import Sequential
from keras.models import clone_model
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras import backend as K

import matplotlib.pyplot as plt
import winsound

DATASET_MAX = 30000
EVAL_SIZE = 5000
EVAL_OVERLAP = 1000
NUM_TS_LAGS = 12
LOSS_MAX = 3.0
MODEL_POOL_SIZE = 5
SUSPECT_SOLUTION_BOUNDARY = 2.0
timeofday = datetime.now()
MAX_OOS_RSQUARED = -3.0
MIN_OOS_RSQUARED = 0.0
COUNTS_AS_SIGNIFICANT = 0.8
CLAMP_PEAK_OBJECTIVE = 0.7
MIN_MODEL_MATURITY_TO_REPLACE = 5
MIN_MODEL_MATURITY_TO_REPRODUCE = 10

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
LOWER_ENV_ADAPTATION_SPEED = 1.0 / 10.0
max_out_of_sample_rsquared = None
previous_best_model = None
previous_MAX_OOS_RSQUARED = None
model_was_confirmed = False

SUPERVISORY_AVERAGE_LEAD = 16  # Geometric mean future price objective number of periods forward

TIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
DATE_FORMAT = '%Y-%m-%d'

prediction_history = []


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def prepare_objective(sequence, timestamps):
    leading = sequence[len(sequence) - 1]
    i = 0
    objective = []
    nexttime = datetime.strptime(timestamps[len(timestamps) - 1], '%Y-%m-%d %H:%M:%S')
    # with open('C:\\Users\\ideav\\Documents\\PythonWork\\FiveFactor\\sequence.csv','w') as f:
    for price in reversed(sequence):
        thistime = datetime.strptime(timestamps[len(timestamps) - i - 1], '%Y-%m-%d %H:%M:%S')
        numperiods = max(1, int((nexttime - thistime).seconds / 5 / 60))
        objective.append((leading - price) / price * 10000)
        for j in range(numperiods):
            leading = ((SUPERVISORY_AVERAGE_LEAD - 1) / SUPERVISORY_AVERAGE_LEAD) * leading + (
                        1 / SUPERVISORY_AVERAGE_LEAD) * price
            # f.write('\n{0}, {1}, {2}'.format(thistime, price, leading))
        nexttime = thistime
        i += 1
    return list(reversed(objective))


def bound_objective(sequence, alt_sequence, proportion):
    # find the min and max. if both have the same sign, print message and return unmodified
    mymin = min(sequence)
    mymax = max(sequence)
    if mymin * mymax > 0:
        print("Sequence is all the same sign, ignoring bounding for this sequence")
        return sequence
    # find whichever is the smallest abs value, and use that to bound the whole sequence
    myabsbound = min(-mymin, mymax)
    pos = 0
    for element in sequence:
        if element > 0 and element > myabsbound * proportion:
            sequence[pos] = myabsbound * proportion
        elif element < 0 and element < -myabsbound * proportion:
            sequence[pos] = -myabsbound * proportion
        pos += 1
    pos = 0
    for element in alt_sequence:
        if element > 0 and element > myabsbound * proportion:
            alt_sequence[pos] = myabsbound * proportion
        elif element < 0 and element < -myabsbound * proportion:
            alt_sequence[pos] = -myabsbound * proportion
        pos += 1
#    with open('C:\\Users\\ideav\\Documents\\PythonWork\\FiveFactor\\sequence.csv', 'w') as f:
#        f.write('\n{0}'.format(sequence))
    return sequence


def divide_input_data_into_three_parts(all_data, objective):
    stub_ES = all_data["ES"].copy()[len(all_data["ES"]) - NUM_TS_LAGS:]
    eval_ES = all_data["ES"].copy()[len(all_data["ES"]) - NUM_TS_LAGS - EVAL_SIZE - EVAL_OVERLAP - 1:len(
        all_data["ES"]) - NUM_TS_LAGS - 1]
    in_ES = all_data["ES"][len(all_data["ES"]) - DATASET_MAX - NUM_TS_LAGS - EVAL_SIZE - 2:len(
        all_data["ES"]) - NUM_TS_LAGS - EVAL_SIZE - 2]
    mean_in_ES = mean(all_data["ES"])
    std_in_ES = std(all_data["ES"])

    stub_ESS = all_data["ESS"].copy()[len(all_data["ESS"]) - NUM_TS_LAGS:]
    eval_ESS = all_data["ESS"].copy()[len(all_data["ESS"]) - NUM_TS_LAGS - EVAL_SIZE - EVAL_OVERLAP - 1:len(
        all_data["ESS"]) - NUM_TS_LAGS - 1]
    in_ESS = all_data["ESS"][len(all_data["ESS"]) - DATASET_MAX - NUM_TS_LAGS - EVAL_SIZE - 2:len(
        all_data["ESS"]) - NUM_TS_LAGS - EVAL_SIZE - 2]
    mean_in_ESS = mean(all_data["ESS"])
    std_in_ESS = std(all_data["ESS"])

    stub_EC = all_data["EC"].copy()[len(all_data["EC"]) - NUM_TS_LAGS:]
    eval_EC = all_data["EC"].copy()[len(all_data["EC"]) - NUM_TS_LAGS - EVAL_SIZE - EVAL_OVERLAP - 1:len(
        all_data["EC"]) - NUM_TS_LAGS - 1]
    in_EC = all_data["EC"][len(all_data["EC"]) - DATASET_MAX - NUM_TS_LAGS - EVAL_SIZE - 2:len(
        all_data["EC"]) - NUM_TS_LAGS - EVAL_SIZE - 2]
    mean_in_EC = mean(all_data["EC"])
    std_in_EC = std(all_data["EC"])

    stub_ECS = all_data["ECS"].copy()[len(all_data["ECS"]) - NUM_TS_LAGS:]
    eval_ECS = all_data["ECS"].copy()[len(all_data["ECS"]) - NUM_TS_LAGS - EVAL_SIZE - EVAL_OVERLAP - 1:len(
        all_data["ECS"]) - NUM_TS_LAGS - 1]
    in_ECS = all_data["ECS"][len(all_data["ECS"]) - DATASET_MAX - NUM_TS_LAGS - EVAL_SIZE - 2:len(
        all_data["ECS"]) - NUM_TS_LAGS - EVAL_SIZE - 2]
    mean_in_ECS = mean(all_data["ECS"])
    std_in_ECS = std(all_data["ECS"])

    ####### Keep this up in the main loop, it's for display
    GC_last_price = all_data["GC"][len(all_data["GC"]) - 1]
    stub_GC = all_data["GC"].copy()[len(all_data["GC"]) - NUM_TS_LAGS:]
    eval_GC = all_data["GC"].copy()[len(all_data["GC"]) - NUM_TS_LAGS - EVAL_SIZE - EVAL_OVERLAP - 1:len(
        all_data["GC"]) - NUM_TS_LAGS - 1]
    in_GC = all_data["GC"][len(all_data["GC"]) - DATASET_MAX - NUM_TS_LAGS - EVAL_SIZE - 2:len(
        all_data["GC"]) - NUM_TS_LAGS - EVAL_SIZE - 2]
    mean_in_GC = mean(all_data["GC"])
    std_in_GC = std(all_data["GC"])

    stub_GCS = all_data["GCS"].copy()[len(all_data["GCS"]) - NUM_TS_LAGS:]
    eval_GCS = all_data["GCS"].copy()[len(all_data["GCS"]) - NUM_TS_LAGS - EVAL_SIZE - EVAL_OVERLAP - 1:len(
        all_data["GCS"]) - NUM_TS_LAGS - 1]
    in_GCS = all_data["GCS"][len(all_data["GCS"]) - DATASET_MAX - NUM_TS_LAGS - EVAL_SIZE - 2:len(
        all_data["GCS"]) - NUM_TS_LAGS - EVAL_SIZE - 2]
    mean_in_GCS = mean(all_data["GCS"])
    std_in_GCS = std(all_data["GCS"])

    stub_US = all_data["US"].copy()[len(all_data["US"]) - NUM_TS_LAGS:]
    eval_US = all_data["US"].copy()[len(all_data["US"]) - NUM_TS_LAGS - EVAL_SIZE - EVAL_OVERLAP - 1:len(
        all_data["US"]) - NUM_TS_LAGS - 1]
    in_US = all_data["US"][len(all_data["US"]) - DATASET_MAX - NUM_TS_LAGS - EVAL_SIZE - 2:len(
        all_data["US"]) - NUM_TS_LAGS - EVAL_SIZE - 2]
    mean_in_US = mean(all_data["US"])
    std_in_US = std(all_data["US"])

    stub_USS = all_data["USS"].copy()[len(all_data["USS"]) - NUM_TS_LAGS:]
    eval_USS = all_data["USS"].copy()[len(all_data["USS"]) - NUM_TS_LAGS - EVAL_SIZE - EVAL_OVERLAP - 1:len(
        all_data["USS"]) - NUM_TS_LAGS - 1]
    in_USS = all_data["USS"][len(all_data["USS"]) - DATASET_MAX - NUM_TS_LAGS - EVAL_SIZE - 2:len(
        all_data["USS"]) - NUM_TS_LAGS - EVAL_SIZE - 2]
    mean_in_USS = mean(all_data["USS"])
    std_in_USS = std(all_data["USS"])

    stub_BC = all_data["BC"].copy()[len(all_data["BC"]) - NUM_TS_LAGS:]
    eval_BC = all_data["BC"].copy()[len(all_data["BC"]) - NUM_TS_LAGS - EVAL_SIZE - EVAL_OVERLAP - 1:len(
        all_data["BC"]) - NUM_TS_LAGS - 1]
    in_BC = all_data["BC"][len(all_data["BC"]) - DATASET_MAX - NUM_TS_LAGS - EVAL_SIZE - 2:len(
        all_data["BC"]) - NUM_TS_LAGS - EVAL_SIZE - 2]
    mean_in_BC = mean(all_data["BC"])
    std_in_BC = std(all_data["BC"])

    ###### Snip the time index feature to the learning set
    in_timeidx = all_data["timeidx"][len(all_data[objective]) - DATASET_MAX - NUM_TS_LAGS:]

    train_data = {"ES": in_ES, "ESS": in_ESS, "EC": in_EC, "ECS": in_ECS, "GC": in_GC, "GCS": in_GCS, "US": in_US, "USS": in_USS, "BC": in_BC}
    eval_data = {"ES": eval_ES, "ESS": eval_ESS, "EC": eval_EC, "ECS": eval_ECS, "GC": eval_GC, "GCS": eval_GCS, "US": eval_US, "USS": eval_USS, "BC": eval_BC}
    front_stub_data = {"ES": stub_ES, "ESS": stub_ESS, "EC": stub_EC, "ECS": stub_ECS, "GC": stub_GC, "GCS": stub_GCS, "US": stub_US, "USS": stub_USS, "BC": stub_BC}
    train_means = {"ES": mean_in_ES, "ESS": mean_in_ESS, "EC": mean_in_EC, "ECS": mean_in_ECS, "GC": mean_in_GC, "GCS": mean_in_GCS, "US": mean_in_US, "USS": mean_in_USS, "BC": mean_in_BC}
    train_stds = {"ES": std_in_ES, "ESS": std_in_ESS, "EC": std_in_EC, "ECS": std_in_ECS, "GC": std_in_GC, "GCS": std_in_GCS, "US": std_in_US, "USS": std_in_USS, "BC": std_in_BC}

    return [train_data, eval_data, front_stub_data, train_means, train_stds, in_timeidx]


def divide_output_data_into_two_parts(out_GC):
    ###### De-mean output series
    avg_out_GC = mean(out_GC)
    out_GC[:] = [number - avg_out_GC for number in out_GC]

    out_eval_GC = [number - avg_out_GC for number in out_GC]

    out_GC = bound_objective(out_GC, out_eval_GC, CLAMP_PEAK_OBJECTIVE)

    # This removes the oldest observations from out_GC
    out_GC = out_GC[len(out_GC) - DATASET_MAX - NUM_TS_LAGS - EVAL_SIZE - 2:len(out_GC) - NUM_TS_LAGS - EVAL_SIZE - 2]
    
    out_eval_GC = out_GC.copy()[len(out_GC) - DATASET_MAX - NUM_TS_LAGS - EVAL_SIZE - EVAL_OVERLAP - 1:len(out_GC) - DATASET_MAX - NUM_TS_LAGS - 1]
    
    return [out_GC, out_eval_GC]


def gauss_normalize_input_data(train_means, train_stds, train_data, eval_data, front_stub_data):
    ###### Normalize the input training series and the most current observation.
    # de-mean and standardize inputs variance, i.e. so that (0,1)
    train_data["ES"] = [(x - train_means["ES"]) / train_stds["ES"] for x in train_data["ES"]]
    train_data["EC"] = [(x - train_means["EC"]) / train_stds["EC"] for x in train_data["EC"]]
    train_data["GC"] = [(x - train_means["GC"]) / train_stds["GC"] for x in train_data["GC"]]
    train_data["US"] = [(x - train_means["US"]) / train_stds["US"] for x in train_data["US"]]
    train_data["BC"] = [(x - train_means["BC"]) / train_stds["BC"] for x in train_data["BC"]]
    front_stub_data["ES"] = [(x - train_means["ES"]) / train_stds["ES"] for x in
                                  front_stub_data["ES"]]
    front_stub_data["EC"] = [(x - train_means["EC"]) / train_stds["EC"] for x in
                                  front_stub_data["EC"]]
    front_stub_data["GC"] = [(x - train_means["GC"]) / train_stds["GC"] for x in
                                  front_stub_data["GC"]]
    front_stub_data["US"] = [(x - train_means["US"]) / train_stds["US"] for x in
                                  front_stub_data["US"]]
    front_stub_data["BC"] = [(x - train_means["BC"]) / train_stds["BC"] for x in
                                  front_stub_data["BC"]]
    train_data["ESS"] = [(x - train_means["ESS"]) / train_stds["ESS"] for x in train_data["ESS"]]
    train_data["ECS"] = [(x - train_means["ECS"]) / train_stds["ECS"] for x in train_data["ECS"]]
    train_data["GCS"] = [(x - train_means["GCS"]) / train_stds["GCS"] for x in train_data["GCS"]]
    train_data["USS"] = [(x - train_means["USS"]) / train_stds["USS"] for x in train_data["USS"]]
    front_stub_data["ESS"] = [(x - train_means["ESS"]) / train_stds["ESS"] for x in
                                   front_stub_data["ESS"]]
    front_stub_data["ECS"] = [(x - train_means["ECS"]) / train_stds["ECS"] for x in
                                   front_stub_data["ECS"]]
    front_stub_data["GCS"] = [(x - train_means["GCS"]) / train_stds["GCS"] for x in
                                   front_stub_data["GCS"]]
    front_stub_data["USS"] = [(x - train_means["USS"]) / train_stds["USS"] for x in
                                   front_stub_data["USS"]]

    #print(front_stub_data)
    ###### Normalize the input evaluation series
    # de-mean and standardize eval inputs variance
    eval_data["ES"] = [(x - train_means["ES"]) / train_stds["ES"] for x in eval_data["ES"]]
    eval_data["EC"] = [(x - train_means["EC"]) / train_stds["EC"] for x in eval_data["EC"]]
    eval_data["GC"] = [(x - train_means["GC"]) / train_stds["GC"] for x in eval_data["GC"]]
    eval_data["US"] = [(x - train_means["US"]) / train_stds["US"] for x in eval_data["US"]]
    eval_data["BC"] = [(x - train_means["BC"]) / train_stds["BC"] for x in eval_data["BC"]]
    eval_data["ESS"] = [(x - train_means["ESS"]) / train_stds["ESS"] for x in eval_data["ESS"]]
    eval_data["ECS"] = [(x - train_means["ECS"]) / train_stds["ECS"] for x in eval_data["ECS"]]
    eval_data["GCS"] = [(x - train_means["GCS"]) / train_stds["GCS"] for x in eval_data["GCS"]]
    eval_data["USS"] = [(x - train_means["USS"]) / train_stds["USS"] for x in eval_data["USS"]]


def pipeline_training_data(train_data, eval_data, stub_data, out_GC, out_eval_GC):
    # convert to [rows, columns] structure
    ###### Pipeline the training data into the input format
    train_data["ES"] = array(train_data["ES"]).reshape((DATASET_MAX, 1))
    train_data["EC"] = array(train_data["EC"]).reshape((DATASET_MAX, 1))
    train_data["GC"] = array(train_data["GC"]).reshape((DATASET_MAX, 1))
    train_data["US"] = array(train_data["US"]).reshape((DATASET_MAX, 1))
    train_data["BC"] = array(train_data["BC"]).reshape((DATASET_MAX, 1))
    train_data["ESS"] = array(train_data["ESS"]).reshape((DATASET_MAX, 1))
    train_data["ECS"] = array(train_data["ECS"]).reshape((DATASET_MAX, 1))
    train_data["GCS"] = array(train_data["GCS"]).reshape((DATASET_MAX, 1))
    train_data["USS"] = array(train_data["USS"]).reshape((DATASET_MAX, 1))

    ###### Pipeline the most current observation data into the input format
    stub_data["ES"] = array(stub_data["ES"]).reshape((NUM_TS_LAGS, 1))
    stub_data["EC"] = array(stub_data["EC"]).reshape((NUM_TS_LAGS, 1))
    stub_data["GC"] = array(stub_data["GC"]).reshape((NUM_TS_LAGS, 1))
    stub_data["US"] = array(stub_data["US"]).reshape((NUM_TS_LAGS, 1))
    stub_data["BC"] = array(stub_data["BC"]).reshape((NUM_TS_LAGS, 1))
    stub_data["ESS"] = array(stub_data["ESS"]).reshape((NUM_TS_LAGS, 1))
    stub_data["ECS"] = array(stub_data["ECS"]).reshape((NUM_TS_LAGS, 1))
    stub_data["GCS"] = array(stub_data["GCS"]).reshape((NUM_TS_LAGS, 1))
    stub_data["USS"] = array(stub_data["USS"]).reshape((NUM_TS_LAGS, 1))

    ###### Pipeline the eval data into the input format
    eval_data["ES"] = array(eval_data["ES"]).reshape((EVAL_SIZE + EVAL_OVERLAP, 1))
    eval_data["EC"] = array(eval_data["EC"]).reshape((EVAL_SIZE + EVAL_OVERLAP, 1))
    eval_data["GC"] = array(eval_data["GC"]).reshape((EVAL_SIZE + EVAL_OVERLAP, 1))
    eval_data["US"] = array(eval_data["US"]).reshape((EVAL_SIZE + EVAL_OVERLAP, 1))
    eval_data["BC"] = array(eval_data["BC"]).reshape((EVAL_SIZE + EVAL_OVERLAP, 1))
    eval_data["ESS"] = array(eval_data["ESS"]).reshape((EVAL_SIZE + EVAL_OVERLAP, 1))
    eval_data["ECS"] = array(eval_data["ECS"]).reshape((EVAL_SIZE + EVAL_OVERLAP, 1))
    eval_data["GCS"] = array(eval_data["GCS"]).reshape((EVAL_SIZE + EVAL_OVERLAP, 1))
    eval_data["USS"] = array(eval_data["USS"]).reshape((EVAL_SIZE + EVAL_OVERLAP, 1))


def model_fitness_score(eval_history):
    return mean(eval_history[1]) - std(eval_history[1]) / 4 + log(eval_history[1].size) / 1000


def initialize_new_model(models, n_steps_in, n_features):
    # define model
    print('Re-creating new models')
    for model_place in range(MODEL_POOL_SIZE):
        model = Sequential()
        model.add(LSTM(120, activation='tanh', input_shape=(n_steps_in, n_features)))
        model.add(Activation("tanh"))
        model.add(Dropout(0.25))
        model.add(Dense(25))
        model.add(Dropout(0.25))
        model.add(Activation("tanh"))
        model.add(Dropout(0.25))
        model.add(Activation("tanh"))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        # leaves models empty initially
        models.append([model_place, array(0), model])
        print("New model. Element has ", len(models[1]), "elements")


def parse_ES_futures_price(strtime, symbol, strprice, esprice, esprice2):
    esspread = None;
    if datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2021-06-14', DATE_FORMAT):
        if symbol.find('ESM') > -1:
            esprice = float(strprice)
            if esprice2 is not None:
                esspread = esprice - esprice2
        if symbol.find('ESU') > -1:
            esprice2 = float(strprice)
            if esprice is not None:
                esspread = esprice - esprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2021-09-19', DATE_FORMAT):
        if symbol.find('ESU') > -1:
            esprice = float(strprice)
            if esprice2 is not None:
                esspread = esprice - esprice2
        if symbol.find('ESZ') > -1:
            esprice2 = float(strprice)
            if esprice is not None:
                esspread = esprice - esprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2021-12-18',DATE_FORMAT):
        if symbol.find('ESZ') > -1:
            esprice = float(strprice)
            if esprice2 is not None:
                esspread = esprice - esprice2
        if symbol.find('ESH') > -1:
            esprice2 = float(strprice)
            if esprice is not None:
                esspread = esprice - esprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2022-03-19', DATE_FORMAT):
        if symbol.find('ESH') > -1:
            esprice = float(strprice)
            if esprice2 is not None:
                esspread = esprice - esprice2
        if symbol.find('ESM') > -1:
            esprice2 = float(strprice)
            if esprice is not None:
                esspread = esprice - esprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2022-06-17', DATE_FORMAT):
        if symbol.find('ESM') > -1:
            esprice = float(strprice)
            if esprice2 is not None:
                esspread = esprice - esprice2
        if symbol.find('ESU') > -1:
            esprice2 = float(strprice)
            if esprice is not None:
                esspread = esprice - esprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2022-09-18', DATE_FORMAT):
        if symbol.find('ESU') > -1:
            esprice = float(strprice)
            if esprice2 is not None:
                esspread = esprice - esprice2
        if symbol.find('ESZ') > -1:
            esprice2 = float(strprice)
            if esprice is not None:
                esspread = esprice - esprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2022-12-18', DATE_FORMAT):
        if symbol.find('ESZ') > -1:
            esprice = float(strprice)
            if esprice2 is not None:
                esspread = esprice - esprice2
        if symbol.find('ESH') > -1:
            esprice2 = float(strprice)
            if esprice is not None:
                esspread = esprice - esprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2023-03-18', DATE_FORMAT):
        if symbol.find('ESH') > -1:
            esprice = float(strprice)
            if esprice2 is not None:
                esspread = esprice - esprice2
        if symbol.find('ESM') > -1:
            esprice2 = float(strprice)
            if esprice is not None:
                esspread = esprice - esprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2023-06-17', DATE_FORMAT):
        if symbol.find('ESM') > -1:
            esprice = float(strprice)
            if esprice2 is not None:
                esspread = esprice - esprice2
        if symbol.find('ESU') > -1:
            esprice2 = float(strprice)
            if esprice is not None:
                esspread = esprice - esprice2
    else:
        if symbol.find('ESU') > -1:
            esprice = float(strprice)
            if esprice2 is not None:
                esspread = esprice - esprice2
        if symbol.find('ESZ') > -1:
            esprice2 = float(strprice)
            if esprice is not None:
                esspread = esprice - esprice2

    return esprice, esprice2, esspread


def parse_EC_futures_price(strtime, symbol, strprice, ecprice, ecprice2):
    ecspread = None;
    if datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2021-03-15', DATE_FORMAT):
        if symbol.find('ECH') > -1:
            ecprice = float(strprice)
            if ecprice2 is not None:
                ecspread = ecprice - ecprice2
        if symbol.find('ECM') > -1:
            ecprice2 = float(strprice)
            if ecprice is not None:
                ecspread = ecprice - ecprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2021-06-14', DATE_FORMAT):
        if symbol.find('ECM') > -1:
            ecprice = float(strprice)
            if ecprice2 is not None:
                ecspread = ecprice - ecprice2
        if symbol.find('ECU') > -1:
            ecprice2 = float(strprice)
            if ecprice is not None:
                ecspread = ecprice - ecprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2021-09-14', DATE_FORMAT):
        if symbol.find('ECU') > -1:
            ecprice = float(strprice)
            if ecprice2 is not None:
                ecspread = ecprice - ecprice2
        if symbol.find('ECZ') > -1:
            ecprice2 = float(strprice)
            if ecprice is not None:
                ecspread = ecprice - ecprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2021-12-11', DATE_FORMAT):
        if symbol.find('ECZ') > -1:
            ecprice = float(strprice)
            if ecprice2 is not None:
                ecspread = ecprice - ecprice2
        if symbol.find('ECH') > -1:
            ecprice2 = float(strprice)
            if ecprice is not None:
                ecspread = ecprice - ecprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2022-03-14', DATE_FORMAT):
        if symbol.find('ECH') > -1:
            ecprice = float(strprice)
            if ecprice2 is not None:
                ecspread = ecprice - ecprice2
        if symbol.find('ECM') > -1:
            ecprice2 = float(strprice)
            if ecprice is not None:
                ecspread = ecprice - ecprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2022-06-14', DATE_FORMAT):
        if symbol.find('ECM') > -1:
            ecprice = float(strprice)
            if ecprice2 is not None:
                ecspread = ecprice - ecprice2
        if symbol.find('ECU') > -1:
            ecprice2 = float(strprice)
            if ecprice is not None:
                ecspread = ecprice - ecprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2022-09-18', DATE_FORMAT):
        if symbol.find('ECU') > -1:
            ecprice = float(strprice)
            if ecprice2 is not None:
                ecspread = ecprice - ecprice2
        if symbol.find('ECZ') > -1:
            ecprice2 = float(strprice)
            if ecprice is not None:
                ecspread = ecprice - ecprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2022-12-18', DATE_FORMAT):
        if symbol.find('ECZ') > -1:
            ecprice = float(strprice)
            if ecprice2 is not None:
                ecspread = ecprice - ecprice2
        if symbol.find('ECH') > -1:
            ecprice2 = float(strprice)
            if ecprice is not None:
                ecspread = ecprice - ecprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2023-03-10', DATE_FORMAT):
        if symbol.find('ECH') > -1:
            ecprice = float(strprice)
            if ecprice2 is not None:
                ecspread = ecprice - ecprice2
        if symbol.find('ECM') > -1:
            ecprice2 = float(strprice)
            if ecprice is not None:
                ecspread = ecprice - ecprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2023-06-17', DATE_FORMAT):
        if symbol.find('ECM') > -1:
            ecprice = float(strprice)
            if ecprice2 is not None:
                ecspread = ecprice - ecprice2
        if symbol.find('ECU') > -1:
            ecprice2 = float(strprice)
            if ecprice is not None:
                ecspread = ecprice - ecprice2
    else:
        if symbol.find('ECU') > -1:
            ecprice = float(strprice)
            if ecprice2 is not None:
                ecspread = ecprice - ecprice2
        if symbol.find('ECZ') > -1:
            ecprice2 = float(strprice)
            if ecprice is not None:
                ecspread = ecprice - ecprice2

    return ecprice, ecprice2, ecspread


def parse_GC_futures_price(strtime, symbol, strprice, gcprice, gcprice2):
    gcspread = None;
    if datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2021-03-28', DATE_FORMAT):
        if symbol.find('GCJ') > -1:
            gcprice = float(strprice)
            if gcprice2 is not None:
                gcspread = gcprice - gcprice2
        if symbol.find('GCM') > -1:
            gcprice2 = float(strprice)
            if gcprice is not None:
                gcspread = gcprice - gcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2021-05-31', DATE_FORMAT):
        if symbol.find('GCM') > -1:
            gcprice = float(strprice)
            if gcprice2 is not None:
                gcspread = gcprice - gcprice2
        if symbol.find('GCQ') > -1:
            gcprice2 = float(strprice)
            if gcprice is not None:
                gcspread = gcprice - gcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2021-07-31', DATE_FORMAT):
        if symbol.find('GCQ') > -1:
            gcprice = float(strprice)
            if gcprice2 is not None:
                gcspread = gcprice - gcprice2
        if symbol.find('GCZ') > -1:
            gcprice2 = float(strprice)
            if gcprice is not None:
                gcspread = (gcprice - gcprice2) / 2  # four month gap, instead of two
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2021-12-26', DATE_FORMAT):
        if symbol.find('GCZ') > -1:
            gcprice = float(strprice)
            if gcprice2 is not None:
                gcspread = gcprice - gcprice2
        if symbol.find('GCG') > -1:
            gcprice2 = float(strprice)
            if gcprice is not None:
                gcspread = gcprice - gcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2022-02-20', DATE_FORMAT):
        if symbol.find('GCG') > -1:
            gcprice = float(strprice)
            if gcprice2 is not None:
                gcspread = gcprice - gcprice2
        if symbol.find('GCJ') > -1:
            gcprice2 = float(strprice)
            if gcprice is not None:
                gcspread = gcprice - gcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2022-04-25',DATE_FORMAT):
        if symbol.find('GCJ') > -1:
            gcprice = float(strprice)
            if gcprice2 is not None:
                gcspread = gcprice - gcprice2
        if symbol.find('GCM') > -1:
            gcprice2 = float(strprice)
            if gcprice is not None:
                gcspread = gcprice - gcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2022-05-29',DATE_FORMAT):
        if symbol.find('GCM') > -1:
            gcprice = float(strprice)
            if gcprice2 is not None:
                gcspread = gcprice - gcprice2
        if symbol.find('GCQ') > -1:
            gcprice2 = float(strprice)
            if gcprice is not None:
                gcspread = gcprice - gcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2022-08-02',DATE_FORMAT):
        if symbol.find('GCQ') > -1:
            gcprice = float(strprice)
            if gcprice2 is not None:
                gcspread = gcprice - gcprice2
        if symbol.find('GCZ') > -1:
            gcprice2 = float(strprice)
            if gcprice is not None:
                gcspread = gcprice - gcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2022-12-09',DATE_FORMAT):
        if symbol.find('GCZ') > -1:
            gcprice = float(strprice)
            if gcprice2 is not None:
                gcspread = gcprice - gcprice2
        if symbol.find('GCG') > -1:
            gcprice2 = float(strprice)
            if gcprice is not None:
                gcspread = gcprice - gcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2023-02-10',DATE_FORMAT):
        if symbol.find('GCG') > -1:
            gcprice = float(strprice)
            if gcprice2 is not None:
                gcspread = gcprice - gcprice2
        if symbol.find('GCJ') > -1:
            gcprice2 = float(strprice)
            if gcprice is not None:
                gcspread = gcprice - gcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2023-04-27',DATE_FORMAT):
        if symbol.find('GCJ') > -1:
            gcprice = float(strprice)
            if gcprice2 is not None:
                gcspread = gcprice - gcprice2
        if symbol.find('GCM') > -1:
            gcprice2 = float(strprice)
            if gcprice is not None:
                gcspread = gcprice - gcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2023-06-28',DATE_FORMAT):
        if symbol.find('GCM') > -1:
            gcprice = float(strprice)
            if gcprice2 is not None:
                gcspread = gcprice - gcprice2
        if symbol.find('GCQ') > -1:
            gcprice2 = float(strprice)
            if gcprice is not None:
                gcspread = gcprice - gcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2023-08-01',DATE_FORMAT):
        if symbol.find('GCQ') > -1:
            gcprice = float(strprice)
            if gcprice2 is not None:
                gcspread = gcprice - gcprice2
        if symbol.find('GCZ') > -1:
            gcprice2 = float(strprice)
            if gcprice is not None:
                gcspread = gcprice - gcprice2
    else:
        if symbol.find('GCZ') > -1:
            gcprice = float(strprice)
            if gcprice2 is not None:
                gcspread = gcprice - gcprice2
        if symbol.find('GCG') > -1:
            gcprice2 = float(strprice)
            if gcprice is not None:
                gcspread = gcprice - gcprice2

    return gcprice, gcprice2, gcspread


def parse_US_futures_price(strtime, symbol, strprice, usprice, usprice2):
    usspread = None;
    if datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2021-06-14', DATE_FORMAT):
        if symbol.find('USM') > -1:
            parts = str.split(strprice, '-')
            if len(parts) > 1:
                usprice = float(float(parts[0]) + float(parts[1]) / 32)
            else:
                usprice = float(strprice)
            if usprice2 is not None:
                usspread = usprice - usprice2
        if symbol.find('USU') > -1:
            parts = str.split(strprice, '-')
            if len(parts) > 1:
                usprice2 = float(float(parts[0]) + float(parts[1]) / 32)
            else:
                usprice2 = float(strprice)
            if usprice is not None:
                usspread = usprice - usprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2021-09-19', DATE_FORMAT):
        if symbol.find('USU') > -1:
            parts = str.split(strprice, '-')
            if len(parts) > 1:
                usprice = float(float(parts[0]) + float(parts[1]) / 32)
            else:
                usprice = float(strprice)
            if usprice2 is not None:
                usspread = usprice - usprice2
        if symbol.find('USZ') > -1:
            parts = str.split(strprice, '-')
            if len(parts) > 1:
                usprice2 = float(float(parts[0]) + float(parts[1]) / 32)
            else:
                usprice2 = float(strprice)
            if usprice is not None:
                usspread = usprice - usprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2021-12-18', DATE_FORMAT):
        if symbol.find('USZ') > -1:
            parts = str.split(strprice, '-')
            if len(parts) > 1:
                usprice = float(float(parts[0]) + float(parts[1]) / 32)
            else:
                usprice = float(strprice)
            if usprice2 is not None:
                usspread = usprice - usprice2
        if symbol.find('USH') > -1:
            parts = str.split(strprice, '-')
            if len(parts) > 1:
                usprice2 = float(float(parts[0]) + float(parts[1]) / 32)
            else:
                usprice2 = float(strprice)
            if usprice is not None:
                usspread = usprice - usprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2022-03-23', DATE_FORMAT):
        if symbol.find('USH') > -1:
            parts = str.split(strprice, '-')
            if len(parts) > 1:
                usprice = float(float(parts[0]) + float(parts[1]) / 32)
            else:
                usprice = float(strprice)
            if usprice2 is not None:
                usspread = usprice - usprice2
        if symbol.find('USM') > -1:
            parts = str.split(strprice, '-')
            if len(parts) > 1:
                usprice2 = float(float(parts[0]) + float(parts[1]) / 32)
            else:
                usprice2 = float(strprice)
            if usprice is not None:
                usspread = usprice - usprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2022-06-19', DATE_FORMAT):
        if symbol.find('USM') > -1:
            parts = str.split(strprice, '-')
            if len(parts) > 1:
                usprice = float(float(parts[0]) + float(parts[1])/32)
            else:
                usprice = float(strprice)
            if usprice2 is not None:
                usspread = usprice - usprice2
        if symbol.find('USU') > -1:
            parts = str.split(strprice, '-')
            if len(parts) > 1:
                usprice2 = float(float(parts[0]) + float(parts[1])/32)
            else:
                usprice2 = float(strprice)
            if usprice is not None:
                usspread = usprice - usprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2022-09-18', DATE_FORMAT):
        if symbol.find('USU') > -1:
            parts = str.split(strprice, '-')
            if len(parts) > 1:
                usprice = float(float(parts[0]) + float(parts[1])/32)
            else:
                usprice = float(strprice)
            if usprice2 is not None:
                usspread = usprice - usprice2
        if symbol.find('USZ') > -1:
            parts = str.split(strprice, '-')
            if len(parts) > 1:
                usprice2 = float(float(parts[0]) + float(parts[1])/32)
            else:
                usprice2 = float(strprice)
            if usprice is not None:
                usspread = usprice - usprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2022-12-18', DATE_FORMAT):
        if symbol.find('USZ') > -1:
            parts = str.split(strprice, '-')
            if len(parts) > 1:
                usprice = float(float(parts[0]) + float(parts[1])/32)
            else:
                usprice = float(strprice)
            if usprice2 is not None:
                usspread = usprice - usprice2
        if symbol.find('USH') > -1:
            parts = str.split(strprice, '-')
            if len(parts) > 1:
                usprice2 = float(float(parts[0]) + float(parts[1])/32)
            else:
                usprice2 = float(strprice)
            if usprice is not None:
                usspread = usprice - usprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2023-03-10', DATE_FORMAT):
        if symbol.find('USH') > -1:
            parts = str.split(strprice, '-')
            if len(parts) > 1:
                usprice = float(float(parts[0]) + float(parts[1])/32)
            else:
                usprice = float(strprice)
            if usprice2 is not None:
                usspread = usprice - usprice2
        if symbol.find('USM') > -1:
            parts = str.split(strprice, '-')
            if len(parts) > 1:
                usprice2 = float(float(parts[0]) + float(parts[1])/32)
            else:
                usprice2 = float(strprice)
            if usprice is not None:
                usspread = usprice - usprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2023-06-17', DATE_FORMAT):
        if symbol.find('USM') > -1:
            parts = str.split(strprice, '-')
            if len(parts) > 1:
                usprice = float(float(parts[0]) + float(parts[1])/32)
            else:
                usprice = float(strprice)
            if usprice2 is not None:
                usspread = usprice - usprice2
        if symbol.find('USU') > -1:
            parts = str.split(strprice, '-')
            if len(parts) > 1:
                usprice2 = float(float(parts[0]) + float(parts[1])/32)
            else:
                usprice2 = float(strprice)
            if usprice is not None:
                usspread = usprice - usprice2
    else:
        if symbol.find('USU') > -1:
            parts = str.split(strprice, '-')
            if len(parts) > 1:
                usprice = float(float(parts[0]) + float(parts[1])/32)
            else:
                usprice = float(strprice)
            if usprice2 is not None:
                usspread = usprice - usprice2
        if symbol.find('USZ') > -1:
            parts = str.split(strprice, '-')
            if len(parts) > 1:
                usprice2 = float(float(parts[0]) + float(parts[1])/32)
            else:
                usprice2 = float(strprice)
            if usprice is not None:
                usspread = usprice - usprice2

    return usprice, usprice2, usspread


def parse_BC_futures_price(strtime, symbol, strprice, bcprice, bcprice2):
    bcspread = None;
    if datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2021-03-28', DATE_FORMAT):
        if symbol.find('BTCH') > -1:
            bcprice = float(strprice)
            if bcprice2 is not None:
                bcspread = bcprice - bcprice2
        if symbol.find('BTCJ') > -1:
            bcprice2 = float(strprice)
            if bcprice is not None:
                bcspread = bcprice - bcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2021-04-25', DATE_FORMAT):
        if symbol.find('BTCJ') > -1:
            bcprice = float(strprice)
            if bcprice2 is not None:
                bcspread = bcprice - bcprice2
        if symbol.find('BTCK') > -1:
            bcprice2 = float(strprice)
            if bcprice is not None:
                bcspread = bcprice - bcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2021-05-23', DATE_FORMAT):
        if symbol.find('BTCK') > -1:
            bcprice = float(strprice)
            if bcprice2 is not None:
                bcspread = bcprice - bcprice2
        if symbol.find('BTCM') > -1:
            bcprice2 = float(strprice)
            if bcprice is not None:
                bcspread = bcprice - bcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2021-06-27', DATE_FORMAT):
        if symbol.find('BTCM') > -1:
            bcprice = float(strprice)
            if bcprice2 is not None:
                bcspread = bcprice - bcprice2
        if symbol.find('BTCN') > -1:
            bcprice2 = float(strprice)
            if bcprice is not None:
                bcspread = bcprice - bcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2021-07-31', DATE_FORMAT):
        if symbol.find('BTCN') > -1:
            bcprice = float(strprice)
            if bcprice2 is not None:
                bcspread = bcprice - bcprice2
        if symbol.find('BTCQ') > -1:
            bcprice2 = float(strprice)
            if bcprice is not None:
                bcspread = bcprice - bcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2021-08-28', DATE_FORMAT):
        if symbol.find('BTCQ') > -1:
            bcprice = float(strprice)
            if bcprice2 is not None:
                bcspread = bcprice - bcprice2
        if symbol.find('BTCU') > -1:
            bcprice2 = float(strprice)
            if bcprice is not None:
                bcspread = bcprice - bcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2021-09-28', DATE_FORMAT):
        if symbol.find('BTCU') > -1:
            bcprice = float(strprice)
            if bcprice2 is not None:
                bcspread = bcprice - bcprice2
        if symbol.find('BTCV') > -1:
            bcprice2 = float(strprice)
            if bcprice is not None:
                bcspread = bcprice - bcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2021-10-29', DATE_FORMAT):
        if symbol.find('BTCV') > -1:
            bcprice = float(strprice)
            if bcprice2 is not None:
                bcspread = bcprice - bcprice2
        if symbol.find('BTCX') > -1:
            bcprice2 = float(strprice)
            if bcprice is not None:
                bcspread = bcprice - bcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2021-11-28', DATE_FORMAT):
        if symbol.find('BTCX') > -1:
            bcprice = float(strprice)
            if bcprice2 is not None:
                bcspread = bcprice - bcprice2
        if symbol.find('BTCZ') > -1:
            bcprice2 = float(strprice)
            if bcprice is not None:
                bcspread = bcprice - bcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2022-01-01', DATE_FORMAT):
        if symbol.find('BTCZ') > -1:
            bcprice = float(strprice)
            if bcprice2 is not None:
                bcspread = bcprice - bcprice2
        if symbol.find('BTCF') > -1:
            bcprice2 = float(strprice)
            if bcprice is not None:
                bcspread = bcprice - bcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2022-01-31', DATE_FORMAT):
        if symbol.find('BTCF') > -1:
            bcprice = float(strprice)
            if bcprice2 is not None:
                bcspread = bcprice - bcprice2
        if symbol.find('BTCG') > -1:
            bcprice2 = float(strprice)
            if bcprice is not None:
                bcspread = bcprice - bcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2022-02-28', DATE_FORMAT):
        if symbol.find('BTCG') > -1:
            bcprice = float(strprice)
            if bcprice2 is not None:
                bcspread = bcprice - bcprice2
        if symbol.find('BTCH') > -1:
            bcprice2 = float(strprice)
            if bcprice is not None:
                bcspread = bcprice - bcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2022-03-26', DATE_FORMAT):
        if symbol.find('BTCH') > -1:
            bcprice = float(strprice)
            if bcprice2 is not None:
                bcspread = bcprice - bcprice2
        if symbol.find('BTCJ') > -1:
            bcprice2 = float(strprice)
            if bcprice is not None:
                bcspread = bcprice - bcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2022-04-30',DATE_FORMAT):
        if symbol.find('BTCJ') > -1:
            bcprice = float(strprice)
            if bcprice2 is not None:
                bcspread = bcprice - bcprice2
        if symbol.find('BTCJ') > -1:
            bcprice2 = float(strprice)
            if bcprice is not None:
                bcspread = bcprice - bcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2022-05-29',DATE_FORMAT):
        if symbol.find('BTCK') > -1:
            bcprice = float(strprice)
            if bcprice2 is not None:
                bcspread = bcprice - bcprice2
        if symbol.find('BTCM') > -1:
            bcprice2 = float(strprice)
            if bcprice is not None:
                bcspread = bcprice - bcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2022-06-29',DATE_FORMAT):
        if symbol.find('BTCM') > -1:
            bcprice = float(strprice)
            if bcprice2 is not None:
                bcspread = bcprice - bcprice2
        if symbol.find('BTCN') > -1:
            bcprice2 = float(strprice)
            if bcprice is not None:
                bcspread = bcprice - bcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2022-07-31',DATE_FORMAT):
        if symbol.find('BTCN') > -1:
            bcprice = float(strprice)
            if bcprice2 is not None:
                bcspread = bcprice - bcprice2
        if symbol.find('BTCQ') > -1:
            bcprice2 = float(strprice)
            if bcprice is not None:
                bcspread = bcprice - bcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2022-08-30',DATE_FORMAT):
        if symbol.find('BTCQ') > -1:
            bcprice = float(strprice)
            if bcprice2 is not None:
                bcspread = bcprice - bcprice2
        if symbol.find('BTCU') > -1:
            bcprice2 = float(strprice)
            if bcprice is not None:
                bcspread = bcprice - bcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2022-09-30',DATE_FORMAT):
        if symbol.find('BTCU') > -1:
            bcprice = float(strprice)
            if bcprice2 is not None:
                bcspread = bcprice - bcprice2
        if symbol.find('BTCV') > -1:
            bcprice2 = float(strprice)
            if bcprice is not None:
                bcspread = bcprice - bcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2022-10-31',DATE_FORMAT):
        if symbol.find('BTCV') > -1:
            bcprice = float(strprice)
            if bcprice2 is not None:
                bcspread = bcprice - bcprice2
        if symbol.find('BTCZ') > -1:
            bcprice2 = float(strprice)
            if bcprice is not None:
                bcspread = bcprice - bcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2022-12-01',DATE_FORMAT):
        if symbol.find('BTCZ') > -1:
            bcprice = float(strprice)
            if bcprice2 is not None:
                bcspread = bcprice - bcprice2
        if symbol.find('BTCF') > -1:
            bcprice2 = float(strprice)
            if bcprice is not None:
                bcspread = bcprice - bcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2023-01-31',DATE_FORMAT):
        if symbol.find('BTCF') > -1:
            bcprice = float(strprice)
            if bcprice2 is not None:
                bcspread = bcprice - bcprice2
        if symbol.find('BTCG') > -1:
            bcprice2 = float(strprice)
            if bcprice is not None:
                bcspread = bcprice - bcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2023-02-28',DATE_FORMAT):
        if symbol.find('BTCG') > -1:
            bcprice = float(strprice)
            if bcprice2 is not None:
                bcspread = bcprice - bcprice2
        if symbol.find('BTCH') > -1:
            bcprice2 = float(strprice)
            if bcprice is not None:
                bcspread = bcprice - bcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2023-03-31',DATE_FORMAT):
        if symbol.find('BTCH') > -1:
            bcprice = float(strprice)
            if bcprice2 is not None:
                bcspread = bcprice - bcprice2
        if symbol.find('BTCJ') > -1:
            bcprice2 = float(strprice)
            if bcprice is not None:
                bcspread = bcprice - bcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2023-05-03',DATE_FORMAT):
        if symbol.find('BTCJ') > -1:
            bcprice = float(strprice)
            if bcprice2 is not None:
                bcspread = bcprice - bcprice2
        if symbol.find('BTCK') > -1:
            bcprice2 = float(strprice)
            if bcprice is not None:
                bcspread = bcprice - bcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2023-05-31',DATE_FORMAT):
        if symbol.find('BTCK') > -1:
            bcprice = float(strprice)
            if bcprice2 is not None:
                bcspread = bcprice - bcprice2
        if symbol.find('BTCM') > -1:
            bcprice2 = float(strprice)
            if bcprice is not None:
                bcspread = bcprice - bcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2023-07-01',DATE_FORMAT):
        if symbol.find('BTCM') > -1:
            bcprice = float(strprice)
            if bcprice2 is not None:
                bcspread = bcprice - bcprice2
        if symbol.find('BTCN') > -1:
            bcprice2 = float(strprice)
            if bcprice is not None:
                bcspread = bcprice - bcprice2
    elif datetime.strptime(strtime, TIME_FORMAT) < datetime.strptime('2023-07-29',DATE_FORMAT):
        if symbol.find('BTCN') > -1:
            bcprice = float(strprice)
            if bcprice2 is not None:
                bcspread = bcprice - bcprice2
        if symbol.find('BTCQ') > -1:
            bcprice2 = float(strprice)
            if bcprice is not None:
                bcspread = bcprice - bcprice2
    else:
        if symbol.find('BTCQ') > -1:
            bcprice = float(strprice)
            if bcprice2 is not None:
                bcspread = bcprice - bcprice2
        if symbol.find('BTCU') > -1:
            bcprice2 = float(strprice)
            if bcprice is not None:
                bcspread = bcprice - bcprice2

    return bcprice, bcprice2, bcspread


def load_latest_data():
    in_timeidx = []
    in_ES = []
    in_EC = []
    in_GC = []
    in_US = []
    in_ESS = []
    in_ECS = []
    in_GCS = []
    in_USS = []
    in_BC = []

    out_ES = []
    out_EC = []
    out_GC = []
    out_US = []

    lasthour = ''
    lastmin = ''
    
    with open('5minuteBars.csv', encoding='utf-8') as f:
        data = f.read().strip()
        # print(data)
        esprice = None
        ecprice = None
        gcprice = None
        usprice = None
        bcprice = None
        esprice2 = None
        ecprice2 = None
        gcprice2 = None
        usprice2 = None
        bcprice2 = None
        timesnap = None
        lastsnapshottime = None

        ###### Read Market Data input module
        for rec in str.split(data, '\n', maxsplit=-1):
            srec = str.split(rec, ',', maxsplit=-1)
            if len(srec) < 3:
                continue
            symbol = srec[0].strip()
            if symbol.startswith('NQ'):
                continue
            strtime = srec[1].strip()
            times = str.split(strtime, ' ')
            datepart = times[0]
            timepart = times[1]
            times = str.split(timepart, ':')
            hour = times[0]
            minute = times[1]
            strprice = srec[2].strip()
            ###### If it's a new timestamp: Initialize state variables, time instance i
            ###### ###### Parse Futures state variable using state of time instance i
            if (lastsnapshottime is None and timesnap is None) or (
                    lastsnapshottime is not None and '{0} {1}:{2}:00'.format(datepart, hour,
                                                                             minute) != timesnap):
                # print('Resetting for a new time period with {} {}. lastsnapshottime={}, timesnap={}'.format(datepart, timepart, lastsnapshottime, timesnap))
                esprice = None
                ecprice = None
                gcprice = None
                usprice = None
                bcprice = None
                esprice2 = None
                ecprice2 = None
                gcprice2 = None
                usprice2 = None
                bcprice2 = None
                esspread = None
                ecspread = None
                gcspread = None
                usspread = None
                bcspread = None
                timesnap = '{0} {1}:{2}:00'.format(datepart, hour, minute)

            if symbol.find('ES') == 0:
                esprice, esprice2, esspread = parse_ES_futures_price(strtime, symbol, strprice, esprice, esprice2)
            elif symbol.find('EC') == 0:
                ecprice, ecprice2, ecspread = parse_EC_futures_price(strtime, symbol, strprice, ecprice, ecprice2)
            elif symbol.find('GC') == 0:
                gcprice, gcprice2, gcspread = parse_GC_futures_price(strtime, symbol, strprice, gcprice, gcprice2)
            elif symbol.find('US') == 0:
                usprice, usprice2, usspread = parse_US_futures_price(strtime, symbol, strprice, usprice, usprice2)
            elif symbol.find('BTC') == 0:
                bcprice, bcprice2, bcspread = parse_BC_futures_price(strtime, symbol, strprice, bcprice, bcprice2)

            # print('{0} {1}:{2}.00'.format(datepart, hour, minute), esprice, ecprice, gcprice, usprice, esspread, ecspread, gcspread, usspread)
            ###### Insert fully populated valid timestamp i state observation, or not.
            #if timesnap is None or timesnap > "2023-07-01" and (esprice is None or ecprice is None \
            #        or esprice2 is None or ecprice2 is None \
            #        or gcprice is None or usprice is None \
            #        or gcprice2 is None or usprice2 is None \
            #        or bcprice is None or bcprice2 is None \
            #        or esspread is None or ecspread is None or gcspread is None or usspread is None):
            #    print('{0} {1}:{2}.00'.format(datepart, hour, minute), esprice, ecprice, gcprice, usprice, esspread, ecspread, gcspread, usspread)

            if timesnap is not None and esprice is not None and esprice > 0 and ecprice is not None and ecprice > 0 \
                    and esprice2 is not None and esprice2 > 0 and ecprice2 is not None and ecprice2 > 0 \
                    and gcprice is not None and gcprice > 0 and usprice is not None and usprice > 0 \
                    and gcprice2 is not None and gcprice2 > 0 and usprice2 is not None and usprice2 > 0 \
                    and bcprice is not None and bcprice > 0 and bcprice2 is not None and bcprice2 > 0 \
                    and esspread is not None and ecspread is not None and gcspread is not None and usspread is not None:
                in_ES.append(esprice)
                in_EC.append(ecprice)
                in_GC.append(gcprice)
                in_US.append(usprice)
                in_BC.append(bcprice)
                in_ESS.append(esspread)
                in_ECS.append(ecspread)
                in_GCS.append(gcspread)
                in_USS.append(usspread)
                in_timeidx.append(timesnap)

                lastsnapshottime = timesnap
            # print('Appending {0} {1}:{2}.00'.format(datepart, hour, minute), esprice, ecprice, gcprice, usprice, esspread, ecspread, gcspread, usspread)
            # elif datetime.strptime(strtime, '%Y-%m-%d %H:%M:%S.%f') > datetime.strptime('2021-11-29', '%Y-%m-%d'):
            #        print('Invalid price skipped {0} {1} {2} {3}'.format(esprice, ecprice, gcprice, usprice), esspread, ecspread, gcspread, usspread)
        # exit()
    return {"ES": in_ES, "EC": in_EC, "GC": in_GC, "US": in_US, "ESS": in_ESS, "ECS": in_ECS, "GCS": in_GCS,
            "USS": in_USS, "BC": in_BC, "timeidx": in_timeidx}


def predict_train_evaluate_select_models(models, train_data, out_train_obj, eval_data, out_eval_obj, stub_data, today,
                                         avg_out_obj, avg_out_obj_winsorized, std_out_obj_winsorized, in_timeidx,
                                         n_steps_in, n_steps_out):
    runtime = datetime.now()
    print('Run time={} Data time={}\nTrend={:}, Skew={:}'.format(runtime, in_timeidx[len(in_timeidx) - 1], avg_out_obj,
                                                                 avg_out_obj_winsorized))

    dataset = hstack((train_data["ES"], train_data["EC"], train_data["GC"], train_data["US"],
                      train_data["BC"], train_data["ESS"], train_data["ECS"], train_data["GCS"],
                      train_data["USS"], out_train_obj))
    # covert into input/output
    X, y = split_sequences(dataset, n_steps_in, n_steps_out)

    evalset = hstack((eval_data["ES"], eval_data["EC"], eval_data["GC"], eval_data["US"],
                      eval_data["BC"], eval_data["ESS"], eval_data["ECS"], eval_data["GCS"],
                      eval_data["USS"],
                      out_eval_obj))
    evalX, evaly = split_sequences(evalset, n_steps_in, n_steps_out)
    #with open('C:\\Users\\ideav\\Documents\\PythonWork\\FiveFactor\\pipelined_outputs-23{0:02}{1:02}-{2:02}{3:02}.csv'.format(runtime.month,runtime.day,runtime.hour,runtime.minute), 'w') as f:
    #    for e1 in evaly:
    #        for e2 in e1:
    #            f.write('{0}'.format(e2))
    #        f.write('\n')

    n_features = X.shape[2]
    #print(X.shape)

    votecount = 0
    average_rsq = 0
    average_count = 0

    ###### Model pool.  Have the model pool do the interation, managing the genetic algo.
    for loop in range(MODEL_POOL_SIZE):
        # horizontally stack columns

        if loop < len(models):
            model = models[loop][2]
            model_place = loop
        else:
            model_place = 0

        # predict
        predictions = model.predict(evalX, batch_size=1000, verbose=0)
        #with open('C:\\Users\\ideav\\Documents\\PythonWork\\FiveFactor\\predicted_outputs-23{0:02}{1:02}-{2:02}{3:02}-{4}.csv'.format(runtime.month,runtime.day,runtime.hour,runtime.minute,loop), 'w') as f:
        #    for e1 in predictions:
        #        for e2 in e1:
        #            f.write('{0}'.format(e2))
        #        f.write('\n')            
        #print('evaly length=', len(reshape(evaly, (1,-1))), ' shape=', reshape(evaly, (1,-1)).shape, ' reshape(predictions, (1,-1)) length=', len(reshape(predictions, (1,-1))), ' shape=', reshape(predictions, (1,-1)).shape);
        slope, intercept, current_r_value, p_value, std_err = scipy.stats.linregress(
            reshape(evaly, (1,-1)), reshape(predictions, (1,-1)))
        current_r_squared = current_r_value*abs(current_r_value)

        # Do the basic prediction from the current models[], and report it
        stub_ES = stub_data["ES"]
        stub_EC = stub_data["EC"]
        stub_GC = stub_data["GC"]
        stub_US = stub_data["US"]
        stub_BC = stub_data["BC"]
        stub_ESS = stub_data["ESS"]
        stub_ECS = stub_data["ECS"]
        stub_GCS = stub_data["GCS"]
        stub_USS = stub_data["USS"]
        x_input = array([stub_ES[len(stub_ES) - 12], stub_ES[len(stub_ES) - 11], stub_ES[len(stub_ES) - 10],
                         stub_ES[len(stub_ES) - 9], stub_ES[len(stub_ES) - 8], stub_ES[len(stub_ES) - 7],
                         stub_ES[len(stub_ES) - 6], stub_ES[len(stub_ES) - 5], stub_ES[len(stub_ES) - 4],
                         stub_ES[len(stub_ES) - 3], stub_ES[len(stub_ES) - 2], today[0],
                         stub_EC[len(stub_EC) - 12], stub_EC[len(stub_EC) - 11], stub_EC[len(stub_EC) - 10],
                         stub_EC[len(stub_EC) - 9], stub_EC[len(stub_EC) - 8], stub_EC[len(stub_EC) - 7],
                         stub_EC[len(stub_EC) - 6], stub_EC[len(stub_EC) - 5], stub_EC[len(stub_EC) - 4],
                         stub_EC[len(stub_EC) - 3], stub_EC[len(stub_EC) - 2], today[1],
                         stub_GC[len(stub_GC) - 12], stub_GC[len(stub_GC) - 11], stub_GC[len(stub_GC) - 10],
                         stub_GC[len(stub_GC) - 9], stub_GC[len(stub_GC) - 8], stub_GC[len(stub_GC) - 7],
                         stub_GC[len(stub_GC) - 6], stub_GC[len(stub_GC) - 5], stub_GC[len(stub_GC) - 4],
                         stub_GC[len(stub_GC) - 3], stub_GC[len(stub_GC) - 2], today[2],
                         stub_US[len(stub_US) - 12], stub_US[len(stub_US) - 11], stub_US[len(stub_US) - 10],
                         stub_US[len(stub_US) - 9], stub_US[len(stub_US) - 8], stub_US[len(stub_US) - 7],
                         stub_US[len(stub_US) - 6], stub_US[len(stub_US) - 5], stub_US[len(stub_US) - 4],
                         stub_US[len(stub_US) - 3], stub_US[len(stub_US) - 2], today[3],
                         stub_BC[len(stub_BC) - 12], stub_BC[len(stub_BC) - 11], stub_BC[len(stub_BC) - 10],
                         stub_BC[len(stub_BC) - 9], stub_BC[len(stub_BC) - 8], stub_BC[len(stub_BC) - 7],
                         stub_BC[len(stub_BC) - 6], stub_BC[len(stub_BC) - 5], stub_BC[len(stub_BC) - 4],
                         stub_BC[len(stub_BC) - 3], stub_BC[len(stub_BC) - 2], today[4],
                         stub_ESS[len(stub_ESS) - 12], stub_ESS[len(stub_ESS) - 11],
                         stub_ESS[len(stub_ESS) - 10], stub_ESS[len(stub_ESS) - 9], stub_ESS[len(stub_ESS) - 8],
                         stub_ESS[len(stub_ESS) - 7], stub_ESS[len(stub_ESS) - 6], stub_ESS[len(stub_ESS) - 5],
                         stub_ESS[len(stub_ESS) - 4], stub_ESS[len(stub_ESS) - 3], stub_ESS[len(stub_ESS) - 2],
                         today[5],
                         stub_ECS[len(stub_ECS) - 12], stub_ECS[len(stub_ECS) - 11],
                         stub_ECS[len(stub_ECS) - 10], stub_ECS[len(stub_ECS) - 9], stub_ECS[len(stub_ECS) - 8],
                         stub_ECS[len(stub_ECS) - 7], stub_ECS[len(stub_ECS) - 6], stub_ECS[len(stub_ECS) - 5],
                         stub_ECS[len(stub_ECS) - 4], stub_ECS[len(stub_ECS) - 3], stub_ECS[len(stub_ECS) - 2],
                         today[6],
                         stub_GCS[len(stub_GCS) - 12], stub_GCS[len(stub_GCS) - 11],
                         stub_GCS[len(stub_GCS) - 10], stub_GCS[len(stub_GCS) - 9], stub_GCS[len(stub_GCS) - 8],
                         stub_GCS[len(stub_GCS) - 7], stub_GCS[len(stub_GCS) - 6], stub_GCS[len(stub_GCS) - 5],
                         stub_GCS[len(stub_GCS) - 4], stub_GCS[len(stub_GCS) - 3], stub_GCS[len(stub_GCS) - 2],
                         today[7],
                         stub_USS[len(stub_USS) - 12], stub_USS[len(stub_USS) - 11],
                         stub_USS[len(stub_USS) - 10], stub_USS[len(stub_USS) - 9], stub_USS[len(stub_USS) - 8],
                         stub_USS[len(stub_USS) - 7], stub_USS[len(stub_USS) - 6], stub_USS[len(stub_USS) - 5],
                         stub_USS[len(stub_USS) - 4], stub_USS[len(stub_USS) - 3], stub_USS[len(stub_USS) - 2],
                         today[8]])
        # x_input = x_input[::-1]

        x_input = x_input.reshape((1, n_steps_in, n_features))
        yhat = model.predict(x_input, verbose=0)
        prediction = (yhat[0][0] + avg_out_obj_winsorized) * std_out_obj_winsorized

        if current_r_squared >= 0.0:
            prediction_history.append(prediction)
        mean_prediction = mean(prediction_history)
        if len(prediction_history) > 30:
            prediction_history.pop(0)

        voting_criterion = (prediction) / 20

        average_rsq += current_r_squared
        average_count += 1

        #print(str.format("model {} avg r {:6.5f}[{:}]: {:8.4f}=>{:8.4f}, r-squared={:6.4f}",
        #                 model_place, mean(models[model_place][1]), models[model_place][1].size,
        #                 prediction, voting_criterion, current_r_squared))
        # if voting_criterion < -2.0 or voting_criterion > 2.0:
        #        winsound.Beep(frequency, duration)
        if voting_criterion < -COUNTS_AS_SIGNIFICANT:
            votecount -= 1
        elif voting_criterion > COUNTS_AS_SIGNIFICANT:
            votecount += 1
        ###predict

        ## train, evaluate, select

        # Train
        # Fit the next candidate model, and find its out of sample r-squared
        stopping_rule = [
            EarlyStopping(
                # Stop training when `loss` is no longer improving
                monitor="loss",
                # "no longer improving" being defined as "no better than 1e-2 less"
                min_delta=1e-2,
                # baseline=250,
                # "no longer improving" being further defined as "for at least 10 epochs"
                patience=200,
                verbose=0,
            )
        ]

        rollback_model = clone_model(model)
        result = model.fit(X, y, epochs=2000, batch_size=1000, validation_data=(evalX, evaly), verbose=0,
                           callbacks=stopping_rule)

        # last_price = train_data["GC"][len(train_data["GC"])-1]
        # logger.write('{},{},{},{}\n'.format(in_timeidx[len(in_timeidx) - 1],
        #                                       result.history['loss'][len(result.history['loss']) - 1],
        #                                       (yhat[0][0] - avg_out_obj_winsorized) * std_out_obj_winsorized,
        #                                       GC_last_price))

        # Evaluate for inheritance
        predictions = model.predict(evalX, batch_size=1000, verbose=0)
        slope, intercept, child_r_value, p_value, std_err = scipy.stats.linregress(
            reshape(evaly, (1,-1)), reshape(predictions, (1,-1)))
        child_r_squared = child_r_value*abs(child_r_value)

        model_to_remove = None

        if len(models) > 0:
            MAX_OOS_RSQUARED = max([mean(x[1]) for x in models])

        # Select
        # Run the genetic fitness criteria. Prodigies replace a line if sufficiently awesome.
        # If not, then if the child r-squareds is good enough, then it inherits, otherwise it reverts
        # to the parent.
        child_replaced_another_line = False

        current_model_score = 0;
#        print(type(models[model_place][1]))
        if models[model_place][1].size > 0:
            current_model_score = model_fitness_score(models[model_place])

        # Max of all the models including this one
        if len(models) > 0:
            MAX_OOS_RSQUARED = max(mean(x[1]) for x in models)

        # Min of all the other models with more than MIN_MODEL_MATURITY_TO_REPLACE in history, if there are any
        if len([x for x in models if x[1].size >= MIN_MODEL_MATURITY_TO_REPLACE and model_place != x[0]]) > 0:
            MIN_OOS_RSQUARED = min(
                mean(x[1]) for x in [y for y in models if y[1].size >= MIN_MODEL_MATURITY_TO_REPLACE and model_place != y[0]])
        else:
            MIN_OOS_RSQUARED = 1000000  # never spawn

        # Mature enough to spawn, and child is a prodigy
        max_model_score = max(model_fitness_score(x) for x in models)
        if models[model_place][1].size >= MIN_MODEL_MATURITY_TO_REPRODUCE and child_r_squared > MAX_OOS_RSQUARED and MIN_OOS_RSQUARED < 0.005:  # max_model_score:
            # Must replace the worst score, among all > 4 cycles aged, and not its parent.
            min_found = 10000000
            worst_model_idx = -1
            min_model_score = min(mean(x[1]) - std(x[1]) / 4 + log(x[1].size) / 1000 for x in models)
            for place in range(len(models)):
                if mean(models[place][1]) - std(models[place][1]) / 4 + log(
                        models[place][1].size) / 1000 <= min_found and models[place][1].size > 4:
                    min_found = mean(models[place][1]) - std(models[place][1]) / 4 + log(
                        models[place][1].size) / 1000
                    worst_model_idx = place

            if worst_model_idx > -1 and model_place != worst_model_idx:
                # Child should inherit, not replace its parent if it was the worst
                models[worst_model_idx] = [worst_model_idx, array(child_r_squared), clone_model(model)]
                models[worst_model_idx][2].compile(optimizer='adam', loss='mse')
                # TODO parameterize output series
                if Path('.\ES_model_loss_{}.csv'.format(worst_model_idx)).is_file():
                    with open('.\ES_model_loss_{}.csv'.format(worst_model_idx), 'r+',
                              encoding='utf-8') as f:
                        f.seek(0)
                        f.write('{:9.6},1'.format(mean(models[worst_model_idx][1])))
                else:
                    with open('.\ES_model_loss_{}.csv'.format(worst_model_idx), "w") as f:
                        f.seek(0)
                        f.write('{:9.6},1'.format(mean(models[worst_model_idx][1])))
                model.save('.\ES_model_{}'.format(worst_model_idx), save_format='h5')
                child_replaced_another_line = True

        if not child_replaced_another_line:
            # The child replaces the parent unless it replaced another model genetic line, or
            # unless it was inferior, i.e. < min(r, historical mean r) - abs difference btw the two.
            # print('Child was {:8.5f}. inheritance criterion was {:8.5f}'.format(child_r_squared, min(current_r_squared, mean(models[model_place][1]))))
            models[model_place][1] = np.append(models[model_place][1], current_r_squared)
            if (child_r_squared < current_r_squared and child_r_squared <= mean(models[model_place][1]) + std(
                    models[model_place][1])) or child_r_squared < current_r_squared * 0.80: # Also cannot inherit with < 80% of the current value, no matter what
                models[model_place][2] = rollback_model
                models[model_place][2].compile(optimizer='adam', loss='mse')
            else:
                rollback_model = None
        else:
            rollback_model = None

        print("model {} avg r {:6.5f}[{:}]: r2={:6.5f}: Child got {:6.5f}, {:6.2f}".format(model_place, mean(models[model_place][1]), models[model_place][1].size, current_r_squared, child_r_squared, prediction))
        if child_replaced_another_line:
            print('Replaced line {}. Parent remains, parent fit={:6.5f}'. \
                  format(worst_model_idx,
                         mean(models[model_place][1]) - std(models[model_place][1]) / 2 + log(
                             models[model_place][1].size) / 1000))
        elif (child_r_squared < current_r_squared and child_r_squared <= mean(models[model_place][1])) or child_r_squared < current_r_squared * 0.80:
            print('Child didn\'t inherit. Rolling mean now {:6.5f}, new fitness={:6.5f}'. \
                  format(mean(models[model_place][1]),
                         mean(models[model_place][1]) - std(models[model_place][1]) / 2 + log(
                             models[model_place][1].size) / 1000))
        else:
            print('Child inherited line. Rolling mean now {:6.5f}, new fitness={:6.5f}'. \
                  format(mean(models[model_place][1]),
                         mean(models[model_place][1]) - std(models[model_place][1]) / 2 + log(
                             models[model_place][1].size) / 1000))

        MIN_OOS_RSQUARED = min([mean(x[1]) for x in models])
        MAX_OOS_RSQUARED = max([mean(x[1]) for x in models])

        if not child_replaced_another_line:
            if Path('.\ES_model_loss_{}.csv'.format(model_place)).is_file():
                with open('.\ES_model_loss_{}.csv'.format(model_place), 'r+', encoding='utf-8') as f:
                    f.seek(0)
                    f.write('{:9.6},{}'.format(mean(models[model_place][1]),
                                               models[model_place][1].size))
                    f.close()
            else:
                with open('.\ES_model_loss_{}.csv'.format(model_place), "w") as f:
                    f.seek(0)
                    f.write('{:9.6},{}'.format(mean(models[model_place][1]),
                                               models[model_place][1].size))
                    f.close()
            model.save('.\ES_model_{}'.format(model_place), save_format='h5')

        # train, evaluate, select

    K.clear_session()

    average_rsq /= average_count
    # logger.flush()
    recommendation = votecount + int(avg_out_obj * 2.0)

    if recommendation > 3 or recommendation < -3:  # -5, -4, +4, or +5
        print('Voted {0} plus trend: MovingAvg={2:6.2} => Recommendation {1} Go risk on.'.format(votecount,
                                                                                                 recommendation,
                                                                                                 mean(
                                                                                                     prediction_history)))
        if average_count == 5:
            model_was_confirmed = True
        # winsound.Beep(frequency, duration)
    elif recommendation > 2 or recommendation < -2:  # -3 or +3
        print('Voted {0} plus trend: MovingAvg={2:6.2} => Recommendation {1} Consider risk on.'.format(
            votecount, recommendation, mean(prediction_history)))
        if average_count == 5:
            model_was_confirmed = True
    elif recommendation > 1 or recommendation < -1:  # -2 or +2
        print('Voted {0} plus trend: MovingAvg={2:6.2} => Recommendation {1} Hold position'.format(votecount,
                                                                                                   recommendation,
                                                                                                   mean(
                                                                                                       prediction_history)))
        if average_count == 5:
            model_was_confirmed = True
    elif recommendation > 0 or recommendation < -0:  # -1 or +1
        print('Voted {0} plus trend: MovingAvg={2:6.2} => Recommendation {1} Hold position/Caution'.format(
            votecount, recommendation, mean(prediction_history)))
        if average_count == 5:
            model_was_confirmed = True
    else:
        print('Voted {0} plus trend: MovingAvg={2:6.2} => Recommendation {1} Exit position.'.format(votecount,
                                                                                                    recommendation,
                                                                                                    mean(
                                                                                                        prediction_history)))
        if average_count == 5:
            model_was_confirmed = True
        # winsound.Beep(frequency, duration)
    # if averageloss > MAX_OOS_RSQUARED + 1:
    #        MAX_OOS_RSQUARED += 0.5

    timeofday = datetime.now()


#logger = open('C:\\Users\\ideav\\Documents\\PythonWork\\FiveFactor\\ES_log_{}{:02}{:02}{:02}{:02}{:02}.csv'.format(timeofday.year,timeofday.month,timeofday.day,timeofday.hour,timeofday.minute,timeofday.second),'w')
#logger.write('time,loss,prediction,price\n')

###### main start
def main():
    models = []
    NUM_MODELS = 5

    obj_name = "ES";

    for model_place in range(NUM_MODELS):

        modelfile = Path('.\{}_model_{}'.format(obj_name, model_place))
        if modelfile.is_file():
            mtime = modelfile.stat().st_mtime
            model = keras.models.load_model('.\{}_model_{}'.format(obj_name, model_place))
            model.compile(optimizer='adam', loss='mse')
            if Path('.\{}_model_loss_{}.csv'.format(obj_name, model_place)).is_file():
                with open('.\{}_model_loss_{}.csv'.format(obj_name, model_place), encoding='utf-8') as f:
                    data = f.read().strip()
                    items = data.split(',')
                    print('Acquired new out of sample r-squared {:9.5f}, age={}'.format(float(items[0]), items[1]))
                    r_sq = float(items[0])
                    length = int(items[1])
            models.append([model_place, array([r_sq] * length), model])
            # print('Appended initial model from file to collection')
            print('Loading model from {}_model_{} with '.format(obj_name, model_place), r_sq)
        else:
                # define model
            print('Re-creating new model')
            initialize_new_model(models, NUM_TS_LAGS, 2);


    all_data = {}
    while True:  # (((timeofday.hour == 5 and timeofday.minute > 30) or (timeofday.hour > 6)) and (timeofday.hour < 23)):
        all_data = load_latest_data()

        out_obj = prepare_objective(all_data[obj_name], all_data["timeidx"])

        # all_data = {"ES" : in_ES, "EC": in_EC, "GC": in_GC, "US": in_US, "ESS": in_ESS, "ECS": in_ECS, "GCS": in_GCS, "USS": in_USS, "BC": in_BC};
        inp = divide_input_data_into_three_parts(all_data, obj_name)

        train_data = inp[0]
        eval_data = inp[1]
        front_stub_data = inp[2]
        train_means = inp[3]
        train_stds = inp[4]
        in_timeidx = inp[5]

        #print(train_means, train_stds)
        today = array([front_stub_data["ES"][len(front_stub_data["ES"])-1],front_stub_data["EC"][len(front_stub_data["EC"])-1],front_stub_data["GC"][len(front_stub_data["GC"])-1],front_stub_data["US"][len(front_stub_data["US"])-1],front_stub_data["BC"][len(front_stub_data["BC"])-1],front_stub_data["ESS"][len(front_stub_data["ESS"])-1],front_stub_data["ECS"][len(front_stub_data["ECS"])-1],front_stub_data["GCS"][len(front_stub_data["GCS"])-1],front_stub_data["USS"][len(front_stub_data["USS"])-1]])

        avg_out = mean(out_obj)

        outp = divide_output_data_into_two_parts(out_obj)  # output: don''t care about the front app prediction stub
        out_train = outp[0]
        out_eval = outp[1]
        ftime = datetime.now()

        # TODO - Save the train_means, and train_stds to file CurrentStats.csv
        if Path('.\CurrentStats.csv').is_file():
            with open('.\CurrentStats.csv', 'r+', encoding='utf-8') as f:
                f.seek(0)
                f.write('{:9.6},{:9.6},{:9.6},{},{}'.format(mean(out_obj), mean(out_train), std(out_train), train_means, train_stds))
 

        #with open('C:\\Users\\ideav\\Documents\\PythonWork\\FiveFactor\\starting_outputs-23{0:02}{1:02}-{2:02}{3:02}.csv'.format(ftime.month,ftime.day,ftime.hour,ftime.minute), 'w') as f:
        #    f.write('\n{0}'.format(out_eval))

        avg_out_winsorized = mean(out_train)
        std_out_winsorized = std(out_train)

        # TODO - Save the train_means, and train_stds to file CurrentStats.csv
        if Path('.\CurrentStats.csv').is_file():
            with open('.\CurrentStats.csv', 'r+', encoding='utf-8') as f:
                f.seek(0)
                f.write('{:9.6},{:9.6},{:9.6},{},{}'.format(mean(out_obj), mean(out_train), std(out_train), train_means, train_stds))
 
        # TODO - Save avg_out, avg_out_winsorized, and std_out_winsorized to file CurrentStats.csv

        gauss_normalize_input_data(train_means, train_stds, train_data, eval_data, front_stub_data)
        ###### Normalize the objective
        # de-mean and standardize output variance, i.e. so that (0,1)
        out_train_obj = [(x - avg_out_winsorized) / std_out_winsorized for x in out_train]
        out_eval_obj = [(x - avg_out_winsorized) / std_out_winsorized for x in out_eval]

        pipeline_training_data(train_data, eval_data, front_stub_data, out_train_obj, out_eval_obj)
        ###### Pipeline the objective data into the output format
        today = today.reshape((9, 1))
        out_train_obj = array(out_train_obj).reshape((DATASET_MAX, 1))
        out_eval_obj = array(out_eval_obj).reshape((EVAL_SIZE + EVAL_OVERLAP, 1))

        n_steps_in, n_steps_out = NUM_TS_LAGS, 1

        predict_train_evaluate_select_models(models, train_data, out_train_obj, eval_data, out_eval_obj,
                                             front_stub_data, today, avg_out, avg_out_winsorized, 
                                             std_out_winsorized, in_timeidx, n_steps_in, n_steps_out)


main();
