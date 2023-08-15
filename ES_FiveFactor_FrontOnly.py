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

tf.get_logger().setLevel('ERROR')

DATASET_MAX = 25000
EVAL_SIZE = 3000
EVAL_OVERLAP = 0
NUM_TS_LAGS = 12
LOSS_MAX = 3.0
MODEL_POOL_SIZE = 5
SUSPECT_SOLUTION_BOUNDARY = 2.0
timeofday = datetime.now()
MAX_OOS_RSQUARED = -3.0
MIN_OOS_RSQUARED = 0.0
COUNTS_AS_SIGNIFICANT = 0.8
CLAMP_PEAK_OBJECTIVE = 0.7
NUM_MODELS = 5
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
model_rs = [0] * NUM_MODELS

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

def gauss_normalize_input_data(means, stds, current_data):
    ###### Normalize the input training series and the most current observation.
    # de-mean and standardize inputs variance, i.e. so that (0,1)
    current_data["ES"] = [(x - means["ES"]) / stds["ES"] for x in
                                  current_data["ES"]]
    current_data["EC"] = [(x - means["EC"]) / stds["EC"] for x in
                                  current_data["EC"]]
    current_data["GC"] = [(x - means["GC"]) / stds["GC"] for x in
                                  current_data["GC"]]
    current_data["US"] = [(x - means["US"]) / stds["US"] for x in
                                  current_data["US"]]
    current_data["BC"] = [(x - means["BC"]) / stds["BC"] for x in
                                  current_data["BC"]]
    current_data["ESS"] = [(x - means["ESS"]) / stds["ESS"] for x in
                                   current_data["ESS"]]
    current_data["ECS"] = [(x - means["ECS"]) / stds["ECS"] for x in
                                   current_data["ECS"]]
    current_data["GCS"] = [(x - means["GCS"]) / stds["GCS"] for x in
                                   current_data["GCS"]]
    current_data["USS"] = [(x - means["USS"]) / stds["USS"] for x in
                                   current_data["USS"]]

def pipeline_training_data(stub_data):
    # convert to [rows, columns] structure
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
        print("New model. Elememnt has ", len(models[1]), "elements")

def load_current_data():
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
    with open('CurrentBars.csv', encoding='utf-8') as f:
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
        currentdataline = 0
        for rec in str.split(data, '\n', maxsplit=-1):
            srec = str.split(rec, ',', maxsplit=-1)
            if len(srec) < 3:
                continue
            currentdataline = currentdataline + 1
            #print('symbol={0}, timestamp={1}, price={2}'.format(srec[0].strip(), srec[1].strip(), srec[2].strip()))
            symbol = srec[0].strip()
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
                #print('Resetting for a new time period with {} {}. lastsnapshottime={}, timesnap={}'.format(datepart, timepart, lastsnapshottime, timesnap))
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

            if symbol.find('ES') == 0 and esprice is None:
                esprice = float(strprice)
            elif symbol.find('ES') == 0 and esprice is not None:
                esprice2 = float(strprice)
                esspread = esprice - esprice2
            elif symbol.find('EC') == 0 and ecprice is None:
                ecprice = float(strprice)
            elif symbol.find('EC') == 0 and ecprice is not None:
                ecprice2 = float(strprice)
                ecspread = ecprice - ecprice2
            elif symbol.find('GC') == 0 and gcprice is None:
                gcprice = float(strprice)
            elif symbol.find('GC') == 0 and gcprice is not None:
                gcprice2 = float(strprice)
                gcspread = gcprice - gcprice2
            elif symbol.find('US') == 0 and usprice is None:
                usprice = float(strprice)
            elif symbol.find('US') == 0 and usprice is not None:
                usprice2 = float(strprice)
                usspread = usprice - usprice2
            elif symbol.find('BTC') == 0 and bcprice is None:
                bcprice = float(strprice)
            elif symbol.find('BTC') == 0 and bcprice is not None:
                bcprice2 = float(strprice)
                bcspread = bcprice - bcprice2
            #print("parsing ", symbol, esprice, esprice2, esspread, ecprice, ecprice2, ecspread, gcprice, gcprice2, gcspread, usprice, usprice2, usspread, bcprice, bcprice2, bcspread, timesnap)

            if symbol.find('NQ') == 0 or bcspread is not None:
                continue;
            
            if timesnap is not None and esprice is not None and esprice > 0 and ecprice is not None and ecprice > 0 \
                    and esprice2 is not None and esprice2 > 0 and ecprice2 is not None and ecprice2 > 0 \
                    and gcprice is not None and gcprice > 0 and usprice is not None and usprice > 0 \
                    and gcprice2 is not None and gcprice2 > 0 and usprice2 is not None and usprice2 > 0 \
                    and bcprice is not None and bcprice > 0 \
                    and esspread is not None and ecspread is not None and gcspread is not None and usspread is not None:
                in_ES.append(esprice)
                #print("Appending just ES=", esprice, " as of ", timesnap, " from file line ", currentdataline)
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

    return {"ES": in_ES, "EC": in_EC, "GC": in_GC, "US": in_US, "ESS": in_ESS, "ECS": in_ECS, "GCS": in_GCS,
            "USS": in_USS, "BC": in_BC, "timeidx": in_timeidx}



#logger = open('C:\\Users\\ideav\\Documents\\PythonWork\\FiveFactor\\ES_log_{}{:02}{:02}{:02}{:02}{:02}.csv'.format(timeofday.year,timeofday.month,timeofday.day,timeofday.hour,timeofday.minute,timeofday.second),'w')
#logger.write('time,loss,prediction,price\n')

def predict_using_model(models, stub_data, today,
                                         avg_out_obj, avg_out_obj_winsorized, std_out_obj_winsorized, data_time,
                                         n_steps_in, n_steps_out):
    runtime = datetime.now()
    #print('Run time={} Data time={}\nTrend={:}, Skew={:}'.format(runtime, data_time, avg_out_obj, avg_out_obj_winsorized))

    ###### Model pool.  Have the model pool do the iteration, managing the genetic algo.
    #print("models length = ", len(models), )
    with open('PythonOutput_ES_FiveFactor_FrontOnly.csv', "w") as fResults:
        fResults.seek(0)
        for loop in range(MODEL_POOL_SIZE):
            # horizontally stack columns

            if loop < len(models):
                model = models[loop][2]
                model_place = loop
            else:
                model_place = 0

            # predict
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

            # n_features is the number of series added to x_input. i.e. 9 features
            #print(x_input)
            x_input = x_input.reshape((1, n_steps_in, 9))
            yhat = model.predict(x_input, verbose=0)
            prediction = (yhat[0][0] + avg_out_obj_winsorized) * std_out_obj_winsorized

            print("model {} predictive result = {:8.4f}, r_squared={:7.5f}".format(model_place, prediction, model_rs[model_place]))
            
            fResults.write('{:9.6f},{:7.4f}\n'.format(prediction, model_rs[model_place]))

        fResults.close()

    K.clear_session()

    timeofday = datetime.now()


###### main start
def main():
    models = []

    obj_name = "ES";

    # TODO read in from the training process means, stds to transform inputs
    # and avg_out_winsorized and std_out_winsorized, to untransform outputs
    means = {}
    stds = {}
    with open('CurrentStats.csv', encoding='utf-8') as fStats:
        data = fStats.read().strip()
        for rec in str.split(data, '\n', maxsplit=-1):
            srec = str.split(rec, ',', maxsplit=-1)
            avg_out = float(srec[0]) # Trend
            avg_out_winsorized = float(srec[1]) # Skewness from trend
            std_out_winsorized = float(srec[2]) # Second moment from trend
            countfeatures = 0
            for item in srec[3:]:
                item_parts = str.split(item, ':', maxsplit=-1)
                if item_parts[0].find('{') > -1:
                    item_parts[0] = item_parts[0][item_parts[0].find('{')+1:]
                if item_parts[1].find('}') > -1:
                    item_parts[1] = item_parts[1][0:item_parts[1].find('}')]
                item_parts[0] = item_parts[0].replace('\'', '')
                countfeatures = countfeatures + 1
                if countfeatures <= 9:
                    means[item_parts[0].strip()] = float(item_parts[1])
                else:
                    stds[item_parts[0].strip()] = float(item_parts[1])        

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
                    #print('Acquired new out of sample r-squared {:9.5f}, age={}'.format(float(items[0]), items[1]))
                    r_sq = float(items[0])
                    length = int(items[1])
            models.append([model_place, array([r_sq] * length), model])
            # print('Appended initial model from file to collection')
            #print('Loading model from {}_model_{} with '.format(obj_name, model_place), r_sq)
        else:
                # define model
            print('Re-creating new model')
            initialize_new_model(models, NUM_TS_LAGS, 2);

        with open(str.format('ES_model_loss_{}.csv', model_place), encoding='utf-8') as fLoss:
            data = fLoss.read().strip()
            items = str.split(data, ',', maxsplit=-1) # assumption: one line in loss file
            model_rs[model_place] = float(items[0])

    current_data = load_current_data()

    today = array([current_data["ES"][len(current_data["ES"])-1],current_data["EC"][len(current_data["EC"])-1],current_data["GC"][len(current_data["GC"])-1],current_data["US"][len(current_data["US"])-1],current_data["BC"][len(current_data["BC"])-1],current_data["ESS"][len(current_data["ESS"])-1],current_data["ECS"][len(current_data["ECS"])-1],current_data["GCS"][len(current_data["GCS"])-1],current_data["USS"][len(current_data["USS"])-1]])
























    gauss_normalize_input_data(means, stds, current_data)
    ###### Normalize the objective
    # de-mean and standardize output variance, i.e. so that (0,1)



    pipeline_training_data(current_data)
    ###### Pipeline the objective data into the output format
    today = today.reshape((9, 1))
    n_steps_in, n_steps_out = NUM_TS_LAGS, 1




    predict_using_model(models, current_data, today, avg_out, avg_out_winsorized, std_out_winsorized,
                        current_data["timeidx"][len(current_data["timeidx"])-1], n_steps_in, n_steps_out)
    

main();
