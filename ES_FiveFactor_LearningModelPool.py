import FuturesHelper;
from FuturesHelper import FuturesHelper;
import logging;
import sys;
from pathlib import Path
import tensorflow as tf;
from tensorflow import keras;
from keras.models import Model;
from keras.models import Sequential;
from keras.models import clone_model;
from keras.callbacks import EarlyStopping;
from keras.callbacks import CSVLogger;
from keras.layers import LSTM;
from keras.layers import Activation;
from keras.layers import Dropout;
from keras.layers import Dense;
from keras.layers import RepeatVector;
from keras.layers import TimeDistributed;
from keras import backend as K;
from datetime import timedelta;

from numpy import array;
from numpy import ndarray;
from numpy import hstack;
from numpy import mean;
from numpy import var;
from numpy import exp;
from numpy import log;
from numpy import std;
from numpy import reshape;

from numpy import transpose
import numpy as np
import scipy
from scipy import stats

import datetime;

class ES_FiveFactor_LearningModelPool:

    def __init__(self):

        self.DATASET_MAX = 50000
        self.EVAL_SIZE = 10000
        self.EVAL_OVERLAP = 0
        self.NUM_TS_LAGS = 12
        self.NUM_FEATURES = 9
        self.LOSS_MAX = 3.0
        self.MODEL_POOL_SIZE = 5
        self.SUSPECT_SOLUTION_BOUNDARY = 2.0
        self.timeofday = datetime.datetime.now()
        self.MAX_OOS_RSQUARED = -3.0
        self.MIN_OOS_RSQUARED = 0.0
        self.COUNTS_AS_SIGNIFICANT = 0.8
        self.CLAMP_PEAK_OBJECTIVE = 0.6
        self.MIN_MODEL_MATURITY_TO_REPLACE = 8
        self.MIN_MODEL_MATURITY_TO_REPRODUCE = 6
        self.MAX_MINRSQUARED_TO_SPAWN = 0.005

        # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        self.LOWER_ENV_ADAPTATION_SPEED = 1.0 / 10.0
        self.max_out_of_sample_rsquared = None
        self.previous_best_model = None
        self.previous_MAX_OOS_RSQUARED = None
        self.model_was_confirmed = False

        self.SUPERVISORY_AVERAGE_LEAD = 16  # Geometric mean future price objective number of periods forward

        self.TIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
        self.DATE_FORMAT = '%Y-%m-%d'

        self._models = []

        self._incubation_pool = []

        starttime = datetime.datetime.now() +  + timedelta(days=-4*365)
        self.helper = FuturesHelper()
        self.helper.get_nth_contract(starttime, 18, "ES")
        self.helper.get_nth_contract(starttime, 18, "EC")
        self.helper.get_nth_contract(starttime, 18, "US")
        self.helper.get_nth_contract(starttime, 23, "GC")
        self.helper.get_nth_contract(starttime, 18, "NQ")
        self.helper.get_nth_contract(starttime, 52, "BTC")

        self._train_data = []
        self._out_train_obj = []
        self._eval_data = []
        self._out_eval_obj = []

        self._logger = logging.getLogger('ES_learning')
        self._logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler('ES_learning.log')
        fh.setLevel(logging.DEBUG)
        self._logger.addHandler(fh)


    def main_loop(self):
        obj_name = "ES"

        for model_place in range(self.MODEL_POOL_SIZE):
            modelfile = Path('.\{}_model_{}'.format(obj_name, model_place))
            if modelfile.is_file():
                mtime = modelfile.stat().st_mtime
                model = keras.models.load_model('.\{}_model_{}'.format(obj_name, model_place))
                model.compile(optimizer='adam', loss='mse')
                if Path('.\{}_model_loss_{}.csv'.format(obj_name, model_place)).is_file():
                    with open('.\{}_model_loss_{}.csv'.format(obj_name, model_place), encoding='utf-8') as f:
                        data = f.read().strip()
                        items = data.split(',')
                        self._logger.info('Acquired new out of sample r-squared {:9.5f}, age={}'.format(float(items[0]), items[1]))
                        r_sq = float(items[0])
                        length = int(items[1])
                        f.close()
                self._models.append([model_place, array([r_sq] * length), model])
                # self._logger.debug('Appended initial model from file to collection')
                self._logger.info('Loading model from {}_model_{} with {}'.format(obj_name, model_place, r_sq))
            else:
                # define model
                self._logger.info('Re-creating new model')
                self.initialize_new_model(model_place);

        all_data = {}

        while True:  # (((timeofday.hour == 5 and timeofday.minute > 30) or (timeofday.hour > 6)) and (timeofday.hour < 23)):
            all_data = self.load_latest_data()

            #print(all_data["OBJ"])
            out_obj = self.prepare_objective(all_data["OBJ"], all_data["ROLL"], all_data["timeidx"])
            #print(out_obj[len(out_obj)-5000:])

            inp = self.divide_input_data_into_two_parts(all_data, obj_name)

            self._train_data = inp[0]
            self._eval_data = inp[1]
            train_means = inp[2]
            train_stds = inp[3]
            in_timeidx = inp[4]

            #print(self._eval_data["ES"])

            outp = self.divide_output_data_into_two_parts(out_obj)  
            self.out_train = outp[0]
            self.out_eval = outp[1]
            ftime = datetime.datetime.now()

            #print(self.out_eval)

            avg_out_winsorized = mean(self.out_train)
            std_out_winsorized = std(self.out_train)


            if Path('.\CurrentStats.csv').is_file():
                with open('.\CurrentStats.csv', 'w', encoding='utf-8') as f:
                    f.seek(0)
                    f.write('{:9.6},{:9.6},{:9.6},{},{}'.format(mean(out_obj), avg_out_winsorized, std_out_winsorized, train_means, train_stds))
 

            self.gauss_normalize_my_input_data(self._train_data, train_means, train_stds)
            self.gauss_normalize_my_input_data(self._eval_data, train_means, train_stds)
            ###### Normalize the objective
            # de-mean and standardize output variance, i.e. so that (0,1)
            self._out_train_obj = [(x - avg_out_winsorized) / std_out_winsorized for x in self.out_train]
            self._out_eval_obj = [(x - avg_out_winsorized) / std_out_winsorized for x in self.out_eval]

            self.pipeline_input_data(self._train_data)
            self.pipeline_input_data(self._eval_data)
            ###### Pipeline the objective data into the output format
            self._out_train_obj = array(self._out_train_obj).reshape((self.DATASET_MAX, 1))
            self._out_eval_obj = array(self._out_eval_obj).reshape((self.EVAL_SIZE + self.EVAL_OVERLAP, 1))

            n_steps_in, n_steps_out = self.NUM_TS_LAGS, 1

            self.new_generation_of_models(avg_out_winsorized, 
                                             std_out_winsorized, in_timeidx[len(in_timeidx) - 1], n_steps_in, n_steps_out)



    def initialize_new_model(self, model_place):
        # define model
        #self._logger.debug('Re-creating new models')

        model = Sequential()
        model.add(LSTM(self.NUM_TS_LAGS, activation='tanh', input_shape=(self.NUM_TS_LAGS, self.NUM_FEATURES))) #"ARI")
        model.add(Dropout(0.25))                                # "integration_filter")
        model.add(Dense(self.NUM_FEATURES * self.NUM_TS_LAGS))  # "condense_into_features_and_lags")
        model.add(Activation('tanh'))                           # "MA")
        model.add(Dropout(0.25))                                # "ARIMA_filter")
        model.add(Activation("tanh"))                           # "subject")
        model.add(Dropout(0.25))                                # "subject_filter")
        model.add(Dense(self.NUM_FEATURES))                     # "patterns_1")
        model.add(Dropout(0.25))                                # "pattern_1_filter")
        model.add(Activation("tanh"))                           # "patterns_2")
        model.add(Dropout(0.25))                                # "pattern_2_filter")
        model.add(Dense(self.NUM_FEATURES))                     # "patterns_3")
        model.add(Dropout(0.25))                                # "pattern_4_filter")
        model.add(Activation("tanh"))                           # "output_patterns")
        model.add(Dense(1))                                     # "output")
        model.compile(optimizer='adam', loss='mse')
        # leaves model perfs empty initially
        self._models.append([model_place, array(0), model])
        model.summary()


    def burn_down_lstm_longshorts(self, model):
        
        return


    def parse_input_data(self, data):
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
        out_obj = []
        out_obj_roll = []

        prev_obj_symbol = None
        obj_symbol = None
        
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
        rolled = False
        lastsnapshottime = None

        for rec in str.split(data, '\n', maxsplit=-1):
            srec = str.split(rec, ',', maxsplit=-1)
            if len(srec) < 3:
                continue
            symbol = srec[0].strip()
            if symbol.startswith('NQ'): ## Data collection dependent. If it's there and not used, omit it.
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
                # self._logger.debug('Resetting for a new time period with {} {}. lastsnapshottime={}, timesnap={}'.format(datepart, timepart, lastsnapshottime, timesnap))
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
                rolled = False

            #print(esprice,ecprice,gcprice,usprice,bcprice,esprice2,ecprice2,gcprice2,usprice2,bcprice2,esspread,ecspread,gcspread,usspread,bcspread,timesnap)

            if symbol.find('ES') == 0:
                esprice, esprice2, esspread, rolled = self.parse_ES_futures_price(strtime, obj_symbol, symbol, strprice, esprice, esprice2)
            elif symbol.find('EC') == 0:
                ecprice, ecprice2, ecspread = self.parse_EC_futures_price(strtime, symbol, strprice, ecprice, ecprice2)
            elif symbol.find('GC') == 0:
                gcprice, gcprice2, gcspread = self.parse_GC_futures_price(strtime, symbol, strprice, gcprice, gcprice2)
            elif symbol.find('US') == 0:
                usprice, usprice2, usspread = self.parse_US_futures_price(strtime, symbol, strprice, usprice, usprice2)
            elif symbol.find('BTC') == 0:
                bcprice, bcprice2, bcspread = self.parse_BC_futures_price(strtime, symbol, strprice, bcprice, bcprice2)
                
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
                out_obj.append(esprice)
                out_obj_roll.append(rolled)

                if rolled and symbol.startswith('ES'):
                    obj_symbol = symbol
                    #print("Rolled:",prev_obj_symbol,obj_symbol, srec)
                    prev_obj_symbol = obj_symbol
                rolled = False

                lastsnapshottime = timesnap

        return {"ES": in_ES,
                "EC": in_EC,
                "GC": in_GC,
                "US": in_US,
                "ESS": in_ESS, "ECS": in_ECS, "GCS": in_GCS, "USS": in_USS,
                "BC": in_BC,
                "timeidx": in_timeidx,
                "OBJ": out_obj,
                "ROLL": out_obj_roll}


    def load_latest_data(self):    
        with open('5minuteBars.csv', encoding='utf-8') as f:
            data = f.read().strip()
            # self._logger.debug(data)
            return self.parse_input_data(data)


    def load_current_data(self):
        with open('CurrentBars.csv', encoding='utf-8') as f:
            data = f.read().strip()
            x = self.parse_input_data(data)
            return self.parse_input_data(data)
        #return {"ES": x["ES"],
        #        "EC": x["EC"],
        #        "GC": x["GC"],
        #        "US": x["US"],
        #        "ESS": x["ESS"], "ECS": x["ECS"], "GCS": x["GCS"], "USS": x["USS"],
        #        "BC": x["BC"],
        #        "timeidx": x["timeidx"]}
        # Needs the extra objective series?
            


    def parse_ES_futures_price(self, strtime, last_obj_symbol, symbol, strprice, esprice, esprice2):
        esspread = None
        roll = False
        ES_contracts = self.helper.Global_Futures_Dictionary["ES"]
        idx = 0;
        for x in ES_contracts._contracts:
            if str(x[0]._non_commercial_last_trading_date) > strtime:
                break
            idx = idx + 1
        if idx < len(self.helper.Global_Futures_Dictionary["ES"]._contracts):
            front_contract = ES_contracts._contracts[idx]
            second_contract = ES_contracts._contracts[idx+1]
            if symbol.find(front_contract[1]._symbol) > -1:
                esprice = float(strprice)
                if esprice2 is not None:
                    if front_contract[1]._symbol != last_obj_symbol:
                        roll = True
                    esspread = esprice - esprice2
            if symbol.find(second_contract[1]._symbol) > -1:
                esprice2 = float(strprice)
                if esprice is not None:
                    esspread = esprice - esprice2
            
        return esprice, esprice2, esspread, roll


    def parse_NQ_futures_price(self, strtime, symbol, strprice, nqprice, nqprice2):
        nqspread = None
        NQ_contracts = self.helper.Global_Futures_Dictionary["NQ"]
        idx = 0;
        for x in NQ_contracts._contracts:
            if str(x[0]._non_commercial_last_trading_date) > strtime:
                break
            idx = idx + 1
        if idx < len(self.helper.Global_Futures_Dictionary["NQ"]._contracts):
            front_contract = NQ_contracts._contracts[idx]
            second_contract = NQ_contracts._contracts[idx+1]
            if symbol.find(front_contract[1]._symbol) > -1:
                nqprice = float(strprice)
                if nqprice2 is not None:
                    nqspread = nqprice - nqprice2
            if symbol.find(second_contract[1]._symbol) > -1:
                nqprice2 = float(strprice)
                if nqprice is not None:
                    nqspread = nqprice - nqprice2
            
        return nqprice, nqprice2, nqspread


    def parse_EC_futures_price(self, strtime, symbol, strprice, ecprice, ecprice2):
        ecspread = None
        EC_contracts = self.helper.Global_Futures_Dictionary["EC"]
        idx = 0;
        for x in EC_contracts._contracts:
            if str(x[0]._non_commercial_last_trading_date) > strtime:
                break
            idx = idx + 1
        if idx < len(self.helper.Global_Futures_Dictionary["EC"]._contracts):
            front_contract = EC_contracts._contracts[idx]
            second_contract = EC_contracts._contracts[idx+1]
            if symbol.find(front_contract[1]._symbol) > -1:
                ecprice = float(strprice)
                if ecprice2 is not None:
                    ecspread = ecprice - ecprice2
            if symbol.find(second_contract[1]._symbol) > -1:
                ecprice2 = float(strprice)
                if ecprice is not None:
                    ecspread = ecprice - ecprice2
            
        return ecprice, ecprice2, ecspread


    def parse_US_futures_price(self, strtime, symbol, strprice, usprice, usprice2):
        usspread = None
        US_contracts = self.helper.Global_Futures_Dictionary["US"]
        idx = 0;
        for x in US_contracts._contracts:
            if str(x[0]._non_commercial_last_trading_date) > strtime:
                break
            idx = idx + 1
        if idx < len(self.helper.Global_Futures_Dictionary["US"]._contracts):
            front_contract = US_contracts._contracts[idx]
            second_contract = US_contracts._contracts[idx+1]
            if symbol.find(front_contract[1]._symbol) > -1:
                parts = str.split(strprice, '-')
                if len(parts) > 1:
                    usprice = float(float(parts[0]) + float(parts[1]) / 32)
                else:
                    usprice = float(strprice)
                if usprice2 is not None:
                    usspread = usprice - usprice2
            if symbol.find(second_contract[1]._symbol) > -1:
                parts = str.split(strprice, '-')
                if len(parts) > 1:
                    usprice2 = float(float(parts[0]) + float(parts[1]) / 32)
                else:
                    usprice2 = float(strprice)
                if usprice is not None:
                    usspread = usprice - usprice2
            
        return usprice, usprice2, usspread


    def parse_GC_futures_price(self, strtime, symbol, strprice, gcprice, gcprice2):
        gcspread = None
        GC_contracts = self.helper.Global_Futures_Dictionary["GC"]
        idx = 0;
        for x in GC_contracts._contracts:
            if str(x[0]._non_commercial_last_trading_date) > strtime:
                break
            idx = idx + 1
        if idx < len(self.helper.Global_Futures_Dictionary["GC"]._contracts):
            front_contract = GC_contracts._contracts[idx]
            second_contract = GC_contracts._contracts[idx+1]
            if symbol.find(front_contract[1]._symbol) > -1:
                gcprice = float(strprice)
                if gcprice2 is not None:
                    gcspread = gcprice - gcprice2
            if symbol.find(second_contract[1]._symbol) > -1:
                gcprice2 = float(strprice)
                if gcprice is not None:
                    gcspread = gcprice - gcprice2
            
        return gcprice, gcprice2, gcspread


    def parse_BC_futures_price(self, strtime, symbol, strprice, bcprice, bcprice2):
        bcspread = None
        BC_contracts = self.helper.Global_Futures_Dictionary["BTC"]
        idx = 0;
        for x in BC_contracts._contracts:
            if str(x[0]._non_commercial_last_trading_date) > strtime:
                break
            idx = idx + 1
        if idx < len(self.helper.Global_Futures_Dictionary["BTC"]._contracts):
            front_contract = BC_contracts._contracts[idx]
            second_contract = BC_contracts._contracts[idx+1]
            if symbol.find(front_contract[1]._symbol) > -1:
                bcprice = float(strprice)
                if bcprice2 is not None:
                    bcspread = bcprice - bcprice2
            if symbol.find(second_contract[1]._symbol) > -1:
                bcprice2 = float(strprice)
                if bcprice is not None:
                    bcspread = bcprice - bcprice2
            
        return bcprice, bcprice2, bcspread


    def prepare_objective(self, sequence, rolls, timestamps):
        leading = sequence[len(sequence) - 1]
        i = 0
        objective = []
        nexttime = datetime.datetime.strptime(timestamps[len(timestamps) - 1], '%Y-%m-%d %H:%M:%S')
        next_num_periods_gap = 1

        rev_seq = list(reversed(sequence))
        #self.write_list_to_file("outp_rev_sequence",rev_seq)
        rev_rolls = list(reversed(rolls))
        #self.write_list_to_file("outp_rev_rolls",rev_rolls)
        #ra = []
        for place in range(len(rev_seq)):
            price = rev_seq[place]
            roll = rev_rolls[place]
            thistime = datetime.datetime.strptime(timestamps[len(timestamps) - i - 1], '%Y-%m-%d %H:%M:%S')
            numperiods = max(1, int((nexttime - thistime).seconds / 5 / 60)) # number of 5 minute bars to skip in any gap
            if not roll:
                objective.append((leading - price) / price * 10000)
            else:
                objective.append(0.0)
            for j in range(next_num_periods_gap):
                if not roll:
                    leading = ((self.SUPERVISORY_AVERAGE_LEAD - 1.0) / self.SUPERVISORY_AVERAGE_LEAD) * leading + (
                            1.0 / self.SUPERVISORY_AVERAGE_LEAD) * price
                else:
                    leading = price
            #ra.append('rev_seq place={} i={} ts place={} thistime={} numperiods={} price={} roll={} leading={} objective={}'.format(place, i, len(timestamps) - i - 1, thistime, numperiods, price, roll, leading, (leading - price) / price * 10000))
            nexttime = thistime
            next_num_periods_gap = numperiods
            i += 1
        #self.write_list_to_file("reversal_algorithm.csv",ra)
        return list(reversed(objective))


    def divide_input_data_into_two_parts(self, all_data, objective):
        eval_ES = all_data["ES"].copy()[len(all_data["ES"]) - self.EVAL_SIZE - self.EVAL_OVERLAP - 1:len(
            all_data["ES"]) - 1]
        in_ES = all_data["ES"][len(all_data["ES"]) - self.DATASET_MAX - self.EVAL_SIZE - 2:len(
            all_data["ES"]) - self.EVAL_SIZE - 2]
        mean_in_ES = mean(all_data["ES"])
        std_in_ES = std(all_data["ES"])

        eval_ESS = all_data["ESS"].copy()[len(all_data["ESS"]) - self.EVAL_SIZE - self.EVAL_OVERLAP - 1:len(
            all_data["ESS"]) - 1]
        in_ESS = all_data["ESS"][len(all_data["ESS"]) - self.DATASET_MAX - self.EVAL_SIZE - 2:len(
            all_data["ESS"]) - self.EVAL_SIZE - 2]
        mean_in_ESS = mean(all_data["ESS"])
        std_in_ESS = std(all_data["ESS"])

        eval_EC = all_data["EC"].copy()[len(all_data["EC"]) - self.EVAL_SIZE - self.EVAL_OVERLAP - 1:len(
            all_data["EC"]) - 1]
        in_EC = all_data["EC"][len(all_data["EC"]) - self.DATASET_MAX - self.EVAL_SIZE - 2:len(
            all_data["EC"]) - self.EVAL_SIZE - 2]
        mean_in_EC = mean(all_data["EC"])
        std_in_EC = std(all_data["EC"])

        eval_ECS = all_data["ECS"].copy()[len(all_data["ECS"]) - self.EVAL_SIZE - self.EVAL_OVERLAP - 1:len(
            all_data["ECS"]) - 1]
        in_ECS = all_data["ECS"][len(all_data["ECS"]) - self.DATASET_MAX - self.EVAL_SIZE - 2:len(
            all_data["ECS"]) - self.EVAL_SIZE - 2]
        mean_in_ECS = mean(all_data["ECS"])
        std_in_ECS = std(all_data["ECS"])

        eval_GC = all_data["GC"].copy()[len(all_data["GC"]) - self.EVAL_SIZE - self.EVAL_OVERLAP - 1:len(
            all_data["GC"]) - 1]
        in_GC = all_data["GC"][len(all_data["GC"]) - self.DATASET_MAX - self.EVAL_SIZE - 2:len(
            all_data["GC"]) - self.EVAL_SIZE - 2]
        mean_in_GC = mean(all_data["GC"])
        std_in_GC = std(all_data["GC"])

        eval_GCS = all_data["GCS"].copy()[len(all_data["GCS"]) - self.EVAL_SIZE - self.EVAL_OVERLAP - 1:len(
            all_data["GCS"]) - 1]
        in_GCS = all_data["GCS"][len(all_data["GCS"]) - self.DATASET_MAX - self.EVAL_SIZE - 2:len(
            all_data["GCS"]) - self.EVAL_SIZE - 2]
        mean_in_GCS = mean(all_data["GCS"])
        std_in_GCS = std(all_data["GCS"])

        eval_US = all_data["US"].copy()[len(all_data["US"]) - self.EVAL_SIZE - self.EVAL_OVERLAP - 1:len(
            all_data["US"]) - 1]
        in_US = all_data["US"][len(all_data["US"]) - self.DATASET_MAX - self.EVAL_SIZE - 2:len(
            all_data["US"]) - self.EVAL_SIZE - 2]
        mean_in_US = mean(all_data["US"])
        std_in_US = std(all_data["US"])

        eval_USS = all_data["USS"].copy()[len(all_data["USS"]) - self.EVAL_SIZE - self.EVAL_OVERLAP - 1:len(all_data["USS"]) - 1]
        in_USS = all_data["USS"][len(all_data["USS"]) - self.DATASET_MAX - self.EVAL_SIZE - 2:len(all_data["USS"]) - self.EVAL_SIZE - 2]
        mean_in_USS = mean(all_data["USS"])
        std_in_USS = std(all_data["USS"])

        eval_BC = all_data["BC"].copy()[len(all_data["BC"]) - self.EVAL_SIZE - self.EVAL_OVERLAP - 1:len(all_data["BC"]) - 1]
        in_BC = all_data["BC"][len(all_data["BC"]) - self.DATASET_MAX - self.EVAL_SIZE - 2:len(all_data["BC"]) - self.EVAL_SIZE - 2]
        mean_in_BC = mean(all_data["BC"])
        std_in_BC = std(all_data["BC"])

        ###### Snip the time index feature to the learning set start up to the latest 
        in_timeidx = all_data["timeidx"][len(all_data[objective]) - self.DATASET_MAX:]

        train_data = {"ES": in_ES, "ESS": in_ESS, "EC": in_EC, "ECS": in_ECS, "GC": in_GC, "GCS": in_GCS, "US": in_US, "USS": in_USS, "BC": in_BC}
        eval_data = {"ES": eval_ES, "ESS": eval_ESS, "EC": eval_EC, "ECS": eval_ECS, "GC": eval_GC, "GCS": eval_GCS, "US": eval_US, "USS": eval_USS, "BC": eval_BC}
        train_means = {"ES": mean_in_ES, "ESS": mean_in_ESS, "EC": mean_in_EC, "ECS": mean_in_ECS, "GC": mean_in_GC, "GCS": mean_in_GCS, "US": mean_in_US, "USS": mean_in_USS, "BC": mean_in_BC}
        train_stds = {"ES": std_in_ES, "ESS": std_in_ESS, "EC": std_in_EC, "ECS": std_in_ECS, "GC": std_in_GC, "GCS": std_in_GCS, "US": std_in_US, "USS": std_in_USS, "BC": std_in_BC}

        return [train_data, eval_data, train_means, train_stds, in_timeidx]


    def write_list_to_file(self, filename, data):
        with open(filename, 'w') as f:
            for x in data:
                f.write('\n{0}'.format(x))



    def divide_output_data_into_two_parts(self, objective):
        ###### De-mean output series
        avg_output = mean(objective)
        objective[:] = [number - avg_output for number in objective]

        out_eval = [number - avg_output for number in objective]

        objective = self.symmetrically_bound_objective(objective, out_eval, self.CLAMP_PEAK_OBJECTIVE)

        # This snips off the oldest observations from objective, and puts the front extent into the eval set
        objective = objective[len(objective) - self.DATASET_MAX - self.NUM_TS_LAGS - self.EVAL_SIZE - 1:len(objective) - self.NUM_TS_LAGS - self.EVAL_SIZE - 1]
    
        out_eval = objective.copy()[len(objective) - self.DATASET_MAX - self.NUM_TS_LAGS - self.EVAL_SIZE - self.EVAL_OVERLAP:len(objective) - self.DATASET_MAX - self.NUM_TS_LAGS]
    
        return [objective, out_eval]


    def symmetrically_bound_objective(self, sequence, alt_sequence, proportion):
        # find the min and max. if both have the same sign, print message and return unmodified
        mymin = min(sequence)
        mymax = max(sequence)
        if mymin * mymax > 0:
            self._logger.info("Sequence is all the same sign, ignoring bounding for this sequence")
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
    #    with open('C:\\Users\\ideav\\Documents\\PythonWork\\FiveFactor\\sequence.csv', 'w') as f:
    #        f.write('\n{0}'.format(sequence))
        return sequence


    # mutates input_data
    def gauss_normalize_my_input_data(self, input_data, train_means, train_stds):
        input_data["ES"] = [(x - train_means["ES"]) / train_stds["ES"] for x in input_data["ES"]]
        input_data["EC"] = [(x - train_means["EC"]) / train_stds["EC"] for x in input_data["EC"]]
        input_data["GC"] = [(x - train_means["GC"]) / train_stds["GC"] for x in input_data["GC"]]
        input_data["US"] = [(x - train_means["US"]) / train_stds["US"] for x in input_data["US"]]
        input_data["BC"] = [(x - train_means["BC"]) / train_stds["BC"] for x in input_data["BC"]]
        input_data["ESS"] = [(x - train_means["ESS"]) / train_stds["ESS"] for x in input_data["ESS"]]
        input_data["ECS"] = [(x - train_means["ECS"]) / train_stds["ECS"] for x in input_data["ECS"]]
        input_data["GCS"] = [(x - train_means["GCS"]) / train_stds["GCS"] for x in input_data["GCS"]]
        input_data["USS"] = [(x - train_means["USS"]) / train_stds["USS"] for x in input_data["USS"]]


    # mutates input_data
    def pipeline_input_data(self, input_data):
        # convert to [rows, columns] structure
        ###### Pipeline the training data into the input format
        input_data["ES"] = array(input_data["ES"]).reshape((len(input_data["ES"]), 1))
        input_data["EC"] = array(input_data["EC"]).reshape((len(input_data["EC"]), 1))
        input_data["GC"] = array(input_data["GC"]).reshape((len(input_data["GC"]), 1))
        input_data["US"] = array(input_data["US"]).reshape((len(input_data["US"]), 1))
        input_data["BC"] = array(input_data["BC"]).reshape((len(input_data["BC"]), 1))
        input_data["ESS"] = array(input_data["ESS"]).reshape((len(input_data["ESS"]), 1))
        input_data["ECS"] = array(input_data["ECS"]).reshape((len(input_data["ECS"]), 1))
        input_data["GCS"] = array(input_data["GCS"]).reshape((len(input_data["GCS"]), 1))
        input_data["USS"] = array(input_data["USS"]).reshape((len(input_data["USS"]), 1))



    # split a multivariate sequence into samples
    def split_sequences(self, sequences, n_steps_in, n_steps_out):
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


    def model_fitness_score(self, eval_history):
        #return mean(eval_history[1]) - std(eval_history[1]) / 4 + log(eval_history[1].size) / 1000
        # desensitize the short end of the longevity virtue but keep about the same magnitude at low scale
        return mean(eval_history[1]) - std(eval_history[1]) / 4 + log(eval_history[1].size + 99) / 5000




    def new_generation_of_models(self, avg_out_obj_winsorized, std_out_obj_winsorized, data_time,
                                             n_steps_in, n_steps_out):
        runtime = datetime.datetime.now()
        self._logger.info('Run time={} Data time={}'.format(runtime, data_time))

        dataset = hstack((self._train_data["ES"], self._train_data["EC"], self._train_data["GC"], self._train_data["US"],
                          self._train_data["BC"], self._train_data["ESS"], self._train_data["ECS"], self._train_data["GCS"],
                          self._train_data["USS"], 
                          self._out_train_obj))
        # covert into input/output
        X, y = self.split_sequences(dataset, n_steps_in, n_steps_out)

        evalset = hstack((self._eval_data["ES"], self._eval_data["EC"], self._eval_data["GC"], self._eval_data["US"],
                          self._eval_data["BC"], self._eval_data["ESS"], self._eval_data["ECS"], self._eval_data["GCS"],
                          self._eval_data["USS"],
                          self._out_eval_obj))
        evalX, evaly = self.split_sequences(evalset, n_steps_in, n_steps_out)

        n_features = X.shape[2]
        #self._logger.debug(X.shape)

        ###### Model pool.  Have the model pool do the interaction, managing the genetic algo.
        for model_place in range(len(self._models)):

            model = self._models[model_place][2]
            # predict using the before model (call this the parent)
            predictions, current_r_squared, child_r_squared, current_model_score = self.before_and_after_models(self._models, X, y, model, model_place, evalX, evaly)

            # Max/min of all the model means
            r_values = [mean(x[1]) for x in self._models]
            r_values.sort()
            if len(self._models) > 0:
                alt_incubation_dropout_criterion = r_values[1]
                others = [y for y in self._models if y[0] != model_place]
                MAX_OOS_RSQUARED = max(mean(x[1]) for x in others)
                MIN_OOS_RSQUARED = min(r_values)

            rollback_model = clone_model(model)
            
            # Run the genetic fitness criteria. Prodigies stand to replace a line if sufficiently awesome.
            # If not, then if the child r-squared is good enough, it inherits, otherwise the line reverts
            # to the parent.
            child_should_incubate_a_new_line = False

            # Mature enough to spawn, and child is a prodigy
            # To minimize impoverishment spinning off badly needed children. Only spawn if the line would not suffer too badly. Try a fixed zero bound.
            if self._models[model_place][1].size >= self.MIN_MODEL_MATURITY_TO_REPRODUCE and child_r_squared > (MAX_OOS_RSQUARED) and current_r_squared > 0.0 and MIN_OOS_RSQUARED < self.MAX_MINRSQUARED_TO_SPAWN:
                inc_place = len(self._incubation_pool)
                self._incubation_pool.append([inc_place, array(child_r_squared), clone_model(model)])
                self._incubation_pool[inc_place][2].compile(optimizer='adam', loss='mse')
                child_should_incubate_a_new_line = True
            
            secondary_inheritance_requirement = mean(self._models[model_place][1]) + std(self._models[model_place][1])
            if not child_should_incubate_a_new_line:
                # The child replaces the parent unless it spawned to incubate another genetic line, or
                # unless it was inferior, i.e. < min(r, (historical mean r) + 1 std dev)
                # Also cannot inherit with < 80% of any positive current value, no matter what - experimental
                # Failed to inherit:
                if (child_r_squared < current_r_squared and child_r_squared <= secondary_inheritance_requirement) or (child_r_squared < current_r_squared * 0.80 and current_r_squared > 0.0):
                    # rollback
                    try:
                        self._models[model_place][1] = np.append(self._models[model_place][1], current_r_squared)
                        self._models[model_place][2] = rollback_model
                        self._models[model_place][2].compile(optimizer='adam', loss='mse')
                    except Exception as ex:
                        self._logger.debug('Exception thrown rolling back model:{} : {}'.format(type(ex), ex.args))
                        self._logger.debug('Rolling back pool model {}'.format(model_place))
                        self._logger.debug('Was {}'.format(self._models[model_place]))

                else:
                    # inherit
                    self._models[model_place][1] = np.append(self._models[model_place][1], child_r_squared)
                    rollback_model = None
            else:
                # rollback to parent, if it spawned a prodigy to incubate
                try:
                    self._models[model_place][1] = np.append(self._models[model_place][1], current_r_squared)
                    self._models[model_place][2] = rollback_model
                    self._models[model_place][2].compile(optimizer='adam', loss='mse')
                except:
                    self._logger.debug('Rolling back pool model {}'.format(model_place))
                    self._logger.debug('Was {}'.format(self._models[model_place]))
                rollback_model = None

            self._logger.info("Line {} avg r {:6.5f}[{:}]: r={:6.5f}: Child got {:6.5f}".format(model_place, mean(self._models[model_place][1]), self._models[model_place][1].size, current_r_squared, child_r_squared))
            if child_should_incubate_a_new_line:
                self._logger.info('Incubate new line. Rolling mean still  {:6.5f}, parent fitness={:6.5f}'. \
                      format(mean(self._models[model_place][1]), self.model_fitness_score(self._models[model_place])))
            elif (child_r_squared < current_r_squared and child_r_squared <= secondary_inheritance_requirement) or (child_r_squared < current_r_squared * 0.80 and current_r_squared > 0.0):
                self._logger.info('Child didn\'t inherit. Rolling mean now {:6.5f}, new fitness={:6.5f}.' . \
                      format(mean(self._models[model_place][1]), self.model_fitness_score(self._models[model_place])))
                #self._logger.debug(self._models[model_place][1])
            else:
                self._logger.info('Child inherited line. Rolling mean now {:6.5f}, new fitness={:6.5f}'. \
                      format(mean(self._models[model_place][1]), self.model_fitness_score(self._models[model_place])))
                #self._logger.debug(self._models[model_place][1])

            if not child_should_incubate_a_new_line:
                if Path('.\ES_model_loss_{}.csv'.format(model_place)).is_file():
                    with open('.\ES_model_loss_{}.csv'.format(model_place), "w") as f:
                        f.seek(0)
                        f.write('{:9.6},{:d}'.format(mean(self._models[model_place][1]),
                                                   self._models[model_place][1].size))
                        f.close()
                else:
                    with open('.\ES_model_loss_{}.csv'.format(model_place), "w") as f:
                        f.seek(0)
                        f.write('{:9.6},{:d}'.format(mean(self._models[model_place][1]),
                                                  self._models[model_place][1].size))
                        f.close()
                model.save('.\ES_model_{}'.format(model_place), save_format='h5')

            # train, evaluate, select

        ######### Incubation pool breeding, drop them quickly if they don't pan out.
        incubation_graduations = []
        fitnesses = [self.model_fitness_score(x) for x in self._models]
        least_fit_place = fitnesses.index(min(x for x in fitnesses))

        if len(self._incubation_pool) > 0:
            self._logger.info('********** Incubator **********')
        for loop in range(len(self._incubation_pool)):
            baby_model = self._incubation_pool[loop][2]
            predictions, current_r_squared, child_r_squared, current_model_score = self.before_and_after_models(self._incubation_pool, X, y, baby_model, loop, evalX, evaly)

            rollback_model = clone_model(baby_model)

            inc_pool_mode_rvals = self._incubation_pool[loop][1]
            secondary_inheritance_requirement = mean(inc_pool_mode_rvals) + std(inc_pool_mode_rvals)
            if (child_r_squared < current_r_squared and child_r_squared <= secondary_inheritance_requirement) or (child_r_squared < current_r_squared * 0.80 and current_r_squared > 0.0):
                # rollback
                self._incubation_pool[loop][1] = np.append(inc_pool_mode_rvals, current_r_squared)
                self._incubation_pool[loop][2] = rollback_model
                self._incubation_pool[loop][2].compile(optimizer='adam', loss='mse')
            else:
                self._incubation_pool[loop][1] = np.append(inc_pool_mode_rvals, child_r_squared)
                rollback_model = None
                # inherit - save 
            if Path('.\incubating_model_loss_{}.csv'.format(loop)).is_file():
                with open('.\incubating_model_loss_{}.csv'.format(loop), "w") as f:
                    f.seek(0)
                    f.write('{:9.6},{:d}'.format(mean(inc_pool_mode_rvals),inc_pool_mode_rvals.size))
                    f.close()
            else:
                with open('.\incubating_model_loss_{}.csv'.format(loop), "w") as f:
                    f.seek(0)
                    f.write('{:9.6},{:d}'.format(mean(inc_pool_mode_rvals),inc_pool_mode_rvals.size))
                    f.close()
            baby_model.save('.\incubating_model_{}'.format(loop), save_format='h5')
            
            # Report what happened above 
            if (child_r_squared < current_r_squared and child_r_squared <= secondary_inheritance_requirement) or (child_r_squared < current_r_squared * 0.80 and current_r_squared > 0.0):
                self._logger.info('Incubating line {} child didn\'t inherit, {} old. Rolling mean now {:6.5f}, new fitness={:6.5f}'. \
                      format(loop, len(self._incubation_pool[loop][1]), mean(self._incubation_pool[loop][1]), self.model_fitness_score(self._incubation_pool[loop])))
            else:
                self._logger.info('Incubating line {} child inherited line, {} old. Rolling mean now {:6.5f}, new fitness={:6.5f}'. \
                      format(loop, len(self._incubation_pool[loop][1]), mean(self._incubation_pool[loop][1]), self.model_fitness_score(self._incubation_pool[loop])))

            # Finalize items for this incubator operation loop
            if self._incubation_pool[loop][1].size >= self.MIN_MODEL_MATURITY_TO_REPLACE and mean(self._incubation_pool[loop][1]) > alt_incubation_dropout_criterion and self.model_fitness_score(self._models[least_fit_place]) <= self.model_fitness_score(self._incubation_pool[loop]):
                incubation_graduations.append(self._incubation_pool[loop])
                self._incubation_pool[loop] = None
            elif mean(self._incubation_pool[loop][1]) <= alt_incubation_dropout_criterion or \
                (self._incubation_pool[loop][1].size >= self.MIN_MODEL_MATURITY_TO_REPLACE and self.model_fitness_score(self._models[least_fit_place]) > self.model_fitness_score(self._incubation_pool[loop])):
                self._logger.info('Dropping incubated line {} due to inferior performance r-sq={:6.5f}, fitness={:6.5f}'. \
                    format(loop, mean(self._incubation_pool[loop][1]), self.model_fitness_score(self._incubation_pool[loop])))
                self._incubation_pool[loop] = None
                
        if len(self._incubation_pool) > 0:
            self._logger.info('********** Incubator **********')

        if len(incubation_graduations) > 0:
            self._logger.info('***** Incubated replacements *****')
        for loop in range(len(incubation_graduations)):            
            self._models[least_fit_place] = incubation_graduations[loop]
            self._logger.info('Replacing line {} with incubated line of r-sq={:6.5f}, new fitness={:6.5f}'. \
                format(least_fit_place, mean(self._models[least_fit_place][1]), self.model_fitness_score(self._models[least_fit_place])))
            if Path('.\ES_model_loss_{}.csv'.format(least_fit_place)).is_file():
                with open('.\ES_model_loss_{}.csv'.format(least_fit_place), "w") as f:
                    f.seek(0)
                    f.write('{:9.6},{:d}'.format(mean(self._models[least_fit_place][1]),len(self._models[least_fit_place][1])))
                    f.close
            else:
                with open('.\ES_model_loss_{}.csv'.format(least_fit_place), "w") as f:
                    f.seek(0)
                    f.write('{:9.6},{:d}'.format(mean(self._models[least_fit_place][1]),len(self._models[least_fit_place][1])))
                    f.close()
            self._models[least_fit_place][2].save('.\ES_model_{}'.format(least_fit_place), save_format='h5')
        if len(incubation_graduations) > 0:
            self._logger.info('***** Incubated replacements *****')

        incubation_graduations = []
        self._incubation_pool = [x for x in self._incubation_pool if x is not None]

        K.clear_session()

        timeofday = datetime.datetime.now()


    def before_and_after_models(self, collection, trainX, trainy, model, model_place, evalX, evaly):
        predictions = model.predict(evalX, batch_size=1000, verbose=0)

        slope, intercept, current_r_value, p_value, std_err = scipy.stats.linregress(
            reshape(evaly, (1,-1)), reshape(predictions, (1,-1)))
        current_r_squared = current_r_value*abs(current_r_value)

        # Train
        # Fit the next candidate model, call it the child, and find its out of sample r-squared
        stopping_rule = [
            EarlyStopping(
                # Stop training when `loss` is no longer improving
                monitor="loss",
                # "no longer improving" being defined as "no better than 1e-2 less"
                min_delta=1e-2,
                # baseline=250,
                # "no longer improving" being further defined as "for at least N epochs"
                patience=50,
                verbose=0,
            )
        ]

        result = model.fit(trainX, trainy, epochs=2000, batch_size=1000, validation_data=(evalX, evaly), verbose=0,
                           callbacks=stopping_rule)

        # last_price = train_data["GC"][len(train_data["GC"])-1]
        # logger.write('{},{},{},{}\n'.format(in_timeidx[len(in_timeidx) - 1],
        #                                       result.history['loss'][len(result.history['loss']) - 1],
        #                                       (yhat[0][0] - avg_out_obj_winsorized) * std_out_obj_winsorized,
        #                                       GC_last_price))

        # Evaluate child model for inheritance
        predictions = model.predict(evalX, batch_size=1000, verbose=0)
        slope, intercept, child_r_value, p_value, std_err = scipy.stats.linregress(
            reshape(evaly, (1,-1)), reshape(predictions, (1,-1)))
        child_r_squared = child_r_value*abs(child_r_value)

        current_model_score = 0;
#        self._logger.debug(type(models[model_place][1]))
        if collection[model_place][1].size > 0:
            current_model_score = self.model_fitness_score(collection[model_place])

        return predictions, current_r_squared, child_r_squared, current_model_score



    def calculate_current_prediction(self):
        obj_name = "ES"

        train_means = {}
        train_stds = {}
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
                        train_means[item_parts[0].strip()] = float(item_parts[1])
                    else:
                        train_stds[item_parts[0].strip()] = float(item_parts[1])        

        models_local = []
        model_rs = [0] * self.MODEL_POOL_SIZE
        for model_place in range(self.MODEL_POOL_SIZE):
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
                        
                models_local.append([model_place, array([r_sq] * length), model])
                # self._logger.debug('Appended initial model from file to collection')
                # self._logger.info('Loading models_local from {}_model_{} with {}'.format(obj_name, model_place, r_sq))
            with open(str.format('ES_model_loss_{}.csv', model_place), encoding='utf-8') as fLoss:
                try:
                    data = fLoss.read().strip()
                    items = str.split(data, ',', maxsplit=-1) # assumption: one line in loss file
                    model_rs[model_place] = float(items[0])
                except:
                    continue

        runtime = datetime.datetime.now()
        current_data = self.load_current_data()

        #print(avg_out_winsorized, std_out_winsorized)

        self.gauss_normalize_my_input_data(current_data, train_means, train_stds)
        
        self.pipeline_input_data(current_data)

        #print(current_data["ES"])
        current_data = hstack((current_data["ES"],
                          current_data["EC"],
                          current_data["GC"],
                          current_data["US"],
                          current_data["BC"],
                          current_data["ESS"],
                          current_data["ECS"],
                          current_data["GCS"],
                          current_data["USS"],
                          [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]))
        curr_x, useless = self.split_sequences(current_data, self.NUM_TS_LAGS, 1)

        with open('PythonOutput_ES_FiveFactor_FrontOnly.csv', "w") as fResults:
            fResults.seek(0)
            for loop in range(self.MODEL_POOL_SIZE):
                # horizontally stack columns

                model = models_local[loop][2]
                model_place = loop

                # n_features is the number of series added to x_input. i.e. self.NUM_FEATURES features
                yhat = model.predict(curr_x, verbose=0)
                print(yhat)
                prediction = (yhat[0][0] + avg_out_winsorized) * std_out_winsorized + avg_out

                print("model {} predictive result {} = {:8.4f}, r_squared={:7.5f}".format(model_place, yhat, prediction, model_rs[model_place]))
                
                fResults.write('{:9.6f},{:7.4f}\n'.format(prediction, model_rs[model_place]))

            fResults.close()


