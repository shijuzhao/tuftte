"""
This file contains several prediction methods.
"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from pmdarima import auto_arima
import numpy as np

MAX = "MAX"
AVG = "AVG"
AVGplus = "AVG+STD"
LR = "LR"
ARIMA = "ARIMA"
RF = "RF"
ALLMETHOD = [MAX, AVGplus, LR, RF]

def predict_traffic_matrix(train_tms, test_tms, hist_len=12, method=MAX):
    """
    Predict traffic matrix.
    Note that here a traffic matrix is a list of n*(n-1) demands, where n is the number of nodes.

    parameters:
        train_tms(list of `list of `float``): a series of historical traffic matrices for training;
        test_tms(list of `list of `float``): traffic matrices for testing;
        hist_len(int): the number of historical traffic matrices of a instance;
        method(str): the prediction method.

    return:
        predicted_tms(list of `list of `float``): traffic matrices predicted by test_tms.
    """
    num_test_tms = len(test_tms) - hist_len
    num_pairs = len(test_tms[0])
    history_for_test_tms = [test_tms[i:i+hist_len] for i in range(num_test_tms)]
    
    if method == MAX:
        predicted_tms = [np.max(history, axis=0) for history in history_for_test_tms]

    elif method == AVG:
        predicted_tms = [np.mean(history, axis=0) for history in history_for_test_tms]

    elif method == AVGplus:
        predicted_tms = [np.mean(history, axis=0) + 2*np.std(history, axis=0) for history in history_for_test_tms]
    
    elif method == LR:
        history_for_train_tms = [np.array(train_tms[i:i+hist_len]).transpose() for i in range(len(train_tms)-hist_len)]
        history_for_test_tms = [np.array(test_tms[i:i+hist_len]).transpose() for i in range(num_test_tms)]
        instance = [[history[i] for history in history_for_train_tms] for i in range(num_pairs)]
        label = np.array(train_tms[hist_len:]).transpose()
        predicted_tms = np.zeros((num_pairs, num_test_tms))
        model = LinearRegression()
        for i in range(num_pairs):
            model.fit(instance[i], label[i])
            predicted_tms[i] = model.predict([history_for_test_tms[j][i] for j in range(num_test_tms)])

        predicted_tms = predicted_tms.transpose()

    elif method == ARIMA:
        predicted_tms = np.zeros((num_test_tms, num_pairs))
        for i in range(num_pairs):
            instance = [tm[i] for tm in train_tms] + [test_tms[j][i] for j in range(hist_len)]
            model = auto_arima(instance, start_p=4, max_d=1, max_p=9)
            for j in range(hist_len, num_test_tms):
                predicted_tms[j-hist_len][i] = model.predict(n_periods=1)
                model.update(test_tms[j][i])

    elif method == RF:
        history_for_train_tms = [np.array(train_tms[i:i+hist_len]).transpose() for i in range(len(train_tms)-hist_len)]
        history_for_test_tms = [np.array(test_tms[i:i+hist_len]).transpose() for i in range(num_test_tms)]
        instance = [[history[i] for history in history_for_train_tms] for i in range(num_pairs)]
        label = np.array(train_tms[hist_len:]).transpose()
        predicted_tms = np.zeros((num_pairs, num_test_tms))
        model = RandomForestRegressor(n_estimators=10)
        for i in range(num_pairs):
            model.fit(instance[i], label[i])
            predicted_tms[i] = model.predict([history_for_test_tms[j][i] for j in range(num_test_tms)])

        predicted_tms = predicted_tms.transpose()

    else:
        assert False, f"Method {method} is not defined!"

    return predicted_tms