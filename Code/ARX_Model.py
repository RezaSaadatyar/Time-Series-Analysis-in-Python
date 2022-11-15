import copy
import numpy as np
import pandas as pd
from Least_Squares import lest_squares
from Plot_Models import plot_models


def arx(data, data_lags, train_size, axs, mu, num_sample):
    nLags = data_lags.shape[1]
    data_lags1 = copy.copy(data_lags)
    alpha = np.zeros(nLags + 1)
    data_lags1.insert(0, "0", data, True)
    data_lags1.columns = list(range(nLags + 1))
    y_train_pred = []
    for i in range(train_size):
        e = data.iloc[i] - alpha.dot(data_lags1.iloc[i])
        alpha = alpha + mu * data_lags1.iloc[i] * e
        y_train_pred.append(alpha.dot(data_lags1.iloc[i]))
    y_train_pred = pd.DataFrame(y_train_pred)
    y_test_pred = np.sum(alpha * data_lags1[train_size:], axis=1)
    plot_models(data, y_train_pred, y_test_pred, axs, [], train_size, num_sample, type_model='ARX')

