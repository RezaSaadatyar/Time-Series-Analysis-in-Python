import numpy as np
import pandas as pd
from sklearn import linear_model
from Plot_Models import plot_models


def linear_regression(data, data_lags, train_size, axs, num_sample):
    mod = linear_model.LinearRegression()
    mod.fit(data_lags[:train_size], data[:train_size])
    y_train_pred = pd.Series(mod.predict(data_lags[:train_size]))
    y_test_pred = pd.Series(mod.predict(data_lags[train_size:]))
    y_test_pred.index = np.arange(data_lags[train_size:].index[0], data_lags[train_size:].index[-1] + 1, 1, dtype='int')
    plot_models(data, y_train_pred, y_test_pred, axs, [], train_size, num_sample=num_sample, type_model='LR')



