import numpy as np
import pandas as pd
from sklearn import tree
from Plot_Models import plot_models


def tree_decision_regression(data, data_lags, train_size, axs, max_depth, num_sample):
    mod = tree.DecisionTreeRegressor(max_depth=max_depth, random_state=0)

    mod.fit(data_lags[:train_size], data[:train_size])
    y_train_pred = pd.Series(mod.predict(data_lags[:train_size]))
    y_test_pred = pd.Series(mod.predict(data_lags[train_size:]))
    y_test_pred.index = np.arange(data_lags[train_size:].index[0], data_lags[train_size:].index[-1] + 1, 1, dtype='int')
    plot_models(data, y_train_pred, y_test_pred, axs, [], train_size, num_sample=num_sample, type_model='DT')

