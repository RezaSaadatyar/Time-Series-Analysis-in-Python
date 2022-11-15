import numpy as np
import pandas as pd
from statsmodels import tsa
from Plot_Models import plot_models


def ar_model(data, train_size, axs, n_lags, num_sample):
    mod = tsa.ar_model.AutoReg(data[:train_size], lags=n_lags).fit()
    y_train_pred = pd.Series(mod.fittedvalues)                         # train
    y_test_pred = pd.Series(mod.model.predict(mod.params, start=train_size, end=len(data)-1))  # For predict Future: end - start samples
    plot_models(data, y_train_pred, y_test_pred, axs, [], train_size, num_sample=num_sample, type_model='AR')
