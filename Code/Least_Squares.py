import numpy as np
from Plot_Models import plot_models


def lest_squares(data, data_lags, train_size, axs, num_sample):
    pinv = np.linalg.pinv(data_lags[:train_size])
    alpha = pinv.dot(data[:train_size])
    if alpha.ndim > 1:
        alpha = alpha.flatten()
    y_train_pred = np.sum(alpha * data_lags[:train_size], axis=1)
    y_test_pred = np.sum(alpha * data_lags[train_size:], axis=1)
    plot_models(data, y_train_pred, y_test_pred, axs, [], train_size, num_sample, type_model='LS')

