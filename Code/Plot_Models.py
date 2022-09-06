import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
# from math import sqrt


def plot_models(data, y_train_pred, y_test_pred, axs, n_lag, train_size, num_sample, type_model):
    plt.rcParams.update({'font.size': 18})

    if train_size + num_sample > len(data):
        ii = len(data)
    else:
        ii = train_size + num_sample

    if type_model == 'Actual_Data':

        axs[0].plot(data, 'k', label="Original data", linewidth=2.5),
        axs[0].axvline(x=train_size, linewidth=4, color='r', ls='--')
        axs[1].axvline(x=train_size, linewidth=4, color='r', ls='--'), axs[2].axvline(x=train_size, linewidth=4, color='r', ls='--')
        axs[3].axvline(x=train_size, linewidth=4, color='r', ls='--'), axs[0].set_xlim(data.index[0], data.index[-1]),
        axs[1].set_xlim(data.index[train_size - num_sample], data.index[ii]), axs[2].set_xlim(data.index[train_size - num_sample], data.index[ii])
        axs[3].set_xlim(data.index[train_size - num_sample], data.index[ii]), axs[3].set_xlim(data.index[train_size - num_sample], data.index[ii])
        axs[0].axvspan(train_size - num_sample, ii, color='yellow', alpha=0.75), axs[1].set_xticks([]), axs[2].set_xticks([])
        axs[0].text(train_size / 2, np.max(data)/1.1, "Training set", ha="center", va="center", rotation=0, size=17,
                    bbox=dict(boxstyle="round,pad=0.6", fc="w", ec="b", lw=2.5))
        axs[0].text(ii - num_sample / 2, np.max(data)/1.1, "Test set", ha="center", va="center", rotation=0, size=17,
                    bbox=dict(boxstyle="round,pad=0.6", fc="w", ec="g", lw=2.5))
        axs[1].plot(data[train_size - num_sample:train_size + ii], 'k', linewidth=4), axs[2].plot(data[train_size - num_sample:train_size + ii], 'k', linewidth=3.5)
        axs[3].plot(data[train_size - num_sample:train_size + ii], 'k', linewidth=3.5), axs[0].legend(fontsize=16, ncol=2)
        axs[1].set_title('Linear regression models', loc='right', y=1, pad=-20, color='deeppink', fontsize=20)
        axs[2].set_title('Machine learning models', loc='right', y=1, pad=-20, color='deeppink', fontsize=20)
        axs[3].set_title('Deep learning models', loc='right', y=1, pad=-20, color='deeppink', fontsize=20)

    elif (type_model == 'LS') | (type_model == 'AR') |(type_model == 'ARX') | (type_model == 'ARIMA'):
        sns.set(style='white')
        r2_tr = metrics.r2_score(data[train_size - len(y_train_pred):train_size], y_train_pred)
        r2_te = metrics.r2_score(data[-len(y_test_pred):], y_test_pred)
        # rmse_tr = sqrt(metrics.mean_squared_error(data[train_size - len(y_train_pred):train_size], y_train_pred))
        # rmse_test = sqrt(metrics.mean_squared_error(data[-len(y_test_pred):], y_test_pred))
        axs[1].plot(pd.concat([y_train_pred[train_size - num_sample:], y_test_pred[:train_size + ii]]), label=type_model + "=" + r'$\ R_{tr}^{2}$:' +
                    str(round(r2_tr, 2)) + "," + r'$\ R_{te}^{2}$:' + str(round(r2_te, 2)), linestyle='--', linewidth=4, alpha=1)
        axs[1].legend(fontsize=15, ncol=2, loc=2, borderaxespad=0, frameon=False)

    elif (type_model == 'LR') | (type_model == 'RF') | (type_model == 'DT') | (type_model == 'Xgboost'):
        sns.set(style='white')
        r2_tr = metrics.r2_score(data[train_size - len(y_train_pred):train_size], y_train_pred)
        r2_te = metrics.r2_score(data[-len(y_test_pred):], y_test_pred)
        axs[2].plot(pd.concat([y_train_pred[train_size - num_sample:], y_test_pred[:train_size + ii]]), label=type_model + "=" + r'$\ R_{tr}^{2}$:' +
                    str(round(r2_tr, 2)) + "," + r'$\ R_{te}^{2}$:' + str(round(r2_te, 2)), linestyle='--', linewidth=4, alpha=1)
        axs[2].legend(fontsize=15, ncol=2, loc=2, borderaxespad=0, frameon=False)

    elif type_model == 'LSTM':
        data_train = data[:train_size]
        data_test = data[train_size:]
        r2_tr = metrics.r2_score(data_train[-len(y_train_pred):], y_train_pred)
        r2_te = metrics.r2_score(data_test[-len(y_test_pred):], y_test_pred)

        axs[3].plot(pd.concat([y_train_pred[train_size - num_sample:], y_test_pred[:train_size + ii]]), label=type_model + "=" + r'$\ R_{tr}^{2}$:' +
                    str(round(r2_tr, 2)) + "," + r'$\ R_{te}^{2}$:' + str(round(r2_te, 2)), linestyle='--', linewidth=4, alpha=1)
        axs[3].legend(fontsize=15, ncol=2, loc=2, borderaxespad=0, frameon=False)

