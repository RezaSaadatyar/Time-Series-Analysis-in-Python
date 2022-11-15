import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics


# from math import sqrt


def plot_models(data, y_train_pred, y_test_pred, axs, n_lag, train_size, num_sample, type_model):
    plt.rcParams.update({'font.size': 18})

    if 'DataFrame' not in str(type(data)):  # Check type data
        data = pd.DataFrame(data)

    if train_size - num_sample <= 0:
        i = 0
        iii = 0.16
        axs[0].set_xticks([])
    else:
        iii = 0.07
        i = train_size - num_sample
        axs[0].tick_params(axis='x', labelsize=16)

    if train_size + num_sample > len(data):
        ii = len(data) - 1
    else:
        ii = train_size + num_sample

    if type_model == 'Actual_Data':

        if data.shape[1] > 1:
            axs[0].plot(data, label=data.columns, linewidth=2.5)
        else:
            axs[0].plot(data, 'k', label="Original data", linewidth=2.5),

        axs[0].axvline(x=train_size, linewidth=4, color='r', ls='--'), axs[1].axvline(x=train_size, linewidth=4, color='r', ls='--')
        axs[2].axvline(x=train_size, linewidth=4, color='r', ls='--'), axs[3].axvline(x=train_size, linewidth=4, color='r', ls='--')
        axs[0].axvspan(i, ii, color='yellow', alpha=0.75), axs[1].plot(data.iloc[i:train_size + ii, 0], 'k', linewidth=4)
        axs[2].plot(data.iloc[i:train_size + ii, 0], 'k', linewidth=3.5), axs[3].plot(data.iloc[i:train_size + ii, 0], 'k', linewidth=3.5),

        axs[0].set_xlim(data.index[0], data.index[-1]), axs[1].set_xlim(data.index[i], data.index[ii]), axs[2].set_xlim(data.index[i], data.index[ii])
        axs[3].set_xlim(data.index[i], data.index[ii])

        axs[0].text(train_size / 2, np.max(np.max(data)) / 1.1, "Training set", ha="center", va="center", rotation=0, size=17,
                    bbox=dict(boxstyle="round,pad=0.6", fc="w", ec="b", lw=2.5))
        axs[0].text(train_size + ((len(data) - train_size) / 2), np.max(np.max(data)) / 1.1, "Test set", ha="center", va="center", rotation=0, size=17,
                    bbox=dict(boxstyle="round,pad=0.6", fc="w", ec="g", lw=2.5))
        axs[1].set_title('Linear regression models', loc='right', y=1 + iii, pad=-20, color='deeppink', fontsize=19, fontstyle='italic')
        axs[2].set_title('Machine learning models', loc='right', y=1 + 0.16, pad=-20, color='deeppink', fontsize=19, fontstyle='italic')
        axs[3].set_title('Deep learning models', loc='right', y=1 + 0.16, pad=-20, color='deeppink', fontsize=19, fontstyle='italic')
        axs[1].set_xticks([]), axs[2].set_xticks([]), axs[0].tick_params(axis='y', labelsize=16), axs[1].tick_params(axis='y', labelsize=16)
        axs[2].tick_params(axis='y', labelsize=16), axs[3].tick_params(axis='y', labelsize=16), axs[3].tick_params(axis='x', labelsize=16)
        axs[0].legend(fontsize=16, ncol=2, frameon=False, loc='best', labelcolor='linecolor', handlelength=0)

    elif (type_model == 'LS') | (type_model == 'AR') | (type_model == 'ARX') | (type_model == 'ARIMA') | (type_model == 'AR+LS'):
        sns.set(style='white')
        r2_tr = metrics.r2_score(data[train_size - len(y_train_pred):train_size], y_train_pred)
        r2_te = metrics.r2_score(data[-len(y_test_pred):], y_test_pred)
        # rmse_tr = sqrt(metrics.mean_squared_error(data[train_size - len(y_train_pred):train_size], y_train_pred))
        # rmse_test = sqrt(metrics.mean_squared_error(data[-len(y_test_pred):], y_test_pred))
        axs[1].plot(pd.concat([y_train_pred[i:], y_test_pred[:train_size + ii]]), label=type_model + "=" + "$R_{tr,te}^{2}$:" +
                                                                                        str(round(r2_tr, 2)) + "; " + str(round(r2_te, 2)), linestyle='--', linewidth=4, alpha=1)
        axs[1].legend(fontsize=16, ncol=2, loc='best', borderaxespad=0, frameon=False, labelcolor='linecolor', handlelength=0)

    elif (type_model == 'LR') | (type_model == 'RF') | (type_model == 'DT') | (type_model == 'Xgboost'):
        sns.set(style='white')
        r2_tr = metrics.r2_score(data[train_size - len(y_train_pred):train_size], y_train_pred)
        r2_te = metrics.r2_score(data[-len(y_test_pred):], y_test_pred)
        axs[2].plot(pd.concat([y_train_pred[i:], y_test_pred[:train_size + ii]]), label=type_model + "=" + "$R_{tr, te}^{2}$:" +
                                                                                        str(round(r2_tr, 2)) + "; " + str(round(r2_te, 2)), linestyle='--', linewidth=4, alpha=1)
        axs[2].legend(fontsize=16, ncol=2, loc='best', borderaxespad=0, frameon=False, labelcolor='linecolor', handlelength=0)

    elif type_model == 'LSTM':
        data_train = data[:train_size]
        data_test = data[train_size:]
        r2_tr = metrics.r2_score(data_train[-len(y_train_pred):], y_train_pred)
        r2_te = metrics.r2_score(data_test[-len(y_test_pred):], y_test_pred)

        axs[3].plot(pd.concat([y_train_pred[i:], y_test_pred[:train_size + ii]]), label=type_model + "=" + r'$R_{tr, te}^{2}$:' +
                                                                                        str(round(r2_tr, 2)) + "; " + str(round(r2_te, 2)), linestyle='--', linewidth=4, alpha=1)
        axs[3].legend(fontsize=16, ncol='2', loc='best', borderaxespad=0, frameon=False, labelcolor='linecolor', handlelength=0)
