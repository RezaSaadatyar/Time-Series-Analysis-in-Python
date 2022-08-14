import matplotlib.pyplot as plt
from sklearn import metrics
from math import sqrt


def plot_models(data, data_train, data_test, y_train_pred, y_test_pred, axs, n_lag, i, Type_model):
    plt.rcParams.update({'font.size': 10})
    if Type_model != 'LSTM':
        n_lag = 0
    if len(data_train) + i > len(data):
        aa = len(data)
    else:
        aa = len(data_train) + i

    if Type_model == 'Actual_Data':
        axs[0].plot(data_train, label="Train Data")
        axs[0].plot(data_test, label="Test Data")
        axs[1].plot(data[len(data_train) - i:len(data_train) + i], color="black", label="Data", linewidth=3.5)
        axs[0].axvspan(len(data_train) - i, aa, color="dimgray", alpha=0.3)
        axs[0].legend(fontsize=6, ncol=2)
    else:
        r2_tr = metrics.r2_score(data_train[-len(y_train_pred):], y_train_pred)
        r2_te = metrics.r2_score(data_test[-len(y_test_pred):], y_test_pred)
        rmse_tr = sqrt(metrics.mean_squared_error(data_train[-len(y_train_pred):], y_train_pred))
        rmse_test = sqrt(metrics.mean_squared_error(data_test[-len(y_test_pred):], y_test_pred))
        axs[1].plot(y_train_pred[-i:], label="Train_" + Type_model + "; RMSE:" + str(round(rmse_tr, 2)) + "; " + r'$\ R^{2}$:'
                                             + str(round(r2_tr, 2)), linestyle='-', linewidth=2.75, alpha=0.99)
        axs[1].plot(y_test_pred[:i - n_lag], label="Test_" + Type_model + "; RMSE:" + str(round(rmse_test, 2)) + "; " + r'$\ R^{2}$:'
                                                   + str(round(r2_te, 2)), linestyle='-', linewidth=2.75, alpha=0.99)
        axs[1].legend(fontsize=6, ncol=4), plt.savefig("squares1.png")
