import pandas as pd
from statsmodels import tsa
from Plot_Models import plot_models


def arima_model(data, train_size, axs, order, seasonal_order, num_sample):
    # mod = pm.auto_arima(X[:train_size], start_p=5, start_q=1, seasonal=True, m=10, d=1, n_fits=50, information_criterion="bic", trace=True, stepwise=True, method='lbfgs')
    mod = tsa.statespace.sarimax.SARIMAX(data[:train_size], order=order, seasonal_order=seasonal_order)
    mod = mod.fit(disp=False)
    y_train_pred = pd.Series(mod.fittedvalues)
    y_test_pred = mod.predict(start=train_size, end=len(data) - 1, dynamic=True, typ='levels')  # predict N steps into the future
    plot_models(data, y_train_pred, y_test_pred, axs, [], train_size, num_sample=num_sample, type_model='ARIMA')
