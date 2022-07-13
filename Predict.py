import warnings
import numpy as np
import pandas as pd
import pmdarima as pm
from scipy import stats, ndimage
import matplotlib.pyplot as plt
from Output_Regression import output_regression
from Plot_Models import plot_models
from Auto_Correlation import auto_correlation
from Test_Stationary import test_stationary
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from sklearn import linear_model, ensemble
from statsmodels import tsa
from statsmodels.tsa.stattools import arma_order_select_ic

# ======================================== Step 1: Load Data ==================================================
warnings.filterwarnings("default")    # "error", "ignore", "always", "default", "module"
data = sm.datasets.sunspots.load_pandas()   # df = pd.read_csv('monthly_milk_production.csv'), df.info(), X = df["Value"].values
data = data.data["SUNACTIVITY"]
# print('Shape of data \t', data.shape)
# print('Original Dataset:\n', data.head())
# print('Values:\n', data)
# ================================ Step 2: Check Stationary Time Series ========================================
# data = test_stationary(data, window=20)
# ==================================== Step 3: Find the lags of AR and etc models ==============================
# auto_correlation(data, nLags=10)
# =========================== Step 4: Split Dataset intro Train and Test =======================================
nLags = 3
Data_Lags = pd.DataFrame(np.zeros((len(data), nLags)))
for i in np.arange(0, nLags):
    Data_Lags[i] = data.shift(i+1)
Data_Lags = Data_Lags[nLags:]
data = data[nLags:]
Data_Lags.index = (np.linspace(0, len(Data_Lags), num=len(Data_Lags), endpoint=False, dtype='int'))
data.index = (np.linspace(0, len(data), num=len(data), endpoint=False, dtype='int'))
train_size = int(len(data) * 0.8)
data_train = data[:train_size]
data_test = data[train_size:]
# --------------------------------------- Step 5: Auto-Regressive (AR) model -----------------------------------
"""1) Plot the time-series. 2)Check the stationary. 3)Determine the parameter p or order of the AR model. 4)Train the model."""
mod = tsa.ar_model.AutoReg(data_train, lags=10).fit()
y_train_pred = pd.Series(mod.fittedvalues)                                                 # train
y_test_pred = pd.Series(mod.model.predict(mod.params, start=train_size, end=len(data)-1))  # For predict Future: end - start samples
fig, axs = plt.subplots(nrows=2, sharey='row', figsize=(12, 6))
plot_models(data, data_train, data_test, y_train_pred, y_test_pred, axs, i=50, Type_model='Actual_Data')
plot_models(data, data_train, data_test, y_train_pred, y_test_pred, axs, i=50, Type_model='AR')
# ----------------------- Step 5: Auto-Regressive Integrated Moving Averages (ARIMA) ------------------------------
"""ARIMA(p,d,q):
p: The order of the auto-regressive (AR) model (i.e., the number of lag observations). 
d: The degree of differencing.
q: The order of the moving average (MA) model. This is essentially the size of the “window” function over your time series data. 
An MA process is a linear combination of past errors."""

# mod = pm.auto_arima(X[:train_size], start_p=5, start_q=1, seasonal=True, m=10, d=1, n_fits=50, information_criterion="bic", trace=True, stepwise=True, method='lbfgs')
ma = (1, 1, 1, 1)
mod = tsa.statespace.sarimax.SARIMAX(data_train, order=(5, 1, ma), seasonal_order=(0, 0, 2, 12))
mod = mod.fit(disp=False)
y_train_pred = pd.Series(mod.fittedvalues)
y_test_pred = mod.predict(start=train_size, end=len(data)-1, dynamic=True, typ='levels')  # predict N steps into the future
plot_models(data, data_train, data_test, y_train_pred, y_test_pred, axs, i=50, Type_model='ARIMA')
# --------------------------------------- Step 5: Linear Regression Model  ----------------------------------------
mod = linear_model.LinearRegression()
y_train_pred, y_test_pred = output_regression(Data_Lags[:train_size],  Data_Lags[train_size:], mod, label_train=data_train)
plot_models(data, data_train, data_test, y_train_pred, y_test_pred, axs, i=50, Type_model='LR')
# ------------------------------------- Step 5: LRandomForestRegressor Models -------------------------------------
mod = ensemble.RandomForestRegressor(n_estimators=100, max_features=nLags, random_state=1)
y_train_pred, y_test_pred = output_regression(Data_Lags[:train_size],  Data_Lags[train_size:], mod, label_train=data_train)
plot_models(data, data_train, data_test, y_train_pred, y_test_pred, axs, i=50, Type_model='RF')
plt.tight_layout(), plt.style.use('ggplot'), plt.show()



