import numpy as np
import pandas as pd
from math import sqrt
import pmdarima as pm
from scipy import stats, ndimage
import matplotlib.pyplot as plt
from Output_Regression import Output_regression
from Plot_Models import Plot_models
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from sklearn import linear_model, metrics, ensemble
from statsmodels.tsa import ar_model, statespace
from statsmodels.tsa.statespace.sarimax import SARIMAX
# ======================================== Step 1: Load Data ==================================================
data = sm.datasets.sunspots.load_pandas()  # df = pd.read_csv('monthly_milk_production.csv'), df.info(), X = df["Value"].values
X = data['endog'].values

mu, std = stats.norm.fit(X)
# print('Shape of data \t', df.shape)
# print('Original Dataset:\n', df.head())
# print('Values:\n', X)
# ================================ Step 2: Check Stationary Time Series ========================================
"""
Augmented Dickey-Fuller Test:
result[0]: When the test statistic is lower than the critical value shown, the time series is stationary
result[1]: p-value >>>> If Test statistic < Critical Value and p-value < 0.05 >>>> the time series is stationary
Stationary means mean, variance and covariance is constant over periods.
"""
"""
result = adfuller(X)
Output_result = pd.Series(result[0:4], index=['Test Statistic', 'p-value', '#lags used', 'number of observations used'])
for key, value in result[4].items():
    Output_result['critical value (%s)' % key] = value
print(Output_result)
if result[0] < result[4]["5%"]:
    print("Reject Ho - Time Series is Stationary")
else:
    print("Failed to Reject Ho - Time Series is Non-Stationary")
    # Converting series to stationary
    X.diff(periods=1)
    X.dropna(inplace=True)
# ==================================== Step 3: Auto-correlation(ACF) ==========================================
"""
"""
Auto-correlation is a mathematical representation of the degree of similarity between a given time series and the lagged version
of itself over successive time intervals. In other words, instead of calculating the correlation between two different series,
we calculate the correlation of the series with an “x” unit lagged version (x∈N) of itself. It is also known as lagged correlation 
or serial correlation. The value of auto-correlation varies between +1 & -1. If the auto-correlation of series is a very small value
that does not mean, there is no correlation.
Auto-correlation: The ACF can be used to identify trends in data and the influence of previously observed values on a current observation
Sharp peaks indicate a sharp correlation in time series, whereas shorter peaks indicate little correlation in the time series.
lag: We can calculate the correlation for current time-series observations with observations of previous time steps called lags and
after lag q, the auto-correlation is not significant anymore.
"""
"""
fig = plt.figure(figsize=(10, 8))
ax1 = plt.subplot(211)
ax1.plot(X)
ax2 = plt.subplot(234)
_, bins, _ = ax2.hist(X, bins='auto', density=True, alpha=0.8)
ax2.plot(bins, stats.norm.pdf(bins, np.mean(X), np.std(X)), linewidth=2), ax2.set_ylabel('Probability'), ax2.set_title('Histogram of data')
ax3 = plt.subplot(235)
plot_acf(X, lags=15, ax=ax3), ax3.set_ylim([-0.2, 1.05]), ax3.set_ylabel('Correlation value'), ax3.set_xlabel('#Lag')
ax4 = plt.subplot(236)
plot_pacf(X, lags=15, method='ywm', ax=ax4), ax4.set_ylim([-0.2, 1.05]), ax4.set_xlabel('#Lag')  # use pacf plot to find the order p
plt.tight_layout(), plt.style.use('ggplot'), plt.show()
# =========================== Step 4: Split Dataset intro Train and Test ==========================================
"""
nLags = 3
Data_Lags = np.zeros((len(X), nLags))
for i in np.arange(0, nLags):
    Data_Lags[:, i] = ndimage.interpolation.shift(X, i+1, cval=np.NaN)
Data_Lags = Data_Lags[nLags:]  # remove missing(NaN) values.
X = X[nLags:]
Time = np.arange(0, len(X))
train_size = int(len(X) * 0.8)
X_train = Data_Lags[:train_size]
X_test = Data_Lags[train_size:]
Label_train = X[:train_size]
Label_test = X[train_size:]
# --------------------------------------- Step 5: Linear Regression Model  ----------------------------------------
mod = linear_model.LinearRegression()
y_train_pred, y_test_pred, RMSE_train, RMSE_test = Output_regression(X_train, Label_train, X_test, Label_test, mod)
fig, axs = plt.subplots(nrows=3, sharey='row', figsize=(12, 6))
Plot_models(Time, X, train_size, y_train_pred, y_test_pred, RMSE_train, RMSE_test, axs, Type_model='Actual_Data')
Plot_models(Time, X, train_size, y_train_pred, y_test_pred, RMSE_train, RMSE_test, axs, Type_model='LR')
# ------------------------------------- Step 5: LRandomForestRegressor Models -------------------------------------
mod = ensemble.RandomForestRegressor(n_estimators=100, max_features=nLags, random_state=1)
y_train_pred, y_test_pred, RMSE_train, RMSE_test = Output_regression(X_train, Label_train, X_test, Label_test, mod)
Plot_models(Time, X, train_size, y_train_pred, y_test_pred, RMSE_train, RMSE_test, axs, Type_model='RF')
# --------------------------------------- Step 5: Auto-Regressive (AR) model --------------------------------------
"""1) Plot the time-series. 2)Check the stationarity. 3)Determine the parameter p or order of the AR model. 4)Train the model."""
test_size = len(X) - train_size
nLags = 10
mod = ar_model.AutoReg(X[:train_size], lags=nLags).fit()
y_test_pred = mod.model.predict(mod.params, start=train_size, end=len(X)-1)  # For predict Future: end - start samples
Plot_models(Time, X, train_size, y_train_pred, y_test_pred, RMSE_train, RMSE_test, axs, Type_model='Regressive')
Plot_models(Time, X, train_size, y_train_pred, y_test_pred, RMSE_train, RMSE_test, axs, Type_model='AR')
# print(mod.summary())
# --------------------------------------- Step 5: Moving Average (MA) model --------------------------------------

# ----------------------- Step 5: Auto-Regressive Integrated Moving Averages (ARIMA) ------------------------------
"""ARIMA(p,d,q):
p: The order of the auto-regressive (AR) model (i.e., the number of lag observations). 
d: The degree of differencing.
q: The order of the moving average (MA) model. This is essentially the size of the “window” function over your time series data. 
An MA process is a linear combination of past errors."""
mod = pm.auto_arima(X[:train_size], start_p=5, start_q=1, seasonal=True, m=10, d=1, n_fits=50, information_criterion="bic",
                    trace=False, stepwise=True, method='lbfgs')
y_test_pred, _ = mod.predict(n_periods=len(X) - train_size, return_conf_int=True)   # predict N steps into the future
Plot_models(Time, X, train_size, y_train_pred, y_test_pred, RMSE_train, RMSE_test, axs, Type_model='ARIMA')
plt.tight_layout(), plt.style.use('ggplot'), plt.show()
# print(mod.summary())
mod = statespace.sarimax.SARIMAX(X[:train_size], order=(4, 1, 3), seasonal_order=(0, 0, 0, 10))
mod = mod.fit()
y_test_pred = mod.predict(n_)
