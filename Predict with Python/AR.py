import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec, pyplot
import pylab as pl
from fontTools.merge import layout
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from sklearn import linear_model
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARIMA
# ======================================== Step 1: Load Data ==================================================
df = pd.read_csv('monthly_milk_production.csv')
X = df["Value"].values
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



"""

    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
         np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))
         """
plt.plot(X)
plt.hist(X)
plt.tight_layout()
# ==================================== Step 3: Auto-correlation(ACF) ==========================================
"""
Auto-correlation is a mathematical representation of the degree of similarity between a given time series and the lagged version
of itself over successive time intervals. In other words, instead of calculating the correlation between two different series,
we calculate the correlation of the series with an “x” unit lagged version (x∈N) of itself. It is also known as lagged correlation 
or serial correlation. The value of auto-correlation varies between +1 & -1. If the auto-correlation of series is a very small value
that does not mean, there is no correlation.
Auto-correlation: The ACF can be used to identify trends in data and the influence of previously observed values on a current observation
Sharp peaks indicate a sharp correlation in time series, whereas shorter peaks indicate little correlation in the time series.
lag: We can calculate the correlation for current time-series observations with observations of previous time steps called lags.
"""
fig, axs = plt.subplots(ncols=3, sharey='row')


grid = plt.GridSpec(nrows=2, ncols=3, wspace=0.4, hspace=0.3)
plt.subplot(grid[0, 0:]), plt.plot(X, axs =grid[0, 0:])
plt.subplot(grid[1,0]), plot_acf(X, lags=15)

plt.show()


plot_pacf(X, lags=15, method='ywm', ax=axs[1])
#axs[2].hist(X, 50, density=True, facecolor='b', alpha=0.75)
sns.distplot(X , color="dodgerblue", ax=axs[2], axlabel='Ideal')
axs[0].set_ylim([-0.2, 1]), axs[1].set_ylim([-0.2, 1.05])
axs[0].set_xlabel('#Lag'), axs[1].set_xlabel('#Lag')
axs[0].set_ylabel('Correlation value'), axs[2].set_ylabel('Counts')
plt.tight_layout()
# ==================================== Step 4: AR ==========================================
X = pd.DataFrame(X, columns=['Values'])
X["Values_shifted"] = X["Values"].shift()
X.dropna(inplace=True)  # The dropna() function is used to remove missing values.
y = X.values[:, 0]      # Original Signal
X = X.values[:, 1]      # Shifted Signal
train_size = int(len(X) * 0.70)
X_train, X_test = X[0:train_size].reshape(-1, 1), X[train_size:len(X)].reshape(-1, 1)
y_train = y[0:train_size]
Time = np.arange(0, len(y))
# --------------------------------------- Train Network ------------------------------------
lr = linear_model.LinearRegression()
lr.fit(X_train, y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)
fig, axs = plt.subplots(nrows=2, sharey='row')
axs[0].plot(Time, y, label="Actual signal")
axs[0].plot(Time[0:train_size], y_train_pred, label="Training_Predicted signal")
axs[0].plot(Time[train_size:], y_test_pred, label="Test_Predicted signal")
i = 100
axs[1].plot(Time[train_size-i:train_size+i], y[train_size-i:train_size+i], label="Actual signal")
axs[1].plot(Time[train_size-i:train_size], y_train_pred[train_size-i:], label="Training_Predicted signal")
axs[1].plot(Time[train_size:train_size+i], y_test_pred[0:i], label="Test_Predicted signal")
axs[1].legend(), plt.tight_layout(), plt.show()
lr.coef_
lr.intercept_
"""
model = ARIMA(y_train, order=(1,0,0))
#%%
model_fit = model.fit()
#%%
print(model_fit.summary())
"""
