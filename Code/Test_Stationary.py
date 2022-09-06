import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa import stattools
import matplotlib.pyplot as plt


def test_stationary(data, window):
    """
    param Data: Data is a ndarray and often 1 * N

    param window: Size of the moving window. If an integer, the fixed number of observations used for each window.
    If an offset, the time period of each window. Each window will be a variable sized based on the observations included
    in the time-period.

    return: Convert non-stationary data to stationary data if Data is Non-Stationary.

    Check Stationary Time Series: 1)Rolling statistics: plot the moving average/variance and see if it varies with
    time. 2) Augmented Dickey-Fuller Test: result[0]: When the test statistic is lower than the critical value shown,
    the time series is stationary result[1]: p-value >>>> If Test statistic < Critical Value and p-value < 0.05 >>>>
    the time series is stationary. Stationary means >>> mean, variance and covariance is constant over periods and
    auto-covariance that does not depend on time.

    Converting Non-stationary data to stationary dataset:
    Log: np.log(Data)
    Differencing simple moving average: MA = Data.rolling(window=window).mean()
    Data = Data - MA
    Data.dropna(inplace=True)
    """
    # ================================ Step 2: Check Stationary Time Series ========================================
    data1 = data
    sns.set(style='white')
    result = stattools.adfuller(data)                                   # Perform Dickey-Fuller Test
    if result[0] < result[4]["5%"]:
        fig, ax1 = plt.subplots(1, 1, sharey='row', figsize=(10, 6))
        plt.rcParams.update({'font.size': 11})
        ax1.set_title('Rolling Mean & Standard Deviation; ' + 'p-value:' + str(round(result[1], 3)) + '; Data is Stationary')
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharey='row', figsize=(10, 6))
        plt.rcParams.update({'font.size': 11})
        ax1.set_title('Rolling Mean & Standard Deviation; ' + 'p-value:' + str(round(result[1], 3)) + '; Data is Non-Stationary')
        data = data - data.rolling(window=window).mean()                 # X.diff(periods=1)
        data.dropna(inplace=True)
        data.index = (np.linspace(0, len(data), num=len(data), endpoint=False, dtype='int'))
        data = pd.Series(data)
        result = stattools.adfuller(data)                                # Perform Dickey-Fuller Test
        ax2.plot(data)
        ax2.plot(data.rolling(window=window).mean())                     # Determine rolling statistics
        ax2.plot(data.rolling(window=window).std())
        ax2.set_title('Rolling Mean & Standard Deviation; ' + 'p-value:' + str(round(result[1], 3)) + '; Data is Stationary')
    output_result = pd.Series(result[0:4], index=['Test Statistic', 'p-value', '#lags used', 'number of observations used'])
    for key, value in result[4].items():
        output_result['critical value (%s)' % key] = value
    print(output_result)
    ax1.plot(data1, label='Data')
    ax1.plot(data1.rolling(window=window).mean(), label='Rolling Mean')   # Determine rolling statistics
    ax1.plot(data1.rolling(window=window).std(), label='Rolling Std')
    ax1.legend(loc='best'), plt.tight_layout(), plt.show()
    return data
