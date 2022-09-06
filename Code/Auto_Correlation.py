import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf


def auto_correlation(data, nLags):
    """
    Args: x: X is an array usually 1*N
    Returns: plot raw data, histogram, pacf and acf        # use pacf plot to find the order p

    Auto-correlation: The ACF can be used to identify trends in data and the influence of previously observed values on a current observation
    Sharp peaks indicate a sharp correlation in time series, whereas shorter peaks indicate little correlation in the time series.
    lag: We can calculate the correlation for current time-series observations with observations of previous time steps called lags and
    after lag q, the auto-correlation is not significant anymore. In other words, instead of calculating the correlation between two different series,
    we calculate the correlation of the series with an “x” unit lagged version (x∈N) of itself. It is also known as lagged correlation
    or serial correlation. The value of auto-correlation varies between +1 & -1. If the auto-correlation of series is a very small value
    that does not mean, there is no correlation.
    """
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 11})
    # plt.rcParams.update({'font.weight': 'bold'})
    ax1 = plt.subplot(211)
    ax1.plot(data), ax1.set_title('Data')

    ax2 = plt.subplot(234)
    _, bins, _ = ax2.hist(data, bins='auto', density=True, alpha=0.8)
    ax2.plot(bins, stats.norm.pdf(bins, np.mean(data), np.std(data)), linewidth=3), ax2.set_ylabel('Probability'), ax2.set_title('Histogram of data')

    ax3 = plt.subplot(235)
    acf_value, acf_interval, _, _ = acf(data, nlags=nLags, qstat=True, alpha=0.05, fft=False)
    time = np.arange(start=0, stop=acf_value.shape[0])
    _, _, baseline = plt.stem(time, acf_value, linefmt='b-.', markerfmt='bo', basefmt='r-')
    plt.setp(baseline, color='r', linewidth=0.5)
    plt.fill_between(x=time[1:], y1=acf_interval[1:, 0] - acf_value[1:], y2=acf_interval[1:, 1] - acf_value[1:], alpha=0.25, linewidth=0, color='red')

    ax4 = plt.subplot(236)
    pacf_value, pacf_interval = pacf(data, nlags=nLags, alpha=0.05)
    _, _, bas = plt.stem(time, pacf_value, linefmt='b-.', markerfmt='bo', basefmt='r-')
    plt.setp(bas, color='r', linewidth=0.50)
    plt.fill_between(x=time[1:], y1=pacf_interval[1:, 0] - pacf_value[1:], y2=pacf_interval[1:, 1] - pacf_value[1:], alpha=0.25, linewidth=0.5, color='red')
    ax3.set_ylabel('Correlation value'), ax3.set_xlabel('#Lag'), ax3.set_title('Autocorrelation'), ax4.set_xlabel('#Lag')
    ax4.set_title('Partial Autocorrelation'), plt.tight_layout(), plt.style.use('ggplot'), plt.savefig("squares.png"), plt.show()
