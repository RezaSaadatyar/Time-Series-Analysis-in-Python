# Time-Series-Forecasting

#### Autoregressive (AR) Models:
Often you an forecast a series based solely o the past values (Yt). Called long-memroy models.
###### AR(p) model:
Yt = a0 + a1Yt-1 + a2Yt-2 + ... + apYt-p + et
#### Moving Average (MA) Models:
You can also forecast a series based solely on the past error values (et). Called short-memory models.
##### MA(q) model:
Yt = a0 + et + b1et-1 + b2et-2 + ... + bqet-q
#### ARMA(p,q) model:
* p is order of AR part
* q is order of MA part
* ARMA(1,1): Yt = a1yt-1 + b1et-1 + et
Examole: 
yt = 0.5yt-1 + 0.2et-1 + et
*ar
Auto-Regressive Integrated Moving Averages (ARIMA):
In statistics and in time series analysis, an ARIMA model is an update of ARMA (autoregressive moving average). The ARMA consists of mainly two components, the autoregressive and moving average; the ARIMA consists of an integrated moving average of autoregressive time series. ARIMA model is useful in the cases where the time series is non-stationary. ARIMA is used to help reduce the number of parameters needed for good estimation in the model.
  
ARIMA(p,d,q):
p: The order of the auto-regressive (AR) model (i.e., the number of lag observations). 
d: The degree of differencing.
q: The order of the moving average (MA) model. This is essentially the size of the “window” function over your time series data. 
An MA process is a linear combination of past errors

 Auto-correlation: 
 The ACF can be used to identify trends in data and the influence of previously observed values on a current observation
 Sharp peaks indicate a sharp correlation in time series, whereas shorter peaks indicate little correlation in the time series.
 lag: We can calculate the correlation for current time-series observations with observations of previous time steps called lags and
 after lag q, the auto-correlation is not significant anymore. In other words, instead of calculating the correlation between two different series,
 we calculate the correlation of the series with an “x” unit lagged version (x∈N) of itself. It is also known as lagged correlation
 or serial correlation. The value of auto-correlation varies between +1 & -1. If the auto-correlation of series is a very small value
 that does not mean, there is no correlation.
