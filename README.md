```diff
Time-Series-Forecasting
1) Plot the time-series. 2)Check the stationary. 3)Determine the parameter p or order of the AR model. 4)Train the model.
```

```diff
Autoregressive (AR) Models:
Often you an forecast a series based solely o the past values (Yt). Called long-memroy models.
AR(p) model:
Yt = a0 + a1Yt-1 + a2Yt-2 + ... + apYt-p + et
```
```diff 
Moving Average (MA) Models:
You can also forecast a series based solely on the past error values (et). Called short-memory models.
MA(q) model:
Yt = a0 + et + b1et-1 + b2et-2 + ... + bqet-q
```
```diff 
ARMA(p,q) model:
* p is order of AR part
* q is order of MA part
* ARMA(1,1): Yt = a1yt-1 + b1et-1 + et
Example: 
yt = 0.5yt-1 + 0.2et-1 + et
* ar_coefs = [1, -0.5]
* ma_coefs = [1, 0.2]
Auto-Regressive Integrated Moving Averages (ARIMA):
In statistics and in time series analysis, an ARIMA model is an update of ARMA (autoregressive moving average). The ARMA consists of 
mainly two components, the autoregressive and moving average; the ARIMA consists of an integrated moving average of autoregressive 
time series. ARIMA model is useful in the cases where the time series is non-stationary. ARIMA is used to help reduce the number of
parameters needed for good estimation in the model.
  
ARIMA(p,d,q):
p: The order of the auto-regressive (AR) model (i.e., the number of lag observations). 
d: The degree of differencing.
q: The order of the moving average (MA) model. This is essentially the size of the “window” function over your time series data. 
An MA process is a linear combination of past errors
```
```diff 
Auto-correlation: 
 * The ACF can be used to identify trends in data and the influence of previously observed values on a current observation Sharp peaks
 indicate a sharp correlation in time series, whereas shorter peaks indicate little correlation in the time series.
 * lag: We can calculate the correlation for current time-series observations with observations of previous time steps called lags and
 after lag q, the auto-correlation is not significant anymore. In other words, instead of calculating the correlation between two different
 series, we calculate the correlation of the series with an “x” unit lagged version (x∈N) of itself. It is also known as lagged correlation
 or serial correlation. The value of auto-correlation varies between +1 & -1. If the auto-correlation of series is a very small value that
 does not mean, there is no correlation.
 * PACF: We can find out the required number of AR terms by inspecting the Partial Autocoreelation plot. The PACF represents the correlation
 between the series and its lags. 
```
#### Long short-term memory (LSTM): 
LSTM is an artificial recurrent neural network (RNN) architecture used in the field of deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections.
 ###### Problems of traditional regression based forecasting models:
 Do not support
 * 1. noise, missing data or outliers.
 * 2. non-linear relationship.
 * 3. multiple fileds to influnce the predictions.
##### LSTM life-cycle in keras:
* 1. Define network
* 2. compile network
* 3. Fit network
* 4. Evaluate network
* 5. Make predictions

LSTMs are sensitive to the scale of the input data, specifically when the sigmoid (default) or tanh activation functions are used. It can be a good practice to rescale th data to the range of 0 to 1, also called normalizing. We can easily normalize the dataset using the MinMaxscaler preproessing class from the scikit-learn library.

We cannot use random way of splitting dataset into train and test as the sequence of events is important for time series. So we take first 70% values for train and the remaining 30% for test.
##### Vanilla LSTM:
A Vanilla LSTM is an LSTM model that has a single hidden layer of LSTM units, and an output layer used to make a prediction.

#### Stacked LSTM:
Multiple hidden LSTM layers can be stacked one on top of another in what is refeered to as a stacked LSTM model.

#### Bidirectional LSTM:
On some sequence prediction problem, it can be benenficial to allow the LSTM model to learn the input sequence both forward, backward and concatenate both interpretations. 

#### Encoder-Decoder model:




