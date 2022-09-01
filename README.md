### $$\textcolor{blue}{\text{Time Series Analysis and Forecasting}}$$

The term time series refers to a series of observations that depend on time. Time is an essential feature in natural processes such as air temperature, a pulse of     the heart, or stock price changes. Analyzing time series and forecasting time series are two different things.

 **Time series analysis:** As a result of time series analysis, we can extract useful information from time series data: trend, cyclic and seasonal deviations, correlations, etc. Time series analysis is the first step to prepare and analyze time series dataset for time series forecasting

 **Time series forecasting** includes: Developing models and Using them to forecast future predictions.Time series forecasting tries to find the most likely
time series values in the future


**An explanation of data set splits**
- ***Train data set:*** A train data set represents the amount of data that machine learning models are fitted with.
- ***Validation data set:*** Validation data sets provide an unbiased evaluation of model fit on train data sets while tuning model hyperparameters.
- ***Test data set:*** A test data set is used to identify whether a model is underfitting (the model performs poorly on the train data set) or overfitting (the model performs well on the train data set but fails to perform well on the test data set). It is determined by looking at the prediction error on both train and test data sets. The test data set is only used after the train and validation data sets have been used to train and validate the model.


---
>![#f03c15](https://via.placeholder.com/15/f03c15/f03c15.png) **Machine Learning for Time Series Forecasting: [Further information](https://github.com/RezaSaadatyar/Machine-Learning-in-Python)** <br/> 
> *The learning process is based on the following steps:*<br/> 
> - 1. Algorithms are fed data. (In this step you can provide additional information to the model, for example, by performing feature extraction).<br/>
> - 2. Train a model using this data.<br/>
> - 3. Test and deploy the model.<br/> 
> - 4. Utilize the deployed model to automate predictive tasks.



**Lag feature**
These are time-shifted values of the actual demand. For example, lag 1 features store the demand of the previous hour/sample relative to the current time stamp. Similarly, we can add lag 2, lag 3, and so on. A combination of lag features is selected during the modeling phase based on the evaluation of the model results.

>![#f03c15](https://via.placeholder.com/15/f03c15/f03c15.png)  **Linear Regression Models**<br/>
>  - [x] ***Linear Correlation:***<br/>
> For two related variables, the correlation measures the association between the two variables. In contrast, a ***linear regression*** is used for the prediction of the value of one variable from another.<br/>
>  - [x] ***Linear Regression:***<br/> We can use the method of Linear Regression when we want to predict the value of one variable from the value(s) of one or more other variables. **$y_{t} = a_{0}+ x_{t} + e_{t}$**<br/>
> **Introduction to Autoregressive and Automated Methods for Time Series Forecasting:**
> - Autoregressive (AR) Model.
>   - The AR(p) notation refers to the autoregressive model which uses p history lag to predict the future.
>   - AR(p) model: $Yt = a0 + a1Yt-1 + a2Yt-2 + ... + apYt-p + et$
> - Auto-Regressive Integrated Moving Averages (ARIMA):
>   - In statistics and in time series analysis, an ARIMA model is an update of ARMA (autoregressive moving average). The ARMA consists of mainly two components, the autoregressive and moving average; the ARIMA consists of an integrated moving average of autoregressive time series. ARIMA is used to help reduce the number of
parameters needed for good estimation in the model.
>   - ARIMA(p,d,q):
     - p: The order of the auto-regressive (AR) model (i.e., the number of lag observations). 
     - d: The degree of differencing.
     - q: The order of the moving average (MA) model. This is essentially the size of the “window” function over your time series data. 
> - Seasonal autoregressive integrated moving average (SARIMA) moddel
     - is an extension of the ARIMA model that also includes the impact of seasonality. It combines the ARIMA model with the seasonally adjusted predictions.
> - Holt-Winters exponential smoothing (HWES):
   - HWES is a way to model three aspects of the time series: an average of a typical value, a trend, and a seasonality. HWES uses exponential smoothing to remember lots of data from the past and predict future values. So this method is suitable for time series with trend and seasonal components.

An MA process is a linear combination of past errors
- Moving Average 
- Autoregressive Moving Average 
- Autoregressive Integrated Moving Average

Linear methods like ARIMA, SARIMA, HWES are popular classical techniques for time series forecasting. But these traditional approaches also have some constraints:
 - Focus on linear relationships and inability to find complex nonlinear ones
Fixed lag observations and incapacity to make feature pre-processing
Missing data are not supported
Working with univariate time series only, but common real-world problems have multiple input variables
One-step predictions: many real-world problems require predictions with a long time horizon

> Moving Average (MA) Models<br/>
You can also forecast a series based solely on the past error values (et). Called short-memory models.
MA(q) model:
$Yt = a0 + et + b1et-1 + b2et-2 + ... + bqet-q$

```diff 
ARMA(p,q) model:
* p is order of AR part
* q is order of MA part
* ARMA(1,1): Yt = a1yt-1 + b1et-1 + et
Example: 
yt = 0.5yt-1 + 0.2et-1 + et
* ar_coefs = [1, -0.5]
* ma_coefs = [1, 0.2]

```
> **Problems of traditional regression based forecasting models:**<br/>
> Do not support<br/>
> * 1. noise, missing data or outliers.
> * 2. non-linear relationship.
> * 3. multiple fileds to influnce the predictions.

> ***Holt Winter Multiplicative method***
>  **$y_{t-1} = (L_t + b_t) * S_{t+1-m}$**
> Auto-correlation: 
>   * The ACF can be used to identify trends in data and the influence of previously observed values on a current observation Sharp peaks
 indicate a sharp correlation in time series, whereas shorter peaks indicate little correlation in the time series.
>   * lag: We can calculate the correlation for current time-series observations with observations of previous time steps called lags and
 after lag q, the auto-correlation is not significant anymore. In other words, instead of calculating the correlation between two different
 series, we calculate the correlation of the series with an “x” unit lagged version (x∈N) of itself. It is also known as lagged correlation
 or serial correlation. The value of auto-correlation varies between +1 & -1. If the auto-correlation of series is a very small value that
 does not mean, there is no correlation.
 >  * PACF: We can find out the required number of AR terms by inspecting the Partial Autocoreelation plot. The PACF represents the correlation
 between the series and its lags. 
** machine learning for forecasting**
Multilayer Perceptron (MLP):
MLP is a deep, artificial neural network connected to mutltiple layers in a directed graph.
** Forecasting with machine learning**
> **Long short-term memory (LSTM):**<br/> LSTM is an artificial recurrent neural network (RNN) architecture used in the field of deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections. LSTMs are sensitive to the scale of the input data, specifically when the sigmoid (default) or tanh activation functions are used. It can be a good practice to rescale th data to the range of 0 to 1, also called normalizing. We can easily normalize the dataset using the MinMaxscaler preproessing class from the scikit-learn library.
>   - LSTM Autoenooder 
>   - Vanilla LSTM<br/> A Vanilla LSTM is an LSTM model that has a single hidden layer of LSTM units, and an output layer used to make a prediction.
>   - Stacked LSTM<br/> Multiple hidden LSTM layers can be stacked one on top of another in what is refeered to as a stacked LSTM model.
>   - Bidirectional LSTM<br/> On some sequence prediction problem, it can be benenficial to allow the LSTM model to learn the input sequence both forward, backward and concatenate both interpretations.
>
> **LSTM life-cycle in keras:**
> * 1. Define network
> * 2. compile network
> * 3. Fit network
> * 4. Evaluate network
> * 5. Make predictions

#### Encoder-Decoder model:

> **Data preprocessing**
> - Data scaling
> - Train-test split<br/> We cannot use random way of splitting dataset into train and test as the sequence of events is important for time series. So we take first 70% values for train and the remaining 30% for test.
> - Create sequences
> - Build a model
