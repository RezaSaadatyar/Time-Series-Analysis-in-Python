**Time Series Analysis and Forecasting**

***This repository Covers:***
- 1. A brief about the Time series
- 2. Preparing the data
     - Normalize data (0-1)
     - Check stationary time series (p < 0.005)
- 3. Find the lags
- 4. Split dataset into train and test
- 5. Types of Regression Algorithms
      - Training the model
      - Prediction and performance check

:arrow_forward: The program will run automatically when you run **code/file Main.py**, and you do not need to run any of the other codes. Depending on your goal, you can execute all steps independently or interdependently within the code/file main. In addition, you can copy and run each section along with its related files in your own code or disable each section with a **#**. To run the program, the only thing you need is your input, which is **data (data = Your data)**.

----
:one: The term time series refers to a series of observations that depend on time. Time is an essential feature in natural processes such as air temperature, a pulse of the heart, or stock price changes. Analyzing time series and forecasting time series are two different things.

 **Time series analysis:** As a result of time series analysis, we can extract useful information from time series data: trends, cyclic and seasonal deviations, correlations, etc. Time series analysis is the first step to preparing and analyzing time series datasets for time series forecasting

 **Time series forecasting** includes: Developing models and Using them to forecast future predictions. Time series forecasting tries to find the most likely
time-series values in the future

---
:two: Data pre-processing is the step where clean data sets from outliers and missing data and create additional features with the raw data to feed the model.<br/> 
- ***Missing values*** can be filled by interpolating between two closest non-missing values or by using different Python functions (e.g., interpolate()) to fill NAN values in the DataFrame or series. 
- ***Normalization*** can be useful, and even required in some machine learning algorithms, when your time series data has input values and features with differing measurements and dimensions. For machine learning algorithms, such as *k-nearest neighbors*, which use distance estimates, *linear regression, and neural networks* that process a weight calibration on input values, normalization is necessary. 
- In ***standardizing*** a data set, the distribution of observed values is rescaled to have a mean of 0 and a standard deviation of 1. Standardization assumes that your observations fit a Gaussian distribution with a well-behaved mean and standard deviation. Algorithms like support vector machines linear and logistic regression and other algorithms have improved performance with Gaussian data.
- ***Check Stationary Time Series:*** Mean, and variance are constant over periods, and auto-covariance does not depend on time. Plot the moving average/variance (Rolling window statistics) and see if it varies with time. Augmented Dickey-Fuller Test: When the test statistic (p-value) is lower than the critical value shown, the time series is stationary.
----

:three: **Lag features**<br/>
They are time-shifted values of the actual demand. For example, lag 1 feature store the demand of the previous hour/sample relative to the current time stamp. Similarly, we can add lag 2, lag 3, and so on. A combination of lag features is selected during the modeling phase based on the evaluation of the model results. The operation of adding lag features is called the sliding window method or Window Features.

***Autocorrelation*** describes the correlation between the output (that is, the target variable that we need to predict) and a specific lagged variable (that is, a group of values at a prior time stamp used as input). ***Autocorrelation plot*** is also often used to check randomness in time series. If the time series is random, autocorrelation values should be near zero for all time lags. If the time series is non-random, then one or more of the autocorrelations will be significantly non-zero. The purpose of *the autocorrelation plot* is to show whether the data points in a time series are positively correlated, negatively correlated, or independent of one another. A plot of the autocorrelation of a time series by lag is also called the ***autocorrelation function (ACF)***.<br/>
**ACF** is an autocorrelation function that provides information about the amount of autocorrelation in a series with its lagged values. In other words, it describes how well present values are related to its past values. A time series consists of several components that include seasonality, trend, cycle, and residuals. The ACF takes all these factors into account while finding correlations, so this is the full auto-correlation plot.

**PACF** is the partial autocorrelation function. Unlike ACF, PACF finds correlations between residuals (the values that remain after removing the other effects) and the subsequent lag, which we will keep it as a feature in our models. thus, in order to avoid *overfitting* data for time series models, it is necessary to find optimum features or order of the autoregression process using the PACF plot. The best order is the lag value after which the PACF plot passes the upper confidence band for the first time. These p lags will act as the number of features used to forecast the time series. In the figure below,  lags up to six have a reasonable correlation before the plot first cuts the upper confidence interval. By combining the first six lags, we can model the given autoregression process.

 ![image](https://user-images.githubusercontent.com/96347878/188177323-4f2fab92-ef86-4bc1-9906-f62e00e4d8c3.png)
 
---
:four: **An explanation of data set splits**
- ***Train data set:*** A train data set represents the amount of data that machine learning models are fitted with.
- ***Validation data set:*** Validation data sets provide an unbiased evaluation of model fit on train data sets while tuning model hyperparameters.
- ***Test data set:*** A test data set is used to identify whether a model is underfitting (the model performs poorly on the train data set) or overfitting (the model performs well on the train data set but fails to perform well on the test data set). It is determined by looking at the prediction error on both train and test data sets. The test data set is only used after the train and validation data sets have been used to train and validate the model.<br/>
![image](https://user-images.githubusercontent.com/96347878/187898924-6b434403-bac1-41d8-ac6f-4acd9053e511.png)
---
:five: **Autoregressive and Automated Methods for Time Series Forecasting**

:black_medium_square: **Linear Regression Models:**<br/>
  - ***Linear Correlation:*** For two related variables, the correlation measures the association between the two variables. In contrast, a ***linear regression*** is used for the prediction of the value of one variable from another.
  - ***Linear Regression (LR):*** We can use the method of Linear Regression when we want to predict the value of one variable from the value(s) of one or more other variables. ***LR model:*** **$y_{t} = a_{0} + x_{t} + e_{t}$**
  - ***Least Squares Regression (LS):*** By minimizing the sum of all offsets or residuals from the plotted curve, the least squares method can be used to identify the best fit for a set of data points. Least squares regression is used for predicting the behavior of dependent variables.<br/> *LS model:*  $Coeff = (X^{T}X)^{-1}X^{T}y$
  - ***Moving Average (MA) Model:*** You can also forecast a series based solely on the past error values (et). Called short-memory models.<br/> *MA(p) model:* $y_{t} = a_{0} + e_{t} + a_{1}e_{t-1} + a_{2}e_{t-2} + ... + a_{p}e_{t-p}$
   - ***Autoregressive (AR) Model:*** The AR(p) notation refers to the autoregressive model which uses p history lag to predict the future. <br/>*AR(p) model:* $y_{t}  = a_{0} + a_{1}y_{t-1} + a_{2}y_{t-2} + ... + a_{p}y_{t-p} + e_{t}$
   - ***Autoregressive Exogenous (ARX) Model:*** The ARX model is a type of autoregressive model that includes an input term, unlike the AR model.<br/>*ARX(p, q) model:*  $y_{t} + a_{1}y_{t-1} + a_{2}y_{t-2} + ... + a_{p}y_{t-p} = b_{1}x_{t} + b_{2}x_{t-1} + ... + b_{p}x_{t-p} + e_{t}$
   - ***Auto-Regressive Integrated Moving Averages (ARIMA) Model:*** In statistics and in time series analysis, an ARIMA model is an update of ARMA (autoregressive moving average). The ARMA consists of mainly two components, the autoregressive and moving average; the ARIMA consists of an integrated moving average of autoregressive time series. ARIMA is used to help reduce the number of parameters needed for good estimation in the model.<br/>
*ARIMA(p,d,q):*  $y_{t}  = C + a_{1}y_{t-1} + a_{2}y_{t-2} + ... + a_{p}y_{t-p} + e_{t} + b_{1}e_{t-1} + b_{2}e_{t-2} + ... + b_{q}e_{t-q}$<br/>
:black_medium_small_square: p: The order of the AR model (i.e., the number of lag observations).<br/> 
:black_medium_small_square: d: The degree of differencing.<br/>
:black_medium_small_square: q: The order of the MA model. This is essentially the size of the “window” function over your time series data.

     |Models Name| Model Equation |
     |--|--|
     |*ARIMA (0, 1, 1) = IAM (1, 1) with constant*| $y_{t} = C + y_{t-1} + e_{t} + b_{1}e_{t-1}$|
     |*ARIMA (0, 1, 1) = IAM (1, 1)*| $y_{t} = y_{t-1} + e_{t} + b_{1}e_{t-1}$|
     |*ARIMA (0, 1, 2) with constant*| $y_{t} = C + y_{t-1} + e_{t} - a_{1}e_{t-1} - a_{2}e_{t-2}$|
     |*ARIMA (1, 1, 1) with constant*| $y_{t} = C + (1+a_{1})y_{t-1} + a_{1}y_{t-2} + e_{t} - b_{1}e_{t-1}$|
     |*ARIMA (1, 1, 1)*| $y_{t} = (1+a_{1})y_{t-1} + a_{1}y_{t-2} + e_{t} - b_{1}e_{t-1}$|
     |*ARIMA (0, 2, 2) with constant*| $y_{t} = C + 2y_{t-1} - y_{t-1} + e_{t} - b_{1}e_{t-1} - b_{2}e_{t-2}$|
Linear methods like AR, ARX, and ARIMA are popular classical techniques for time series forecasting. But these traditional approaches also have some constraints:<br/>
    :black_small_square: Focus on linear relationships and inability to find complex nonlinear ones.<br/>
    :black_small_square: Fixed lag observations and incapacity to make feature pre-processing.<br/>
    :black_small_square: Missing data & noise are not supported.<br/>
    :black_small_square: Working with univariate time series only, but common real-world problems have multiple input variables.<br/>
    :black_small_square: One-step predictions while many real-world problems require predictions with a long time horizon.<br/>

:black_medium_square: **Machine Learning for Time Series Forecasting: [Further information](https://github.com/RezaSaadatyar/Machine-Learning-in-Python)** 
  - ***Xgboost Regression***
  - ***Linear Regression***
  - ***Decision Trees (DT) Regression***
  - ***Random Forest (RF) Regression***
  
  ***The learning process is based on the following steps:***<br/>:black_small_square: Algorithms are fed data. (In this step you can provide additional information to the model, for example, by performing feature extraction).<br/>:black_small_square: Train a model using this data.<br/>:black_small_square: Test and deploy the model.<br/>:black_small_square: Utilize the deployed model to automate predictive tasks.

:black_medium_square: **Deep Learning for Time Series Forecasting: [Further information](https://github.com/RezaSaadatyar/Deep-Learning-in-python)** 
  - ***Long short-term memory (LSTM):***<br/> LSTM is an artificial recurrent neural network (RNN) architecture used in the field of deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections. LSTMs are sensitive to the scale of the input data, specifically when the sigmoid (default) or tanh activation functions are used. It can be a good practice to rescale the data to the range of 0 to 1, also called normalizing. We can easily normalize the dataset using the MinMaxscaler preprocessing class from the scikit-learn library.<br/><br/>***There are several types of x, including:***<br/>:black_small_square: *LSTM Autoenooder*<br/>:black_small_square: *Vanilla LSTM:* A Vanilla LSTM is an LSTM model that has a single hidden layer of LSTM units, and an output layer used to make a prediction.<br/>:black_small_square: *Stacked LSTM:* Multiple hidden LSTM layers can be stacked one on top of another in what is referred to as a stacked LSTM model.<br/>:black_small_square: *Bidirectional LSTM:* On some sequence prediction problem, it can be beneficial to allow the LSTM model to learn the input sequence both forward, backward and concatenate both interpretations.<br/><br/>***LSTM life-cycle in keras:***<br>:black_small_square: Define network.<br/>:black_small_square: compile network.<br/>:black_small_square: Fit network.<br/>:black_small_square: Evaluate network.<br/>:black_small_square: Make predictions
  
 ----
 **Install the required packages (if required)**<br/>:black_small_square: pip install numpy<br/>:black_small_square: pip install scipy<br/>:black_small_square: pip install pandas<br/>:black_small_square: pip install seaborn<br/>:black_small_square: pip install matplotlib<br/>:black_small_square: pip install scikit-learn<br/>:black_small_square: pip install keras
 
 ---
 **The five steps to effective time series forecasting are as follows:**
```
# ============================================= Import Libraries ========================================
import os
import numpy as np
import pandas as pd
from ARX_Model import arx
import statsmodels.api as sm
from AR_Model import ar_model
import matplotlib.pyplot as plt
from ARIMA_Model import arima_model
from Plot_Models import plot_models
from Least_Squares import lest_squares
from Normalize_Data import normalize_data
from Sequences_Data import sequences_data
from Test_Stationary import test_stationary
from Auto_Correlation import auto_correlation
from Linear_Regression import linear_regression
from Xgboost_Regression import xgboost_regression
from keras import models, layers, optimizers, utils
from Random_Forest_Regression import random_forest_regression
from Tree_Decision_Regression import tree_decision_regression
```
***Step 1:***
```
# ======================================== Step 1: Load Data ==================================================
os.system('cls')
data = sm.datasets.sunspots.load_pandas()  # df = pd.read_csv('monthly_milk_production.csv'), df.info(), X = df["Value"].values
data = data.data["SUNACTIVITY"]
# print('Shape of data \t', data.shape)
# print('Original Dataset:\n', data.head())
# print('Values:\n', data)
```
***Step 2:***
```
# ================================ Step 2.1: Normalize Data (0-1) ================================================
data, normalize = normalize_data(data, Type_Normalize='MinMaxScaler', Display_Figure='on')  # Type_Normalize: 'MinMaxScaler', 'normalize',
```
![Normalize_data](https://user-images.githubusercontent.com/96347878/188679304-9ad61a53-9f8d-44a4-94b3-b5cbe2e6c2af.png)

```
# ================================ Step 2.2: Check Stationary Time Series ========================================
data = test_stationary(data, window=20)
```
![Check Stationary Time Series](https://user-images.githubusercontent.com/96347878/188680819-9b1ac97e-086d-42b6-89af-e835492c0086.png)

***Step 3***:
```
# ==================================== Step 3: Find the lags of AR and etc models ==============================
auto_correlation(data, nLags=10)
```
![Find the lags of AR and etc models](https://user-images.githubusercontent.com/96347878/188682420-b9bf0369-9c31-4b19-a29d-3fdf2cfd113d.png)

***Step 4:***
```
# =========================== Step 4: Split Dataset intro Train and Test =======================================
nLags = 3
Data_Lags = pd.DataFrame(np.zeros((len(data), nLags)))
for i in range(0, nLags):
    Data_Lags[i] = data.shift(i + 1)
Data_Lags = Data_Lags[nLags:]
data = data[nLags:]
Data_Lags.index = np.arange(0, len(Data_Lags), 1, dtype=int)
data.index = np.arange(0, len(data), 1, dtype=int)
train_size = int(len(data) * 0.8)
```
***Step 5:***
```
# ================================= Step 5: Autoregressive and Automated Methods ===============================
sns.set(style='white')
fig, axs = plt.subplots(nrows=4, ncols=1, sharey='row', figsize=(16, 10))
plot_models(data, [], [], axs, nLags, train_size, num_sample=50, type_model='Actual_Data')
# -------------------------------------------  Least Squares ---------------------------------------------------
lest_squares(data, Data_Lags, train_size, axs, num_sample=50)
# -------------------------------------------- Auto-Regressive (AR) model --------------------------------------
ar_model(data, train_size, axs, n_lags=nLags, num_sample=50)
# ------------------------------------------------  ARX --------------------------------------------------------
arx(data, Data_Lags, train_size, axs, mu=0.9, num_sample=50)
# ----------------------------- Auto-Regressive Integrated Moving Averages (ARIMA) -----------------------------
arima_model(data, train_size, axs, order=(5, 1, (1, 1, 1, 1)), seasonal_order=(0, 0, 2, 12), num_sample=50)
# ======================================= Step 5: Machine Learning Models ======================================
# ------------------------------------------- Linear Regression Model  -----------------------------------------
linear_regression(data, Data_Lags, train_size, axs, num_sample=50)
# ------------------------------------------ RandomForestRegressor Model ---------------------------------------
random_forest_regression(data, Data_Lags, train_size, axs, n_estimators=100, max_features=nLags, num_sample=50)
# -------------------------------------------- Decision Tree Model ---------------------------------------------
tree_decision_regression(data, Data_Lags, train_size, axs, max_depth=2, num_sample=50)
# ---------------------------------------------- xgboost -------------------------------------------------------
xgboost_regression(data, Data_Lags, train_size, axs, n_estimators=1000, num_sample=50)
# -----------------------------------------------  LSTM model --------------------------------------------------
train_x, train_y = sequences_data(np.array(data[:train_size]), nLags)  # Convert to a time series dimension:[samples, nLags, n_features]
test_x, test_y = sequences_data(np.array(data[train_size:]), nLags)
mod = models.Sequential()  # Build the model
# mod.add(layers.ConvLSTM2D(filters=64, kernel_size=(1, 1), activation='relu', input_shape=(None, nLags)))  # ConvLSTM2D
# mod.add(layers.Flatten())
mod.add(layers.LSTM(units=100, activation='tanh', input_shape=(None, nLags)))
mod.add(layers.Dropout(rate=0.2))
# mod.add(layers.LSTM(units=100, activation='tanh'))  # Stacked LSTM
# mod.add(layers.Bidirectional(layers.LSTM(units=100, activation='tanh'), input_shape=(None, 1)))     # Bidirectional LSTM: forward and backward
mod.add(layers.Dense(32))
mod.add(layers.Dense(1))   # A Dense layer of 1 node is added in order to predict the label(Prediction of the next value)
mod.compile(optimizer='adam', loss='mse')
mod.fit(train_x, train_y, validation_data=(test_x, test_y), verbose=2, epochs=100)
y_train_pred = pd.Series(mod.predict(train_x).ravel())
y_test_pred = pd.Series(mod.predict(test_x).ravel())
y_train_pred.index = np.arange(nLags, len(y_train_pred)+nLags, 1, dtype=int)
y_test_pred.index = np.arange(train_size + nLags, len(data), 1, dtype=int)
plot_models(data, y_train_pred, y_test_pred, axs, nLags, train_size, num_sample=50, type_model='LSTM')
# data_train = normalize.inverse_transform((np.array(data_train)).reshape(-1, 1))
mod.summary(), plt.tight_layout(),plt.xticks(fontsize=15), plt.yticks(fontsize=15), plt.show()
```
![Autoregressive and Automated Methods](https://user-images.githubusercontent.com/96347878/188715358-252f5f1c-ee91-4003-ae82-0403d41fc001.png)

<table>
 <tr><th>Training </th><th>Test</th></tr>
<tr><td>
 
| Method | $${R^2}$$|
| ------ | ----- |
|LS  | 0.83 |  0.87 |
|AR  | 0.83 |  0.21 | 
|ARX | 0.95 | ***0.99***|   
|ARIMA| 0.84 | 0.25 | 
|LR| 0.83 | 0.87 |    
|DT | 0.67 | 0.56 |   
|RF | 0.97 | 0.84 | 
|XGBoost | 1 | 0.83 | 
|LSTM | 0.86 | 0.89 |
 
</td><td>

 $${R^2}$$|
| ----- |
|  0.87 |
|  0.21 | 
|0.99 |   
| 0.25 | 
| 0.87 |    
| 0.56 |   
| 0.84 | 
| 0.83 | 
| ***0.89*** |

</td></tr> </table>

Reza.Saadatyar92@gmail.com
