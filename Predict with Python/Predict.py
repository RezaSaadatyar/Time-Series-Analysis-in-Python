# ============================================= Import Libraries ========================================
import warnings
# import keras.optimizers
import numpy as np
import pandas as pd
import pmdarima as pm
from scipy import stats, ndimage
import matplotlib.pyplot as plt
from Output_Regression import output_regression
from Plot_Models import plot_models
from Sequences_Data import sequences
from Normalize_Data import normalize_data
from Auto_Correlation import auto_correlation
from Test_Stationary import test_stationary
import statsmodels.api as sm
import xgboost as xgb
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.api import VAR
from sklearn import linear_model, ensemble, tree
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels import tsa
from statsmodels.tsa.stattools import arma_order_select_ic
# from tensorflow import keras
# from keras import models, layers, optimizers, utils

# ======================================== Step 1: Load Data ==================================================
warnings.filterwarnings("default")  # "error", "ignore", "always", "default", "module"
data = sm.datasets.sunspots.load_pandas()  # df = pd.read_csv('monthly_milk_production.csv'), df.info(), X = df["Value"].values
data = data.data["SUNACTIVITY"]
# print('Shape of data \t', data.shape)
# print('Original Dataset:\n', data.head())
# print('Values:\n', data)
# ================================ Step 2: Normalize Data (0-1) ================================================
# data, normalize = normalize_data(data, Type_Normalize='MinMaxScaler', Display_Figure='off')  # Type_Normalize: 'MinMaxScaler', 'normalize',
# ================================ Step 2: Check Stationary Time Series ========================================
# data = test_stationary(data, window=20)
# ==================================== Step 3: Find the lags of AR and etc models ==============================
# auto_correlation(data, nLags=10)
# =========================== Step 4: Split Dataset intro Train and Test =======================================
nLags = 5
Data_Lags = pd.DataFrame(np.zeros((len(data), nLags)))
for i in np.arange(0, nLags):
    Data_Lags[i] = data.shift(i + 1)
Data_Lags = Data_Lags[nLags:]
data = data[nLags:]
Data_Lags.index = np.arange(0, len(Data_Lags), 1, dtype=int)
data.index = np.arange(0, len(data), 1, dtype=int)
train_size = int(len(data) * 0.8)
data_train = data[:train_size]
data_test = data[train_size:]


mod = xgb.XGBRegressor(n_estimators=1000)
y_train_pred, y_test_pred = output_regression(Data_Lags[:train_size],  Data_Lags[train_size:], mod, label_train=data_train)
plot_models(data, data_train, data_test, y_train_pred, y_test_pred, axs, nLags, i=50, Type_model='DT')


"""
mod = ExponentialSmoothing(data_train, seasonal='add')
a = mod.fit(optimized=True)
y = a.predict(len(data_test))


# --------------------------------------- Step 5: Auto-Regressive (AR) model -----------------------------------
mod = tsa.ar_model.AutoReg(data_train, lags=10).fit()
y_train_pred = pd.Series(mod.fittedvalues)                                                 # train
y_test_pred = pd.Series(mod.model.predict(mod.params, start=train_size, end=len(data)-1))  # For predict Future: end - start samples
fig, axs = plt.subplots(nrows=2, sharey='row', figsize=(12, 6))
plot_models(data, data_train, data_test, y_train_pred, y_test_pred, axs, nLags, i=50, Type_model='Actual_Data')
plot_models(data, data_train, data_test, y_train_pred, y_test_pred, axs, nLags, i=50, Type_model='AR')
# ----------------------- Step 5: Auto-Regressive Integrated Moving Averages (ARIMA) ------------------------------
# mod = pm.auto_arima(X[:train_size], start_p=5, start_q=1, seasonal=True, m=10, d=1, n_fits=50, information_criterion="bic", trace=True, stepwise=True, method='lbfgs')
ma = (1, 1, 1, 1)
mod = tsa.statespace.sarimax.SARIMAX(data_train, order=(5, 1, ma), seasonal_order=(0, 0, 2, 12))
mod = mod.fit(disp=False)
y_train_pred = pd.Series(mod.fittedvalues)
y_test_pred = mod.predict(start=train_size, end=len(data)-1, dynamic=True, typ='levels')  # predict N steps into the future
plot_models(data, data_train, data_test, y_train_pred, y_test_pred, axs, nLags, i=50, Type_model='ARIMA')
# --------------------------------------- Step 5: Linear Regression Model  ----------------------------------------
mod = linear_model.LinearRegression()
y_train_pred, y_test_pred = output_regression(Data_Lags[:train_size],  Data_Lags[train_size:], mod, label_train=data_train)
plot_models(data, data_train, data_test, y_train_pred, y_test_pred, axs, nLags, i=50, Type_model='LR')
# ------------------------------------- Step 5: RandomForestRegressor Models --------------------------------------
mod = ensemble.RandomForestRegressor(n_estimators=100, max_features=nLags, random_state=1)
y_train_pred, y_test_pred = output_regression(Data_Lags[:train_size],  Data_Lags[train_size:], mod, label_train=data_train)
plot_models(data, data_train, data_test, y_train_pred, y_test_pred, axs, nLags, i=50, Type_model='RF')
# ------------------------------------- Step 5: Decision Tree Models ---------------------------------------------
mod = tree.DecisionTreeRegressor(max_depth=2, random_state=0)
y_train_pred, y_test_pred = output_regression(Data_Lags[:train_size],  Data_Lags[train_size:], mod, label_train=data_train)
plot_models(data, data_train, data_test, y_train_pred, y_test_pred, axs, nLags, i=50, Type_model='DT')
# --------------------------------------- Step 5: xgboost --------------------------------------------------------
mod = xgb.XGBRegressor(n_estimators=1000)
y_train_pred, y_test_pred = output_regression(Data_Lags[:train_size],  Data_Lags[train_size:], mod, label_train=data_train)
plot_models(data, data_train, data_test, y_train_pred, y_test_pred, axs, nLags, i=50, Type_model='DT')
# ------------------------------------------ Step 5: MLP model ---------------------------------------------------
train_x, train_y = sequences(np.array(data_train), nLags)  # Convert to a time series dimension:[samples, nLags, n_features]
test_x, test_y = sequences(np.array(data_test), nLags)
"""
"""
mod = models.Sequential()  # Build the model
mod.add(layers.Dense(50, activation='relu', input_shape=(None, nLags), input_dim=nLags))
mod.add(layers.Dense(25, activation='relu'))
mod.add(layers.Dropout(0.2))             # Prevent over fitting
mod.add(layers.Dense(12, activation='relu'))
mod.add(layers.Dropout(0.2))
mod.add(layers.Dense(6, activation='relu'))
mod.add(layers.Dropout(0.2))
mod.add(layers.Dense(1, activation='relu'))
# utils.plot_model(mod, show_shapes=True, show_layer_names=True)
mod.compile(optimizer=optimizers.Adam(0.001), loss='mse')

mod.fit(train_x, train_y, validation_data=(test_x, test_y), validation_split=0.1, batch_size=2, verbose=1, epochs=20, shuffle=False)
y_train_pred = pd.Series(mod.predict(train_x).ravel())
y_test_pred = pd.Series(mod.predict(test_x).ravel())
plt.subplot(211)
plt.plot(data_train, 'b')
plt.plot(y_train_pred, 'r')
plt.subplot(212)
plt.plot(data_test, 'b')
plt.plot(y_test_pred, 'r')
plt.show()

# ----------------------------------------- Step 5: LSTM model ---------------------------------------------------
mod = models.Sequential()  # Build LSTM Autoencoder model
mod.add(layers.LSTM(units=128, activation='tanh', input_shape=(train_x.shape[1], train_x.shape[2])))
mod.add(layers.Dropout(rate=0.2))
mod.add(layers.RepeatVector(train_x.shape[1]))
mod.add(layers.LSTM(units=128, activation='tanh', return_sequences=True))
mod.add(layers.Dropout(rate=0.2))
mod.add(layers.TimeDistributed(layers.Dense(train_x.shape[2])))
mod.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
mod.fit(train_x, train_y, epochs=50, batch_size=5, validation_split=0.1, callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')], shuffle=False)
"""
"""
mod = models.Sequential()  # Build the model
# mod.add(layers.ConvLSTM2D(filters=64, kernel_size=(1, 1), activation='relu', input_shape=(None, nLags)))  # ConvLSTM2D
# mod.add(layers.Flatten())
mod.add(layers.LSTM(units=100, activation='tanh', input_shape=(None, nLags)))
# mod.add(layers.LSTM(units=100, activation='tanh'))  # Stacked LSTM
# mod.add(layers.Bidirectional(layers.LSTM(units=100, activation='tanh'), input_shape=(None, 1)))     # Bidirectional LSTM: forward and backward
mod.add(layers.Dense(32))
mod.add(layers.Dense(1))   # A Dense layer of 1 node is added in order to predict the label(Prediction of the next value)
mod.compile(optimizer='adam', loss='mse')
mod.fit(train_x, train_y, validation_data=(test_x, test_y), verbose=2, epochs=100)
y_train_pred = pd.Series(mod.predict(train_x).ravel())
y_test_pred = pd.Series(mod.predict(test_x).ravel())
y_train_pred.index = np.arange(nLags, len(y_train_pred)+nLags, 1, dtype=int)
y_test_pred.index = np.arange(len(data_train) + nLags, data_test.index[-1]+1, 1, dtype=int)
plot_models(data, data_train, data_test, y_train_pred, y_test_pred, axs, nLags, i=50, Type_model='LSTM')
# data_train = normalize.inverse_transform((np.array(data_train)).reshape(-1, 1))
mod.summary()
plt.tight_layout(), plt.style.use('ggplot'), plt.show()
"""