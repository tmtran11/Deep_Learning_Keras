# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 17:03:14 2017
Following instruction, I create a recurrent neural network to predct Google stock price
As a follow up, I implement by myself additional indicator, which is th Microsoft stock prices, in order to tuning up the model
csv files can be found in Deep_learning directory
@author: FPTShop
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# put cvs file in Pandas DataFrame
dataset_train = pd.read_csv('GOOG.csv')
training_set = dataset_train.iloc[:, 1:2].values
dataset_indicator = pd.read_csv('MSFT.csv')
indicator_set = dataset_indicator.iloc[:, 1:2].values

# Using Scaler from sklearn to transform to weight the training set according to one standard
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(training_set)
indicator_set_scaled = sc.transform(indicator_set)

# Using data of 120 days before to predict the stock price of one day
# Training set, Using Google stock price itself
X_train = []
y_train = []
for i in range(120, 1762):
    X_train.append(training_set_scaled[(i-120):i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Indicator set, Using Microsoftword stock price
# Expect this extra parameter to better tune the predictive model
X_indicator = []
y_indicator = []
for i in range(120, 1762):
    X_indicator.append(indicator_set_scaled[(i-120):i, 0])
    y_indicator.append(indicator_set_scaled[i, 0])
X_indicator, y_indicator = np.array(X_indicator), np.array(y_indicator)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_indicator = np.reshape(X_indicator, (X_indicator.shape[0], X_indicator.shape[1], 1))

# 120 column from training set + 120 column from indicator set
X_train = np.concatenate((X_train, X_indicator), axis = 2)

# Build recurrent neural neural network
from keras.models import Sequential # sequential layers
from keras.layers import Dense      # Basics Neural Network layers
from keras.layers import LSTM       # Long Short Term Memory cells 
from keras.layers import Dropout    # prevent over fitting


regressor = Sequential()

regressor.add(LSTM(units = 70, return_sequences = True, input_shape = (X_train.shape[1], 2)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 70, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 70, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 70, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 70, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 70))
regressor.add(Dropout(0.2))
# add 2 more layers
regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
# pass in by batches of 32 samples and iterate for 100 time

# prepare test set
dataset_test1 = pd.read_csv('GOOG_test.csv')
dataset_test2 = pd.read_csv('MSFT_test.csv')
real_stock_price_1 = dataset_test1.iloc[:, 1:2].values
real_stock_price_2 = dataset_test2.iloc[:, 1:2].values

dataset_total1 = pd.concat((dataset_train['Open'], dataset_test1['Open']), axis = 0)
dataset_total2 = pd.concat((dataset_indicator['Open'], dataset_test2['Open']), axis = 0)
inputs1 = dataset_total1[(len(dataset_total1)-len(dataset_test1)-120):].values
inputs1 = inputs1.reshape(-1, 1)
inputs1 = sc.transform(inputs1)
inputs2 = dataset_total2[(len(dataset_total2)-len(dataset_test2)-120):].values
inputs2 = inputs2.reshape(-1, 1)
inputs2 = sc.transform(inputs2)

X_test1 = []
for i in range(120, 367):
    X_test1.append(inputs1[(i-120):i, 0])   
X_test1 = np.array(X_test1)
X_test1 = np.reshape(X_test1, (X_test1.shape[0], X_test1.shape[1], 1))


X_test2 = []
for i in range(120, 367): 
    X_test2.append(inputs2[(i-120):i, 0])   
X_test2 = np.array(X_test2)
X_test2 = np.reshape(X_test2, (X_test2.shape[0], X_test2.shape[1], 1))

X_test = np.concatenate((X_test1, X_test2), axis = 2)

pred_stock_price = regressor.predict(X_test)
pred_stock_price = sc.inverse_transform(pred_stock_price)

# plot to see the result
plt.plot(real_stock_price_1, color = 'red', label = 'real')
plt.plot(pred_stock_price, color = 'blue', label = 'pred')
plt.legend()
plt.show()
