import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from commonTools import *
from math import inf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

def build_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def detect_errors_nn(data):
    pressure = data['Pressure_Tank103']
    set_pressure = data['SetPressureTank103_manual']
    tlmTime = pd.to_datetime(data['MeasureTime'])

    scaled_data, scaler = preprocess_data(pressure.values.reshape(-1, 1))
    time_step = 100
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    train_size = 286#int(len(X) * 0.5)
    test_size = len(X) - train_size
    X_train, X_test = X[0:train_size], X[train_size:len(X)]
    y_train, y_test = y[0:train_size], y[train_size:len(y)]

    model = build_model()
    model.fit(X_train, y_train, batch_size=1, epochs=1)

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    train_residuals = y_train - train_predict
    test_residuals = y_test - test_predict

    faultTimes = []
    for i in range(len(test_predict)):
        if test_predict[i] < set_pressure.values[train_size + i + time_step + 1] - 30:
            faultTimes.append(tlmTime[train_size + i + time_step + 1])

    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    plt.plot(tlmTime, pressure, label='Pressure')
    plt.plot(tlmTime, set_pressure, label='Set Pressure')
    for fault_time in faultTimes:
        plt.axvline(x=fault_time, color='r', linestyle='--')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(tlmTime[time_step:train_size + time_step], train_residuals, label='Train Residuals')
    plt.plot(tlmTime[train_size + time_step:train_size + time_step + len(test_residuals)], test_residuals, label='Test Residuals')
    plt.legend()
    plt.show()

    return faultTimes

def main():
    fileName = 'DANE/F4_data.csv'
    script_dir = os.path.dirname(__file__)
    fileName = os.path.join(script_dir, fileName)
    cycleName = 'cycles_F4'

    tlmTime, data, cycles = read_data(fileName, cycleName, [-inf, inf])
    detectCycle(data)
    errorTimes = detect_errors_nn(data)
    print(f"Detected error times: {errorTimes}")

if __name__ == '__main__':
    main()
