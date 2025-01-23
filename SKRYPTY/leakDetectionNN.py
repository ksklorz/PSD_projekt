import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from commonTools import *
from math import inf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

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

def detect_leaks_nn(data):
    level_total = data['KQ1001_Level_B101'] + data['KQ1001_Level_B102']
    tlmTime = pd.to_datetime(data['MeasureTime'])

    scaled_data, scaler = preprocess_data(level_total.values.reshape(-1, 1))
    time_step = 100
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = build_model()
    model.fit(X, y, batch_size=1, epochs=1)

    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)

    faultTimes = []
    for i in range(len(predictions)):
        if predictions[i] < 190:
            faultTimes.append(tlmTime[i + time_step + 1])

    plt.plot(tlmTime, level_total)
    for fault_time in faultTimes:
        plt.axvline(x=fault_time, color='r', linestyle='--')
    plt.show()

    return faultTimes

def main():
    fileName = 'DANE/F3_data.csv'
    script_dir = os.path.dirname(__file__)
    fileName = os.path.join(script_dir, fileName)
    cycleName = 'cycles_F3'

    tlmTime, data, cycles = read_data(fileName, cycleName, [-inf, inf])
    detectCycle(data)
    leakTimes = detect_leaks_nn(data)
    print(f"Detected leak times: {leakTimes}")

if __name__ == '__main__':
    main()
