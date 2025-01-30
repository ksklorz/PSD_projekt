from commonTools import *
from math import inf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

def train_model(data):
    trainData = data['Pressure_Tank103'].values - data['SetPressureTank103_manual'].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    trainData = scaler.fit_transform(trainData.reshape(-1, 1)).flatten()

    XTrain1 = trainData[0:345]
    YTrain1 = trainData[1:346]
    XTrain2 = trainData[634:910]
    YTrain2 = trainData[635:911]
    XTrain = np.concatenate((XTrain1, XTrain2))
    YTrain = np.concatenate((YTrain1, YTrain2))

    XTest1 = trainData[1197:1480]
    YTest1 = trainData[1198:1481]
    XTest2 = trainData[1764:2047]
    YTest2 = trainData[1765:2048]

    XTest = np.concatenate((XTest1, XTest2))
    YTest = np.concatenate((YTest1, YTest2))

    XTrain = np.reshape(XTrain, (XTrain.size, 1, 1))
    YTrain = np.reshape(YTrain, (YTrain.size, 1, 1))

    XTest = np.reshape(XTest, (XTest.size, 1, 1))
    YTest = np.reshape(YTest, (YTest.size, 1, 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, activation='tanh', input_shape=(XTrain.shape[1], XTrain.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1, activation='linear'))

    optim = Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.99,
        epsilon=1e-06
    )

    model.compile(optimizer=optim, loss='mse', metrics=['mse'])
    hist = model.fit(XTrain, YTrain, epochs=30, batch_size=32, verbose=1, validation_split=0.1)

    YPred = model.predict(XTest)

    YTest = scaler.inverse_transform(YTest.flatten().reshape(-1, 1))
    YPred = scaler.inverse_transform(YPred.flatten().reshape(-1, 1))

    plt.figure(figsize=(10, 5))
    plt.plot(YTest, 'b', label='Real')
    plt.plot(YPred, 'r', label='Predicted')
    plt.legend()
    plt.xlabel('Sample number')
    plt.ylabel('Pressure')
    plt.grid(True)
    plt.title('Model LSTM na danych testowych')
    plt.savefig('FIG/model_learn.pdf')
    # plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(hist.history['loss'],'b-', label='Loss for Training Set')
    plt.plot(hist.history['val_loss'],'r-', label='Loss for Validation Set')
    plt.legend() 
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.grid(True)
    plt.title('Model LSTM - Loss')
    plt.savefig('FIG/model_loss.pdf')
    # plt.show()

    model.summary()

    Residuum = YTest - YPred
    plt.figure(figsize=(10, 5))
    plt.plot(Residuum, 'r', label='Residuum')
    # plt.legend()
    plt.xlabel('Sample number')
    plt.ylabel('Pressure')
    plt.grid(True)
    plt.title('Residuum')
    plt.savefig('FIG/model_residuum.pdf')
    # plt.show()

    plt.figure(figsize=(10, 5))
    plt.hist(Residuum, bins=30)
    plt.savefig('FIG/model_residuum_hist.pdf')

    AVG = np.mean(Residuum)
    STD = np.std(Residuum)
    Talpha = 3
    TH = (AVG+Talpha*STD)*np.ones(Residuum.size)
    TL = (AVG-Talpha*STD)*np.ones(Residuum.size)

    plt.figure(figsize=(10, 5))
    plt.plot(Residuum)
    plt.plot(TH,'r-')
    plt.plot(TL,'r-')
    plt.xlabel('Sample number')
    plt.ylabel('Residuum')
    plt.grid(True)
    plt.savefig('FIG/model_residuum_limits.pdf')
    
    # plt.show()
    return model, scaler


def test_model(dane, model, scaler):

    test_data = dane['Pressure_Tank103'].values # - dane['SetPressureTank103_manual'].values
    test_data = scaler.transform(test_data.reshape(-1, 1)).flatten()
    XTest = test_data[0:-1]
    YTest = test_data[1:]
    XTest = np.reshape(XTest, (XTest.size, 1, 1))
    YTest = np.reshape(YTest, (YTest.size, 1, 1))

    YPred = model.predict(XTest)
    YTest = scaler.inverse_transform(YTest.flatten().reshape(-1, 1))
    YPred = scaler.inverse_transform(YPred.flatten().reshape(-1, 1))

    Residuum = YTest - YPred
    plt.figure(figsize=(10, 5))
    plt.plot(Residuum, 'r', label='Residuum')
    # plt.legend()
    plt.xlabel('Sample number')
    plt.ylabel('Residuum')
    plt.grid(True)
    plt.title('Model LSTM na danych testowych')
    plt.savefig('FIG/model_test.pdf')
    plt.show()


def main():
    fileName = 'DANE/F2_data.csv'
    script_dir = os.path.dirname(__file__)
    fileName = os.path.join(script_dir, fileName)
    cycleName = 'cycles_F2'

    tlmTime, data, cycles = read_data(fileName, cycleName, [-inf, inf])
    detectCycle(data)
    SetErrors(data, cycles)
    
    model, scaler = train_model(data)
    test_model(data, model, scaler)

    

if __name__ == '__main__':
    main()
