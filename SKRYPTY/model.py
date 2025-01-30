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
    trainData = data[['Pressure_Tank103', 'SetPressureTank103_manual']].values
    outData = data['Pressure_Tank103'].shift(-1).values

    scaler_train = MinMaxScaler(feature_range=(0, 1))
    scaler_out = MinMaxScaler(feature_range=(0, 1))
    trainData = scaler_train.fit_transform(trainData)
    outData = scaler_out.fit_transform(outData.reshape(-1, 1))

    XTrain1 = trainData[0:345]
    YTrain1 = outData[0:345]
    XTrain2 = trainData[634:910]
    YTrain2 = outData[634:910]
    XTrain = np.concatenate((XTrain1, XTrain2))
    YTrain = np.concatenate((YTrain1, YTrain2))

    XTest1 = trainData[1197:1480]
    YTest1 = outData[1197:1480]
    XTest2 = trainData[1764:2047]
    YTest2 = outData[1764:2047]

    XTest = np.concatenate((XTest1, XTest2))
    YTest = np.concatenate((YTest1, YTest2))

    XTrain = np.reshape(XTrain, (XTrain.shape[0], 1, XTrain.shape[1]))
    YTrain = np.reshape(YTrain, (YTrain.shape[0], 1, 1))

    XTest = np.reshape(XTest, (XTest.shape[0], 1, XTest.shape[1]))
    YTest = np.reshape(YTest, (YTest.shape[0], 1, 1))

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
    YTest = scaler_out.inverse_transform(YTest.flatten().reshape(-1, 1))
    YPred = scaler_out.inverse_transform(YPred.flatten().reshape(-1, 1))
    # YPred = scaler.inverse_transform(YPred.flatten().reshape(-1, 1))

    XTest = scaler_train.inverse_transform(XTest.flatten().reshape(-1, 2))

    plt.figure(figsize=(10, 5))
    plt.plot(YTest, 'b', label='Real')
    plt.plot(YPred, 'r', label='Predicted')
    # plt.plot(data['SetPressureTank103_manual'] , 'g', label='Setpoint')
    plt.plot(XTest[:, 1], 'g', label='Setpoint')
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
    plt.plot(Residuum, label='Residuum')
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
    th = AVG + Talpha * STD
    tl = AVG - Talpha * STD
    TH = th*np.ones(Residuum.size)
    TL = tl*np.ones(Residuum.size)
    
    
    plt.figure(figsize=(10, 5))
    plt.plot(Residuum)
    plt.plot(TH,'r-')
    plt.plot(TL,'r-')
    plt.xlabel('Sample number')
    plt.ylabel('Residuum')
    plt.grid(True)
    plt.savefig('FIG/model_residuum_limits.pdf')
    
    # plt.show()
    return model, scaler_train, scaler_out, th, tl

def test_model(dane, model, scaler_train, scaler_out, TH, TL, nazwa):
    test_data = dane[['Pressure_Tank103', 'SetPressureTank103_manual']].values
    test_data = scaler_train.transform(test_data)
    # test_data = scaler.transform(test_data)
    
    XTest = test_data
    YTest = dane['Pressure_Tank103'].shift(-1).values
    XTest = np.reshape(XTest, (XTest.shape[0], 1, XTest.shape[1]))

    YPred = model.predict(XTest)
    # YTest = scaler_out.inverse_transform(YTest.reshape(-1, 1))
    YTest = YTest.reshape(-1, 1)
    YPred = scaler_out.inverse_transform(YPred.flatten().reshape(-1, 1))
    # YPred = scaler.inverse_transform(YPred.flatten().reshape(-1, 1))

    Residuum = YTest - YPred
    plt.figure(figsize=(10, 5))
    plt.plot(Residuum, label='Residuum')
    plt.plot(TH*np.ones(Residuum.size),'r-')
    plt.plot(TL*np.ones(Residuum.size),'r-')
    plt.plot(dane['Error']*50, 'g', label='Error')
    # plt.legend()
    # plt.plot(YPred)
    # plt.plot(YTest)
    plt.xlabel('Sample number')
    plt.ylabel('Residuum')
    plt.grid(True)
    plt.title('Model LSTM na danych testowych')
    plt.savefig(f'FIG/model_test_{nazwa}.pdf')

def main():
    fileName = 'DANE/F2_data.csv'
    script_dir = os.path.dirname(__file__)
    fileName = os.path.join(script_dir, fileName)
    cycleName = 'cycles_F2'

    # plt.show

    tlmTime, data, cycles = read_data(fileName, cycleName, [-inf, inf])
    detectCycle(data)
    SetErrors(data, cycles)

    plt.figure(figsize=(10, 5))
    plt.plot(tlmTime, data['Pressure_Tank103'], label='Pressure')
    plt.plot(tlmTime, data['SetPressureTank103_manual'], label='Set Pressure')
    plt.legend()
    plt.grid(True)
    plt.savefig('FIG/model_data.pdf')
    
    model, scaler_train, scaler_out, TH, TL = train_model(data)

    test_model(data, model, scaler_train, scaler_out, TH, TL, 'F2')

    fileName = 'DANE/F1_data.csv'
    script_dir = os.path.dirname(__file__)
    fileName = os.path.join(script_dir, fileName)
    cycleName = 'cycles_F1'
    tlmTime, data, cycles = read_data(fileName, cycleName, [-inf, inf])
    SetErrors(data, cycles)
    test_model(data, model, scaler_train, scaler_out, TH, TL, 'F1')

    fileName = 'DANE/F3_data.csv'
    script_dir = os.path.dirname(__file__)
    fileName = os.path.join(script_dir, fileName)
    cycleName = 'cycles_F3'
    tlmTime, data, cycles = read_data(fileName, cycleName, [-inf, inf])
    SetErrors(data, cycles)
    test_model(data, model, scaler_train, scaler_out, TH, TL, 'F3')

    fileName = 'DANE/F4_data.csv'
    script_dir = os.path.dirname(__file__)
    fileName = os.path.join(script_dir, fileName)
    cycleName = 'cycles_F4'
    tlmTime, data, cycles = read_data(fileName, cycleName, [-inf, inf])
    SetErrors(data, cycles)
    test_model(data, model, scaler_train, scaler_out, TH, TL, 'F4')


    # plt.show()



if __name__ == '__main__':
    main()
