import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from Extract import extract
input_len = 16

def model_1():
    reg = Sequential()
    reg.add(LSTM(units = 100, return_sequences = True, input_shape = (input_len,1)))
    reg.add(Dropout(0.2))
    reg.add(LSTM(units = 100, return_sequences=True))
    reg.add(Dropout(0.2))
    reg.add(LSTM(units = 100, return_sequences= True))
    reg.add(Dropout(0.2))
    reg.add(LSTM(units = 100))
    reg.add(Dropout(0.2))
    reg.add(Dense(units=1))
    return reg

def set_generator():
    x_train = []
    y_train = []
    #x_test = []
    #y_test = []
    datasets = extract()
    df = pd.DataFrame()
    for i in range(12, datasets.shape[0]):
        x_train.append(datasets.iloc[i-12:i,:])
        y_train.append(datasets.iloc[i,-1])

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    #print(df)

    return x_train, y_train

def train_model():
    #reg = model_1()
    #reg.compile(optimizer='rmsprop', loss = 'mean_squared_error')
    x_train, y_train = set_generator()
    #x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    print(x_train.shape)
    print(x_train)
    #reg.fit(x_train, y_train, epochs = 50, batch_size = 32)
    return None


