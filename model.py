from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU
from keras.metrics import MeanSquaredError
from keras.initializers.initializers_v2 import Zeros
from typing import List
from numpy.typing import NDArray

Layers = List[NDArray]


def lstm():
    model = Sequential(layers=[
        LSTM(128, input_shape=(10, 1)),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64),
        Dropout(0.2),
        Dense(32),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=[MeanSquaredError()])
    return model


def bidirectional_lstm():
    model = Sequential(layers=[
        Bidirectional(LSTM(128),
                      input_shape=(10, 1),
                      kernel_initializer=Zeros(),
                      bias_initializer=Zeros()),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64),
        Dropout(0.2),
        Dense(32),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=[MeanSquaredError()])
    return model


def mlp():
    model = Sequential(layers=[
        Dense(16, activation='linear', input_shape=(10, 1)),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=[MeanSquaredError()])
    return model


def gru():
    model = Sequential(layers=[
        GRU(128, input_shape=(10, 1)),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64),
        Dropout(0.2),
        Dense(32),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=[MeanSquaredError()])
    return model