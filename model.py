from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.metrics import MeanSquaredError
from typing import List
from numpy.typing import NDArray

Layers = List[NDArray]


def lstm():
    model = Sequential(layers=[
        LSTM(128, return_sequences=False, input_shape=(10, 1)),
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
        Bidirectional(LSTM(128, return_sequences=False), input_shape=(10, 1)),
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
        Dense(32, input_shape=(10, 1), activation='relu'),
        Dense(16),
        Dense(8),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=[MeanSquaredError()])
    return model


def svr():
    model = Sequential(layers=[
        Dense(64, input_shape=(10, 1), activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=[MeanSquaredError()])
    return model