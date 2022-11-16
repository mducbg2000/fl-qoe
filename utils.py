import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras import regularizers
from keras.metrics import MeanSquaredError
from numpy.typing import NDArray
from typing import Any, List, Tuple
from functools import reduce

NDArray = NDArray[Any]
NDArrays = List[NDArray]


def load_data(file_name):
    dataset = pd.read_csv(file_name, index_col='ID')

    rand = np.random.rand(len(dataset)) < 0.8

    train = dataset[rand]
    test = dataset[~rand]
    X_train = train.drop(labels="VMOS", inplace=False, axis=1)
    y_train = train["VMOS"]

    X_test = test.drop(labels="VMOS", inplace=False, axis=1)
    y_test = test["VMOS"]

    return (X_train, y_train), (X_test, y_test)


def build_model(number_of_feature):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True,
              input_shape=(number_of_feature, 1)))
    model.add(Dense(128, activation='relu',
              activity_regularizer=regularizers.l2(1e-4)))
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(Dropout(0.2))
    # model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))

    model.compile(optimizer='adam',
                  loss='mse', metrics=[MeanSquaredError()])
    return model


def aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime
