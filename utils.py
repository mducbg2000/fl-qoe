import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Any, List, Tuple
from functools import reduce

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from flwr.server.strategy import FedAvg

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.metrics import MeanSquaredError

NDArray = NDArray[Any]
NDArrays = List[NDArray]
LSTM_MODEL = Sequential


def read_raw_dataset(url: str,
                     index_col: str,
                     target_name: str,
                     drop: List[str] = []):
    dataset = pd.read_csv(url, index_col=index_col)
    drop_cols = drop + [target_name]
    X_raw = dataset.drop(labels=drop_cols, axis=1)
    y = dataset[target_name]
    return X_raw, y


def select_features(X: pd.DataFrame, y: pd.Series, k=10):
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X, y)
    return X[X.columns[selector.get_support(indices=True)]]


def read_chunk_and_split(id: int):
    X_raw, y = read_raw_dataset("datasets/chunk{}.csv".format(id), "ID",
                                "VMOS")
    X = select_features(X_raw, y)
    return train_test_split(X, y, train_size=0.8)


def build_model(number_of_feature) -> Sequential:
    model = Sequential()
    model.add(
        Bidirectional(LSTM(128, return_sequences=False),
                      input_shape=(number_of_feature, 1)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mse', metrics=[MeanSquaredError()])
    return model


def fed_avg(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [[layer * num_examples for layer in weights]
                        for weights, num_examples in results]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime
