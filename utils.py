import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Any, List, Tuple
from functools import reduce

from sklearn.feature_selection import SelectKBest, f_regression

NDArray = NDArray[Any]
NDArrays = List[NDArray]


def read_raw_dataset(url: str, index_col: str, target_name: str, drop: List[str] = []):
    dataset = pd.read_csv(url, index_col=index_col)
    drop_cols = drop + [target_name]
    X_raw = dataset.drop(labels=drop_cols, axis=1)
    y = dataset[target_name]
    return X_raw, y


def select_features(X: pd.DataFrame, y: pd.Series, k=10):
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X, y)
    return X[X.columns[selector.get_support(indices=True)]], selector



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
