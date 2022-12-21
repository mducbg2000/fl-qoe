import pandas as pd
from typing import List, Tuple

from sklearn.feature_selection import SelectKBest, f_regression
from numpy.typing import NDArray
from keras.models import Sequential
from sklearn.metrics import mean_squared_error

LSTM_MODEL = Sequential
NUMBER_OF_FEATURES = 10
Layers = List[NDArray]
ClientParam = Tuple[Layers, int]


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
    return X[select_features_name(X, y, k)]


def select_features_name(X: pd.DataFrame, y: pd.Series, k=10):
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X, y)
    return X.columns[selector.get_support(indices=True)]


def read_chunk(dataset: str, chunk_id,
               select_cols) -> Tuple[pd.DataFrame, pd.Series]:
    index_col = "id" if dataset == 'pokemon' else "ID"
    target_col = "MOS" if dataset == 'pokemon' else "VMOS"
    drop_cols = ["user_id"] if dataset == 'pokemon' else []
    X_raw, y = read_raw_dataset(
        "datasets/{}/chunk{}.csv".format(dataset, chunk_id), index_col,
        target_col, drop_cols)
    return X_raw[select_cols], y


def train_with_data(model, init_weights: Layers,
                    dataset: Tuple[pd.DataFrame, pd.Series]) -> ClientParam:
    model.set_weights(init_weights)
    model.fit(dataset[0],
              dataset[1],
              validation_split=0.05,
              batch_size=128,
              epochs=50,
              shuffle=True)
    return model.get_weights(), len(dataset[0])