import pandas as pd
from typing import List, Tuple

from sklearn.feature_selection import SelectKBest, f_regression
from numpy.typing import NDArray
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.metrics import MeanSquaredError
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


def read_chunk(dataset: str, chunk_id: int,
               select_cols) -> Tuple[pd.DataFrame, pd.Series]:
    index_col = "id" if dataset == 'pokemon' else "ID"
    target_col = "MOS" if dataset == 'pokemon' else "VMOS"
    drop_cols = ["user_id"] if dataset == 'pokemon' else []
    X_raw, y = read_raw_dataset(
        "datasets/{}/chunk{}.csv".format(dataset, chunk_id), index_col,
        target_col, drop_cols)
    return X_raw[select_cols], y


def train_with_chunk(init_weights: Layers, dataset: str, chunk_id: int):
    X_train, y_train = read_chunk(dataset, chunk_id)
    lstm = build_model(NUMBER_OF_FEATURES)
    lstm.set_weights(init_weights)
    lstm.fit(X_train,
             y_train,
             validation_split=0.05,
             batch_size=128,
             epochs=50,
             shuffle=True)
    return ClientParam(lstm.get_weights(), len(X_train))


def train_with_data(init_weights: Layers,
                    dataset: Tuple[pd.DataFrame, pd.Series]) -> ClientParam:
    lstm = build_model(NUMBER_OF_FEATURES)
    lstm.set_weights(init_weights)
    lstm.fit(dataset[0],
             dataset[1],
             validation_split=0.05,
             batch_size=128,
             epochs=50,
             shuffle=True)
    return lstm.get_weights(), len(dataset[0])


def build_model(number_of_feature=10) -> Sequential:
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


def compute_loss(model: LSTM_MODEL, X_test, y_test) -> float:
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)
