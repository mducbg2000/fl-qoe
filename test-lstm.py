from typing import Tuple
from sklearn.metrics import mean_squared_error
import pandas as pd
from utils import select_features, build_model, NDArrays, LSTM_MODEL, read_raw_dataset, fed_avg

TRAIN_RESULT = Tuple[LSTM_MODEL, int]
AGGREGATE_PARAM = Tuple[NDArrays, int]
NUMBER_OF_FEATURES = 10


def read_chunk(id: int) -> Tuple[pd.DataFrame, pd.Series]:
    X_raw, y = read_raw_dataset(
        "datasets/pokemon_users/chunk{}.csv".format(id), "id", "MOS",
        ["user_id"])
    X = select_features(X_raw, y, k=NUMBER_OF_FEATURES)
    return X, y


def train(id: int, init_weights: NDArrays) -> TRAIN_RESULT:
    X_train, y_train = read_chunk(id)
    lstm = build_model(NUMBER_OF_FEATURES)
    lstm.set_weights(init_weights)
    lstm.fit(X_train,
             y_train,
             validation_split=0.05,
             batch_size=128,
             epochs=50,
             shuffle=True)
    return lstm, len(X_train)


def compute_loss(model: LSTM_MODEL, X_test, y_test) -> float:
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)


if __name__ == "__main__":
    global_model: LSTM_MODEL = build_model(NUMBER_OF_FEATURES)
    init_weights = global_model.get_weights()

    train_results = [train(id + 1, init_weights) for id in range(180)]

    global_weights = fed_avg([(result[0].get_weights(), result[1])
                              for result in train_results])

    global_model.set_weights(global_weights)

    X_test, y_test = read_chunk(181)

    local_mse = [
        compute_loss(model, X_test, y_test) for (model, _) in train_results
    ]

    global_mse = compute_loss(global_model, X_test, y_test)

    print("Local mse: \n")
    print(local_mse)
    print("Global mse: \n")
    print(global_mse)
