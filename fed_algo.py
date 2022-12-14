from pandas import DataFrame, Series
from functools import reduce
from typing import List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from utils import build_model, compute_loss

Layers = List[NDArray]
ClientParam = Tuple[Layers, int]


class FedAlgo(ABC):

    @abstractmethod
    def name() -> str:
        pass

    @abstractmethod
    def aggregate(self, clients: List[ClientParam]) -> Layers:
        pass

    @abstractmethod
    def predict(self) -> float:
        pass

    @abstractmethod
    def get_weights(self) -> Layers:
        pass


class FedAvg(FedAlgo):

    def __init__(self, init_weights: Layers, X_test: DataFrame,
                 y_test: Series) -> None:
        super().__init__()
        self.model = build_model()
        self.model.set_weights(init_weights)
        self.X_test = X_test
        self.y_test = y_test

    def name():
        return "FedAvg"

    def aggregate(self, clients: List[ClientParam]):
        weights = self.fed_avg(clients)
        self.model.set_weights(weights)
        return weights

    def predict(self) -> float:
        return compute_loss(self.model, self.X_test, self.y_test)

    def get_weights(self) -> Layers:
        return self.model.get_weights()

    def fed_avg(self, clients: List[ClientParam]) -> Layers:
        num_examples_total = sum([num_examples for _, num_examples in clients])

        weighted_weights = [[layer * num_examples for layer in layers]
                            for layers, num_examples in clients]

        weights_prime: Layers = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        return weights_prime


class FedAdam(FedAvg):

    def __init__(self, init_weights, X_test: DataFrame,
                 y_test: Series) -> None:
        super().__init__(init_weights, X_test, y_test)
        self.m_t = None
        self.v_t = None

    def name():
        return "FedAdam"

    def aggregate(self, clients: List[ClientParam]):
        return self.fed_adam(clients)

    def fed_adam(
        self,
        clients: List[ClientParam],
    ) -> Layers:
        fedavg_params_aggregated: Layers = self.fed_avg(clients)

        delta_t: Layers = [
            x - y
            for x, y in zip(fedavg_params_aggregated, self.model.get_weights())
        ]

        if not self.m_t:
            self.m_t = [np.zeros_like(x) for x in delta_t]
        self.m_t = [
            np.multiply(0.9, x) + 0.1 * y for x, y in zip(self.m_t, delta_t)
        ]

        if not self.v_t:
            self.v_t = [np.zeros_like(x) for x in delta_t]
        self.v_t = [
            0.99 * x + 0.01 * np.multiply(y, y)
            for x, y in zip(self.v_t, delta_t)
        ]

        new_weights = [
            x + (1e-1) * y / (np.sqrt(z) + (1e-9))
            for x, y, z in zip(self.model.get_weights(), self.m_t, self.v_t)
        ]

        self.model.set_weights(new_weights)
        return new_weights


class FedYogi(FedAvg):

    def __init__(self, init_weights, X_test: DataFrame,
                 y_test: Series) -> None:
        super().__init__(init_weights, X_test, y_test)
        self.m_t = None
        self.v_t = None

    def name():
        return "FedYogi"

    def aggregate(self, clients: List[ClientParam]):
        return self.fed_yogi(clients)

    def fed_yogi(self, clients: List[ClientParam]):
        fedavg_params_aggregated: Layers = self.fed_avg(clients)

        delta_t: Layers = [
            x - y
            for x, y in zip(fedavg_params_aggregated, self.model.get_weights())
        ]

        if not self.m_t:
            self.m_t = [np.zeros_like(x) for x in delta_t]
        self.m_t = [
            np.multiply(0.9, x) + 0.1 * y for x, y in zip(self.m_t, delta_t)
        ]

        if not self.v_t:
            self.v_t = [np.zeros_like(x) for x in delta_t]
        self.v_t = [
            x - 0.01 * np.multiply(y, y) * np.sign(x - np.multiply(y, y))
            for x, y in zip(self.v_t, delta_t)
        ]

        new_weights = [
            x + (1e-1) * y / (np.sqrt(z) + (1e-9))
            for x, y, z in zip(self.model.get_weights(), self.m_t, self.v_t)
        ]

        self.model.set_weights(new_weights)
        return new_weights


class FedAdagrad(FedAvg):

    def __init__(self, init_weights, X_test: DataFrame,
                 y_test: Series) -> None:
        super().__init__(init_weights, X_test, y_test)
        self.m_t = None
        self.v_t = None

    def name():
        return "FedAdagrad"

    def aggregate(self, clients: List[ClientParam]):
        return self.fed_adagrad(clients)

    def fed_adagrad(self, clients: List[ClientParam]):
        fedavg_params_aggregated: Layers = self.fed_avg(clients)

        delta_t: Layers = [
            x - y
            for x, y in zip(fedavg_params_aggregated, self.model.get_weights())
        ]

        if not self.m_t:
            self.m_t = [np.zeros_like(x) for x in delta_t]
        self.m_t = [
            np.multiply(0.9, x) + 0.1 * y for x, y in zip(self.m_t, delta_t)
        ]

        if not self.v_t:
            self.v_t = [np.zeros_like(x) for x in delta_t]
        self.v_t = [x + np.multiply(y, y) for x, y in zip(self.v_t, delta_t)]

        new_weights = [
            x + (1e-1) * y / (np.sqrt(z) + (1e-9))
            for x, y, z in zip(self.model.get_weights(), self.m_t, self.v_t)
        ]

        self.model.set_weights(new_weights)
        return new_weights