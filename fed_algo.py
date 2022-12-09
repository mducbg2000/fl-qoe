from dataclasses import dataclass
from numpy.typing import NDArray
from functools import reduce
from typing import Any, List, Optional, Tuple
import numpy as np

NDArray = NDArray[Any]
NDArrays = List[NDArray]


@dataclass(frozen=True)
class ClientParam:
    layers: NDArrays
    num_examples: int


def fed_avg(clients: List[ClientParam]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([client.num_examples for client in clients])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * client.num_examples for layer in client.layers] for client in clients
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


def fed_adam(
    clients: List[ClientParam],
    old_global_weights: NDArrays,
    old_m_t: Optional[NDArrays] = None,
    old_v_t: Optional[NDArrays] = None
):
    fedavg_params_aggregated: NDArrays = fed_avg(clients)
    delta_t: NDArrays = [
        x-y for x, y in zip(fedavg_params_aggregated, old_global_weights)
    ]

    m_t = old_m_t or [np.zeros_like(x) for x in delta_t]
    new_m_t = [
        np.multiply(0.9, x) + 0.1*y 
        for x, y in zip(m_t, delta_t)
    ]

    v_t = old_v_t or [np.zeros_like(x) for x in delta_t]
    new_v_t = [
        0.99*x + 0.01*np.multiply(y, y) 
        for x, y in zip(v_t, delta_t)
    ]

    new_weights = [
        x + (1e-1)*y / (np.sqrt(z) + (1e-9))
        for x, y, z in zip(old_global_weights, new_m_t, new_v_t)
    ]

    return new_weights, new_m_t, new_v_t


def fed_yogi(
    clients: List[ClientParam],
    old_global_weights: NDArrays,
    old_m_t: Optional[NDArrays] = None,
    old_v_t: Optional[NDArrays] = None
):
    fedavg_params_aggregated: NDArrays = fed_avg(clients)

    delta_t: NDArrays = [
        x-y for x, y in zip(fedavg_params_aggregated, old_global_weights)
    ]

    m_t = old_m_t or [np.zeros_like(x) for x in delta_t]
    new_m_t = [
        np.multiply(0.9, x) + 0.1*y
        for x, y in zip(m_t, delta_t)
    ]

    v_t = old_v_t or [np.zeros_like(x) for x in delta_t]
    new_v_t = [
        x - 0.01*np.multiply(y, y)*np.sign(x-np.multiply(y, y))
        for x, y in zip(v_t, delta_t)
    ]

    new_weights = [
        x + (1e-1)*y / (np.sqrt(z) + (1e-9))
        for x, y, z in zip(old_global_weights, new_m_t, new_v_t)
    ]

    return new_weights, new_m_t, new_v_t

def fed_adagrad(    
    clients: List[ClientParam],
    old_global_weights: NDArrays,
    old_m_t: Optional[NDArrays] = None,
    old_v_t: Optional[NDArrays] = None
):
    fedavg_params_aggregated: NDArrays = fed_avg(clients)

    delta_t: NDArrays = [
        x-y for x, y in zip(fedavg_params_aggregated, old_global_weights)
    ]

    m_t = old_m_t or [np.zeros_like(x) for x in delta_t]
    new_m_t = [
        np.multiply(0, x) + y
        for x, y in zip(m_t, delta_t)
    ]

    v_t = old_v_t or [np.zeros_like(x) for x in delta_t]
    new_v_t = [
        x + np.multiply(y, y)
        for x, y in zip(v_t, delta_t)
    ]

    new_weights = [
        x + (1e-1) * y / (np.sqrt(z) + 1e-9)
        for x,y,z in zip(old_global_weights, new_m_t, new_v_t)
    ]

    return new_weights, new_m_t, new_v_t