"""
Networks for agent.
"""
from typing import Any

import torch
from numpy import ndarray, dtype
from torch import nn

LAYER_DIM = 64
P_DROPOUT = 0.5


class QNetwork(nn.Module):
    """
    Basic Q Network.
    """

    def __init__(self, state_dim: int, n_actions: int, seed: int):
        """
        Initialisation.
        :param state_dim: int. Dimension of state space.
        :param n_actions: int. Number of actions for one dimensional case containing several/n_actions actions.
        :param seed: int. Seed.
        """
        super().__init__()  # type:ignore

        self.seed = torch.manual_seed(seed)
        self.layers = nn.Sequential(
            nn.Linear(state_dim, LAYER_DIM),
            nn.ReLU(),
            nn.Linear(LAYER_DIM, LAYER_DIM),
            nn.ReLU(),
            nn.Linear(LAYER_DIM, n_actions)
        )

    def forward(self, state: ndarray[Any, dtype[Any]]) -> Any:
        """
        Forward step.
        :param state: ndarray[Any, dtype[Any]]. State to be propagated.
        :return: Any.
        """
        return self.layers(state)


class DuelingQNetwork(nn.Module):
    """
    Dueling network.
    """

    def __init__(self, state_dim: int, n_actions: int, seed: int):
        """
        Initialisation.
        :param state_dim: int. Dimension of state space.
        :param n_actions: int. Number of actions for one dimensional case containing several/n_actions actions.
        :param seed: int. Seed.
        """
        super().__init__()  # type:ignore

        self.seed = torch.manual_seed(seed)

        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, LAYER_DIM),
            nn.ReLU(),
            nn.Linear(LAYER_DIM, LAYER_DIM),
            nn.ReLU(),
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(LAYER_DIM, LAYER_DIM),
            nn.ReLU(),
            nn.Linear(LAYER_DIM, n_actions),
        )

        self.value_layer = nn.Sequential(
            nn.Linear(LAYER_DIM, LAYER_DIM),
            nn.ReLU(),
            nn.Linear(LAYER_DIM, 1),
        )

    def forward(self, state: ndarray[Any, dtype[Any]]) -> Any:
        """
        Forward step.
        :param state: ndarray[Any, dtype[Any]]. State to be propagated.
        :return: Any.
        """
        feature = self.feature_layer(state)

        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)

        q = value + advantage - advantage.mean(dim=-1, keepdim=True)

        return q


class QNetworkDropout(nn.Module):
    """
    Basic Q Network.
    """

    def __init__(self, state_dim: int, n_actions: int, seed: int):
        """
        Initialisation.
        :param state_dim: int. Dimension of state space.
        :param n_actions: int. Number of actions for one dimensional case containing several/n_actions actions.
        :param seed: int. Seed.
        """
        super().__init__()  # type:ignore

        self.seed = torch.manual_seed(seed)
        self.layers = nn.Sequential(
            nn.Linear(state_dim, LAYER_DIM),
            nn.Dropout(p=P_DROPOUT),
            nn.ReLU(),
            nn.Linear(LAYER_DIM, LAYER_DIM),
            nn.Dropout(p=P_DROPOUT),
            nn.ReLU(),
            nn.Linear(LAYER_DIM, n_actions)
        )

    def forward(self, state: ndarray[Any, dtype[Any]]) -> Any:
        """
        Forward step.
        :param state: ndarray[Any, dtype[Any]]. State to be propagated.
        :return: Any.
        """
        return self.layers(state)


class DuelingQNetworkDropout(nn.Module):
    """
    Dueling network.
    """

    def __init__(self, state_dim: int, n_actions: int, seed: int):
        """
        Initialisation.
        :param state_dim: int. Dimension of state space.
        :param n_actions: int. Number of actions for one dimensional case containing several/n_actions actions.
        :param seed: int. Seed.
        """
        super().__init__()  # type:ignore

        self.seed = torch.manual_seed(seed)

        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, LAYER_DIM),
            nn.Dropout(p=P_DROPOUT),
            nn.ReLU(),
            nn.Linear(LAYER_DIM, LAYER_DIM),
            nn.Dropout(p=P_DROPOUT),
            nn.ReLU(),
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(LAYER_DIM, LAYER_DIM),
            nn.Dropout(p=P_DROPOUT),
            nn.ReLU(),
            nn.Linear(LAYER_DIM, n_actions),
        )

        self.value_layer = nn.Sequential(
            nn.Linear(LAYER_DIM, LAYER_DIM),
            nn.Dropout(p=P_DROPOUT),
            nn.ReLU(),
            nn.Linear(LAYER_DIM, 1),
        )

    def forward(self, state: ndarray[Any, dtype[Any]]) -> Any:
        """
        Forward step.
        :param state: ndarray[Any, dtype[Any]]. State to be propagated.
        :return: Any.
        """
        feature = self.feature_layer(state)

        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)

        q = value + advantage - advantage.mean(dim=-1, keepdim=True)

        return q
