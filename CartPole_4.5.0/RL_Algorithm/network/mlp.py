"""
network/mlp.py

Shared MLP (Multi-Layer Perceptron) backbone used as the foundation for
actor and critic networks across multiple algorithms (AC, A2C, PPO, TD3, SAC).

Build this before implementing any of those algorithm files.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Configurable fully-connected network.

    Architecture:
        input → [Linear → Activation] × len(hidden_dims) → Linear → output

    Args:
        input_dim   (int):       Input feature size.
        output_dim  (int):       Output feature size.
        hidden_dims (list[int]): Width of each hidden layer, e.g. [256, 256].
        activation  (str):       'relu' | 'elu' | 'tanh'

    Example:
        >>> net = MLP(4, 1, [256, 256], activation='relu')
        >>> net(torch.zeros(8, 4)).shape
        torch.Size([8, 1])
    """

    _ACTIVATIONS = {
        'relu': nn.ReLU,
        'elu':  nn.ELU,
        'tanh': nn.Tanh,
    }

    def __init__(
        self,
        input_dim:   int,
        output_dim:  int,
        hidden_dims: list = [256, 256],
        activation:  str  = 'relu',
    ):
        super().__init__()

        if activation not in self._ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation '{activation}'. "
                f"Choose from {list(self._ACTIVATIONS.keys())}."
            )

        act_cls = self._ACTIVATIONS[activation]

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act_cls())
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)