"""
Standard MLP utility layers shared across the project.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def mlp_block(in_dim: int, out_dim: int, activation=nn.ReLU) -> nn.Sequential:
    """Single linear layer + batch-norm + activation."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        activation(),
    )


def build_mlp(
    in_dim: int,
    hidden_sizes: Tuple[int, ...],
    out_dim: int,
    activation=nn.ReLU,
) -> nn.Sequential:
    """Build a multi-layer perceptron with LayerNorm after each hidden layer."""
    layers: list = []
    prev = in_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.LayerNorm(h))
        layers.append(activation())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)
