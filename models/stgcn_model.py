from typing import Tuple

import numpy as np
import torch
from torch import nn


class SimpleSTGCN(nn.Module):
    """Lightweight spatio-temporal model: 1D temporal conv + linear spatial mixing.
    Fast enough to train on CPU in minutes for 207-node METR-LA."""

    def __init__(self, in_features: int, hidden_size: int, horizon: int):
        super().__init__()
        self.in_proj = nn.Linear(in_features, hidden_size)
        # Temporal: 1D conv over time axis
        self.temporal1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.temporal2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        # Spatial: simple linear mix across nodes (acts like a graph filter)
        self.spatial = nn.Linear(hidden_size, hidden_size)
        self.act = nn.ReLU()
        self.out_proj = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, nodes, features)
        B, T, N, F = x.shape
        x = self.in_proj(x)  # (B, T, N, H)

        # Temporal conv: reshape to (B*N, H, T)
        x = x.permute(0, 2, 3, 1).reshape(B * N, -1, T)
        x = self.act(self.temporal1(x))
        x = self.act(self.temporal2(x))
        # Take last timestep: (B*N, H)
        x = x[:, :, -1]
        x = x.reshape(B, N, -1)  # (B, N, H)

        # Spatial mixing
        x = self.act(self.spatial(x))  # (B, N, H)

        return self.out_proj(x)  # (B, N, horizon)


def create_tensors(x: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def train_one_epoch(model: nn.Module, optimizer, x: torch.Tensor, y: torch.Tensor, edge_index: torch.Tensor) -> float:
    model.train()
    optimizer.zero_grad()
    pred = model(x, edge_index)
    # y: (batch, horizon, nodes); pred: (batch, nodes, horizon)
    loss = ((pred.permute(0, 2, 1) - y) ** 2).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    return float(loss.detach().cpu().item())
