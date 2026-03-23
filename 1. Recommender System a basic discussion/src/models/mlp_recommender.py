"""
MLP-based Matrix Factorisation Recommender.

FIX from original notebook:
  - Deduplicated 3 near-identical classes into one parameterised class.
  - Mutable default argument (list) replaced with tuple.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset


# ==========================================================================
# Dataset
# ==========================================================================

class RatingDataset(Dataset):
    """PyTorch Dataset wrapping a sparse user-item matrix (non-zero entries only)."""

    def __init__(self, user_item_matrix: csr_matrix):
        users, items = user_item_matrix.nonzero()
        self.users = users.astype(np.int64)
        self.items = items.astype(np.int64)
        self.ratings = np.asarray(
            user_item_matrix[users, items]
        ).flatten().astype(np.float32)

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int):
        return self.users[idx], self.items[idx], self.ratings[idx]


# ==========================================================================
# Model
# ==========================================================================

class MLPRecommender(nn.Module):
    """
    MLP-based collaborative filtering.

    Concatenates user and item embeddings, passes through fully-connected
    layers, outputs a scalar rating prediction.

    Parameters
    ----------
    num_users, num_items : int
    embedding_dim : int
    hidden_layers : sequence of int
        E.g. (32, 16, 8, 4, 2) — each element is a hidden layer width.
    dropout : float
    l2_lambda : float
        Weight for manual L2 regularisation term added to the loss.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        hidden_layers: Sequence[int] = (32, 16, 8, 4, 2),
        dropout: float = 0.5,
        l2_lambda: float = 0.001,
    ):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embedding_dim)
        self.item_embed = nn.Embedding(num_items, embedding_dim)
        self.l2_lambda = l2_lambda

        layers: list[nn.Module] = []
        in_dim = embedding_dim * 2
        for h in hidden_layers:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        u = self.user_embed(user_ids)
        i = self.item_embed(item_ids)
        x = torch.cat([u, i], dim=1)
        return self.mlp(x).squeeze(-1)

    def l2_regularization(self) -> torch.Tensor:
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for p in self.parameters():
            total = total + torch.norm(p, 2)
        return self.l2_lambda * total

    @torch.no_grad()
    def predict_all_items(self, user_id: int, num_items: int, device: torch.device) -> np.ndarray:
        """Predict ratings for *one* user across all items."""
        self.set_eval_mode()
        uids = torch.full((num_items,), user_id, dtype=torch.long, device=device)
        iids = torch.arange(num_items, dtype=torch.long, device=device)
        return self(uids, iids).cpu().numpy()

    def set_eval_mode(self):
        """Switch to evaluation mode (disables dropout, batchnorm)."""
        self.training = False
        for module in self.children():
            module.training = False
        return self

    def set_train_mode(self):
        """Switch to training mode."""
        self.training = True
        for module in self.children():
            module.training = True
        return self
