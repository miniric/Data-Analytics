"""
Neural Matrix Factorisation with bias terms (improved MF).

FIX from original notebook:
  - Deduplicated ImprovedMF / ImprovedMF2 / ImprovedMF3 into one class.
  - WeightedMSELoss kept as-is (slightly upweights ratings > 3).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class NeuralMatrixFactorization(nn.Module):
    """
    Matrix factorisation with user/item embeddings, biases, and a global bias.

    Parameters
    ----------
    num_users, num_items : int
    embedding_dim : int
    """

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embedding_dim)
        self.item_embed = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.user_embed.weight, std=0.01)
        nn.init.normal_(self.item_embed.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(
        self, user_ids: torch.Tensor, item_ids: torch.Tensor
    ) -> torch.Tensor:
        u = self.user_embed(user_ids)
        i = self.item_embed(item_ids)
        dot = (u * i).sum(dim=1)
        return (
            dot
            + self.user_bias(user_ids).squeeze(-1)
            + self.item_bias(item_ids).squeeze(-1)
            + self.global_bias
        )

    @torch.no_grad()
    def predict_all_items(
        self, user_id: int, num_items: int, device: torch.device
    ) -> np.ndarray:
        self.train(False)
        uids = torch.full((num_items,), user_id, dtype=torch.long, device=device)
        iids = torch.arange(num_items, dtype=torch.long, device=device)
        return self(uids, iids).cpu().numpy()


class WeightedMSELoss(nn.Module):
    """MSE with higher weight for ratings > 3 (positive signal is more valuable)."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = torch.ones_like(target)
        weights = torch.where(target > 3, torch.tensor(1.2, device=target.device), weights)
        return (weights * (pred - target) ** 2).mean()
