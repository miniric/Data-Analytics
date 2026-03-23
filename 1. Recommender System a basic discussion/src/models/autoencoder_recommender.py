"""
AutoEncoder-based Recommender (user-based and item-based).

FIX from original notebook:
  - Deduplicated 3 AutoEncoder + 3 AutoEncoderRecommender classes.
  - Fixed device mismatch in item-based variant.
  - Unified into one parameterised class with mode="user"|"item".
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix


class AutoEncoder(nn.Module):
    """
    Symmetric autoencoder with configurable layer widths.

    Parameters
    ----------
    input_dim : int
    encoding_dims : sequence of int
        Encoder layer widths (decoder mirrors them).
    dropout : float
    """

    def __init__(
        self,
        input_dim: int,
        encoding_dims: Sequence[int] = (256, 128, 64),
        dropout: float = 0.5,
    ):
        super().__init__()
        self.encoder = self._build(input_dim, list(encoding_dims), dropout)
        decoder_dims = list(reversed(encoding_dims[:-1])) + [input_dim]
        self.decoder = self._build(encoding_dims[-1], decoder_dims, dropout)

    @staticmethod
    def _build(in_dim: int, dims: list[int], dropout: float) -> nn.Sequential:
        layers: list[nn.Module] = []
        for d in dims:
            layers += [nn.Linear(in_dim, d), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = d
        # Remove trailing ReLU + Dropout from last layer
        return nn.Sequential(*layers[:-2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AutoEncoderRecommender:
    """
    Wrapper that handles training and prediction for AutoEncoder.

    Parameters
    ----------
    input_dim : int
        Number of items (user-based) or users (item-based).
    encoding_dims : sequence of int
    device : torch.device
    mode : "user" or "item"
        "user" = each row is a user's ratings over items.
        "item" = transpose so each row is an item's ratings over users.
    """

    def __init__(
        self,
        input_dim: int,
        encoding_dims: Sequence[int] = (256, 128, 64),
        device: torch.device | None = None,
        mode: str = "user",
    ):
        self.device = device or torch.device("cpu")
        self.mode = mode
        self.model = AutoEncoder(input_dim, encoding_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), weight_decay=1e-5)

    def fit(self, X: np.ndarray, epochs: int = 16, batch_size: int = 64,
            progress_callback=None) -> list[float]:
        """
        Train the autoencoder on a dense matrix X.

        Returns per-epoch loss history.
        """
        data = X.T if self.mode == "item" else X
        tensor = torch.FloatTensor(data).to(self.device)
        dataset = torch.utils.data.TensorDataset(tensor, tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        losses = []
        self.model.train(True)
        for epoch in range(epochs):
            epoch_loss = 0.0
            for bx, by in loader:
                self.optimizer.zero_grad()
                out = self.model(bx)
                loss = _sparse_mse(by, out)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            avg = epoch_loss / len(loader)
            losses.append(avg)
            if progress_callback:
                progress_callback(epoch, epochs, avg)
        return losses

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict reconstructed ratings for all users."""
        self.model.train(False)
        data = X.T if self.mode == "item" else X
        tensor = torch.FloatTensor(data).to(self.device)
        out = self.model(tensor).cpu().numpy()
        return out.T if self.mode == "item" else out

    @torch.no_grad()
    def predict_for_user(self, ratings_matrix: csr_matrix, user_index: int) -> np.ndarray:
        """Predict ratings for a single user."""
        self.model.train(False)
        if self.mode == "item":
            data = torch.FloatTensor(ratings_matrix.T.toarray()).to(self.device)
            out = self.model(data)
            return out.T[user_index].cpu().numpy()
        else:
            row = ratings_matrix[user_index].toarray()
            tensor = torch.FloatTensor(row).to(self.device)
            return self.model(tensor).cpu().numpy().flatten()


def _sparse_mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """MSE loss computed only on non-zero entries."""
    mask = (y_true != 0).float()
    return (mask * (y_true - y_pred) ** 2).mean()
