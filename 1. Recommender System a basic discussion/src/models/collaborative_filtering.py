"""
User-based Collaborative Filtering with multiple similarity metrics.

FIX from original notebook:
  - Eliminated O(n*m) user lookup via np.where (now uses direct index).
  - Deduplicated 3 nearly-identical predict functions into one base class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity


class _BaseCollaborativeFilter(ABC):
    """Base class for user-based collaborative filtering."""

    def __init__(self, k: int = 5):
        self.k = k
        self.similarity_matrix = None

    def fit(self, ratings_matrix: sparse.csr_matrix) -> "_BaseCollaborativeFilter":
        self.ratings_matrix = ratings_matrix
        self.similarity_matrix = self._compute_similarity(ratings_matrix)
        return self

    @abstractmethod
    def _compute_similarity(self, matrix: sparse.csr_matrix) -> np.ndarray:
        ...

    def predict_for_user(self, user_index: int) -> np.ndarray:
        """Predict ratings for a single user by index (not user ID)."""
        sim_scores = self.similarity_matrix[user_index]
        # Top-k similar users (exclude self)
        neighbours = np.argsort(sim_scores)[::-1]
        neighbours = neighbours[neighbours != user_index][: self.k]

        n_items = self.ratings_matrix.shape[1]
        predictions = np.zeros(n_items)

        for item in range(n_items):
            neighbour_ratings = self.ratings_matrix[neighbours, item].toarray().flatten()
            rated_mask = neighbour_ratings > 0
            if rated_mask.any():
                predictions[item] = neighbour_ratings[rated_mask].mean()
            else:
                col_data = self.ratings_matrix[:, item].data
                predictions[item] = col_data.mean() if len(col_data) > 0 else 3.0

        return predictions

    def predict_cold_start(self) -> np.ndarray:
        """Cold-start: per-item mean rating across all users."""
        n_items = self.ratings_matrix.shape[1]
        means = np.zeros(n_items)
        for item in range(n_items):
            col_data = self.ratings_matrix[:, item].data
            means[item] = col_data.mean() if len(col_data) > 0 else 2.5
        return means


class CosineCollaborativeFilter(_BaseCollaborativeFilter):
    """Standard cosine similarity on the full rating vectors."""

    def _compute_similarity(self, matrix: sparse.csr_matrix) -> np.ndarray:
        dense = matrix.toarray()
        return cosine_similarity(dense)


class AdjustedCosineCollaborativeFilter(_BaseCollaborativeFilter):
    """Cosine similarity computed only on co-rated items (min_common_items filter)."""

    def __init__(self, k: int = 5, min_common: int = 3):
        super().__init__(k)
        self.min_common = min_common

    def _compute_similarity(self, matrix: sparse.csr_matrix) -> np.ndarray:
        dense = matrix.toarray()
        n = dense.shape[0]
        sim = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                mask = (dense[i] != 0) & (dense[j] != 0)
                if mask.sum() >= self.min_common:
                    ui, uj = dense[i, mask], dense[j, mask]
                    denom = np.linalg.norm(ui) * np.linalg.norm(uj)
                    if denom > 0:
                        sim[i, j] = sim[j, i] = np.dot(ui, uj) / denom
        return sim


class PearsonCollaborativeFilter(_BaseCollaborativeFilter):
    """Pearson correlation on co-rated items, with mean-centred predictions."""

    def _compute_similarity(self, matrix: sparse.csr_matrix) -> np.ndarray:
        dense = matrix.toarray()
        n = dense.shape[0]
        sim = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                mask = (dense[i] != 0) & (dense[j] != 0)
                if mask.sum() > 1:
                    ri, rj = dense[i, mask], dense[j, mask]
                    if np.std(ri) == 0 or np.std(rj) == 0:
                        sim[i, j] = 1.0 if np.array_equal(ri, rj) else 0.0
                    else:
                        corr, _ = pearsonr(ri, rj)
                        sim[i, j] = corr
                    sim[j, i] = sim[i, j]
        return np.nan_to_num(sim)

    def predict_for_user(self, user_index: int) -> np.ndarray:
        """Mean-centred weighted prediction (Resnick formula)."""
        sim_scores = self.similarity_matrix[user_index]
        neighbours = np.argsort(sim_scores)[::-1]
        neighbours = neighbours[neighbours != user_index][: self.k]

        dense = self.ratings_matrix.toarray()
        user_ratings = dense[user_index]
        user_mean = user_ratings[user_ratings != 0].mean() if (user_ratings != 0).any() else 3.0

        n_items = self.ratings_matrix.shape[1]
        predictions = np.zeros(n_items)

        neighbour_means = np.array([
            dense[n][dense[n] != 0].mean() if (dense[n] != 0).any() else 0.0
            for n in neighbours
        ])

        for item in range(n_items):
            nr = dense[neighbours, item]
            rated = nr > 0
            if rated.any():
                weights = sim_scores[neighbours[rated]]
                deviations = nr[rated] - neighbour_means[rated]
                w_sum = np.abs(weights).sum()
                if w_sum > 0:
                    predictions[item] = user_mean + np.dot(weights, deviations) / w_sum
                else:
                    predictions[item] = user_mean
            else:
                col_data = self.ratings_matrix[:, item].data
                predictions[item] = col_data.mean() if len(col_data) > 0 else 3.5

        return predictions
