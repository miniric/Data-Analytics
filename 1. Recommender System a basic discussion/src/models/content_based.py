"""
Content-Based Filtering recommender using TF-IDF genre vectors.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


class ContentBasedRecommender:
    """
    Content-based recommender that builds user profiles from genre TF-IDF vectors
    and predicts ratings via dot-product similarity.
    """

    def __init__(self):
        self._tfidf = TfidfVectorizer(token_pattern=r"\b\w+\b")
        self.movie_features = None

    def fit(self, genres_series) -> "ContentBasedRecommender":
        """Fit TF-IDF on movie genres (pipe-separated strings)."""
        self.movie_features = self._tfidf.fit_transform(genres_series)
        return self

    def predict_for_user(self, user_ratings) -> np.ndarray:
        """
        Predict ratings for one user given their observed ratings vector.

        Uses a weighted-average user profile over rated movie features.
        """
        ratings = _to_dense(user_ratings)
        profile = self._build_user_profile(ratings)
        return self._predict_from_profile(profile)

    def predict_cold_start(self, ratings_matrix) -> np.ndarray:
        """
        Cold-start prediction: average profile over ALL users in the matrix.
        """
        profiles = []
        n_users = ratings_matrix.shape[0]
        for i in range(n_users):
            row = ratings_matrix[i]
            ratings = _to_dense(row)
            profiles.append(self._build_user_profile(ratings))
        avg_profile = np.mean(profiles, axis=0)
        return self._predict_from_profile(avg_profile)

    # ----- internal -----

    def _build_user_profile(self, ratings: np.ndarray) -> np.ndarray:
        rated_idx = np.nonzero(ratings)[0]
        if len(rated_idx) == 0:
            return np.zeros(self.movie_features.shape[1])

        rated_features = self.movie_features[rated_idx]
        weights = ratings[rated_idx]

        weighted = rated_features.multiply(weights[:, np.newaxis])
        profile = np.asarray(weighted.sum(axis=0)).flatten()
        profile /= weights.sum()
        return profile

    def _predict_from_profile(self, profile: np.ndarray) -> np.ndarray:
        raw = np.asarray(self.movie_features.dot(profile)).flatten()
        # Normalise to [1, 5]
        r_min, r_max = raw.min(), raw.max()
        if r_max - r_min < 1e-10:
            return np.full_like(raw, 3.0)
        return 1.0 + 4.0 * (raw - r_min) / (r_max - r_min)


def _to_dense(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.toarray()).flatten()
    return np.asarray(x).flatten()
