"""
Data loading, preprocessing, and train/test splitting for the Recommender System.

Responsibilities:
  - Load MovieLens CSV files into a sparse utility matrix
  - Sample users and remove inactive ones
  - Split into train, warm-test, and cold-test matrices
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .config import DataConfig


# ---------------------------------------------------------------------------
# Immutable data container returned by the data pipeline
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RecommenderData:
    """Immutable container holding train/test split matrices."""
    train_matrix: csr_matrix          # Ratings used for training (warm users only)
    test_warm: csr_matrix             # Held-out ratings for warm users
    test_cold: csr_matrix             # All ratings for cold users (evaluation only)
    warm_user_indices: np.ndarray     # Row indices of warm users
    cold_user_indices: np.ndarray     # Row indices of cold users
    user_id_map: Dict[int, int]       # original_user_id -> matrix_row_index
    movie_id_map: Dict[int, int]      # original_movie_id -> matrix_col_index
    index_to_user: Dict[int, int]     # matrix_row_index -> original_user_id
    index_to_movie: Dict[int, int]    # matrix_col_index -> original_movie_id
    movies_df: pd.DataFrame           # Movie metadata aligned to matrix columns
    ratings_df: pd.DataFrame          # Raw merged ratings DataFrame


def load_and_preprocess(cfg: DataConfig = DataConfig()) -> RecommenderData:
    """Full data pipeline: load -> clean -> build sparse matrix -> sample -> split."""

    # --- Load CSVs ---
    ratings_df = pd.read_csv(cfg.ratings_csv)
    movies_df = pd.read_csv(cfg.movies_csv)

    if "timestamp" in ratings_df.columns:
        ratings_df = ratings_df.drop(columns=["timestamp"])

    merged = pd.merge(ratings_df, movies_df[["movieId", "genres", "title"]], on="movieId")

    # --- Extract year from title ---
    merged["year"] = merged["title"].apply(_extract_year)
    merged = merged.dropna(subset=["year"])
    merged["year"] = merged["year"].astype(int)

    # --- Build sparse utility matrix ---
    user_ids = merged["userId"].unique()
    movie_ids = merged["movieId"].unique()

    user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
    movie_id_map = {mid: idx for idx, mid in enumerate(movie_ids)}

    rows = merged["userId"].map(user_id_map).values
    cols = merged["movieId"].map(movie_id_map).values
    data = merged["rating"].values

    full_matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(user_ids), len(movie_ids)),
    )

    # --- Sample users ---
    sampled, sampled_user_map = _sample_users(
        full_matrix, user_id_map, cfg.sample_fraction, cfg.random_seed
    )

    # --- Filter inactive users (inline — before split) ---
    pre_filter_n = sampled.shape[0]
    row_nnz = np.diff(sampled.indptr)
    active_mask = row_nnz >= cfg.min_ratings_per_user
    sampled = sampled[active_mask]

    # Rebuild user_id_map to reflect new row indices after filtering
    old_to_new: Dict[int, int] = {}
    new_idx = 0
    for old_idx in range(pre_filter_n):
        if active_mask[old_idx]:
            old_to_new[old_idx] = new_idx
            new_idx += 1

    sampled_user_map = {
        uid: old_to_new[old_row]
        for uid, old_row in sampled_user_map.items()
        if old_row in old_to_new
    }

    # --- Split into train / warm-test / cold-test ---
    train_matrix, test_warm, test_cold, warm_indices, cold_indices = _split_train_test(
        sampled, cfg.cold_user_fraction, cfg.test_ratio, cfg.random_seed,
    )

    # --- Build reverse maps and aligned movie metadata ---
    index_to_user = {idx: uid for uid, idx in sampled_user_map.items()}
    index_to_movie = {idx: mid for mid, idx in movie_id_map.items()}

    aligned_movies = pd.DataFrame(
        {"movieId": [index_to_movie[i] for i in range(len(movie_ids))]}
    )
    aligned_movies = aligned_movies.merge(movies_df, on="movieId", how="left")

    return RecommenderData(
        train_matrix=train_matrix,
        test_warm=test_warm,
        test_cold=test_cold,
        warm_user_indices=warm_indices,
        cold_user_indices=cold_indices,
        user_id_map=sampled_user_map,
        movie_id_map=movie_id_map,
        index_to_user=index_to_user,
        index_to_movie=index_to_movie,
        movies_df=aligned_movies,
        ratings_df=merged,
    )


# ---------------------------------------------------------------------------
# Internal helpers (all return new objects — no mutation)
# ---------------------------------------------------------------------------

def _extract_year(title: str):
    match = re.search(r"\((\d{4})\)", str(title))
    return int(match.group(1)) if match else None


def _sample_users(
    matrix: csr_matrix,
    user_id_map: Dict[int, int],
    fraction: float,
    random_seed: int,
) -> Tuple[csr_matrix, Dict[int, int]]:
    """Sample a fraction of users — returns (new_matrix, new_user_id_map)."""
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    rng = np.random.RandomState(random_seed)
    original_ids = list(user_id_map.keys())
    n_sampled = int(len(original_ids) * fraction)

    sampled_ids = rng.choice(original_ids, n_sampled, replace=False)
    new_map = {uid: i for i, uid in enumerate(sampled_ids)}

    rows, cols, data = [], [], []
    for uid in sampled_ids:
        old_idx = user_id_map[uid]
        row = matrix[old_idx].tocoo()
        new_idx = new_map[uid]
        rows.extend([new_idx] * len(row.col))
        cols.extend(row.col)
        data.extend(row.data)

    sampled = csr_matrix(
        (data, (rows, cols)),
        shape=(n_sampled, matrix.shape[1]),
    )
    return sampled, new_map


def _split_train_test(
    full_matrix: csr_matrix,
    cold_user_fraction: float,
    test_ratio: float,
    random_seed: int,
) -> Tuple[csr_matrix, csr_matrix, csr_matrix, np.ndarray, np.ndarray]:
    """Split the matrix into train, warm-test, and cold-test sets.

    Cold users have ALL their ratings held out into test_cold.
    Warm users have test_ratio of their ratings held out into test_warm;
    the remainder goes into train_matrix.

    Returns:
        train_matrix, test_warm, test_cold, warm_user_indices, cold_user_indices
    """
    rng = np.random.RandomState(random_seed)
    n_users, n_items = full_matrix.shape

    all_user_indices = np.arange(n_users)
    n_cold = int(n_users * cold_user_fraction)
    cold_user_indices = np.sort(rng.choice(all_user_indices, n_cold, replace=False))
    warm_user_indices = np.setdiff1d(all_user_indices, cold_user_indices)

    train_rows, train_cols, train_data = [], [], []
    test_warm_rows, test_warm_cols, test_warm_data = [], [], []
    test_cold_rows, test_cold_cols, test_cold_data = [], [], []

    cold_set = set(cold_user_indices.tolist())

    for uid in range(n_users):
        row = full_matrix[uid].tocoo()
        cols = row.col
        vals = row.data
        if len(cols) == 0:
            continue

        if uid in cold_set:
            test_cold_rows.extend([uid] * len(cols))
            test_cold_cols.extend(cols)
            test_cold_data.extend(vals)
        else:
            n_test = max(1, int(len(cols) * test_ratio))
            test_mask = np.zeros(len(cols), dtype=bool)
            test_indices = rng.choice(len(cols), n_test, replace=False)
            test_mask[test_indices] = True

            train_rows.extend([uid] * int((~test_mask).sum()))
            train_cols.extend(cols[~test_mask])
            train_data.extend(vals[~test_mask])

            test_warm_rows.extend([uid] * int(test_mask.sum()))
            test_warm_cols.extend(cols[test_mask])
            test_warm_data.extend(vals[test_mask])

    shape = (n_users, n_items)
    train_matrix = csr_matrix(
        (train_data, (train_rows, train_cols)), shape=shape
    )
    test_warm = csr_matrix(
        (test_warm_data, (test_warm_rows, test_warm_cols)), shape=shape
    )
    test_cold = csr_matrix(
        (test_cold_data, (test_cold_rows, test_cold_cols)), shape=shape
    )

    return train_matrix, test_warm, test_cold, warm_user_indices, cold_user_indices
