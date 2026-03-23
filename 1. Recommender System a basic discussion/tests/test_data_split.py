"""Validation tests for the train/test split logic."""
import numpy as np
from scipy.sparse import csr_matrix

from src.data_utils import _split_train_test


def _make_test_matrix(n_users=50, n_items=100, density=0.1, seed=42):
    """Create a small test sparse matrix with known properties."""
    rng = np.random.RandomState(seed)
    rows, cols, data = [], [], []
    for u in range(n_users):
        n_ratings = max(5, int(n_items * density))
        rated_items = rng.choice(n_items, n_ratings, replace=False)
        for item in rated_items:
            rows.append(u)
            cols.append(item)
            data.append(rng.uniform(0.5, 5.0))
    return csr_matrix((data, (rows, cols)), shape=(n_users, n_items))


def test_no_overlap_train_test_warm():
    mat = _make_test_matrix()
    train, test_warm, _, _, _ = _split_train_test(mat, 0.2, 0.2, 42)
    overlap = train.multiply(test_warm)
    assert overlap.nnz == 0

def test_no_overlap_train_test_cold():
    mat = _make_test_matrix()
    train, _, test_cold, _, _ = _split_train_test(mat, 0.2, 0.2, 42)
    overlap = train.multiply(test_cold)
    assert overlap.nnz == 0

def test_cold_users_empty_in_train():
    mat = _make_test_matrix()
    train, _, _, _, cold_indices = _split_train_test(mat, 0.2, 0.2, 42)
    for uid in cold_indices:
        assert train[uid].nnz == 0

def test_warm_users_reconstruct():
    mat = _make_test_matrix()
    train, test_warm, _, warm_indices, _ = _split_train_test(mat, 0.2, 0.2, 42)
    for uid in warm_indices:
        original = mat[uid].toarray().flatten()
        reconstructed = train[uid].toarray().flatten() + test_warm[uid].toarray().flatten()
        np.testing.assert_array_almost_equal(original, reconstructed)

def test_cold_users_reconstruct():
    mat = _make_test_matrix()
    _, _, test_cold, _, cold_indices = _split_train_test(mat, 0.2, 0.2, 42)
    for uid in cold_indices:
        original = mat[uid].toarray().flatten()
        cold = test_cold[uid].toarray().flatten()
        np.testing.assert_array_almost_equal(original, cold)

def test_total_nnz_preserved():
    mat = _make_test_matrix()
    train, test_warm, test_cold, _, _ = _split_train_test(mat, 0.2, 0.2, 42)
    assert train.nnz + test_warm.nnz + test_cold.nnz == mat.nnz

def test_cold_fraction_approximate():
    mat = _make_test_matrix(n_users=100)
    _, _, _, _, cold_indices = _split_train_test(mat, 0.2, 0.2, 42)
    assert 15 <= len(cold_indices) <= 25

def test_reproducibility():
    mat = _make_test_matrix()
    r1 = _split_train_test(mat, 0.2, 0.2, 42)
    r2 = _split_train_test(mat, 0.2, 0.2, 42)
    assert (r1[0] - r2[0]).nnz == 0
    assert np.array_equal(r1[3], r2[3])
    assert np.array_equal(r1[4], r2[4])
