"""
Matrix Completion methods: SVT (Singular Value Thresholding) and ALS.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse, stats
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.utils.extmath import randomized_svd
from tqdm import trange


# ==========================================================================
# SVT Matrix Completion
# ==========================================================================

class SVTMatrixCompletion:
    """
    Singular Value Thresholding (SVT) for matrix completion.

    Reference: Cai, Candès & Shen (2010) — "A Singular Value Thresholding
    Algorithm for Matrix Rank Minimization".
    """

    def __init__(self, beta: float = 0.3, tau: float = 100.0,
                 tolerance: float = 1e-4, max_iter: int = 100):
        self.beta = beta
        self.tau = tau
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.completed_matrix = None

    def fit(self, M: np.ndarray, verbose: bool = True) -> "SVTMatrixCompletion":
        """
        Run SVT on matrix M (dense, 0 = missing).

        Parameters
        ----------
        M : np.ndarray  (users x items)
            Input matrix with 0 for missing entries.
        """
        D, N = M.shape
        omega = _construct_omega(M)

        # Scale
        scaled, scaler = _scale_matrix(M)
        proj_omega = _project(scaled, omega)

        Y = np.zeros((D, N))

        iterator = trange(self.max_iter, desc="SVT", disable=not verbose)
        for it in iterator:
            U, S, VT = randomized_svd(Y, n_components=min(Y.shape) - 1)
            S_shrink = np.maximum(S - self.tau, 0)
            X = U @ np.diag(S_shrink) @ VT

            Y = Y + self.beta * _project(scaled - X, omega)

            residual = (
                np.linalg.norm(proj_omega - _project(X, omega), "fro")
                / max(np.linalg.norm(proj_omega, "fro"), 1e-10)
            )
            iterator.set_postfix(residual=f"{residual:.6f}")

            if residual <= self.tolerance:
                iterator.close()
                if verbose:
                    print(f"  Converged at iteration {it + 1}")
                break

        self.completed_matrix = scaler.inverse_transform(X)
        return self

    def predict(self, user_index: int) -> np.ndarray:
        return self.completed_matrix[user_index].flatten()


# ==========================================================================
# ALS Matrix Completion
# ==========================================================================

class ALSMatrixCompletion:
    """
    Alternating Least Squares for matrix factorisation / completion.
    """

    def __init__(self, rank: int = 20, lambda_reg: float = 0.01,
                 max_iter: int = 100, tolerance: float = 1e-4):
        self.rank = rank
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.completed_matrix = None

    def fit(self, R: np.ndarray, verbose: bool = True) -> "ALSMatrixCompletion":
        """
        Factorise R ≈ U @ V^T via ALS.

        Parameters
        ----------
        R : np.ndarray  (users x items)
            Rating matrix (0 = missing).
        """
        n_users, n_items = R.shape
        mask = (R != 0).astype(float)
        K = self.rank
        lam = self.lambda_reg

        rng = np.random.RandomState(42)
        U = rng.normal(0, 0.1, (n_users, K))
        V = rng.normal(0, 0.1, (n_items, K))

        prev_loss = float("inf")
        identity_k = np.eye(K)

        iterator = trange(self.max_iter, desc="ALS", disable=not verbose)
        for it in iterator:
            # Fix V, solve for U
            for i in range(n_users):
                rated = mask[i] > 0
                if not rated.any():
                    continue
                V_rated = V[rated]
                R_rated = R[i, rated]
                U[i] = np.linalg.solve(
                    V_rated.T @ V_rated + lam * identity_k,
                    V_rated.T @ R_rated,
                )

            # Fix U, solve for V
            for j in range(n_items):
                rated = mask[:, j] > 0
                if not rated.any():
                    continue
                U_rated = U[rated]
                R_rated = R[rated, j]
                V[j] = np.linalg.solve(
                    U_rated.T @ U_rated + lam * identity_k,
                    U_rated.T @ R_rated,
                )

            pred = U @ V.T
            loss = np.sum(mask * (R - pred) ** 2) + lam * (
                np.sum(U ** 2) + np.sum(V ** 2)
            )
            iterator.set_postfix(loss=f"{loss:.2f}")

            if abs(prev_loss - loss) < self.tolerance:
                iterator.close()
                if verbose:
                    print(f"  Converged at iteration {it + 1}")
                break
            prev_loss = loss

        self.completed_matrix = U @ V.T
        return self

    def predict(self, user_index: int) -> np.ndarray:
        return self.completed_matrix[user_index].flatten()


# ==========================================================================
# Pre-fill strategies for matrix completion input
# ==========================================================================

def prefill_user_mean(matrix: np.ndarray, std: float = 0.5) -> np.ndarray:
    """Fill missing entries with truncated Gaussian around user mean."""
    filled = matrix.copy()
    for i in range(filled.shape[0]):
        row = filled[i]
        rated = row[row != 0]
        mean = rated.mean() if len(rated) > 0 else _global_nonzero_mean(matrix)

        missing = row == 0
        n_missing = missing.sum()
        if n_missing == 0:
            continue

        values = stats.truncnorm(
            (1 - mean) / std, (5 - mean) / std, loc=mean, scale=std
        ).rvs(size=n_missing)
        filled[i, missing] = np.clip(values, 1, 5)
    return filled


def prefill_global_mean(matrix: np.ndarray, std: float = 1.0) -> np.ndarray:
    """Fill missing entries with Gaussian around global mean (for cold-start)."""
    filled = matrix.copy()
    global_mean = _global_nonzero_mean(matrix)
    missing = filled == 0
    n = missing.sum()
    if n > 0:
        values = np.random.normal(global_mean, std, size=n)
        filled[missing] = np.clip(values, 1, 5)
    return filled


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _construct_omega(M):
    if sparse.issparse(M):
        rows, cols = M.nonzero()
        vals = M.data
        mask = vals != 0
        return rows[mask], cols[mask]
    return np.where(M != 0)


def _project(M, omega):
    proj = np.zeros_like(M)
    proj[omega] = M[omega]
    return proj


def _scale_matrix(M):
    if sparse.issparse(M):
        scaler = MaxAbsScaler()
    else:
        scaler = StandardScaler()
    return scaler.fit_transform(M), scaler


def _global_nonzero_mean(matrix: np.ndarray) -> float:
    vals = matrix[matrix != 0]
    return float(vals.mean()) if len(vals) > 0 else 3.0
