"""
Unified evaluation module for all recommender models.

Consolidates the 4 near-identical evaluation functions from the original notebook
into a single, consistent implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
)


@dataclass(frozen=True)
class EvalResult:
    """Immutable container for evaluation metrics."""
    recall: float
    precision: float
    accuracy: float
    rmse: float
    f1: float

    def __repr__(self) -> str:
        return (
            f"Recall={self.recall:.4f}  Precision={self.precision:.4f}  "
            f"Accuracy={self.accuracy:.4f}  RMSE={self.rmse:.4f}  F1={self.f1:.4f}"
        )

    def as_dict(self) -> dict:
        return {
            "recall": self.recall,
            "precision": self.precision,
            "accuracy": self.accuracy,
            "rmse": self.rmse,
            "f1": self.f1,
        }


def evaluate(
    true_ratings: np.ndarray,
    predicted_ratings: np.ndarray,
    threshold: float = 3.5,
) -> EvalResult:
    """
    Evaluate predicted ratings against ground truth.

    Only considers positions where the true rating is non-zero (i.e., observed).
    Converts sparse matrices to dense automatically.

    Parameters
    ----------
    true_ratings : array-like or sparse
        Ground truth ratings (1-D).
    predicted_ratings : array-like or sparse
        Model predictions (1-D, same length as true_ratings).
    threshold : float
        Rating threshold for binary liked / not-liked classification.

    Returns
    -------
    EvalResult with recall, precision, accuracy, RMSE, F1.
    """
    true = _to_array(true_ratings)
    pred = _to_array(predicted_ratings)

    if len(true) != len(pred):
        raise ValueError(
            f"Length mismatch: true={len(true)}, pred={len(pred)}"
        )

    # Keep only observed (non-zero) and finite entries
    valid = (true != 0) & np.isfinite(true) & np.isfinite(pred)
    true = true[valid]
    pred = pred[valid]

    if len(true) == 0:
        return EvalResult(
            recall=float("nan"),
            precision=float("nan"),
            accuracy=float("nan"),
            rmse=float("nan"),
            f1=float("nan"),
        )

    true_liked = true >= threshold
    pred_liked = pred >= threshold

    return EvalResult(
        recall=recall_score(true_liked, pred_liked, zero_division=0),
        precision=precision_score(true_liked, pred_liked, zero_division=0),
        accuracy=accuracy_score(true_liked, pred_liked),
        rmse=float(np.sqrt(mean_squared_error(true, pred))),
        f1=f1_score(true_liked, pred_liked, zero_division=0),
    )


def aggregate_results(results: list[EvalResult]) -> EvalResult:
    """Average a list of EvalResult, ignoring NaN entries."""
    if not results:
        raise ValueError("Cannot aggregate empty results list")

    fields = ["recall", "precision", "accuracy", "rmse", "f1"]
    averages = {}
    for field in fields:
        values = [getattr(r, field) for r in results if np.isfinite(getattr(r, field))]
        averages[field] = float(np.mean(values)) if values else float("nan")

    return EvalResult(**averages)


def scale_predictions(predictions: np.ndarray, new_min: float = 1.0, new_max: float = 5.0) -> np.ndarray:
    """
    Min-max scale predictions into [new_min, new_max].

    FIX: Original notebook had a division-by-zero bug when all predictions
    were identical (old_min == old_max).
    """
    pred = np.asarray(predictions, dtype=np.float64)
    old_min, old_max = pred.min(), pred.max()

    if old_max - old_min < 1e-10:
        # All predictions identical — map to midpoint
        return np.full_like(pred, (new_min + new_max) / 2)

    return new_min + (new_max - new_min) * (pred - old_min) / (old_max - old_min)


@dataclass(frozen=True)
class SubsetEvalResult:
    """Evaluation result for a subset of users, carrying SSE for pooled RMSE."""
    eval_result: EvalResult
    total_sse: float
    total_rated_count: int


def evaluate_on_subset(
    test_matrix: csr_matrix,
    predict_fn,
    user_indices: np.ndarray,
    threshold: float = 3.5,
    desc: str = "Evaluating",
) -> SubsetEvalResult:
    """Evaluate predict_fn on a subset of users against a test matrix."""
    results = []
    total_sse = 0.0
    total_count = 0
    for uid in tqdm(user_indices, desc=f"  {desc}", leave=False, unit="user"):
        true = test_matrix[uid].toarray().flatten()
        pred = np.clip(predict_fn(uid), 0.5, 5.0)
        result = evaluate(true, pred, threshold=threshold)
        results.append(result)
        valid = (true != 0) & np.isfinite(true) & np.isfinite(pred)
        if valid.any():
            total_sse += float(np.sum((true[valid] - pred[valid]) ** 2))
            total_count += int(valid.sum())

    return SubsetEvalResult(
        eval_result=aggregate_results(results),
        total_sse=total_sse,
        total_rated_count=total_count,
    )


def evaluate_overall(
    warm: SubsetEvalResult,
    cold: SubsetEvalResult,
) -> EvalResult:
    """Combine warm and cold results. Uses pooled RMSE (not averaged)."""
    n_warm = warm.total_rated_count
    n_cold = cold.total_rated_count
    total = n_warm + n_cold
    if total == 0:
        return EvalResult(
            recall=float("nan"), precision=float("nan"),
            accuracy=float("nan"), rmse=float("nan"), f1=float("nan"),
        )
    w_warm = n_warm / total
    w_cold = n_cold / total
    wr = warm.eval_result
    cr = cold.eval_result

    pooled_rmse = float(np.sqrt((warm.total_sse + cold.total_sse) / total))

    return EvalResult(
        recall=wr.recall * w_warm + cr.recall * w_cold,
        precision=wr.precision * w_warm + cr.precision * w_cold,
        accuracy=wr.accuracy * w_warm + cr.accuracy * w_cold,
        rmse=pooled_rmse,
        f1=wr.f1 * w_warm + cr.f1 * w_cold,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_array(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.toarray()).flatten()
    if isinstance(x, np.matrix):
        return np.asarray(x).flatten()
    return np.asarray(x).flatten()
