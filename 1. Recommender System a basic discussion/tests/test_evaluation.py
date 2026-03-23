"""Tests for warm/cold/overall evaluation functions."""
import numpy as np
from src.evaluation import EvalResult, SubsetEvalResult, evaluate_overall


def test_overall_pure_warm():
    warm = SubsetEvalResult(
        eval_result=EvalResult(recall=0.8, precision=0.7, accuracy=0.75, rmse=0.5, f1=0.74),
        total_sse=25.0, total_rated_count=100,
    )
    cold = SubsetEvalResult(
        eval_result=EvalResult(recall=0.0, precision=0.0, accuracy=0.0, rmse=0.0, f1=0.0),
        total_sse=0.0, total_rated_count=0,
    )
    result = evaluate_overall(warm, cold)
    assert abs(result.f1 - 0.74) < 1e-6
    assert abs(result.rmse - 0.5) < 1e-6  # sqrt(25/100) = 0.5


def test_overall_equal_weights():
    warm = SubsetEvalResult(
        eval_result=EvalResult(recall=0.8, precision=0.8, accuracy=0.8, rmse=0.4, f1=0.8),
        total_sse=8.0, total_rated_count=50,
    )
    cold = SubsetEvalResult(
        eval_result=EvalResult(recall=0.6, precision=0.6, accuracy=0.6, rmse=0.8, f1=0.6),
        total_sse=32.0, total_rated_count=50,
    )
    result = evaluate_overall(warm, cold)
    assert abs(result.f1 - 0.7) < 1e-6
    # Pooled RMSE = sqrt((8+32)/100) = sqrt(0.4) ≈ 0.6325
    assert abs(result.rmse - np.sqrt(0.4)) < 1e-6


def test_overall_pooled_rmse_differs_from_average():
    """Pooled RMSE should NOT equal simple average of warm and cold RMSE."""
    warm = SubsetEvalResult(
        eval_result=EvalResult(recall=0.8, precision=0.8, accuracy=0.8, rmse=1.0, f1=0.8),
        total_sse=100.0, total_rated_count=100,
    )
    cold = SubsetEvalResult(
        eval_result=EvalResult(recall=0.6, precision=0.6, accuracy=0.6, rmse=2.0, f1=0.6),
        total_sse=40.0, total_rated_count=10,
    )
    result = evaluate_overall(warm, cold)
    naive_avg = (1.0 * 100 + 2.0 * 10) / 110  # weighted average
    pooled = np.sqrt((100.0 + 40.0) / 110)     # correct pooled
    assert abs(result.rmse - pooled) < 1e-6
    assert abs(result.rmse - naive_avg) > 0.01  # they differ
