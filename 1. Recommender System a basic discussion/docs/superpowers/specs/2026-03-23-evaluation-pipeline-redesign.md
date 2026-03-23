# Evaluation Pipeline Redesign

**Date**: 2026-03-23
**Status**: Approved
**Scope**: Complete rewrite of `data_utils.py`, `evaluation.py`, `train.py`, `config.py`

## Problem Statement

The current evaluation pipeline has fundamental flaws:

1. **Single cold-start user** — only 1 user (hardcoded) is simulated as cold-start
2. **No train/test separation** — models train on nearly the same data they are evaluated against (data leakage for all non-target users)
3. **Diluted cold-start signal** — the 1 cold user's metrics are averaged across ~1,384 warm users

These issues mean the reported metrics measure **reconstruction ability** rather than **recommendation ability**.

## Design

### Core Principle

Only non-zero entries (observed ratings) participate in train/test splits. Zeros in the sparse matrix represent "unknown" and are never used as ground truth.

### Data Split Strategy

```
Sampled users (1% of full dataset, filtered to min 5 ratings)
    │
    ├─ 80% → Warm Users
    │   └─ Each user's non-zero ratings: 80% → train, 20% → test
    │
    └─ 20% → Cold Users
        └─ 0% → train (completely absent from training matrix)
        └─ 100% → test (all their ratings are held out)
```

### Deprecated Fields (REMOVED)

The following fields from the old design are **deleted**, not retained:

- `DataConfig.removal_percentage` — replaced by `test_ratio`
- `RecommenderData.target_user_id` — replaced by `cold_user_indices`
- `RecommenderData.ground_truth` / `partial_ratings` / `cold_start` — replaced by `train_matrix` / `test_warm` / `test_cold`
- Helper functions `_remove_user_ratings()`, `_remove_all_user_ratings()` — no longer needed

### Configuration

New fields in `DataConfig`:

```python
@dataclass(frozen=True)
class DataConfig:
    movies_csv: Path
    ratings_csv: Path
    sample_fraction: float = 0.01
    min_ratings_per_user: int = 5
    random_seed: int = 42
    cold_user_fraction: float = 0.20    # 20% of users become cold-start
    test_ratio: float = 0.20            # 20% of warm user ratings held out
    # REMOVED: removal_percentage (was 30.0)
```

### Data Container

Replace `RecommenderData` with a new structure that clearly separates train from test:

```python
@dataclass(frozen=True)
class RecommenderData:
    # Training data (what models see)
    train_matrix: csr_matrix          # warm users' 80% ratings + cold users' 0 rows

    # Test data (what we evaluate against)
    test_warm: csr_matrix             # warm users' held-out 20% ratings
    test_cold: csr_matrix             # cold users' full ratings

    # Index sets
    warm_user_indices: np.ndarray     # row indices of warm users in the matrix
    cold_user_indices: np.ndarray     # row indices of cold users in the matrix

    # Mappings (unchanged)
    user_id_map: dict                 # original_user_id → matrix_row_index
    movie_id_map: dict                # original_movie_id → matrix_col_index
    index_to_user: dict
    index_to_movie: dict

    # Metadata
    movies_df: pd.DataFrame
    ratings_df: pd.DataFrame
```

Key properties:
- `train_matrix` has the same shape as the full sampled matrix, but cold users' rows are all zeros, and warm users are missing 20% of their ratings
- `test_warm` has the same shape, but only contains the held-out 20% entries for warm users (everything else is 0)
- `test_cold` has the same shape, but only contains cold users' ratings (everything else is 0)
- No overlap between train and test: `train_matrix.multiply(test_warm)` has zero non-zero entries

### Pipeline Ordering

The processing order is strict and must not be rearranged:

```
1. Load CSVs, merge, extract year
2. Build full sparse utility matrix
3. Sample users (sample_fraction=0.01)
4. Remove inactive users (min_ratings_per_user=5)   ← BEFORE split
5. Split into warm/cold users and train/test         ← AFTER filtering
```

**Rationale**: Filtering must happen BEFORE the split. Otherwise cold users (who have 0 train ratings) would be removed by the inactive-user filter. After filtering, all remaining users have >= 5 ratings, so even after the 80/20 split, warm users retain at least 4 training ratings.

### Split Algorithm

Uses `np.random.RandomState(cfg.random_seed)` for deterministic reproducibility.

```
def split_train_test(full_matrix, cfg):
    rng = np.random.RandomState(cfg.random_seed)

    1. Randomly select 20% of user row indices → cold_user_indices
       Remaining 80% → warm_user_indices
       (using rng.choice, no replacement)

    2. For each warm user:
       - Get their non-zero column indices
       - Randomly select 20% of these → held_out_cols (using rng)
       - Remaining 80% → train_cols
       - Put train entries in train_matrix
       - Put held-out entries in test_warm

    3. For each cold user:
       - Put nothing in train_matrix (row stays all-zero)
       - Put all their non-zero entries in test_cold

    4. Return RecommenderData with all matrices and indices
```

### Evaluation

Replace the single `evaluate_all_users()` with three separate evaluation passes:

```python
def evaluate_warm_users(model, data) -> EvalResult:
    """Evaluate only on warm users' held-out ratings."""
    for uid in data.warm_user_indices:
        true = data.test_warm[uid]      # held-out 20%
        pred = model.predict(uid)
        # compare only at non-zero positions of test_warm[uid]

def evaluate_cold_users(model, data) -> EvalResult:
    """Evaluate only on cold users' full ratings."""
    for uid in data.cold_user_indices:
        true = data.test_cold[uid]      # all their ratings
        pred = model.predict(uid)
        # compare only at non-zero positions of test_cold[uid]

def evaluate_overall(warm_result, cold_result, n_warm_ratings, n_cold_ratings) -> EvalResult:
    """Weighted average by number of test ratings."""
```

### Results Format

```json
{
  "svt": {
    "SVT-tau100": {
      "warm": {"recall": 0.xx, "precision": 0.xx, "accuracy": 0.xx, "rmse": 0.xx, "f1": 0.xx},
      "cold": {"recall": 0.xx, "precision": 0.xx, "accuracy": 0.xx, "rmse": 0.xx, "f1": 0.xx},
      "overall": {"recall": 0.xx, "precision": 0.xx, "accuracy": 0.xx, "rmse": 0.xx, "f1": 0.xx}
    }
  }
}
```

### Model Runner Changes

Each model runner must adapt to the new data structure. The key change: models train on `data.train_matrix` and are evaluated separately on warm and cold users.

| Model | Training input | Warm prediction | Cold prediction |
|-------|---------------|----------------|----------------|
| CB | `movies_df["genres"]` (unchanged) | Use user's train ratings to find similar content | No user history → global popularity fallback |
| CF | `data.train_matrix` for similarity | Predict using similar users from train | No similarity signal → global mean fallback |
| SVT | Pre-filled `data.train_matrix` | Completed matrix row | Completed matrix row (cold rows were pre-filled) |
| ALS | `data.train_matrix` | Factored matrix row | Factored matrix row |
| MLP | Non-zero entries from `data.train_matrix` | `model.predict(uid, all_items)` | Same (embedding untrained) |
| AE | `data.train_matrix.toarray()` rows | Reconstruct from train vector | Input is all-zero → learned bias output |
| MF | Non-zero entries from `data.train_matrix` | `model.predict(uid, all_items)` | Same (embedding untrained) |

### Files to Change

| File | Change | Size |
|------|--------|------|
| `src/config.py` | Add `cold_user_fraction`, `test_ratio` | Small |
| `src/data_utils.py` | Rewrite `load_and_preprocess()`, new split logic, new `RecommenderData` | Large |
| `src/evaluation.py` | Add `evaluate_warm_users()`, `evaluate_cold_users()`, `evaluate_overall()` | Medium |
| `src/train.py` | Rewrite all model runners to use new data structure and 3-way evaluation | Large |
| `src/models/*.py` | Minimal changes — add cold-start fallback where needed | Small |

### Neural Model Internal Train/Val Split

Neural models (MLP, AE, MF) use `TrainConfig.train_split=0.8` to split `train_matrix`'s non-zero entries into an internal train/val set for early stopping. This is **separate from** the data-level train/test split:

```
full_matrix
  ├─ data-level split (DataConfig.test_ratio=0.20)
  │   ├─ train_matrix (80% of warm users' ratings)    ← models see this
  │   └─ test_warm / test_cold                         ← final evaluation
  │
  └─ neural model internal split (TrainConfig.train_split=0.80)
      ├─ 80% of train_matrix non-zero entries → gradient updates
      └─ 20% of train_matrix non-zero entries → early stopping validation
```

The internal val set is used ONLY for early stopping and learning rate scheduling. It is NOT used for reporting metrics. Final metrics always come from `test_warm` / `test_cold`.

### Prediction Interface

All model predictions must be wrapped into a common signature for the evaluation functions:

```python
predict_fn(user_index: int) -> np.ndarray  # returns 1-D array of length n_items
```

Each model runner creates a lambda/closure that adapts its model's native API to this interface. Predictions should be clipped to [0.5, 5.0] before evaluation.

### Overall Metrics Weighting

`evaluate_overall` computes a weighted average where the weight is the **total number of test ratings** (not number of users):

```python
w_warm = test_warm.nnz
w_cold = test_cold.nnz
overall_metric = (warm_metric * w_warm + cold_metric * w_cold) / (w_warm + w_cold)
```

For RMSE: pool all (true, pred) pairs from warm and cold, then compute a single RMSE (not an average of two RMSEs, since RMSE is non-linear).

### What Does NOT Change

- Model architectures (MLP, AE, MF, SVT, ALS, CF, CB internals)
- Training hyperparameters and configs (MLPConfig, AEConfig, etc.)
- CLI interface (`--models`, `--epochs`, etc.)
- Checkpoint saving for neural models

### Validation Criteria

1. `train_matrix` and `test_warm` have zero overlap: `train_matrix.multiply(test_warm).nnz == 0`
2. `train_matrix` and `test_cold` have zero overlap: `train_matrix.multiply(test_cold).nnz == 0`
3. Cold users have exactly 0 non-zero entries in `train_matrix`
4. Warm users' train + test ratings reconstruct their original ratings
5. All models produce warm, cold, and overall metrics
6. Results are reproducible with the same random seed
