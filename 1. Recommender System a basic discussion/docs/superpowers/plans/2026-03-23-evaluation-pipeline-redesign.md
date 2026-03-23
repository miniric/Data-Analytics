# Evaluation Pipeline Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-user cold-start simulation with a rigorous train/test split that separately evaluates warm users (leave-20%-out) and cold users (20% of users, fully held out).

**Architecture:** Rewrite the data pipeline to produce `train_matrix`, `test_warm`, `test_cold` from a single split function. Evaluation becomes three separate passes (warm/cold/overall). Model runners adapt to the new data container; model architectures stay unchanged.

**Tech Stack:** Python 3.11+, scipy.sparse, numpy, pandas, PyTorch, tqdm

**Spec:** `docs/superpowers/specs/2026-03-23-evaluation-pipeline-redesign.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/config.py` | Modify | Add `cold_user_fraction`, `test_ratio`; remove `removal_percentage` |
| `src/data_utils.py` | Rewrite | New `RecommenderData`, new `load_and_preprocess()` with train/test split |
| `src/evaluation.py` | Modify | Add `evaluate_warm_users()`, `evaluate_cold_users()`, `evaluate_overall()` |
| `src/train.py` | Rewrite | Adapt all 7 model runners to new data structure + 3-way eval |
| `tests/test_data_split.py` | Create | Validate split correctness (no overlap, cold users empty, reconstruction) |
| `tests/test_evaluation.py` | Create | Validate warm/cold/overall eval logic |

---

### Task 1: Update `src/config.py`

**Files:**
- Modify: `src/config.py:32-40` (DataConfig)

- [ ] **Step 1: Edit DataConfig**

Replace `removal_percentage` with new fields:

```python
@dataclass(frozen=True)
class DataConfig:
    """Immutable data configuration."""
    movies_csv: Path = DATA_DIR / "movies.csv"
    ratings_csv: Path = DATA_DIR / "ratings.csv"
    sample_fraction: float = 0.01
    min_ratings_per_user: int = 5
    random_seed: int = 42
    cold_user_fraction: float = 0.20
    test_ratio: float = 0.20
```

- [ ] **Step 2: Verify imports still work**

Run: `cd /Users/chang.yy/Documents/NAS/7_Code/1.\ Git_Repos/1.\ Recommender\ System\ a\ basic\ discussion && python -c "from src.config import DataConfig; print(DataConfig())"`

Expected: prints DataConfig with new fields, no errors.

- [ ] **Step 3: Commit**

```bash
git add src/config.py
git commit -m "refactor: replace removal_percentage with cold_user_fraction and test_ratio in DataConfig"
```

---

### Task 2: Rewrite `src/data_utils.py` — Data Container

**Files:**
- Modify: `src/data_utils.py:28-40` (RecommenderData)

- [ ] **Step 1: Replace RecommenderData**

```python
@dataclass(frozen=True)
class RecommenderData:
    """Immutable container holding train/test split matrices."""
    # Training data (what models see)
    train_matrix: csr_matrix

    # Test data (what we evaluate against)
    test_warm: csr_matrix
    test_cold: csr_matrix

    # Index sets
    warm_user_indices: np.ndarray
    cold_user_indices: np.ndarray

    # Mappings
    user_id_map: Dict[int, int]
    movie_id_map: Dict[int, int]
    index_to_user: Dict[int, int]
    index_to_movie: Dict[int, int]

    # Metadata
    movies_df: pd.DataFrame
    ratings_df: pd.DataFrame
```

- [ ] **Step 2: Verify dataclass is valid**

Run: `python -c "from src.data_utils import RecommenderData; print('OK')"`

Expected: `OK`

---

### Task 3: Rewrite `src/data_utils.py` — Split Logic

**Files:**
- Modify: `src/data_utils.py` (replace `load_and_preprocess` and internal helpers)

- [ ] **Step 1: Write the split function**

Replace `_remove_user_ratings`, `_remove_all_user_ratings` with:

```python
def _split_train_test(
    full_matrix: csr_matrix,
    cold_user_fraction: float,
    test_ratio: float,
    random_seed: int,
) -> Tuple[csr_matrix, csr_matrix, csr_matrix, np.ndarray, np.ndarray]:
    """
    Split observed ratings into train/test for warm users and hold out cold users entirely.

    Returns: (train_matrix, test_warm, test_cold, warm_user_indices, cold_user_indices)
    """
    rng = np.random.RandomState(random_seed)
    n_users, n_items = full_matrix.shape

    # 1. Split users into warm / cold
    all_user_indices = np.arange(n_users)
    n_cold = int(n_users * cold_user_fraction)
    cold_user_indices = np.sort(rng.choice(all_user_indices, n_cold, replace=False))
    warm_user_indices = np.setdiff1d(all_user_indices, cold_user_indices)

    # 2. Build COO lists for each matrix
    train_rows, train_cols, train_data = [], [], []
    test_warm_rows, test_warm_cols, test_warm_data = [], [], []
    test_cold_rows, test_cold_cols, test_cold_data = [], [], []

    cold_set = set(cold_user_indices)

    for uid in range(n_users):
        row = full_matrix[uid].tocoo()
        cols = row.col
        vals = row.data

        if len(cols) == 0:
            continue

        if uid in cold_set:
            # Cold user: all ratings go to test_cold
            test_cold_rows.extend([uid] * len(cols))
            test_cold_cols.extend(cols)
            test_cold_data.extend(vals)
        else:
            # Warm user: split ratings
            n_test = max(1, int(len(cols) * test_ratio))
            test_mask = np.zeros(len(cols), dtype=bool)
            test_indices = rng.choice(len(cols), n_test, replace=False)
            test_mask[test_indices] = True

            # Train entries
            train_rows.extend([uid] * int((~test_mask).sum()))
            train_cols.extend(cols[~test_mask])
            train_data.extend(vals[~test_mask])

            # Test entries
            test_warm_rows.extend([uid] * int(test_mask.sum()))
            test_warm_cols.extend(cols[test_mask])
            test_warm_data.extend(vals[test_mask])

    shape = (n_users, n_items)
    train_matrix = csr_matrix((train_data, (train_rows, train_cols)), shape=shape)
    test_warm = csr_matrix((test_warm_data, (test_warm_rows, test_warm_cols)), shape=shape)
    test_cold = csr_matrix((test_cold_data, (test_cold_rows, test_cold_cols)), shape=shape)

    return train_matrix, test_warm, test_cold, warm_user_indices, cold_user_indices
```

- [ ] **Step 2: Rewrite `load_and_preprocess()`**

```python
def load_and_preprocess(cfg: DataConfig = DataConfig()) -> RecommenderData:
    """Full pipeline: load → clean → build sparse → sample → filter → split."""
    # Load CSVs
    ratings_df = pd.read_csv(cfg.ratings_csv)
    movies_df = pd.read_csv(cfg.movies_csv)
    if "timestamp" in ratings_df.columns:
        ratings_df = ratings_df.drop(columns=["timestamp"])

    merged = pd.merge(ratings_df, movies_df[["movieId", "genres", "title"]], on="movieId")
    merged["year"] = merged["title"].apply(_extract_year)
    merged = merged.dropna(subset=["year"])
    merged["year"] = merged["year"].astype(int)

    # Build sparse matrix
    user_ids = merged["userId"].unique()
    movie_ids = merged["movieId"].unique()
    user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
    movie_id_map = {mid: idx for idx, mid in enumerate(movie_ids)}

    rows = merged["userId"].map(user_id_map).values
    cols = merged["movieId"].map(movie_id_map).values
    data = merged["rating"].values
    full_matrix = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(movie_ids)))

    # Sample users
    sampled, sampled_user_map = _sample_users(full_matrix, user_id_map, cfg.sample_fraction, cfg.random_seed)

    # Filter inactive users — BEFORE split
    sampled = _remove_inactive_users(sampled, threshold=cfg.min_ratings_per_user)

    # Split into train/test
    train_matrix, test_warm, test_cold, warm_indices, cold_indices = _split_train_test(
        sampled, cfg.cold_user_fraction, cfg.test_ratio, cfg.random_seed,
    )

    # Build metadata
    index_to_user = {idx: uid for uid, idx in sampled_user_map.items()}
    index_to_movie = {idx: mid for mid, idx in movie_id_map.items()}
    aligned_movies = pd.DataFrame({"movieId": [index_to_movie[i] for i in range(len(movie_ids))]})
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
```

- [ ] **Step 3: Rebuild user_id_map after filtering**

`_remove_inactive_users` removes rows and changes matrix shape. The old `sampled_user_map` becomes stale. Add this after the filter step in `load_and_preprocess`:

```python
    # Filter inactive users — BEFORE split
    pre_filter_n = sampled.shape[0]
    row_nnz = np.diff(sampled.indptr)
    active_mask = row_nnz >= cfg.min_ratings_per_user
    sampled = sampled[active_mask]

    # Rebuild user_id_map to reflect new row indices
    old_to_new = {}
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
```

This replaces the `_remove_inactive_users()` call + manual rebuild.

- [ ] **Step 4: Remove dead code**

Delete `_remove_user_ratings()` and `_remove_all_user_ratings()`.

- [ ] **Step 5: Commit**

```bash
git add src/data_utils.py
git commit -m "refactor: rewrite data_utils with train/test split and warm/cold user separation"
```

---

### Task 4: Write split validation tests

**Files:**
- Create: `tests/test_data_split.py`

- [ ] **Step 1: Write tests**

```python
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
    """Train and test_warm must have zero overlapping entries."""
    mat = _make_test_matrix()
    train, test_warm, _, _, _ = _split_train_test(mat, 0.2, 0.2, 42)
    overlap = train.multiply(test_warm)
    assert overlap.nnz == 0, f"Found {overlap.nnz} overlapping entries"


def test_no_overlap_train_test_cold():
    """Train and test_cold must have zero overlapping entries."""
    mat = _make_test_matrix()
    train, _, test_cold, _, _ = _split_train_test(mat, 0.2, 0.2, 42)
    overlap = train.multiply(test_cold)
    assert overlap.nnz == 0, f"Found {overlap.nnz} overlapping entries"


def test_cold_users_empty_in_train():
    """Cold users must have exactly 0 non-zero entries in train_matrix."""
    mat = _make_test_matrix()
    train, _, _, _, cold_indices = _split_train_test(mat, 0.2, 0.2, 42)
    for uid in cold_indices:
        assert train[uid].nnz == 0, f"Cold user {uid} has {train[uid].nnz} train entries"


def test_warm_users_reconstruct():
    """Warm users' train + test_warm entries must reconstruct original ratings."""
    mat = _make_test_matrix()
    train, test_warm, _, warm_indices, _ = _split_train_test(mat, 0.2, 0.2, 42)
    for uid in warm_indices:
        original = mat[uid].toarray().flatten()
        reconstructed = train[uid].toarray().flatten() + test_warm[uid].toarray().flatten()
        np.testing.assert_array_almost_equal(original, reconstructed)


def test_cold_users_reconstruct():
    """Cold users' test_cold entries must equal original ratings."""
    mat = _make_test_matrix()
    _, _, test_cold, _, cold_indices = _split_train_test(mat, 0.2, 0.2, 42)
    for uid in cold_indices:
        original = mat[uid].toarray().flatten()
        cold = test_cold[uid].toarray().flatten()
        np.testing.assert_array_almost_equal(original, cold)


def test_total_nnz_preserved():
    """Total non-zero entries across all 3 matrices must equal original."""
    mat = _make_test_matrix()
    train, test_warm, test_cold, _, _ = _split_train_test(mat, 0.2, 0.2, 42)
    assert train.nnz + test_warm.nnz + test_cold.nnz == mat.nnz


def test_cold_fraction_approximate():
    """Number of cold users should be approximately cold_user_fraction * n_users."""
    mat = _make_test_matrix(n_users=100)
    _, _, _, _, cold_indices = _split_train_test(mat, 0.2, 0.2, 42)
    assert 15 <= len(cold_indices) <= 25  # 20% of 100 = 20, allow some tolerance


def test_reproducibility():
    """Same seed must produce identical splits."""
    mat = _make_test_matrix()
    r1 = _split_train_test(mat, 0.2, 0.2, 42)
    r2 = _split_train_test(mat, 0.2, 0.2, 42)
    assert (r1[0] - r2[0]).nnz == 0
    assert np.array_equal(r1[3], r2[3])
    assert np.array_equal(r1[4], r2[4])
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/chang.yy/Documents/NAS/7_Code/1.\ Git_Repos/1.\ Recommender\ System\ a\ basic\ discussion && python -m pytest tests/test_data_split.py -v`

Expected: All 8 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_data_split.py
git commit -m "test: add split validation tests for train/test separation"
```

---

### Task 5: Update `src/evaluation.py`

**Files:**
- Modify: `src/evaluation.py`

- [ ] **Step 1: Add warm/cold/overall evaluation functions**

Append after existing `aggregate_results()`:

```python
@dataclass(frozen=True)
class SubsetEvalResult:
    """Evaluation result for a subset of users, carrying SSE for pooled RMSE."""
    eval_result: EvalResult
    total_sse: float          # sum of squared errors (for pooled RMSE)
    total_rated_count: int    # number of rated entries evaluated


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
        # Accumulate SSE for pooled RMSE
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

    # Pooled RMSE: sqrt((SSE_warm + SSE_cold) / (n_warm + n_cold))
    pooled_rmse = float(np.sqrt((warm.total_sse + cold.total_sse) / total))

    return EvalResult(
        recall=wr.recall * w_warm + cr.recall * w_cold,
        precision=wr.precision * w_warm + cr.precision * w_cold,
        accuracy=wr.accuracy * w_warm + cr.accuracy * w_cold,
        rmse=pooled_rmse,
        f1=wr.f1 * w_warm + cr.f1 * w_cold,
    )
```

Add `from tqdm import tqdm` to imports at top of file.

**Note:** `evaluate_on_subset` now clips predictions to [0.5, 5.0] internally, so model runners do not need to clip individually. `SubsetEvalResult` carries SSE for correct pooled RMSE computation.

- [ ] **Step 2: Write eval tests**

Create `tests/test_evaluation.py`:

```python
"""Tests for warm/cold/overall evaluation functions."""
import numpy as np
from src.evaluation import EvalResult, evaluate_overall


def test_overall_pure_warm():
    """When there are no cold ratings, overall equals warm."""
    warm = EvalResult(recall=0.8, precision=0.7, accuracy=0.75, rmse=0.5, f1=0.74)
    cold = EvalResult(recall=0.0, precision=0.0, accuracy=0.0, rmse=0.0, f1=0.0)
    result = evaluate_overall(warm, cold, n_warm_ratings=100, n_cold_ratings=0)
    assert abs(result.f1 - 0.74) < 1e-6


def test_overall_equal_weights():
    """When equal ratings, overall is simple average."""
    warm = EvalResult(recall=0.8, precision=0.8, accuracy=0.8, rmse=0.4, f1=0.8)
    cold = EvalResult(recall=0.6, precision=0.6, accuracy=0.6, rmse=0.8, f1=0.6)
    result = evaluate_overall(warm, cold, n_warm_ratings=50, n_cold_ratings=50)
    assert abs(result.f1 - 0.7) < 1e-6
    assert abs(result.rmse - 0.6) < 1e-6
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_evaluation.py -v`

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/evaluation.py tests/test_evaluation.py
git commit -m "feat: add warm/cold/overall evaluation functions with weighted averaging"
```

---

### Task 6: Rewrite `src/train.py` — Evaluation Helper

**Files:**
- Modify: `src/train.py`

- [ ] **Step 1: Replace `evaluate_all_users` with 3-way evaluation**

Replace the old `evaluate_all_users()` function (lines 185-198) with:

```python
def run_evaluation(
    data: RecommenderData,
    predict_fn,
    threshold: float = 3.5,
    desc: str = "Model",
) -> dict:
    """Run warm/cold/overall evaluation and return results dict."""
    warm = evaluate_on_subset(
        data.test_warm, predict_fn, data.warm_user_indices,
        threshold=threshold, desc=f"{desc} Warm",
    )
    cold = evaluate_on_subset(
        data.test_cold, predict_fn, data.cold_user_indices,
        threshold=threshold, desc=f"{desc} Cold",
    )
    overall_result = evaluate_overall(warm, cold)

    print(f"  Warm:    {warm.eval_result}")
    print(f"  Cold:    {cold.eval_result}")
    print(f"  Overall: {overall_result}")

    return {
        "warm": warm.eval_result.as_dict(),
        "cold": cold.eval_result.as_dict(),
        "overall": overall_result.as_dict(),
    }
```

- [ ] **Step 2: Update imports**

Replace `from src.evaluation import EvalResult, aggregate_results, evaluate, scale_predictions` with:

```python
from src.evaluation import (
    EvalResult,
    aggregate_results,
    evaluate,
    evaluate_on_subset,
    evaluate_overall,
    scale_predictions,
)
from src.data_utils import RecommenderData, load_and_preprocess
```

- [ ] **Step 3: Commit**

```bash
git add src/train.py
git commit -m "refactor: replace evaluate_all_users with 3-way warm/cold/overall evaluation"
```

---

### Task 7: Rewrite `src/train.py` — Model Runners (Traditional)

**Files:**
- Modify: `src/train.py` (functions `run_content_based`, `run_collaborative_filtering`, `run_svt`, `run_als`)

- [ ] **Step 1: Rewrite `run_content_based`**

```python
def run_content_based(data: RecommenderData, cfg: TrainConfig) -> dict:
    print("\n" + "=" * 60)
    print("CONTENT-BASED FILTERING")
    print("=" * 60)

    cb = ContentBasedRecommender()
    cb.fit(data.movies_df["genres"])

    def predict_fn(uid):
        user_train_ratings = data.train_matrix[uid]
        if user_train_ratings.nnz == 0:
            return cb.predict_cold_start(data.train_matrix)
        return cb.predict_for_user(user_train_ratings)

    return run_evaluation(data, predict_fn, threshold=cfg.rating_threshold, desc="CB")
```

- [ ] **Step 2: Rewrite `run_collaborative_filtering`**

```python
def run_collaborative_filtering(data: RecommenderData, cfg: TrainConfig) -> dict:
    print("\n" + "=" * 60)
    print("COLLABORATIVE FILTERING")
    print("=" * 60)

    results = {}
    filters = {
        "Cosine": CosineCollaborativeFilter(k=5),
        "AdjustedCosine": AdjustedCosineCollaborativeFilter(k=5, min_common=3),
        "Pearson": PearsonCollaborativeFilter(k=5),
    }
    for name, filt in filters.items():
        print(f"\n  --- {name} ---")
        filt.fit(data.train_matrix)
        cold_pred = filt.predict_cold_start()

        def predict_fn(uid, f=filt, cp=cold_pred):
            if data.train_matrix[uid].nnz == 0:
                return cp
            return f.predict_for_user(uid)

        results[name] = run_evaluation(data, predict_fn, threshold=cfg.rating_threshold, desc=f"CF-{name}")
    return results
```

- [ ] **Step 3: Rewrite `run_svt`**

```python
def run_svt(data: RecommenderData, cfg: TrainConfig) -> dict:
    print("\n" + "=" * 60)
    print("SVT MATRIX COMPLETION")
    print("=" * 60)

    results = {}
    train_dense = data.train_matrix.toarray()
    filled = prefill_user_mean(train_dense)

    for name, svt_cfg in SVT_VARIANTS.items():
        print(f"\n  --- {name} ---")
        svt = SVTMatrixCompletion(
            beta=svt_cfg.beta, tau=svt_cfg.tau,
            tolerance=svt_cfg.tolerance, max_iter=svt_cfg.max_iter,
        )
        svt.fit(filled)
        results[name] = run_evaluation(
            data, lambda uid, s=svt: s.predict(uid),
            threshold=cfg.rating_threshold, desc=f"SVT-{name}",
        )
    return results
```

- [ ] **Step 4: Rewrite `run_als`**

```python
def run_als(data: RecommenderData, cfg: TrainConfig) -> dict:
    print("\n" + "=" * 60)
    print("ALS MATRIX COMPLETION")
    print("=" * 60)

    train_dense = data.train_matrix.toarray()
    filled = prefill_global_mean(train_dense)
    als = ALSMatrixCompletion(rank=20).fit(filled)

    return run_evaluation(
        data, lambda uid, a=als: a.predict(uid),
        threshold=cfg.rating_threshold, desc="ALS",
    )
```

- [ ] **Step 5: Commit**

```bash
git add src/train.py
git commit -m "refactor: rewrite traditional model runners for train/test split evaluation"
```

---

### Task 8: Rewrite `src/train.py` — Model Runners (Neural)

**Files:**
- Modify: `src/train.py` (functions `run_mlp`, `run_autoencoder`, `run_matrix_factorization`)

- [ ] **Step 1: Rewrite `run_mlp`**

```python
def run_mlp(data: RecommenderData, cfg: TrainConfig, device: torch.device) -> dict:
    print("\n" + "=" * 60)
    print("MLP RECOMMENDER")
    print("=" * 60)

    n_users, n_items = data.train_matrix.shape
    train_loader, val_loader = prepare_dataloaders(data.train_matrix, cfg)

    results = {}
    for name, mlp_cfg in MLP_VARIANTS.items():
        print(f"\n  --- {name} ---")
        model = MLPRecommender(
            n_users, n_items,
            embedding_dim=mlp_cfg.embedding_dim,
            hidden_layers=mlp_cfg.layers,
            dropout=mlp_cfg.dropout,
            l2_lambda=mlp_cfg.l2_lambda,
        ).to(device)

        train_neural_model(
            model, train_loader, val_loader,
            criterion=nn.MSELoss(), device=device, cfg=cfg,
            checkpoint_name=f"mlp_{name}", add_l2=True,
        )

        def predict_fn(uid, m=model):
            return m.predict_all_items(uid, n_items, device)

        results[name] = run_evaluation(data, predict_fn, threshold=cfg.rating_threshold, desc=f"MLP-{name}")
    return results
```

- [ ] **Step 2: Rewrite `run_autoencoder`**

```python
def run_autoencoder(data: RecommenderData, cfg: TrainConfig, device: torch.device) -> dict:
    print("\n" + "=" * 60)
    print("AUTOENCODER RECOMMENDER")
    print("=" * 60)

    n_users, n_items = data.train_matrix.shape
    results = {}

    for name, ae_cfg in AE_VARIANTS.items():
        print(f"\n  --- {name} (mode={ae_cfg.mode}) ---")
        input_dim = n_items if ae_cfg.mode == "user" else n_users

        ae = AutoEncoderRecommender(
            input_dim, encoding_dims=ae_cfg.encoding_dims,
            device=device, mode=ae_cfg.mode,
        )
        ae.fit(data.train_matrix.toarray(), epochs=cfg.epochs, batch_size=cfg.batch_size)

        def predict_fn(uid, _ae=ae):
            preds = _ae.predict_for_user(data.train_matrix, uid)
            return scale_predictions(preds)

        results[name] = run_evaluation(data, predict_fn, threshold=cfg.rating_threshold, desc=f"AE-{name}")
    return results
```

- [ ] **Step 3: Rewrite `run_matrix_factorization`**

```python
def run_matrix_factorization(data: RecommenderData, cfg: TrainConfig, device: torch.device) -> dict:
    print("\n" + "=" * 60)
    print("NEURAL MATRIX FACTORIZATION")
    print("=" * 60)

    n_users, n_items = data.train_matrix.shape
    train_loader, val_loader = prepare_dataloaders(data.train_matrix, cfg)

    results = {}
    for name, mf_cfg in MF_VARIANTS.items():
        print(f"\n  --- {name} (dim={mf_cfg.embedding_dim}) ---")
        model = NeuralMatrixFactorization(
            n_users, n_items, embedding_dim=mf_cfg.embedding_dim,
        ).to(device)

        train_neural_model(
            model, train_loader, val_loader,
            criterion=WeightedMSELoss(), device=device, cfg=cfg,
            checkpoint_name=f"mf_{name}",
        )

        def predict_fn(uid, m=model):
            return m.predict_all_items(uid, n_items, device)

        results[name] = run_evaluation(data, predict_fn, threshold=cfg.rating_threshold, desc=f"MF-{name}")
    return results
```

- [ ] **Step 4: Update `main()` — remove old partial/cold logic**

Update `main()` to:
- Call `load_and_preprocess(data_cfg)` (returns new RecommenderData)
- Print train/test stats instead of ground_truth stats
- Each model runner is called once (not twice for partial/cold)
- Save results in new format (warm/cold/overall per variant)

```python
def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()

    data_cfg = DataConfig(
        sample_fraction=args.sample_fraction,
        random_seed=args.seed,
        **({"movies_csv": Path(args.data_dir) / "movies.csv",
            "ratings_csv": Path(args.data_dir) / "ratings.csv"} if args.data_dir else {}),
    )
    train_cfg = TrainConfig(
        batch_size=args.batch_size, epochs=args.epochs,
        learning_rate=args.lr, random_seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("LOADING DATA & SPLITTING")
    print("=" * 60)
    data = load_and_preprocess(data_cfg)
    print(f"  Matrix shape:       {data.train_matrix.shape}")
    print(f"  Train entries:      {data.train_matrix.nnz}")
    print(f"  Test warm entries:  {data.test_warm.nnz}")
    print(f"  Test cold entries:  {data.test_cold.nnz}")
    print(f"  Warm users:         {len(data.warm_user_indices)}")
    print(f"  Cold users:         {len(data.cold_user_indices)}")

    all_results = {}
    model_runners = {
        "cb": lambda: run_content_based(data, train_cfg),
        "cf": lambda: run_collaborative_filtering(data, train_cfg),
        "svt": lambda: run_svt(data, train_cfg),
        "als": lambda: run_als(data, train_cfg),
        "mlp": lambda: run_mlp(data, train_cfg, device),
        "ae": lambda: run_autoencoder(data, train_cfg, device),
        "mf": lambda: run_matrix_factorization(data, train_cfg, device),
    }

    total_start = time.time()
    for model_name in args.models:
        t0 = time.time()
        all_results[model_name] = model_runners[model_name]()
        print(f"\n  [{model_name}] completed in {time.time() - t0:.1f}s")

    results_path = CHECKPOINT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nALL DONE — {time.time() - total_start:.1f}s")
    print(f"Results: {results_path}")
```

- [ ] **Step 5: Commit**

```bash
git add src/train.py
git commit -m "refactor: rewrite neural model runners and main() for new evaluation pipeline"
```

---

### Task 9: Integration Test — Full Pipeline

**Files:**
- No new files; run existing pipeline

- [ ] **Step 1: Dry run with content-based only (fastest model)**

Run: `cd /Users/chang.yy/Documents/NAS/7_Code/1.\ Git_Repos/1.\ Recommender\ System\ a\ basic\ discussion && python -m src.train --models cb`

Expected: prints train/test stats, warm/cold/overall metrics for CB. No errors.

- [ ] **Step 2: Run all tests**

Run: `python -m pytest tests/ -v`

Expected: All tests pass.

- [ ] **Step 3: Full pipeline run**

Run: `python -m src.train`

Expected: All 7 models run, `checkpoints/results.json` contains warm/cold/overall for each variant.

- [ ] **Step 4: Validate results.json format**

```python
python -c "
import json
with open('checkpoints/results.json') as f:
    r = json.load(f)
for model, variants in r.items():
    if isinstance(variants, dict) and 'warm' in variants:
        print(f'{model}: warm/cold/overall present')
    else:
        for var, metrics in variants.items():
            assert 'warm' in metrics, f'{model}/{var} missing warm'
            print(f'{model}/{var}: warm/cold/overall present')
print('Format OK')
"
```

- [ ] **Step 5: Verify results.json (do NOT git commit — checkpoints are gitignored)**

Confirm the file exists and has the correct warm/cold/overall structure. Results are regenerated by running the pipeline, not version-controlled.

---

## Execution Order & Dependencies

```
Task 1 (config)
    ↓
Task 2 (data container) → Task 3 (split logic) → Task 4 (split tests)
    ↓
Task 5 (evaluation functions)
    ↓
Task 6 (eval helper in train.py)
    ↓
Task 7 (traditional runners) ─┐
Task 8 (neural runners)       ├→ Task 9 (integration test)
                              ─┘
```

Tasks 7 and 8 can be done in parallel.
