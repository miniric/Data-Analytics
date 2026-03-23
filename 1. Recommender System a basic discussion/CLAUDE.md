# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A research project comparing traditional and modern movie recommender systems on the MovieLens dataset, focusing on data sparsity and cold start problems. The study compares content-based filtering, collaborative filtering, matrix completion (SVT/ALS), MLP, AutoEncoder, and neural matrix factorisation under normal-user and cold-start scenarios.

## Project Structure

```
├── src/                          # Training code (Python scripts)
│   ├── config.py                 # All configs (DataConfig, TrainConfig, model variants)
│   ├── data_utils.py             # Data loading, sparse matrix, scenario simulation
│   ├── evaluation.py             # Unified evaluate() + scale_predictions()
│   ├── train.py                  # Main entry point — trains all models
│   └── models/
│       ├── content_based.py      # TF-IDF content-based recommender
│       ├── collaborative_filtering.py  # Cosine / Adjusted Cosine / Pearson
│       ├── matrix_completion.py  # SVT + ALS + pre-fill strategies
│       ├── mlp_recommender.py    # MLP with embeddings (parameterised)
│       ├── autoencoder_recommender.py  # AutoEncoder (user/item mode)
│       └── matrix_factorization.py     # Neural MF with bias terms
├── notebooks/
│   └── EDA.ipynb                 # Detailed exploratory data analysis
├── data/                         # movies.csv + ratings.csv (gitignored)
├── checkpoints/                  # Saved .pt models + results.json (gitignored)
├── run_pipeline.sh               # One-command full training pipeline
└── "Basic Movie Recommenders discussions .ipynb"  # Original notebook (legacy)
```

## Commands

```bash
# Full training pipeline (all models)
./run_pipeline.sh

# Train specific models
python -m src.train --models mlp mf ae

# Train with custom params (CUDA auto-detected)
python -m src.train --epochs 100 --batch-size 128 --lr 0.0005

# Override data directory
python -m src.train --data-dir /path/to/data/

# Run EDA notebook
jupyter notebook notebooks/EDA.ipynb
```

Available model keys: `cb` (content-based), `cf` (collaborative filtering), `svt`, `als`, `mlp`, `ae` (autoencoder), `mf` (matrix factorisation).

## Architecture Decisions

- **Immutable data containers**: `RecommenderData` (frozen dataclass) holds all matrices and mappings. Functions return new objects rather than mutating.
- **Parameterised models**: Each neural architecture (MLP, AE, MF) is a single class with configurable layers/dims — variants defined in `config.py` (`MLP_VARIANTS`, `AE_VARIANTS`, `MF_VARIANTS`).
- **Unified evaluation**: One `evaluate()` function replaces the 4 inconsistent versions from the original notebook. Returns frozen `EvalResult` dataclass.
- **Device auto-detection**: `get_device()` in config.py picks CUDA > MPS > CPU automatically.

## Data Flow

1. `load_and_preprocess()` → `RecommenderData` (ground_truth, partial_ratings, cold_start matrices)
2. Each model runner receives `RecommenderData` + configs
3. `evaluate_all_users()` iterates users with tqdm, calls `evaluate()` per user, returns aggregated `EvalResult`
4. Results saved as JSON to `checkpoints/results.json`

## Key Parameters

- **sample_fraction**: 0.01 (1% of users sampled for tractable experiments)
- **removal_percentage**: 30% of target user's ratings removed for partial scenario
- **rating_threshold**: 3.5 (like/dislike boundary for classification metrics)
- **min_ratings_per_user**: 5 (users below this are filtered out)
