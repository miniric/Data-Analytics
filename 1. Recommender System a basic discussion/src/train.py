#!/usr/bin/env python3
"""
Main training script for the Recommender System project.

Usage:
    python -m src.train                    # Train all models
    python -m src.train --models mlp mf    # Train specific models

Features:
    - CUDA/MPS auto-detection
    - tqdm progress bars for all training loops
    - Per-epoch status: loss, learning rate, elapsed time
    - Model checkpointing with early stopping
    - Full reproducibility (seeded everything)
    - Warm / cold / overall evaluation for every model
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm, trange

# ---- project imports ----
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    AE_VARIANTS,
    CHECKPOINT_DIR,
    DataConfig,
    MF_VARIANTS,
    MLP_VARIANTS,
    SVT_VARIANTS,
    TrainConfig,
    get_device,
)
from src.data_utils import RecommenderData, load_and_preprocess
from src.evaluation import (
    EvalResult,
    SubsetEvalResult,
    aggregate_results,
    evaluate,
    evaluate_on_subset,
    evaluate_overall,
    scale_predictions,
)
from src.models.autoencoder_recommender import AutoEncoderRecommender
from src.models.collaborative_filtering import (
    AdjustedCosineCollaborativeFilter,
    CosineCollaborativeFilter,
    PearsonCollaborativeFilter,
)
from src.models.content_based import ContentBasedRecommender
from src.models.matrix_completion import (
    ALSMatrixCompletion,
    SVTMatrixCompletion,
    prefill_global_mean,
    prefill_user_mean,
)
from src.models.matrix_factorization import NeuralMatrixFactorization, WeightedMSELoss
from src.models.mlp_recommender import MLPRecommender, RatingDataset


# ==========================================================================
# Reproducibility
# ==========================================================================

def seed_everything(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==========================================================================
# Generic neural training loop (MLP & MF share this)
# ==========================================================================

def train_neural_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    criterion: nn.Module,
    device: torch.device,
    cfg: TrainConfig,
    checkpoint_name: str,
    add_l2: bool = False,
) -> dict:
    """
    Train a neural model with early stopping, LR scheduling, and tqdm.

    Returns a dict of training history.
    """
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.5)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "lr": [], "epoch_time": []}

    ckpt_path = CHECKPOINT_DIR / f"{checkpoint_name}.pt"

    epoch_bar = trange(cfg.epochs, desc=f"Training {checkpoint_name}", unit="epoch")
    for epoch in epoch_bar:
        t0 = time.time()

        # ---- Train ----
        model.train(True)
        train_loss = 0.0
        batch_bar = tqdm(train_loader, desc=f"  Epoch {epoch+1}", leave=False, unit="batch")
        for user_ids, item_ids, ratings in batch_bar:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            ratings = ratings.float().to(device)

            optimizer.zero_grad()
            preds = model(user_ids, item_ids)
            loss = criterion(preds, ratings)
            if add_l2 and hasattr(model, "l2_regularization"):
                loss = loss + model.l2_regularization()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")
        batch_bar.close()

        avg_train = train_loss / len(train_loader)

        # ---- Validate ----
        model.train(False)
        val_loss = 0.0
        with torch.no_grad():
            for user_ids, item_ids, ratings in val_loader:
                user_ids = user_ids.to(device)
                item_ids = item_ids.to(device)
                ratings = ratings.float().to(device)
                preds = model(user_ids, item_ids)
                val_loss += criterion(preds, ratings).item()
        avg_val = val_loss / len(val_loader)

        # ---- Bookkeeping ----
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["lr"].append(current_lr)
        history["epoch_time"].append(elapsed)

        scheduler.step(avg_val)

        epoch_bar.set_postfix(
            train=f"{avg_train:.4f}",
            val=f"{avg_val:.4f}",
            lr=f"{current_lr:.2e}",
            time=f"{elapsed:.1f}s",
        )

        # ---- Early stopping ----
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                print(f"  Early stopping at epoch {epoch+1} (best val={best_val_loss:.4f})")
                break

    epoch_bar.close()
    # Restore best weights
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    print(f"  Best checkpoint loaded from {ckpt_path} (val_loss={best_val_loss:.4f})")
    return history


# ==========================================================================
# Evaluation helper — runs warm / cold / overall
# ==========================================================================

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


# ==========================================================================
# Data preparation helper for embedding-based models
# ==========================================================================

def prepare_dataloaders(matrix, cfg: TrainConfig, seed: int = 42):
    dataset = RatingDataset(matrix)
    gen = torch.Generator().manual_seed(seed)
    train_size = int(cfg.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=gen)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader


# ==========================================================================
# Individual model runners
# ==========================================================================

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


# ==========================================================================
# Main
# ==========================================================================

ALL_MODELS = ["cb", "cf", "svt", "als", "mlp", "ae", "mf"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train Recommender System Models")
    parser.add_argument(
        "--models", nargs="+", default=ALL_MODELS,
        choices=ALL_MODELS,
        help="Which models to train (default: all)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sample-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()

    data_cfg_kwargs = {"sample_fraction": args.sample_fraction, "random_seed": args.seed}
    if args.data_dir:
        data_cfg_kwargs["movies_csv"] = Path(args.data_dir) / "movies.csv"
        data_cfg_kwargs["ratings_csv"] = Path(args.data_dir) / "ratings.csv"
    data_cfg = DataConfig(**data_cfg_kwargs)

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
    print(f"\n{'=' * 60}")
    print(f"ALL DONE — total time: {time.time() - total_start:.1f}s")
    print(f"Results saved to: {results_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
