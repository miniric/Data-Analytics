"""
Shared configuration for the Recommender System project.
"""

from dataclasses import dataclass, field
from pathlib import Path

import torch


def get_device() -> torch.device:
    """Auto-detect best available device: MPS > CUDA > CPU."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("[Device] Apple MPS")
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[Device] CUDA — {torch.cuda.get_device_name(0)}")
        print(f"         Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        dev = torch.device("cpu")
        print("[Device] CPU")
    return dev


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class DataConfig:
    """Immutable data configuration."""
    movies_csv: Path = DATA_DIR / "movies.csv"
    ratings_csv: Path = DATA_DIR / "ratings.csv"
    sample_fraction: float = 0.01
    min_ratings_per_user: int = 5
    cold_user_fraction: float = 0.20
    test_ratio: float = 0.20
    random_seed: int = 42


@dataclass(frozen=True)
class TrainConfig:
    """Immutable training configuration."""
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10
    train_split: float = 0.8
    rating_threshold: float = 3.5
    random_seed: int = 42


@dataclass(frozen=True)
class MLPConfig:
    """MLP architecture variants."""
    embedding_dim: int = 32
    layers: tuple = (32, 16, 8, 4, 2)
    dropout: float = 0.5
    l2_lambda: float = 0.001


@dataclass(frozen=True)
class AutoEncoderConfig:
    """AutoEncoder architecture variants."""
    encoding_dims: tuple = (256, 128, 64)
    dropout: float = 0.5
    mode: str = "user"  # "user" or "item"


@dataclass(frozen=True)
class MFConfig:
    """Neural Matrix Factorization configuration."""
    embedding_dim: int = 64


@dataclass(frozen=True)
class SVTConfig:
    """SVT Matrix Completion configuration."""
    beta: float = 0.3
    tau: float = 100.0
    tolerance: float = 1e-4
    max_iter: int = 100


# ---------------------------------------------------------------------------
# Pre-defined architecture variants for experiments
# ---------------------------------------------------------------------------

MLP_VARIANTS = {
    "MLP-Small": MLPConfig(embedding_dim=32, layers=(32, 8, 2)),
    "MLP-Medium": MLPConfig(embedding_dim=32, layers=(32, 16, 8, 4, 2)),
    "MLP-Large": MLPConfig(embedding_dim=256, layers=(256, 128, 64, 32, 16, 8, 4, 2)),
}

AE_VARIANTS = {
    "AE-User-Small": AutoEncoderConfig(encoding_dims=(256, 128, 64), mode="user"),
    "AE-User-Large": AutoEncoderConfig(encoding_dims=(1024, 512, 256, 128, 64), mode="user"),
    "AE-Item": AutoEncoderConfig(encoding_dims=(256, 128, 64), mode="item"),
}

MF_VARIANTS = {
    "MF-16": MFConfig(embedding_dim=16),
    "MF-64": MFConfig(embedding_dim=64),
    "MF-1024": MFConfig(embedding_dim=1024),
}

SVT_VARIANTS = {
    "SVT-tau1000": SVTConfig(beta=0.3, tau=1000.0),
    "SVT-tau100": SVTConfig(beta=0.3, tau=100.0),
    "SVT-beta1": SVTConfig(beta=1.0, tau=100.0),
}
