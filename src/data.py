"""Loading, validating, cleaning and splitting the WDBC dataset."""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def download_dataset(dest: Path = config.DATA_FILE) -> Path:
    """Download the Kaggle mirror of WDBC and copy data.csv into data/raw/."""
    import kagglehub  # imported lazily: only needed for the download step

    source_dir = Path(kagglehub.dataset_download(config.KAGGLE_HANDLE))
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source_dir / "data.csv", dest)
    return dest


def load_raw(path: Path = config.DATA_FILE) -> pd.DataFrame:
    """Read the raw CSV; fail with an actionable message if it is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.download_data` "
            "or follow the instructions in data/README.md."
        )
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-informative columns and encode the target (M -> 1, B -> 0)."""
    df = df.drop(columns=[c for c in config.DROP_COLUMNS if c in df.columns])
    df[config.TARGET] = df[config.TARGET].map({"M": 1, "B": 0})
    return df


def validate(df: pd.DataFrame) -> None:
    """Integrity checks (these replace the assert-based checks in the notebook)."""
    if df.shape[1] != config.N_FEATURES + 1:
        raise ValueError(f"Expected {config.N_FEATURES + 1} columns, got {df.shape[1]}.")
    if df.isna().any().any():
        raise ValueError("Dataset contains missing values (or unmapped target labels).")
    if set(df[config.TARGET].unique()) != {0, 1}:
        raise ValueError("Target must be binary with values {0, 1}.")


def load_dataset(path: Path = config.DATA_FILE) -> pd.DataFrame:
    """Load, clean and validate the dataset in one call."""
    df = clean(load_raw(path))
    validate(df)
    return df


def split_data(df: pd.DataFrame, test_size: float = config.TEST_SIZE, seed: int = config.SEED):
    """Stratified train/test split. Returns X_train, X_test, y_train, y_test."""
    X = df.drop(columns=[config.TARGET])
    y = df[config.TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
