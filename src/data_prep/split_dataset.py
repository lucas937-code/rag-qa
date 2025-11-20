# src/data_prep/split_dataset.py

"""
Utilities to create deterministic train/validation splits from a raw
train Parquet file and to inspect splits locally.

Typical usage in Colab (full run):
----------------------------------
from src.data_prep.load_dataset import prepare_full_dataset
from src.data_prep.split_dataset import create_train_val_split

# 1) Full dataset -> Parquet (e.g. in Google Drive)
paths = prepare_full_dataset(output_dir=RAW_DIR, ...)

# 2) Deterministic split: first 7 900 examples become validation
split_paths = create_train_val_split(
    raw_train_path=paths["train"],
    output_dir=PROCESSED_DIR + "/splits",
    val_size=7900,
)

Typical usage locally (quick inspection):
-----------------------------------------
from src.data_prep.split_dataset import get_local_split_preview

df_preview = get_local_split_preview("data/processed/splits/train.parquet", max_examples=50)
print(df_preview.head())
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Literal
import json
import logging

import pandas as pd

logger = logging.getLogger(__name__)


# =====================
# Low-level helpers
# =====================

def _load_parquet(path: str | Path) -> pd.DataFrame:
    """
    Load a Parquet file into a pandas DataFrame.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    df = pd.read_parquet(path)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def _save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """
    Save a DataFrame as Parquet and return the path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info("Saved %d rows to %s", len(df), path)
    return path


def _write_meta_file(meta: Dict, path: str | Path) -> Path:
    """
    Write a small JSON metadata file next to the splits.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Wrote split metadata to %s", path)
    return path


# =====================
# Public API: main split creation (Colab / full run)
# =====================

def create_train_val_split(
    raw_train_path: str | Path,
    output_dir: str | Path,
    val_size: int = 7900,
    strategy: Literal["first_n", "random"] = "first_n",
    random_state: int = 42,
    keep_original_index: bool = False,
) -> Dict[str, Path]:
    """
    Create a deterministic train/validation split from a raw train Parquet file.

    Default behaviour (strategy="first_n"):
        - Take the first `val_size` examples (after reset_index) as validation.
        - The remaining examples become train.

    This matches typical project requirements like "use the first 7 900 examples
    as validation set".

    Parameters
    ----------
    raw_train_path:
        Path to the raw train Parquet file, as produced by `prepare_full_dataset`.
    output_dir:
        Directory where the split Parquet files will be written.
        Typically something like `PROCESSED_DIR + "/splits"`.
    val_size:
        Number of examples to put into the validation split (default: 7 900).
    strategy:
        "first_n" (default): use the first `val_size` rows as validation.
        "random": randomly sample `val_size` rows for validation.
    random_state:
        Random seed used when strategy="random".
    keep_original_index:
        If False (default), reset the index of the output DataFrames.
        If True, preserve the original row index from the raw file.

    Returns
    -------
    Dict[str, Path] with keys:
        - "train": path to the new train Parquet file
        - "val": path to the new validation Parquet file
        - "meta": path to a JSON metadata file

    Notes
    -----
    - This function should be run once per dataset version (e.g. once in Colab).
    - The resulting parquet files can be shared via Google Drive and reused
      by all team members.
    """
    raw_train_path = Path(raw_train_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_parquet(raw_train_path)

    n_total = len(df)
    if val_size >= n_total:
        raise ValueError(
            f"val_size ({val_size}) must be smaller than the number of rows "
            f"in the raw train set ({n_total})."
        )

    if strategy == "first_n":
        df = df.reset_index(drop=False, names="orig_index")
        val_df = df.iloc[:val_size].copy()
        train_df = df.iloc[val_size:].copy()
        if not keep_original_index:
            val_df = val_df.reset_index(drop=True)
            train_df = train_df.reset_index(drop=True)
    elif strategy == "random":
        df = df.reset_index(drop=False, names="orig_index")
        val_df = df.sample(n=val_size, random_state=random_state)
        train_df = df.drop(val_df.index)
        if not keep_original_index:
            val_df = val_df.reset_index(drop=True)
            train_df = train_df.reset_index(drop=True)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    train_path = output_dir / "train.parquet"
    val_path = output_dir / f"val_{val_size}.parquet"
    meta_path = output_dir / "splits_meta.json"

    _save_parquet(train_df, train_path)
    _save_parquet(val_df, val_path)

    meta = {
        "raw_train_path": str(raw_train_path),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "n_total_raw_train": n_total,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "val_size": val_size,
        "strategy": strategy,
        "random_state": random_state if strategy == "random" else None,
        "keep_original_index": keep_original_index,
    }
    _write_meta_file(meta, meta_path)

    return {
        "train": train_path,
        "val": val_path,
        "meta": meta_path,
    }


# =====================
# Public API: local preview of existing splits
# =====================

def get_local_split_preview(
    split_path: str | Path,
    max_examples: int = 50,
    as_pandas: bool = True,
) -> pd.DataFrame:
    """
    Load a small preview of an existing split (train or val) for
    local development.

    Parameters
    ----------
    split_path:
        Path to a Parquet file, e.g. "processed/splits/train.parquet".
    max_examples:
        Maximum number of examples to return.
    as_pandas:
        Reserved flag in case you later want a different format.
        Currently always returns a pandas.DataFrame.

    Returns
    -------
    pandas.DataFrame with at most `max_examples` rows.
    """
    df = _load_parquet(split_path)
    if max_examples is not None and max_examples < len(df):
        df = df.iloc[:max_examples].copy()
    if not as_pandas:
        # For now we only support pandas; extend if needed.
        logger.warning("as_pandas=False is not implemented, returning DataFrame.")
    return df
