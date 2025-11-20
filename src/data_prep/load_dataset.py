# src/data_prep/load_dataset.py

"""
Utilities to load the raw QA dataset both for local development (small samples)
and for full processing in Colab (full splits saved to disk).

Design:
- get_local_sample(...)  -> small subset for quick experiments on a laptop
- prepare_full_dataset(...) -> download full dataset and write splits to disk
"""

from typing import Optional, Sequence, Dict
from pathlib import Path
import logging

import pandas as pd
from datasets import load_dataset, Dataset, IterableDataset
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


# =====================
# Low-level helper
# =====================

def _load_hf_split(
    dataset_name: str,
    subset: Optional[str],
    split: str,
    streaming: bool = False,
) -> Dataset | IterableDataset:
    """
    Load a single split from Hugging Face Datasets.

    Parameters
    ----------
    dataset_name:
        Name on the HF Hub, e.g. "trivia_qa".
    subset:
        Optional subset/config, e.g. "rc.wikipedia".
    split:
        Split name, e.g. "train", "validation", "test".
    streaming:
        If True, use streaming mode (IterableDataset).

    Returns
    -------
    HF Dataset or IterableDataset.
    """
    if subset is None:
        ds = load_dataset(dataset_name, split=split, streaming=streaming)
    else:
        ds = load_dataset(dataset_name, subset, split=split, streaming=streaming)

    logger.info(
        "Loaded HF split %s (%s, subset=%s, streaming=%s)",
        split, dataset_name, subset, streaming
    )
    return ds


def _save_split_to_parquet(
    ds: Dataset,
    out_path: Path,
    max_examples: Optional[int] = None,
) -> Path:
    """
    Save a (non-streaming) HF Dataset split to Parquet.

    Parameters
    ----------
    ds:
        HF Dataset (not streaming).
    out_path:
        Target Parquet file.
    max_examples:
        Optionally limit the number of examples (e.g. for dev).

    Returns
    -------
    Path to the written Parquet file.
    """
    if isinstance(ds, IterableDataset):
        raise ValueError("Streaming dataset cannot be directly converted to Parquet")

    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    df = ds.to_pandas()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)

    logger.info("Saved %d rows to %s", len(df), out_path)
    return out_path


# =====================
# Public API: local dev
# =====================

def get_local_sample(
    max_examples: int = 500,
    split: str = "train",
    dataset_name: str = "trivia_qa",
    subset: Optional[str] = "rc.wikipedia",
    streaming: bool = False,
    as_pandas: bool = True,
) -> pd.DataFrame | Dataset | IterableDataset:
    """
    Get a small sample from the dataset for local experimentation.

    Intended for use on laptops without downloading or materializing everything.

    Parameters
    ----------
    max_examples:
        Number of examples to load (approximate for streaming).
    split:
        Which split to use, e.g. "train".
    dataset_name:
        HF dataset name.
    subset:
        HF config / subset, e.g. "rc.wikipedia".
    streaming:
        If True, use streaming mode to avoid loading the full split.
    as_pandas:
        If True and not streaming, return a pandas DataFrame.
        If False, return the HF Dataset / IterableDataset.

    Returns
    -------
    - If streaming:
        IterableDataset (you can manually iterate over a few examples).
    - If not streaming:
        pandas.DataFrame with at most max_examples rows (default),
        or HF Dataset if as_pandas=False.
    """
    ds = _load_hf_split(dataset_name, subset, split, streaming=streaming)

    # streaming mode: we iterate manually up to max_examples and build a DataFrame
    if streaming:
        logger.info(
            "Using streaming mode for local sample (max_examples=%d)", max_examples
        )
        rows = []
        for i, example in enumerate(ds):
            rows.append(example)
            if i + 1 >= max_examples:
                break
        df = pd.DataFrame(rows)
        return df if as_pandas else ds

    # non-streaming mode: we can use select to get a subset
    if max_examples is not None:
        ds_small = ds.select(range(min(max_examples, len(ds))))
    else:
        ds_small = ds

    if as_pandas:
        return ds_small.to_pandas()
    return ds_small


# =====================
# Public API: full dataset (e.g. Colab)
# =====================

def stream_hf_to_parquet(
    dataset_name: str,
    subset: str | None,
    split: str,
    out_file: str,
    batch_size: int = 1000,
):
    """
    Stream a HF Dataset split and write it incrementally to Parquet.

    - Does NOT load entire dataset into RAM.
    - Safe for Colab.

    Parameters
    ----------
    dataset_name, subset, split: HF identifiers
    out_file: absolute path to Parquet file (in Google Drive)
    batch_size: how many rows to accumulate before writing a chunk
    """
    from datasets import load_dataset
    import pandas as pd
    from pathlib import Path

    ds_stream = load_dataset(
        dataset_name,
        subset,
        split=split,
        streaming=True
    )

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Arrow/Parquet writer prepared later
    writer = None
    total_rows = 0
    batch = []

    for row in ds_stream:
        batch.append(row)

        if len(batch) >= batch_size:
            table = pa.Table.from_pylist(batch)
            batch = []

            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema)
            writer.write_table(table)

            total_rows += table.num_rows
            print(f"Wrote {total_rows} rows...")

    # letzte (unvollständige) Batch
    if batch:
        table = pa.Table.from_pylist(batch)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
        total_rows += table.num_rows

    if writer is not None:
        writer.close()

    print(f"Finished streaming {total_rows} rows to {out_path}")
    return str(out_path)


def prepare_full_dataset(
    output_dir: str,
    dataset_name: str = "trivia_qa",
    subset: Optional[str] = "rc.wikipedia",
    splits: Sequence[str] = ("train", "validation"),
    use_streaming: bool = True,
    batch_size: int = 1000,
) -> Dict[str, Path]:
    """
    Download/load the full dataset splits and save them as Parquet files.

    Intended for use in Colab (e.g. with a Google Drive-backed output_dir).

    Parameters
    ----------
    output_dir:
        Directory where Parquet files should be written.
    dataset_name:
        HF dataset name.
    subset:
        HF config / subset.
    splits:
        List of split names to materialize.
    max_examples_per_split:
        Optional dict split->max_examples (mainly for debugging),
        e.g. {"train": 100000} to cap the train split during development.

    Returns
    -------
    Dict mapping split name -> Path to Parquet file.
    """
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}

    for split in splits:
        out_file = base / f"{split}.parquet"

        if use_streaming:
            # Streaming-Version (für große Splits / Colab)
            stream_hf_to_parquet(
                dataset_name=dataset_name,
                subset=subset,
                split=split,
                out_file=str(out_file),
                batch_size=batch_size,
            )
        else:
            # Fallback: kleine Splits klassisch laden (nicht streaming)
            ds = _load_hf_split(dataset_name, subset, split, streaming=False)
            _save_split_to_parquet(ds, out_file, max_examples=None)

        paths[split] = out_file

    return paths
