# src/data_prep/chunk_documents.py

"""
Chunking utilities for transforming long evidence documents into
smaller, fixed-length passages suitable for retrieval in a RAG setup.

Design decisions:
- We chunk the `evidence_text` field (plus optional doc_title as metadata).
- Chunking is token-based, using the tokenizer of the embedding model.
- Sliding window with:
    - chunk_size_tokens = 512
    - overlap_tokens    = 128
- Output is stored as Parquet:
    - e.g. train_chunks.parquet, val_chunks.parquet

Each chunk contains:
- chunk_id:      unique identifier ({split}_{example_id}_{chunk_index})
- split:         name of the originating split ("train" or "val")
- example_id:    stable reference to the original row (e.g. question_id)
- doc_title:     title of the source document (if available)
- chunk_text:    decoded text of the chunk
- start_token:   index of first token in original tokenized document
- end_token:     index of last token (exclusive) in original tokenized document
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict
import logging
import json

import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer

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
    logger.info("Saved %d chunk rows to %s", len(df), path)
    return path


def _sliding_window_chunks(
    input_ids: List[int],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Dict[str, int]]:
    """
    Compute sliding windows (in token index space) over a sequence of token IDs.

    Returns a list of dicts with 'start_token' and 'end_token' (exclusive).
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})."
        )

    n_tokens = len(input_ids)
    if n_tokens == 0:
        return []

    step = chunk_size - chunk_overlap
    windows: List[Dict[str, int]] = []

    start = 0
    while start < n_tokens:
        end = min(start + chunk_size, n_tokens)
        windows.append({"start_token": start, "end_token": end})
        if end == n_tokens:
            break
        start += step

    return windows


# =====================
# Public API
# =====================

def chunk_dataset(
    split_path: str | Path,
    out_path: str | Path,
    split_name: Optional[str] = None,
    text_column: str = "evidence_text",
    title_column: str = "doc_titles_json",
    example_id_column: str = "question_id",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size_tokens: int = 512,
    chunk_overlap_tokens: int = 128,
    max_rows: Optional[int] = None,
) -> Path:
    """
    Chunk a split (train/val) of the dataset into fixed-length token-based chunks.

    Parameters
    ----------
    split_path:
        Path to the input Parquet file for a split (e.g. train.parquet, val_7900.parquet).
    out_path:
        Path to the output Parquet file where chunked data will be stored
        (e.g. train_chunks.parquet, val_chunks.parquet).
    split_name:
        Name of the split for metadata, e.g. "train" or "val".
        If None, it will be inferred from the input filename where possible.
    text_column:
        Name of the column containing the main evidence text to chunk.
    title_column:
        Name of the column containing the document title (optional metadata).
    example_id_column:
        Name of the column used as example/document identifier.
        If not present, the row index will be used instead.
    model_name:
        Name of the HF model to load the tokenizer from. Should match the
        embedding model used later for consistency.
    chunk_size_tokens:
        Number of tokens per chunk.
    chunk_overlap_tokens:
        Overlap between consecutive chunks (in tokens).
    max_rows:
        Optional maximum number of rows to process (useful for debugging on a subset).

    Returns
    -------
    Path to the Parquet file with chunked data.

    Notes
    -----
    - This function should typically be run once per split (train/val).
    - The resulting file can be shared via Google Drive among team members.
    """
    print("Neue Version funktioniert!")
    split_path = Path(split_path)
    out_path = Path(out_path)

    logger.info("Chunking dataset from %s to %s", split_path, out_path)
    df = _load_parquet(split_path)

    if max_rows is not None and max_rows < len(df):
        df = df.iloc[:max_rows].copy()
        logger.info("Using only the first %d rows for chunking (debug mode).", max_rows)

    # Infer split_name from file name if not explicitly provided
    if split_name is None:
        # crude heuristic: strip extension, take base name
        split_name = split_path.stem

    # Load tokenizer from the same family as the embedding model
    logger.info("Loading tokenizer for model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    all_chunks = []
    logger.info(
        "Starting token-based sliding window chunking: chunk_size=%d, overlap=%d",
        chunk_size_tokens,
        chunk_overlap_tokens,
    )

    for row_idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Chunking {split_name}"):
        text = row.get(text_column, None)

        # Skip rows with missing or empty text
        if not isinstance(text, str) or not text.strip():
            continue

        # Determine example_id (fallback: row index)
        example_id = row.get(example_id_column, None)
        if example_id is None:
            example_id = row_idx

        raw_titles = row.get(title_column, None)
        doc_title = None

        # doc_titles_json is a JSON-encoded list of titles; we take the first one if available
        if isinstance(raw_titles, str):
            try:
                titles = json.loads(raw_titles)
                if isinstance(titles, list) and titles:
                    doc_title = titles[0]
            except json.JSONDecodeError:
                # Fallback: keep the raw string if JSON is malformed for some reason
                doc_title = raw_titles


        # Tokenize
        encoding = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids: List[int] = encoding["input_ids"]

        windows = _sliding_window_chunks(
            input_ids=input_ids,
            chunk_size=chunk_size_tokens,
            chunk_overlap=chunk_overlap_tokens,
        )

        if not windows:
            continue

        # Construct chunks for this document
        for chunk_index, window in enumerate(windows):
            start_token = window["start_token"]
            end_token = window["end_token"]

            chunk_ids = input_ids[start_token:end_token]
            # Decode token IDs back to text
            chunk_text = tokenizer.decode(
                chunk_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            chunk_id = f"{split_name}_{example_id}_{chunk_index}"

            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "split": split_name,
                    "example_id": example_id,
                    "doc_title": doc_title,
                    "chunk_text": chunk_text,
                    "start_token": start_token,
                    "end_token": end_token,
                }
            )

    chunks_df = pd.DataFrame(all_chunks)
    logger.info(
        "Finished chunking. Generated %d chunks from %d input rows.",
        len(chunks_df),
        len(df),
    )

    return _save_parquet(chunks_df, out_path)
