"""
High-quality embedding pipeline for RAG chunk corpus.

This module takes the chunked dataset produced by `chunk_documents.py`
(e.g. train_chunks.parquet, val_chunks.parquet) and computes dense
embeddings using a SentenceTransformer model.

Outputs:
- <split>_embeddings.npy         : float32 array of shape [N_chunks, dim]
- <split>_embeddings_meta.parquet: metadata for each embedding row
    - embedding_idx (0..N-1)
    - chunk_id
    - example_id / question_id
    - split
    - doc_title
    - chunk_text
    - start_token
    - end_token
    - model_name
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Tuple
import logging

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import torch


logger = logging.getLogger(__name__)


# ============================
# Helper: device selection
# ============================

def _get_default_device() -> str:
    """Select GPU if available, otherwise CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # for Apple Silicon
        return "mps"
    return "cpu"


# ============================
# Core embedding function
# ============================

def embed_chunks(
    chunks_path: str | Path,
    embeddings_out_path: str | Path,
    meta_out_path: str | Path,
    model_name: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    batch_size: int = 128,
    normalize_embeddings: bool = True,
    use_doc_title: bool = True,
    max_chunks: Optional[int] = None,
    device: Optional[str] = None,
) -> Tuple[Path, Path]:
    """
    Compute dense embeddings for all chunks in a Parquet file.

    Parameters
    ----------
    chunks_path:
        Path to the chunks Parquet file (e.g. train_chunks.parquet).
    embeddings_out_path:
        Target .npy file for the embeddings array.
    meta_out_path:
        Target .parquet file for the embedding metadata.
    model_name:
        Hugging Face / SentenceTransformers model name.
        For higher quality retrieval, a QA-optimized model like
        "sentence-transformers/multi-qa-mpnet-base-dot-v1" is recommended.
    batch_size:
        Batch size for encoding.
    normalize_embeddings:
        If True, L2-normalize embeddings so that dot product ~= cosine similarity.
    use_doc_title:
        If True, prepend the document title (if available) to the chunk text
        to form the passage fed into the encoder.
    max_chunks:
        Optional limit on the number of chunks to embed (for debugging).

    Returns
    -------
    (embeddings_out_path, meta_out_path)
    """

    chunks_path = Path(chunks_path)
    embeddings_out_path = Path(embeddings_out_path)
    meta_out_path = Path(meta_out_path)

    logger.info("Loading chunks from %s", chunks_path)
    df = pd.read_parquet(chunks_path)
    df = df.reset_index(drop=True)

    if max_chunks is not None:
        df = df.iloc[:max_chunks].copy()
        logger.info("Limiting to first %d chunks for embedding.", max_chunks)

    if df.empty:
        raise ValueError(f"No chunks found in {chunks_path}")

    # Build passages: doc_title + chunk_text (optional)
    if "chunk_text" not in df.columns:
        raise KeyError(
            f"Expected column 'chunk_text' in chunks file, found: {list(df.columns)}"
        )

    passages: List[str] = []
    for _, row in df.iterrows():
        text = row.get("chunk_text", None)
        if not isinstance(text, str) or not text.strip():
            passages.append("")  # will be skipped later or encoded as empty
            continue

        title = None
        if use_doc_title and "doc_title" in df.columns:
            title = row.get("doc_title", None)

        if use_doc_title and isinstance(title, str) and title.strip():
            passage = f"{title.strip()}\n\n{text.strip()}"
        else:
            passage = text.strip()

        passages.append(passage)

    # Filter out completely empty passages (if any)
    idx_non_empty = [i for i, p in enumerate(passages) if p]
    if not idx_non_empty:
        raise ValueError("All passages are empty after preprocessing; nothing to embed.")

    df = df.iloc[idx_non_empty].reset_index(drop=True)
    passages = [passages[i] for i in idx_non_empty]

    logger.info(
        "Prepared %d non-empty passages for embedding (from %d chunks).",
        len(passages),
        len(idx_non_empty),
    )

    # Load model
    if device is None:
        device = _get_default_device()

    # Report device and CUDA info for the user
    try:
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False

    logger.info("Requested device: %s", device)
    logger.info("Torch reports CUDA available: %s", cuda_available)
    if cuda_available:
        try:
            dev_idx = torch.cuda.current_device()
            dev_name = torch.cuda.get_device_name(dev_idx)
            logger.info("Using CUDA device %s: %s", dev_idx, dev_name)
        except Exception:
            logger.info("CUDA available but failed to query device name.")

    logger.info("Loading SentenceTransformer model '%s' on device '%s'", model_name, device)
    model = SentenceTransformer(model_name, device=device)

    # Encode in batches with an explicit tqdm progress bar
    all_embeddings: List[np.ndarray] = []

    num_batches = (len(passages) + batch_size - 1) // batch_size
    logger.info("Starting encoding with batch_size=%d (%d batches)...", batch_size, num_batches)
    for start in tqdm(range(0, len(passages), batch_size), desc="Embedding chunks", total=num_batches, unit="batch"):
        end = min(start + batch_size, len(passages))
        batch = passages[start:end]

        emb = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        all_embeddings.append(emb)

    embeddings = np.vstack(all_embeddings).astype(np.float32)

    logger.info(
        "Finished encoding. Embeddings shape: %s (dtype=%s)",
        embeddings.shape,
        embeddings.dtype,
    )

    # Optional L2 normalization (for cosine similarity via dot product)
    if normalize_embeddings:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        embeddings = embeddings / norms
        logger.info("Applied L2 normalization to embeddings.")

    # Save embeddings array
    embeddings_out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_out_path, embeddings)
    logger.info("Saved embeddings to %s", embeddings_out_path)

    # Build and save metadata
    n = len(df)
    if embeddings.shape[0] != n:
        raise RuntimeError(
            f"Number of embeddings ({embeddings.shape[0]}) "
            f"does not match number of rows ({n})."
        )

    meta_df = pd.DataFrame({
        "embedding_idx": np.arange(n, dtype=np.int32),
    })

    # Copy useful metadata columns if available
    for col in ["chunk_id", "split", "example_id", "doc_title", "chunk_text", "start_token", "end_token"]:
        if col in df.columns:
            meta_df[col] = df[col].values

    # Add model + config info for reproducibility
    meta_df["model_name"] = model_name
    meta_df["normalized"] = normalize_embeddings

    meta_out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_df.to_parquet(meta_out_path)
    logger.info("Saved embedding metadata to %s", meta_out_path)

    return embeddings_out_path, meta_out_path


# ============================
# Optional: helper for queries
# ============================

def embed_queries(
    queries: List[str],
    model_name: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    batch_size: int = 32,
    normalize_embeddings: bool = True,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Compute embeddings for a list of query strings.

    This function uses the same SentenceTransformer model as the corpus
    embedding and applies the same normalization logic, so that you can
    directly compare query embeddings against chunk embeddings via
    dot product (cosine similarity).
    """
    if device is None:
        device = _get_default_device()

    model = SentenceTransformer(model_name, device=device)

    all_embeddings: List[np.ndarray] = []
    for start in range(0, len(queries), batch_size):
        end = min(start + batch_size, len(queries))
        batch = queries[start:end]
        emb = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        all_embeddings.append(emb)

    embeddings = np.vstack(all_embeddings).astype(np.float32)

    if normalize_embeddings:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        embeddings = embeddings / norms

    return embeddings
