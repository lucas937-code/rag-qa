from pathlib import Path
import os
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import torch
import pandas as pd
import json

MODEL_NAME = "all-MiniLM-L6-v2"  # or "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 512
EMBEDDINGS_FILE = "corpus_embeddings_unique.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_chunks_from_parquet(chunks_path, title_column="doc_title", text_column="chunk_text"):
    df = pd.read_parquet(chunks_path)
    passages = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading chunks from parquet"):
        text = row.get(text_column, "")
        if not isinstance(text, str) or not text.strip():
            continue

        raw_title = row.get(title_column)
        title = "No Title"

        if isinstance(raw_title, str):
            try:
                titles = json.loads(raw_title)
                if isinstance(titles, list) and titles:
                    title = titles[0]
                else:
                    title = raw_title
            except json.JSONDecodeError:
                title = raw_title

        passages.append(f"{title}: {text}")

    return passages


def compute_and_save_embeddings(
    base_dir,
    chunks_parquet_path,
    embeddings_file=EMBEDDINGS_FILE,
    batch_size=BATCH_SIZE,
    model_name=MODEL_NAME,
    device=DEVICE,
):
    """
    Compute embeddings from precomputed Parquet chunks.
    """
    # Normalize paths
    base_dir = Path(base_dir)
    chunks_parquet_path = Path(chunks_parquet_path)

    if not chunks_parquet_path.exists():
        raise FileNotFoundError(f"Chunks parquet not found: {chunks_parquet_path}")

    embeddings_path = base_dir / embeddings_file
    base_dir.mkdir(parents=True, exist_ok=True)

    # If embeddings already exist, load and return
    if embeddings_path.exists():
        print(f"Loading existing embeddings from {embeddings_path}")
        with open(embeddings_path, "rb") as f:
            data = pickle.load(f)
        return data["passages"], data["embeddings"]
    
    print("Using device:", DEVICE)

    print("Embeddings not found, computing embeddings...")

    # Load chunked corpus
    corpus = load_chunks_from_parquet(chunks_parquet_path)
    print(f"Loaded {len(corpus)} chunks")

    # Remove duplicates
    before = len(corpus)
    corpus = list(dict.fromkeys(corpus))
    after = len(corpus)
    print(f"Removed {before - after} duplicates → {after} unique passages.")

    # Load model
    model = SentenceTransformer(model_name, device=device)

    # Embed in batches
    corpus_embeddings = []
    for i in tqdm(range(0, len(corpus), batch_size), desc="Embedding"):
        batch = corpus[i:i+batch_size]
        emb = model.encode(batch, convert_to_numpy=True)
        corpus_embeddings.append(emb)

    corpus_embeddings = np.vstack(corpus_embeddings)

    # Save
    with open(embeddings_path, "wb") as f:
        pickle.dump({"passages": corpus, "embeddings": corpus_embeddings}, f)

    print(f"Saved embeddings to {embeddings_path}")
    return corpus, corpus_embeddings


