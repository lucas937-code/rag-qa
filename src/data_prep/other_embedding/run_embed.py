from pathlib import Path
import sys

# Determine repository root relative to this file and add it to sys.path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

# Import after ensuring repo root is on sys.path
from data_prep.other_embedding.embed_chunks import embed_chunks

# Default directories (change if your layout differs)
CHUNK_DIR = repo_root / "data" / "processed" / "chunks"
EMB_DIR = repo_root / "data" / "processed" / "embeddings"

# Make sure the output directory exists
EMB_DIR.mkdir(parents=True, exist_ok=True)

# Input: chunked train data
train_chunks_path = CHUNK_DIR / "train_chunks.parquet"

# Output: embeddings + metadata
train_embeddings_path = EMB_DIR / "train_embeddings.npy"
train_meta_path = EMB_DIR / "train_embeddings_meta.parquet"

print("Embedding train chunks if file exists:", train_chunks_path)
if train_chunks_path.exists():
    train_embeddings_path, train_meta_path = embed_chunks(
        chunks_path=train_chunks_path,
        embeddings_out_path=train_embeddings_path,
        meta_out_path=train_meta_path,
        # Model choice: good default for QA retrieval
        model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        batch_size=128,
        normalize_embeddings=True,
        use_doc_title=True,
        max_chunks=None,
    )

    print("Train embeddings saved to:", train_embeddings_path)
    print("Train metadata saved to:", train_meta_path)
else:
    print("Train chunks file not found, skipping train embeddings.")

# Input: chunked validation data
val_chunks_path = CHUNK_DIR / "val_chunks.parquet"

# Output: embeddings + metadata
val_embeddings_path = EMB_DIR / "val_7900_embeddings.npy"
val_meta_path = EMB_DIR / "val_7900_embeddings_meta.parquet"

print("Embedding validation chunks if file exists:", val_chunks_path)
if val_chunks_path.exists():
    val_embeddings_path, val_meta_path = embed_chunks(
        chunks_path=val_chunks_path,
        embeddings_out_path=val_embeddings_path,
        meta_out_path=val_meta_path,
        model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        batch_size=128,
        normalize_embeddings=True,
        use_doc_title=True,
        max_chunks=None,
    )

    print("Val embeddings saved to:", val_embeddings_path)
    print("Val metadata saved to:", val_meta_path)
else:
    print("Val chunks file not found, skipping val embeddings.")
