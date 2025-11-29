import os
import pickle
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk, concatenate_datasets
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import Config, DEFAULT_CONFIG

# Optional FAISS import
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


# ======================================================
# Configuration
# ======================================================
SHARD_PREFIX = "shard_"
TOP_K_VALUES = [1, 3, 5, 7, 10]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEV_LIMIT = 1000   # number of samples used per split for evaluation


# ======================================================
# Load embeddings / index
# ======================================================
def load_embeddings(config: Config):
    embeddings_file = Path(config.EMBEDDINGS_FILE)
    faiss_index_file = Path(config.FAISS_INDEX_FILE)
    passages_file = Path(config.PASSAGES_FILE)
    if not FAISS_AVAILABLE:
        raise ImportError("faiss is required. Install faiss-cpu and build the index via compute_embeddings.")
    if not (faiss_index_file.exists() and passages_file.exists()):
        raise FileNotFoundError("FAISS index/passages missing. Run src.compute_embeddings to build them.")

    with open(passages_file, "rb") as f:
        passages = pickle.load(f)["passages"]
    index = faiss.read_index(str(faiss_index_file)) # type: ignore
    print(f"🔹 Loaded FAISS index with {len(passages)} passages")
    return passages, None, index


# ======================================================
# Load dataset shards
# ======================================================
def load_all_shards(base_dir):
    shards = sorted([
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if d.startswith(SHARD_PREFIX)
    ])

    if not shards:
        raise RuntimeError(f"⚠ No shards found in: {base_dir}")

    datasets = []
    for shard in tqdm(shards, desc=f"Loading {base_dir}"):
        datasets.append(load_from_disk(shard))

    dataset = concatenate_datasets(datasets)
    return dataset.select(range(min(DEV_LIMIT, len(dataset))))


# ======================================================
# Compute Recall@K
# ======================================================
def evaluate_recall(model, corpus, embeddings, dataset, top_k_values, faiss_index=None):
    recalls = {k: [] for k in top_k_values}

    for ex in tqdm(dataset, desc="Evaluating Recall"):
        question = ex["question"]
        aliases = ex["answer"]["normalized_aliases"]

        # Encode question
        q_emb = model.encode([question], convert_to_numpy=True, device=DEVICE)
        if faiss_index is None:
            raise RuntimeError("FAISS index not loaded.")
        norm = np.linalg.norm(q_emb, axis=1, keepdims=True)
        norm[norm == 0] = 1e-10
        q_norm = (q_emb / norm).astype(np.float32)
        _, idx = faiss_index.search(q_norm, max(top_k_values))
        sorted_idx = idx[0]

        # Evaluate Recall@K
        for k in top_k_values:
            retrieved = [corpus[i].lower() for i in sorted_idx[:k]]
            found = any(any(a in p for a in aliases) for p in retrieved)
            recalls[k].append(int(found))

    return {k: np.mean(v) for k, v in recalls.items()}


# ======================================================
# ORCHESTRATION FUNCTION
# ======================================================
def run_evaluation(config: Config = DEFAULT_CONFIG):
    DATA_DIRS = {
        "train": config.TRAIN_DIR,
        "validation": config.VAL_DIR,
        "test": config.TEST_DIR
    }
    corpus, emb, faiss_index = load_embeddings(config)
    model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

    for name, path in DATA_DIRS.items():
        print(f"\n=== 🔥 Evaluating {name.upper()} — first {DEV_LIMIT} samples ===")
        dataset = load_all_shards(path)

        results = evaluate_recall(model, corpus, emb, dataset, TOP_K_VALUES, faiss_index)
        for k, score in results.items():
            print(f"Recall@{k}: {score:.4f}")


if __name__ == "__main__":
    run_evaluation()
