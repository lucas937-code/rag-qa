import os
import pickle
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk, concatenate_datasets
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ======================================================
# Configuration
# ======================================================
DATA_DIRS = {
    "train": "data/train",
    "validation": "data/validation",
    "test": "data/test"
}

EMBEDDINGS_FILE = Path("corpus_embeddings_unique.pkl")
SHARD_PREFIX = "shard_"
TOP_K_VALUES = [1, 3, 5, 7, 10]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEV_LIMIT = 1000   # number of samples used per split for evaluation


# ======================================================
# Load embeddings
# ======================================================
def load_embeddings():
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(f"❌ Cannot find embeddings → {EMBEDDINGS_FILE}")

    with open(EMBEDDINGS_FILE, "rb") as f:
        data = pickle.load(f)

    corpus = data["passages"]
    emb = data["embeddings"]

    print(f"🔹 Loaded {len(corpus)} passages with embeddings shape {emb.shape}")
    return corpus, emb


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
def evaluate_recall(model, corpus, embeddings, dataset, top_k_values):
    recalls = {k: [] for k in top_k_values}

    for ex in tqdm(dataset, desc="Evaluating Recall"):
        question = ex["question"]
        aliases = ex["answer"]["normalized_aliases"]

        # Encode question
        q_emb = model.encode([question], convert_to_numpy=True, device=DEVICE)
        sims = cosine_similarity(q_emb, embeddings)[0]
        sorted_idx = np.argsort(-sims)

        # Evaluate Recall@K
        for k in top_k_values:
            retrieved = [corpus[i].lower() for i in sorted_idx[:k]]
            found = any(any(a in p for a in aliases) for p in retrieved)
            recalls[k].append(int(found))

    return {k: np.mean(v) for k, v in recalls.items()}


# ======================================================
# ORCHESTRATION FUNCTION
# ======================================================
def run_evaluation():
    corpus, emb = load_embeddings()
    model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

    for name, path in DATA_DIRS.items():
        print(f"\n=== 🔥 Evaluating {name.upper()} — first {DEV_LIMIT} samples ===")
        dataset = load_all_shards(path)

        results = evaluate_recall(model, corpus, emb, dataset, TOP_K_VALUES)
        for k, score in results.items():
            print(f"Recall@{k}: {score:.4f}")


if __name__ == "__main__":
    run_evaluation()
