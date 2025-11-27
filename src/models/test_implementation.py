import os
import pickle
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch

# ============================================
# CONFIG
# ============================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

EMBEDDINGS_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "embeddings",
    "corpus_embeddings_unique.pkl"
)

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================
# LOAD EMBEDDINGS
# ============================================
def load_corpus_embeddings():
    with open(EMBEDDINGS_FILE, "rb") as f:
        data = pickle.load(f)
        return data["passages"], data["embeddings"]

# ============================================
# RETRIEVAL
# ============================================
def retrieve(query, corpus, corpus_embeddings, embed_model, top_k=TOP_K, original_indices=None):
    """
    query: str
    corpus: list of passages (subset)
    corpus_embeddings: embeddings for the subset (N x d)
    original_indices: np.array of original indices of each passage in the full corpus
    """
    query_emb = embed_model.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(query_emb, corpus_embeddings)
    top_idx = np.argsort(-sims[0])[:top_k]

    results = []
    for idx in top_idx:
        # Map back to original corpus index if provided
        orig_idx = int(original_indices[idx]) if original_indices is not None else int(idx)
        results.append({
            "rank": len(results) + 1,
            "subset_index": int(idx),
            "original_index": orig_idx,
            "passage": corpus[idx],
            "score": float(sims[0][idx])
        })

    return results

# ============================================
# MANUAL TEST
# ============================================
if __name__ == "__main__":
    print("\nLoading corpus embeddings from:")
    print(EMBEDDINGS_FILE)

    corpus, corpus_embeddings = load_corpus_embeddings()

    # --- RANDOM SAMPLE OF 10 PASSAGES FROM FULL CORPUS ---
    num_samples = 10
    idxs = random.sample(range(len(corpus)), num_samples)
    subset_indices = np.array(idxs)

    corpus_subset = [corpus[i] for i in subset_indices]
    corpus_embeddings_subset = corpus_embeddings[subset_indices]

    print(f"\nUsing a random subset of {num_samples} passages.")
    print(f"Original indices selected: {list(subset_indices)}\n")

    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)

    query = "Who discovered penicillin?"
    print(f"\nQuery: {query}\n")

    results = retrieve(
        query,
        corpus_subset,
        corpus_embeddings_subset,
        embed_model,
        top_k=TOP_K,
        original_indices=subset_indices
    )

    print("\n===== RETRIEVAL RESULTS =====\n")
    for r in results:
        print(
            f"[Rank {r['rank']}]  "
            f"Score: {r['score']:.4f}  "
            f"(Subset idx {r['subset_index']} | Original idx {r['original_index']})"
        )
        print(r["passage"])
        print("-" * 80)
