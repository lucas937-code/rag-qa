import pickle
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.config import Config, DEFAULT_CONFIG
from src.load_data import load_all_shards

# Optional FAISS import
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================
# Load embeddings / index
# ======================================================
def load_embeddings(config: Config):
    embeddings_file = Path(config.embeddings_file)
    faiss_index_file = Path(config.faiss_index_file)
    passages_file = Path(config.passages_file)
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
# Compute Recall@K
# ======================================================
def evaluate_recall(model, corpus, embeddings, dataset, candidates, top_k, faiss_index=None, reranker=None):
    recalls = {k: [] for k in top_k}
    top_candidates = max(candidates, max(top_k))

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
        _, idx = faiss_index.search(q_norm, top_candidates)
        candidates_idx = idx[0]

        if reranker is not None:
            pairs = [(question, corpus[i]) for i in candidates_idx]
            scores = reranker.predict(pairs)
            order = np.argsort(-scores)
            sorted_idx = candidates_idx[order]
        else:
            sorted_idx = candidates_idx

        # Evaluate Recall@K
        for k in top_k:
            retrieved = [corpus[i].lower() for i in sorted_idx[:k]]
            found = any(any(a in p for a in aliases) for p in retrieved)
            recalls[k].append(int(found))

    return {k: np.mean(v) for k, v in recalls.items()}


# ======================================================
# ORCHESTRATION FUNCTION
# ======================================================
def run_evaluation(config: Config,
                   sample_limit=100,
                   candidates=100,
                   top_k=(1,3,5,10),
                   data_dirs=None):
    data_dirs = (config.train_dir, config.val_dir, config.test_dir) if data_dirs is None else data_dirs
    corpus, emb, faiss_index = load_embeddings(config)
    model = SentenceTransformer(config.embedding_model, device=DEVICE)
    reranker = CrossEncoder(config.rerank_model, device=DEVICE)

    for path in data_dirs:
        print(f"\n=== 🔥 Evaluating {path} — first {sample_limit} samples ===")
        dataset = load_all_shards(path, config.shard_prefix, sample_limit)

        results = evaluate_recall(model, corpus, emb, dataset, candidates, top_k, faiss_index, reranker)
        for k, score in results.items():
            print(f"Recall@{k}: {score:.4f}")


if __name__ == "__main__":
    run_evaluation(DEFAULT_CONFIG)
