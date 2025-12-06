from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_embed_model = None
_embed_model_name = None
_reranker = None

def get_embedder(model_name):
    global _embed_model, _embed_model_name
    if _embed_model is None or _embed_model_name != model_name:
        print(f"🔹 Loading embedding model {model_name}...")
        _embed_model = SentenceTransformer(model_name, device=DEVICE)
        _embed_model_name = model_name
    return _embed_model

def get_reranker(model_name: str):
    global _reranker
    if _reranker is None:
        print("🔹 Loading cross-encoder reranker...")
        _reranker = CrossEncoder(model_name, device=DEVICE)
    return _reranker

# ==============================
# Retrieve passages
# ==============================
def retrieve_top_k(query, corpus, embeddings, emb_model_name, reranker_model_name, candidates=100, k=3):
    if embeddings is None:
        raise RuntimeError("FAISS index not loaded. Call load_embeddings() first.")

    embed_model = get_embedder(emb_model_name)
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    norm = np.linalg.norm(q_emb, axis=1, keepdims=True)
    norm[norm == 0] = 1e-10
    q_norm = (q_emb / norm).astype(np.float32)
    scores, idx = embeddings.search(q_norm, candidates)
    candidates_idx = idx[0]

    reranker = get_reranker(reranker_model_name)
    pairs = [(query, corpus[i]) for i in candidates_idx]
    rerank_scores = reranker.predict(pairs)
    order = np.argsort(-rerank_scores)
    top_idx = candidates_idx[order][:k]
    return [corpus[i] for i in top_idx], rerank_scores[order][:k]