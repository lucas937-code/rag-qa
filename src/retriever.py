from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import numpy as np

class Retriever:
    def __init__(self) -> None:
        self.embed_model = None
        self.embed_model_name = None
        self.reranker = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _get_embedder(self, model_name):
        if self.embed_model is None or self.embed_model_name != model_name:
            print(f"🔹 Loading embedding model {model_name}...")
            self.embed_model = SentenceTransformer(model_name, device=self.device)
            self.embed_model_name = model_name
        return self.embed_model
    
    def _get_reranker(self, model_name: str):
        if self.reranker is None:
            print("🔹 Loading cross-encoder reranker...")
            self.reranker = CrossEncoder(model_name, device=self.device)
        return self.reranker

    def retrieve_top_k(self, query, corpus, embeddings, emb_model_name, reranker_model_name, candidates=100, k=3):
        if embeddings is None:
            raise RuntimeError("FAISS index not loaded. Call load_embeddings() first.")

        embed_model = self._get_embedder(emb_model_name)
        q_emb = embed_model.encode([query], convert_to_numpy=True)
        norm = np.linalg.norm(q_emb, axis=1, keepdims=True)
        norm[norm == 0] = 1e-10
        q_norm = (q_emb / norm).astype(np.float32)
        scores, idx = embeddings.search(q_norm, candidates)
        candidates_idx = idx[0]

        reranker = self._get_reranker(reranker_model_name)
        pairs = [(query, corpus[i]) for i in candidates_idx]
        rerank_scores = reranker.predict(pairs)
        order = np.argsort(-rerank_scores)
        top_idx = candidates_idx[order][:k]
        return [corpus[i] for i in top_idx], rerank_scores[order][:k]
