import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import Config

class Retriever:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = config.embedding_model

    def _get_embedder(self, emb_model_name: str, device: str):
        return SentenceTransformer(emb_model_name, device=device)
        
    def retrieve(self, query: str, corpus, faiss_index, top_k: int = 5):
        emb_model = self._get_embedder(self.model_name, self.device)
        if emb_model is None:
            raise ValueError("Embedding model is not initialized.")
        
        query_embedding = emb_model.encode([query], convert_to_numpy=True)
        norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        norm[norm == 0] = 1e-10
        query_norm = (query_embedding / norm).astype(np.float32)
        scores, idx = faiss_index.search(query_norm, top_k)
        retrieved_passages_idx = idx[0]
        return [corpus[i] for i in idx[0]], scores[0], retrieved_passages_idx