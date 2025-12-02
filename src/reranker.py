import torch
import numpy as np
from sentence_transformers import CrossEncoder
from src.config import Config

class Reranker:
    def __init__(self, 
                 config: Config,
                 faiss_candidates: int = 100) -> None:
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.faiss_candidates = faiss_candidates
        # Load reranking model
        self.reranker = self._get_reranker(config.rerank_model, self.device)

    # ==============================
    # Local helper functions
    # ==============================
    def _get_reranker(self, model_name: str, device: str):
        return CrossEncoder(model_name, device=device)
    
    def rerank_and_get_top_k(self, query: str, corpus: list, candidates_idx: list, top_k: int = 5):
        if self.reranker is None:
            raise ValueError("Reranking model is not initialized.")
        
        # Prepare inputs for reranking
        pairs = [[query, corpus[i]] for i in candidates_idx]
        # Get reranking scores
        scores = self.reranker.predict(pairs)
        order = np.argsort(-scores)
        top_idx = candidates_idx[order][:top_k]
        return [corpus[i] for i in top_idx], scores[order][:top_k]