from sentence_transformers import SentenceTransformer, CrossEncoder
import torch

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