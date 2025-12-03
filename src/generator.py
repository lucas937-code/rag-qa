import os
import pickle
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.config import Config, DEFAULT_CONFIG
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoConfig,
)
# Optional FAISS import
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# ==============================
# Config
# ==============================
GEN_MODEL_NAME = "google/flan-t5-base"
TOP_K = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_GEN_TOKENS = 48
MAX_INPUT_LENGTH = 2048
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FAISS_CANDIDATES = 50


# ==============================
# Load embeddings / index
# ==============================
def load_embeddings(config: Config = DEFAULT_CONFIG):
    """
    Load FAISS index + passages. Fail fast if artifacts are missing.
    """
    global _faiss_index, _faiss_passages
    FAISS_INDEX_FILE = Path(config.FAISS_INDEX_FILE)
    PASSAGES_FILE = Path(config.PASSAGES_FILE)

    if not FAISS_AVAILABLE:
        raise ImportError("faiss is required. Install faiss-cpu and build the index via compute_embeddings.")

    if not (FAISS_INDEX_FILE.exists() and PASSAGES_FILE.exists()):
        raise FileNotFoundError("FAISS index/passages missing. Run src.compute_embeddings to build them.")

    with open(PASSAGES_FILE, "rb") as f:
        data = pickle.load(f)
        passages = data["passages"]
    _faiss_passages = passages
    _faiss_index = faiss.read_index(str(FAISS_INDEX_FILE)) # type: ignore
    print(f"🔹 Loaded FAISS index with {len(passages)} passages")
    return passages, None


# ==============================
# Init models (lazy-loaded)
# ==============================
_embed_model = None
_embed_model_name = None
_gen_model = None
_tokenizer = None
_faiss_index = None
_faiss_passages = None
_reranker = None


def get_embedder(model_name):
    global _embed_model, _embed_model_name
    if _embed_model is None or _embed_model_name != model_name:
        print(f"🔹 Loading embedding model {model_name}...")
        _embed_model = SentenceTransformer(model_name, device=DEVICE)
        _embed_model_name = model_name
    return _embed_model


def get_generator():
    global _gen_model, _tokenizer

    if _gen_model is None:
        print(f"🔹 Loading generator: {GEN_MODEL_NAME}")

        # Load config and tokenizer
        cfg = AutoConfig.from_pretrained(GEN_MODEL_NAME)
        _tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME, use_fast=True)

        # Proper detection: encoder-decoder (T5/BART/etc.) vs decoder-only (GPT/etc.)
        is_encoder_decoder = getattr(cfg, "is_encoder_decoder", False)

        if is_encoder_decoder:
            # Flan-T5, T5, BART, etc. → Seq2Seq
            print("➡ Detected encoder-decoder (Seq2Seq) model.")
            _gen_model = AutoModelForSeq2SeqLM.from_pretrained(
                GEN_MODEL_NAME,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            ).to(DEVICE)
        else:
            # GPT-style models → CausalLM
            print("➡ Detected decoder-only (CausalLM) model.")
            if _tokenizer.pad_token_id is None:
                _tokenizer.pad_token = _tokenizer.eos_token
            _gen_model = AutoModelForCausalLM.from_pretrained(
                GEN_MODEL_NAME,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                device_map="auto" if DEVICE == "cuda" else None,
            )

        _gen_model.eval()

    return _tokenizer, _gen_model



def get_reranker():
    global _reranker
    if _reranker is None:
        print("🔹 Loading cross-encoder reranker...")
        _reranker = CrossEncoder(RERANK_MODEL_NAME, device=DEVICE)
    return _reranker


# ==============================
# Retrieve passages
# ==============================
def retrieve_top_k(query, corpus, embeddings, model_name, k=TOP_K):
    if _faiss_index is None:
        raise RuntimeError("FAISS index not loaded. Call load_embeddings() first.")

    embed_model = get_embedder(model_name)
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    norm = np.linalg.norm(q_emb, axis=1, keepdims=True)
    norm[norm == 0] = 1e-10
    q_norm = (q_emb / norm).astype(np.float32)
    scores, idx = _faiss_index.search(q_norm, FAISS_CANDIDATES)
    candidates_idx = idx[0]

    reranker = get_reranker()
    pairs = [(query, corpus[i]) for i in candidates_idx]
    rerank_scores = reranker.predict(pairs)
    order = np.argsort(-rerank_scores)
    top_idx = candidates_idx[order][:k]
    return [corpus[i] for i in top_idx], rerank_scores[order][:k]


# ==============================
# Generate answer from combined top-K context
# ==============================
def generate_answer_combined(query, corpus, embeddings, top_k=5, config: Config = DEFAULT_CONFIG):
    tokenizer, model = get_generator()
    if tokenizer is None or model is None:
        raise RuntimeError("Generator model or tokenizer not loaded.")

    top_passages, _ = retrieve_top_k(query, corpus, embeddings, config.EMBEDDING_MODEL, k=top_k)
    context_block = "\n---\n".join(top_passages)

    # Simple text prompt for Flan-T5 (no chat template)
    prompt = (
        "You are a concise QA assistant. Answer ONLY from the provided context. "
        "If you are unsure, reply \"I don't know\". Respond in <=20 words.\n\n"
        f"Context:\n{context_block}\n\nQuestion: {query}\nAnswer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_GEN_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    answer = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    return answer, top_passages



# ==============================
# Manual test (cmd terminal)
# ==============================
if __name__ == "__main__":
    corpus, emb = load_embeddings()
    q = "The medical condition glaucoma affects which part of the body?"

    passages, _ = retrieve_top_k(q, corpus, emb, DEFAULT_CONFIG.EMBEDDING_MODEL, k=3)
    print("\nRetrieved passages:")
    for i, p in enumerate(passages):
        print(f"{i+1}. {p[:200].replace(chr(10),' ')}...")

    print("\nGenerated Answer:")
    print(generate_answer_combined(q, corpus, emb, top_k=5)[0])
