import os
import pickle
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional FAISS import
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# ==============================
# Config
# ==============================
EMBEDDINGS_FILE = Path("corpus_embeddings_unique.pkl")
FAISS_INDEX_FILE = Path("corpus_faiss.index")
PASSAGES_FILE = Path("corpus_passages.pkl")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
TOP_K = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_GEN_TOKENS = 128
MAX_INPUT_LENGTH = 2048


# ==============================
# Load embeddings / index
# ==============================
def load_embeddings(file_path=EMBEDDINGS_FILE):
    """
    Prefer FAISS index + passages if available; otherwise fallback to pickle embeddings.
    """
    global _faiss_index, _faiss_passages

    if FAISS_AVAILABLE and FAISS_INDEX_FILE.exists() and PASSAGES_FILE.exists():
        with open(PASSAGES_FILE, "rb") as f:
            data = pickle.load(f)
            passages = data["passages"]
        _faiss_passages = passages
        _faiss_index = faiss.read_index(str(FAISS_INDEX_FILE))
        print(f"🔹 Loaded FAISS index with {len(passages)} passages")
        return passages, None

    # Fallback to dense embeddings pickle
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Embedding file not found → {file_path}")

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    print(f"🔹 Loaded {len(data['passages'])} passages from {file_path}")
    return data["passages"], data["embeddings"]


# ==============================
# Init models (lazy-loaded)
# ==============================
_embed_model = None
_gen_model = None
_tokenizer = None
_faiss_index = None
_faiss_passages = None


def get_embedder():
    global _embed_model
    if _embed_model is None:
        print("🔹 Loading embedding model...")
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)
    return _embed_model


def get_generator():
    global _gen_model, _tokenizer
    if _gen_model is None:
        print("🔹 Loading Mistral-7B-Instruct...")
        _tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME, use_fast=True)
        # Mistral has no pad token
        if _tokenizer.pad_token_id is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        _gen_model = AutoModelForCausalLM.from_pretrained(
            GEN_MODEL_NAME,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None,
        )
        _gen_model.eval()
    return _tokenizer, _gen_model


# ==============================
# Retrieve passages
# ==============================
def retrieve_top_k(query, corpus, embeddings, k=TOP_K):
    # Use FAISS if available and loaded
    if FAISS_AVAILABLE and _faiss_index is not None:
        embed_model = get_embedder()
        q_emb = embed_model.encode([query], convert_to_numpy=True)
        norm = np.linalg.norm(q_emb, axis=1, keepdims=True)
        norm[norm == 0] = 1e-10
        q_norm = (q_emb / norm).astype(np.float32)
        scores, idx = _faiss_index.search(q_norm, k)
        return [corpus[i] for i in idx[0]], scores[0]

    embed_model = get_embedder()
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, embeddings)[0]

    idx = np.argsort(-sims)[:k]
    return [corpus[i] for i in idx], sims[idx]


# ==============================
# Generate answer from combined top-K context
# ==============================
def generate_answer_combined(query, corpus, embeddings, top_k=5):
    tokenizer, model = get_generator()
    if tokenizer is None or model is None:
        raise RuntimeError("Generator model or tokenizer not loaded.")
    top_passages, _ = retrieve_top_k(query, corpus, embeddings, k=top_k)
    context_block = "\n---\n".join(top_passages)

    # Use instruct/chat template
    messages = [
        {"role": "system", "content": "You are a factual assistant. Answer only using the provided context."},
        {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}\nAnswer:"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
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
            do_sample=False,       # or True + temperature for diversity
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

    passages, _ = retrieve_top_k(q, corpus, emb, k=3)
    print("\nRetrieved passages:")
    for i, p in enumerate(passages):
        print(f"{i+1}. {p[:200].replace(chr(10),' ')}...")

    print("\nGenerated Answer:")
    print(generate_answer_combined(q, corpus, emb, top_k=5)[0])
