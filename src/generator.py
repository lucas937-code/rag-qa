import os
import pickle
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==============================
# Config
# ==============================
EMBEDDINGS_FILE = Path("corpus_embeddings_unique.pkl")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TOP_K = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_GEN_TOKENS = 128
MAX_INPUT_LENGTH = 2048


# ==============================
# Load embeddings from file
# ==============================
def load_embeddings(file_path=EMBEDDINGS_FILE):
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


def get_embedder():
    global _embed_model
    if _embed_model is None:
        print("🔹 Loading embedding model...")
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)
    return _embed_model


def get_generator():
    global _gen_model, _tokenizer
    if _gen_model is None:
        print("🔹 Loading Llama-3.1-8B-Instruct...")
        _tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME, use_fast=True)
        _tokenizer.pad_token = _tokenizer.eos_token  # Llama has no pad token
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

    prompt = (
        "You are a factual assistant. Answer only with information from the context.\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\nAnswer:"
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
