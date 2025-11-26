import os
import pickle
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ------------------------------
# Config
# ------------------------------
EMBEDDINGS_FILE = "corpus_embeddings_unique.pkl"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-large"
TOP_K = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_GEN_TOKENS = 128   # generation only — no max input length

# ==============================
# Load embeddings + passages
# ==============================
def load_corpus_embeddings(pickle_path=EMBEDDINGS_FILE):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
        return data["passages"], data["embeddings"]

# ==============================
# Load models
# ==============================
def load_models():
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME).to(DEVICE)
    gen_model.eval()

    return embed_model, tokenizer, gen_model

# ==============================
# Retrieve relevant passages
# ==============================
def retrieve(query, corpus, corpus_embeddings, embed_model, top_k=TOP_K):
    query_emb = embed_model.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(query_emb, corpus_embeddings)
    top_idx = np.argsort(-sims[0])[:top_k]
    return [corpus[i] for i in top_idx]

# ==============================
# Generate using top passage(s)
# — NO MAX INPUT LENGTH USED —
# ==============================
def generate_answer_single(query, passages, tokenizer, gen_model):
    for chunk in passages:
        prompt = (
            f"Answer the question using the context.\n"
            f"Context:\n{chunk}\n\n"
            f"Question: {query}\nAnswer:"
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(DEVICE)

        with torch.no_grad():
            output_ids = gen_model.generate(
                **inputs,
                max_new_tokens=MAX_GEN_TOKENS
            )

        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        if len(answer) > 0:
            return answer

    return "No useful answer found."

# ==============================
# Exported function for notebook
# ==============================
def ask(question, top_k=TOP_K):
    corpus, corpus_embeddings = load_corpus_embeddings()
    embed_model, tokenizer, gen_model = load_models()

    top_passages = retrieve(question, corpus, corpus_embeddings, embed_model, top_k)
    answer = generate_answer_single(question, top_passages, tokenizer, gen_model)

    return answer, top_passages
