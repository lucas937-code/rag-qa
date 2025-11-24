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
EMBEDDINGS_FILE = "corpus_embeddings.pkl"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-large"
TOP_K = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_GEN_TOKENS = 128
MAX_INPUT_LENGTH = 512

# ------------------------------
# Load corpus embeddings
# ------------------------------
with open(EMBEDDINGS_FILE, "rb") as f:
    data = pickle.load(f)
    corpus_embeddings = data["embeddings"]
    corpus = data["passages"]

print(f"Loaded {len(corpus)} passages and embeddings.")

# ------------------------------
# Load embedding model
# ------------------------------
embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)

# ------------------------------
# Load Flan-T5 generator
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME).to(DEVICE)
model.eval()

# ------------------------------
# Retrieve top-k passages
# ------------------------------
def retrieve(query, top_k=TOP_K):
    query_emb = embed_model.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(query_emb, corpus_embeddings)
    top_idx = np.argsort(-sims[0])[:top_k]
    return [corpus[i] for i in top_idx]

# ------------------------------
# Generate single answer
# ------------------------------
def generate_answer_single(query, context_passages):
    """
    Generate an answer using the top retrieved chunks.
    Returns only the first non-empty answer.
    """
    for chunk in context_passages[:TOP_K]:
        prompt = f"Answer the question based on the context below:\nContext: {chunk}\nQuestion: {query}\nAnswer:"
        # print(prompt)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_LENGTH
        ).to(DEVICE)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_GEN_TOKENS
            )

        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        if answer:
            return answer

    return "No answer found."

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    query = "The medical condition glaucoma affects which part of the body?"
    top_passages = retrieve(query)

    print("Retrieved top passages:")
    for i, p in enumerate(top_passages):
        print(f"{i+1}. {p[:200].replace(chr(10), ' ')}...")

    answer = generate_answer_single(query, top_passages)
    print("\nGenerated Answer:")
    print(answer)
