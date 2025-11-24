# rag_lama_base_reduced_topk.py
import os
import pickle
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------------------
# Config
# ------------------------------
EMBEDDINGS_FILE = "corpus_embeddings.pkl"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # LLaMA-2 7B HF model
TOP_K = 3  # Reduced number of passages for memory efficiency
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_GEN_TOKENS = 128
MAX_PASSAGES_PER_CHUNK = 2  # Process 1–2 passages at a time to save memory
MAX_INPUT_LENGTH = 512       # Truncate input sequences

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
# Load LLaMA generator
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    GEN_MODEL_NAME,
    device_map="auto",  # Automatically place layers on available GPUs
    torch_dtype=torch.float16
)
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
# Generate answer in chunks
# ------------------------------
def generate_answer_chunked(query, context_passages):
    chunked_answers = []

    for i in range(0, len(context_passages), MAX_PASSAGES_PER_CHUNK):
        chunk = context_passages[i:i + MAX_PASSAGES_PER_CHUNK]
        context = "\n\n".join(chunk)
        prompt = f"Answer the question based on the context below:\nContext: {context}\nQuestion: {query}\nAnswer:"

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_LENGTH
        ).to(DEVICE)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_GEN_TOKENS,
                do_sample=True,        # Optional: for more natural output
                temperature=0.7
            )

        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        chunked_answers.append(answer.strip())

    final_answer = " ".join(chunked_answers)
    return final_answer

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    query = "In which US state is Lower Lake?"
    top_passages = retrieve(query)

    print("Retrieved top passages:")
    for i, p in enumerate(top_passages):
        print(f"{i+1}. {p[:400].replace(chr(10), ' ')}...")

    answer = generate_answer_chunked(query, top_passages)
    print("\nGenerated Answer:")
    print(answer)
