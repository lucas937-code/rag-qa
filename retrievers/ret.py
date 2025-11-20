import os
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss

# ---- File paths ----
CORPUS_FILE = "corpus.npy"
EMBEDDINGS_FILE = "corpus_embeddings.npy"

# ---- Load TriviaQA rc.wikipedia ----
trivia_qa = load_dataset("trivia_qa", name="rc.wikipedia")
train_dataset = trivia_qa["train"]

# ---- Flatten corpus into strings ----
if os.path.exists(CORPUS_FILE) and os.path.exists(EMBEDDINGS_FILE):
    corpus = np.load(CORPUS_FILE, allow_pickle=True)
    corpus_embeddings = np.load(EMBEDDINGS_FILE)
    print(f"Loaded corpus ({len(corpus)}) and embeddings from disk.")
else:
    corpus = []
    for ex in train_dataset:
        for doc in ex["search_results"]:
            # Ensure we always store a string
            if isinstance(doc, dict) and "search_context" in doc:
                corpus.append(str(doc["search_context"]))
            else:
                corpus.append(str(doc))

    # ---- Compute embeddings ----
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    corpus_embeddings = model.encode(corpus, convert_to_numpy=True, show_progress_bar=True, batch_size=64)
    corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

    # ---- Save corpus and embeddings ----
    np.save(CORPUS_FILE, np.array(corpus, dtype=object))
    np.save(EMBEDDINGS_FILE, corpus_embeddings)
    print(f"Saved corpus ({len(corpus)}) and embeddings to disk.")

# ---- Build FAISS index ----
dim = corpus_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(corpus_embeddings)
print(f"FAISS index built with {index.ntotal} vectors.")

# ---- Initialize model for queries ----
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ---- Retrieval function ----
def retrieve(question, top_k=5):
    q_emb = model.encode(question, convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb)
    scores, doc_ids = index.search(np.array([q_emb]), top_k)
    retrieved_docs = [corpus[i] for i in doc_ids[0]]
    return list(zip(retrieved_docs, scores[0]))

# ---- Example usage ----
if __name__ == "__main__":
    question = "Who was the first emperor of Rome?"
    results = retrieve(question, top_k=3)

    for i, (doc, score) in enumerate(results):
        print(f"\n----- Document {i+1} (score={score:.4f}) -----\n")
        print(doc[:500], "...\n")  # print first 500 chars
