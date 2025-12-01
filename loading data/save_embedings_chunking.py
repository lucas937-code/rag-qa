# embeddings_with_progress_gpu_chunked.py
import os
import pickle
from datasets import load_from_disk, concatenate_datasets
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import torch

# ------------------------------
# Config
# ------------------------------
DATA_DIR = "../train_dataset"
EMBEDDINGS_FILE = "corpus_embeddings.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"
SHARD_PREFIX = "shard_"
BATCH_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU if available

# Chunking parameters
CHUNK_SIZE = 200  # number of tokens/words per chunk
CHUNK_OVERLAP = 50  # number of tokens/words to overlap

# ------------------------------
# Helper: Load all shards
# ------------------------------
def load_all_shards(data_dir):
    shards = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if d.startswith(SHARD_PREFIX)]
    shards = sorted(shards)
    datasets = []
    for shard in tqdm(shards, desc="Loading shards"):
        datasets.append(load_from_disk(shard))
    return concatenate_datasets(datasets)

# ------------------------------
# Helper: Chunk passages with overlap
# ------------------------------
def chunk_passage(title, text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Splits a text into overlapping chunks. Each chunk includes the title at the start.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_text = " ".join(words[start:end])
        chunks.append(f"{title}: {chunk_text}")
        start += chunk_size - overlap
    return chunks

# ------------------------------
# Helper: Extract and chunk all passages
# ------------------------------
def extract_passages(dataset):
    passages = []
    for example in tqdm(dataset, desc="Extracting passages"):
        entity_pages = example.get("entity_pages", {})
        titles = entity_pages.get("title", [])
        wiki_contexts = entity_pages.get("wiki_context", [])

        # Make sure lists are the same length
        n_pages = min(len(titles), len(wiki_contexts))

        for i in range(n_pages):
            title = titles[i] or "No Title"
            text = wiki_contexts[i]
            passages.extend(chunk_passage(title, text))
    return passages

# ------------------------------
# Main
# ------------------------------
def main():
    if os.path.exists(EMBEDDINGS_FILE):
        print(f"Loading saved embeddings from {EMBEDDINGS_FILE}...")
        with open(EMBEDDINGS_FILE, "rb") as f:
            data = pickle.load(f)
            corpus_embeddings = data["embeddings"]
            corpus = data["passages"]
        print(f"Loaded {len(corpus)} passages.")
    else:
        print("Embeddings not found, computing embeddings...")

        # 1. Load dataset
        dataset = load_all_shards(DATA_DIR)
        print(f"Loaded dataset with {len(dataset)} examples.")
        # print(dataset[0])
        
        # DEBUG TODO REMOVE LATER!!!!
        # dataset = dataset.select(range(1000))  # limit to first 100 examples for testing

        # 2. Extract and chunk passages
        corpus = extract_passages(dataset)
        print(f"Total chunks to embed: {len(corpus)}")

        # 3. Load SentenceTransformer model on GPU
        model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        print(f"Using device: {DEVICE}")
        
        # 4. Compute embeddings in batches with GPU
        corpus_embeddings = []
        for i in tqdm(range(0, len(corpus), BATCH_SIZE), desc="Computing embeddings"):
            batch = corpus[i:i+BATCH_SIZE]
            emb = model.encode(batch, convert_to_numpy=True, device=DEVICE)
            corpus_embeddings.append(emb)
        corpus_embeddings = np.vstack(corpus_embeddings)

        # 5. Save embeddings
        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump({"passages": corpus, "embeddings": corpus_embeddings}, f)
        print(f"Saved embeddings to {EMBEDDINGS_FILE}")

    # ------------------------------
    # Optional: test retrieval
    # ------------------------------

    # we contain duplicates due to the nature of the dataset, many pages are repeated
    # only print the number of unique passages
    unique_passages = set(corpus)
    print(f"Number of unique passages: {len(unique_passages)}")
    # number of total passages
    print(f"Number of total passages: {len(corpus)}")
    # number of embeddings
    print(f"Number of embeddings: {corpus_embeddings.shape[0]}")
    
    # number of dimensions
    print(f"Embedding dimensions: {corpus_embeddings.shape[1]}")

    print(f"{corpus_embeddings[:5]}")  # print first 5 embeddings


    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    from sklearn.metrics.pairwise import cosine_similarity
    query = "What is the top prize at the Cannes Film Festival?"
    query_embedding = model.encode([query], convert_to_numpy=True, device=DEVICE)
    sims = cosine_similarity(query_embedding, corpus_embeddings)
    top_idx = np.argsort(-sims[0])[:3]
    print("\nTop k passages:")
    for i in top_idx:
        print("-", corpus[i].replace("\n", " "))
        print("---")
        
if __name__ == "__main__":
    main()
