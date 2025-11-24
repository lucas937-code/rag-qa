# embeddings_with_progress_gpu.py
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
# Helper: Extract all passages
# ------------------------------
def extract_passages(dataset):
    passages = []
    for example in tqdm(dataset, desc="Extracting passages"):
        passages.extend(example['entity_pages']['wiki_context'])
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

        # 2. Extract passages
        corpus = extract_passages(dataset)
        print(f"Total passages to embed: {len(corpus)}")

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
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    from sklearn.metrics.pairwise import cosine_similarity
    # query = "The medical condition glaucoma affects which part of the body?"
    query = "What is the capital of france?"
    query_embedding = model.encode([query], convert_to_numpy=True, device=DEVICE)
    sims = cosine_similarity(query_embedding, corpus_embeddings)
    top_idx = np.argsort(-sims[0])[:5]
    print("\nTop 5 passages:")
    for i in top_idx:
        print("-", corpus[i][:].replace("\n", " "))
        print("---")
        

if __name__ == "__main__":
    main()
