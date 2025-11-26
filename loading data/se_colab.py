import os
import pickle
from datasets import load_from_disk, concatenate_datasets
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import torch

# ------------------------------
# Default config
# ------------------------------
SHARD_PREFIX = "shard_"
BATCH_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 250
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_FILE = "corpus_embeddings_unique.pkl"

# ------------------------------
# Helper functions
# ------------------------------

def load_all_shards(data_dir):
    shards = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if d.startswith(SHARD_PREFIX)]
    shards = sorted(shards)
    datasets = []
    for shard in tqdm(shards, desc="Loading shards"):
        datasets.append(load_from_disk(shard))
    return concatenate_datasets(datasets)

def chunk_passage(title, text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_text = " ".join(words[start:end])
        chunks.append(f"{title}: {chunk_text}")
        start += chunk_size - overlap
    return chunks

def extract_passages(dataset):
    passages = []
    for example in tqdm(dataset, desc="Extracting passages"):
        entity_pages = example.get("entity_pages", {})
        titles = entity_pages.get("title", [])
        wiki_contexts = entity_pages.get("wiki_context", [])
        n_pages = min(len(titles), len(wiki_contexts))
        for i in range(n_pages):
            title = titles[i] or "No Title"
            text = wiki_contexts[i]
            passages.extend(chunk_passage(title, text))
    return passages

# ------------------------------
# Main function to export
# ------------------------------
def compute_and_save_embeddings(base_dir,
                                embeddings_file=EMBEDDINGS_FILE,
                                batch_size=BATCH_SIZE,
                                model_name=MODEL_NAME,
                                device=DEVICE):
    """
    Loads shards from train_dataset in base_dir, computes embeddings with SentenceTransformer,
    and saves embeddings and passages to a pickle file.
    
    Args:
        base_dir (str): Base folder where train_dataset exists.
        embeddings_file (str): Name of the pickle file to save embeddings.
        batch_size (int): Number of chunks per batch for embedding.
        model_name (str): SentenceTransformer model name.
        device (str): 'cuda' or 'cpu'.
    """
    train_dir = os.path.join(base_dir, "train_dataset")
    embeddings_path = os.path.join(base_dir, embeddings_file)

    if os.path.exists(embeddings_path):
        print(f"Loading saved embeddings from {embeddings_path}...")
        with open(embeddings_path, "rb") as f:
            data = pickle.load(f)
            corpus_embeddings = data["embeddings"]
            corpus = data["passages"]
        print(f"Loaded {len(corpus)} passages.")
        return corpus, corpus_embeddings

    print("Embeddings not found, computing embeddings...")

    # 1. Load dataset shards
    dataset = load_all_shards(train_dir)
    print(f"Loaded dataset with {len(dataset)} examples.")

    # 2. Extract and chunk passages
    corpus = extract_passages(dataset)

    # 3. Remove duplicates
    before = len(corpus)
    corpus = list(dict.fromkeys(corpus))
    after = len(corpus)
    print(f"Removed {before-after} duplicate passages ({after} unique).")
    print(f"Total chunks to embed: {len(corpus)}")

    # 4. Load model
    model = SentenceTransformer(model_name, device=device)
    print(f"Using device: {device}")

    # 5. Compute embeddings in batches
    corpus_embeddings = []
    for i in tqdm(range(0, len(corpus), batch_size), desc="Computing embeddings"):
        batch = corpus[i:i+batch_size]
        emb = model.encode(batch, convert_to_numpy=True, device=device)
        corpus_embeddings.append(emb)
    corpus_embeddings = np.vstack(corpus_embeddings)

    # 6. Save embeddings
    with open(embeddings_path, "wb") as f:
        pickle.dump({"passages": corpus, "embeddings": corpus_embeddings}, f)
    print(f"Saved embeddings to {embeddings_path}")

    return corpus, corpus_embeddings
