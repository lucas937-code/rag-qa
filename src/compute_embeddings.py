import os
import pickle
import numpy as np
from tqdm import tqdm
import torch
from datasets import load_from_disk, concatenate_datasets
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from config.paths import TRAIN_DIR

# ------------------------------
# Config
# ------------------------------
EMBEDDINGS_FILE = os.path.join(os.path.dirname(__file__), "..", "corpus_embeddings_unique.pkl")
MODEL_NAME = "all-MiniLM-L6-v2"
SHARD_PREFIX = "shard_"
BATCH_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Chunking parameters
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128

# ------------------------------
# Helper: Load all shards
# ------------------------------
def load_all_shards(data_dir=TRAIN_DIR):
    shards = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if d.startswith(SHARD_PREFIX)]
    shards = sorted(shards)
    datasets = []
    for shard in tqdm(shards, desc="Loading shards"):
        datasets.append(load_from_disk(shard))
    if datasets:
        return concatenate_datasets(datasets)
    return None

# ------------------------------
# Helper: Chunk passages
# ------------------------------
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

# ------------------------------
# Helper: Extract passages from dataset
# ------------------------------
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
# Compute embeddings (with file existence check)
# ------------------------------
def compute_embeddings(embeddings_file=EMBEDDINGS_FILE, force_recompute=False):
    """
    Computes embeddings or loads them if EMBEDDINGS_FILE exists.
    Set force_recompute=True to overwrite existing embeddings.
    Returns: corpus (list of passages), corpus_embeddings (numpy array)
    """
    if os.path.exists(embeddings_file) and not force_recompute:
        print(f"Loading saved embeddings from {embeddings_file}...")
        with open(embeddings_file, "rb") as f:
            data = pickle.load(f)
            corpus_embeddings = data["embeddings"]
            corpus = data["passages"]
        print(f"Loaded {len(corpus)} passages.")
        return corpus, corpus_embeddings

    print("Embeddings not found or force_recompute=True, computing embeddings...")

    # 1. Load dataset
    dataset = load_all_shards()
    if dataset is None:
        print("No shards found! Make sure TRAIN_DIR has shards.")
        return None, None
    print(f"Loaded dataset with {len(dataset)} examples.")

    # 2. Extract passages
    corpus = extract_passages(dataset)
    
    # 3. Remove duplicates
    before = len(corpus)
    corpus = list(dict.fromkeys(corpus))
    after = len(corpus)
    print(f"Removed {before-after} duplicate passages ({after} unique).")

    # 4. Load model
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    print(f"Using device: {DEVICE}")

    # 5. Compute embeddings in batches
    corpus_embeddings = []
    for i in tqdm(range(0, len(corpus), BATCH_SIZE), desc="Computing embeddings"):
        batch = corpus[i:i+BATCH_SIZE]
        emb = model.encode(batch, convert_to_numpy=True, device=DEVICE)
        corpus_embeddings.append(emb)
    corpus_embeddings = np.vstack(corpus_embeddings)

    # 6. Save embeddings
    with open(embeddings_file, "wb") as f:
        pickle.dump({"passages": corpus, "embeddings": corpus_embeddings}, f)
    print(f"Saved embeddings to {embeddings_file}")

    return corpus, corpus_embeddings

# ------------------------------
# Retrieval function
# ------------------------------
def retrieve_top_k(query, corpus, corpus_embeddings, model_name=MODEL_NAME, device=DEVICE, top_k=3):
    model = SentenceTransformer(model_name, device=device)
    query_embedding = model.encode([query], convert_to_numpy=True, device=device)
    sims = cosine_similarity(query_embedding, corpus_embeddings)
    top_idx = np.argsort(-sims[0])[:top_k]
    results = [corpus[i] for i in top_idx]
    return results, sims[0][top_idx]

# ------------------------------
# Entry point for script
# ------------------------------
if __name__ == "__main__":
    corpus, corpus_embeddings = compute_embeddings()
    if corpus is not None and corpus_embeddings is not None:
        # Example test retrieval
        query = "What is the top prize at the Cannes Film Festival?"
        results, scores = retrieve_top_k(query, corpus, corpus_embeddings)
        print("\nTop retrieved passages:")
        for passage, score in zip(results, scores):
            print(f"[score: {score:.4f}] {passage}\n---")
