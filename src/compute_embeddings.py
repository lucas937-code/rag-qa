import os
import pickle
import numpy as np
from tqdm import tqdm
import torch
from datasets import load_from_disk, concatenate_datasets
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import Config, DEFAULT_CONFIG

# Optional FAISS import
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# ------------------------------
# Config
# ------------------------------
BATCH_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Chunking parameters (token-based; MiniLM has a ~256 token limit)
CHUNK_TOKENS = 240
CHUNK_OVERLAP = 60

# ------------------------------
# Helper: Load all shards from one or multiple directories
# ------------------------------
def load_all_shards(data_dirs=None):
    if data_dirs is None:
        data_dirs = [DEFAULT_CONFIG.TRAIN_DIR, DEFAULT_CONFIG.VAL_DIR, DEFAULT_CONFIG.TEST_DIR]

    datasets = []
    for data_dir in data_dirs:
        if not os.path.isdir(data_dir):
            continue
        shards = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if d.startswith(DEFAULT_CONFIG.SHARD_PREFIX)]
        shards = sorted(shards)
        for shard in tqdm(shards, desc=f"Loading shards from {data_dir}"):
            datasets.append(load_from_disk(shard))

    if datasets:
        return concatenate_datasets(datasets)
    return None

# ------------------------------
# Helper: Chunk passages
# ------------------------------
def chunk_passage(title, text, tokenizer, chunk_tokens=CHUNK_TOKENS, overlap=CHUNK_OVERLAP):
    """
    Token-based chunking to avoid encoder truncation. Keeps simple overlaps.
    """
    # Tokenize with truncation disabled to get full sequence of ids
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_tokens
        token_slice = tokens[start:end]
        chunk_text = tokenizer.decode(token_slice, skip_special_tokens=True).strip()
        chunks.append(f"{title}: {chunk_text}")
        # Move window with overlap
        start += max(1, chunk_tokens - overlap)
    return chunks

# ------------------------------
# Helper: Extract passages from dataset
# ------------------------------
def extract_passages(dataset, tokenizer):
    passages = []
    for example in tqdm(dataset, desc="Extracting passages"):
        entity_pages = example.get("entity_pages", {})
        titles = entity_pages.get("title", [])
        wiki_contexts = entity_pages.get("wiki_context", [])
        n_pages = min(len(titles), len(wiki_contexts))
        for i in range(n_pages):
            title = titles[i] or "No Title"
            text = wiki_contexts[i]
            passages.extend(chunk_passage(title, text, tokenizer))
    return passages

# ------------------------------
# Compute embeddings (with file existence check)
# ------------------------------
def compute_embeddings(config: Config, force_recompute=False):
    """
    Computes embeddings or loads them if EMBEDDINGS_FILE exists.
    Set force_recompute=True to overwrite existing embeddings.
    Returns: corpus (list of passages), corpus_embeddings (numpy array)
    """
    if os.path.exists(config.EMBEDDINGS_FILE) and not force_recompute:
        print(f"Loading saved embeddings from {config.EMBEDDINGS_FILE}...")
        with open(config.EMBEDDINGS_FILE, "rb") as f:
            data = pickle.load(f)
            corpus_embeddings = data["embeddings"]
            corpus = data["passages"]
        print(f"Loaded {len(corpus)} passages.")
        return corpus, corpus_embeddings

    print("Embeddings not found or force_recompute=True, computing embeddings...")

    # 1. Load dataset
    dataset = load_all_shards([config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR])
    if dataset is None:
        print("No shards found! Make sure TRAIN/VAL/TEST dirs have shards.")
        return None, None
    print(f"Loaded dataset with {len(dataset)} examples.")

    # 2. Load model (also needed for tokenizer)
    model = SentenceTransformer(config.EMBEDDING_MODEL, device=DEVICE)
    tokenizer = model.tokenizer
    print(f"Using device: {DEVICE}")

    # 3. Extract passages (token-based chunking)
    corpus = extract_passages(dataset, tokenizer)
    
    # 4. Remove duplicates
    before = len(corpus)
    corpus = list(dict.fromkeys(corpus))
    after = len(corpus)
    print(f"Removed {before-after} duplicate passages ({after} unique).")

    # 5. Compute embeddings in batches
    corpus_embeddings = []
    for i in tqdm(range(0, len(corpus), BATCH_SIZE), desc="Computing embeddings"):
        batch = corpus[i:i+BATCH_SIZE]
        emb = model.encode(batch, convert_to_numpy=True, device=DEVICE)
        corpus_embeddings.append(emb)
    corpus_embeddings = np.vstack(corpus_embeddings)

    # 6. Save embeddings (pickle) for backward compatibility
    with open(config.EMBEDDINGS_FILE, "wb") as f:
        pickle.dump({"passages": corpus, "embeddings": corpus_embeddings}, f)
    print(f"Saved embeddings to {config.EMBEDDINGS_FILE}")

    # 7. Save passages separately (FAISS uses its own binary)
    with open(config.PASSAGES_FILE, "wb") as f:
        pickle.dump({"passages": corpus}, f)
    print(f"Saved passages to {config.PASSAGES_FILE}")

    # 8. Build FAISS index (inner product on L2-normalized vectors)
    if FAISS_AVAILABLE:
        norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        emb_normalized = corpus_embeddings / norms
        dim = emb_normalized.shape[1]
        index = faiss.IndexFlatIP(dim) # type: ignore
        index.add(emb_normalized.astype(np.float32))
        faiss.write_index(index, config.FAISS_INDEX_FILE) # type: ignore
        print(f"Saved FAISS index to {config.FAISS_INDEX_FILE} (dim={dim}, n={len(corpus)})")
    else:
        print("⚠ FAISS not installed. Install faiss-cpu to enable fast retrieval.")

    return corpus, corpus_embeddings

# ------------------------------
# Retrieval function
# ------------------------------
def retrieve_top_k(query, corpus, corpus_embeddings, config: Config, device=DEVICE, top_k=3):
    """
    Fallback retrieval using in-memory embeddings (cosine similarity).
    """
    model = SentenceTransformer(config.EMBEDDING_MODEL, device=device)
    query_embedding = model.encode([query], convert_to_numpy=True, device=device)
    sims = cosine_similarity(query_embedding, corpus_embeddings)
    top_idx = np.argsort(-sims[0])[:top_k]
    results = [corpus[i] for i in top_idx]
    return results, sims[0][top_idx]

# ------------------------------
# Entry point for script
# ------------------------------
if __name__ == "__main__":
    corpus, corpus_embeddings = compute_embeddings(DEFAULT_CONFIG)
    if corpus is not None and corpus_embeddings is not None:
        # Example test retrieval
        query = "What is the top prize at the Cannes Film Festival?"
        results, scores = retrieve_top_k(query, corpus, corpus_embeddings, DEFAULT_CONFIG)
        print("\nTop retrieved passages:")
        for passage, score in zip(results, scores):
            print(f"[score: {score:.4f}] {passage}\n---")
