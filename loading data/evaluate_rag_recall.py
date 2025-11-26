# evaluate_rag_recall_dev.py
import os
import pickle
from datasets import load_from_disk, concatenate_datasets
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import torch

# ------------------------------
# Config
# ------------------------------
DATA_DIRS = {
    "train": "../train_dataset",
    "test": "../test_dataset"
}
EMBEDDINGS_FILE = "corpus_embeddings_unique.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K_VALUES = [1,3, 5,7,10]
SHARD_PREFIX = "shard_"
DEV_LIMIT = 1000  # Only first 10 examples for development

# ------------------------------
# Helper: Load all shards
# ------------------------------
def load_all_shards(data_dir):
    shards = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if d.startswith(SHARD_PREFIX)]
    shards = sorted(shards)
    datasets = []
    for shard in tqdm(shards, desc=f"Loading shards from {data_dir}"):
        datasets.append(load_from_disk(shard))
    dataset = concatenate_datasets(datasets)
    # Limit to first DEV_LIMIT examples
    return dataset.select(range(min(DEV_LIMIT, len(dataset))))

# ------------------------------
# Evaluate Recall@k
# ------------------------------
def evaluate_recall(model, corpus, corpus_embeddings, dataset, top_k_list):
    recalls = {k: [] for k in top_k_list}
    
    for example in tqdm(dataset, desc="Evaluating recall"):
        question = example['question']
        normalized_answers = example['answer']['normalized_aliases']

        # Encode query
        query_emb = model.encode([question], convert_to_numpy=True, device=DEVICE)

        # Compute similarity
        sims = cosine_similarity(query_emb, corpus_embeddings)[0]
        top_indices = np.argsort(-sims)

        # Evaluate Recall@k
        for k in top_k_list:
            top_k_passages = [corpus[idx].lower() for idx in top_indices[:k]]
            found = any(any(ans in passage for ans in normalized_answers) for passage in top_k_passages)
            recalls[k].append(int(found))
    
    # Compute mean recall for each k
    return {k: np.mean(v) for k, v in recalls.items()}

# ------------------------------
# Main
# ------------------------------
def main():
    # Load embeddings
    if not os.path.exists(EMBEDDINGS_FILE):
        raise FileNotFoundError(f"{EMBEDDINGS_FILE} not found. Please compute embeddings first.")
    with open(EMBEDDINGS_FILE, "rb") as f:
        data = pickle.load(f)
        corpus = data["passages"]
        corpus_embeddings = data["embeddings"]
    print(f"Loaded {len(corpus)} passages and embeddings of shape {corpus_embeddings.shape}")

    # Load SentenceTransformer model
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    # Evaluate for each dataset split
    for split_name, data_dir in DATA_DIRS.items():
        print(f"\n=== Evaluating {split_name} dataset (first {DEV_LIMIT} examples) ===")
        dataset = load_all_shards(data_dir)
        recall_results = evaluate_recall(model, corpus, corpus_embeddings, dataset, TOP_K_VALUES)
        for k, recall in recall_results.items():
            print(f"Recall@{k}: {recall:.4f}")

if __name__ == "__main__":
    main()
