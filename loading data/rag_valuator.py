"""
RAG Evaluation Module — retrieval + generation + accuracy scoring
Exports run_rag_eval() for notebook execution
"""

import os
import json
import pickle
import torch
import numpy as np
from datasets import load_from_disk, concatenate_datasets
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ==============================================================
# CONFIG — YOU CAN OVERRIDE THESE WHEN CALLING run_rag_eval()
# ==============================================================

DEFAULT_BASEDIR = "/content/drive/MyDrive/rag-matthias"
EMBED_FILE_NAME = "corpus_embeddings_unique.pkl"
TEST_DATA_FOLDER = "test_dataset"       # resolved as BASEDIR/test_dataset
SHARD_PREFIX = "shard_"
TOP_K = 3
MAX_QUESTIONS = 100                      # evaluation size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBED_MODEL = "all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-large"


# ==============================================================
# LOAD TEST SAMPLE QUESTIONS (first 100)
# ==============================================================
def load_test_set(base_dir):
    path = os.path.join(base_dir, TEST_DATA_FOLDER)
    shards = [os.path.join(path, d) for d in os.listdir(path) if d.startswith(SHARD_PREFIX)]
    shards = sorted(shards)
    datasets = [load_from_disk(sh) for sh in shards]
    return concatenate_datasets(datasets).select(range(MAX_QUESTIONS))


# ==============================================================
# RAG pipeline functions
# ==============================================================
def retrieve(query, embed_model, corpus, emb, k=TOP_K):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, emb)[0]
    top_idx = np.argsort(-sims)[:k]
    return [corpus[i] for i in top_idx]


def generate(query, retrieved, tokenizer, model):
    context = "\n\n".join(retrieved)
    prompt = f"Answer using the context.\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(**inputs)       # no max length → full generation
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


# ==============================================================
# METRIC scoring
# ==============================================================
def exact_match(pred, gold):
    return int(pred.lower().strip() == gold.lower().strip())

def contains_match(pred, gold_aliases):
    pred = pred.lower()
    return int(any(a.lower() in pred for a in gold_aliases))


# ==============================================================
# MAIN exported function for Notebook
# ==============================================================
def run_rag_eval(base_dir=DEFAULT_BASEDIR):

    # ---- load embeddings ----
    embed_path = os.path.join(base_dir, EMBED_FILE_NAME)
    print("Loading embeddings:", embed_path)
    with open(embed_path, "rb") as f:
        data = pickle.load(f)
        corpus, corpus_emb = data["passages"], data["embeddings"]

    print(f"Loaded {len(corpus)} passages\n")

    # ---- load models ----
    embed_model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL).to(DEVICE).eval()

    # ---- load test questions ----
    print("Loading test set (100 q)...\n")
    test = load_test_set(base_dir)

    EMs, CONTAINS = [], []

    print("Running Evaluation...\n")
    for ex in tqdm(test):
        q       = ex["question"]
        gold    = ex["answer"]["normalized_value"]
        aliases = ex["answer"]["normalized_aliases"] + [gold]

        retrieved = retrieve(q, embed_model, corpus, corpus_emb)
        pred      = generate(q, retrieved, tokenizer, gen_model)

        EMs.append(exact_match(pred, gold))
        CONTAINS.append(contains_match(pred, aliases))

    print("\n============== Results ==============")
    print(f"Exact-Match Accuracy:       {np.mean(EMs):.4f}")
    print(f"Contains-Answer Accuracy:   {np.mean(CONTAINS):.4f}")

    return np.mean(EMs), np.mean(CONTAINS)
