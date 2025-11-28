"""
Full RAG Evaluation — Retrieval + Generation + Metrics
Evaluates RAG on first 100 test questions.
"""

import os
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from datasets import load_from_disk, concatenate_datasets
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# ============================= CONFIG ============================= #
TEST_PATH = "data/test"                  # <--- updated to your new folder
SHARD_PREFIX = "shard_"                  
MAX_QUESTIONS = 100                      

EMBEDDINGS_FILE = Path("corpus_embeddings_unique.pkl")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-large"

TOP_K = 5          # retrieve 5 passages for generation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_FILE = "rag_eval_output.jsonl"      # enable write block to store results


# ======================== LOAD TEST SHARDS ======================== #
def load_test_100():
    shards = sorted([
        os.path.join(TEST_PATH, d)
        for d in os.listdir(TEST_PATH)
        if d.startswith(SHARD_PREFIX)
    ])

    if not shards:
        raise RuntimeError(f"⚠ No shards found in {TEST_PATH}")

    ds = concatenate_datasets([load_from_disk(sh) for sh in shards])
    return ds.select(range(min(MAX_QUESTIONS, len(ds))))


# ============================= RAG CORE ============================= #
def retrieve(query, embed_model, corpus, corpus_emb, k=TOP_K):
    q_emb = embed_model.encode([query], convert_to_numpy=True, device=DEVICE)
    sims = cosine_similarity(q_emb, corpus_emb)[0]
    top_idx = np.argsort(-sims)[:k]
    return [corpus[i] for i in top_idx]


def generate(query, retrieved, tokenizer, model):
    context = "\n---\n".join(retrieved)
    prompt = (
        "You are a helpful factual assistant.\n"
        "Answer only based on the given context.\n"
        f"\nContext:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=128)

    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


# ============================== METRICS ============================== #
def exact_match(pred, gold):
    return int(pred.lower().strip() == gold.lower().strip())


def contains_match(pred, gold_aliases):
    pred = pred.lower()
    return int(any(a.lower() in pred for a in gold_aliases))


# =============================== MAIN =============================== #
def run_full_rag_eval():
    print("\n=== Loading embeddings ===")
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError("❌ Embedding file missing. Compute embeddings first.")

    data = pickle.load(open(EMBEDDINGS_FILE, "rb"))
    corpus, corpus_emb = data["passages"], data["embeddings"]

    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME).to(DEVICE).eval()

    print("\n=== Loading Test dataset (100 samples) ===")
    test = load_test_100()

    em_results, contains_results = [], []
    output_log = []  # stored for optional write

    print("\n=== Running RAG Evaluation ===")
    for ex in tqdm(test):
        q = ex["question"]
        gold = ex["answer"]["normalized_value"]
        aliases = ex["answer"]["normalized_aliases"] + [gold]

        retrieved = retrieve(q, embed_model, corpus, corpus_emb, k=TOP_K)
        pred = generate(q, retrieved, tokenizer, gen_model)

        em_results.append(exact_match(pred, gold))
        contains_results.append(contains_match(pred, aliases))

        output_log.append({
            "question": q,
            "gold": gold,
            "aliases": aliases,
            "predicted": pred,
            "retrieved": [r[:200] for r in retrieved],
            "exact_match": em_results[-1],
            "contains_match": contains_results[-1]
        })

    # Optionally save
    # with open(SAVE_FILE, "w") as f:
    #     for row in output_log:
    #         f.write(json.dumps(row) + "\n")

    print("\n================ Final Results ================")
    print(f"Exact Match Accuracy:       {np.mean(em_results):.4f}")
    print(f"Contains-Match Accuracy:    {np.mean(contains_results):.4f}")
    # print(f"Results saved to {SAVE_FILE}")


if __name__ == "__main__":
    run_full_rag_eval()
