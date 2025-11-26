"""
Full RAG Evaluation — Retrieval + Generation + Metrics
Runs on first 100 test questions and logs results + performance scores
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

# ------------------------------ CONFIG ------------------------------ #
TEST_DATA = "../test_dataset"
SHARD_PREFIX = "shard_"
MAX_QUESTIONS = 1000

EMBEDDINGS_FILE = "corpus_embeddings_unique.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-large"

TOP_K = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_INPUT_LEN = 512
MAX_GEN_TOKENS = 128

SAVE_FILE = "rag_eval_output.jsonl"


# ------------------------------ LOAD TEST SHARDS ------------------------------ #
def load_test_100():
    shards = [os.path.join(TEST_DATA, d) for d in os.listdir(TEST_DATA) if d.startswith(SHARD_PREFIX)]
    shards = sorted(shards)
    datasets = [load_from_disk(sh) for sh in shards]
    return concatenate_datasets(datasets).select(range(MAX_QUESTIONS))


# ------------------------------ RAG PIPELINE ------------------------------ #
def retrieve(query, embed_model, corpus, corpus_emb, k=TOP_K):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, corpus_emb)[0]
    top_idx = np.argsort(-sims)[:k]
    return [corpus[i] for i in top_idx]


def generate(query, retrieved, tokenizer, gen_model):
    context_str = "\n\n".join(retrieved)
    prompt = f"Answer the question using the context.\nContext:\n{context_str}\n\nQuestion: {query}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LEN).to(DEVICE)

    with torch.no_grad():
        output = gen_model.generate(**inputs, max_new_tokens=MAX_GEN_TOKENS)

    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


# ------------------------------ METRICS ------------------------------ #
def exact_match(pred, gold):
    return int(pred.lower().strip() == gold.lower().strip())


def contains_match(pred, gold_aliases):
    pred = pred.lower()
    return int(any(a.lower() in pred for a in gold_aliases))


# ------------------------------ MAIN ------------------------------ #
def main():
    print("\n=== Loading Embeddings ===")
    with open(EMBEDDINGS_FILE, "rb") as f:
        data = pickle.load(f)
        corpus, corpus_emb = data["passages"], data["embeddings"]

    embed_model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL).to(DEVICE).eval()

    print("\n=== Loading Test Data (100 questions) ===")
    test = load_test_100()

    out = open(SAVE_FILE, "w")

    em_results = []
    contains_results = []

    print("\n=== Running RAG Evaluation ===")
    for ex in tqdm(test):
        q = ex["question"]
        gold = ex["answer"]["normalized_value"]
        aliases = ex["answer"]["normalized_aliases"] + [gold]

        retrieved = retrieve(q, embed_model, corpus, corpus_emb)
        pred = generate(q, retrieved, tokenizer, gen_model)

        em = exact_match(pred, gold)
        contains = contains_match(pred, aliases)

        em_results.append(em)
        contains_results.append(contains)

        out.write(json.dumps({
            "question": q,
            "gold_answer": gold,
            "aliases_used": aliases,
            "predicted": pred,
            "retrieved_passages": retrieved,
            "exact_match": em,
            "contains_match": contains
        }) + "\n")

    out.close()

    print("\n==================== Final Metrics ====================")
    print(f"Exact Match Accuracy:       {np.mean(em_results):.4f}")
    print(f"Contains-Match Accuracy:    {np.mean(contains_results):.4f}")
    print("Output saved to:", SAVE_FILE)


if __name__ == "__main__":
    main()
