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
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Optional FAISS import
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


# ============================= CONFIG ============================= #
TEST_PATH = "data/test"                  # <--- updated to your new folder
SHARD_PREFIX = "shard_"                  
MAX_QUESTIONS = 100                      

EMBEDDINGS_FILE = Path("corpus_embeddings_unique.pkl")
FAISS_INDEX_FILE = Path("corpus_faiss.index")
PASSAGES_FILE = Path("corpus_passages.pkl")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

TOP_K = 5          # retrieve 5 passages for generation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_GEN_TOKENS = 128
MAX_INPUT_LENGTH = 2048

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
    messages = [
        {"role": "system", "content": "You are a factual assistant. Answer only using the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        padding=True,
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_GEN_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


# ============================== METRICS ============================== #
def exact_match(pred, gold):
    return int(pred.lower().strip() == gold.lower().strip())


def contains_match(pred, gold_aliases):
    pred = pred.lower()
    return int(any(a.lower() in pred for a in gold_aliases))


# =============================== MAIN =============================== #
def run_full_rag_eval(embeddings_file=EMBEDDINGS_FILE):
    print("\n=== Loading embeddings ===")
    faiss_index = None
    corpus_emb = None
    if not FAISS_AVAILABLE:
        raise ImportError("faiss is required. Install faiss-cpu and build the index via compute_embeddings.")
    if not (FAISS_INDEX_FILE.exists() and PASSAGES_FILE.exists()):
        raise FileNotFoundError("FAISS index/passages missing. Run src.compute_embeddings to build them.")
    with open(PASSAGES_FILE, "rb") as f:
        corpus = pickle.load(f)["passages"]
    faiss_index = faiss.read_index(str(FAISS_INDEX_FILE))
    print(f"🔹 Loaded FAISS index with {len(corpus)} passages")

    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token  # Mistral has no pad token
    gen_model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    ).to(DEVICE if DEVICE == "cuda" else "cpu").eval()

    print("\n=== Loading Test dataset (100 samples) ===")
    test = load_test_100()

    em_results, contains_results = [], []
    output_log = []  # stored for optional write

    print("\n=== Running RAG Evaluation ===")
    for ex in tqdm(test):
        q = ex["question"]
        gold = ex["answer"]["normalized_value"]
        aliases = ex["answer"]["normalized_aliases"] + [gold]

        # Retrieval: prefer FAISS if available
        q_emb = embed_model.encode([q], convert_to_numpy=True, device=DEVICE)
        norm = np.linalg.norm(q_emb, axis=1, keepdims=True)
        norm[norm == 0] = 1e-10
        q_norm = (q_emb / norm).astype(np.float32)
        scores, idx = faiss_index.search(q_norm, TOP_K)
        retrieved = [corpus[i] for i in idx[0]]
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
