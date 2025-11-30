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
import re
import string
from tqdm import tqdm
from src.config import Config, DEFAULT_CONFIG

# Optional FAISS import
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


# ============================= CONFIG ============================= #
MAX_QUESTIONS = 100
GEN_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

TOP_K = 5          # retrieve 5 passages for generation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_GEN_TOKENS = 128
MAX_INPUT_LENGTH = 2048


# ======================== LOAD TEST SHARDS ======================== #
def load_test_100(config: Config):
    shards = sorted([
        os.path.join(config.TEST_DIR, d)
        for d in os.listdir(config.TEST_DIR)
        if d.startswith(DEFAULT_CONFIG.SHARD_PREFIX)
    ])

    if not shards:
        raise RuntimeError(f"⚠ No shards found in {config.TEST_DIR}")

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
        {
            "role": "system",
            "content": (
                "You are a concise QA assistant. Answer ONLY from the provided context. "
                "If unsure, reply \"I don't know\". Respond in <=20 words."
            ),
        },
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

# TriviaQA-style normalization and EM/F1
def normalize_answer(s: str):
    def lower(text):
        return text.lower()
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def em_score(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))

def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    num_same = sum(min(pred_tokens.count(tok), gold_tokens.count(tok)) for tok in common)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# =============================== MAIN =============================== #
def run_full_rag_eval(config: Config = DEFAULT_CONFIG):
    embed_model_name = Path(config.EMBEDDING_MODEL)
    faiss_index_file = Path(config.FAISS_INDEX_FILE)
    passages_file = Path(config.PASSAGES_FILE)
    print("\n=== Loading embeddings ===")
    faiss_index = None
    corpus_emb = None
    if not FAISS_AVAILABLE:
        raise ImportError("faiss is required. Install faiss-cpu and build the index via compute_embeddings.")
    if not (faiss_index_file.exists() and passages_file.exists()):
        raise FileNotFoundError("FAISS index/passages missing. Run src.compute_embeddings to build them.")
    with open(passages_file, "rb") as f:
        corpus = pickle.load(f)["passages"]
    faiss_index = faiss.read_index(str(faiss_index_file)) # type: ignore
    print(f"🔹 Loaded FAISS index with {len(corpus)} passages")

    embed_model = SentenceTransformer(embed_model_name, device=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token  # Mistral has no pad token
    gen_model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    ).to(DEVICE if DEVICE == "cuda" else "cpu").eval()

    print("\n=== Loading Test dataset (100 samples) ===")
    test = load_test_100(config)

    em_results, f1_results = [], []
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

        em_results.append(max(em_score(pred, a) for a in aliases))
        f1_results.append(max(f1_score(pred, a) for a in aliases))

        output_log.append({
            "question": q,
            "gold": gold,
            "aliases": aliases,
            "predicted": pred,
            "retrieved": [r[:200] for r in retrieved],
            "exact_match": em_results[-1],
            "f1": f1_results[-1],
        })

    # Optionally save
    # with open(SAVE_FILE, "w") as f:
    #     for row in output_log:
    #         f.write(json.dumps(row) + "\n")

    print("\n================ Final Results ================")
    print(f"Exact Match (EM):           {np.mean(em_results):.4f}")
    print(f"F1:                         {np.mean(f1_results):.4f}")
    # print(f"Results saved to {SAVE_FILE}")


if __name__ == "__main__":
    run_full_rag_eval()
