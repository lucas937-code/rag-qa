import os
import argparse
import pickle
import random
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm


# -----------------------
# Editable defaults
# -----------------------
# Edit these values directly in the file when you want to run the script
# without passing CLI arguments.
CONFIG = {
    "repo_root": ".",
    "embeddings": None,  # None -> use default path under repo_root
    "val_parquet": None,  # None -> use default path under repo_root
    "embed_model": "all-MiniLM-L6-v2",
    "top_k": 5,
    "num_questions": 1000,
    "max_examples": None,
    "output": None,
    "use_cuda": True,
    # set to None -> generate a new random seed every run
    "seed": None,
}


def safe_get_question(row):
    # try a few common keys and shapes
    for key in ("question", "Question", "question_text"):
        if key in row:
            q = row[key]
            if isinstance(q, list):
                return q[0] if q else ""
            if isinstance(q, dict) and "value" in q:
                return q["value"]
            return str(q)
    # fallback: try to find a field containing 'question'
    for k in row.index:
        if "question" in str(k).lower():
            return str(row[k])
    return ""


def extract_gold_titles(row):
    # Try to extract titles from common locations
    # 1) entity_pages -> title
    if "entity_pages" in row and isinstance(row["entity_pages"], dict):
        titles = row["entity_pages"].get("title")
        if isinstance(titles, list) and titles:
            return [t for t in titles if isinstance(t, str)]

    # 2) doc_titles_json (JSON-encoded list)
    if "doc_titles_json" in row and isinstance(row["doc_titles_json"], str):
        try:
            parsed = json.loads(row["doc_titles_json"])
            if isinstance(parsed, list) and parsed:
                return [t for t in parsed if isinstance(t, str)]
        except Exception:
            pass

    # 3) doc_title string
    if "doc_title" in row and isinstance(row["doc_title"], str):
        return [row["doc_title"]]

    # 4) fallback to empty list
    return []


def load_embeddings(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    passages = data["passages"]
    embeddings = data["embeddings"]
    return passages, embeddings


def passage_title(passage_str):
    # embedding passages were saved as "<title>: <text>" in the pipeline
    if not isinstance(passage_str, str):
        return ""
    parts = passage_str.split(":", 1)
    if len(parts) == 2:
        return parts[0].strip()
    return ""


def evaluate(args):
    # Determine seed: if None, generate a new random seed each run
    if args.seed is None:
        # generate a 32-bit random seed from os.urandom for unpredictability
        seed = int.from_bytes(os.urandom(8), "big") % (2 ** 32)
        print("Generated random seed:", seed)
    else:
        seed = int(args.seed)

    random.seed(seed)
    np.random.seed(seed)

    # defaults relative to repo
    repo_root = Path(args.repo_root)
    embeddings_path = Path(args.embeddings) if args.embeddings else repo_root / "data" / "processed" / "embeddings" / "corpus_embeddings_unique.pkl"
    val_parquet = Path(args.val_parquet) if args.val_parquet else repo_root / "data" / "processed" / "splits" / "val_7900.parquet"

    print("Loading embeddings from:", embeddings_path)
    passages, embeddings = load_embeddings(embeddings_path)
    embeddings = np.asarray(embeddings)

    # Precompute passage titles
    passage_titles = [passage_title(p) for p in passages]

    print("Loading validation questions from:", val_parquet)
    df = pd.read_parquet(val_parquet)
    # If `max_examples` is provided, subsample the dataframe for quicker runs.
    # Otherwise, keep the full validation set — do not default to 100 examples.
    if args.max_examples:
        df = df.sample(n=min(args.max_examples, len(df)), random_state=seed)

    samples = df.sample(n=min(args.num_questions, len(df)), random_state=seed)
    print(f"Sampling {len(samples)} questions for evaluation")

    # Load embedder
    device = "cuda" if args.use_cuda and __import__("torch").cuda.is_available() else "cpu"
    print("Using device for embedder:", device)
    embed_model = SentenceTransformer(args.embed_model, device=device)

    topk = args.top_k
    successes = 0
    rows = []

    # iterate with a progress bar so the terminal shows progress while encoding/retrieving
    for i, (_, row) in enumerate(tqdm(samples.iterrows(), total=len(samples), desc="Evaluating queries"), start=1):
        q_text = safe_get_question(row)
        gold_titles = extract_gold_titles(row)

        q_emb = embed_model.encode([q_text], convert_to_numpy=True, device=device)
        sims = cosine_similarity(q_emb, embeddings)[0]
        top_idx = np.argsort(-sims)[:topk]

        retrieved = []
        hit = False
        for rank, idx in enumerate(top_idx, start=1):
            title = passage_titles[idx]
            retrieved.append({"rank": rank, "idx": int(idx), "title": title, "score": float(sims[idx])})
            # compare titles (case-insensitive)
            for g in gold_titles:
                if g and title and g.strip().lower() == title.strip().lower():
                    hit = True
        if hit:
            successes += 1

        rows.append({
            "q_index": i,
            "question": q_text,
            "gold_titles": gold_titles,
            "retrieved": retrieved,
            "hit": hit,
        })

    recall_at_k = successes / len(rows)
    print(f"\nRecall@{topk}: {recall_at_k:.3f} ({successes}/{len(rows)})")

    # Save detailed results
    out_path = Path(args.output) if args.output else Path("retrieval_eval_results.jsonl")
    print("Writing results to:", out_path)
    with open(out_path, "w", encoding="utf8") as fout:
        for r in tqdm(rows, desc="Writing results"):
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    # Use in-file CONFIG only (ignore CLI args).
    # This makes it easy to run the script by editing the CONFIG dict at the top.
    print("Running retrieval evaluation using in-file CONFIG (no CLI args).")
    # Build a simple namespace object with the config values
    args = argparse.Namespace(**CONFIG)
    evaluate(args)


if __name__ == "__main__":
    main()
