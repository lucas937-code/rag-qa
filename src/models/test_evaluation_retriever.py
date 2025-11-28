import os
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch

# ============================================
# CONFIG
# ============================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

EMBEDDINGS_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "embeddings",
    "corpus_embeddings_unique.pkl"
)

# Example path — adapt this to your actual file
TRIVIA_QA_DEV_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "triviaqa_dev_with_orig_index.pkl"   # e.g. a pickled pandas DataFrame
)

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================
# LOAD EMBEDDINGS
# ============================================
def load_corpus_embeddings():
    with open(EMBEDDINGS_FILE, "rb") as f:
        data = pickle.load(f)
        corpus = data["passages"]
        corpus_embeddings = data["embeddings"]

        # Optional: if you stored original_indices when deduplicating / chunking
        original_indices = data.get("original_indices", None)

        return corpus, corpus_embeddings, original_indices


# ============================================
# SINGLE QUERY RETRIEVAL (your existing function)
# ============================================
def retrieve(query, corpus, corpus_embeddings, embed_model, top_k=TOP_K, original_indices=None):
    """
    query: str
    corpus: list of passages (N)
    corpus_embeddings: np.ndarray (N x d)
    original_indices: np.array of original indices of each passage in the full corpus (optional)
    """
    query_emb = embed_model.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(query_emb, corpus_embeddings)
    top_idx = np.argsort(-sims[0])[:top_k]

    results = []
    for idx in top_idx:
        orig_idx = int(original_indices[idx]) if original_indices is not None else int(idx)
        results.append({
            "rank": len(results) + 1,
            "subset_index": int(idx),
            "original_index": orig_idx,
            "passage": corpus[idx],
            "score": float(sims[0][idx])
        })

    return results


# ============================================
# RETRIEVAL EVALUATION
# ============================================
def evaluate_retrieval(
    qa_df,
    corpus,
    corpus_embeddings,
    embed_model,
    original_indices=None,
    k_values=(1, 5, 20),
    max_questions=None
):
    """
    qa_df: pandas DataFrame with columns:
        - 'question'      (str)
        - 'orig_index'    (int) -> ground-truth passage index
          This 'orig_index' should be comparable to:
            - original_indices[idx] if original_indices is not None
            - or directly to passage index 'idx' if original_indices is None.

    original_indices:
        - np.array of shape (N,), where original_indices[i] is the "original corpus index"
          for corpus[i]. If None, we assume i itself is the index.

    k_values: tuple of K's for recall@K
    max_questions: if not None, evaluate only on first max_questions rows (for speed).

    Returns:
        - recall_at_k: dict {k: value}
        - mrr: float
    """

    # Metrics counters
    hits_at_k = {k: 0 for k in k_values}
    mrr_sum = 0.0

    # Optionally sample subset for speed
    if max_questions is not None and max_questions < len(qa_df):
        qa_df = qa_df.sample(n=max_questions, random_state=42).reset_index(drop=True)

    n_questions = len(qa_df)
    print(f"Evaluating on {n_questions} questions...")

    for i, row in qa_df.iterrows():
        question_text = row["question"]
        gt_orig_index = int(row["orig_index"])  # ground-truth original index

        # Encode question
        q_emb = embed_model.encode([question_text], convert_to_numpy=True)
        sims = cosine_similarity(q_emb, corpus_embeddings)[0]
        ranked_passage_indices = np.argsort(-sims)  # indices into 'corpus'

        # Map ranking to original indices (if provided)
        if original_indices is not None:
            ranked_original_indices = [int(original_indices[idx]) for idx in ranked_passage_indices]
        else:
            ranked_original_indices = [int(idx) for idx in ranked_passage_indices]

        # Find rank of first correct passage
        rank_of_gt = None
        for rank, orig_idx in enumerate(ranked_original_indices, start=1):
            if orig_idx == gt_orig_index:
                rank_of_gt = rank
                break

        # If the correct passage isn't in the corpus / mapping, count as miss
        if rank_of_gt is None:
            continue

        # Update Recall@K / Hit@K
        for k in k_values:
            if rank_of_gt <= k:
                hits_at_k[k] += 1

        # Update MRR
        mrr_sum += 1.0 / rank_of_gt

        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{n_questions} questions...")

    recall_at_k = {k: hits_at_k[k] / n_questions for k in k_values}
    mrr = mrr_sum / n_questions

    return recall_at_k, mrr


# ============================================
# MAIN (example usage)
# ============================================
if __name__ == "__main__":
    print("\nLoading corpus embeddings from:")
    print(EMBEDDINGS_FILE)
    corpus, corpus_embeddings, original_indices = load_corpus_embeddings()

    print("\nLoading dev questions from:")
    print(TRIVIA_QA_DEV_FILE)
    qa_df = pd.read_pickle(TRIVIA_QA_DEV_FILE)  # or pd.read_csv(...) depending on your format

    print("\nLoading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)

    # ---------- 1. Manual sanity check (what you already had) ----------
    query = "Who discovered penicillin?"
    print(f"\nManual sanity check — Query: {query}\n")
    manual_results = retrieve(
        query,
        corpus,
        corpus_embeddings,
        embed_model,
        top_k=TOP_K,
        original_indices=original_indices
    )
    print("\n===== MANUAL RETRIEVAL RESULTS =====\n")
    for r in manual_results:
        print(
            f"[Rank {r['rank']}]  "
            f"Score: {r['score']:.4f}  "
            f"(Passage idx {r['subset_index']} | Original idx {r['original_index']})"
        )
        print(r["passage"])
        print("-" * 80)

    # ---------- 2. Proper evaluation over many questions ----------
    k_values = (1, 5, 20)
    recall_at_k, mrr = evaluate_retrieval(
        qa_df,
        corpus,
        corpus_embeddings,
        embed_model,
        original_indices=original_indices,
        k_values=k_values,
        max_questions=1000,   # set None to use all questions
    )

    print("\n===== EVALUATION METRICS =====")
    for k in k_values:
        print(f"Recall@{k}: {recall_at_k[k]:.4f}")
    print(f"MRR: {mrr:.4f}")
