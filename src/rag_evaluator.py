"""
Full RAG Evaluation — Retrieval + Generation + Metrics
Uses the dynamic RAG stack from src.generator (FAISS + reranker + generator).
"""

import os
import json
import numpy as np
import re
import string
from datasets import load_from_disk, concatenate_datasets
from tqdm import tqdm

from src.config import Config, DEFAULT_CONFIG
from src.retriever import Retriever
from src.reranker import Reranker
from src.generator import Generator


class RAGEvaluator:
    def __init__(self, config: Config):
        self.test_dir = config.test_dir
    
    # ===== Internal Methods =====

    def _load_test_100(self, max_questions: int):
        shards = sorted([
            os.path.join(self.test_dir, d)
            for d in os.listdir(self.test_dir)
            if d.startswith(DEFAULT_CONFIG.shard_prefix)
        ])

        if not shards:
            raise RuntimeError(f"⚠ No shards found in {self.test_dir}")

        ds = concatenate_datasets([load_from_disk(sh) for sh in shards])
        return ds.select(range(min(max_questions, len(ds))))


    def _normalize_answer(self, s: str):
        def lower(text):
            return text.lower()
        def remove_punc(text):
            return "".join(ch for ch in text if ch not in string.punctuation)
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)
        def white_space_fix(text):
            return " ".join(text.split())
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def _calc_em_score(self, pred: str, gold: str) -> int:
        return int(self._normalize_answer(pred) == self._normalize_answer(gold))

    def _calc_f1_score(self, pred: str, gold: str) -> float:
        pred_tokens = self._normalize_answer(pred).split()
        gold_tokens = self._normalize_answer(gold).split()
        if not pred_tokens and not gold_tokens:
            return 1.0
        if not pred_tokens or not gold_tokens:
            return 0.0
        common = set(pred_tokens) & set(gold_tokens)
        num_same = sum(
            min(pred_tokens.count(tok), gold_tokens.count(tok)) for tok in common
        )
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)


    # ===== Public Methods =====
    def run_full_rag_eval(self,
                          corpus,
                          embeddings,
                          retriever: Retriever,
                          reranker: Reranker, 
                          generator: Generator,
                          max_questions: int = 100,
                          candidates: int = 100,
                          top_k: int = 3,
                          save_file: str | None = None):
        print("\n=== Loading Test dataset ===")
        test = self._load_test_100(max_questions=max_questions)

        em_results, f1_results = [], []
        output_log = []

        print("\n=== Running RAG Evaluation ===")
        for ex in tqdm(test):
            q = ex["question"]
            gold = ex["answer"]["normalized_value"]
            aliases = ex["answer"]["normalized_aliases"] + [gold]

            retrieved_passages, _, candidates_idx = retriever.retrieve(
            query=q, 
            corpus=corpus,
            faiss_index=embeddings, 
            top_k=candidates)

            context, _ = reranker.rerank_and_get_top_k(
            query=q, 
            corpus=corpus, 
            candidates_idx=candidates_idx,
            top_k=top_k)

            # RAG: retrieval + generation (uses FAISS + reranker + generator from src.generator)
            pred = generator.generate(q, retrieved_passages)

            # Metrics: max over all aliases (TriviaQA-style)
            em = max(self._calc_em_score(pred, a) for a in aliases)
            f1 = max(self._calc_f1_score(pred, a) for a in aliases)

            em_results.append(em)
            f1_results.append(f1)

            output_log.append({
                "question": q,
                "gold": gold,
                "aliases": aliases,
                "predicted": pred,
                "retrieved": [r[:200] for r in context],
                "exact_match": em,
                "f1": f1,
            })

        if save_file is not None:
            with open(save_file, "w", encoding="utf-8") as f:
                for row in output_log:
                    f.write(json.dumps(row) + "\n")
            print(f"\n📁 Results saved to {save_file}")

        print("\n================ Final Results ================")
        print(f"Exact Match (EM): {np.mean(em_results):.4f}")
        print(f"F1:              {np.mean(f1_results):.4f}")