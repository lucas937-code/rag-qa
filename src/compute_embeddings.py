import os
import pickle
import numpy as np
from tqdm import tqdm
import torch
from datasets import load_from_disk, concatenate_datasets
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import Config, DEFAULT_CONFIG, LocalConfig
import faiss

class Embedder:
    def __init__(self, config: Config):
        self.chunk_tokens = config.chunk_tokens
        self.chunk_overlap = config.chunk_overlap
        self.batch_size = config.embeddings_batch_size
        self.model_name = config.embedding_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== Internal Methods =====
    def _load_all_shards(self, data_dirs=None, max_shards_per_dir=None):
        if data_dirs is None:
            data_dirs = [DEFAULT_CONFIG.train_dir, DEFAULT_CONFIG.val_dir, DEFAULT_CONFIG.test_dir]

        datasets = []
        for data_dir in data_dirs:
            if not os.path.isdir(data_dir):
                continue
            shards = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if d.startswith(DEFAULT_CONFIG.shard_prefix)]
            shards = sorted(shards)
            if max_shards_per_dir is not None:
                shards = shards[:max_shards_per_dir]
            for shard in tqdm(shards, desc=f"Loading shards from {data_dir}"):
                datasets.append(load_from_disk(shard))

        if datasets:
            return concatenate_datasets(datasets)
        return None

    def _chunk_passage(self, title, text, tokenizer):
        """
        Token-based chunking to avoid encoder truncation. Keeps simple overlaps.
        """
        # Tokenize with truncation disabled to get full sequence of ids
        tokens = tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + self.chunk_tokens
            token_slice = tokens[start:end]
            chunk_text = tokenizer.decode(token_slice, skip_special_tokens=True).strip()
            chunks.append(f"{title}: {chunk_text}")
            # Move window with overlap
            start += max(1, self.chunk_tokens - self.chunk_overlap)
        return chunks

    def _extract_passages(self, dataset, tokenizer):
        passages = []
        for example in tqdm(dataset, desc="Extracting passages"):
            entity_pages = example.get("entity_pages", {})
            titles = entity_pages.get("title", [])
            wiki_contexts = entity_pages.get("wiki_context", [])
            n_pages = min(len(titles), len(wiki_contexts))
            for i in range(n_pages):
                title = titles[i] or "No Title"
                text = wiki_contexts[i]
                passages.extend(self._chunk_passage(title, text, tokenizer))
        return passages

    # ===== Public Methods =====
    def compute_embeddings(
        self, 
        config: Config,
        force_recompute=False,
        recompute_passages=False,
        data_dirs=None,
        max_shards_per_dir=None):
        """
        - If embeddings file exists and force_recompute=False: load and return.
        - Otherwise, reuse cached passages if available (unless recompute_passages=True),
        then compute embeddings + FAISS index.
        Returns: corpus (list of passages), corpus_embeddings (numpy array)
        """
        print()
        if os.path.exists(config.embeddings_file) and not force_recompute:
            print(f"Loading saved embeddings from {config.embeddings_file}...")
            with open(config.embeddings_file, "rb") as f:
                data = pickle.load(f)
                corpus_embeddings = data["embeddings"]
                corpus = data["passages"]
            print(f"Loaded {len(corpus)} passages.")
            return corpus, corpus_embeddings

        print("Embeddings not found or force_recompute=True, computing embeddings...")

        # 1. Passages: reuse cached file if available and not recomputing
        corpus = None
        if (not recompute_passages) and os.path.exists(config.passages_file):
            with open(config.passages_file, "rb") as f:
                corpus = pickle.load(f)["passages"]
            print(f"Reused cached passages from {config.passages_file} ({len(corpus)} passages).")
        else:
            dirs_to_load = data_dirs if data_dirs is not None else [config.train_dir, config.val_dir, config.test_dir]
            dataset = self._load_all_shards(dirs_to_load, max_shards_per_dir=max_shards_per_dir)
            if dataset is None:
                print("No shards found! Make sure TRAIN/VAL/TEST dirs have shards.")
                return None, None
            print(f"Loaded dataset with {len(dataset)} examples.")

            # 2. Load model (also needed for tokenizer)
            model_tmp = SentenceTransformer(config.embedding_model, device=self.device)
            tokenizer = model_tmp.tokenizer
            print(f"Using device: {self.device}")

            # 3. Extract passages (token-based chunking)
            corpus = self._extract_passages(dataset, tokenizer)
            
            # 4. Remove duplicates
            before = len(corpus)
            corpus = list(dict.fromkeys(corpus))
            after = len(corpus)
            print(f"Removed {before-after} duplicate passages ({after} unique).")

            # 5. Save passages separately (FAISS uses its own binary)
            with open(config.passages_file, "wb") as f:
                pickle.dump({"passages": corpus}, f)
            print(f"Saved passages to {config.passages_file}")

        # 2. Load model for embeddings
        model = SentenceTransformer(config.embedding_model, device=self.device)
        print(f"Using device: {self.device}")

        # 3. Compute embeddings in batches
        corpus_embeddings = []
        for i in tqdm(range(0, len(corpus), self.batch_size), desc="Computing embeddings"):
            batch = corpus[i:i+self.batch_size]
            emb = model.encode(batch, convert_to_numpy=True, device=self.device)
            corpus_embeddings.append(emb)
        corpus_embeddings = np.vstack(corpus_embeddings)

        # 4. Build FAISS index (inner product on L2-normalized vectors)
        norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        emb_normalized = corpus_embeddings / norms
        dim = emb_normalized.shape[1]
        index = faiss.IndexFlatIP(dim) # type: ignore
        index.add(emb_normalized.astype(np.float32))
        faiss.write_index(index, config.faiss_index_file) # type: ignore
        print(f"Saved FAISS index to {config.faiss_index_file} (dim={dim}, n={len(corpus)})")

        return corpus, corpus_embeddings
