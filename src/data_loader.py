from datasets import load_dataset, Dataset, DownloadConfig
import os
from src.config import Config, DEFAULT_CONFIG
import faiss
import pickle
from pathlib import Path

class DataLoader:
    def __init__(self, config: Config):
        self.train_dir = config.train_dir
        self.val_dir = config.val_dir
        self.test_dir = config.test_dir
        self.cache_dir = config.hf_cache_dir
        self.faiss_index_file = config.faiss_index_file
        self.passages_file = config.passages_file
        self.val_split_size = config.val_split_size
        self.batch_size = config.shard_batch_size

    # ===== Internal Methods =====
    def _save_shard(self, buffer, base_dir, idx):
        shard_dir = os.path.join(base_dir, f"shard_{idx}")
        os.makedirs(shard_dir, exist_ok=True)
        Dataset.from_list(buffer).save_to_disk(shard_dir)
        print(f"Saved shard_{idx} → {shard_dir}")

    def _save_stream_to_disk_shards(self, stream, base_path):
        buf, idx = [], 0
        for ex in stream:
            buf.append(ex)
            if len(buf) >= self.batch_size:
                self._save_shard(buf, base_path, idx)
                buf, idx = [], idx + 1
        if buf:
            self._save_shard(buf, base_path, idx)

    def _download_and_prepare_data(self):
        download_config = DownloadConfig(cache_dir=self.cache_dir)

        # ---- STREAM TRAIN ----
        stream = load_dataset(
            "trivia_qa",
            name="rc.wikipedia",
            split="train",
            streaming=True,
            download_config=download_config
        )

        train_buffer, val_buffer = [], []
        train_idx = val_idx = 0

        for i, ex in enumerate(stream):
            if i < self.val_split_size:
                val_buffer.append(ex)
                if len(val_buffer) >= self.batch_size:
                    self._save_shard(val_buffer, self.val_dir, val_idx)
                    val_buffer, val_idx = [], val_idx + 1
            else:
                train_buffer.append(ex)
                if len(train_buffer) >= self.batch_size:
                    self._save_shard(train_buffer, self.train_dir, train_idx)
                    train_buffer, train_idx = [], train_idx + 1

            if i % 1000 == 0:
                print(f"Processed {i} samples...")

        if val_buffer:
            self._save_shard(val_buffer, self.val_dir, val_idx)
        if train_buffer:
            self._save_shard(train_buffer, self.train_dir, train_idx)

        # ---- STREAM TEST ----
        test_stream = load_dataset(
            "trivia_qa",
            name="rc.wikipedia",
            split="validation",
            streaming=True,
            download_config=download_config
        )
        self._save_stream_to_disk_shards(test_stream, self.test_dir)

        print("✔ Dataset download finished.")

    # ===== Public Methods =====
    def ensure_data_available(self):
        # 🔹 Ensure all required directories exist
        for d in [self.train_dir, self.val_dir, self.test_dir]:
            os.makedirs(d, exist_ok=True)

        # 🔹 Only download if train/val shards are missing
        if len(os.listdir(self.train_dir)) > 0 and len(os.listdir(self.val_dir)) > 0:
            print("✔ Dataset already downloaded — skipping.")
            return

        print("⬇ Downloading TriviaQA streaming dataset...")
        self._download_and_prepare_data()

    def load_existing_embeddings(self):
        """
        Load FAISS index + passages. Fail fast if artifacts are missing.
        """
        faiss_index_file = Path(self.faiss_index_file)
        passages_file = Path(self.passages_file)

        if not (faiss_index_file.exists() and passages_file.exists()):
            raise FileNotFoundError("FAISS index/passages missing. Run src.compute_embeddings to build them.")

        with open(passages_file, "rb") as f:
            data = pickle.load(f)
            passages = data["passages"]
        faiss_index = faiss.read_index(str(faiss_index_file)) # type: ignore
        print(f"🔹 Loaded FAISS index with {len(passages)} passages")
        return passages, faiss_index