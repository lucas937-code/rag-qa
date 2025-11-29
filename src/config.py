import sys
import os

class Config():
    def __init__(self, 
                 base_dir: str,
                 hf_cache_dir=".hf_cache",
                 data_dir="data",
                 train_dir="data/train",
                 val_dir="data/validation",
                 test_dir="data/test",
                 embeddings_file="corpus_embeddings_unique.pkl",
                 faiss_index_file="corpus_faiss.index",
                 passages_file="corpus_passages.pkl",
                 embedding_model="all-MiniLM-L6-v2") -> None:
        self.BASE_DIR = base_dir
        self.HF_CACHE_DIR = os.path.join(self.BASE_DIR, hf_cache_dir)
        self.DATA_DIR     = os.path.join(self.BASE_DIR, data_dir)
        self.TRAIN_DIR    = os.path.join(self.DATA_DIR, train_dir)
        self.VAL_DIR      = os.path.join(self.DATA_DIR, val_dir)
        self.TEST_DIR     = os.path.join(self.DATA_DIR, test_dir)
        self.EMBEDDINGS_FILE = os.path.join(self.BASE_DIR, embeddings_file)
        self.FAISS_INDEX_FILE = os.path.join(self.BASE_DIR, faiss_index_file)
        self.PASSAGES_FILE =  os.path.join(self.BASE_DIR, passages_file)
        self.EMBEDDING_MODEL = embedding_model
        self.SHARD_PREFIX = "shard_"

    def ensure_dirs(self):
        for p in [self.HF_CACHE_DIR, self.DATA_DIR, self.TRAIN_DIR, self.VAL_DIR, self.TEST_DIR]:
            os.makedirs(p, exist_ok=True)
            print(f"✅ Ensured directory exists: {p}")

class ColabConfig(Config):
    def __init__(self, base_dir="/content/drive/MyDrive/rag-matthias") -> None:
        from google.colab import drive # type: ignore
        drive.mount("/content/drive")
        super().__init__(base_dir=base_dir)

class LocalConfig(Config):
    def __init__(self) -> None:
        super().__init__(os.getcwd())
    

def is_colab():
    return "google.colab" in sys.modules

DEFAULT_CONFIG = LocalConfig()