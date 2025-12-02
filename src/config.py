import sys
import os

class Config():
    def __init__(self, 
                 base_dir: str,
                 hf_cache_dir: str,
                 data_dir: str,
                 train_dir: str,
                 val_dir: str,
                 test_dir: str,
                 embeddings_file: str,
                 faiss_index_file: str,
                 passages_file: str,
                 embedding_model: str,
                 rerank_model: str,
                 generator_model: str) -> None:
        self.base_dir = base_dir
        self.hf_cache_dir = os.path.join(self.base_dir, hf_cache_dir)
        self.data_dir     = os.path.join(self.base_dir, data_dir)
        self.train_dir    = os.path.join(self.data_dir, train_dir)
        self.val_dir      = os.path.join(self.data_dir, val_dir)
        self.test_dir     = os.path.join(self.data_dir, test_dir)
        self.embeddings_file = os.path.join(self.data_dir, embeddings_file)
        self.faiss_index_file = os.path.join(self.data_dir, faiss_index_file)
        self.passages_file =  os.path.join(self.data_dir, passages_file)
        self.embedding_model = embedding_model
        self.rerank_model = rerank_model
        self.generator_model = generator_model
        self.shard_prefix = "shard_"

    def ensure_dirs(self):
        for p in [self.hf_cache_dir, self.data_dir, self.train_dir, self.val_dir, self.test_dir]:
            os.makedirs(p, exist_ok=True)
            print(f"✅ Ensured directory exists: {p}")

class ColabConfig(Config):
    def __init__(self,
                 base_dir="/content/drive/MyDrive/rag-matthias",
                 hf_cache_dir=".hf_cache",
                 data_dir="data",
                 train_dir="train",
                 val_dir="validation",
                 test_dir="test",
                 embeddings_file="corpus_embeddings_unique.pkl",
                 faiss_index_file="corpus_faiss.index",
                 passages_file="corpus_passages.pkl",
                 embedding_model="all-MiniLM-L6-v2",
                 rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                 generator_model="google/flan-t5-large") -> None:
        from google.colab import drive # type: ignore
        drive.mount("/content/drive")
        super().__init__(base_dir=base_dir,
                         hf_cache_dir=hf_cache_dir,
                         data_dir=data_dir,
                         train_dir=train_dir,
                         val_dir=val_dir,
                         test_dir=test_dir,
                         embeddings_file=embeddings_file,
                         faiss_index_file=faiss_index_file,
                         passages_file=passages_file,
                         embedding_model=embedding_model,
                         rerank_model=rerank_model,
                         generator_model=generator_model)

class LocalConfig(Config):
    def __init__(self,
                 base_dir=os.getcwd(),
                 hf_cache_dir=".hf_cache",
                 data_dir="data",
                 train_dir="train",
                 val_dir="validation",
                 test_dir="test",
                 embeddings_file="corpus_embeddings_unique.pkl",
                 faiss_index_file="corpus_faiss.index",
                 passages_file="corpus_passages.pkl",
                 embedding_model="all-MiniLM-L6-v2",
                 rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                 generator_model="google/flan-t5-large") -> None:
        super().__init__(base_dir=base_dir,
                         hf_cache_dir=hf_cache_dir,
                         data_dir=data_dir,
                         train_dir=train_dir,
                         val_dir=val_dir,
                         test_dir=test_dir,
                         embeddings_file=embeddings_file,
                         faiss_index_file=faiss_index_file,
                         passages_file=passages_file,
                         embedding_model=embedding_model,
                         rerank_model=rerank_model,
                         generator_model=generator_model)

class OllamaConfig(Config):
    def __init__(self,
                 base_dir=os.getcwd(),
                 hf_cache_dir=".hf_cache",
                 data_dir="data",
                 train_dir="train",
                 val_dir="validation",
                 test_dir="test",
                 embeddings_file="corpus_embeddings_unique.pkl",
                 faiss_index_file="corpus_faiss.index",
                 passages_file="corpus_passages.pkl",
                 embedding_model="all-MiniLM-L6-v2",
                 generator_model="llama3.1:8b",
                 rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                 ollama_url="http://127.0.0.1:11434/api/chat") -> None:
        super().__init__(base_dir=base_dir,
                         hf_cache_dir=hf_cache_dir,
                         data_dir=data_dir,
                         train_dir=train_dir,
                         val_dir=val_dir,
                         test_dir=test_dir,
                         embeddings_file=embeddings_file,
                         faiss_index_file=faiss_index_file,
                         passages_file=passages_file,
                         embedding_model=embedding_model,
                         rerank_model=rerank_model,
                         generator_model=generator_model)
        self.ollama_url = ollama_url
    

def is_colab():
    return "google.colab" in sys.modules

DEFAULT_CONFIG = LocalConfig()