# RAG QA System

A comprehensive Retrieval-Augmented Generation (RAG) question-answering system implemented in Python. This project demonstrates advanced RAG architecture with configurable embedding models, reranking, and multiple generation backends.

## 🎯 Project Overview

This RAG system was developed as part of an academic assignment to build a complete question-answering pipeline. The implementation focuses on modular design, performance optimization, and flexibility across different deployment environments.

### Key Features

- **Modular Architecture**: Separate components for embedding, retrieval, reranking, and generation
- **Multiple Deployment Options**: Local, Google Colab, and Ollama support
- **Advanced Reranking**: Cross-encoder reranking for improved retrieval accuracy
- **Efficient Chunking**: Token-based text chunking with configurable overlap
- **Comprehensive Evaluation**: EM and F1 scoring for both retrieval and end-to-end performance

## 🏗️ Architecture

```
Query → Embedding → FAISS Retrieval → Cross-Encoder Reranking → LLM Generation → Answer
```

### Core Components

- **Embedder**: `BAAI/bge-base-en` model with FAISS indexing (768-dim vectors)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` for relevance scoring
- **Generator**: Configurable between `google/flan-t5-large` (HF) and `llama3.1:8b` (Ollama)
- **Chunking**: Token-based with 240 tokens per chunk, 60 token overlap

### Technical Implementation Details

#### Data Processing Pipeline
- **Dataset**: TriviaQA `rc.wikipedia` with streaming download for memory efficiency
- **Sharded Storage**: Data saved in 1000-example shards for scalable loading
- **Validation Split**: First 7900 samples from training set used for validation
- **Passage Extraction**: Entity pages with title-prefixed chunking (`"Title: chunk_content"`)

#### Embedding & Retrieval System
- **FAISS Index**: Inner Product (IP) search on L2-normalized vectors
- **Batch Processing**: 512 passages per embedding batch for GPU efficiency
- **Duplicate Removal**: Python dict-based deduplication preserving order
- **Caching Strategy**: Separate passage cache and embedding cache for incremental updates

#### Two-Stage Retrieval Architecture
1. **FAISS Retrieval**: Top 100 candidates using cosine similarity
2. **Cross-Encoder Reranking**: Relevance scoring on query-passage pairs
3. **Final Selection**: Top 5 passages passed to generator

#### Generation Strategies
- **HuggingFace**: Auto-detection of encoder-decoder vs. decoder-only models
- **Ollama Integration**: RESTful API calls with configurable timeouts
- **Prompt Engineering**: Context-aware QA prompts with length constraints
- **Generation Parameters**: Deterministic decoding (no sampling) with 48-token limit

## 📊 Implementation Decisions

### Model Selection Rationale

| Component | Model Choice | Reasoning |
|-----------|--------------|-----------|
| **Embedding** | `BAAI/bge-base-en` | Balanced performance (768-dim) with strong retrieval capabilities |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Lightweight yet effective for passage re-ranking |
| **Generator** | `google/flan-t5-large` / `llama3.1:8b` | T5 for accessibility, Llama for local deployment |

### Key Design Decisions

1. **Chunk Size**: 240 tokens with 60 token overlap (25% overlap ratio)
   - Balances context preservation with embedding efficiency
   - Prevents information loss at chunk boundaries
   - Title prefixing maintains entity context in each chunk

2. **FAISS Indexing**: Inner Product search on L2-normalized vectors
   - Enables fast similarity search on ~978K passages
   - Normalization ensures cosine similarity equivalence
   - Memory-efficient binary format for large-scale deployment

3. **Two-Stage Retrieval Architecture**: 
   - FAISS retrieval (top 100 candidates) for speed
   - Cross-encoder reranking (top 5 final passages) for accuracy
   - Reduces computational cost while maintaining precision

4. **Streaming Data Processing**:
   - Memory-efficient handling of large datasets
   - Sharded storage enables incremental processing
   - Automatic deduplication prevents redundant embeddings

5. **Flexible Configuration System**:
   - Environment-specific configs (Local/Colab/Ollama)
   - Abstract base classes enable easy model swapping
   - Device-aware model loading (CUDA/CPU)

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Basic Usage

1. **Configure the system** (first cell in `pipeline.ipynb`):
```python
from src.config import LocalConfig, OllamaConfig

# Local setup
config = LocalConfig()

# Or Ollama setup (requires running Ollama server)
config = OllamaConfig(ollama_url="http://localhost:11434/api/chat")
```

2. **Run the complete pipeline**:
```bash
jupyter notebook pipeline.ipynb
```

The notebook includes:
- Data loading and preprocessing
- Embedding computation and indexing
- Retrieval and generation testing
- Performance evaluation

## 📈 Performance Results

**🚩🚩TODO MAKE CONSISTENT WITH POSTER DEMONSTRATION**

### Test Set Evaluation (1000 questions)
- **Exact Match (EM)**: 65.0%
- **F1 Score**: 75.86%

### Dataset Statistics
- **Corpus Size**: ~978K passages
- **Training/Validation/Test Split**: Configurable (default 7900 validation samples)
- **Average Question Length**: Analyzed in `plots/` directory

## 🔧 Configuration Options

### Key Parameters

```python
config.chunk_tokens = 240        # Chunk size in tokens
config.chunk_overlap = 60        # Overlap between chunks
config.embeddings_batch_size = 512  # Embedding computation batch size
config.val_split_size = 7900     # Validation set size
```

### Model Choices

```python
config.embedding_model = "BAAI/bge-base-en"
config.rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
config.generator_model = "google/flan-t5-large"  # or "llama3.1:8b"
```

## 🔬 Evaluation Methods

### Retrieval Evaluation
- **FAISS Performance**: Cosine similarity scoring with normalized vectors
- **Reranking Effectiveness**: Cross-encoder relevance scoring on query-passage pairs
- **Top-k Analysis**: Retrieval accuracy at different k values (5, 10, 20, 100)

### End-to-End RAG Evaluation
- **Exact Match (EM)**: Strict string matching after normalization
- **F1 Score**: Token-level overlap with comprehensive normalization:
  - Lowercasing and punctuation removal
  - Stop word removal ("a", "an", "the")
  - Whitespace normalization
- **Evaluation Pipeline**:
  ```python
  # Normalization pipeline
  normalize_answer() → tokenize() → calculate_precision_recall() → compute_f1()
  ```

### Dataset Splits
- **Training**: Remaining samples after validation split
- **Validation**: First 7900 samples from original training set
- **Test**: TriviaQA validation set (used for final evaluation)
- **Shard-based Loading**: Enables memory-efficient evaluation on large datasets

## 🌟 Advanced Features

### Ollama Integration
- Local LLM deployment with `llama3.1:8b`
- Reduced dependency on external APIs
- Configurable host/port settings

### Efficient Data Handling
- **Sharded Dataset Loading**: 1000-example shards for memory efficiency
- **Streaming Downloads**: TriviaQA dataset processed without full memory loading
- **Incremental Caching**: Separate passage and embedding caches for flexible updates
- **Batch Processing**: Configurable batch sizes for embedding computation (default: 512)
- **Device Optimization**: Automatic CUDA detection with fallback to CPU
- **FAISS Integration**: Binary index format for fast similarity search at scale

## 📁 Project Structure

```
rag-qa/
├── src/                    # Core implementation
│   ├── config.py          # Configuration management (Local/Colab/Ollama)
│   ├── embedder.py        # Embedding computation + FAISS indexing
│   ├── retriever.py       # FAISS-based document retrieval
│   ├── reranker.py        # Cross-encoder passage reranking
│   ├── generator.py       # LLM generation (HF + Ollama backends)
│   ├── data_loader.py     # Streaming data processing + sharding
│   ├── evaluate_rag_full.py # End-to-end RAG evaluation
│   ├── evaluate_retrieve.py # Retrieval-only evaluation
│   ├── analyze_data.py    # Dataset analysis and visualization
│   ├── explore_data.py    # Data exploration utilities
│   └── data_prep/         # Dataset loading and preprocessing
│       └── load_dataset.py # TriviaQA streaming download
├── pipeline.ipynb         # Main pipeline notebook
├── plots/                 # Data analysis visualizations
│   ├── *_question_length_distribution.png
│   └── *_answer_length_distribution.png
├── requirements.txt       # Python dependencies
└── notes.md              # Development notes and ideas
```