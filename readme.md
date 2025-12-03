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

- **Embedder**: `BAAI/bge-base-en` model with FAISS indexing
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` for relevance scoring
- **Generator**: Configurable between `google/flan-t5-large` (HF) and `llama3.1:8b` (Ollama)
- **Chunking**: Token-based with 240 tokens per chunk, 60 token overlap

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

2. **FAISS Indexing**: Enables fast similarity search on ~1M passages
   - Supports efficient retrieval at scale
   - Reduces memory footprint compared to brute-force search

3. **Two-Stage Retrieval**: 
   - Initial FAISS retrieval (top 100 candidates)
   - Cross-encoder reranking (top 5 final passages)
   - Improves accuracy while maintaining performance

4. **Flexible Configuration System**:
   - Environment-specific configs (Local/Colab/Ollama)
   - Easy parameter tuning for experiments

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
- Top-k accuracy metrics
- Relevance scoring with cross-encoder

### End-to-End RAG Evaluation
- **Exact Match (EM)**: Strict string matching
- **F1 Score**: Token-level overlap with normalization:
  - Lowercasing and punctuation removal
  - Stop word removal ("a", "an", "the")
  - Whitespace normalization

## 🌟 Advanced Features

### Ollama Integration
- Local LLM deployment with `llama3.1:8b`
- Reduced dependency on external APIs
- Configurable host/port settings

### Efficient Data Handling
- Sharded dataset loading for memory efficiency
- Lazy loading of large embeddings
- Batch processing for embedding computation