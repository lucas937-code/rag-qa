# RAG QA System

A comprehensive Retrieval-Augmented Generation (RAG) system for question answering built as an academic project. This implementation demonstrates key design decisions and evaluation approaches for modern QA systems.

## 🎯 Project Overview

This project implements a complete RAG pipeline that:
- Retrieves relevant passages from a large Wikipedia corpus using dense embeddings
- Reranks retrieved passages for better relevance
- Generates answers using a sequence-to-sequence model
- Evaluates both retrieval and end-to-end performance

## 🏗️ Architecture & Key Design Decisions

### **Embedding Model Choice: BAAI/bge-base-en**
We selected the BAAI/bge-base-en model over alternatives because:
- Superior performance on English retrieval tasks
- Good balance between model size and accuracy
- Strong semantic understanding for question-passage matching

### **Two-Stage Retrieval Pipeline**
Our system uses a **retrieve → rerank** approach:
1. **Initial Retrieval**: FAISS index with cosine similarity for fast approximate nearest neighbor search
2. **Cross-Encoder Reranking**: MS MARCO cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for improved relevance scoring

*Why this approach?* Initial retrieval provides speed and scalability, while reranking significantly improves precision without prohibitive computational cost.

### **Generator: FLAN-T5-Base**
Chosen for its:
- Strong instruction-following capabilities
- Efficient encoder-decoder architecture
- Good performance on extractive QA tasks

### **FAISS for Vector Search**
We use FAISS instead of alternatives because:
- Optimized for large-scale similarity search
- Memory-efficient indexing
- Fast query times even with ~1M passages

### **Smart Passage Chunking**
- **Token-based chunking**: 240 tokens per chunk with 60-token overlap
- Prevents encoder truncation while maintaining context
- Title-prefixed passages for better semantic understanding

## 📊 Dataset

**TriviaQA Wikipedia Dataset** - 9,000 examples across train/validation/test splits
- **Questions**: Real trivia questions from various sources
- **Context**: Wikipedia passages with entity annotations
- **Answers**: Wikipedia entities with multiple aliases

### Data Characteristics
- Average question length: 12-13 words
- Answers include multiple aliases for robust evaluation
- Disjoint train/validation/test splits for reliable evaluation
- **978,526 unique passages** after chunking and deduplication

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the Complete Pipeline
```bash
jupyter notebook pipeline.ipynb
```

The notebook walks through:
1. **Setup & Configuration** - Automatic environment detection (Local/Colab)
2. **Data Loading** - Downloads and processes TriviaQA dataset
3. **Embedding Computation** - Creates FAISS index and embeddings
4. **Retrieval Testing** - Tests passage retrieval with sample queries
5. **Generation** - End-to-end QA with retrieved context
6. **Evaluation** - Comprehensive performance metrics

## 📈 Performance Results

### Retrieval Performance (Recall@K)
| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@7 | Recall@10 |
|---------|----------|----------|----------|----------|-----------|
| Train   | 81.3%    | 90.5%    | 92.6%    | 93.9%    | 94.1%     |
| Validation | 72.6%  | 83.2%    | 85.8%    | 87.0%    | 88.2%     |
| Test    | 78.2%    | 87.9%    | 90.3%    | 91.4%    | 92.5%     |

### End-to-End QA Performance (Test Set, 100 samples)
- **Exact Match**: 60.0%
- **F1 Score**: 64.0%

## 📁 Project Structure

```
├── src/                    # Core implementation
│   ├── config.py          # Configuration management (Local/Colab)
│   ├── load_data.py       # Dataset downloading and streaming
│   ├── compute_embeddings.py  # Embedding computation and FAISS indexing
│   ├── generator.py       # Answer generation with retrieval & reranking
│   ├── evaluate_retrieve.py   # Retrieval evaluation with Recall@K
│   ├── evaluate_rag_full.py   # End-to-end evaluation (EM/F1)
│   ├── explore_data.py    # Data analysis utilities
│   └── analyze_data.py    # Statistical analysis and plotting
├── plots/                 # Data analysis visualizations
├── pipeline.ipynb         # Main pipeline notebook
├── fine_tune_embedding.ipynb  # Embedding fine-tuning experiments
└── notes.md              # Project notes and ideas
```

## 🔬 Evaluation Methodology

### **Retrieval Evaluation**
We use **Recall@K** to measure if the correct answer appears in the top-K retrieved passages. This directly measures the retriever's ability to find relevant context.

**Key Implementation Details:**
- **FAISS Candidates**: 50 initial candidates retrieved
- **Reranking**: Cross-encoder reorders candidates for better precision
- **Alias Matching**: Checks if any answer alias appears in retrieved passages
- **Evaluation Limit**: 1000 samples per split for efficient evaluation

### **QA Evaluation**
- **Exact Match (EM)**: Strict string matching after normalization
- **F1 Score**: Token-level overlap accounting for partial matches
- **Alias Handling**: Takes maximum score across all answer aliases

**Normalization Process**:
1. Lowercase conversion
2. Punctuation removal
3. Stop word removal ("a", "an", "the")
4. Whitespace normalization

## 🎨 Visualizations

The `plots/` directory contains length distribution analyses:
- Question length distributions across splits
- Answer length distributions
- Helps understand data characteristics and inform model choices

## 💡 Key Insights & Lessons Learned

### **What Worked Well**
- **Two-stage retrieval** significantly improves answer quality (72.6% → 88.2% Recall@10)
- **Cross-encoder reranking** worth the computational cost for precision
- **BGE embeddings** provide strong semantic matching out of the box
- **Token-based chunking** prevents information loss better than naive splitting

### **Challenges & Solutions**
- **Memory Management**: Used streaming data loading and batched embedding computation
- **Speed vs Accuracy**: Balanced FAISS candidates (50) with reranking for optimal performance
- **Answer Variations**: Implemented robust alias matching for fair evaluation
- **Model Detection**: Automatic encoder-decoder vs decoder-only detection in generator

### **Technical Decisions Explained**

#### **Why TriviaQA over Natural Questions?**
- More focused trivia-style questions better suited for RAG demonstration
- Cleaner entity-based answers with comprehensive aliases
- Smaller dataset size manageable for academic project

#### **Why 240-token chunks with 60-token overlap?**
- Fits within BGE's context window while preserving context
- Overlap ensures no information loss at chunk boundaries
- Empirically balanced retrieval quality vs computational cost

#### **Why MS MARCO cross-encoder?**
- Specifically trained for passage relevance
- Lightweight compared to larger alternatives
- Strong performance on question-passage matching

### **Future Improvements** (from `notes.md`)
- **Llama 3.1**: Experiment with local generation models
- **Few-shot prompting**: Improve generation quality with examples
- **Hybrid retrieval**: Combine dense and sparse embeddings
- **Domain fine-tuning**: Adapt embeddings to trivia domain
- **Advanced chunking**: Semantic-aware passage splitting

## 🛠️ Configuration

The system supports both local and Google Colab environments:
- **LocalConfig**: For local development with automatic paths
- **ColabConfig**: For Google Colab with Drive integration

**Key Configuration Options:**
```python
config = LocalConfig(
    embedding_model="BAAI/bge-base-en",  # Change embedding model
    base_dir="/path/to/project"         # Custom project path
)
```

Automatic environment detection ensures the right configuration is used.

## 📚 References

- **TriviaQA Dataset**: [Joshi et al., 2017]
- **BGE Embeddings**: [BAAI, 2023]
- **FAISS**: [Johnson et al., 2019]
- **FLAN-T5**: [Chung et al., 2022]
- **MS MARCO Cross-Encoder**: [Nogueira & Cho, 2019]
