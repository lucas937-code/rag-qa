# Data Preparation Overview

*(TriviaQA → Flattened Parquet → Deterministic Splits)*

This document provides an overview of the **data preparation stage** used in our RAG project.
It describes:

* how the raw TriviaQA dataset is loaded
* how nested structures are flattened into a Parquet-friendly schema
* how full splits are stored in Google Drive (for the whole team)
* how deterministic train/validation splits are produced
* how developers can locally inspect small samples

The relevant implementation can be found in:

* **`src/data_prep/load_dataset.py`** 
* **`src/data_prep/split_dataset.py`** 

---

# 1. Goals of the Data Preparation Stage

The main goals of this component are:

1. **Load the raw TriviaQA `rc.wikipedia` dataset**

   * either fully (in Colab, using Google Drive for persistence), or
   * partially (locally, for debugging).

2. **Flatten the dataset**
   Convert deeply nested TriviaQA rows into a *simple, tabular format* that can be stored as Parquet without errors.

3. **Produce deterministic splits**
   Create a validation set of the *first 7,900* examples and use the rest as training data.

4. **Ensure reproducibility and team-wide accessibility**
   All large data artifacts (raw Parquet files, splits) live in a shared Google Drive folder.

---

# 2. Loading the Raw Dataset (`load_dataset.py`)

The entry points are:

### **`prepare_full_dataset(...)`**

Used in Colab to download the *entire* dataset and save each split as a Parquet file.
This function automatically applies flattening when `dataset_name="trivia_qa"` and `subset="rc.wikipedia"`.

### **`get_local_sample(...)`**

Used locally on a laptop to load only a small subset (e.g. 300 examples).
Also flattening-aware.

Internally, loading is performed via `datasets.load_dataset(...)`, with optional streaming support.

---

# 3. Why Flattening Is Required

TriviaQA’s raw structure contains **heavily nested fields**, for example:

```python
{
  "question": "...",
  "answer": {
      "aliases": [...],
      "normalized_aliases": [...]
  },
  "entity_pages": {
      "title": [...],
      "filename": [...],
      "wiki_context": [...]
  },
  "search_results": {... huge nested structure ...},
  "evidence": {... nested lists/dicts ...}
}
```

Storing this directly as a Parquet file causes PyArrow to throw errors such as:

```
ArrowNotImplementedError: Nested data conversions not implemented
```

This happens because:

* Parquet does not support arbitrary Python objects (dicts, lists of dicts, etc.)
* Some TriviaQA fields are *dicts of lists*, others are *lists of dicts*
* Structures like `search_results` and `evidence` are deeply nested and irregular

Therefore, we flatten the dataset into a normalized, Parquet-friendly format.

---

# 4. Flattening: How It Works

Flattening is implemented in:

### **`_flatten_triviaqa_row(row)`** in `load_dataset.py` 

Its purpose is to extract **only the fields needed for our RAG pipeline**, while removing or simplifying everything else.

## 4.1 What the *raw* format looks like (simplified)

```python
{
  "question_id": "tc_3",
  "question": "Where in England was Dame Judi Dench born?",
  "answer": {
      "aliases": [...],
      "normalized_aliases": [...]
  },
  "entity_pages": {
      "title": ["England", "Judi Dench"],
      "wiki_context": [
          "...long text about England...",
          "...long text about Judi Dench..."
      ]
  },
  "search_results": {
      "rank": [...],
      "filename": [...],
      "search_context": [...]
  },
  "evidence": {...}
}
```

This is **not storable** in Parquet as-is.

---

## 4.2 What the *flattened* format looks like

After flattening, each row becomes a *fully scalar* record:

```python
{
  "question_id": "tc_3",
  "question": "Where in England was Dame Judi Dench born?",

  "answer_aliases_json": "[\"Park Grove School\", \"York, England\", ...]",
  "answer_normalized_json": "[\"york england\", \"park grove school\", ...]",

  "doc_titles_json": "[\"England\", \"Judi Dench\"]",

  "evidence_text": "England is a country... \n\n Judi Dench is an English actress ..."
}
```

### Key differences:

* **No nested structures** (all columns are plain strings or None)
* Lists (e.g., aliases, titles) become **JSON strings**
* Multiple evidence passages become **a single concatenated text string**
* Unneeded structures like `search_results` and `evidence` are discarded

This format is:

* Parquet-safe
* Easy to chunk and embed
* Reproducible
* Consistent across the team

---

## 4.3 What fields are retained (important for RAG)

We keep:

* `question_id`
* `question`
* `answer_aliases_json`
* `answer_normalized_json`
* `doc_titles_json`
* `evidence_text` (the large text that will later be chunked and embedded)

We lose:

* All annotation/debugging structures (`search_results`, nested `evidence`)
* Extra provenance metadata not relevant for retrieval

This is intentional: our RAG pipeline only needs the question, answer aliases (for evaluation), and the evidence texts (for retrieval).

---

# 5. Saving the Flattened Dataset to Parquet

In `prepare_full_dataset(...)`, each split is converted via:

### **`_save_split_to_parquet(...)`**

* Applies flattening automatically for TriviaQA rc.wikipedia
* Writes a clean Parquet file
* Stores it in the chosen Google Drive directory

The resulting files are large (several hundred MBs to multiple GBs) because they contain the full Wikipedia evidence texts.

---

# 6. Deterministic Train/Validation Split (`split_dataset.py`)

Once the full raw training set is flattened and stored as Parquet, the file is passed to:

### **`create_train_val_split(...)`** 

This function:

1. Loads the flattened raw training data
2. Sorts and resets indices
3. Creates a deterministic validation set of size **7,900**
4. Writes:

   * `train.parquet`
   * `val_7900.parquet`
   * `splits_meta.json`

We use the `"first_n"` strategy by default, matching the project requirements.

---

# 7. Local Inspection

### **`get_local_sample(...)`**

Allows quick local exploration by loading a small number of flattened examples.

### **`get_local_split_preview(...)`**

Loads the first *k* rows of an existing split for inspection.

Both are useful for development without needing the full dataset.