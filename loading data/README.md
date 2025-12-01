# TriviaQA Dataset Sharding - Exploration

This repository contains a sharded version of the **TriviaQA (Wikipedia RC)** dataset for memory-efficient processing and easy exploration.

## File Structure
- `load.py`: Downloads the structure as described below.
- `structure.py` Views an overview of the dataset dimensionality and datatypes. An output of this is saved in `structure.txt`
- `structure.txt`: Output of `structure.py`
- `explore.py`: Get some distribution information as well as sample questions with answers.
- `explore.txt`: Output of explore
- `save_embedings.py`: Save Embeding. Also gives out sample retreives for a given question.
- `save_embedings_chunking.py`: Now with chunking. Parameters can be specified.

Runtime approx. 1h40min on i7-8700 CPU @ 3.20GHz & NVIDIA GeForce RTX 2080 

- `rag.py`: Performs the generation part as well for a sample question. Note: Right now the performance is poor, possibly due to a to small model.
- `rag_lama.py`: In theory working rag with lama, but due to a lack of disk space i could not download it as for now (15gb).


**Personal thoughts at this point:**
- The retrieve part gives way too long results. -> smaller batch size. Important: Use overlap and always include the title
- I hope due to the smaller retrieve output, even the smaller generator model can perform a better answer.



# Explanation

---

## 1. Dataset Download and Streaming

* The dataset is downloaded and loaded using Hugging Face `datasets` in **streaming mode** to avoid memory overload:

```python
from datasets import load_dataset, DownloadConfig

cache_dir = "/proj/ciptmp/du69limo/to/rag-qa/hf_cache"
download_config = DownloadConfig(cache_dir=cache_dir)

train_val_stream = load_dataset(
    "trivia_qa",
    name="rc.wikipedia",
    split="train",
    streaming=True,
    download_config=download_config
)
```

* `streaming=True` ensures **examples are processed one at a time** without loading the entire dataset into memory.

---

## 2. Train / Validation Split

* The first 7,900 examples are used for **validation**.
* Remaining examples are used for **training**.
* Data is stored in **buffers** and saved to disk once the buffer reaches 1,000 examples.

---

## 3. Sharding and Disk Storage

* Data is saved in **shards** using Hugging Face `Dataset.save_to_disk`:

```python
Dataset.from_list(buffer).save_to_disk(shard_path)
```

* Each shard is a directory containing Arrow files and metadata (`dataset_info.json`).
* This allows **loading shards independently** without reading the entire dataset.

---

## 4. Directory Structure

After processing, the dataset is organized as follows:

```
../train_dataset/
    shard_0/
    shard_1/
    ...
../validation_dataset/
    shard_0/
    shard_1/
    ...
../test_dataset/
    shard_0/
    shard_1/
    ...
```

* **Shards** contain subsets of examples (default 1,000 per shard).
* You can load individual shards using:

```python
from datasets import load_from_disk
train_shard = load_from_disk("../train_dataset/shard_0")
```

---

## 5. Data Structure Overview

### Exploring Train set:

Total examples across all shards: 3000
Columns: ['question', 'question_id', 'question_source', 'entity_pages', 'search_results', 'answer']

Column types:

* question: Value('string')
* question_id: Value('string')
* question_source: Value('string')
* entity_pages: {'doc_source': List(Value('string')), 'filename': List(Value('string')), 'title': List(Value('string')), 'wiki_context': List(Value('string'))}
* search_results: {'description': List(Value('null')), 'filename': List(Value('null')), 'rank': List(Value('null')), 'search_context': List(Value('null')), 'title': List(Value('null')), 'url': List(Value('null'))}
* answer: {'aliases': List(Value('string')), 'matched_wiki_entity_name': Value('string'), 'normalized_aliases': List(Value('string')), 'normalized_matched_wiki_entity_name': Value('string'), 'normalized_value': Value('string'), 'type': Value('string'), 'value': Value('string')}

Sample data from first 3 examples (strings truncated to 50 chars):

*(omitted for brevity — see dataset)*

Basic text statistics (lengths in words) for columns:
Column 'question': mean=13.22, min=5, max=46
Column 'question_id': mean=1.00, min=1, max=1
Column 'question_source': mean=1.00, min=1, max=1

Top 10 most common labels in 'answer':
[('canada', 14), ('fish', 11), ('australia', 11), ('four', 10), ('red', 10), ('india', 10), ('france', 9), ('three', 9), ('two', 9), ('blue', 9)]

### Exploring Validation set:

Total examples across all shards: 3000
Columns: ['question', 'question_id', 'question_source', 'entity_pages', 'search_results', 'answer']

Column types: *(same as train set)*

Sample data from first 3 examples (strings truncated to 50 chars)

Basic text statistics (lengths in words) for columns:
Column 'question': mean=11.86, min=5, max=43
Column 'question_id': mean=1.00, min=1, max=1
Column 'question_source': mean=1.00, min=1, max=1

Top 10 most common labels in 'answer':
[('three', 8), ('ireland', 8), ('spain', 8), ('portugal', 7), ('norway', 6), ('italy', 6), ('6', 6), ('two', 6), ('chicago', 5), ('switzerland', 5)]

### Exploring Test set:

Total examples across all shards: 3000
Columns: ['question', 'question_id', 'question_source', 'entity_pages', 'search_results', 'answer']

Column types: *(same as train set)*

Sample data from first 3 examples (strings truncated to 50 chars)

Basic text statistics (lengths in words) for columns:
Column 'question': mean=13.32, min=5, max=55
Column 'question_id': mean=1.00, min=1, max=1
Column 'question_source': mean=1.00, min=1, max=1

Top 10 most common labels in 'answer':
[('france', 12), ('argentina', 9), ('three', 8), ('red', 8), ('australia', 8), ('spain', 8), ('new zealand', 7), ('switzerland', 7), ('scotland', 7), ('blue', 7)]


```python
{
    "question": "The medical condition glaucoma affects which part ...",
    "question_id": "qb_6735",
    "question_source": "http://www.quizballs.com/",
    "entity_pages": {
        "doc_source": ["TagMe", "TagMe", "TagMe"],
        "filename": ["Disease.txt", "Glaucoma.txt", "Human_body.txt"],
        "title": ["Disease", "Glaucoma", "Human body"],
        "wiki_context": [
            "A disease  is a particular abnormal condition, a d...",
            "Glaucoma is a group of eye diseases which result i...",
            "The human body is the entire structure of a human ...",
        ],
    },
    "search_results": {
        "description": [],
        "filename": [],
        "rank": [],
        "search_context": [],
        "title": [],
        "url": [],
    },
    "answer": {
        "aliases": [
            "Eye (anatomy)",
            "Eye",
            "Eye balls",
            "Schizochroal eye",
            "Ocular globe",
            "Ommateum",
            "Simple eye",
            "Oculars",
            "Animal eyes",
            "Eyes",
            "Compound Eyes",
            "Apposition eye",
            "Robotic eye",
            "Eye ball",
            "Facet eyes",
            "Compound Eye",
            "Conjunctival disorders",
            "Compound eyes",
            "Eyeball",
            "Cyber-eye",
            "Eye (vertebrate)",
            "Eye (invertebrate)",
            "Ommotidium",
            "Fly's eye lens",
            "Peeper (organ)",
            "Camera-type eye",
            "Ocular",
            "Compound eye",
            "Eye membrane",
            "Pinhole eye",
        ],
        "matched_wiki_entity_name": "",
        "normalized_aliases": [
            "compound eyes",
            "oculars",
            "facet eyes",
            "cyber eye",
            "ommateum",
            "peeper organ",
            "eye balls",
            "compound eye",
            "simple eye",
            "eye vertebrate",
            "animal eyes",
            "camera type eye",
            "eye membrane",
            "apposition eye",
            "eyeball",
            "eye",
            "ommotidium",
            "eyes",
            "robotic eye",
            "ocular globe",
            "eye anatomy",
            "pinhole eye",
            "eye ball",
            "conjunctival disorders",
            "fly s eye lens",
            "ocular",
            "schizochroal eye",
            "eye invertebrate",
        ],
        "normalized_matched_wiki_entity_name": "",
        "normalized_value": "eye",
        "type": "WikipediaEntity",
        "value": "Eye",
    },
}
```