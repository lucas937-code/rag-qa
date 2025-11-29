import os
from collections import Counter
import pandas as pd
from datasets import load_from_disk, Value

MAX_LEN = 50  # Max length for string truncation in samples

def load_shards(base_dir, max_shards=None):
    """
    Load shards from a directory into a list of datasets.
    """
    shards = sorted([os.path.join(base_dir, d) for d in os.listdir(base_dir)
                     if os.path.isdir(os.path.join(base_dir, d))])
    if max_shards:
        shards = shards[:max_shards]
    datasets = [load_from_disk(shard) for shard in shards]
    return datasets

def truncate_nested(obj, max_len=MAX_LEN):
    """
    Recursively truncate strings inside dicts, lists, or other objects.
    """
    if isinstance(obj, str):
        return obj if len(obj) <= max_len else obj[:max_len] + "..."
    elif isinstance(obj, list):
        return [truncate_nested(x, max_len) for x in obj]
    elif isinstance(obj, dict):
        return {k: truncate_nested(v, max_len) for k, v in obj.items()}
    else:
        return obj

def explore_dataset(datasets, name="dataset", max_len=MAX_LEN):
    """
    Explore dataset(s): print structure, dimensionality, column types, sample data.
    """
    if not datasets:
        print(f"No shards found for {name}.")
        return

    print(f"\nExploring {name}:")
    total_examples = sum(len(ds) for ds in datasets)
    print(f"Total examples across all shards: {total_examples}")

    first_shard = datasets[0]
    print(f"Columns: {first_shard.column_names}")

    print("\nColumn types:")
    for col, feat in first_shard.features.items():
        print(f" - {col}: {feat}")

    print(f"\nSample data from first 3 examples (strings truncated to {max_len} chars):")
    for i, row in enumerate(first_shard):
        if i >= 3:
            break
        truncated_row = truncate_nested(row, max_len)
        print(truncated_row)

    # Detect text columns
    text_cols = [col for col, feat in first_shard.features.items() 
                 if isinstance(feat, Value) and feat.dtype == "string"]

    if text_cols:
        print("\nBasic text statistics (lengths in words) for columns:")
        for col in text_cols:
            lengths = [len(row[col].split()) for ds in datasets for row in ds]
            if lengths:
                lengths_series = pd.Series(lengths)
                print(f"Column '{col}': mean={lengths_series.mean():.2f}, min={lengths_series.min()}, max={lengths_series.max()}")

    # Show label distribution if exists
    label_col = None
    if "answer" in first_shard.column_names:
        label_col = "answer"
    elif "answers" in first_shard.column_names:
        label_col = "answers"

    if label_col:
        counter = Counter()
        for ds in datasets:
            for row in ds:
                val = row[label_col]
                if isinstance(val, dict):
                    key = val.get("normalized_value") or val.get("value") or str(val)
                    counter[key] += 1
                elif isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict):
                            key = item.get("normalized_value") or item.get("value") or str(item)
                            counter[key] += 1
                        else:
                            counter[item] += 1
                else:
                    counter[val] += 1
        print(f"\nTop 10 most common labels in '{label_col}':")
        print(counter.most_common(10))
