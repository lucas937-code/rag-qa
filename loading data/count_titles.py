#!/usr/bin/env python3
import os
from collections import Counter
from datasets import load_from_disk, concatenate_datasets
import matplotlib.pyplot as plt

# ===========================
#     CONFIGURE PATHS
# ===========================
TRAIN_DIR = "../train_dataset/"
VAL_DIR   = "../validation_dataset/"
TEST_DIR  = "../test_dataset/"

OUTPUT_DIR = "title_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================
#     LOAD SHARDS
# ===========================
def load_shards(base_dir):
    """Load all dataset shards in a directory into one dataset."""
    if not os.path.exists(base_dir):
        print(f"[!] Directory not found: {base_dir}")
        return None

    all_shards = []
    for shard in sorted(os.listdir(base_dir)):
        shard_path = os.path.join(base_dir, shard)
        if os.path.isdir(shard_path):
            try:
                ds = load_from_disk(shard_path)
                all_shards.append(ds)
            except Exception as e:
                print(f"Failed to load {shard_path}: {e}")

    if not all_shards:
        return None

    return all_shards[0] if len(all_shards) == 1 else concatenate_datasets(all_shards)


# ===========================
#     TITLE COUNTING
# ===========================
def count_titles(ds, name):
    """Count how many documents each title appears in."""
    title_counter = Counter()

    for item in ds:
        titles = item.get("entity_pages", {}).get("title", [])
        if isinstance(titles, list):
            for t in set(titles):   # ensure unique per-document
                title_counter[t] += 1

    print(f"\n===== {name} Title Frequencies =====")
    for title, freq in title_counter.most_common():
        print(f"{title}: {freq}")

    return title_counter


# ===========================
#     PLOTTING UTILITIES
# ===========================
def plot_distribution(counter: Counter, name: str):
    counts = list(counter.values())

    # ---- Histogram for occurrence distribution ----
    plt.figure()
    plt.hist(counts, bins=30, edgecolor='black')
    plt.title(f"{name} Title Frequency Distribution")
    plt.xlabel("Number of documents title appears in")
    plt.ylabel("Count of titles")
    plt.savefig(f"{OUTPUT_DIR}/{name.lower()}_hist.png")
    plt.close()

    # ---- Bar plot of top 50 most frequent titles ----
    most_common = counter.most_common(50)
    labels, values = zip(*most_common)

    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)
    plt.xticks(rotation=90)
    plt.title(f"Top 50 Most Frequent Titles — {name}")
    plt.ylabel("Document count")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{name.lower()}_top50.png")
    plt.close()

    print(f"[+] Saved plots for {name} → {OUTPUT_DIR}/")


# ===========================
#           MAIN
# ===========================
def main():
    print("\nLoading datasets...")

    train_ds = load_shards(TRAIN_DIR)
    val_ds   = load_shards(VAL_DIR)
    test_ds  = load_shards(TEST_DIR)

    if train_ds:
        train_titles = count_titles(train_ds, "Train")
        plot_distribution(train_titles, "Train")

    if val_ds:
        val_titles = count_titles(val_ds, "Validation")
        plot_distribution(val_titles, "Validation")

    if test_ds:
        test_titles = count_titles(test_ds, "Test")
        plot_distribution(test_titles, "Test")

    print("\nDone. Plots saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
