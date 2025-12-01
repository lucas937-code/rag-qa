import os
from collections import Counter
import matplotlib.pyplot as plt
from datasets import load_from_disk, concatenate_datasets

# Directory to save plots (project-root relative)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Load all shards and concatenate ---
def load_shards_concat(base_dir):
    """Load all shards from a directory and concatenate into one dataset."""
    all_shards = []
    for shard_name in sorted(os.listdir(base_dir)):
        shard_path = os.path.join(base_dir, shard_name)
        if os.path.isdir(shard_path):
            ds = load_from_disk(shard_path)
            all_shards.append(ds)
    if not all_shards:
        return None
    if len(all_shards) == 1:
        return all_shards[0]
    return concatenate_datasets(all_shards)

# --- Dataset info ---
def dataset_info(ds, name):
    print(f"\n{name} dataset info:")
    print(f"Number of examples: {len(ds)}")
    print(f"Features: {ds.features}")
    print("Sample example:")
    print(ds[0])

# --- Analyze question/answer lengths ---
def analyze_lengths(ds, field, name):
    lengths = []
    for x in ds:
        value = x.get(field, "")
        if isinstance(value, list):
            value = value[0] if value else ""
            if isinstance(value, dict) and "value" in value:
                value = value["value"]
        elif isinstance(value, dict) and "value" in value:
            value = value["value"]
        elif not isinstance(value, str):
            value = str(value)
        lengths.append(len(value.split()))

    avg_len = sum(lengths) / len(lengths)
    print(f"{name} {field} average length: {avg_len:.2f} words")

    # Plot histogram
    plt.figure(figsize=(6,4))
    plt.hist(lengths, bins=30, color='skyblue', edgecolor='black')
    plt.title(f"{name} {field} length distribution")
    plt.xlabel("Number of words")
    plt.ylabel("Frequency")
    plt.tight_layout()

    # Save plot to PLOTS_DIR
    filename = os.path.join(PLOTS_DIR, f"{name.lower()}_{field}_length_distribution.png")
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot to {filename}")

# --- Most common answers ---
def most_common_answers(ds, top_k=10):
    answers = []
    for x in ds:
        val = x.get('answer', [])
        if isinstance(val, list):
            answers.extend(val)
        elif isinstance(val, dict):
            if 'aliases' in val:
                answers.extend(val['aliases'])
            elif 'value' in val:
                answers.append(val['value'])
            else:
                answers.append(str(val))
        else:
            answers.append(str(val))
    counter = Counter(answers)
    print(f"\nTop {top_k} most common answers:")
    for ans, freq in counter.most_common(top_k):
        print(f"{ans}: {freq}")

# --- Print sample Q&A ---
def print_sample_qa(ds, name, n=20):
    print(f"\n{name} dataset sample Q&A:")
    examples = list(ds)[:n]
    for i, item in enumerate(examples):
        # Extract question
        question = item.get('question', "")
        if isinstance(question, list):
            question = question[0] if question else ""
            if isinstance(question, dict) and "value" in question:
                question = question["value"]
        elif isinstance(question, dict) and "value" in question:
            question = question["value"]

        # Extract answers (first 3)
        answers = item.get('answer', [])
        first3 = []
        if isinstance(answers, dict) and 'aliases' in answers:
            first3 = answers['aliases'][:3]
        elif isinstance(answers, list):
            first3 = answers[:3]
        answer_text = ", ".join(first3)
        print(f"{i+1}. {question}")
        print(answer_text)
        print()