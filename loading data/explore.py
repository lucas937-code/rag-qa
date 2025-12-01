from collections import Counter
import matplotlib.pyplot as plt

from datasets import load_from_disk, concatenate_datasets
import os

# Paths to the dataset shards
TRAIN_DIR = "../train_dataset/"
VAL_DIR = "../validation_dataset/"
TEST_DIR = "../test_dataset/"

def load_shards(base_dir):
    """Load all shards from a directory into a single dataset."""
    all_shards = []
    for shard_name in sorted(os.listdir(base_dir)):
        shard_path = os.path.join(base_dir, shard_name)
        if os.path.isdir(shard_path):
            ds = load_from_disk(shard_path)
            all_shards.append(ds)
    if all_shards:
        if len(all_shards) > 1:
            return concatenate_datasets(all_shards)  # fixed
        else:
            return all_shards[0]
    return None

# Load datasets
train_ds = load_shards(TRAIN_DIR)
val_ds = load_shards(VAL_DIR)
test_ds = load_shards(TEST_DIR)

# Basic info
def dataset_info(ds, name):
    print(f"\n{name} dataset info:")
    print(f"Number of examples: {len(ds)}")
    print(f"Features: {ds.features}")
    print("Sample example:")
    print(ds[0])

dataset_info(train_ds, "Train")
dataset_info(val_ds, "Validation")
dataset_info(test_ds, "Test")

# Analyze question/answer lengths
def analyze_lengths(ds, field, name):
    lengths = []
    for x in ds:
        value = x[field]
        if isinstance(value, list):
            # Take the first element
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
    import matplotlib.pyplot as plt
    plt.hist(lengths, bins=30, color='skyblue', edgecolor='black')
    plt.title(f"{name} {field} length distribution")
    plt.xlabel("Number of words")
    plt.ylabel("Frequency")
    # save to file
    plt.savefig(f"{name.lower()}_{field}_length_distribution.png")

# analyze_lengths(train_ds, "question", "Train")
# analyze_lengths(train_ds, "answer", "Train")

# Most common answer types (optional)
def most_common_answers(ds, top_k=10):
    answers = [ans for x in ds for ans in x['answer']]
    counter = Counter(answers)
    print(f"Top {top_k} most common answers:")
    for ans, freq in counter.most_common(top_k):
        print(f"{ans}: {freq}")

# most_common_answers(train_ds)






def print_sample_qa(ds, name, n=20):
    print(f"\n{name} dataset sample Q&A:")
    examples = list(ds)[:n]  # first n examples
    for i, item in enumerate(examples):
        # Extract question
        question = item['question']
        if isinstance(question, list):
            question = question[0] if question else ""
            if isinstance(question, dict) and "value" in question:
                question = question["value"]
        elif isinstance(question, dict) and "value" in question:
            question = question["value"]

        # Extract answers (first 3)
        answers = item['answer']
        first3 = []
        if isinstance(answers, dict) and 'aliases' in answers:
            first3 = answers['aliases'][:3]  # first 3 aliases
        elif isinstance(answers, list):
            first3 = answers[:3]
        
        # Join answers by comma
        answer_text = ", ".join(first3)

        print(f"{i+1}. {question}")
        print(answer_text)
        print()

# Print for each split
print_sample_qa(train_ds, "Train")
print_sample_qa(val_ds, "Validation")
print_sample_qa(test_ds, "Test")
