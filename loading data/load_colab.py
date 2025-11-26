from datasets import load_dataset, Dataset, DownloadConfig
import os

def download_and_save_triviaqa(base_dir="/content/drive/MyDrive/rag-matthias",
                                batch_size=1000,
                                val_size=7900):
    """
    Downloads TriviaQA (rc.wikipedia) and saves train, validation, and test sets in shards.

    Args:
        base_dir (str): Base folder where everything will be saved.
        batch_size (int): Number of examples per shard.
        val_size (int): Number of examples to use for validation from the train split.
    """

    # Hugging Face cache directory
    cache_dir = os.path.join(base_dir, "hf_cache")
    download_config = DownloadConfig(cache_dir=cache_dir)

    # Paths for datasets
    train_dir = os.path.join(base_dir, "train_dataset")
    val_dir = os.path.join(base_dir, "validation_dataset")
    test_dir = os.path.join(base_dir, "test_dataset")

    # Ensure directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Function to save streamed data in shards
    def save_stream_to_disk_shards(streamed_data, base_path, batch_size=batch_size, prefix="shard"):
        buffer = []
        shard_idx = 0
        for i, example in enumerate(streamed_data):
            buffer.append(example)
            if len(buffer) >= batch_size:
                shard_path = os.path.join(base_path, f"{prefix}_{shard_idx}")
                os.makedirs(shard_path, exist_ok=True)
                Dataset.from_list(buffer).save_to_disk(shard_path)
                buffer = []
                shard_idx += 1
                print(f"Saved shard {shard_idx} with {batch_size} examples to {shard_path}")
        if buffer:
            shard_path = os.path.join(base_path, f"{prefix}_{shard_idx}")
            os.makedirs(shard_path, exist_ok=True)
            Dataset.from_list(buffer).save_to_disk(shard_path)
            print(f"Saved final shard {shard_idx} with {len(buffer)} examples to {shard_path}")

    # Streaming the train set
    train_val_stream = load_dataset(
        "trivia_qa",
        name="rc.wikipedia",
        split="train",
        streaming=True,
        download_config=download_config
    )

    # Split and save train/validation on the fly
    train_buffer = []
    val_buffer = []
    train_shard_idx = 0
    val_shard_idx = 0

    for i, example in enumerate(train_val_stream):
        if i < val_size:
            val_buffer.append(example)
            if len(val_buffer) >= batch_size:
                shard_path = os.path.join(val_dir, f"shard_{val_shard_idx}")
                os.makedirs(shard_path, exist_ok=True)
                Dataset.from_list(val_buffer).save_to_disk(shard_path)
                print(f"Saved validation shard {val_shard_idx}")
                val_buffer = []
                val_shard_idx += 1
        else:
            train_buffer.append(example)
            if len(train_buffer) >= batch_size:
                shard_path = os.path.join(train_dir, f"shard_{train_shard_idx}")
                os.makedirs(shard_path, exist_ok=True)
                Dataset.from_list(train_buffer).save_to_disk(shard_path)
                print(f"Saved train shard {train_shard_idx}")
                train_buffer = []
                train_shard_idx += 1
        if i % 1000 == 0:
            print(f"Overall processed {i} examples...")

    # Save remaining buffers
    if val_buffer:
        shard_path = os.path.join(val_dir, f"shard_{val_shard_idx}")
        os.makedirs(shard_path, exist_ok=True)
        Dataset.from_list(val_buffer).save_to_disk(shard_path)
        print(f"Saved final validation shard {val_shard_idx}")

    if train_buffer:
        shard_path = os.path.join(train_dir, f"shard_{train_shard_idx}")
        os.makedirs(shard_path, exist_ok=True)
        Dataset.from_list(train_buffer).save_to_disk(shard_path)
        print(f"Saved final train shard {train_shard_idx}")

    # Streaming validation/test set
    test_stream = load_dataset(
        "trivia_qa",
        name="rc.wikipedia",
        split="validation",
        streaming=True,
        download_config=download_config
    )
    save_stream_to_disk_shards(test_stream, test_dir)
