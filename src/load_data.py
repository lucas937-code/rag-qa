from datasets import load_dataset, Dataset, DownloadConfig
import os
from src.config import Config, DEFAULT_CONFIG

VAL_SPLIT_SIZE = 7900
BATCH_SIZE = 1000

def save_shard(buffer, base_dir, idx):
    shard_dir = os.path.join(base_dir, f"shard_{idx}")
    os.makedirs(shard_dir, exist_ok=True)
    Dataset.from_list(buffer).save_to_disk(shard_dir)
    print(f"Saved shard_{idx} → {shard_dir}")

def save_stream_to_disk_shards(stream, base_path, batch_size=BATCH_SIZE):
    buf, idx = [], 0
    for ex in stream:
        buf.append(ex)
        if len(buf) >= batch_size:
            save_shard(buf, base_path, idx)
            buf, idx = [], idx + 1
    if buf:
        save_shard(buf, base_path, idx)

def download_and_prepare_data(config: Config):
    download_config = DownloadConfig(cache_dir=config.HF_CACHE_DIR)

    # ---- STREAM TRAIN ----
    stream = load_dataset(
        "trivia_qa",
        name="rc.wikipedia",
        split="train",
        streaming=True,
        download_config=download_config
    )

    train_buffer, val_buffer = [], []
    train_idx = val_idx = 0

    for i, ex in enumerate(stream):
        if i < VAL_SPLIT_SIZE:
            val_buffer.append(ex)
            if len(val_buffer) >= BATCH_SIZE:
                save_shard(val_buffer, config.VAL_DIR, val_idx)
                val_buffer, val_idx = [], val_idx + 1
        else:
            train_buffer.append(ex)
            if len(train_buffer) >= BATCH_SIZE:
                save_shard(train_buffer, config.TRAIN_DIR, train_idx)
                train_buffer, train_idx = [], train_idx + 1

        if i % 1000 == 0:
            print(f"Processed {i} samples...")

    if val_buffer:
        save_shard(val_buffer, config.VAL_DIR, val_idx)
    if train_buffer:
        save_shard(train_buffer, config.TRAIN_DIR, train_idx)

    # ---- STREAM TEST ----
    test_stream = load_dataset(
        "trivia_qa",
        name="rc.wikipedia",
        split="validation",
        streaming=True,
        download_config=download_config
    )
    save_stream_to_disk_shards(test_stream, config.TEST_DIR)

    print("✔ Dataset download finished.")

def ensure_data_available(config: Config = DEFAULT_CONFIG):
    # 🔹 Ensure all required directories exist
    for d in [config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR]:
        os.makedirs(d, exist_ok=True)

    # 🔹 Only download if train/val shards are missing
    if len(os.listdir(config.TRAIN_DIR)) > 0 and len(os.listdir(config.VAL_DIR)) > 0:
        print("✔ Dataset already downloaded — skipping.")
        return

    print("⬇ Downloading TriviaQA streaming dataset...")
    download_and_prepare_data(config)
