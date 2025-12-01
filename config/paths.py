import os
import sys

def running_in_colab():
    return "google.colab" in sys.modules

def get_base_dir():
    if running_in_colab():
        from google.colab import drive
        drive.mount("/content/drive")
        return "/content/drive/MyDrive/rag-matthias"
    return os.getcwd()

BASE_DIR = get_base_dir()

HF_CACHE_DIR = os.path.join(BASE_DIR, ".hf_cache")
DATA_DIR     = os.path.join(BASE_DIR, "data")
TRAIN_DIR    = os.path.join(DATA_DIR, "train")
VAL_DIR      = os.path.join(DATA_DIR, "validation")
TEST_DIR     = os.path.join(DATA_DIR, "test")

# 🔥 Ensure all required dirs exist before import
for p in [HF_CACHE_DIR, DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR]:
    os.makedirs(p, exist_ok=True)
    print(f"✅ Ensured directory exists: {p}")