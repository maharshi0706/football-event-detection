import torch
from pathlib import Path


TRAIN_TFRECORD_PATH = Path(r"E:\Football Highlight Generation\TFRecords Combined\train_merged_v1.tfrecord")
# TRAIN_TFRECORD_PATH = Path(r"D:\Football Highlight Generation\TFRecords\train.tfrecord")
VAL_TFRECORD_PATH = Path(r"D:\Football Highlight Generation\TFRecords\val.tfrecord")

MODEL_NAME         = "MCG-NJU/videomae-base"  # or "base" if VRAM allows

DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE         = 4
ACCUMULATION_STEPS = 8               # effective batch = 32
NUM_EPOCHS         = 20
LR                 = 3e-5
WEIGHT_DECAY       = 0.05
NUM_CLASSES        = 14              

SEED               = 42
NUM_WORKERS        = 4

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_VERSION = Path("checkpoints/v2") # Training Version II
TRAINING_VERSION.mkdir(exist_ok=True)

LOG_DIR        = Path("runs")

PREFETCH_FACTOR = 2