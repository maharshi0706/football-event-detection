import torch
from pathlib import Path


TRAIN_TFRECORD_PATH = Path(r"E:\Football Highlight Generation\TFRecords Combined\train_merged_v1.tfrecord")
VAL_TFRECORD_PATH = Path(r"D:\Football Highlight Generation\TFRecords\val.tfrecord")

MODEL_NAME         = "MCG-NJU/videomae-base-finetuned-kinetics"  # or "base" if VRAM allows

DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE         = 4
ACCUMULATION_STEPS = 8               # effective batch = 32
NUM_EPOCHS         = 1
LR                 = 1e-5
WEIGHT_DECAY       = 0.05
NUM_CLASSES        = 14              # Update to your real number

SEED               = 42
NUM_WORKERS        = 0

CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR        = Path("runs")