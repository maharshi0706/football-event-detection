import torch
from pathlib import Path


# Dataset Version I
# TRAIN_TFRECORD_PATH = Path(r"D:\Football Highlight Generation\TFRecords\train.tfrecord")  
# VAL_TFRECORD_PATH = Path(r"D:\Football Highlight Generation\TFRecords\val.tfrecord")

# Dataset Version II
TRAIN_TFRECORD_PATH = Path(r"D:\Football Event Detection\Dataset\Data\train_v2.tfrecord")  
VAL_TFRECORD_PATH = Path(r"D:\Football Event Detection\Dataset\Data\val_v2.tfrecord")  

MODEL_NAME         = "MCG-NJU/videomae-base"

DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE         = 2
ACCUMULATION_STEPS = 16               # effective batch = 32
NUM_EPOCHS         = 20
LR                 = 3e-5
WEIGHT_DECAY       = 0.05
NUM_CLASSES        = 14              

SEED               = 42
NUM_WORKERS        = 4

CHECKPOINT_DIR = Path("checkpoints")
# CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_VERSION = Path("checkpoints/v5") # Training Version V
# TRAINING_VERSION.mkdir(exist_ok=True)

LOG_DIR        = Path("runs")

PREFETCH_FACTOR = 2