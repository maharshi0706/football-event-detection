import struct
import random
from pathlib import Path
import tensorflow as tf

def get_offsets(path):
    offsets = []
    with open(path, "rb") as f:
        while True:
            start = f.tell()
            header = f.read(8)
            if len(header) < 8:
                break
            length = struct.unpack("<Q", header)[0]
            f.read(4)
            f.seek(length, 1)
            f.read(4)
            offsets.append((start + 8 + 4, length))
    return offsets

def read_at(path, offset, length):
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read(length)

INPUT     = Path(r"E:\Football Dataset\train_merged_v2.tfrecord")
TRAIN_OUT = Path(r"D:\Football Event Detection\Dataset\Data\train_v2.tfrecord")
VAL_OUT   = Path(r"D:\Football Event Detection\Dataset\Data\val_v2.tfrecord")

VAL_RATIO = 0.15
SEED      = 42
random.seed(SEED)

print("Indexing...")
offsets = get_offsets(INPUT)
print(f"Total: {len(offsets)} records")

indices = list(range(len(offsets)))
random.shuffle(indices)

split_idx   = int(len(indices) * VAL_RATIO)
val_idx     = indices[:split_idx]
train_idx   = indices[split_idx:]

print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")

train_writer = tf.io.TFRecordWriter(str(TRAIN_OUT))
for i, idx in enumerate(train_idx):
    offset, length = offsets[idx]
    train_writer.write(read_at(INPUT, offset, length))
    if (i + 1) % 1000 == 0:
        print(f"  Train: {i+1}/{len(train_idx)}", end="\r")
train_writer.close()

val_writer = tf.io.TFRecordWriter(str(VAL_OUT))
for i, idx in enumerate(val_idx):
    offset, length = offsets[idx]
    val_writer.write(read_at(INPUT, offset, length))
    if (i + 1) % 1000 == 0:
        print(f"  Val: {i+1}/{len(val_idx)}", end="\r")
val_writer.close()

print(f"\nDone.")