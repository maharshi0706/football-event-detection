import torch
import struct

import sys
sys.path.append("..")
import ML.config as config

import numpy as np
import random
import torchvision.transforms as T
from collections import defaultdict
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tensorflow.core.example.example_pb2 import Example

def build_offset_index(tfrecord_path):
    offsets = []
    with open(tfrecord_path, "rb") as f:
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

def read_record_at(tfrecord_path, offset, length):
    with open(tfrecord_path, "rb") as f:
        f.seek(offset)
        return f.read(length)
    

def parse_example(raw_bytes):
    example = Example()
    example.ParseFromString(raw_bytes)
    result = {}
    for key, feature in example.features.feature.items():
        kind = feature.WhichOneof("kind")
        if kind == "bytes_list":
            result[key] = feature.bytes_list.value[0]
        elif kind == "int64_list":
            result[key] = list(feature.int64_list.value)
        elif kind == "float_list":
            result[key] = list(feature.float_list.value)
    return result



class VideoAugment:
    def __init__(self):
        self.color_jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)

    def __call__(self, video):
        if random. random() > 0.5:
            video = torch.flip(video, dims=[3])

        brightness = random.uniform(0.6, 1.4)
        contrast = random.uniform(0.7, 1.3)
        video = torch.clamp(video * brightness, 0, 1)
        video = torch.clamp((video - 0.5) * contrast + 0.5, 0, 1)

        T, C, H, W = video.shape

        crop_size = random.randint(int(0.75 * H), H)
        top = random.randint(0, H - crop_size)
        left = random.randint(0, W - crop_size)
        video = video[:, :, top:top+crop_size, left:left+crop_size]
        video = torch.nn.functional.interpolate(
            video, size=(224, 224), mode="bilinear", align_corners=False
        )

        if random.random() > 0.8:
            gray = video.mean(dim=1, keepdim=True).expand_as(video)
            video = gray

        return video
        
        # i, j, h, w = T.RandomResizedCrop.get_params(
        #     video[0], scale=(0.6, 1.0), ratio=(0.75, 1.33)
        # )
        # video = torch.stack([TF.resized_crop(f, i, j, h, w, (224, 224,)) for f in video])

        # video = torch.stack([self.color_jitter(f) for f in video])

        # return video

# def get_weighted_sampler(dataset):
#     class_counts = defaultdict(int)

#     for _, label in dataset:
#         class_counts[label.item()] += 1

#     weights = [1.0 / class_counts[label.item()] for _, label in dataset]
#     return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

def get_weighted_sampler(dataset):
    import tensorflow as tf

    class_counts = defaultdict(int)
    labels = []


    # Use TF dataset to read only labels — no video loaded into RAM
    tf_dataset = tf.data.TFRecordDataset(
        str(dataset.tfrecord_path),
        buffer_size=64 * 1024 * 1024
    ).apply(tf.data.experimental.ignore_errors())

    for i, raw in enumerate(tf_dataset):
        parsed = tf.io.parse_single_example(
            raw, {"label": tf.io.FixedLenFeature([], tf.int64)}
        )
        label_id = int(parsed["label"].numpy())
        labels.append(label_id)
        class_counts[label_id] += 1
        if (i + 1) % 1000 == 0:
            print(f"  Scanned {i+1} records...", end="\r")

    print(f"\nDone. Total: {len(labels)} records.")
    weights = [1.0 / class_counts[l] for l in labels]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)



class FootballTFRecordDataset(Dataset):
    def __init__(self, tfrecord_path, transform=None):
        self.tfrecord_path = tfrecord_path
        self.transform = transform

        self.offsets = build_offset_index(tfrecord_path)

    def __len__(self):
        return len(self.offsets)
    
    def __getitem__(self, index):
        offset, length = self.offsets[index]

        raw_bytes = read_record_at(self.tfrecord_path, offset, length)
        record = parse_example(raw_bytes)

        video_bytes = record["video"]
        num_bytes = 16 * 224 * 224 * 3
        raw_pixels = video_bytes[-num_bytes:]
        video = np.frombuffer(raw_pixels, dtype=np.uint8).copy()
        video = video.reshape(16, 224, 224, 3)

        video = torch.from_numpy(video).float().div(255.0)
        video = video.permute(0, 3, 1, 2)

        label = torch.tensor(record["label"][0], dtype=torch.long)

        # sample = {"video":video, "label": label}

        if self.transform:
            video = self.transform(video)

        return video, label
    
def get_dataloaders():
    train_ds = FootballTFRecordDataset(config.TRAIN_TFRECORD_PATH, transform=VideoAugment())
    sampler = get_weighted_sampler(dataset=train_ds)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        # shuffle=True,  # Does not work with sampler
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=config.PREFETCH_FACTOR,
        persistent_workers=True
    )
    
    val_loader = None
    if config.VAL_TFRECORD_PATH.exists():
        val_ds = FootballTFRecordDataset(config.VAL_TFRECORD_PATH, transform=None)
        val_loader = DataLoader(
            val_ds,
            batch_size=config.BATCH_SIZE,
            # shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            prefetch_factor=config.PREFETCH_FACTOR,
            persistent_workers=True
        )
    
    return train_loader, val_loader