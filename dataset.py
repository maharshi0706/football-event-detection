import torch
import struct
import config
import numpy as np
import random
import torchvision.transforms as T
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset, DataLoader
from tensorflow.core.example.example_pb2 import Example
# from tfrecord.torch.dataset import TFRecordDataset, MultiTFRecordDataset
# from albumentations import Compose, HorizontalFlip, RandomBrightnessContrast, ShiftScaleRotate, GaussNoise

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

        
        i, j, h, w = T.RandomResizedCrop.get_params(
            video[0], scale=(0.6, 1.0), ratio=(0.75, 1.33)
        )
        video = torch.stack([TF.resized_crop(f, i, j, h, w, (224, 224,)) for f in video])

        video = torch.stack([self.color_jitter(f) for f in video])

        return video
    
class FootballTFRecordDataset(Dataset):
    def __init__(self, tfrecord_path, transform=None):
        self.tfrecord_path = tfrecord_path
        self.transform = VideoAugment()

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
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
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