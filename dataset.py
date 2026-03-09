# # import torch
# # from transformers import AutoImageProcessor
# # from torch.utils.data import Dataset, DataLoader
# # import tensorflow as tf
# # import numpy as np
# # from albumentations import Compose, HorizontalFlip, RandomBrightnessContrast, ShiftScaleRotate, GaussNoise
# # import config

# # class FootballTFRecordDataset(Dataset):
# #     def __init__(self, tfrecord_path, processor, augment=False):
# #         self.dataset = tf.data.TFRecordDataset(str(tfrecord_path))
# #         self.processor = processor
# #         self.augment = augment
# #         self.length = sum(1 for _ in self.dataset)  # expensive — can cache later if needed

# #         # Augmentation pipeline (applied per frame)
# #         self.transform = Compose([
# #             HorizontalFlip(p=0.5),
# #             RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
# #             ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=8, p=0.3),
# #             GaussNoise(var_limit=(5, 25), p=0.2),
# #         ]) if augment else None

# #     def __len__(self):
# #         return self.length

# #     def __getitem__(self, idx):
# #         # Sequential access (acceptable for ~28k items)
# #         for i, raw in enumerate(self.dataset):
# #             if i == idx:
# #                 try:
# #                     example = tf.io.parse_single_example(raw, {
# #                         'video': tf.io.FixedLenFeature([], tf.string),
# #                         'label': tf.io.FixedLenFeature([], tf.int64),
# #                     })
# #                     video_bytes = example['video']
# #                     label = int(example['label'])

# #                     # Decode to uint8 numpy (16, 224, 224, 3)
# #                     video_np = tf.io.decode_raw(video_bytes, tf.uint8).numpy()
# #                     expected_size = 16 * 224 * 224 * 3

# #                     if len(video_np) != expected_size:
# #                         print(f"Skipping bad clip at index {i} (size {len(video_np)} != {expected_size})")
# #                         # Return a dummy zero clip (or raise to skip batch)
# #                         return torch.zeros((3, 16, 224, 224)), label  # dummy
                    
# #                     video_np = np.reshape(video_np, (16, 224, 224, 3))

# #                     # Apply augmentation if enabled (on uint8 images)
# #                     if self.transform:
# #                         aug_video = np.zeros_like(video_np, dtype=np.float32)
# #                         for f in range(16):
# #                             aug_frame = self.transform(image=video_np[f])['image']
# #                             aug_video[f] = aug_frame.astype(np.float32) / 255.0
# #                         video_np = aug_video  # already float [0,1]
# #                     else:
# #                         video_np = video_np.astype(np.float32) / 255.0  # normalize to [0,1]

# #                     # Transpose to (C, T, H, W) — required by timm models
# #                     video_np = np.transpose(video_np, (3, 0, 1, 2))

# #                     frames_list = [frame for frame in video_np]
# #                     inputs = self.processor(frames_list, return_tensors="pt")
# #                     pixel_values = inputs['pixel_values'].squeeze(0)

# #                     return pixel_values, label
                    
# #                     # # To torch tensor
# #                     # video_tensor = torch.from_numpy(video_np).permute(3, 0, 1, 2).float()

# #                     # return video_tensor, label
                
# #                 except Exception as e:
# #                     print(f"Parse error at index {i}: {e}")
# #                     return torch.zeros((3, 16, 224, 224), dtype=torch.float32), label  # dummy

# #         raise IndexError("Index out of range")

# # def get_dataloaders():
# #     processor = AutoImageProcessor.from_pretrained(config.MODEL_NAME)

# #     train_ds = FootballTFRecordDataset(config.TRAIN_TFRECORD_PATH, processor, augment=True)
# #     train_loader = DataLoader(
# #         train_ds,
# #         batch_size=config.BATCH_SIZE,
# #         shuffle=True,
# #         num_workers=config.NUM_WORKERS,
# #         pin_memory=True
# #     )
    
# #     val_loader = None
# #     if config.VAL_TFRECORD_PATH.exists():
# #         val_ds = FootballTFRecordDataset(config.VAL_TFRECORD_PATH, processor, augment=False)
# #         val_loader = DataLoader(
# #             val_ds,
# #             batch_size=config.BATCH_SIZE,
# #             shuffle=False,
# #             num_workers=config.NUM_WORKERS,
# #             pin_memory=True
# #         )
    
# #     return train_loader, val_loader



# import torch
# from torch.utils.data import Dataset, DataLoader
# import tensorflow as tf
# import numpy as np
# from albumentations import Compose, HorizontalFlip, RandomBrightnessContrast, ShiftScaleRotate, GaussNoise
# import config

# class FootballTFRecordDataset(Dataset):
#     def __init__(self, tfrecord_path, augment=False):
#         self.dataset = tf.data.TFRecordDataset(str(tfrecord_path))
#         self.augment = augment
#         self.length = sum(1 for _ in self.dataset)

#         self.transform = Compose([
#             HorizontalFlip(p=0.5),
#             RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
#             ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=8, p=0.3),
#             GaussNoise(var_limit=(5, 25), p=0.2),
#         ]) if augment else None

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         for i, raw in enumerate(self.dataset):
#             if i == idx:
#                 try:
#                     example = tf.io.parse_single_example(raw, {
#                         'video': tf.io.FixedLenFeature([], tf.string),
#                         'label': tf.io.FixedLenFeature([], tf.int64),
#                     })
#                     video_bytes = example['video']
#                     label = int(example['label'])

#                     video_np = tf.io.decode_raw(video_bytes, tf.uint8).numpy()
#                     expected_size = 16 * 224 * 224 * 3

#                     if len(video_np) != expected_size:
#                         print(f"Skipping bad clip at index {i} (size {len(video_np)} != {expected_size})")
#                         return torch.zeros((3, 16, 224, 224), dtype=torch.float32), label

#                     video_np = np.reshape(video_np, (16, 224, 224, 3))  # (T, H, W, C)

#                     # Augment if enabled
#                     if self.transform:
#                         aug_video = np.zeros_like(video_np, dtype=np.float32)
#                         for f in range(16):
#                             aug_frame = self.transform(image=video_np[f])['image']
#                             aug_video[f] = aug_frame / 255.0
#                         video_np = aug_video
#                     else:
#                         video_np = video_np.astype(np.float32) / 255.0

#                     # Transpose to (C, T, H, W) — VideoMAE expects channels first
#                     video_np = np.transpose(video_np, (3, 0, 1, 2))

#                     video_tensor = torch.from_numpy(video_np).float()

#                     return video_tensor, label

#                 except Exception as e:
#                     print(f"Error at index {i}: {e}")
#                     return torch.zeros((3, 16, 224, 224), dtype=torch.float32), label

#         raise IndexError("Index out of range")

# def get_dataloaders():
#     train_ds = FootballTFRecordDataset(config.TRAIN_TFRECORD_PATH, augment=True)
#     train_loader = DataLoader(
#         train_ds,
#         batch_size=config.BATCH_SIZE,
#         shuffle=True,
#         num_workers=0,  # Safe for TF on Windows
#         pin_memory=True
#     )
    
#     val_loader = None
#     if config.VAL_TFRECORD_PATH.exists():
#         val_ds = FootballTFRecordDataset(config.VAL_TFRECORD_PATH, augment=False)
#         val_loader = DataLoader(
#             val_ds,
#             batch_size=config.BATCH_SIZE,
#             shuffle=False,
#             num_workers=0,
#             pin_memory=True
#         )
    
#     return train_loader, val_loader


# dataset.py (final version – no processor, channels-first for VideoMAE)
import torch
import config
import numpy as np
import tensorflow as tf
from transformers import AutoImageProcessor
from torch.utils.data import Dataset, DataLoader
from tfrecord.torch.dataset import TFRecordDataset as TorchTFRecordDataset
from albumentations import Compose, HorizontalFlip, RandomBrightnessContrast, ShiftScaleRotate, GaussNoise

class FootballTFRecordDataset(Dataset):
    def __init__(self, tfrecord_path, processor, augment=False):
        self.dataset = tf.data.TFRecordDataset(str(tfrecord_path))
        self.processor = processor
        self.augment = augment
        self.length = sum(1 for _ in self.dataset)

        self.transform = Compose([
            HorizontalFlip(p=0.5),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=8, p=0.3),
            GaussNoise(var_limit=(5, 25), p=0.2),
        ]) if augment else None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        for i, raw in enumerate(self.dataset):
            if i == idx:
                try:
                    example = tf.io.parse_single_example(raw, {
                        'video': tf.io.FixedLenFeature([], tf.string),
                        'label': tf.io.FixedLenFeature([], tf.int64),
                    })
                    video_bytes = example['video']
                    label = int(example['label'])

                    video_np = tf.io.decode_raw(video_bytes, tf.uint8).numpy()
                    expected_size = 16 * 224 * 224 * 3

                    if len(video_np) != expected_size:
                        print(f"Skipping bad clip at index {i} (size {len(video_np)} != {expected_size})")
                        return torch.zeros((16, 3, 224, 224), dtype=torch.float32), label

                    video_np = np.reshape(video_np, (16, 3, 224, 224)) 

                    # Augment if enabled (on uint8)
                    # if self.transform:
                    #     aug_video = np.zeros_like(video_np, dtype=np.float32)
                    #     for f in range(16):
                    #         aug_frame = self.transform(image=video_np[f])['image']
                    #         aug_video[f] = aug_frame 
                    #     video_np = (aug_video * 255).astype(np.uint8)
                    #     # video_np = aug_video
                    # else:
                    #     pass
                        # video_np = video_np.astype(np.float32) / 255.0

                    # frames_list = [frame for frame in video_np]

                    inputs = self.processor(video_np, return_tensors="pt")
                    pixel_values = inputs['pixel_values'].squeeze(0)
                    return pixel_values, label
                    # # Transpose to (C, T, H, W) – VideoMAE expects channels first
                    # video_np = np.transpose(video_np, (3, 0, 1, 2))

                    # video_tensor = torch.from_numpy(video_np).float()

                    # return video_tensor, label

                except Exception as e:
                    print(f"Error at index {i}: {e}")
                    return torch.zeros((3, 16, 224, 224), dtype=torch.float32), label

        raise IndexError("Index out of range")

def get_dataloaders():
    processor = AutoImageProcessor.from_pretrained(config.MODEL_NAME, do_rescale=False)

    train_ds = FootballTFRecordDataset(config.TRAIN_TFRECORD_PATH, processor, augment=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = None
    if config.VAL_TFRECORD_PATH.exists():
        val_ds = FootballTFRecordDataset(config.VAL_TFRECORD_PATH, processor, augment=False)
        val_loader = DataLoader(
            val_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
    
    return train_loader, val_loader