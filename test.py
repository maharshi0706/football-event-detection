# # import struct
# # import torch
# # import numpy as np
# # from torch.utils.data import Dataset, DataLoader


# # def read_tfrecords(path):
# #     records = []
# #     with open(path, "rb") as f:
# #         while True:
# #             header = f.read(8)
# #             if len(header) < 8:
# #                 break
# #             length = struct.unpack("<Q", header)[0]
# #             f.read(4)                   # length crc
# #             data = f.read(length)
# #             f.read(4)                   # data crc
# #             if len(data) < length:
# #                 break
# #             records.append(data)
# #     return records


# # def parse_record(raw_bytes):
# #     """Parse tf.train.Example bytes using protobuf package only."""
# #     # Inline protobuf definition — no TF needed
# #     from google.protobuf import descriptor_pool, descriptor_pb2, symbol_database
# #     from google.protobuf import reflection, descriptor

# #     # Use a simple manual protobuf parse via struct
# #     # tf.train.Example is well-documented, parse it field by field
# #     result = {}

# #     # Use protobuf's Any parser on raw bytes
# #     # Easier: use the pre-compiled tf example proto from the protobuf package
# #     try:
# #         # Try using tensorflow_metadata if available
# #         from tensorflow.core.example.example_pb2 import Example
# #     except ImportError:
# #         # Fallback: minimal inline proto parsing
# #         from google.protobuf import descriptor_pb2
# #         pass

# #     example = Example()
# #     example.ParseFromString(raw_bytes)

# #     for key, feature in example.features.feature.items():
# #         kind = feature.WhichOneof("kind")
# #         if kind == "bytes_list":
# #             result[key] = feature.bytes_list.value[0]   # raw bytes
# #         elif kind == "int64_list":
# #             result[key] = list(feature.int64_list.value)
# #         elif kind == "float_list":
# #             result[key] = list(feature.float_list.value)

# #     return result


# # class TFRecordDataset(Dataset):
# #     """
# #     Pure PyTorch map-style TFRecord dataset.
# #     - No TensorFlow needed
# #     - Fully picklable → works with num_workers > 0 on Windows
# #     - Supports shuffle, random access
# #     """

# #     def __init__(self, tfrecord_path, transform=None):
# #         self.transform = transform
# #         print(f"Loading records from {tfrecord_path}...")
# #         self.raw_records = read_tfrecords(tfrecord_path)
# #         print(f"Loaded {len(self.raw_records)} records.")

# #     def __len__(self):
# #         return len(self.raw_records)

# #     def __getitem__(self, idx):
# #         record = parse_record(self.raw_records[idx])

# #         # ── Decode video (stored as raw float32 bytes) ──────────────────
# #         video_bytes = record["video"]
# #         video = np.frombuffer(video_bytes, dtype=np.float32).copy()
# #         # Reshape to your actual shape — adjust (16, 224, 224, 3) as needed
# #         video = video.reshape(16, 224, 224, 3)
# #         video = torch.from_numpy(video).permute(0, 3, 1, 2)  # → (T, C, H, W)

# #         # ── Decode label ─────────────────────────────────────────────────
# #         label = torch.tensor(record["label"][0], dtype=torch.long)

# #         sample = {"video": video, "label": label}

# #         if self.transform:
# #             sample = self.transform(sample)

# #         return sample

# # dataset = TFRecordDataset(
# #     tfrecord_path=r'E:\Football Highlight Generation\TFRecords Augmented\train_aug_20260306_2048.tfrecord',
# # )

# # print(f"Dataset size: {len(dataset)}")
# # sample = dataset[0]
# # print(f"video shape : {sample['video'].shape}")
# # print(f"label       : {sample['label']}")

# # # ✅ num_workers > 0 works on Windows with map-style dataset
# # dataloader = DataLoader(
# #     dataset,
# #     batch_size=4,
# #     num_workers=4,
# #     shuffle=True,
# #     prefetch_factor=2,
# #     persistent_workers=True,
# #     pin_memory=True
# # )

# # batch = next(iter(dataloader))
# # print(f"\nBatch video : {batch['video'].shape}")
# # print(f"Batch label : {batch['label']}")




# import struct
# import torch
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# from tensorflow.core.example.example_pb2 import Example


# def build_offset_index(tfrecord_path):
#     """
#     Instead of loading all records into RAM, store (offset, length) 
#     for each record. Reads from disk only when __getitem__ is called.
#     """
#     offsets = []
#     with open(tfrecord_path, "rb") as f:
#         while True:
#             start = f.tell()
#             header = f.read(8)
#             if len(header) < 8:
#                 break
#             length = struct.unpack("<Q", header)[0]
#             f.read(4)           # skip length crc
#             f.seek(length, 1)   # skip data
#             f.read(4)           # skip data crc
#             # store: where the data starts, how long it is
#             offsets.append((start + 8 + 4, length))
#     return offsets


# def read_record_at(tfrecord_path, offset, length):
#     """Read a single record's raw bytes from disk by offset."""
#     with open(tfrecord_path, "rb") as f:
#         f.seek(offset)
#         return f.read(length)


# def parse_example(raw_bytes):
#     """Parse tf.train.Example protobuf bytes."""
#     example = Example()
#     example.ParseFromString(raw_bytes)
#     result = {}
#     for key, feature in example.features.feature.items():
#         kind = feature.WhichOneof("kind")
#         if kind == "bytes_list":
#             result[key] = feature.bytes_list.value[0]
#         elif kind == "int64_list":
#             result[key] = list(feature.int64_list.value)
#         elif kind == "float_list":
#             result[key] = list(feature.float_list.value)
#     return result


# class TFRecordDataset(Dataset):
#     """
#     Memory-efficient, pure PyTorch map-style TFRecord dataset.

#     - Stores only byte offsets in RAM (not the data itself)
#     - Reads each record from disk on demand in __getitem__
#     - Fully picklable → supports num_workers > 0 on Windows
#     - Videos stored as tf.io.serialize_tensor(uint8) are decoded correctly
#     """

#     def __init__(self, tfrecord_path, transform=None):
#         self.tfrecord_path = tfrecord_path
#         self.transform = transform

#         print(f"Indexing {tfrecord_path} ...")
#         self.offsets = build_offset_index(tfrecord_path)
#         print(f"Found {len(self.offsets)} records.  "
#               f"RAM used: ~{len(self.offsets) * 16 / 1e6:.1f} MB for index.")

#     def __len__(self):
#         return len(self.offsets)

#     def __getitem__(self, idx):
#         offset, length = self.offsets[idx]

#         # Read only this one record from disk
#         raw_bytes = read_record_at(self.tfrecord_path, offset, length)
#         record = parse_example(raw_bytes)

#         # ── Video ────────────────────────────────────────────────────────
#         # Stored as tf.io.serialize_tensor(uint8 tensor of shape 16,224,224,3)
#         # tf.io.serialize_tensor prepends a small header before the raw bytes.
#         # We strip it by finding the actual pixel data (16*224*224*3 bytes).
#         video_bytes = record["video"]
#         num_bytes = 16 * 224 * 224 * 3           # exact expected size
#         raw_pixels = video_bytes[-num_bytes:]     # strip tf tensor header
#         video = np.frombuffer(raw_pixels, dtype=np.uint8).copy()
#         video = video.reshape(16, 224, 224, 3)
#         # Normalize to float32 [0, 1] and convert to (T, C, H, W)
#         video = torch.from_numpy(video).float().div(255.0)
#         video = video.permute(0, 3, 1, 2)        # → (16, 3, 224, 224)

#         # ── Label ────────────────────────────────────────────────────────
#         label = torch.tensor(record["label"][0], dtype=torch.long)

#         sample = {"video": video, "label": label}

#         if self.transform:
#             sample = self.transform(sample)

#         return sample


# if __name__ == "__main__":
#     TFRECORD_PATH = r'E:\Football Highlight Generation\TFRecords Augmented\train_aug_20260306_2048.tfrecord'

#     dataset = TFRecordDataset(tfrecord_path=TFRECORD_PATH)

#     print(f"\nDataset size : {len(dataset)}")
#     sample = dataset[0]
#     print(f"video shape  : {sample['video'].shape}")   # (16, 3, 224, 224)
#     print(f"video dtype  : {sample['video'].dtype}")   # float32
#     print(f"video range  : [{sample['video'].min():.3f}, {sample['video'].max():.3f}]")
#     print(f"label        : {sample['label']}")

#     # ✅ num_workers > 0 works — only offsets (ints) are pickled, not data
#     dataloader = DataLoader(
#         dataset,
#         batch_size=4,
#         num_workers=4,
#         shuffle=True,
#         prefetch_factor=2,
#         persistent_workers=True,
#         pin_memory=True
#     )

#     batch = next(iter(dataloader))
#     print(f"\nBatch video  : {batch['video'].shape}")  # (4, 16, 3, 224, 224)
#     print(f"Batch label  : {batch['label']}")



from collections import Counter
from dataset import FootballTFRecordDataset
import config

ds = FootballTFRecordDataset(config.TRAIN_TFRECORD_PATH)

labels = []
for i in range(len(ds)):
    _, label = ds[i]
    labels.append(label.item())

counts = Counter(labels)
total = len(labels)

print(f"Total samples: {total}")
print(f"Num classes: {len(counts)}\n")
for cls_id, count in sorted(counts.items()):
    print(f"  Class {cls_id}: {count:5d} samples  ({100*count/total:.1f}%)")