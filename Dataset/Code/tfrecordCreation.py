import os
import cv2
import glob
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from pathlib import Path

CLIPS_DIR = Path(r"E:\Football Dataset\Event Clips Split\train")
OUTPUT    = Path(r"E:\Football Dataset\Event Clips Split\train_recovered.tfrecord")

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

class_names = sorted(os.listdir(CLIPS_DIR))
class_to_id = {cls: i for i, cls in enumerate(class_names)}
print("Classes:", class_to_id)

writer = tf.io.TFRecordWriter(str(OUTPUT))
total = 0

for cls in class_names:
    class_path = CLIPS_DIR / cls
    files = glob.glob(str(class_path / "*.mp4"))

    for f in tqdm(files, desc=cls):
        cap = cv2.VideoCapture(f)
        frames = []
        for _ in range(16):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        cap.release()

        if len(frames) == 16:
            frames_np = np.array(frames, dtype=np.uint8)
            video_raw = tf.io.serialize_tensor(
                tf.convert_to_tensor(frames_np, dtype=tf.uint8)
            ).numpy()
            feature = {
                "video": _bytes_feature(video_raw),
                "label": _int64_feature(class_to_id[cls])
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            total += 1

writer.close()
print(f"Written {total} records to {OUTPUT}")