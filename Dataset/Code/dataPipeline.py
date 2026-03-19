import re
import os
import json
import time
import shutil
import struct
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import defaultdict
from SoccerNet.utils import getListGames
from SoccerNet.Downloader import SoccerNetDownloader

# Paths
EXTERNAL_MATCHES_DIR = Path("D:\Football")
MATCHES_DIR          = Path("E:\Football Dataset\Match Footage")
ANNOTATIONS_DIR      = Path("E:\Football Dataset\Annotations")
OUTPUT_TFRECORD      = Path(r"E:\Football Dataset\TFRecords New\train_new.tfrecord")
OUTPUT_TFRECORD.parent.mkdir(parents=True, exist_ok=True)

STORAGE_THRESHOLD = 20 * (2**30)

CLASSES = [
    "Ball out of play", "Clearance", "Corner", "Direct free-kick",
    "Foul", "Goal", "Indirect free-kick", "Kick-off", "Offside",
    "Shots off target", "Shots on target", "Substitution",
    "Throw-in", "Yellow card"
]
CLASS_TO_ID = {cls: i for i, cls in enumerate(CLASSES)}

matchDownloader = SoccerNetDownloader(LocalDirectory=str(MATCHES_DIR))
matchDownloader.password = 's0cc3rn3t'

annotationDownloader = SoccerNetDownloader(LocalDirectory=str(ANNOTATIONS_DIR))
annotationDownloader.password = 's0cc3rn3t'


# ── TFRecord helpers ──────────────────────────────────────────────────────────

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def extract_frames(video_path, timestamp, num_frames=16, duration=6):
    """Extract num_frames evenly sampled from clip window around timestamp."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps        = cap.get(cv2.CAP_PROP_FPS) or 30
    total_secs = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    start_time = max(0, timestamp - 3)
    end_time   = min(start_time + duration, total_secs)

    timestamps = np.linspace(start_time, end_time, num_frames)
    frames = []
    for t in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return np.array(frames, dtype=np.uint8) if len(frames) == num_frames else None


def download_with_retry(downloader: SoccerNetDownloader, game, files, spl, retries=2, wait=10):
    """Download with automatic retry on network failure."""
    for attempt in range(retries):
        try:
            downloader.downloadGame(game, files=files, spl=spl)
            return True
        except KeyboardInterrupt:
            raise  
        except Exception as e:
            print(f"  Attempt {attempt+1}/{retries} failed: {e}")
            if attempt < retries - 1:
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
    print(f"Failed to download {game} after {retries} attempts, skipping.")
    return False
# ── Count existing records in single TFRecord ─────────────────────────────────

# def get_tfrecord_counts():
#     """
#     Count per-class records already written in the single TFRecord file.
#     Reads byte offsets only — no RAM loading.
#     """
#     counts = defaultdict(int)

#     if not OUTPUT_TFRECORD.exists():
#         return counts

#     # Need to parse labels to count per class
#     # We read each record and parse just the label field
#     import struct
#     from tensorflow.core.example.example_pb2 import Example

#     with open(OUTPUT_TFRECORD, "rb") as f:
#         while True:
#             header = f.read(8)
#             if len(header) < 8:
#                 break
#             length = struct.unpack("<Q", header)[0]
#             f.read(4)
#             data = f.read(length)
#             f.read(4)

#             example = Example()
#             example.ParseFromString(data)
#             label_id = example.features.feature["label"].int64_list.value[0]
#             # reverse lookup class name
#             cls = CLASSES[label_id]
#             counts[cls] += 1

#     print("Existing record counts in TFRecord:")
#     for cls in CLASSES:
#         print(f"  {cls:<25} {counts[cls]}")

#     return counts

# def get_tfrecord_counts():
#     counts = defaultdict(int)

#     if not OUTPUT_TFRECORD.exists() or OUTPUT_TFRECORD.stat().st_size == 0:
#         print("No existing TFRecord found, starting fresh.")
#         return counts

#     print(f"Reading existing counts from {OUTPUT_TFRECORD}...")
#     from tensorflow.core.example.example_pb2 import Example

#     try:
#         with open(OUTPUT_TFRECORD, "rb") as f:
#             while True:
#                 header = f.read(8)
#                 if len(header) < 8:
#                     break
#                 length = struct.unpack("<Q", header)[0]
#                 f.read(4)
#                 data = f.read(length)
#                 f.read(4)
#                 if len(data) < length:
#                     break    
#                 example = Example()
#                 example.ParseFromString(data)
#                 label_id = example.features.feature["label"].int64_list.value[0]
#                 counts[CLASSES[label_id]] += 1
#     except Exception as e:
#         print(f"Warning: error reading existing TFRecord ({e}), counts may be incomplete.")

#     print("Existing record counts:")
#     for cls in CLASSES:
#         print(f"  {cls:<25} {counts[cls]}")

#     return counts


def get_tfrecord_counts(tfrecord_paths):
    """
    Count per-class records across multiple TFRecord files/directories.
    
    Args:
        tfrecord_paths: list of Path objects — can be individual .tfrecord 
                        files or directories containing .tfrecord files
    """
    from tensorflow.core.example.example_pb2 import Example

    counts = defaultdict(int)
    files_read = []

    # Collect all .tfrecord files from the provided paths
    for path in tfrecord_paths:
        path = Path(path)
        if path.is_dir():
            found = list(path.glob("*.tfrecord"))
            files_read.extend(found)
        elif path.is_file() and path.suffix == ".tfrecord":
            files_read.append(path)
        else:
            print(f"Path not found or not a TFRecord: {path}")

    if not files_read:
        print("No existing TFRecord files found, starting fresh.")
        return counts

    print(f"Reading counts from {len(files_read)} TFRecord file(s):")
    for f in files_read:
        print(f"  {f}")

    for tfrecord_file in files_read:
        if tfrecord_file.stat().st_size == 0:
            print(f"Skipping empty file: {tfrecord_file}")
            continue

        file_counts = defaultdict(int)
        try:
            with open(tfrecord_file, "rb") as f:
                while True:
                    header = f.read(8)
                    if len(header) < 8:
                        break
                    length = struct.unpack("<Q", header)[0]
                    f.read(4)
                    data = f.read(length)
                    f.read(4)
                    if len(data) < length:
                        break
                    example = Example()
                    example.ParseFromString(data)
                    label_id = example.features.feature["label"].int64_list.value[0]
                    cls = CLASSES[label_id]
                    counts[cls] += 1
                    file_counts[cls] += 1

            total_in_file = sum(file_counts.values())
            print(f"{tfrecord_file.name}: {total_in_file} records")

        except Exception as e:
            print(f" Error reading {tfrecord_file}: {e}, skipping.")

    print("\nCombined counts across all files:")
    for cls in CLASSES:
        print(f"  {cls:<25} {counts[cls]}")
    print(f"  {'TOTAL':<25} {sum(counts.values())}")

    return counts

# ── Core helpers ──────────────────────────────────────────────────────────────

def get_disk_space():
    total, used, free = shutil.disk_usage("E:\\")
    return free // (2**30)

def match_exists(game):
    return (MATCHES_DIR / game).exists() or (EXTERNAL_MATCHES_DIR / game).exists()

def annotation_exists(game):
    return (ANNOTATIONS_DIR / game).exists()

def process_match(game, writer, counts, max_per_class=2000):
    """Extract frames from match video and write directly to single TFRecord."""
    annotation_file = ANNOTATIONS_DIR / game / "Labels-v2.json"
    if not annotation_file.exists():
        print(f"  No annotation file for {game}, skipping.")
        return

    with open(annotation_file) as f:
        data = json.load(f)

    for idx, annotation in enumerate(data["annotations"]):
        event_label = annotation["label"]

        if event_label not in CLASS_TO_ID:
            continue

        if counts[event_label] >= max_per_class:
            continue

        game_time  = annotation["gameTime"]
        half       = int(game_time.split("-", 1)[0].strip())
        video_file = str(MATCHES_DIR / game / f"{half}_720p.mkv")

        if not Path(video_file).exists():
            continue

        minutes, seconds = map(int, game_time.split("-", 1)[1].strip().split(":"))
        timestamp = minutes * 60 + seconds

        frames = extract_frames(video_file, timestamp)
        if frames is None:
            continue

        # Write to single TFRecord
        video_raw = tf.io.serialize_tensor(
            tf.convert_to_tensor(frames, dtype=tf.uint8)
        ).numpy()

        feature = {
            "video": _bytes_feature(video_raw),
            "label": _int64_feature(CLASS_TO_ID[event_label])
        }
        example = tf.train.Example(
            features=tf.train.Features(feature=feature)
        )
        writer.write(example.SerializeToString())
        counts[event_label] += 1

        print(f"  [{event_label}] {counts[event_label]}/{max_per_class}")


def download_missing_matches(competitions=None, seasons=None, max_per_class=2000):
    all_games = getListGames(split="all", dataset="SoccerNet")
    # print(all_games[0])

    # Filter by competition and season
    if competitions or seasons:
        filtered = []
        for game in all_games:
            parts = game.split('\\')
            comp, season = parts[0], parts[1]
            if (competitions is None or comp in competitions) and \
               (seasons is None or season in seasons):
                filtered.append(game)
        all_games = filtered

    print(f"Found {len(all_games)} games matching filters.")

    # Load existing counts from TFRecord on disk
    # counts = get_tfrecord_counts()

    counts = get_tfrecord_counts([
        # Path(r"E:\Football Dataset\TFRecords New\train_new.tfrecord"),
        Path(r"E:\Football Dataset\All records\train_new_1.tfrecord"),
        Path(r"E:\Football Dataset\All records\train_new_2.tfrecord"),
        Path(r"E:\Football Dataset\All records\train_new_3.tfrecord"),
        Path(r"E:\Football Dataset\All records\train_new_4.tfrecord"),
        Path(r"E:\Football Dataset\All records\train_new_5.tfrecord"),
        Path(r"E:\Football Dataset\All records\train_new_6.tfrecord"),
        Path(r"E:\Football Dataset\All records\train_new.tfrecord"),
        Path(r"E:\train_new.tfrecord"),
    ])

    # Check if all classes already done before opening writer
    if all(counts[cls] >= max_per_class for cls in CLASSES):
        print("All classes already at target. Nothing to do.")
        return

    # Open in append mode — 'a' adds to existing file without overwriting
    writer = tf.io.TFRecordWriter(str(OUTPUT_TFRECORD))

    written_this_session = 0

    try:
        for game in all_games:

            if all(counts[cls] >= max_per_class for cls in CLASSES):
                print("All classes reached target. Done.")
                break

            # if get_disk_space() < STORAGE_THRESHOLD:
            #     print("Low disk space, stopping.")
            #     break

            if match_exists(game):
                print(f"Skipping download {game} (footage exists)")
                # if annotation_exists(game):
                # process_match(game, writer, counts, max_per_class)
                continue

            # if not annotation_exists(game):
            successAnnotation = download_with_retry(
                annotationDownloader,
                game, files=["Labels-v2.json"], spl="train"
            )

            if successAnnotation:
                print(f"\nDownloading {game}...")
                success = download_with_retry(
                    matchDownloader, game,
                    files=["1_720p.mkv", "2_720p.mkv"], spl="train"
                )

                if not success:
                    continue
            
            elif not successAnnotation:
                continue


            before = sum(counts.values())
            process_match(game, writer, counts, max_per_class)
            written_this_session += sum(counts.values()) - before

            try:
                os.remove(str(MATCHES_DIR / game / "1_720p.mkv"))
                os.remove(str(MATCHES_DIR / game / "2_720p.mkv"))
                print(f"Deleted footage: {game}")
            except Exception as e:
                print(f"Failed to delete {game}: {e}")

    finally:
        writer.close()
        print(f"\nWrote {written_this_session} new records this session.")
        print("\nFinal counts:")
        for cls in CLASSES:
            print(f"  {cls:<25} {counts[cls]}")


if __name__ == "__main__":
    NEW_SEASONS = [
        # ("england_epl",        "2015-2016"),
        # ("england_epl",        "2016-2017"),
        # ("germany_bundesliga", "2017-2018"),
        ("france_ligue-1",     "2017-2018"),
        # ("europe_uefa-champions-league",       "2017-2018"),
    ]

    for competition, season in NEW_SEASONS:
        print(f"\n{'='*50}")
        print(f"  {competition}  {season}")
        print(f"{'='*50}")
        download_missing_matches(
            competitions=[competition],
            seasons=[season],
            max_per_class=1500
        )