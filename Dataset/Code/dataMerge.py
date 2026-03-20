import shutil
from pathlib import Path

# ── Configure paths ───────────────────────────────────────────────────────────

OLD_TFRECORD = Path(r"E:\Football Dataset\Event Clips Split\train_recovered.tfrecord")

NEW_TFRECORDS = [
    Path(r"E:\Football Dataset\All records\train_new_1.tfrecord"),
    Path(r"E:\Football Dataset\All records\train_new_2.tfrecord"),
    Path(r"E:\Football Dataset\All records\train_new_3.tfrecord"),
    Path(r"E:\Football Dataset\All records\train_new_4.tfrecord"),
    Path(r"E:\Football Dataset\All records\train_new_5.tfrecord"),
    Path(r"E:\Football Dataset\All records\train_new_6.tfrecord"),
    Path(r"E:\Football Dataset\All records\train_new_7.tfrecord"),
    Path(r"E:\Football Dataset\All records\train_new.tfrecord"),
]

OUTPUT = Path(r"E:\Football Dataset\train_merged_v2.tfrecord")

all_files = [OLD_TFRECORD] + NEW_TFRECORDS

print(f"Merging {len(all_files)} files into {OUTPUT}...")

# total_size = sum(f.stat().st_size for f in all_files if f.exists()) / (2**30)
# print(f"Total size to merge: {total_size:.2f} GB")

for f in all_files:
    if f.exists():
        print(f"{f.name}: {f.stat().st_size / (2**30):.2f} GB")
    else:
        print(f"Missing: {f}")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT, "wb") as outfile:
    for path in all_files:
        if not path.exists():
            print(f"  ⚠️  Skipping missing file: {path}")
            continue
        size_gb = path.stat().st_size / (2**30)
        print(f"  Merging {path.name} ({size_gb:.2f} GB)...")
        with open(path, "rb") as infile:
            shutil.copyfileobj(infile, outfile, length=64 * 1024 * 1024)  # 64MB chunks

print(f"\nDone. Output: {OUTPUT} ({OUTPUT.stat().st_size / (2**30):.2f} GB)")





