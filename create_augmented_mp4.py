# augment_mp4.py – run once to create augmented .mp4 files
import cv2
import moviepy.editor as mpy
import albumentations as A
import os
import glob
from pathlib import Path
from tqdm import tqdm

INPUT_ROOT = Path(r"D:\Football Highlight Generation\Event Clips Split\train")
OUTPUT_ROOT = Path(r"E:\Football Highlight Generation\Event Clips Augmented\train")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

aug_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=8, p=0.3),
    A.GaussNoise(p=0.2),
])

def augment_and_save(input_path, output_dir, clip_id):
    clip = mpy.VideoFileClip(str(input_path))
    
    def augment_frame(get_frame, t):
        frame = get_frame(t)
        aug = aug_pipeline(image=frame)['image']
        return aug

    for aug_id in range(2):  # create 2 augmented versions per clip
        aug_clip = clip.fl(augment_frame)
        output_path = output_dir / f"{clip_id.stem}_aug{aug_id}.mp4"
        aug_clip.write_videofile(str(output_path), codec='libx264', fps=30, logger=None)

# Loop over all classes
for class_dir in tqdm(INPUT_ROOT.iterdir()):
    if not class_dir.is_dir():
        continue
    output_class_dir = OUTPUT_ROOT / class_dir.name
    output_class_dir.mkdir(exist_ok=True)
    
    for clip_path in class_dir.glob("*.mp4"):
        augment_and_save(clip_path, output_class_dir, clip_path)