import cv2
import torch
import numpy as np
from pathlib import Path
from transformers import VideoMAEForVideoClassification


import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

CLASSES = [
    "Ball out of play", "Clearance", "Corner", "Direct free-kick",
    "Foul", "Goal", "Indirect free-kick", "Kick-off", "Offside",
    "Shots off target", "Shots on target", "Substitution",
    "Throw-in", "Yellow card"
]

CHECKPOINT = Path(r"D:\Football Event Detection\checkpoints\best_acc_0.6627.pth")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES = 16
IMG_SIZE   = 224

def load_model(checkpoint_path):
    model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base",
        num_labels=len(CLASSES),
        ignore_mismatched_sizes=True
    )

    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(model.config.hidden_size, len(CLASSES))
    )

    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict=state_dict)
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model


def extract_frames(video_path, num_frames=NUM_FRAMES):
    """Evenly sample num_frames from entire video clip."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(total_frames * 0.2)
    indices = np.linspace(start_frame, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if len(frames) != num_frames:
        raise ValueError(f"Expected {num_frames} frames, got {len(frames)}")

    return np.array(frames, dtype=np.uint8)  # (16, 224, 224, 3)


def preprocess(frames):
    """Convert frames to model input tensor."""
    video = torch.from_numpy(frames).float().div(255.0)  # (16, 224, 224, 3)
    video = video.permute(0, 3, 1, 2)                    # (16, 3, 224, 224)
    video = video.unsqueeze(0).to(DEVICE)                 # (1, 16, 3, 224, 224)
    return video


def predict(model, video_path, top_k=3):
    """
    Run inference on a video clip.
    Returns top-k predictions with confidence scores.
    """
    frames  = extract_frames(video_path)
    video   = preprocess(frames)

    with torch.no_grad():
        outputs = model(video)
        probs   = torch.softmax(outputs.logits, dim=-1)[0]

    top_probs, top_indices = torch.topk(probs, k=top_k)

    results = []
    for prob, idx in zip(top_probs.cpu(), top_indices.cpu()):
        results.append({
            "class": CLASSES[idx.item()],
            "confidence": round(prob.item() * 100, 2)
        })

    return results


def print_predictions(video_path, predictions):
    print(f"\nVideo: {Path(video_path).name}")
    print("─" * 35)
    for i, pred in enumerate(predictions):
        bar = "█" * int(pred["confidence"] / 5)
        print(f"  {i+1}. {pred['class']:<22} {pred['confidence']:>6.2f}%  {bar}")
    print()


if __name__ == "__main__":
    model = load_model(CHECKPOINT)

    # Single video
    video_path = r"E:\Football Dataset\Event Clips Split\test\Goal\Goal_2015-03-10 - 22-45 FC Porto 4 - 0 Basel_37.mp4"
    # video_path = r"E:\Football Dataset\Event Clips Split\test\Goal\Goal_2014-11-04 - 22-45 Dortmund 4 - 1 Galatasaray_179.mp4"
    # video_path = r"E:\Football Dataset\Event Clips Split\test\Goal\Goal_2014-11-04 - 22-45 Arsenal 3 - 3 Anderlecht_59.mp4"
    preds = predict(model, video_path, top_k=3)
    print_predictions(video_path, preds)

    # Batch inference on a folder
    # clips_dir = Path(r"path\to\clips\folder")
    # for clip in clips_dir.glob("*.mp4"):
    #     try:
    #         preds = predict(model, clip, top_k=3)
    #         print_predictions(clip, preds)
    #     except Exception as e:
    #         print(f"Error on {clip.name}: {e}")

    # TEST_DIR = Path(r"E:\Football Dataset\Event Clips Split\test")

    # all_preds  = []
    # all_labels = []
    # errors     = []

    # for cls_folder in sorted(TEST_DIR.iterdir()):
    #     if not cls_folder.is_dir():
    #         continue
    #     true_label = cls_folder.name
    #     if true_label not in CLASSES:
    #         continue

    #     for clip in cls_folder.glob("*.mp4"):
    #         try:
    #             preds = predict(model, clip, top_k=1)
    #             predicted = preds[0]["class"]
    #             confidence = preds[0]["confidence"]

    #             all_preds.append(predicted)
    #             all_labels.append(true_label)

    #             if predicted != true_label:
    #                 errors.append({
    #                     "file": clip.name,
    #                     "true": true_label,
    #                     "predicted": predicted,
    #                     "confidence": confidence
    #                 })
    #         except Exception as e:
    #             print(f"Error on {clip.name}: {e}")

    # # Accuracy
    # correct = sum(p == l for p, l in zip(all_preds, all_labels))
    # print(f"\nAccuracy: {correct}/{len(all_labels)} = {correct/len(all_labels)*100:.1f}%")

    # # Top confused pairs
    # from collections import Counter
    # confused_pairs = Counter(
    #     (e["true"], e["predicted"]) for e in errors
    # )
    # print("\nTop confused pairs (true → predicted):")
    # for (true, pred), count in confused_pairs.most_common(10):
    #     print(f"  {true:<22} → {pred:<22} {count} times")

    # # Confusion matrix
    # cm = confusion_matrix(all_labels, all_preds, labels=CLASSES)
    # fig, ax = plt.subplots(figsize=(14, 12))
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    # disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    # plt.title("Confusion Matrix — v5 Model")
    # plt.tight_layout()
    # plt.savefig("confusion_matrix.png", dpi=150)
    # plt.show()
    # print("Saved confusion_matrix.png")

