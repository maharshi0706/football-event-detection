import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import sys
sys.path.append("..")

from Inference.modelTesting import predict, load_model, CHECKPOINT, CLASSES

if __name__ == "__main__":
    model = load_model(CHECKPOINT)

    TEST_DIR = Path(r"E:\Football Dataset\Event Clips Split\test")

    all_preds  = []
    all_labels = []
    errors     = []

    for cls_folder in sorted(TEST_DIR.iterdir()):
        if not cls_folder.is_dir():
            continue
        true_label = cls_folder.name
        if true_label not in CLASSES:
            continue

        for clip in cls_folder.glob("*.mp4"):
            try:
                preds = predict(model, clip, top_k=1)
                predicted = preds[0]["class"]
                confidence = preds[0]["confidence"]

                all_preds.append(predicted)
                all_labels.append(true_label)

                if predicted != true_label:
                    errors.append({
                        "file": clip.name,
                        "true": true_label,
                        "predicted": predicted,
                        "confidence": confidence
                    })
            except Exception as e:
                print(f"Error on {clip.name}: {e}")

    # Accuracy
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    print(f"\nAccuracy: {correct}/{len(all_labels)} = {correct/len(all_labels)*100:.1f}%")

    # Top confused pairs
    from collections import Counter
    confused_pairs = Counter(
        (e["true"], e["predicted"]) for e in errors
    )
    print("\nTop confused pairs (true → predicted):")
    for (true, pred), count in confused_pairs.most_common(10):
        print(f"  {true:<22} → {pred:<22} {count} times")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.title("Confusion Matrix — v5 Model")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("Saved confusion_matrix.png")

