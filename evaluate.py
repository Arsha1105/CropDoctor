"""
evaluate.py — Model evaluation for CropDoctor
==============================================
Loads the best saved checkpoint and evaluates it on the held-out validation
split.  Produces:
  outputs/confusion_matrix.png
  outputs/classification_report.txt
  outputs/metrics.json

IMPORTANT DISCLAIMER
--------------------
These metrics measure performance on the held-out portion of the PlantVillage
dataset.  PlantVillage images are taken under controlled, lab-like conditions.
Performance on real farm photos taken in variable lighting, with partial leaves
or background clutter, will typically be lower.  Always validate the model
on your own representative images before making production decisions.

Usage:
    python evaluate.py --data_dir data/PlantVillage
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torchvision import models

from dataset import DEFAULT_CLASSES, check_dataset_path, make_splits

SEED           = 42
MODELS_DIR     = Path("models")
OUTPUTS_DIR    = Path("outputs")
CHECKPOINT     = MODELS_DIR / "best_model.pth"
CLASS_JSON     = MODELS_DIR / "class_names.json"
CONFIG_JSON    = MODELS_DIR / "config.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_model(checkpoint_path: Path, num_classes: int, device: torch.device):
    """Re-build MobileNetV2 with the correct head and load saved weights."""
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features, num_classes),
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_class_names() -> list:
    if not CLASS_JSON.exists():
        print(f"[ERROR] class_names.json not found at '{CLASS_JSON}'.")
        print("  Please run train.py first.")
        sys.exit(1)
    with open(CLASS_JSON) as f:
        return json.load(f)


def collect_predictions(model, loader, device):
    """Return (all_true_labels, all_predicted_labels) as Python lists."""
    all_true, all_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds   = outputs.argmax(dim=1).cpu().tolist()
            all_pred.extend(preds)
            all_true.extend(labels.tolist())
    return all_true, all_pred


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def save_confusion_matrix(y_true, y_pred, class_names, output_path):
    cm = confusion_matrix(y_true, y_pred)
    # Normalise by true-class totals so each cell shows the fraction
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    short_names = [n.replace("___", "\n").replace("_", " ") for n in class_names]

    fig_side = max(10, len(class_names))
    fig, ax = plt.subplots(figsize=(fig_side, fig_side - 1))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=short_names,
        yticklabels=short_names,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Normalised Confusion Matrix (row = true class)", fontsize=14)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Confusion matrix saved -> {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def evaluate(args):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  CropDoctor — Evaluation  ({device.type.upper()})")
    print(f"{'='*60}")

    # Check checkpoint
    if not CHECKPOINT.exists():
        print(f"[ERROR] Checkpoint not found: '{CHECKPOINT}'")
        print("  Please run train.py first.")
        sys.exit(1)

    # Load class names (use saved ones to guarantee order matches training)
    class_names = load_class_names()
    num_classes  = len(class_names)
    print(f"[INFO] Classes ({num_classes}): {class_names}")

    # Check dataset
    if not check_dataset_path(args.data_dir, class_names):
        sys.exit(1)

    # Build the val loader with the same split used during training
    print(f"\n[INFO] Rebuilding validation split from: {args.data_dir}")
    _, val_loader, _ = make_splits(
        root_dir     = args.data_dir,
        class_names  = class_names,
        val_fraction = 0.20,
        random_seed  = SEED,
        batch_size   = args.batch_size,
        num_workers  = 0,
    )
    print(f"[INFO] Validation batches: {len(val_loader)}")

    # Load model
    print(f"\n[INFO] Loading checkpoint: {CHECKPOINT}")
    model = load_model(CHECKPOINT, num_classes, device)

    # Collect predictions
    print("[INFO] Running inference on validation set …")
    y_true, y_pred = collect_predictions(model, val_loader, device)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    overall_acc    = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall   = recall_score   (y_true, y_pred, average="macro", zero_division=0)
    macro_f1       = f1_score       (y_true, y_pred, average="macro", zero_division=0)

    print(f"\n{'='*60}")
    print("  EVALUATION RESULTS (validation split — PlantVillage dataset)")
    print(f"{'='*60}")
    print(f"  Overall Accuracy  : {overall_acc:.4f}  ({overall_acc*100:.2f} %)")
    print(f"  Macro Precision   : {macro_precision:.4f}")
    print(f"  Macro Recall      : {macro_recall:.4f}")
    print(f"  Macro F1-Score    : {macro_f1:.4f}")
    print(f"{'='*60}")
    print("\n  NOTE: These metrics apply to the controlled PlantVillage images.")
    print("  Performance on real farm photos may differ significantly.")
    print(f"{'='*60}\n")

    # Full classification report (per-class)
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    print(report)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    # 1. metrics.json
    metrics = {
        "note": (
            "Metrics measured on the 20% held-out PlantVillage validation split. "
            "Do not assume similar performance on out-of-distribution field photos."
        ),
        "overall_accuracy":  round(overall_acc,      4),
        "macro_precision":   round(macro_precision,  4),
        "macro_recall":      round(macro_recall,     4),
        "macro_f1":          round(macro_f1,         4),
        "num_classes":       num_classes,
        "num_val_samples":   len(y_true),
        "class_names":       class_names,
    }
    metrics_path = OUTPUTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved -> {metrics_path}")

    # 2. classification_report.txt
    report_path = OUTPUTS_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write("CropDoctor — Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Overall Accuracy : {overall_acc:.4f}\n")
        f.write(f"Macro Precision  : {macro_precision:.4f}\n")
        f.write(f"Macro Recall     : {macro_recall:.4f}\n")
        f.write(f"Macro F1-Score   : {macro_f1:.4f}\n\n")
        f.write("Per-Class Report:\n")
        f.write(report)
        f.write("\n\nDISCLAIMER: Metrics are from the held-out PlantVillage "
                "validation split and do not guarantee real-world performance.\n")
    print(f"[INFO] Classification report saved -> {report_path}")

    # 3. confusion_matrix.png
    cm_path = OUTPUTS_DIR / "confusion_matrix.png"
    save_confusion_matrix(y_true, y_pred, class_names, str(cm_path))

    print(f"\n[DONE] All evaluation outputs saved to '{OUTPUTS_DIR}/'.\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the saved CropDoctor model"
    )
    parser.add_argument(
        "--data_dir", default="data/PlantVillage",
        help="Root directory of the PlantVillage dataset",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for inference (default: 32)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
