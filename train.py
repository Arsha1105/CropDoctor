"""
train.py — Transfer-learning training script for CropDoctor
============================================================
Uses MobileNetV2 pretrained on ImageNet.  Only the final classifier head is
trained by default; fine-tuning (unfreezing the full network) can be added
as a second phase.

Quick start:
    python train.py --data_dir data/PlantVillage --epochs 10 --batch_size 32

Full options:
    python train.py --help
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on any machine
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision import models

from dataset import DEFAULT_CLASSES, check_dataset_path, make_splits

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def build_model(num_classes: int) -> nn.Module:
    """
    Load pretrained MobileNetV2 and replace the classifier head.

    The pretrained feature extractor is frozen initially so that training
    only updates the new head weights on the first pass.  This is a standard
    transfer-learning practice: the ImageNet features are already very useful
    for leaf images, so we first adapt the head before (optionally) fine-
    tuning the full network.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze all pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier:  original  [Dropout, Linear(1280 -> 1000)]
    in_features = model.classifier[1].in_features   # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features, num_classes),
    )
    # The new head parameters are trainable by default
    return model


def unfreeze_model(model: nn.Module):
    """Unfreeze all parameters for fine-tuning after head training."""
    for param in model.parameters():
        param.requires_grad = True


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    phase: str,
) -> tuple:
    """Run one full pass over loader; return (avg_loss, accuracy)."""
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    running_loss    = 0.0
    correct         = 0
    total           = 0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss    = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds         = outputs.argmax(dim=1)
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)

    avg_loss = running_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------------
def save_training_curves(history: dict, output_path: str):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss",  marker="o")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",    marker="s")
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train Accuracy", marker="o")
    axes[1].plot(epochs, history["val_acc"],   label="Val Accuracy",   marker="s")
    axes[1].set_title("Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Training curves saved -> {output_path}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(args):
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  CropDoctor — Training  ({device.type.upper()})")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Dataset
    # ------------------------------------------------------------------
    if not check_dataset_path(args.data_dir, DEFAULT_CLASSES):
        sys.exit(1)

    print(f"\n[INFO] Loading dataset from: {args.data_dir}")
    train_loader, val_loader, class_names = make_splits(
        root_dir          = args.data_dir,
        class_names       = DEFAULT_CLASSES,
        val_fraction      = 0.20,
        random_seed       = SEED,
        batch_size        = args.batch_size,
        num_workers       = 0,           # 0 is safest on Windows
        samples_per_class = args.samples_per_class,
    )
    num_classes = len(class_names)
    print(f"[INFO] Classes ({num_classes}): {class_names}")
    print(f"[INFO] Train batches: {len(train_loader)} | "
          f"Val batches: {len(val_loader)}")

    # ------------------------------------------------------------------
    # 2. Model
    # ------------------------------------------------------------------
    model     = build_model(num_classes)
    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Phase 1: train head only
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"\n[INFO] Phase 1 — head-only training "
          f"({len(trainable)} trainable parameter tensors)")

    optimizer = Adam(trainable, lr=args.lr)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

    history = {"train_loss": [], "train_acc": [],
               "val_loss":   [], "val_acc":   []}

    best_val_acc   = 0.0
    best_epoch     = 0
    models_dir     = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = models_dir / "best_model.pth"

    # ------------------------------------------------------------------
    # 3. Epoch loop
    # ------------------------------------------------------------------
    total_epochs = args.epochs
    for epoch in range(1, total_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, "train"
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, "val"
        )
        scheduler.step()
        elapsed = time.time() - t0

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Console output
        print(
            f"  Epoch {epoch:02d}/{total_epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_epoch    = epoch
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"    ✓ New best checkpoint saved  (val_acc={best_val_acc:.4f})")

    # ------------------------------------------------------------------
    # 4. Save artefacts
    # ------------------------------------------------------------------
    # Class names
    classes_path = models_dir / "class_names.json"
    with open(classes_path, "w") as f:
        json.dump(class_names, f, indent=2)

    # Config
    config = {
        "data_dir":    args.data_dir,
        "epochs":      args.epochs,
        "batch_size":  args.batch_size,
        "lr":          args.lr,
        "num_classes": num_classes,
        "seed":        SEED,
        "best_epoch":  best_epoch,
        "best_val_acc": round(best_val_acc, 4),
        "device":      device.type,
    }
    config_path = models_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Training history
    history_path = models_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Training curves plot
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    save_training_curves(history, str(outputs_dir / "training_curves.png"))

    print(f"\n{'='*60}")
    print(f"  Training complete.")
    print(f"  Best val accuracy : {best_val_acc:.4f}  (epoch {best_epoch})")
    print(f"  Model checkpoint  : {best_ckpt_path}")
    print(f"  Class names       : {classes_path}")
    print(f"  Config            : {config_path}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CropDoctor with MobileNetV2 transfer learning"
    )
    parser.add_argument(
        "--data_dir", default="data/PlantVillage",
        help="Root directory of the PlantVillage dataset",
    )
    parser.add_argument(
        "--models_dir", default="models",
        help="Directory to save model checkpoints and metadata",
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Mini-batch size (default: 32)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate for the head optimizer (default: 0.001)",
    )
    parser.add_argument(
        "--samples_per_class", type=int, default=0,
        help=(
            "Cap each class to this many images for faster CPU training. "
            "0 = use all images (default). "
            "Recommended: 200 for a ~10-min run, 0 for full training."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
