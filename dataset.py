"""
dataset.py — PlantVillage dataset loading and preprocessing
============================================================
Handles:
  - Verifying that the expected class folders exist in the dataset root
  - Building train / val splits (stratified, reproducible, no leakage)
  - Applying the correct MobileNetV2 / ImageNet normalisation
  - Printing class distribution and split sizes

Usage (standalone check):
    python dataset.py --data_dir data/PlantVillage
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Supported subset of PlantVillage classes
# These names match the folder names used in the *colour* PlantVillage release
# (https://www.kaggle.com/datasets/emmarex/plantdisease).
# You can add / remove entries to match what you downloaded.
# ---------------------------------------------------------------------------
DEFAULT_CLASSES: List[str] = [
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Miner",
    "Tomato_healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
]

# ImageNet mean / std — required by pretrained MobileNetV2
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224

# Minimum images per class for a usable split
MIN_IMAGES_PER_CLASS = 10


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def get_train_transform() -> transforms.Compose:
    """Augmented transform for training images."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        # ColorJitter removed: it is the single most expensive CPU transform
        # and adds little value when only the head is being trained.
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform() -> transforms.Compose:
    """Deterministic transform for validation / inference."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------
class PlantVillageDataset(Dataset):
    """
    Custom Dataset for PlantVillage.

    Scans the given root directory for the specified class folders and builds
    a flat list of (image_path, label_index) pairs.

    Parameters
    ----------
    root_dir : str | Path
        Path to the PlantVillage directory that contains one sub-folder per
        class (e.g.  data/PlantVillage/).
    class_names : list[str]
        Ordered list of class folder names to include.
    transform : callable, optional
        torchvision transform applied to each image.
    """

    def __init__(
        self,
        root_dir: str,
        class_names: List[str],
        transform=None,
    ):
        self.root_dir    = Path(root_dir)
        self.class_names = class_names
        self.transform   = transform
        self.class_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(class_names)
        }

        self.samples: List[Tuple[Path, int]] = []
        self._scan_directories()

    # ------------------------------------------------------------------
    def _scan_directories(self):
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

        for cls_name in self.class_names:
            cls_dir = self.root_dir / cls_name
            if not cls_dir.is_dir():
                print(f"  [WARNING] Folder not found, skipping: {cls_dir}")
                continue
            candidates = sorted(
                p for p in cls_dir.iterdir()
                if p.suffix.lower() in valid_extensions
            )
            # Full-decode validation: open + convert to RGB.
            # verify() only checks headers and misses truncated pixel data,
            # so we do the actual decode here once to guarantee the sample
            # list is crash-free during training.
            good, bad = 0, 0
            for f in candidates:
                try:
                    with Image.open(f) as img:
                        img.convert("RGB")   # full pixel decode
                    self.samples.append((f, self.class_to_idx[cls_name]))
                    good += 1
                except Exception:
                    bad += 1
            if bad:
                print(
                    f"  [WARNING] '{cls_name}': skipped {bad} unreadable "
                    f"file(s) out of {good + bad} found."
                )
            if good < MIN_IMAGES_PER_CLASS:
                print(
                    f"  [WARNING] Only {good} valid image(s) in '{cls_name}' "
                    f"(minimum expected: {MIN_IMAGES_PER_CLASS}). "
                    "Consider adding more images."
                )

        if not self.samples:
            raise FileNotFoundError(
                f"No valid images found under '{self.root_dir}' for the "
                "specified classes. Please check your dataset path and class "
                "names.\n\nExpected layout:\n"
                "  data/PlantVillage/<ClassName>/<image.jpg>"
            )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        # All files in self.samples were full-decoded at scan time, so this
        # open is expected to succeed.  If it somehow fails (e.g. file deleted
        # after scan), skip gracefully by returning the next sample.
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            print(f"  [WARNING] Skipping unreadable sample at runtime: "
                  f"{img_path} ({exc})")
            # Return the previous sample as a safe fallback
            alt_idx = (idx - 1) % len(self.samples)
            img_path, label = self.samples[alt_idx]
            image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ---------------------------------------------------------------------------
# Split helper
# ---------------------------------------------------------------------------
def make_splits(
    root_dir: str,
    class_names: List[str],
    val_fraction: float = 0.20,
    random_seed: int = 42,
    batch_size: int = 32,
    num_workers: int = 0,
    samples_per_class: int = 0,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Build stratified train / validation DataLoaders.

    Parameters
    ----------
    samples_per_class : int
        If > 0, randomly cap each class to this many images before splitting.
        Useful for fast CPU training runs (e.g. 200 -> ~10 min for 8 classes).
        0 means use all available images.

    Returns
    -------
    train_loader, val_loader, active_class_names
        active_class_names is the filtered list of classes that actually have
        images — may be shorter than the input class_names.
    """
    # Build a base dataset (no transform yet — we only need the label list)
    # This also validates images and warns about corrupt files.
    base_dataset = PlantVillageDataset(
        root_dir=root_dir,
        class_names=class_names,
        transform=None,
    )

    # Determine which classes actually have samples (some may be missing)
    present_indices = {label for _, label in base_dataset.samples}
    active_class_names = [
        name for idx, name in enumerate(class_names) if idx in present_indices
    ]
    if len(active_class_names) < len(class_names):
        dropped = [n for n in class_names if n not in active_class_names]
        print(
            f"[INFO] {len(dropped)} class(es) have no images and will be "
            f"excluded from training: {dropped}"
        )
    if len(active_class_names) < 2:
        raise ValueError(
            f"Need at least 2 classes with images to train. "
            f"Found: {active_class_names}. "
            "Please add more class folders to your dataset directory."
        )

    # Rebuild datasets using only the active classes so label indices are
    # contiguous (0 … N-1) and the model head size matches exactly.
    base_dataset = PlantVillageDataset(
        root_dir=root_dir,
        class_names=active_class_names,
        transform=None,
    )

    labels = [label for _, label in base_dataset.samples]

    # Optional per-class cap — keeps training fast on CPU
    if samples_per_class > 0:
        rng = np.random.default_rng(random_seed)
        kept = []
        for cls_idx in range(len(active_class_names)):
            cls_indices = [i for i, l in enumerate(labels) if l == cls_idx]
            if len(cls_indices) > samples_per_class:
                cls_indices = rng.choice(
                    cls_indices, size=samples_per_class, replace=False
                ).tolist()
            kept.extend(cls_indices)
        kept.sort()
        labels = [labels[i] for i in kept]
        # Remap base_dataset samples to only the kept indices
        base_dataset.samples = [base_dataset.samples[i] for i in kept]
        print(
            f"[INFO] Capped to {samples_per_class} images/class -> "
            f"{len(labels)} total samples"
        )

    # Stratified split
    indices = list(range(len(base_dataset)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=random_seed,
        stratify=labels,
    )

    # Build separate dataset objects (with the correct transform each)
    train_dataset = PlantVillageDataset(
        root_dir=root_dir, class_names=active_class_names,
        transform=get_train_transform(),
    )
    val_dataset = PlantVillageDataset(
        root_dir=root_dir, class_names=active_class_names,
        transform=get_val_transform(),
    )

    train_subset = Subset(train_dataset, train_idx)
    val_subset   = Subset(val_dataset,   val_idx)

    # Class distribution report (use active_class_names — contiguous indices)
    train_labels = [labels[i] for i in train_idx]
    val_labels   = [labels[i] for i in val_idx]
    _print_distribution(active_class_names, train_labels, val_labels)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=False,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=False,
    )

    return train_loader, val_loader, active_class_names


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _print_distribution(
    class_names: List[str],
    train_labels: List[int],
    val_labels: List[int],
):
    print("\n" + "=" * 60)
    print(f"{'Class':<45} {'Train':>6} {'Val':>6} {'Total':>6}")
    print("-" * 60)
    for idx, name in enumerate(class_names):
        t = train_labels.count(idx)
        v = val_labels.count(idx)
        print(f"  {name:<43} {t:>6} {v:>6} {t+v:>6}")
    print("-" * 60)
    print(f"  {'TOTAL':<43} {len(train_labels):>6} {len(val_labels):>6} "
          f"{len(train_labels)+len(val_labels):>6}")
    print("=" * 60 + "\n")


def check_dataset_path(data_dir: str, class_names: List[str]) -> bool:
    """
    Quick sanity-check: verify the data directory exists and contains at
    least some of the expected class folders.  Returns True if OK.
    """
    root = Path(data_dir)
    if not root.is_dir():
        print(f"[ERROR] Dataset directory not found: '{data_dir}'")
        print(
            "\nPlease download the PlantVillage dataset and place it so that "
            "each class is a sub-folder, e.g.:\n"
            "  data/PlantVillage/Tomato_Early_blight/*.jpg\n"
            "  data/PlantVillage/Potato___healthy/*.jpg\n"
            "  ...\n"
            "Dataset download options:\n"
            "  Kaggle  : https://www.kaggle.com/datasets/emmarex/plantdisease\n"
            "  GitHub  : https://github.com/spMohanty/PlantVillage-Dataset\n"
        )
        return False

    found = [c for c in class_names if (root / c).is_dir()]
    missing = [c for c in class_names if not (root / c).is_dir()]
    print(f"[INFO] Found {len(found)}/{len(class_names)} expected class "
          f"folders under '{data_dir}'.")
    if missing:
        print("[WARNING] Missing folders:")
        for m in missing:
            print(f"  {m}")
    return len(found) > 0


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify PlantVillage dataset")
    parser.add_argument(
        "--data_dir", default="data/PlantVillage",
        help="Path to PlantVillage root directory",
    )
    args = parser.parse_args()

    ok = check_dataset_path(args.data_dir, DEFAULT_CLASSES)
    if ok:
        try:
            make_splits(args.data_dir, DEFAULT_CLASSES)
            print("[OK] Dataset looks good — ready to train.")
        except FileNotFoundError as e:
            print(e)
            sys.exit(1)
    else:
        sys.exit(1)
