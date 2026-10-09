"""
predictor.py — Reusable inference module for CropDoctor
=========================================================
Loads the saved model once and exposes a simple predict() function that
the Streamlit app (and any other caller) can import directly.

Usage (standalone test):
    python predictor.py path/to/leaf_image.jpg
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models

from dataset import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, get_val_transform

MODELS_DIR  = Path("models")
CHECKPOINT  = MODELS_DIR / "best_model.pth"
CLASS_JSON  = MODELS_DIR / "class_names.json"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _build_mobilenet(num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features, num_classes),
    )
    return model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class CropDoctorPredictor:
    """
    Load-once wrapper around the MobileNetV2 checkpoint.

    Parameters
    ----------
    checkpoint_path : str | Path, optional
        Path to the .pth weights file.  Defaults to models/best_model.pth.
    class_json_path : str | Path, optional
        Path to the class_names.json file.  Defaults to models/class_names.json.
    device : str, optional
        'cpu' or 'cuda'.  Defaults to auto-detect.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        class_json_path: Optional[str] = None,
        device: Optional[str]           = None,
    ):
        self._checkpoint = Path(checkpoint_path or CHECKPOINT)
        self._class_json = Path(class_json_path or CLASS_JSON)
        self._device     = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._transform  = get_val_transform()
        self._model:      Optional[nn.Module] = None
        self._class_names: Optional[List[str]] = None

    # ------------------------------------------------------------------
    @property
    def class_names(self) -> List[str]:
        if self._class_names is None:
            self._load()
        return self._class_names  # type: ignore

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    # ------------------------------------------------------------------
    def _load(self):
        """Load model and class names from disk (called lazily)."""
        # Validate files
        if not self._class_json.exists():
            raise FileNotFoundError(
                f"Class names file not found: '{self._class_json}'\n"
                "Please run train.py first."
            )
        if not self._checkpoint.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: '{self._checkpoint}'\n"
                "Please run train.py first."
            )

        with open(self._class_json) as f:
            self._class_names = json.load(f)

        model = _build_mobilenet(len(self._class_names))
        state_dict = torch.load(self._checkpoint, map_location=self._device)
        model.load_state_dict(state_dict)
        model.to(self._device)
        model.eval()
        self._model = model

    # ------------------------------------------------------------------
    def predict(
        self,
        image_source,
        top_k: int = 3,
    ) -> Dict:
        """
        Predict the disease class for a single leaf image.

        Parameters
        ----------
        image_source : str | Path | PIL.Image.Image
            The leaf image.  Can be a file path or an already-opened PIL image.
        top_k : int
            Number of top predictions to return (default 3).

        Returns
        -------
        dict with keys:
            predicted_class  : str   — name of the top-1 class
            confidence       : float — probability of the top-1 class (0–1)
            top_k            : list  — [{class, probability}, …] top-k entries
            is_healthy       : bool  — True when 'healthy' is in the class name
            error            : str | None — set if something went wrong
        """
        # Lazy load
        if self._model is None:
            try:
                self._load()
            except FileNotFoundError as exc:
                return {"error": str(exc)}

        # Load & validate image
        try:
            if isinstance(image_source, (str, Path)):
                image = Image.open(image_source).convert("RGB")
            elif isinstance(image_source, Image.Image):
                image = image_source.convert("RGB")
            else:
                return {"error": f"Unsupported image_source type: {type(image_source)}"}
        except Exception as exc:
            return {"error": f"Could not open image: {exc}"}

        # Preprocess
        tensor = self._transform(image).unsqueeze(0).to(self._device)  # [1,3,H,W]

        # Inference
        with torch.no_grad():
            logits      = self._model(tensor)
            probs       = F.softmax(logits, dim=1).squeeze(0)  # [num_classes]

        # Top-k
        top_probs, top_indices = torch.topk(probs, k=min(top_k, self.num_classes))
        top_results = [
            {
                "class":       self._class_names[idx.item()],
                "probability": round(prob.item(), 4),
            }
            for idx, prob in zip(top_indices, top_probs)
        ]

        best = top_results[0]
        return {
            "predicted_class": best["class"],
            "confidence":      best["probability"],
            "top_k":           top_results,
            "is_healthy":      "healthy" in best["class"].lower(),
            "error":           None,
        }


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------
_default_predictor: Optional[CropDoctorPredictor] = None


def get_predictor() -> CropDoctorPredictor:
    """Return a module-level singleton predictor (cached after first call)."""
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = CropDoctorPredictor()
    return _default_predictor


def predict_image(image_source, top_k: int = 3) -> Dict:
    """Convenience wrapper — calls get_predictor().predict(image_source)."""
    return get_predictor().predict(image_source, top_k=top_k)


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predictor.py <path_to_image>")
        sys.exit(1)

    img_path = sys.argv[1]
    print(f"\n[CropDoctor] Predicting: {img_path}")

    predictor = CropDoctorPredictor()
    result    = predictor.predict(img_path)

    if result.get("error"):
        print(f"[ERROR] {result['error']}")
        sys.exit(1)

    print(f"\n  Predicted Class : {result['predicted_class']}")
    print(f"  Confidence      : {result['confidence']*100:.2f} %")
    print(f"  Is Healthy      : {result['is_healthy']}")
    print("\n  Top predictions:")
    for entry in result["top_k"]:
        bar = "█" * int(entry["probability"] * 30)
        print(f"    {entry['class']:<45} {entry['probability']*100:5.1f}%  {bar}")
