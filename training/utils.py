"""
Shared utilities for training ML and DL models on the SkinDisease dataset.
"""

from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np


def get_project_root() -> Path:
    """Return the project root (directory containing app.py / requirements.txt)."""
    return Path(__file__).resolve().parents[1]


def get_skin_image_root() -> Path:
    """
    Return the root directory that contains the SkinDisease image folders.

    Expected structure (already present in your workspace):
        <project_root>/SkinDisease/SkinDisease/train
        <project_root>/SkinDisease/SkinDisease/test
    """
    root = get_project_root() / "SkinDisease" / "SkinDisease"
    if not root.is_dir():
        raise FileNotFoundError(
            f"Expected SkinDisease image dataset under: {root}\n"
            "Please make sure the dataset is placed there."
        )
    return root


def extract_color_histogram(
    image_bgr: np.ndarray, bins: Tuple[int, int, int] = (8, 8, 8)
) -> np.ndarray:
    """
    Simple handcrafted feature extractor using a 3D color histogram in HSV space.

    This is used for the classical ML models (SVM / RF / Logistic Regression)
    as Dataset A-style tabular features.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def load_image_for_features(image_path: Path, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Load an image from disk, resize, and return as a BGR numpy array.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    img = cv2.resize(img, target_size)
    return img


def compute_severity_level(confidence: float) -> str:
    """
    Map a confidence score in [0, 1] to a simple severity level.
    This is *not* clinically validated, just an educational heuristic.
    """
    if confidence >= 0.85:
        return "High"
    if confidence >= 0.6:
        return "Medium"
    return "Low"


def allowed_image_file(filename: str, allowed_exts: List[str] = None) -> bool:
    """Check if an uploaded filename has an allowed image extension."""
    if allowed_exts is None:
        allowed_exts = [".jpg", ".jpeg", ".png"]
    suffix = Path(filename).suffix.lower()
    return suffix in allowed_exts


