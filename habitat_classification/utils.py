"""
Utility functions for habitat classification.
"""

import numpy as np
import pandas as pd
import base64
from pathlib import Path


def decode_patch(encoded: str) -> np.ndarray:
    """
    Decode base64 string to numpy array.

    Args:
        encoded: base64-encoded string representing a (15, 35, 35) float32 array

    Returns:
        numpy array of shape (15, 35, 35)
    """
    patch_bytes = base64.b64decode(encoded)
    patch = np.frombuffer(patch_bytes, dtype=np.float32)
    return patch.reshape(15, 35, 35)


def encode_patch(patch: np.ndarray) -> str:
    """
    Encode numpy array to base64 string.

    Args:
        patch: numpy array of shape (15, 35, 35)

    Returns:
        base64-encoded string
    """
    patch_float32 = patch.astype(np.float32)
    return base64.b64encode(patch_float32.tobytes()).decode("utf-8")


def load_training_data():
    """
    Load training patches and labels.

    Returns:
        patches: numpy array of shape (N, 15, 35, 35)
        labels: numpy array of shape (N,) with class indices 0-70
    """
    data_dir = Path(__file__).parent / "data"
    training_data = Path(__file__).parent / "training"
    patches = np.load(training_data / "train/patches.npy")
    labels_df = pd.read_csv(data_dir / "train.csv")
    return patches, labels_df["vistgerd_idx"].values


def load_class_names():
    """
    Load class name mappings.

    Returns:
        dict with 'vistgerd' and 'vistlendi' mappings
    """
    import json
    data_dir = Path(__file__).parent / "data"
    with open(data_dir / "class_names.json", "r") as f:
        return json.load(f)


def load_hierarchy():
    """
    Load hierarchy mapping (vistgerd -> vistlendi).

    Returns:
        dict mapping vistgerd index to vistlendi index
    """
    import json
    data_dir = Path(__file__).parent / "data"
    with open(data_dir / "hierarchy.json", "r") as f:
        return json.load(f)


def extract_features(patch: np.ndarray) -> np.ndarray:
    """
    Extract simple features from a patch.

    Args:
        patch: numpy array of shape (15, 35, 35)

    Returns:
        feature vector (30 values: 15 means + 15 stds)
    """
    # Band statistics
    means = patch.mean(axis=(1, 2))  # 15 values
    stds = patch.std(axis=(1, 2))    # 15 values

    features = np.concatenate([means, stds])

    return features
