"""Shared constants and validation helpers for the 128x128 retraining path."""

from __future__ import annotations

from pathlib import Path
import numpy as np

IMAGE_SIZE = 128
DATA_FOLDER = "data_128"
MODEL_FOLDER = "models_128"
RESULTS_FOLDER = "results_128"
OUTPUT_FOLDER = "outputs_128"
SEQUENCE_LENGTH = 16


def validate_128_dataset(folder: str = DATA_FOLDER) -> tuple[tuple[int, ...], tuple[int, ...]]:
    folder_path = Path(folder)
    frames_path = folder_path / "frames.npy"
    actions_path = folder_path / "actions.npy"

    if not frames_path.exists() or not actions_path.exists():
        raise FileNotFoundError(
            f"Expected {frames_path} and {actions_path}. "
            "Collect fresh 128x128 data first."
        )

    frames = np.load(frames_path, mmap_mode="r")
    actions = np.load(actions_path, mmap_mode="r")

    if frames.ndim != 3 or tuple(frames.shape[1:]) != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"{frames_path} has shape {frames.shape}; expected "
            f"(N, {IMAGE_SIZE}, {IMAGE_SIZE}). Do not reuse the 64x64 dataset."
        )

    if actions.ndim != 1:
        raise ValueError(f"{actions_path} has shape {actions.shape}; expected (N,).")

    if len(frames) != len(actions):
        raise ValueError(
            f"frames/actions length mismatch: {len(frames)} vs {len(actions)}."
        )

    if len(frames) <= SEQUENCE_LENGTH + 1:
        raise ValueError(
            f"Dataset has only {len(frames)} frames; need more than "
            f"{SEQUENCE_LENGTH + 1} for sequence training."
        )

    return tuple(frames.shape), tuple(actions.shape)
