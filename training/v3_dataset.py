from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class V3SequenceDataset(Dataset):
    """Contiguous sequence dataset with explicit train/validation ranges."""

    def __init__(
        self,
        folder: str = "data_v3",
        sequence_length: int = 32,
        start: int = 0,
        end: int | None = None,
    ):
        folder_path = Path(folder)
        frames = np.load(folder_path / "frames.npy", mmap_mode="r")
        actions = np.load(folder_path / "actions.npy", mmap_mode="r")

        if frames.ndim != 3 or tuple(frames.shape[1:]) != (128, 128):
            raise ValueError(
                f"V3 expects frames shaped (N,128,128); got {frames.shape}."
            )
        if actions.ndim != 1 or len(actions) != len(frames):
            raise ValueError("V3 actions must be shaped (N,) and match frames.")

        self.frames_np = frames
        self.actions_np = actions
        self.sequence_length = int(sequence_length)
        self.start = int(start)
        self.end = len(frames) if end is None else min(int(end), len(frames))

        if self.start < 0 or self.end <= self.start:
            raise ValueError("Invalid dataset range.")
        if self.end - self.start <= self.sequence_length:
            raise ValueError("Dataset range is too short for the sequence length.")

    def __len__(self):
        return self.end - self.start - self.sequence_length

    def __getitem__(self, index):
        base = self.start + int(index)
        stop = base + self.sequence_length + 1
        frames = torch.from_numpy(
            np.asarray(self.frames_np[base:stop]).copy()
        ).float().unsqueeze(1)
        actions = torch.from_numpy(
            np.asarray(
                self.actions_np[base:base + self.sequence_length]
            ).copy()
        ).long()
        return frames, actions


def dataset_split_points(folder: str = "data_v3", train_fraction: float = 0.8):
    frames = np.load(Path(folder) / "frames.npy", mmap_mode="r")
    split = int(len(frames) * float(train_fraction))
    return len(frames), split
