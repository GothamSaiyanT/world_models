import os
import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceRangeDataset(Dataset):
    """Contiguous frame/action sequences for the final 64x64 experiment.

    The saved dataset is intentionally uint8 to keep it small in Kaggle and on disk.
    Samples are converted to float32 in [0, 1] only when they are loaded.
    """

    def __init__(self, folder, start, end, sequence_length=16):
        self.frames_path = os.path.join(str(folder), "frames.npy")
        self.actions_path = os.path.join(str(folder), "actions.npy")

        if not os.path.exists(self.frames_path):
            raise FileNotFoundError(self.frames_path)
        if not os.path.exists(self.actions_path):
            raise FileNotFoundError(self.actions_path)

        self.frames = np.load(self.frames_path, mmap_mode="r")
        self.actions = np.load(self.actions_path, mmap_mode="r")

        self.start = int(start)
        self.end = int(end)
        self.sequence_length = int(sequence_length)

        if self.start < 0 or self.end > len(self.frames):
            raise ValueError("Dataset range is outside the saved frames.")
        if self.end - self.start <= self.sequence_length:
            raise ValueError("Dataset range is too short for the requested sequence length.")

    def __len__(self):
        return self.end - self.start - self.sequence_length

    def __getitem__(self, index):
        base = self.start + int(index)
        stop = base + self.sequence_length + 1

        frame_np = np.asarray(self.frames[base:stop]).copy()
        action_np = np.asarray(
            self.actions[base:base + self.sequence_length]
        ).copy()

        frame_tensor = torch.from_numpy(frame_np).float()
        if frame_tensor.max() > 1.0:
            frame_tensor = frame_tensor / 255.0

        frame_tensor = frame_tensor.unsqueeze(1)
        action_tensor = torch.from_numpy(action_np).long()

        return frame_tensor, action_tensor


def load_normalized_frames(folder):
    """Load the complete frame array as float32 in [0, 1]."""
    frames = np.load(os.path.join(str(folder), "frames.npy"))
    frames = frames.astype(np.float32)
    if frames.max() > 1.0:
        frames /= 255.0
    return frames


def load_actions(folder):
    return np.load(os.path.join(str(folder), "actions.npy")).astype(np.int64)
