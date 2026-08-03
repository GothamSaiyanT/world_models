import os
import numpy as np
import torch
from torch.utils.data import Dataset


class WorldModelDataset(Dataset):
    """
    Kept for backward compatibility / single-step evaluation
    (used by evaluate.py). Not used for training anymore.
    """

    def __init__(self, folder="data"):

        self.frames = torch.from_numpy(
            np.load(
                os.path.join(folder, "frames.npy")
            )
        ).float()

        self.actions = torch.from_numpy(
            np.load(
                os.path.join(folder, "actions.npy")
            )
        ).long()

    def __len__(self):

        return len(self.frames) - 1

    def __getitem__(self, index):

        current_frame = self.frames[index].unsqueeze(0)

        next_frame = self.frames[index + 1].unsqueeze(0)

        action = self.actions[index]

        return (
            current_frame,
            action,
            next_frame
        )


class WorldModelSequenceDataset(Dataset):
    """
    Returns contiguous sequences of (seq_len + 1) frames and seq_len
    actions, instead of single (frame_t, action_t, frame_t+1) triples.

    This is the key fix for the noisy predictions: the previous
    dataset shuffled individual transitions and reset the hidden
    state to zero every batch, so the recurrent DynamicsModel never
    learned anything sequential. Training on contiguous sequences
    and carrying `hidden` forward within each sequence (see Trainer)
    lets the GRU actually learn temporal dependencies such as ball
    direction/velocity in Breakout.

    Only sequence *start indices* are shuffled by the DataLoader;
    the frames within a sequence stay in their original order.
    """

    def __init__(self, folder="data", seq_len=16):

        self.frames = torch.from_numpy(
            np.load(os.path.join(folder, "frames.npy"))
        ).float()

        self.actions = torch.from_numpy(
            np.load(os.path.join(folder, "actions.npy"))
        ).long()

        self.seq_len = seq_len

    def __len__(self):

        # Last valid start index leaves room for seq_len + 1 frames
        return len(self.frames) - self.seq_len - 1

    def __getitem__(self, index):

        # (seq_len + 1, 1, H, W) -- includes the frame *before* the
        # first action and the frame *after* the last action
        frames = self.frames[
            index: index + self.seq_len + 1
        ].unsqueeze(1)

        actions = self.actions[
            index: index + self.seq_len
        ]

        return frames, actions