import os
import numpy as np
import torch
from torch.utils.data import Dataset


class WorldModelDataset(Dataset):

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