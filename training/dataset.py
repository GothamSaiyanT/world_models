import os
import numpy as np
import torch
from torch.utils.data import Dataset

class WorldModelDataset(Dataset):

    def __init__(self, folder="data"):

        self.frames = np.load(
            os.path.join(folder, "frames.npy")
        )

        self.actions = np.load(
            os.path.join(folder, "actions.npy")
        )

    def __len__(self):

        return len(self.frames) - 1

    def __getitem__(self, index):

        current_frame = torch.tensor(
            self.frames[index],
            dtype=torch.float32
        ).unsqueeze(0)

        next_frame = torch.tensor(
            self.frames[index + 1],
            dtype=torch.float32
        ).unsqueeze(0)

        action = torch.tensor(
            self.actions[index],
            dtype=torch.long
        )

        return (
            current_frame,
            action,
            next_frame
        )