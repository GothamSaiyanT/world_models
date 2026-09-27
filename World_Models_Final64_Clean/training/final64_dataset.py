import os

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceRangeDataset(Dataset):
    """
    Contiguous frame/action sequences for Final64.

    A sequence is accepted only when every frame belongs to
    the same episode. This prevents the GRU from being trained
    across artificial environment-reset boundaries.
    """

    def __init__(
        self,
        folder,
        start,
        end,
        sequence_length=16,
    ):
        folder = str(folder)

        self.frames_path = os.path.join(
            folder,
            "frames.npy",
        )

        self.actions_path = os.path.join(
            folder,
            "actions.npy",
        )

        self.episode_ids_path = os.path.join(
            folder,
            "episode_ids.npy",
        )

        if not os.path.exists(
            self.frames_path
        ):
            raise FileNotFoundError(
                self.frames_path
            )

        if not os.path.exists(
            self.actions_path
        ):
            raise FileNotFoundError(
                self.actions_path
            )

        if not os.path.exists(
            self.episode_ids_path
        ):
            raise FileNotFoundError(
                self.episode_ids_path
            )

        self.frames = np.load(
            self.frames_path,
            mmap_mode="r",
        )

        self.actions = np.load(
            self.actions_path,
            mmap_mode="r",
        )

        self.episode_ids = np.load(
            self.episode_ids_path,
            mmap_mode="r",
        )

        if not (
            len(self.frames)
            == len(self.actions)
            == len(self.episode_ids)
        ):
            raise ValueError(
                "frames.npy, actions.npy and "
                "episode_ids.npy must have equal length."
            )

        self.start = int(start)
        self.end = int(end)

        self.sequence_length = int(
            sequence_length
        )

        if (
            self.start < 0
            or self.end > len(self.frames)
        ):
            raise ValueError(
                "Dataset range is outside "
                "the saved frames."
            )

        if (
            self.end - self.start
            <= self.sequence_length
        ):
            raise ValueError(
                "Dataset range is too short "
                "for the requested sequence length."
            )

        # -----------------------------------------------------
        # Build valid starting positions
        # -----------------------------------------------------
        #
        # For sequence length 16 we require:
        #
        # frame[t] ... frame[t+16]
        #
        # to all belong to exactly the same episode.
        # -----------------------------------------------------

        self.valid_indices = []

        last_possible_start = (
            self.end
            - self.sequence_length
        )

        for base in range(
            self.start,
            last_possible_start,
        ):
            stop = (
                base
                + self.sequence_length
                + 1
            )

            episode_slice = (
                self.episode_ids[
                    base:stop
                ]
            )

            if (
                len(episode_slice)
                != self.sequence_length + 1
            ):
                continue

            if np.all(
                episode_slice
                == episode_slice[0]
            ):
                self.valid_indices.append(
                    base
                )

        if not self.valid_indices:
            raise RuntimeError(
                "No valid within-episode sequences "
                "were found for this dataset range."
            )

        print(
            f"SequenceRangeDataset "
            f"[{self.start}:{self.end}] | "
            f"valid sequences: "
            f"{len(self.valid_indices):,}"
        )

    def __len__(self):
        return len(
            self.valid_indices
        )

    def __getitem__(self, index):

        base = self.valid_indices[
            int(index)
        ]

        stop = (
            base
            + self.sequence_length
            + 1
        )

        frame_np = np.asarray(
            self.frames[
                base:stop
            ]
        ).copy()

        action_np = np.asarray(
            self.actions[
                base:
                base + self.sequence_length
            ]
        ).copy()

        frame_tensor = torch.from_numpy(
            frame_np
        ).float()

        if frame_tensor.max() > 1.0:
            frame_tensor = (
                frame_tensor / 255.0
            )

        frame_tensor = (
            frame_tensor.unsqueeze(1)
        )

        action_tensor = (
            torch.from_numpy(
                action_np
            ).long()
        )

        return (
            frame_tensor,
            action_tensor,
        )


def load_normalized_frames(folder):

    frames = np.load(
        os.path.join(
            str(folder),
            "frames.npy",
        )
    )

    frames = frames.astype(
        np.float32
    )

    if frames.max() > 1.0:
        frames /= 255.0

    return frames


def load_actions(folder):

    return np.load(
        os.path.join(
            str(folder),
            "actions.npy",
        )
    ).astype(np.int64)


def load_episode_ids(folder):

    return np.load(
        os.path.join(
            str(folder),
            "episode_ids.npy",
        )
    ).astype(np.int64)