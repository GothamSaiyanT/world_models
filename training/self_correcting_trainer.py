"""Shared trainer for adaptive and fixed-interval correction pipelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class EpochResult:
    """Summary of one completed training epoch."""

    average_loss: float
    correction_events: int
    corrected_samples: int
    mean_correction_error: float | None
    average_base_loss: float
    average_motion_loss: float
    average_foreground_loss: float
    average_motion_fraction: float
    mean_drift_error: float
    p90_drift_error: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SelfCorrectingTrainer:

    def __init__(
        self,
        model,
        dataset: Dataset,
        corrector,
        drift_detector,
        validation_dataset: Dataset | None = None,
        config=None,
        learning_rate: float | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
        optimizer: str | None = None,
        device: str | torch.device | None = None,
        warmup_epochs: int | None = None,
        motion_weight: float | None = None,
        foreground_weight: float | None = None,
        motion_threshold: float | None = None,
        foreground_threshold: float | None = None,
        motion_dilation: int | None = None,
        gradient_clip: float | None = None,
    ) -> None:
        self.model = model
        self.corrector = corrector
        self.drift_detector = drift_detector

        learning_rate = self._resolve(
            learning_rate, config, "learning_rate", 0.001
        )
        batch_size = self._resolve(
            batch_size, config, "batch_size", 32
        )
        num_workers = self._resolve(
            num_workers, config, "num_workers", 2
        )
        optimizer_name = self._resolve(
            optimizer, config, "optimizer", "adam"
        )
        requested_device = self._resolve(
            device, config, "device", None
        )

        self.warmup_epochs = int(self._resolve(
            warmup_epochs, config, "warmup_epochs", 3
        ))
        self.motion_weight = float(self._resolve(
            motion_weight, config, "motion_weight", 10.0
        ))
        self.foreground_weight = float(self._resolve(
            foreground_weight, config, "foreground_weight", 2.0
        ))
        self.motion_threshold = float(self._resolve(
            motion_threshold, config, "motion_threshold", 0.02
        ))
        self.foreground_threshold = float(self._resolve(
            foreground_threshold, config, "foreground_threshold", 0.05
        ))
        self.motion_dilation = int(self._resolve(
            motion_dilation, config, "motion_dilation", 3
        ))
        self.gradient_clip = float(self._resolve(
            gradient_clip, config, "gradient_clip", 1.0
        ))