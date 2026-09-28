from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Subset

from config import (
    DataConfig,
    DriftConfig,
    ModelConfig,
    TrainingConfig,
    serialise_configs,
)
from core_nn.correction import CorrectionStrategy
from core_nn.drift import DriftDetector
from core_nn.world_model import WorldModel
from training.self_correcting_trainer import EpochResult, SelfCorrectingTrainer


class SelfCorrectingPipeline(ABC):
    """Template method for setup, training, validation, reporting, and checkpointing."""

    pipeline_name: str
    model_filename: str
    checkpoint_filename: str
    history_filename: str

    def __init__(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        drift_config: DriftConfig,
        training_config: TrainingConfig,
        dataset: Dataset | None = None,
    ) -> None:
        self.data_config = data_config
        self.model_config = model_config
        self.drift_config = drift_config
        self.training_config = training_config

        self.dataset = dataset if dataset is not None else self._build_dataset()

        self.train_dataset, self.validation_dataset = self._split_dataset(
            self.dataset
        )

        print("Training sequences:", len(self.train_dataset))
        print("Validation sequences:", len(self.validation_dataset))

        self.model = WorldModel(model_config)
        self.drift_detector = DriftDetector(metric=drift_config.metric)
        self.corrector = self.build_corrector()
        self.validate_pipeline()

        self.trainer = SelfCorrectingTrainer(
            model=self.model,
            dataset=self.train_dataset,
            validation_dataset=self.validation_dataset,
            corrector=self.corrector,
            drift_detector=self.drift_detector,
            config=training_config,
        )
        self.history: list[dict[str, Any]] = []

    def _build_dataset(self) -> Dataset:
        try:
            from training.dataset import WorldModelSequenceDataset
        except ImportError as exc:
            raise ImportError(
                "Could not import training.dataset.WorldModelSequenceDataset. "
                "Place this refactor at your project root beside training/dataset.py."
            ) from exc

        return WorldModelSequenceDataset(
            folder=self.data_config.folder,
            seq_len=self.data_config.sequence_length,
        )

    def _split_dataset(
        self,
        dataset: Dataset,
    ) -> tuple[Dataset, Dataset]:
        """
        Create a contiguous 80/20 split.

        Because sequence samples overlap in time, random_split() is avoided.
        A sequence-length gap is left before validation to prevent leakage.
        """
        sequence_length = self.data_config.sequence_length

        total_frames = len(dataset) + sequence_length
        split_frame = int(total_frames * 0.8)

        train_end = split_frame - sequence_length