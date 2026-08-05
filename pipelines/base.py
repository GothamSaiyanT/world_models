from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

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
    """Template method for setup, training, reporting, and checkpointing."""

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
        self.model = WorldModel(model_config)
        self.drift_detector = DriftDetector(metric=drift_config.metric)
        self.corrector = self.build_corrector()
        self.validate_pipeline()

        self.trainer = SelfCorrectingTrainer(
            model=self.model,
            dataset=self.dataset,
            corrector=self.corrector,
            drift_detector=self.drift_detector,
            config=training_config,
        )
        self.history: list[dict[str, Any]] = []

    def _build_dataset(self) -> Dataset:
        # Lazy import lets these OOP classes be tested independently. In the
        # user's existing project, keep training/dataset.py in its current place.
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

    @abstractmethod
    def build_corrector(self) -> CorrectionStrategy:
        raise NotImplementedError

    def validate_pipeline(self) -> None:
        """Subclasses may add pipeline-specific configuration checks."""

    def run(self) -> list[dict[str, Any]]:
        checkpoint_dir = Path(self.training_config.checkpoint_folder)
        history_dir = Path(self.training_config.history_folder)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        history_dir.mkdir(parents=True, exist_ok=True)

        print(f"Pipeline: {self.pipeline_name}")
        print(f"Training device: {self.trainer.device}")

        best_loss = float("inf")
        for epoch in range(1, self.training_config.epochs + 1):
            result = self.trainer.train_epoch()
            epoch_record = {"epoch": epoch, **result.to_dict()}
            self.history.append(epoch_record)
            self.report_epoch(epoch, result)

            if result.average_loss < best_loss:
                best_loss = result.average_loss
                self.save_best_model(epoch, result)
                print("  Best model updated.")

        self.save_history()
        print(f"Training complete. Best loss: {best_loss:.6f}")
        return self.history

    def report_epoch(self, epoch: int, result: EpochResult) -> None:
        mean_error_text = (
            f"{result.mean_correction_error:.6f}"
            if result.mean_correction_error is not None
            else "n/a"
        )
        print(
            f"Epoch {epoch}/{self.training_config.epochs} | "
            f"loss={result.average_loss:.6f} | "
            f"events={result.correction_events} | "
            f"samples_corrected={result.corrected_samples} | "
            f"mean_correction_error={mean_error_text}"
        )
        self.after_epoch(result)

    def after_epoch(self, result: EpochResult) -> None:
        """Optional subclass hook for warnings or specialised reporting."""

    def save_best_model(self, epoch: int, result: EpochResult) -> None:
        checkpoint_dir = Path(self.training_config.checkpoint_folder)

        # Raw state_dict keeps compatibility with rendering code that expects
        # torch.load(path) to return model weights directly.
        torch.save(
            self.model.state_dict(),
            checkpoint_dir / self.model_filename,
        )

        torch.save(
            {
                "pipeline": self.pipeline_name,
                "epoch": epoch,
                "best_loss": result.average_loss,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.trainer.optimizer.state_dict(),
                "configs": serialise_configs(
                    data=self.data_config,
                    model=self.model_config,
                    drift=self.drift_config,
                    training=self.training_config,
                ),
            },
            checkpoint_dir / self.checkpoint_filename,
        )

    def save_history(self) -> None:
        history_path = Path(self.training_config.history_folder) / self.history_filename
        with history_path.open("w", encoding="utf-8") as file:
            json.dump(self.history, file, indent=2)
