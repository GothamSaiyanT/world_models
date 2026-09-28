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
from training.splits import split_train_validation


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
        self.train_dataset, self.validation_dataset = split_train_validation(
            self.dataset,
            validation_fraction=data_config.validation_fraction,
            gap=data_config.sequence_length,
        )
        self.model = WorldModel(model_config)
        self.drift_detector = DriftDetector(metric=drift_config.metric)
        self.corrector = self.build_corrector()
        self.validate_pipeline()

        self.trainer = SelfCorrectingTrainer(
            model=self.model,
            dataset=self.train_dataset,
            corrector=self.corrector,
            drift_detector=self.drift_detector,
            config=training_config,
            validation_dataset=self.validation_dataset,
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

        if self.trainer.has_validation:
            print(
                f"Training sequences: {len(self.train_dataset)} | "
                f"Validation sequences: {len(self.validation_dataset)}"
            )
            selected_on = "validation"
        else:
            print("Validation is off: the best model is chosen on training loss.")
            selected_on = "training"

        best_loss = float("inf")
        for epoch in range(1, self.training_config.epochs + 1):
            result = self.trainer.train_epoch()

            val_result = None
            if self.trainer.has_validation:
                val_result = self.trainer.validate()

            epoch_record = {"epoch": epoch, **result.to_dict()}
            if val_result is not None:
                for name, value in val_result.to_dict().items():
                    epoch_record[f"val_{name}"] = value
            self.history.append(epoch_record)
            self.report_epoch(epoch, result, val_result)

            # Judge each epoch on data the model has not trained on
            # whenever we have it.
            score = (
                val_result.average_loss
                if val_result is not None
                else result.average_loss
            )
            if score < best_loss:
                best_loss = score
                self.save_best_model(epoch, best_loss, selected_on)
                print("  Best model updated.")

        self.save_history()
        print(f"Training complete. Best {selected_on} loss: {best_loss:.6f}")
        return self.history

    def report_epoch(
        self,
        epoch: int,
        result: EpochResult,
        val_result: EpochResult | None = None,
    ) -> None:
        mean_error_text = (
            f"{result.mean_correction_error:.6f}"
            if result.mean_correction_error is not None
            else "n/a"
        )
        val_text = (
            f"val_loss={val_result.average_loss:.6f} | "
            if val_result is not None
            else ""
        )
        print(
            f"Epoch {epoch}/{self.training_config.epochs} | "
            f"loss={result.average_loss:.6f} | "
            f"{val_text}"
            f"events={result.correction_events} | "
            f"samples_corrected={result.corrected_samples} | "
            f"mean_correction_error={mean_error_text}"
        )
        self.after_epoch(result)

    def after_epoch(self, result: EpochResult) -> None:
        """Optional subclass hook for warnings or specialised reporting."""

    def save_best_model(
        self,
        epoch: int,
        best_loss: float,
        selected_on: str,
    ) -> None:
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
                "best_loss": best_loss,
                "selected_on": selected_on,
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
