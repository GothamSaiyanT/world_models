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
from training.self_correcting_trainer import (
    EpochResult,
    SelfCorrectingTrainer,
)


class SelfCorrectingPipeline(ABC):
    """Shared base pipeline for fixed-interval and adaptive models."""

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

        self.dataset = (
            dataset
            if dataset is not None
            else self._build_dataset()
        )

        (
            self.train_dataset,
            self.validation_dataset,
        ) = self._split_dataset(
            self.dataset
        )

        print(
            "Training sequences:",
            len(self.train_dataset),
        )

        print(
            "Validation sequences:",
            len(self.validation_dataset),
        )

        self.model = WorldModel(
            model_config
        )

        self.drift_detector = DriftDetector(
            metric=drift_config.metric
        )

        self.corrector = (
            self.build_corrector()
        )

        self.validate_pipeline()

        self.trainer = SelfCorrectingTrainer(
            model=self.model,
            dataset=self.train_dataset,
            validation_dataset=(
                self.validation_dataset
            ),
            corrector=self.corrector,
            drift_detector=(
                self.drift_detector
            ),
            config=training_config,
        )

        self.history: list[
            dict[str, Any]
        ] = []

    def _build_dataset(
        self,
    ) -> Dataset:
        try:
            from training.dataset import (
                WorldModelSequenceDataset,
            )
        except ImportError as exc:
            raise ImportError(
                "Could not import "
                "training.dataset.WorldModelSequenceDataset."
            ) from exc

        return WorldModelSequenceDataset(
            folder=self.data_config.folder,
            seq_len=(
                self.data_config.sequence_length
            ),
        )

    def _split_dataset(
        self,
        dataset: Dataset,
    ) -> tuple[Dataset, Dataset]:
        """
        Contiguous 80/20 split.

        WorldModelSequenceDataset currently has 9983 sequences
        for 10,000 frames with sequence length 16, meaning its
        length is total_frames - sequence_length - 1.

        The split is therefore based on the original 10,000 frames:
        training sequence starts:   0..7983
        validation sequence starts: 8000..9982

        The omitted starts 7984..7999 form a 16-step safety gap.
        """

        sequence_length = (
            self.data_config.sequence_length
        )

        dataset_length = len(dataset)

        # Dataset length = frames - seq_len - 1
        total_frames = (
            dataset_length
            + sequence_length
            + 1
        )

        split_frame = int(
            total_frames * 0.8
        )

        train_end = (
            split_frame
            - sequence_length
        )

        validation_start = (
            split_frame
        )

        print(
            "Dataset sequences:",
            dataset_length,
        )
        print(
            "Estimated total frames:",
            total_frames,
        )
        print(
            "Split frame:",
            split_frame,
        )
        print(
            "Train end:",
            train_end,
        )
        print(
            "Validation start:",
            validation_start,
        )

        if train_end <= 0:
            raise ValueError(
                "Training split is empty."
            )

        if validation_start >= dataset_length:
            raise ValueError(
                "Validation split is empty."
            )

        train_dataset = Subset(
            dataset,
            range(
                0,
                train_end,
            ),
        )

        validation_dataset = Subset(
            dataset,
            range(
                validation_start,
                dataset_length,
            ),
        )

        return (
            train_dataset,
            validation_dataset,
        )

    @abstractmethod
    def build_corrector(
        self,
    ) -> CorrectionStrategy:
        raise NotImplementedError

    def validate_pipeline(
        self,
    ) -> None:
        """Subclasses may add pipeline-specific checks."""

    def run(
        self,
    ) -> list[dict[str, Any]]:
        checkpoint_dir = Path(
            self.training_config.checkpoint_folder
        )

        history_dir = Path(
            self.training_config.history_folder
        )

        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        history_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"Pipeline: {self.pipeline_name}"
        )

        print(
            f"Training device: "
            f"{self.trainer.device}"
        )

        best_validation_loss = float(
            "inf"
        )

        best_epoch = None

        for epoch in range(
            1,
            self.training_config.epochs + 1,
        ):
            train_result = (
                self.trainer.train_epoch(
                    epoch=epoch - 1
                )
            )

            validation_result = (
                self.trainer.validate_epoch()
            )

            epoch_record = {
                "epoch": epoch,
                "training_loss":
                    train_result.average_loss,
                "validation_loss":
                    validation_result.average_loss,
                "training_base_loss":
                    train_result.average_base_loss,
                "validation_base_loss":
                    validation_result.average_base_loss,
                "training_motion_loss":
                    train_result.average_motion_loss,
                "validation_motion_loss":
                    validation_result.average_motion_loss,
                "training_foreground_loss":
                    train_result.average_foreground_loss,
                "validation_foreground_loss":
                    validation_result.average_foreground_loss,
                "training_mean_drift":
                    train_result.mean_drift_error,
                "validation_mean_drift":
                    validation_result.mean_drift_error,
                "training_p90_drift":
                    train_result.p90_drift_error,
                "validation_p90_drift":
                    validation_result.p90_drift_error,
                "correction_events":
                    train_result.correction_events,
                "corrected_samples":
                    train_result.corrected_samples,
                "mean_correction_error":
                    train_result.mean_correction_error,
            }

            self.history.append(
                epoch_record
            )

            self.report_epoch(
                epoch,
                train_result,
                validation_result,
            )

            if (
                validation_result.average_loss
                < best_validation_loss
            ):
                best_validation_loss = (
                    validation_result.average_loss
                )

                best_epoch = epoch

                self.save_best_model(
                    epoch=epoch,
                    train_result=train_result,
                    validation_result=(
                        validation_result
                    ),
                )

                print(
                    "  Best validation "
                    "model updated."
                )

        self.save_history()

        print(
            "\nTraining complete."
        )

        print(
            "Best epoch:",
            best_epoch,
        )

        print(
            "Best validation loss:",
            f"{best_validation_loss:.6f}",
        )

        return self.history

    def report_epoch(
        self,
        epoch: int,
        train_result: EpochResult,
        validation_result: EpochResult,
    ) -> None:
        mean_error_text = (
            f"{train_result.mean_correction_error:.6f}"
            if train_result.mean_correction_error is not None
            else "n/a"
        )

        print(
            f"\nEpoch "
            f"{epoch}/"
            f"{self.training_config.epochs}"
        )

        print(
            f"  Training Loss:              "
            f"{train_result.average_loss:.6f}"
        )

        print(
            f"  Validation Loss:            "
            f"{validation_result.average_loss:.6f}"
        )

        print(
            f"  Training Motion Loss:       "
            f"{train_result.average_motion_loss:.6f}"
        )

        print(
            f"  Validation Motion Loss:     "
            f"{validation_result.average_motion_loss:.6f}"
        )

        print(
            f"  Training Foreground Loss:   "
            f"{train_result.average_foreground_loss:.6f}"
        )

        print(
            f"  Validation Foreground Loss: "
            f"{validation_result.average_foreground_loss:.6f}"
        )

        print(
            f"  Training Mean Drift:        "
            f"{train_result.mean_drift_error:.6f}"
        )

        print(
            f"  Validation Mean Drift:      "
            f"{validation_result.mean_drift_error:.6f}"
        )

        print(
            f"  Correction events:          "
            f"{train_result.correction_events}"
        )

        print(
            f"  Samples corrected:          "
            f"{train_result.corrected_samples}"
        )

        print(
            f"  Mean correction error:      "
            f"{mean_error_text}"
        )

        self.after_epoch(
            train_result
        )

    def after_epoch(
        self,
        result: EpochResult,
    ) -> None:
        """Optional subclass hook."""

    def save_best_model(
        self,
        epoch: int,
        train_result: EpochResult,
        validation_result: EpochResult,
    ) -> None:
        checkpoint_dir = Path(
            self.training_config.checkpoint_folder
        )

        # Raw state_dict for current render.py compatibility.
        torch.save(
            self.model.state_dict(),
            checkpoint_dir / self.model_filename,
        )

        # Full checkpoint for reproducibility.
        torch.save(
            {
                "pipeline":
                    self.pipeline_name,
                "epoch":
                    epoch,
                "best_loss":
                    validation_result.average_loss,
                "best_validation_loss":
                    validation_result.average_loss,
                "training_loss":
                    train_result.average_loss,
                "validation_loss":
                    validation_result.average_loss,
                "training_metrics":
                    train_result.to_dict(),
                "validation_metrics":
                    validation_result.to_dict(),
                "model_state_dict":
                    self.model.state_dict(),
                "optimizer_state_dict":
                    self.trainer.optimizer.state_dict(),
                "configs":
                    serialise_configs(
                        data=self.data_config,
                        model=self.model_config,
                        drift=self.drift_config,
                        training=(
                            self.training_config
                        ),
                    ),
            },
            checkpoint_dir
            / self.checkpoint_filename,
        )

    def save_history(
        self,
    ) -> None:
        history_path = (
            Path(
                self.training_config.history_folder
            )
            / self.history_filename
        )

        with history_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.history,
                file,
                indent=2,
            )
