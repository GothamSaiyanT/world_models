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

from core_nn.correction import (
    CorrectionStrategy,
)

from core_nn.drift import (
    DriftDetector,
)

from core_nn.world_model import (
    WorldModel,
)

from training.self_correcting_trainer import (
    EpochResult,
    ValidationResult,
    SelfCorrectingTrainer,
)


class SelfCorrectingPipeline(ABC):
    """
    Shared pipeline for adaptive and
    fixed-interval correction.
    """

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

        # =====================================================
        # LOAD DATASET
        # =====================================================

        self.dataset = (
            dataset
            if dataset is not None
            else self._build_dataset()
        )

        # =====================================================
        # CONTIGUOUS 80/20 TRAIN/VALIDATION SPLIT
        # =====================================================

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

        # =====================================================
        # MODEL
        # =====================================================

        self.model = WorldModel(
            model_config
        )

        # =====================================================
        # DRIFT DETECTOR
        # =====================================================

        self.drift_detector = DriftDetector(
            metric=drift_config.metric
        )

        # =====================================================
        # CORRECTION STRATEGY
        # =====================================================

        self.corrector = (
            self.build_corrector()
        )

        self.validate_pipeline()

        # =====================================================
        # TRAINER
        # =====================================================

        self.trainer = (
            SelfCorrectingTrainer(
                model=self.model,
                train_dataset=self.train_dataset,
                validation_dataset=(
                    self.validation_dataset
                ),
                corrector=self.corrector,
                drift_detector=(
                    self.drift_detector
                ),
                config=training_config,
            )
        )

        self.history: list[
            dict[str, Any]
        ] = []

    # =========================================================
    # DATA
    # =========================================================

    def _build_dataset(
        self,
    ) -> Dataset:

        from training.dataset import (
            WorldModelSequenceDataset,
        )

        return WorldModelSequenceDataset(
            folder=self.data_config.folder,
            seq_len=(
                self.data_config
                .sequence_length
            ),
        )

    def _split_dataset(
        self,
        dataset: Dataset,
    ) -> tuple[Dataset, Dataset]:
        """
        Contiguous 80/20 split.

        This avoids leakage between overlapping
        training and validation sequences.
        """

        sequence_length = (
            self.data_config
            .sequence_length
        )

        # -----------------------------------------------------
        # Normal project dataset
        # -----------------------------------------------------

        if hasattr(
            dataset,
            "frames",
        ):

            total_frames = len(
                dataset.frames
            )

            validation_frame = int(
                total_frames * 0.8
            )

            # Prevent a training sequence from
            # crossing into validation frames.
            train_end = (
                validation_frame
                - sequence_length
            )

            validation_start = (
                validation_frame
            )

        # -----------------------------------------------------
        # Fallback
        # -----------------------------------------------------

        else:

            validation_start = int(
                len(dataset) * 0.8
            )

            train_end = max(
                1,
                validation_start
                - sequence_length,
            )

        if train_end <= 0:

            raise ValueError(
                "Dataset is too small "
                "for training split."
            )

        if (
            validation_start
            >= len(dataset)
        ):

            raise ValueError(
                "Dataset is too small "
                "for validation split."
            )

        train_indices = range(
            0,
            train_end,
        )

        validation_indices = range(
            validation_start,
            len(dataset),
        )

        return (
            Subset(
                dataset,
                train_indices,
            ),
            Subset(
                dataset,
                validation_indices,
            ),
        )

    # =========================================================
    # PIPELINE-SPECIFIC CORRECTOR
    # =========================================================

    @abstractmethod
    def build_corrector(
        self,
    ) -> CorrectionStrategy:

        raise NotImplementedError

    def validate_pipeline(
        self,
    ) -> None:

        """
        Subclasses may perform
        additional validation.
        """

    # =========================================================
    # CHECKPOINT SELECTION SCORE
    # =========================================================

    @staticmethod
    def calculate_validation_score(
        validation_result:
            ValidationResult,
    ) -> float:
        """
        Select checkpoints based on the parts
        of the frame that matter most to this
        project.

        Average whole-frame MSE can look very
        good even when the tiny moving ball is
        missing.

        Therefore checkpoint selection uses:

            motion loss
            +
            foreground loss

        instead of whole-frame validation loss.
        """

        return float(
            validation_result
            .average_motion_loss
            +
            validation_result
            .average_foreground_loss
        )

    # =========================================================
    # TRAINING
    # =========================================================

    def run(
        self,
    ) -> list[dict[str, Any]]:

        checkpoint_dir = Path(
            self.training_config
            .checkpoint_folder
        )

        history_dir = Path(
            self.training_config
            .history_folder
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
            f"Pipeline: "
            f"{self.pipeline_name}"
        )

        print(
            f"Training device: "
            f"{self.trainer.device}"
        )

        # -----------------------------------------------------
        # We now select checkpoints using the
        # motion + foreground validation score.
        # -----------------------------------------------------

        best_selection_score = float(
            "inf"
        )

        best_validation_loss = float(
            "inf"
        )

        best_epoch = None

        for epoch in range(
            1,
            self.training_config.epochs
            + 1,
        ):

            # =================================================
            # TRAIN
            # =================================================

            train_result = (
                self.trainer
                .train_epoch(
                    epoch=epoch - 1
                )
            )

            # =================================================
            # VALIDATE
            # =================================================

            validation_result = (
                self.trainer
                .validate_epoch()
            )

            # =================================================
            # CHECKPOINT SELECTION SCORE
            # =================================================

            validation_score = (
                self.calculate_validation_score(
                    validation_result
                )
            )

            # =================================================
            # SAVE HISTORY
            # =================================================

            epoch_record = {

                "epoch":
                    epoch,

                "training_loss":
                    train_result
                    .average_loss,

                "validation_loss":
                    validation_result
                    .average_loss,

                "validation_motion_loss":
                    validation_result
                    .average_motion_loss,

                "validation_foreground_loss":
                    validation_result
                    .average_foreground_loss,

                "validation_selection_score":
                    validation_score,

                "training":
                    train_result
                    .to_dict(),

                "validation":
                    validation_result
                    .to_dict(),
            }

            self.history.append(
                epoch_record
            )

            # =================================================
            # REPORT RESULTS
            # =================================================

            self.report_epoch(
                epoch,
                train_result,
                validation_result,
                validation_score,
            )

            # =================================================
            # SAVE BEST MODEL
            #
            # IMPORTANT:
            # Selection uses motion + foreground
            # instead of normal validation loss.
            # =================================================

            if (
                validation_score
                < best_selection_score
            ):

                best_selection_score = (
                    validation_score
                )

                best_validation_loss = (
                    validation_result
                    .average_loss
                )

                best_epoch = epoch

                self.save_best_model(
                    epoch=epoch,
                    train_result=(
                        train_result
                    ),
                    validation_result=(
                        validation_result
                    ),
                    validation_score=(
                        validation_score
                    ),
                )

                print(
                    "  Best motion-aware "
                    "validation model saved."
                )

        # =====================================================
        # SAVE TRAINING HISTORY
        # =====================================================

        self.save_history()

        print(
            "\nTraining complete."
        )

        print(
            "Best epoch:",
            best_epoch,
        )

        print(
            "Best whole-frame "
            "validation loss:",
            f"{best_validation_loss:.6f}",
        )

        print(
            "Best motion-aware "
            "selection score:",
            f"{best_selection_score:.6f}",
        )

        return self.history

    # =========================================================
    # REPORTING
    # =========================================================

    def report_epoch(
        self,
        epoch: int,
        train_result: EpochResult,
        validation_result:
            ValidationResult,
        validation_score: float,
    ) -> None:

        mean_error_text = (
            f"{train_result.mean_correction_error:.6f}"
            if (
                train_result
                .mean_correction_error
                is not None
            )
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
            f"  Validation Motion Loss:     "
            f"{validation_result.average_motion_loss:.6f}"
        )

        print(
            f"  Validation Foreground Loss: "
            f"{validation_result.average_foreground_loss:.6f}"
        )

        print(
            f"  Validation Selection Score: "
            f"{validation_score:.6f}"
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
            f"  Validation P90 Drift:       "
            f"{validation_result.p90_drift_error:.6f}"
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

        """
        Optional subclass hook.
        """

    # =========================================================
    # CHECKPOINTING
    # =========================================================

    def save_best_model(
        self,
        epoch: int,
        train_result: EpochResult,
        validation_result:
            ValidationResult,
        validation_score: float,
    ) -> None:

        checkpoint_dir = Path(
            self.training_config
            .checkpoint_folder
        )

        # -----------------------------------------------------
        # Simple model weights
        # Used by render scripts.
        # -----------------------------------------------------

        torch.save(
            self.model.state_dict(),
            checkpoint_dir
            / self.model_filename,
        )

        # -----------------------------------------------------
        # Full checkpoint
        # Useful for reproducibility / research results.
        # -----------------------------------------------------

        torch.save(
            {
                "pipeline":
                    self.pipeline_name,

                "epoch":
                    epoch,

                "training_loss":
                    train_result
                    .average_loss,

                "validation_loss":
                    validation_result
                    .average_loss,

                "validation_motion_loss":
                    validation_result
                    .average_motion_loss,

                "validation_foreground_loss":
                    validation_result
                    .average_foreground_loss,

                "validation_selection_score":
                    validation_score,

                "training_metrics":
                    train_result
                    .to_dict(),

                "validation_metrics":
                    validation_result
                    .to_dict(),

                "model_state_dict":
                    self.model
                    .state_dict(),

                "optimizer_state_dict":
                    self.trainer
                    .optimizer
                    .state_dict(),

                "configs":
                    serialise_configs(
                        data=(
                            self.data_config
                        ),
                        model=(
                            self.model_config
                        ),
                        drift=(
                            self.drift_config
                        ),
                        training=(
                            self.training_config
                        ),
                    ),
            },

            checkpoint_dir
            / self.checkpoint_filename,
        )

    # =========================================================
    # HISTORY
    # =========================================================

    def save_history(
        self,
    ) -> None:

        history_path = (
            Path(
                self.training_config
                .history_folder
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