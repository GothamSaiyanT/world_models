from core.correction import (
    CorrectionStrategy,
    FixedIntervalCorrector,
)
from pipelines.base import SelfCorrectingPipeline
from training.self_correcting_trainer import EpochResult


class FixedIntervalPipeline(SelfCorrectingPipeline):
    """Pipeline 2: scheduled correction after a fixed number of steps."""

    pipeline_name = "Pipeline 2 - Fixed Interval Correction"
    model_filename = "best_fixed_interval_model.pt"
    checkpoint_filename = "best_fixed_interval_checkpoint.pt"
    history_filename = "fixed_interval_history.json"

    def build_corrector(self) -> CorrectionStrategy:
        return FixedIntervalCorrector(
            interval=self.drift_config.fixed_interval,
        )

    def validate_pipeline(self) -> None:
        if self.drift_config.fixed_interval > self.data_config.sequence_length:
            print(
                "Warning: fixed_interval is greater than sequence_length; "
                "no scheduled correction will occur within a rollout."
            )

    def after_epoch(self, result: EpochResult) -> None:
        if result.correction_events == 0:
            print("  Warning: no fixed-interval correction occurred this epoch.")
