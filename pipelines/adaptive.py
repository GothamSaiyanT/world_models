from core_nn.correction import AdaptiveCorrector, CorrectionStrategy
from pipelines.base import SelfCorrectingPipeline
from training.self_correcting_trainer import EpochResult


class AdaptivePipeline(SelfCorrectingPipeline):
    """Pipeline 3: error-triggered, per-sample correction."""

    pipeline_name = "Pipeline 3 - Adaptive Drift Correction"
    model_filename = "best_adaptive_model.pt"
    checkpoint_filename = "best_adaptive_checkpoint.pt"
    history_filename = "adaptive_history.json"

    def build_corrector(self) -> CorrectionStrategy:
        return AdaptiveCorrector(
            threshold=self.drift_config.adaptive_threshold,
        )

    def after_epoch(self, result: EpochResult) -> None:
        if result.correction_events == 0:
            print(
                "  Warning: no adaptive correction occurred; the threshold "
                "may be too high for the current error scale."
            )
