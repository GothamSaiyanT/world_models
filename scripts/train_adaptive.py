"""Entry point for Pipeline 1: adaptive self-correction.

Run from the project root with:
    python -m scripts.train_adaptive
"""

from config import (
    DataConfig,
    DriftConfig,
    ModelConfig,
    TrainingConfig,
)
from pipelines import AdaptivePipeline


def main() -> None:
    """Configure and train the adaptive self-correcting pipeline."""

    pipeline = AdaptivePipeline(
        data_config=DataConfig(
            folder="data",
            sequence_length=16,
        ),
        model_config=ModelConfig(
            latent_size=128,
            hidden_size=128,
            image_size=64,
            input_channels=1,
            num_actions=4,
            action_embedding_size=32,
        ),
        drift_config=DriftConfig(
            metric="mse",
            # Raised from 0.01: an untrained model's per-step error
            # starts around 0.16-0.17, so 0.01 fired on ~100% of
            # samples from epoch 1 (see training logs: events=4992,
            # samples_corrected=159728 -- literally every sample at
            # every step). That severed the GRU's gradient chain at
            # every step and prevented it from ever learning
            # multi-step dynamics. 0.08 stays below typical
            # untrained error, so correction stays rare during
            # warm-up, but is low enough to catch real drift once
            # the model predicts reasonably well. Re-tune once you
            # see your own post-warmup error distribution.
            adaptive_threshold=0.05,
            fixed_interval=10,
        ),
        training_config=TrainingConfig(
            epochs=70,
            learning_rate=0.001,
            batch_size=32,
            num_workers=2,
            # Switched from "sgd" -- already working well for the
            # baseline, and this training loop (autoregressive +
            # correction) has a messier gradient landscape than the
            # baseline's teacher-forced loop, so Adam's adaptive
            # step sizes matter more here, not less.
            optimizer="adam",
            device=None,
            checkpoint_folder="models",
            history_folder="results",
        ),
    )

    pipeline.run()


if __name__ == "__main__":
    main()