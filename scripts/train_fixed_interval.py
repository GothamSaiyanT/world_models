"""Entry point for Pipeline 2: fixed-interval correction."""

from config import DataConfig, DriftConfig, ModelConfig, TrainingConfig
from pipelines import FixedIntervalPipeline


def main() -> None:
    pipeline = FixedIntervalPipeline(
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
            # adaptive_threshold is unused by FixedIntervalCorrector
            # (it only reads fixed_interval), but kept at the same
            # 0.08 as the adaptive pipeline for consistent logging
            # if you ever compare the two DriftConfigs directly.
            adaptive_threshold=0.08,
            fixed_interval=10,
        ),
        training_config=TrainingConfig(
            epochs=70,
            learning_rate=0.001,
            batch_size=32,
            num_workers=2,
            # Switched from "sgd" to match the baseline's working
            # setup and to cope with the harder autoregressive
            # training loop.
            optimizer="adam",
            device=None,
            checkpoint_folder="models",
            history_folder="results",
        ),
    )

    pipeline.run()


if __name__ == "__main__":
    main()