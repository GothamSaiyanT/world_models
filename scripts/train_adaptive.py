"""Entry point for Pipeline 3: adaptive self-correction."""

from config import (
    DataConfig,
    DriftConfig,
    ModelConfig,
    TrainingConfig,
)

from pipelines import AdaptivePipeline


def main() -> None:

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
            adaptive_threshold=0.05,
            fixed_interval=10,
        ),

        training_config=TrainingConfig(
            epochs=20,
            learning_rate=0.001,
            batch_size=32,
            num_workers=2,
            optimizer="adam",
            device=None,
            checkpoint_folder="models",
            history_folder="results",
        ),
    )

    pipeline.run()


if __name__ == "__main__":
    main()