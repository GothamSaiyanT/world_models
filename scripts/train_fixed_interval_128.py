"""Train the fixed-interval self-correcting pipeline at 128x128."""

import argparse

from config import DataConfig, DriftConfig, ModelConfig, TrainingConfig
from highres_128 import (
    DATA_FOLDER, IMAGE_SIZE, MODEL_FOLDER, RESULTS_FOLDER, SEQUENCE_LENGTH,
    validate_128_dataset,
)
from pipelines import FixedIntervalPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--data-folder", default=DATA_FOLDER)
    parser.add_argument("--model-folder", default=MODEL_FOLDER)
    parser.add_argument("--results-folder", default=RESULTS_FOLDER)
    args = parser.parse_args()

    frame_shape, action_shape = validate_128_dataset(args.data_folder)
    print("Dataset frames:", frame_shape)
    print("Dataset actions:", action_shape)

    pipeline = FixedIntervalPipeline(
        data_config=DataConfig(
            folder=args.data_folder,
            sequence_length=SEQUENCE_LENGTH,
        ),
        model_config=ModelConfig(
            latent_size=128,
            hidden_size=128,
            image_size=IMAGE_SIZE,
            input_channels=1,
            num_actions=4,
            action_embedding_size=32,
        ),
        drift_config=DriftConfig(
            metric="mse",
            adaptive_threshold=0.05,
            fixed_interval=args.interval,
        ),
        training_config=TrainingConfig(
            epochs=args.epochs,
            learning_rate=0.001,
            batch_size=args.batch_size,
            num_workers=2,
            optimizer="adam",
            device=None,
            checkpoint_folder=args.model_folder,
            history_folder=args.results_folder,
        ),
    )
    pipeline.run()


if __name__ == "__main__":
    main()
