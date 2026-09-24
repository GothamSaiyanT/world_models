from __future__ import annotations

import argparse

from training.v3_dataset import V3SequenceDataset, dataset_split_points
from training.v3_trainer import V3Trainer
from v3.config import V3Config
from v3.world_model import WorldModelV3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline",
        required=True,
        choices=["baseline", "fixed_interval", "adaptive"],
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--data-folder", default="data_v3")
    parser.add_argument("--model-folder", default="models_v3")
    parser.add_argument("--results-folder", default="results_v3")
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--interval", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()

    total_frames, split = dataset_split_points(args.data_folder, 0.8)
    print("Total frames:", total_frames)
    print("Train frames:", split)
    print("Validation frames:", total_frames - split)

    train_dataset = V3SequenceDataset(
        args.data_folder,
        args.sequence_length,
        start=0,
        end=split,
    )
    val_dataset = V3SequenceDataset(
        args.data_folder,
        args.sequence_length,
        start=split,
        end=total_frames,
    )

    config = V3Config(sequence_length=args.sequence_length)
    model = WorldModelV3(config)
    parameters = sum(p.numel() for p in model.parameters())
    print("Pipeline:", args.pipeline)
    print("Model parameters:", f"{parameters:,}")

    trainer = V3Trainer(
        model,
        train_dataset,
        val_dataset,
        pipeline=args.pipeline,
        output_folder=args.model_folder,
        results_folder=args.results_folder,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        fixed_interval=args.interval,
    )
    trainer.train(args.epochs)


if __name__ == "__main__":
    main()
