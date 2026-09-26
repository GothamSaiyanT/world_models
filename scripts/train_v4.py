from __future__ import annotations

import argparse

from training.v4_dataset import V4SequenceDataset, dataset_split_points
from training.v4_trainer import V4Trainer
from v4.config import V4Config
from v4.world_model import WorldModelV4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline",
        required=True,
        choices=["baseline", "fixed_interval", "adaptive"],
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--data-folder", default="data_v4_64")
    parser.add_argument("--model-folder", default="models_v4_64")
    parser.add_argument("--results-folder", default="results_v4_64")
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--interval", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "tpu", "cpu"],
    )
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    total_frames, split = dataset_split_points(args.data_folder, 0.8)
    print("Total frames:", total_frames)
    print("Train frames:", split)
    print("Validation frames:", total_frames - split)

    train_dataset = V4SequenceDataset(
        args.data_folder,
        args.sequence_length,
        start=0,
        end=split,
    )
    val_dataset = V4SequenceDataset(
        args.data_folder,
        args.sequence_length,
        start=split,
        end=total_frames,
    )

    config = V4Config(sequence_length=args.sequence_length)
    model = WorldModelV4(config)
    parameters = sum(p.numel() for p in model.parameters())
    print("Pipeline:", args.pipeline)
    print("Model parameters:", f"{parameters:,}")

    trainer = V4Trainer(
        model,
        train_dataset,
        val_dataset,
        pipeline=args.pipeline,
        output_folder=args.model_folder,
        results_folder=args.results_folder,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        fixed_interval=args.interval,
        device=args.device,
        num_workers=args.num_workers,
        patience=args.patience,
    )
    trainer.train(args.epochs)


if __name__ == "__main__":
    main()
