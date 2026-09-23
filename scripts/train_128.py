"""Train a fresh baseline world model on native 128x128 observations."""

import argparse
import os

from core.world_model import WorldModel
from highres_128 import (
    DATA_FOLDER, IMAGE_SIZE, MODEL_FOLDER, SEQUENCE_LENGTH, validate_128_dataset
)
from training.data_collector import DataCollector
from training.dataset import WorldModelSequenceDataset
from training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--data-folder", default=DATA_FOLDER)
    parser.add_argument("--model-folder", default=MODEL_FOLDER)
    parser.add_argument(
        "--collect-if-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    os.makedirs(args.data_folder, exist_ok=True)
    os.makedirs(args.model_folder, exist_ok=True)

    frames_path = os.path.join(args.data_folder, "frames.npy")
    actions_path = os.path.join(args.data_folder, "actions.npy")

    if not (os.path.exists(frames_path) and os.path.exists(actions_path)):
        if not args.collect_if_missing:
            raise FileNotFoundError(
                "128x128 data is missing. Run python -m scripts.collect_128 first."
            )
        print(f"Collecting {args.steps} fresh {IMAGE_SIZE}x{IMAGE_SIZE} frames...")
        DataCollector(
            image_size=IMAGE_SIZE,
            save_folder=args.data_folder,
        ).collect(num_steps=args.steps)

    frame_shape, action_shape = validate_128_dataset(args.data_folder)
    print("Dataset frames:", frame_shape)
    print("Dataset actions:", action_shape)

    dataset = WorldModelSequenceDataset(
        folder=args.data_folder,
        seq_len=SEQUENCE_LENGTH,
    )

    model = WorldModel(
        latent_size=128,
        hidden_size=128,
        image_size=IMAGE_SIZE,
    )

    checkpoint = os.path.join(args.model_folder, "best_world_model.npz")
    if os.path.exists(checkpoint):
        os.remove(checkpoint)
        print("Previous 128x128 baseline checkpoint deleted.")

    trainer = Trainer(
        model=model,
        dataset=dataset,
        learning_rate=0.001,
        batch_size=args.batch_size,
    )

    best_loss = float("inf")
    print(f"Training baseline at {IMAGE_SIZE}x{IMAGE_SIZE}")
    print("Training sequences:", len(dataset))
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)

    for epoch in range(args.epochs):
        loss = trainer.train_epoch()
        print(f"Epoch {epoch + 1}/{args.epochs} Loss: {loss:.6f}")

        if loss < best_loss:
            best_loss = loss
            model.save(checkpoint, epoch=epoch + 1, best_loss=best_loss)
            print("Best 128x128 baseline model saved.")

    print("Training complete.")
    print(f"Best loss: {best_loss:.6f}")
    print("Checkpoint:", checkpoint)


if __name__ == "__main__":
    main()
